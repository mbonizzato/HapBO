#!/usr/bin/env python3
"""Continuous LLAMBO runner for the synthetic panel.

Runs the pinned upstream LLAMBO-discriminative loop natively on the analytic test
functions (mirror of scripts/colabo_llm_continuous.py: the objective is evaluated
DIRECTLY at the proposed continuous point - no finite pool, no snapping; nothing
about the LLAMBO algorithm is modified, only the evaluation source).

Two official initialization modes (the paper's):
  random    - Mode 1: n_init points sampled uniformly in the box (seeded);
  zero-shot - Mode 2: n_init points proposed by the LLM via zero-shot prompting
              (upstream exp_warmstarting FullTemplate skeleton), with the selected
              formula card as context (the same information HapBO's elicitor and
              the colabo-llm (c,s) elicitation received). n_init=5 per the paper.
The formula card is also part of the per-iteration task context. The hidden objective is
selected by SYNTH_TRUTH (scripts/synthetic_truth.py; default ``well``).

Run in the LLAMBO environment (python 3.8):
  LLAMBO_ROOT=external/LLAMBO \
  <llambo_python> scripts/llambo_continuous.py --family hartmann6 --mode zero-shot --rho 90 --seeds 10

Outputs <out-root>/<family>-<card>-<mode>[-seedK].json
(per-seed best-so-far regret trajectories, checkpointed after every seed).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "baselines" / "workers"))

# ---- analytic families: identical constants to scripts/colabo_llm_continuous.py ----
_HARTMANN_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_HARTMANN_A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14],
                        [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
_HARTMANN_P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991],
                               [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]])


def _hartmann6(x):
    x = np.asarray(x, float).reshape(-1)
    inner = np.sum(_HARTMANN_A * (x[None, :] - _HARTMANN_P) ** 2, axis=1)
    return float(-np.sum(_HARTMANN_ALPHA * np.exp(-inner)))


def _styblinski(x):
    x = np.asarray(x, float).reshape(-1)
    return float(0.5 * np.sum(x ** 4 - 16.0 * x ** 2 + 5.0 * x))


def _levy(x):
    x = np.asarray(x, float).reshape(-1)
    w = 1.0 + (x - 1.0) / 4.0
    t1 = np.sin(np.pi * w[0]) ** 2
    t3 = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2)
    mid = np.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0) ** 2))
    return float(t1 + mid + t3)


def _ackley(x):
    x = np.asarray(x, float).reshape(-1)
    d = x.size
    a, b, c = 20.0, 0.2, 2.0 * np.pi
    return float(-a * np.exp(-b * np.sqrt(np.sum(x ** 2) / d)) - np.exp(np.sum(np.cos(c * x)) / d) + a + np.e)


_FAMILIES = {
    "hartmann6": (6, 0.0, 1.0, _hartmann6, -3.322368),
    "styblinski_tang7": (7, -5.0, 5.0, _styblinski, -39.16616570 * 7),
    "levy10": (10, -10.0, 10.0, _levy, 0.0),
    "ackley15": (15, -32.768, 32.768, _ackley, 0.0),
}
# Hidden objective of the synthetic panel (scripts/synthetic_truth.py; env SYNTH_TRUTH, default well)
_FAMILIES = __import__('synthetic_truth').apply(_FAMILIES)
BUDGETS = {"hartmann6": 40, "styblinski_tang7": 50, "levy10": 50, "ackley15": 150}
CARD_DIR = {"hartmann6": "hartmann6", "styblinski_tang7": "styblinski_tang7",
            "levy10": "levy10", "ackley15": "ackley15"}
N_INIT = 5  # the paper's initialization count (both modes)


RHO_CHOICES = ["90", "well50", "well10"]


def card_key(rho):
    """'90' -> 'rho90'; 'well50'/'well10' -> the well cards (scripts/build_synthetic_cards.py)."""
    rho = str(rho)
    return rho if rho.startswith("well") else "rho" + rho


def _card_text(family, rho="90"):
    path = (REPO / "experiments" / "landscape_estimates" / CARD_DIR[family]
            / ("card_formula_%s_domain.txt" % card_key(rho)))
    return path.read_text(encoding="utf-8").strip()


def _axis_sentence(dim, lo, hi):
    """LLAMBO and LGBO see unit-cube coordinates while the card's model is in native units."""
    return ("Search coordinates: x0 ... x%d are shown rescaled to [0, 1]; the model's x_i equals "
            "%g + %g * (shown value of x_i)." % (dim - 1, lo, hi - lo))


def _zero_shot_units(family, dim, lo, hi, seed, base_url, api_key, model, timeout, rho="90"):
    """Mode 2: LLM proposes N_INIT native configs from the rho card; returns
    unit-cube rows (clamped). Shortfall filled with seeded uniform points (logged)."""
    names = ["x%d" % k for k in range(dim)]
    ranges = ", ".join("%s (float between [%g, %g])" % (n, lo, hi) for n in names)
    prompt = (
        "You are assisting me with automated machine learning.\n"
        + _card_text(family, rho) + "\n"
        + "I'm exploring a subset of hyperparameters detailed as: " + ranges + ".\n"
        + "Provide %d diverse and efficient configuration suggestions to commence a "
          "Bayesian Optimization procedure aimed at fine-tuning hyperparameters.\n" % N_INIT
        + "You mustn't include 'None' in the configurations.\n"
        + "Your response should include only a list of dictionaries, where each dictionary "
          "describes one recommended configuration. Do not enumerate the dictionaries."
    )

    def call(text):
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "temperature": 0.0, "max_tokens": 3072,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode("utf-8")
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions", data=payload,
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key})
        # transport retry only (same request, temperature 0): a busy shared server
        # must not end the run on a socket timeout
        for k in range(6):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read())["choices"][0]["message"]["content"]
            except (socket.timeout, urllib.error.URLError, ConnectionError) as e:
                if k == 5:
                    raise
                print("  zero-shot init: transport error (%s), retry %d/5 in %ds" % (e, k + 1, 30 * (k + 1)),
                      flush=True)
                time.sleep(30 * (k + 1))

    rows, attempt, error = [], prompt, ""
    for _ in range(4):
        raw = call(attempt)
        try:
            start = raw.index("[")
            proposals, _ = json.JSONDecoder().raw_decode(raw[start:])
            for p in proposals:
                u = [(min(max(float(p[n]), lo), hi) - lo) / (hi - lo) for n in names]
                if not any(np.allclose(u, r, atol=1e-9) for r in rows):
                    rows.append(u)
            if len(rows) >= N_INIT:
                break
            error = "only %d distinct configurations" % len(rows)
        except (ValueError, KeyError, TypeError) as exc:
            error = "unparseable reply (%s)" % exc
        attempt = prompt + ("\n\nYour previous answer was insufficient: %s. Answer again "
                            "with %d DISTINCT configurations as a plain JSON list." % (error, N_INIT))
    print("[warmstart] %d LLM-proposed init points" % min(len(rows), N_INIT), flush=True)
    if len(rows) < N_INIT:
        rng = np.random.default_rng(seed)
        while len(rows) < N_INIT:
            rows.append(list(rng.random(dim)))
        print("[warmstart] shortfall filled with seeded uniform points", flush=True)
    return [list(map(float, r)) for r in rows[:N_INIT]]


class _AcqGiveUp(Exception):
    """Raised when continuous LLAMBO cannot generate candidates for `giveup_after`
    queries in a row — the loop is aborted and the trajectory padded flat."""


def _install_acquisition_fallback(optimizer, dim, seed, *, max_outer=2, giveup_after=6):
    """Make continuous LLAMBO robust to candidate-generation failure.

    Upstream ``get_candidate_points`` retries 4x internally, then RAISES if it
    cannot parse >=5 candidate configs from the LLM. In continuous space (free-form
    float vectors) that failure is frequent at high dimension (~100% for 15-D
    Ackley). Here we allow up to ``max_outer`` full attempts; on persistent failure
    we gracefully SKIP the query by re-proposing the incumbent (best observed point,
    with sub-1e-3 jitter so it stays a fresh row) -> best-so-far is unchanged and the
    run continues instead of crashing. Occasional (transient) failures are absorbed
    this way. If generation fails ``giveup_after`` queries in a row the case is
    hopeless (e.g. Ackley-15): we raise :class:`_AcqGiveUp`, and run_one pads the
    remaining budget flat — same honest 'no further progress' trajectory, without
    burning the LLM on ~200 pointless selection calls. ``stats['fallbacks']`` counts
    skipped queries; ``stats['gaveup']`` records the early abort.
    """
    inner = optimizer.acq_func.get_candidate_points
    rng = np.random.default_rng(int(seed) + 91731)
    stats = {"fallbacks": 0, "consecutive": 0, "gaveup": False}

    def _incumbent(observed_configs, observed_fvals):
        vals = observed_fvals.iloc[:, 0].astype(float)
        row = observed_configs.loc[[vals.idxmin()]].reset_index(drop=True)
        jitter = (rng.random(row.shape[1]) - 0.5) * 2e-4
        for j, col in enumerate(row.columns):
            row.at[0, col] = float(min(max(float(row.at[0, col]) + jitter[j], 0.0), 1.0))
        return row

    def robust(observed_configs, observed_fvals, *a, **k):
        for _ in range(max_outer):
            try:
                out = inner(observed_configs, observed_fvals, *a, **k)
                stats["consecutive"] = 0
                return out
            except Exception:
                pass
        stats["fallbacks"] += 1
        stats["consecutive"] += 1
        if stats["consecutive"] >= giveup_after:
            stats["gaveup"] = True
            raise _AcqGiveUp(
                "%d consecutive candidate-generation failures" % giveup_after)
        return _incumbent(observed_configs, observed_fvals), 0.0, 0.0

    optimizer.acq_func.get_candidate_points = robust
    return stats


def run_one(family, seed, mode, budget, rho="90"):
    dim, lo, hi, f, f_opt = _FAMILIES[family]
    base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("LLM_API_KEY", "none")
    model = os.environ.get("LLM_MODEL", "Qwen3.5-122B-A10B-GPTQ-Int4")
    os.environ.setdefault("OPENAI_API_KEY", api_key)
    os.environ.setdefault("OPENAI_API_BASE", base_url)
    os.environ.setdefault("OPENAI_API_TYPE", "open_ai")
    os.environ.setdefault("OPENAI_API_VERSION", "")

    root = os.environ.get("LLAMBO_ROOT")
    if not root or not os.path.isdir(root):
        raise RuntimeError("LLAMBO_ROOT must point to the pinned upstream checkout")
    if root not in sys.path:
        sys.path.insert(0, root)
    cwd = os.getcwd()
    os.chdir(root)

    import openai  # noqa: E402
    import llambo_worker as W  # noqa: E402  (shim + task context + score bound)
    from llambo.llambo import LLAMBO  # noqa: E402

    W._install_chat_shim(openai, base_url=base_url, api_key=api_key, model=model,
                         timeout=120.0, seed=seed)  # shorter: a hung SGLang call recovers fast

    if mode == "zero-shot":
        units = _zero_shot_units(family, dim, lo, hi, seed, base_url, api_key, model, 120.0, rho)
    else:
        rng = np.random.default_rng(seed)
        units = [list(map(float, rng.random(dim))) for _ in range(N_INIT)]

    def initialization(count):
        if count != N_INIT:
            raise ValueError("expected %d init points, got %d" % (N_INIT, count))
        return [{"x%d" % k: u[k] for k in range(dim)} for u in units]

    best = {"val": np.inf}
    traj = []

    def objective(candidate):
        u = np.array([min(max(float(candidate["x%d" % k]), 0.0), 1.0) for k in range(dim)])
        fval = f(lo + (hi - lo) * u)  # DIRECT continuous evaluation
        best["val"] = min(best["val"], fval)
        traj.append(max(best["val"] - f_opt, 0.0))
        snapped = {"x%d" % k: float(u[k]) for k in range(dim)}
        return snapped, {"score": fval, "generalization_score": fval}

    # Feed the formula card into LLAMBO's per-iteration task context, so the in-loop LLM
    # receives the same approximate model as every other prior-accepting method. Upstream
    # builds its prompt templates with str.format, so the card's literal braces
    # (e.g. "sum_{k=1..4}") are doubled to render as single braces.
    config = {"model": ("an analytic black-box objective. Approximate model of the "
                        "objective (may be imperfect):\n" + _card_text(family, rho)
                        + "\n" + _axis_sentence(dim, lo, hi)).replace("{", "{{").replace("}", "}}"),
              "task": "regression", "metric": "objective value"}
    # upstream picks the printed precision from each bound's decimals ([0, 1] bounds would print 0
    # decimals): print 4 decimals (2 in the generative surrogate, where upstream adds 2).
    import llambo.acquisition_function as _llambo_af  # noqa: E402
    import llambo.discriminative_sm_utils as _llambo_ds  # noqa: E402
    import llambo.generative_sm_utils as _llambo_gs  # noqa: E402
    _llambo_af.LLM_ACQ._count_decimal_places = lambda self, n: 4
    _llambo_ds._count_decimal_places = lambda n: 4
    _llambo_gs._count_decimal_places = lambda n: 2  # upstream adds 2 here
    optimizer = LLAMBO(
        W._task_context(dim, 0, False, config),  # lower_is_better=True (minimise)
        sm_mode="discriminative", n_candidates=10, n_templates=2, n_gens=10,
        alpha=0.1, n_initial_samples=N_INIT, n_trials=budget - N_INIT,
        init_f=initialization, bbox_eval_f=objective, chat_engine=model,
    )
    optimizer.seed = seed
    W._bound_acquisition_scores(optimizer)
    fb = _install_acquisition_fallback(optimizer, dim, seed)
    try:
        optimizer.optimize()
    except _AcqGiveUp as exc:  # hopeless continuous generation (e.g. Ackley-15) — pad flat
        print("  [acq-giveup] %s; padding remaining budget flat" % exc, flush=True)
    except Exception as exc:  # surrogate/selection failure — keep the partial trajectory
        print("  [warn] optimize() aborted (%s); padding trajectory" % exc, flush=True)
    os.chdir(cwd)
    if not traj:
        traj = [float(best["val"] - f_opt) if np.isfinite(best["val"]) else 1e9]
    if len(traj) < budget:
        traj.extend([traj[-1]] * (budget - len(traj)))
    return traj[:budget], fb["fallbacks"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(_FAMILIES))
    ap.add_argument("--mode", required=True, choices=["random", "zero-shot"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--only-seed", type=int, default=None)
    ap.add_argument("--max-new-seeds", type=int, default=0)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--rho", default="90", choices=RHO_CHOICES)
    ap.add_argument("--out-root", default="results/estimate_panel/llambo")
    args = ap.parse_args()

    budget = args.budget if args.budget is not None else BUDGETS[args.family]
    stem = "%s-%s-%s" % (args.family, card_key(args.rho), args.mode)
    if args.only_seed is not None:
        stem += "-seed%d" % args.only_seed
    out = REPO / args.out_root
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / (stem + ".json")
    seed_list = [args.only_seed] if args.only_seed is not None else list(range(args.seeds))

    trajs = []
    fallbacks = []
    if out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        if prev.get("budget") == budget and prev.get("mode") == args.mode \
                and prev.get("only_seed") == args.only_seed:
            trajs = [list(t) for t in prev.get("trajectories", [])][: len(seed_list)]
            fallbacks = list(prev.get("fallbacks", []))[: len(trajs)]
            while len(fallbacks) < len(trajs):
                fallbacks.append(0)
            if trajs:
                print("resuming: %d seed(s) checkpointed" % len(trajs), flush=True)

    def write():
        finals = [t[-1] for t in trajs]
        out_path.write_text(json.dumps({
            "family": args.family, "mode": args.mode, "budget": budget, "n_init": N_INIT,
            "only_seed": args.only_seed, "seed_list": seed_list,
            "trajectories": trajs, "final_regret": finals, "fallbacks": fallbacks,
            "mean_final_regret": (sum(finals) / len(finals)) if finals else None,
            "seeds_done": len(trajs), "seeds_target": len(seed_list),
        }, indent=2), encoding="utf-8")

    start = len(trajs)
    stop = len(seed_list) if args.max_new_seeds <= 0 else min(len(seed_list), start + args.max_new_seeds)
    for index in range(start, stop):
        seed = seed_list[index]
        print("== llambo-%s %s b%d seed %d ==" % (args.mode, args.family, budget, seed), flush=True)
        traj, nfb = run_one(args.family, seed, args.mode, budget, args.rho)
        trajs.append(traj)
        fallbacks.append(nfb)
        print("  seed %d: final regret %.4f (%d/%d queries skipped)"
              % (seed, traj[-1], nfb, budget), flush=True)
        write()
    write()
    print("LLAMBO_CONT_%s %s %d/%d seeds" %
          ("PARTIAL" if len(trajs) < len(seed_list) else "DONE", stem, len(trajs), len(seed_list)), flush=True)


if __name__ == "__main__":
    main()
