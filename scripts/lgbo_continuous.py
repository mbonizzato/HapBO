#!/usr/bin/env python3
"""Continuous LGBO (paper equation, posterior lift) on the synthetic panel.

LGBO run natively on the analytic test functions: the objective is evaluated directly at
the proposed continuous point (no finite pool, no snapping). The operator is the one of
the finite worker ``src/baselines/workers/lgbo_worker.py`` (imported, not re-implemented):
each iteration the LLM returns a point or region preference with confidence ``c`` for the
formula card, the posterior mean is shifted by ``c * Sigma(X, G) a / sqrt(a^T Sigma(G, G) a)``
(posterior covariances; covariance unchanged), and log-EI on the shifted posterior is
maximized over the box by a Sobol screen (2048 points) followed by 8 L-BFGS-B restarts.
The Matern-5/2 hyperparameters are refit by marginal likelihood each iteration; two
LLM-chosen points form the initial design (Sobol fallback). The run gives up after 6
consecutive guidance failures and pads the trajectory flat.

The hidden objective is selected by ``SYNTH_TRUTH`` (scripts/synthetic_truth.py; default
``well``). Run in the LLAMBO environment (numpy/scipy/aiohttp):

  LLM_BASE_URL=http://localhost:8000/v1 LLM_MODEL=<model> \
  <lgbo_python> scripts/lgbo_continuous.py --family hartmann6 --rho 90 --seeds 10

Default output: results/synthetic/<world>/<card>/lgbo/<family>-<card>-lgbo[-seedK].json
(per-seed best-so-far regret trajectories, checkpointed after every seed; resumable).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src" / "baselines" / "workers"))

import lgbo_common as common  # noqa: E402  guidance prompt/parser, grid + weights, erf, Sobol
import lgbo_worker as base  # noqa: E402  the finite LGBO worker (GP, instrumented chat, initial design)

METHOD_SLUG = "lgbo"


class _HardTimeout(Exception):
    """Raised by a SIGALRM handler to interrupt a chat() call that exceeds the hard time
    limit; a backstop to aiohttp's ClientTimeout."""


def _alarm_handler(signum, frame):
    raise _HardTimeout("chat hard-timeout")

# ---- analytic families (same constants as scripts/llambo_continuous.py) ----
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
_FAMILIES = __import__('synthetic_truth').apply(_FAMILIES)  # env SYNTH_TRUTH
BUDGETS = {"hartmann6": 40, "styblinski_tang7": 50, "levy10": 50, "ackley15": 150}
CARD_DIR = {"hartmann6": "hartmann6", "styblinski_tang7": "styblinski_tang7",
            "levy10": "levy10", "ackley15": "ackley15"}


RHO_CHOICES = ["90", "well50", "well10"]


def card_key(rho):
    """'90' -> 'rho90'; 'well50'/'well10' -> the well cards. Also the output stem tag."""
    rho = str(rho)
    return rho if rho.startswith("well") else "rho" + rho


def _card_text(family, rho="90"):
    path = (REPO / "experiments" / "landscape_estimates" / CARD_DIR[family]
            / ("card_formula_%s_domain.txt" % card_key(rho)))
    return path.read_text(encoding="utf-8").strip()


class _AcqGiveUp(Exception):
    """Raised when LGBO acquisition fails for `giveup_after` queries in a row -- the
    loop is aborted and the trajectory padded flat (parity with llambo_continuous)."""


def _sobol_unit(n, d, seed):
    from scipy.stats import qmc
    return qmc.Sobol(d=d, scramble=True, seed=int(seed)).random(int(n))



N_INIT = 2  # lgbo adapter: paper_n_initial
SETTINGS = {  # = src/baselines/lgbo.py worker_config defaults
    "n_grid": 64, "decay": 4.0, "length_scale": 0.2, "noise_std": 1e-3,
    "max_tokens": 512, "request_timeout": 300.0,
}
HARD_TIMEOUT = 330  # SIGALRM backstop above the 300 s client timeout (aiohttp may fail to cancel a wedged call)
_LOG_TINY = float(np.log(np.finfo(float).tiny))
_CHUNK = 256  # rows per kernel block (the GP kernel materializes an (n, m, d) difference array)


def default_out_root():
    import synthetic_truth

    return "results/synthetic/%s" % synthetic_truth.world()


class _PosteriorLift:
    """lgbo_worker._lgbo_step's lift for one BO iteration, with the per-iteration pieces cached.

    Same grid/weights call and seed as the finite step, same guards as _regional_shift_z;
    shift(U) = lam * (K(U,G) a - (L^-1 K(X,U))^T (L^-1 K(X,G) a)) = lam * gp.posterior_cross_cov(U, G) @ a.
    """

    def __init__(self, gp, guidance, dim, n_grid, seed, decay, eps=1e-12):
        self.gp, self.lam, self.q, self.G, self.a, self.proj = gp, 0.0, 0.0, None, None, None
        if guidance is None:
            return
        kind, lb, ub, confidence = guidance
        G, a = common._grid_and_weights(kind, lb, ub, dim, n_grid, seed, decay)
        a = np.asarray(a, dtype=float).reshape(-1)
        sigma_gg = gp.posterior_cov(G)
        q = float(a @ sigma_gg @ a)
        self.q = q
        if not np.isfinite(q) or q <= eps or confidence <= 0.0:
            return
        self.lam, self.G, self.a = float(confidence) / np.sqrt(q), G, a
        if not getattr(gp, "_degenerate_", False) and gp.X_train_.shape[0] > 0:
            self.proj = np.linalg.solve(gp.L_, gp._kernel(gp.X_train_, G) @ a)

    def shift(self, U):
        U = np.atleast_2d(np.asarray(U, dtype=float))
        if self.lam == 0.0:
            return np.zeros(U.shape[0])
        out = self.gp._kernel(U, self.G) @ self.a
        if self.proj is not None:
            vx = np.linalg.solve(self.gp.L_, self.gp._kernel(self.gp.X_train_, U))
            out = out - vx.T @ self.proj
        out = self.lam * out
        return out if np.all(np.isfinite(out)) else np.zeros(U.shape[0])


def _log_ei(gp, lift, U, best_observed):
    """lgbo_worker._lgbo_step's score (log-EI on the shifted mean) at arbitrary unit-cube points."""
    U = np.atleast_2d(np.asarray(U, dtype=float))
    if U.shape[0] > _CHUNK:
        return np.concatenate([_log_ei(gp, lift, U[i:i + _CHUNK], best_observed) for i in range(0, U.shape[0], _CHUNK)])
    mean, std = gp.predict(U)
    mean = mean + lift.shift(U) * gp.y_scale_
    std = np.maximum(std, 0.0)  # the exponential tilt does not change covariance
    improvement = mean - float(best_observed)
    ei = np.maximum(improvement, 0.0)
    positive = std > 1e-12
    if np.any(positive):
        z = improvement[positive] / std[positive]
        cdf = 0.5 * (1.0 + common._erf_vec(z / np.sqrt(2.0)))
        pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)
        ei[positive] = improvement[positive] * cdf + std[positive] * pdf
    log_ei = np.log(np.maximum(ei, np.finfo(float).tiny))
    return np.nan_to_num(log_ei, nan=-np.inf, posinf=np.inf, neginf=-np.inf)


def _task_config(family, rho, dim, lo, hi):
    """Task text of every synthetic LGBO run: the formula card plus the unit-cube coordinate mapping."""
    return {
        "model": "an analytic black-box objective",
        "metric": "objective value",
        "lower_is_better": True,  # we minimize f (g = -f is maximized internally)
        "description": "Approximate model of the objective (may be imperfect):\n"
                       + _card_text(family, rho)
                       + "\nSearch coordinates: x0 ... x%d are shown rescaled to [0, 1]; the model's x_i equals "
                         "%g + %g * (shown value of x_i)." % (dim - 1, lo, hi - lo),
    }


def _timed(fn, seconds=HARD_TIMEOUT):
    """Run fn() under lgbo_continuous's SIGALRM hard cap (POSIX only; plain call elsewhere)."""
    if not hasattr(signal, "SIGALRM"):
        return fn()
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(int(seconds))
    try:
        return fn()
    finally:
        signal.alarm(0)


def _telemetry_summary(calls):
    guidance = [c for c in calls if c.get("kind") == "guidance"]
    return {
        "n_calls": len(calls), "n_calls_ok": sum(1 for c in calls if c.get("ok")),
        "n_initial_calls": sum(1 for c in calls if c.get("kind") == "initial"), "n_guidance_calls": len(guidance),
        "prompt_tokens_total": int(sum(c.get("prompt_tokens", 0) for c in calls)),
        "completion_tokens_total": int(sum(c.get("completion_tokens", 0) for c in calls)),
        "total_tokens_total": int(sum(c.get("total_tokens", 0) for c in calls)),
        "llm_latency_seconds_total": round(float(sum(c.get("latency_seconds", 0.0) for c in calls)), 3),
    }


def run_one(family, seed, budget, rho="90", chat=None, n_sobol=2048, n_restarts=8, giveup_after=6, verbose=False):
    """One seed. Returns (trajectory, guidance_failures, gaveup, info). ``chat`` overrides the LLM (tests)."""
    from scipy.optimize import minimize

    st = SETTINGS
    dim, lo, hi, f, f_opt = _FAMILIES[family][:5]
    calls = []
    if chat is None:
        chat = base._make_instrumented_chat(
            base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("LLM_API_KEY", "none"),
            model=os.environ.get("LLM_MODEL", "Qwen3.5-122B-A10B-GPTQ-Int4"),
            timeout=float(st["request_timeout"]), seed=seed, sink=calls)

    def to_native(u):
        return lo + (hi - lo) * np.asarray(u, float)

    best = {"val": np.inf}
    traj = []

    def record(fval):
        best["val"] = min(best["val"], float(fval))
        traj.append(max(best["val"] - f_opt, 0.0))

    config = _task_config(family, rho, dim, lo, hi)
    n_init = max(1, min(N_INIT, budget))

    # Initial design: 2 LLM-chosen locations (continuous: used as given); seeded Sobol fallback.
    U, init_source = None, "llm"
    try:
        reply = _timed(lambda: chat([{"role": "user", "content": base._initial_prompt(dim, config, n_init)}],
                                    temperature=0.0, max_tokens=int(st["max_tokens"]), kind="initial", query_after=0))
        U = base._parse_initial_points(reply, dim, n_init)
    except Exception as exc:
        print("  [init-llm-fail] %s; using seeded Sobol fallback" % exc, flush=True)
    if U is None:
        U, init_source = _sobol_unit(n_init, dim, seed), "sobol-fallback"
    else:  # the finite worker never repeats a candidate: replace an exact repeat by a seeded Sobol point
        sob = _sobol_unit(n_init, dim, seed)
        for i in range(1, len(U)):
            if np.min(np.abs(U[:i] - U[i]).max(axis=1)) < 1e-12:
                U[i], init_source = sob[i], "llm+sobol-dedup"
    X_obs, y_obs, history = [], [], []

    def observe(u):
        u = np.clip(np.asarray(u, float), 0.0, 1.0)
        fval = f(to_native(u))
        X_obs.append([float(v) for v in u])
        y_obs.append(-fval)  # maximization of g = -f
        history.append(("[" + ", ".join("%.4f" % v for v in u) + "]", fval))
        record(fval)

    for u in U:
        if len(y_obs) < budget:
            observe(u)

    consecutive = fallbacks = 0
    gaveup = False
    lifts = {"guided": 0, "zero_lift": 0, "point": 0, "region": 0}
    try:
        while len(y_obs) < budget:
            gp = base._GP(length_scale=float(st["length_scale"]), noise_std=float(st["noise_std"])).fit(
                np.asarray(X_obs, float), np.asarray(y_obs, float))
            lines = ["  %s -> %.6g" % row for row in history[-30:]]
            guidance = None
            try:
                reply = _timed(lambda: chat([{"role": "user", "content": common._guidance_prompt(dim, config, lines)}],
                                            temperature=0.0, max_tokens=int(st["max_tokens"]),
                                            kind="guidance", query_after=len(y_obs)))
                guidance = common._parse_guidance(reply, dim)
                if guidance is None:
                    consecutive += 1
                    fallbacks += 1
                    print("  [parse-fail] guidance unparseable (%d in a row)" % consecutive, flush=True)
            except Exception as exc:
                consecutive += 1
                fallbacks += 1
                print("  [llm-fail] %s (%d in a row)" % (exc, consecutive), flush=True)
            else:
                if guidance is not None:
                    consecutive = 0
            if consecutive >= giveup_after:
                gaveup = True
                raise _AcqGiveUp("%d consecutive guidance failures" % giveup_after)

            lift = _PosteriorLift(gp, guidance, dim, int(st["n_grid"]), seed + 1000 + len(y_obs), float(st["decay"]))
            if guidance is not None:
                lifts["guided"] += 1
                lifts[guidance[0]] = lifts.get(guidance[0], 0) + 1
                lifts["zero_lift"] += int(lift.lam == 0.0)
                if verbose:
                    print("  q%d %s c=%.3g q=%.4g lambda=%.4g" % (len(y_obs), guidance[0], guidance[3], lift.q,
                                                                lift.lam), flush=True)
            best_obs = float(np.max(y_obs))

            def score(Umat):
                s = _log_ei(gp, lift, Umat, best_obs)
                return np.where(np.isfinite(s), s, np.where(s > 0, -_LOG_TINY, _LOG_TINY))

            cand = _sobol_unit(n_sobol, dim, seed + 13 + len(y_obs))
            vals = score(cand)
            order = np.argsort(-vals)
            starts = cand[order[:n_restarts]]
            best_u = np.clip(np.asarray(starts[0], float), 0.0, 1.0)
            best_val = float(vals[order[0]])
            for x0 in starts:
                res = minimize(lambda u: -float(score(np.clip(u, 0.0, 1.0))[0]), x0, method="L-BFGS-B",
                               bounds=[(0.0, 1.0)] * dim)
                cu = np.asarray(res.x, float)
                if not np.all(np.isfinite(cu)):
                    continue
                cu = np.clip(cu, 0.0, 1.0)
                val = -float(res.fun)
                if np.isfinite(val) and val > best_val:
                    best_val, best_u = val, cu
            if not np.all(np.isfinite(best_u)):
                best_u = np.random.default_rng(seed + len(y_obs)).random(dim)
            observe(best_u)
    except _AcqGiveUp as exc:
        print("  [acq-giveup] %s; padding remaining budget flat" % exc, flush=True)
    except Exception as exc:  # GP or optimizer failure: the partial trajectory is padded below
        print("  [warn] loop aborted (%s); padding trajectory" % exc, flush=True)

    n_real = len(traj)
    if not traj:
        traj = [float(best["val"] - f_opt) if np.isfinite(best["val"]) else 1e9]
    if len(traj) < budget:
        traj.extend([traj[-1]] * (budget - len(traj)))
    info = {"init_source": init_source, "n_real_evaluations": n_real, "lifts": lifts,
            "telemetry": _telemetry_summary(calls)}
    return traj[:budget], fallbacks, gaveup, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(_FAMILIES))
    ap.add_argument("--rho", default="90", choices=RHO_CHOICES)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--only-seed", type=int, default=None)
    ap.add_argument("--max-new-seeds", type=int, default=0)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--out-root", default=None,
                    help="default results/synthetic/<SYNTH_TRUTH>/<card>/lgbo")
    ap.add_argument("--verbose", action="store_true", help="print the lift (q, lambda) every iteration")
    args = ap.parse_args()

    budget = args.budget if args.budget is not None else BUDGETS[args.family]
    stem = "%s-%s-%s" % (args.family, card_key(args.rho), METHOD_SLUG)
    if args.only_seed is not None:
        stem += "-seed%d" % args.only_seed
    out = REPO / (args.out_root or "%s/%s/%s" % (default_out_root(), card_key(args.rho), METHOD_SLUG))
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / (stem + ".json")
    seed_list = [args.only_seed] if args.only_seed is not None else list(range(args.seeds))
    import synthetic_truth

    world = {"truth": synthetic_truth.world()}

    trajs, fallbacks, infos = [], [], []
    if out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        if prev.get("world") not in (None, world):
            raise SystemExit("refusing to resume %s: written under %s, now %s" % (out_path, prev.get("world"), world))
        if prev.get("budget") == budget and prev.get("rho") == args.rho and prev.get("only_seed") == args.only_seed:
            trajs = [list(t) for t in prev.get("trajectories", [])][: len(seed_list)]
            fallbacks = list(prev.get("fallbacks", []))[: len(trajs)]
            infos = list(prev.get("seed_info", []))[: len(trajs)]
            while len(fallbacks) < len(trajs):
                fallbacks.append(0)
            while len(infos) < len(trajs):
                infos.append({})
            if trajs:
                print("resuming: %d seed(s) checkpointed" % len(trajs), flush=True)

    def write():
        finals = [t[-1] for t in trajs]
        out_path.write_text(json.dumps({
            "family": args.family, "rho": args.rho, "budget": budget, "n_init": N_INIT,
            "method": "lgbo (paper equation, posterior lift; finite worker operator, continuous acquisition)",
            "world": world, "f_opt": float(_FAMILIES[args.family][4]), "settings": SETTINGS,
            "only_seed": args.only_seed, "seed_list": seed_list,
            "trajectories": trajs, "final_regret": finals, "fallbacks": fallbacks, "seed_info": infos,
            "mean_final_regret": (sum(finals) / len(finals)) if finals else None,
            "seeds_done": len(trajs), "seeds_target": len(seed_list),
        }, indent=2), encoding="utf-8")

    start = len(trajs)
    stop = len(seed_list) if args.max_new_seeds <= 0 else min(len(seed_list), start + args.max_new_seeds)
    for index in range(start, stop):
        seed = seed_list[index]
        print("== %s %s rho%s b%d seed %d (%s) ==" % (METHOD_SLUG, args.family, args.rho, budget, seed, world),
              flush=True)
        traj, nfb, gaveup, info = run_one(args.family, seed, budget, args.rho, verbose=args.verbose)
        info["gaveup"] = gaveup
        trajs.append(traj)
        fallbacks.append(nfb)
        infos.append(info)
        print("  seed %d: final regret %.4f (%d guidance failures%s; init %s; lifts %s; tokens %d)"
              % (seed, traj[-1], nfb, ", gaveup" if gaveup else "", info["init_source"], info["lifts"],
                 info["telemetry"]["total_tokens_total"]), flush=True)
        write()
    write()
    print("LGBO_CONT_%s %s %d/%d seeds -> %s"
          % ("PARTIAL" if len(trajs) < len(seed_list) else "DONE", stem, len(trajs), len(seed_list), out_path),
          flush=True)


if __name__ == "__main__":
    main()
