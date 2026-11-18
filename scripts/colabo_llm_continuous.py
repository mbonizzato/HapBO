#!/usr/bin/env python3
"""Continuous colabo-llm runner: upstream ColaBO (qPriorLogExpectedImprovement) over the box.

The continuous counterpart of the finite ColaBO worker (src/baselines/workers/colabo_worker.py)
on the synthetic panel. It builds the published ``colabo_logei`` Ax GenerationStrategy of the
finite worker (the same GenerationSteps, qPriorLogExpectedImprovement acquisition, SingleTaskGP
with the lognormal covariance prior and DefaultPrior user prior); the objective is the analytic
test function evaluated directly at the continuous point Ax proposes (no finite candidate pool,
no snapping to a candidate index). ColaBO's continuous BoTorch acquisition optimization is used
as published; only the evaluation source differs from the finite worker (a direct callable
instead of a pool lookup).

The user prior is the BORA supplement Sec. C.2 construction: N(c, sigma) with the confidence
float passed to DefaultPrior set to (1 - s + eps) in normalized [0,1] box coordinates (DefaultPrior
normalizes c into the unit cube and, for a float confidence, uses it directly as the per-dim std in
that normalized space). This matches how the finite worker passed a scalar ``confidence``.

The hidden objective is selected by SYNTH_TRUTH (scripts/synthetic_truth.py; default ``well``).
Run in the ColaBO environment with the pinned Ax/botorch trees on sys.path:
    COLABO_ROOT=external/colabo \
    <colabo_python> scripts/colabo_llm_continuous.py --family hartmann6 --only-seed 0 \
        --cs-json experiments/colabo_llm_priors/synthetic-continuous-hartmann6/hartmann6-rho90.json \
        --out-root results/synthetic/well/rho90/colabo

Writes <out-root>/<out-name or family>[-seed<k>].json (per-seed best-so-far regret trajectory + final regret).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ----- analytic families (self-contained copy so the runner needs no repo src import) ------
import numpy as np

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


# name -> (dim, lo, hi, f, f_opt)
_FAMILIES = {
    "hartmann6": (6, 0.0, 1.0, _hartmann6, -3.322368),
    "styblinski_tang7": (7, -5.0, 5.0, _styblinski, -39.16616570 * 7),
    "levy10": (10, -10.0, 10.0, _levy, 0.0),
    "ackley15": (15, -32.768, 32.768, _ackley, 0.0),
}
# Hidden objective of the synthetic panel (scripts/synthetic_truth.py; env SYNTH_TRUTH, default well)
_FAMILIES = __import__('synthetic_truth').apply(_FAMILIES)
BUDGETS = {"hartmann6": 40, "styblinski_tang7": 50, "levy10": 50, "ackley15": 150}


def _setup_colabo_paths():
    root = os.environ.get("COLABO_ROOT")
    if not root or not os.path.isdir(root):
        raise RuntimeError("COLABO_ROOT must point to the pinned upstream checkout")
    ax_root = os.path.join(root, "Ax")
    botorch_root = os.path.join(root, "botorch")
    if not os.path.isdir(ax_root) or not os.path.isdir(botorch_root):
        raise RuntimeError("COLABO_ROOT missing patched Ax or botorch tree")
    sys.path[0:0] = [ax_root, botorch_root]
    os.chdir(botorch_root)
    return botorch_root


def run_one(family, seed, c_native, confidence_float, budget, n_init):
    dim, lo, hi, f, f_opt = _FAMILIES[family]
    import torch
    from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
    from ax.modelbridge.registry import Models
    from ax.models.torch.botorch_modular.surrogate import Surrogate
    from ax.service.ax_client import AxClient
    from ax.service.utils.instantiation import ObjectiveProperties
    from benchmarking.mapping.gp_priors import get_covar_module
    from botorch.acquisition.prior_monte_carlo import qPriorLogExpectedImprovement
    from botorch.models import SingleTaskGP
    from botorch.utils.prior import DefaultPrior

    torch.manual_seed(seed)
    np.random.seed(seed)
    bounds = torch.tensor([[0.0] * dim, [1.0] * dim], dtype=torch.float)
    # normalized c in [0,1]
    c_unit = (np.asarray(c_native, float) - lo) / (hi - lo)
    c_unit = np.clip(c_unit, 0.0, 1.0)
    prior_location = torch.tensor(c_unit, dtype=torch.float)
    # BORA Sec C.2: sigma = 1 - s + eps, passed as a float confidence => per-dim std in [0,1] space
    user_prior = DefaultPrior(bounds, prior_location, confidence=float(confidence_float))

    gp_params = {"const": {"loc": 0.0, "scale": 1.0}, "ls": {"loc": 0.0, "scale": 1.0},
                 "ops": {"loc": 0.0, "scale": 1.0}, "noise": {"loc": 0.0, "scale": 1.0}}
    model_options = get_covar_module("singletask_lognormal", dim, prior_location=prior_location,
                                     gp_params=gp_params, gp_constraints={})

    steps = [GenerationStep(model=Models.PRIOR, num_trials=1,
                            model_kwargs={"seed": seed, "prior": user_prior})]
    if n_init > 1:
        steps.append(GenerationStep(model=Models.PRIOR, num_trials=n_init - 1,
                     model_kwargs={"seed": seed, "prior": user_prior, "generate_default": False}))
    num_bo = budget - n_init
    if num_bo > 0:
        acquisition_options = {"user_prior": user_prior, "decay_beta": float(budget),
                               "num_samples": 76_800, "resampling_fraction": 0.0067}
        optimizer_options = {"raw_samples": 1_024, "num_restarts": 20,
                             "retry_on_optimization_warning": False,
                             "options": {"nonnegative": False, "sample_around_best": True,
                                         "sample_around_best_sigma": 0.1, "maxiter": 100, "batch_limit": 48}}
        steps.append(GenerationStep(model=Models.BOTORCH_MODULAR, num_trials=num_bo,
                     model_kwargs={"surrogate": Surrogate(botorch_model_class=SingleTaskGP,
                                                          model_options=model_options),
                                   "botorch_acqf_class": qPriorLogExpectedImprovement,
                                   "acquisition_options": acquisition_options,
                                   "refit_on_update": True, "refit_params": {}},
                     model_gen_kwargs={"model_gen_options": {"optimizer_kwargs": optimizer_options}}))

    client = AxClient(generation_strategy=GenerationStrategy(steps=steps))
    client.create_experiment(
        name=f"colabo_llm_{family}_{seed}",
        parameters=[{"name": f"x_{k + 1}", "type": "range", "bounds": [0.0, 1.0]} for k in range(dim)],
        objectives={"objective": ObjectiveProperties(minimize=False)},  # Ax maximizes -> feed -f
    )
    best_min = np.inf
    traj = []
    for _ in range(budget):
        params, ti = client.get_next_trial()
        u = np.array([float(params[f"x_{k + 1}"]) for k in range(dim)])
        x_native = lo + (hi - lo) * u  # DIRECT continuous evaluation, no pool
        fval = f(x_native)  # native minimization value
        best_min = min(best_min, fval)
        traj.append(max(best_min - f_opt, 0.0))
        client.complete_trial(trial_index=ti, raw_data={"objective": -fval})  # maximize -f
    return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(_FAMILIES))
    ap.add_argument("--cs-json", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--max-new-seeds", type=int, default=0,
                    help="run at most N new seeds in this process then exit (0 = all). "
                         "Ax/BoTorch memory grows across repeated in-process runs, so the "
                         "driver runs one seed per process and relies on checkpoint/resume.")
    ap.add_argument("--out-root", default="results/colabo_llm")
    ap.add_argument("--budget", type=int, default=None,
                    help="override the family budget (default 40/50/50/150)")
    ap.add_argument("--out-name", default=None,
                    help="output file stem (default: family); use per-rho stems so "
                         "scenarios do not share a checkpoint file")
    ap.add_argument("--only-seed", type=int, default=None,
                    help="run exactly this one seed into its own checkpoint file "
                         "(parallelize long-budget families across processes)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _setup_colabo_paths()

    cs = json.loads((repo_root / args.cs_json).read_text() if not os.path.isabs(args.cs_json)
                    else Path(args.cs_json).read_text())
    c = cs["c"]
    s = float(cs["s"])
    eps = float(cs.get("eps", 1e-6))
    confidence_float = 1.0 - s + eps  # normalized-space std for DefaultPrior

    dim, lo, hi, f, f_opt = _FAMILIES[args.family]
    budget = args.budget if args.budget is not None else BUDGETS[args.family]
    n_init = 2 * dim
    print(f"== colabo-llm {args.family} d={dim} b{budget} n_init={n_init} "
          f"c[0]={c[0]} s={s} confidence(norm std)={confidence_float:.6f} ==", flush=True)

    out = repo_root / args.out_root
    out.mkdir(parents=True, exist_ok=True)
    stem = args.out_name or args.family
    if args.only_seed is not None:
        stem = f"{stem}-seed{args.only_seed}"
    out_path = out / f"{stem}.json"
    seed_list = [args.only_seed] if args.only_seed is not None else list(range(args.seeds))

    # Per-seed checkpoint/resume (IO only -- the per-seed run is unchanged and fully seeded, so
    # resuming at seed k is identical to a fresh straight-through run).
    trajs, finals = [], []
    if out_path.exists():
        prev = json.loads(out_path.read_text())
        if (prev.get("budget") == budget and prev.get("n_init") == n_init
                and prev.get("c") == c and float(prev.get("s", -1)) == s
                and prev.get("only_seed") == args.only_seed):
            trajs = [list(t) for t in prev.get("trajectories", [])][: len(seed_list)]
            finals = [t[-1] for t in trajs]
            if trajs:
                print(f"  resuming: {len(trajs)} seed(s) already checkpointed", flush=True)

    def _write():
        out_path.write_text(json.dumps({
            "family": args.family, "dim": dim, "budget": budget, "n_init": n_init,
            "c": c, "s": s, "confidence_float": confidence_float,
            "only_seed": args.only_seed, "seed_list": seed_list,
            "trajectories": trajs, "final_regret": finals,
            "mean_final_regret": float(np.mean(finals)) if finals else None,
            "seeds_done": len(trajs), "seeds_target": len(seed_list),
        }, indent=2))

    start = len(trajs)
    stop = len(seed_list) if args.max_new_seeds <= 0 else min(len(seed_list), start + args.max_new_seeds)
    for index in range(start, stop):
        seed = seed_list[index]
        t = run_one(args.family, seed, c, confidence_float, budget, n_init)
        trajs.append(t)
        finals.append(t[-1])
        print(f"  seed {seed}: colabo-llm final regret={t[-1]:.4f}", flush=True)
        _write()

    _write()
    if len(trajs) < len(seed_list):
        print(f"COLABO_LLM_PARTIAL {stem} seeds_done={len(trajs)}/{len(seed_list)}", flush=True)
    else:
        print(f"COLABO_LLM_DONE {stem} mean_final={float(np.mean(finals)):.4f}", flush=True)


if __name__ == "__main__":
    main()
