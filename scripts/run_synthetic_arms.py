#!/usr/bin/env python3
"""Run the HapBO, vanilla BO, control, random and TuRBO arms of the synthetic panel.

Functions Hartmann-6, Styblinski-Tang-7, Levy-10, Ackley-15. scripts/run_synthetic_panel.py runs
every arm of the panel through this script and the baseline runners. The objective is the hidden
objective selected by SYNTH_TRUTH (scripts/synthetic_truth.py; default ``well``); regret is measured
against its true optimum; the card only changes the prior. Budgets per BUDGETS below (40/50/50;
150 for ackley15), paired seeds 0..N-1.

Arms (select with --arms):
  hapbo   - run_continuous_bo + the compiled prior document given by --prior-file. The
            prior-maximizer search is seeded with the document's own peak coordinates
            (landscape_generator.document_peak_points) unless --no-document-peaks.
  vanilla - run_continuous_bo without a prior (paired seeds)
  vanilla-peaks
          - vanilla EI (unshaped acquisition) whose candidate batch also holds the prior peaks the
            hapbo arm appends for the same --prior-file, alpha and seed
            (run_continuous_bo(shape_acquisition=False)); needs --prior-file
  pi-bo, prior-to-mean, prior-warm-start, prior-greedy
          - prior-injection arms (run_continuous_bo(prior_strategy=<arm>); continuous_bo module
            docstring), the continuous counterparts of the finite-grid arms of fig_ablations with the
            prior of --prior-file: piBO (Hvarfner et al. 2022; the prior mode, then points sampled
            from the prior, beta = N / 10),
            the GP prior mean compiled alpha x standardized prior, the highest-prior initial
            design, and the highest-prior points without a GP. They locate and score the same
            prior peaks as the hapbo arm; --alpha-override does not apply.
  random  - uniform random baseline
  turbo   - TuRBO-1 native (src/baselines/turbo_continuous.py). Needs torch+gpytorch:
            run the turbo arm under the ColaBO environment's python (lazy imports keep the
            other arms torch-free). The script itself is python-3.9 compatible.

hapbo, vanilla, vanilla-peaks and prior-to-mean share the seeded Sobol initial design.

Output ({family: {arm: [[per-query regret] x seeds]}}):
  <out-root>/<label>/trajectories.json (+ continuous_results.csv, run_provenance.json)
Incremental partial files are written after every seed so a killed job resumes
by rerunning (completed family/arm pairs are reloaded from the partial file).

Examples (alpha 0.325 is the setting of the paper's HapBO arm):
  python scripts/run_synthetic_arms.py --label hartmann6-draw1 --families hartmann6 --arms hapbo \
      --prior-file experiments/synthetic_priors/rho90/hartmann6/draw1/prior.json --alpha-override 0.325
  python scripts/run_synthetic_arms.py --label hartmann6-draw1-vanilla-peaks --families hartmann6 --arms vanilla-peaks \
      --prior-file experiments/synthetic_priors/rho90/hartmann6/draw1/prior.json --alpha-override 0.325
  python scripts/run_synthetic_arms.py --label hartmann6-draw1 --families hartmann6 \
      --arms pi-bo,prior-to-mean,prior-warm-start,prior-greedy \
      --prior-file experiments/synthetic_priors/rho90/hartmann6/draw1/prior.json
  python scripts/run_synthetic_arms.py --label none --arms vanilla,random
  <colabo_python> scripts/run_synthetic_arms.py --label none-turbo --arms turbo
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# prior-injection arms: arm name = run_continuous_bo prior_strategy
INJECTION_ARMS = ("pi-bo", "prior-to-mean", "prior-warm-start", "prior-greedy")
PI_BO_BETA = "auto"  # beta = N / 10, N = budget - n_init BO iterations (Hvarfner et al. 2022, Sec. 4)
CANDIDATE_SET = "alpha-pi first acquisition batch: sobol(n_sobol, seed + n_init) + prior peaks, distinct points"

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


# name -> (dim, lo, hi, f, f_opt)
FAMILIES = {
    "hartmann6": (6, 0.0, 1.0, _hartmann6, -3.322368),
    "styblinski_tang7": (7, -5.0, 5.0, _styblinski, -39.16616570 * 7),
    "levy10": (10, -10.0, 10.0, _levy, 0.0),
    "ackley15": (15, -32.768, 32.768, _ackley, 0.0),
}
# Hidden objective of the synthetic panel (scripts/synthetic_truth.py; env SYNTH_TRUTH, default well)
FAMILIES = __import__('synthetic_truth').apply(FAMILIES)
BUDGETS = {"hartmann6": 40, "styblinski_tang7": 50, "levy10": 50, "ackley15": 150}


def load_prior_fn(family, dim, prior_file, document_peaks=True):
    sys.path.insert(0, str(ROOT / "src"))
    from prior.landscape.landscape_generator import compile_prior, document_peak_points

    record = json.loads(Path(prior_file).read_text(encoding="utf-8"))
    compiled = compile_prior(record)
    names = [f"x{i}" for i in range(dim)]

    def prior_fn(x):
        return float(compiled.pi({names[i]: float(x[i]) for i in range(dim)}))

    # the document's own peaks seed the prior-maximizer search of continuous_bo
    if document_peaks:
        prior_fn.peak_candidates = np.asarray([[p[n] for n in names] for p in document_peak_points(record)], float)
    return prior_fn, float(compiled.alpha)


def random_trajectory(f, bounds, f_opt, budget, seed):
    rng = np.random.default_rng(seed)
    lo, hi = bounds[:, 0], bounds[:, 1]
    X = lo + (hi - lo) * rng.random((budget, bounds.shape[0]))
    fvals = np.array([f(x) for x in X])
    best = np.minimum.accumulate(fvals)
    return np.maximum(best - f_opt, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="output subdir under --out-root")
    ap.add_argument("--out-root", default="results/estimate_panel", help="output root (relative to the repo)")
    ap.add_argument("--arms", required=True,
                    help="comma list from {hapbo,vanilla,vanilla-peaks,random,turbo,"
                         "pi-bo,prior-to-mean,prior-warm-start,prior-greedy}")
    ap.add_argument("--prior-file", default=None,
                    help="prior.json document; required for the hapbo, vanilla-peaks and injection arms")
    ap.add_argument("--prior-id", default=None, help="label of the prior recorded in the outputs")
    ap.add_argument("--no-document-peaks", action="store_true",
                    help="do not seed the prior-maximizer search with the document's peak coordinates")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=0,
                    help="first seed to run (for per-seed job splitting; combine with "
                         "--seeds <k+1> to run exactly seed k under a distinct --label)")
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--n-init", type=int, default=3,
                    help="initial design size of the hapbo, vanilla, vanilla-peaks and prior-injection "
                         "arms (default 3, the synthetic panel's setting). TuRBO keeps its native "
                         "2*dim initial design.")
    ap.add_argument("--alpha-override", type=float, default=None,
                    help="fixed alpha for the hapbo arm instead of the compiled prior trust; the "
                         "prior.json file is not modified. When absent, the compiled alpha is used. "
                         "The vanilla-peaks arm takes the same alpha (it gates the prior-peak search, "
                         "not the acquisition).")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = [a for a in arms if a not in ("hapbo", "vanilla", "vanilla-peaks", "random", "turbo",
                                            *INJECTION_ARMS)]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}")
    prior_arms = [a for a in arms if a in ("hapbo", "vanilla-peaks", *INJECTION_ARMS)]
    if prior_arms and not args.prior_file:
        raise SystemExit(f"--prior-file is required for the {' and '.join(prior_arms)} "
                         f"arm{'s' if len(prior_arms) > 1 else ''}")
    prior_label = args.prior_id or (str(Path(args.prior_file).as_posix()) if args.prior_file else "")

    run_continuous_bo = None
    if prior_arms or "vanilla" in arms:
        sys.path.insert(0, str(ROOT / "src"))
        from bayesian_optimization.continuous_bo import run_continuous_bo as _rcb
        run_continuous_bo = _rcb
    run_turbo = None
    if "turbo" in arms:
        # import the module directly: the baselines package __init__ pulls in base.py, whose
        # `float | Mapping[...]` alias needs Python >= 3.10, while TuRBO runs in a python-3.9 env.
        sys.path.insert(0, str(ROOT / "src" / "baselines"))
        from turbo_continuous import run_turbo as _rt
        run_turbo = _rt

    out = ROOT / args.out_root / args.label
    out.mkdir(parents=True, exist_ok=True)
    partial_path = out / "trajectories.partial.json"
    traj = {}
    if partial_path.exists():
        traj = json.loads(partial_path.read_text(encoding="utf-8"))
        print(f"resuming from partial: "
              f"{ {f: {a: len(v) for a, v in d.items()} for f, d in traj.items()} }", flush=True)

    rows = []
    pi_bo_beta_used: dict = {}  # family -> beta of the pi-bo runs (ContinuousBOResult.pi_bo_beta)
    for family in families:
        dim, lo, hi, f, f_opt = FAMILIES[family]
        bounds = np.array([[lo, hi]] * dim, dtype=float)
        budget = BUDGETS[family]
        n_init = int(args.n_init)
        prior_fn = alpha = None
        alpha_compiled = None
        if prior_arms:
            prior_fn, alpha = load_prior_fn(family, dim, args.prior_file, not args.no_document_peaks)
            alpha_compiled = alpha
            if args.alpha_override is not None:
                alpha = float(args.alpha_override)
        traj.setdefault(family, {})
        print(f"\n== {family} (d={dim}, b{budget}, n_init={n_init}"
              + (f", alpha={alpha:.3f} [{prior_label}]" if alpha is not None else "")
              + (f" (compiled={alpha_compiled:.3f}, OVERRIDDEN)"
                 if alpha is not None and args.alpha_override is not None else "")
              + f") arms={arms} ==", flush=True)
        for arm in arms:
            done = traj[family].setdefault(arm, [])
            for seed in range(max(len(done), args.seed_start), args.seeds):
                if arm == "vanilla":
                    r = run_continuous_bo(f, bounds, f_opt, budget=budget, n_initial=n_init, seed=seed,
                                          n_restarts=args.restarts)
                    regret = list(map(float, r.simple_regret))
                elif arm == "hapbo":
                    # the seeded Sobol initial design of vanilla, then the alpha-pi shaped EI loop
                    r = run_continuous_bo(f, bounds, f_opt, budget=budget, n_initial=n_init, seed=seed,
                                          n_restarts=args.restarts, alpha=alpha, prior_fn=prior_fn)
                    regret = list(map(float, r.simple_regret))
                elif arm == "vanilla-peaks":
                    # same prior, alpha and seed as hapbo, so the same prior peaks join every candidate
                    # batch; the acquisition stays unshaped and the init is the vanilla Sobol design.
                    r = run_continuous_bo(f, bounds, f_opt, budget=budget, n_initial=n_init, seed=seed,
                                          n_restarts=args.restarts, alpha=alpha, prior_fn=prior_fn,
                                          shape_acquisition=False)
                    regret = list(map(float, r.simple_regret))
                elif arm in INJECTION_ARMS:
                    # same prior, peaks, candidate batches and seed as hapbo; only the injection site
                    # differs. prior-to-mean scales the standardized prior by the compiled alpha
                    r = run_continuous_bo(f, bounds, f_opt, budget=budget, n_initial=n_init, seed=seed,
                                          n_restarts=args.restarts, prior_fn=prior_fn,
                                          prior_strategy=arm, pi_bo_beta=PI_BO_BETA,
                                          prior_mean_beta=alpha_compiled)
                    regret = list(map(float, r.simple_regret))
                    if r.pi_bo_beta is not None:
                        pi_bo_beta_used[family] = r.pi_bo_beta
                elif arm == "random":
                    regret = list(map(float, random_trajectory(f, bounds, f_opt, budget, seed)))
                else:  # turbo — native baseline keeps its own 2*dim initialization
                    regret = list(map(float, run_turbo(
                        f, bounds, budget=budget, n_init=2 * dim, seed=seed, optimum_value=f_opt)))
                done.append(regret)
                print(f"  {arm} seed {seed}: final={regret[-1]:.4f}", flush=True)
                partial_path.write_text(json.dumps(traj), encoding="utf-8")

    for family, arm_map in traj.items():
        dim, lo, hi, f, f_opt = FAMILIES[family]
        for arm, seed_trajs in arm_map.items():
            for seed, regret in enumerate(seed_trajs):
                rows.append({"family": family, "dim": dim, "budget": BUDGETS[family],
                             "label": args.label, "prior_id": prior_label,
                             "arm": arm, "seed": seed, "final_regret": float(regret[-1])})
    (out / "trajectories.json").write_text(json.dumps(traj), encoding="utf-8")
    # Provenance: the alpha used by the multiplier and the hidden objective.
    provenance = {
        "label": args.label, "prior_id": prior_label, "arms": arms,
        "truth": __import__("synthetic_truth").world(),
        "document_peaks": not args.no_document_peaks,
        "families": families, "seeds": args.seeds, "seed_start": args.seed_start,
        "n_init": int(args.n_init), "restarts": int(args.restarts),
        "alpha_override": args.alpha_override,
    }
    if "vanilla-peaks" in arms:
        # vanilla-peaks uses alpha only for the peak search
        provenance["vanilla_peaks"] = {"shape_acquisition": False, "prior_peak_candidates": True,
                                       "document_peaks": not args.no_document_peaks}
    injection = [a for a in arms if a in INJECTION_ARMS]
    if "pi-bo" in arms:  # families whose pi-bo seeds were all reloaded from the partial file
        from bayesian_optimization.acquisition import pi_bo_beta
        for family in families:
            pi_bo_beta_used.setdefault(family, pi_bo_beta(BUDGETS[family], args.n_init))
    if injection:
        from bayesian_optimization.acquisition import PI_BO_PRIOR_EPSILON
        from bayesian_optimization.continuous_bo import DISTINCT_TOL
        # the alpha that prior-to-mean uses is the compiled one (alpha_override applies to hapbo only)
        provenance["injection"] = {
            "arms": injection, "candidate_set": CANDIDATE_SET,
            "distinct_tol": DISTINCT_TOL, "pi_bo_beta": PI_BO_BETA, "pi_bo_epsilon": PI_BO_PRIOR_EPSILON,
            "pi_bo_beta_resolved": pi_bo_beta_used,
            "pi_bo_initial_design": "prior mode, then n_init - 1 points sampled from the prior",
            "prior_mean_beta": "compiled alpha", "document_peaks": not args.no_document_peaks,
        }
    (out / "run_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    with (out / "continuous_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n=== synthetic arms [{args.label}] mean final regret (n={args.seeds}) ===", flush=True)
    header = f"{'family':<18}" + "".join(f"{a:>15}" for a in arms)
    print(header)
    for family in families:
        means = []
        for arm in arms:
            vals = [t[-1] for t in traj[family].get(arm, [])]
            means.append(float(np.mean(vals)) if vals else float("nan"))
        print(f"{family:<18}" + "".join(f"{m:>15.4f}" for m in means))
    print("SYNTHETIC_ARMS_DONE", flush=True)


if __name__ == "__main__":
    main()
