"""Cortical EXPLORATION PERFORMANCE (OptimizeNeurostim.m, perf_explore = MPm(P_max)/mMPm), applied uniformly
to every method's observed query sequence.

At each query q, a GP is fitted to the responses OBSERVED so far (never the true means); P_max(q) = the tested
electrode with the highest GP posterior mean; score(q) = true mean response at P_max(q) / best true mean of the map.
The true means only grade the recommendation; the recommender never sees them.

GP as in mbonizzato/OptimizeNeurostim matlab_GPML/OptimizeNeurostim.m (rat settings): zero mean, Matern-5/2 ARD
on the electrode grid coordinates (ch2xy = channel_x, channel_y), Gaussian likelihood, y = observed /
max response seen so far; hyperparameters MAP under smooth box priors -> here bounded maximum marginal likelihood:
lengthscales in [0.01, 3], signal sd in [0.01, 100], noise sd in [0.001, 0.1]. Deviation: GPML's
minimize(hyp, ..., -10) does 10 function evaluations warm-started from the previous query; here L-BFGS-B
(<= 100 evaluations) warm-started from the previous query's solution plus one restart from the GPML initial
hyp (ell = sf = 1, sn clipped to 0.1), keeping the better marginal likelihood. Ties: first tested electrode
(GPML: random among ties).

Writes <out>/<study>.json = {"<problem>/<method>/<budget>/<seed>": [score_q1, ..., score_qB], ...}.
usage: python scripts/score_exploration.py --results results/studies/cortical --out results/exploration_scores/cortical
           --data benchmark/NeuroMOSAIC/dataset1/rat --studies human-map llm-geometry llm-expert-hint human-map-all
           human-map-colabo human-map-llambo human-map-lgbo human-map-bora [--workers N]
"""
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

BOUNDS = [(math.log(0.01), math.log(3.0)), (math.log(0.01), math.log(3.0)),
          (math.log(0.01), math.log(100.0)), (math.log(0.001), math.log(0.1))]
INIT = np.array([0.0, 0.0, 0.0, math.log(0.1)])
S5 = math.sqrt(5.0)


def _kernel(A, B, ell, sf2):
    d = (A[:, None, :] - B[None, :, :]) / ell
    r = np.sqrt(np.maximum((d ** 2).sum(-1), 0.0))
    return sf2 * (1.0 + S5 * r + 5.0 / 3.0 * r ** 2) * np.exp(-S5 * r)


def _nlml(theta, X, y):
    ell = np.exp(theta[:2]); sf2 = math.exp(2 * theta[2]); sn2 = math.exp(2 * theta[3])
    K = _kernel(X, X, ell, sf2) + (sn2 + 1e-10) * np.eye(len(X))
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return 1e10
    a = np.linalg.solve(L.T, np.linalg.solve(L, y))
    return float(0.5 * y @ a + np.log(np.diag(L)).sum() + 0.5 * len(X) * math.log(2 * math.pi))


def _fit(X, y, warm):
    best = None
    for start in ([warm] if warm is not None else []) + [INIT]:
        r = minimize(_nlml, np.clip(start, [b[0] for b in BOUNDS], [b[1] for b in BOUNDS]), args=(X, y),
                     method="L-BFGS-B", bounds=BOUNDS, options={"maxfun": 100})
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def _post_mean(theta, X, y, Xs):
    ell = np.exp(theta[:2]); sf2 = math.exp(2 * theta[2]); sn2 = math.exp(2 * theta[3])
    K = _kernel(X, X, ell, sf2) + (sn2 + 1e-10) * np.eye(len(X))
    return _kernel(Xs, X, ell, sf2) @ np.linalg.solve(K, y)


def score_run(path: Path):
    st = path / "status.json"
    ev = path / "evaluations.jsonl"
    if not (st.is_file() and ev.is_file()) or json.loads(st.read_text()).get("status") != "completed":
        return None
    recs = [json.loads(l) for l in ev.read_text().splitlines() if l.strip()]
    xs = np.array([[r["candidate"]["channel_x"], r["candidate"]["channel_y"]] for r in recs], float)
    obs = np.array([r["objective"] for r in recs], float)
    gt = np.array([r["ground_truth_objective"] for r in recs], float)
    return xs, obs, gt


def explore_scores(xs, obs, gt_by_xy, opt):
    scores, warm = [], None
    for q in range(1, len(obs) + 1):
        X, o = xs[:q], obs[:q]
        mx = np.max(o) if np.max(o) > 0 else 1.0
        y = o / mx
        warm = _fit(X, y, warm)
        tested = np.unique(X, axis=0)
        mu = _post_mean(warm, X, y, tested)
        best = tested[int(np.argmax(mu))]
        scores.append(gt_by_xy[(int(best[0]), int(best[1]))] / opt)
    return scores


def _work(args):
    key, path, opt, gt_map = args
    r = score_run(Path(path))
    if r is None:
        return key, None
    xs, obs, gt = r
    return key, explore_scores(xs, obs, {tuple(map(int, k.split(","))): v for k, v in gt_map.items()}, opt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/studies/cortical"))
    ap.add_argument("--out", type=Path, default=Path("results/exploration_scores/cortical"))
    ap.add_argument("--studies", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--data", type=Path, default=Path("benchmark/NeuroMOSAIC/dataset1/rat"), help="NeuroMOSAIC dataset1/rat directory (.mat files)")
    a = ap.parse_args()
    import sys
    sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src")]
    from datasets.neuromosaic_dataset1_rat import load_neuromosaic_dataset1_rat_problem
    truth = {}  # problem_id -> (opt, {"x,y": true mean})
    a.out.mkdir(parents=True, exist_ok=True)
    for study in a.studies:
        outf = a.out / f"{study}.json"
        if outf.is_file():
            print("skip (exists)", study, flush=True)
            continue
        jobs = []
        for pdir in sorted((a.results / study / "runs").glob("neuromosaic-dataset1-rat*/rat*")):
            if pdir.name not in truth:
                rat, slug = pdir.name.split("-", 1)
                muscle = "".join(w.capitalize() for w in slug.split("-"))
                p = load_neuromosaic_dataset1_rat_problem(subject_path=next(a.data.glob(f"{rat}_*.mat")),
                                                         muscle=muscle, subject=rat)
                gtm = {f"{c['channel_x']},{c['channel_y']}": float(v)
                       for c, v in zip(p.native_candidates, p.ground_truth)}
                truth[pdir.name] = (float(np.max(p.ground_truth)), gtm)
            opt, gtm = truth[pdir.name]
            for rd in sorted(pdir.glob("*/budget-*/seed-*")):
                key = f"{pdir.name}/{rd.parent.parent.name}/{rd.parent.name}/{rd.name}"
                jobs.append((key, str(rd), opt, gtm))
        res = {}
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for key, sc in ex.map(_work, jobs, chunksize=4):
                if sc is not None:
                    res[key] = sc
        outf.write_text(json.dumps(res), encoding="utf-8")
        print(f"{study}: {len(res)} runs scored", flush=True)


if __name__ == "__main__":
    main()
