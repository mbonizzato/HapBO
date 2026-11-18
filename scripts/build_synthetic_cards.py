#!/usr/bin/env python3
"""Build the formula cards of the synthetic panel and the frozen LLM inputs rendered from them.

Steps (run in this order; each overwrites its outputs):

  specs   experiments/landscape_estimates/<fam>/formula_spec_rho90.json
          The perturbation draws are fixed per family (formula_estimate_lib._draws); the true function is the
          shifted benchmark (synthetic_truth, world ``shift``) and the model's location parameters carry the same
          shift; the knob t is calibrated so that Pearson(model, shifted truth) over the box equals 0.9.
  cards   experiments/landscape_estimates/<fam>/{card_formula_rho90_domain.txt, optimization_formula.txt,
          schema.json}: the shared card text every method receives (neutral framing, facts only) and the HapBO
          card sections 2-3 (search space, objective + open reasoning questions).
  well    experiments/landscape_estimates/<fam>/{card_formula_well{50,10}_domain.txt, formula_spec_well{50,10}.json}:
          cards of fidelity rho = .5 / .1 with respect to the well truth (see below). Slow (minutes per card).
  tasks   experiments/synthetic_cards/<card>/<fam>/TASK.md for card in rho90, well50, well10: the rendered task
          card (domain = formula card, search space = schema.json, optimization = optimization_formula.txt), plus
          experiments/synthetic_cards/{PSL_VOCABULARY.md, psl_schema.json}, the vocabulary and schema shown to the
          LLM next to TASK.md (copies of src/prior/expert_knowledge/{vocabulary_card.md, psl_schema.json}).

Well cards. q = optimum of the well truth T; c90 = optimum of the rho90 card model; perf(x) = (Tmax - T(x)) /
(Tmax - T(q)) with Tmax the maximum of T on a fixed 100k uniform sample. The card optimum c_target is the first
point with perf = target (.5 / .1) on the path Gamma: leg 1 = the ray from q through c90 to the box exit z_e;
leg 2 (only if perf never reaches the target on leg 1) = the segment from z_e to the nearest sample point with
perf <= target. Constants: the same functional form and print format as the rho90 card, location parameters
pinned so that the model optimum is c_target, every other constant moved by the family's fixed draws scaled by
one knob t >= t90 (t90 = the rho90 knob):
  Hartmann-6       the Gaussian nearest the t90 model optimum is moved until the model optimum is c_target; if
                   another basin is then deeper, its alpha is raised by the minimal factor that makes c_target the
                   global optimum (margin 0.005). When the knob cannot bring Pearson down to the target, every
                   A[k][j] of the rho90 constants is multiplied by (1 + t - t90) instead (narrower Gaussians).
  Styblinski-Tang  s_i = c_i - y*_i / lam_i (y*_i = global minimiser of y^4 - a_i y^2 + b_i y).
  Levy / Ackley    o = c_target / c = c_target; Ackley's b moves as b90 + 0.2 |db| (t - t90).
Stage A: the smallest t >= t90 with Pearson(card, T) on the sample = target (grid + bisection). Stage B (when the
correlation at c_target is already below the target for every t): t = t90 and c_target = the last point of Gamma
before the perf target where Pearson = target. The rule used and every calibration grid are recorded in the spec.

usage: python scripts/build_synthetic_cards.py {specs,cards,well,tasks} [--fams ...] [--keys well50,well10] [--dry]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

os.environ["SYNTH_TRUTH"] = "shift"  # specs are calibrated against the shifted benchmark (cards describe it)
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import formula_estimate_lib as FEL  # noqa: E402
import synthetic_truth as ST  # noqa: E402
from estimate_lib import FAMS  # noqa: E402
from prior.expert_knowledge.prompt_card import render_prompt_card, search_space_from_schema  # noqa: E402

LE = ROOT / "experiments" / "landscape_estimates"
CARDS_OUT = ROOT / "experiments" / "synthetic_cards"
VOCAB = ROOT / "src" / "prior" / "expert_knowledge"
FAMILIES = list(ST.FAMILIES)
BUDGETS = {"hartmann6": 40, "styblinski_tang7": 50, "levy10": 50, "ackley15": 150}
TASK_CARDS = ("rho90", "well50", "well10")
RHO_KEYS = ("rho90",)

MODEL_PARA = (
    "No expert intuition about the location of the optimum is available. An approximate analytic model of the "
    "objective is available. Its functional form is believed to describe the objective's general shape, but its "
    "parameter values are only approximately calibrated (fidelity unknown), so the locations and relative depths "
    "of its minima may differ from those of the true objective. The model is:\n\n{formula}"
)

OPTIMIZATION_TXT = (
    "Objective: a scalar black-box value, lower is better.\n"
    "Direction: minimize\n"
    "Evaluation budget: about {budget} evaluations.\n"
    "Known constraints or prior observations: none; the only information available is the approximate analytic "
    "model in the domain description above.\n\n"
    "Questions to consider before writing the prior:\n"
    "- Where in the search box are the model's low (good) regions, how many are there, and how do their depths "
    "compare?\n"
    "- Which parameters does the model value depend on most strongly, and which interact?\n"
    "- How much should an approximately calibrated model's minimum locations be trusted?"
)


def formula_text_clean(spec: dict) -> str:
    return FEL.formula_text(spec).replace("  nan  nan", "  -  -")


def domain_text(family: str, spec: dict) -> str:
    d, lo, hi = ST.BOUNDS[family]
    text = formula_text_clean(spec)
    assert "nan" not in text, family
    return (f"A {d}-parameter black-box objective, to be MINIMIZED; each input ranges [{lo:g}, {hi:g}]. "
            + MODEL_PARA.format(formula=text))


# ------------------------------------------------------------------------------------------------ specs / cards
def build_specs(fams) -> None:
    for fam in fams:
        for rho in RHO_KEYS:
            spec = FEL.build_spec(fam, rho)
            spec["shift_delta"] = ST.shift_delta(fam).tolist()
            FEL.spec_path(fam, rho).write_text(json.dumps(spec), encoding="utf-8")
            print(f"{fam} {rho}: t={spec['t']:.3f} pearson={spec['achieved_pearson']:.3f} "
                  f"spearman={spec['achieved_spearman']:.3f}", flush=True)


def build_cards(fams) -> None:
    for fam in fams:
        d, lo, hi = ST.BOUNDS[fam]
        fam_dir = LE / fam
        fam_dir.mkdir(parents=True, exist_ok=True)
        (fam_dir / "optimization_formula.txt").write_text(
            OPTIMIZATION_TXT.format(budget=BUDGETS[fam]), encoding="utf-8", newline="\n")
        (fam_dir / "schema.json").write_text(json.dumps({
            "dimensions": [{"name": f"x{i}", "type": "continuous", "range": [lo, hi], "scale": "linear"}
                           for i in range(d)]}, indent=2), encoding="utf-8", newline="\n")
        for rho_key in RHO_KEYS:
            spec = json.loads(FEL.spec_path(fam, rho_key).read_text(encoding="utf-8"))
            domain = domain_text(fam, spec)
            (fam_dir / f"card_formula_{rho_key}_domain.txt").write_text(domain, encoding="utf-8", newline="\n")
            print(f"{fam} {rho_key}: achieved P={spec['achieved_pearson']:.3f}, {len(domain)} chars")


# ------------------------------------------------------------------------------------------------ TASK.md
def render_task(fam: str, card: str) -> str:
    fd = LE / fam
    dims = json.loads((fd / "schema.json").read_text(encoding="utf-8"))["dimensions"]
    return render_prompt_card(domain=(fd / f"card_formula_{card}_domain.txt").read_text(encoding="utf-8"),
                              search_space=search_space_from_schema(dims),
                              optimization_description=(fd / "optimization_formula.txt").read_text(encoding="utf-8"))


def build_tasks(fams, cards=TASK_CARDS) -> None:
    CARDS_OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(VOCAB / "vocabulary_card.md", CARDS_OUT / "PSL_VOCABULARY.md")
    shutil.copyfile(VOCAB / "psl_schema.json", CARDS_OUT / "psl_schema.json")
    for card in cards:
        for fam in fams:
            o = CARDS_OUT / card / fam
            o.mkdir(parents=True, exist_ok=True)
            (o / "TASK.md").write_text(render_task(fam, card), encoding="utf-8", newline="\n")
            md5 = hashlib.md5((o / "TASK.md").read_bytes()).hexdigest()
            print(f"{card} {fam} -> {o.relative_to(ROOT).as_posix()}/TASK.md md5 {md5}")


# ------------------------------------------------------------------------------------------------ well cards
KEYS = {"well50": 0.5, "well10": 0.1}
TOL = 0.005
N_SCAN = 4001
N_POINTS = 100_000
T_GRID_EXTRA = np.r_[0.0, np.geomspace(0.002, 64.0, 160)]
T_GRID_H6 = np.r_[0.0, np.geomspace(0.005, 32.0, 48)]   # coarser: each H6 point needs a basin relocation
WELL_H6_MARGIN = 0.005
H6_SCREEN = 4000
H6_STARTS = 10
H6_MAX_BOOST = 3.0


def sample(fam: str, n: int = N_POINTS) -> np.ndarray:
    """Fixed uniform evaluation sample of a family (frozen seed string)."""
    d, lo, hi = ST.BOUNDS[fam]
    seed = int(hashlib.sha256(("agentic-fid:" + fam).encode()).hexdigest()[:8], 16)
    return lo + (hi - lo) * np.random.default_rng(seed).random((n, d))


def minimize_numeric(fn, fam, X, vals, extra_starts=(), n_starts=64):
    """Global minimum of a box-bounded fn: best n_starts sample points + extra starts -> L-BFGS-B, then Powell
    polish of the 4 best local minima. Returns (f_min, x_min)."""
    from scipy.optimize import minimize
    d, lo, hi = ST.BOUNDS[fam]
    bnds = [(lo, hi)] * d
    f1 = lambda x: float(fn(np.clip(x, lo, hi)[None, :])[0])  # noqa: E731
    starts = [X[i] for i in np.argsort(vals)[:n_starts]] + [np.clip(np.asarray(s, float), lo, hi) for s in extra_starts]
    res = []
    for s in starts:
        r = minimize(f1, s, method="L-BFGS-B", bounds=bnds)
        fs = f1(s)
        res.append((float(r.fun), np.clip(r.x, lo, hi)) if float(r.fun) <= fs else (fs, np.asarray(s, float)))
    res.sort(key=lambda t: t[0])
    best = res[0]
    for v, x in res[:4]:
        r = minimize(f1, x, method="Powell", bounds=bnds, options={"xtol": 1e-10, "ftol": 1e-14, "maxfev": 40000})
        if float(r.fun) < best[0]:
            best = (float(r.fun), np.clip(r.x, lo, hi))
    return best[0], np.asarray(best[1], float)


def context(fam):
    d, lo, hi = ST.BOUNDS[fam]
    W, sq = hi - lo, np.sqrt(d)
    X = sample(fam)
    data = ST.load_well_truth()
    t_well = data["well"][fam]
    M = ST.card_model(fam)
    fm = M(X)
    Tv = ST.window_eval(fam, X, t_well, M=M)
    T = ST.well_truth(fam)
    fq, q = ST.well_optimum(fam)
    if "card_model" in data:
        c90 = np.asarray(data["card_model"][fam]["x_opt"], float)
    else:
        c90 = minimize_numeric(M, fam, X, fm)[1]
    zq = (q - lo) / W
    u = (c90 - q) / W
    if np.linalg.norm(u) < 1e-12:   # q == c90: the well construction's basin ray
        u = np.asarray(t_well["basin_ray"], float)
    u = u / np.linalg.norm(u)
    s_exit = min(((1 - zq[i]) / u[i]) if u[i] > 0 else (-zq[i] / u[i]) for i in range(d) if abs(u[i]) > 1e-12) / sq
    t90 = json.loads((LE / fam / "formula_spec_rho90.json").read_text(encoding="utf-8"))["t"]
    Tmax = float(Tv.max())
    return dict(fam=fam, d=d, lo=lo, hi=hi, W=W, sq=sq, X=X, Tv=Tv, T=T, fq=fq, q=q, c90=c90, zq=zq, u=u,
                s_exit=float(s_exit), t90=float(t90), Tmax=Tmax, draws=FEL._draws(fam),
                perf_sample=(Tmax - Tv) / (Tmax - fq), Xs=X[:20000])


def perf(ctx, Xn):
    return (ctx["Tmax"] - ctx["T"](np.atleast_2d(Xn))) / (ctx["Tmax"] - ctx["fq"])


def diag(ctx, a, b):
    return float(np.linalg.norm((np.asarray(a) - np.asarray(b)) / ctx["W"]) / ctx["sq"])


def _bisect(pred, a, b, n=60):
    """pred(a) False, pred(b) True -> boundary."""
    for _ in range(n):
        m = 0.5 * (a + b)
        if pred(m):
            b = m
        else:
            a = m
    return float(b)


def build_path(ctx, target):
    """Gamma for this target: point(s) for s in [0, s_end] and s_perf (first s with perf = target)."""
    zq, u, sq, lo, W = ctx["zq"], ctx["u"], ctx["sq"], ctx["lo"], ctx["W"]
    se = ctx["s_exit"]
    path = {"s_exit": se, "leg2": None}

    def leg1(s):
        return lo + W * (zq + np.minimum(s, se) * sq * u)
    s = np.linspace(0.0, se, N_SCAN)
    pr = perf(ctx, lo + W * (zq + (s * sq)[:, None] * u))
    path["leg1_perf_min"] = float(pr.min())
    path["leg1_perf_at_exit"] = float(pr[-1])
    path["perf_at_c90"] = float(perf(ctx, ctx["c90"])[0])
    hit = np.where(pr <= target)[0]
    if len(hit):
        path.update(point=leg1, s_end=se, kind="ray",
                    s_perf=_bisect(lambda v: perf(ctx, leg1(v))[0] <= target, s[hit[0] - 1], s[hit[0]]))
        return path
    Z = (ctx["X"] - lo) / W
    ze = zq + se * sq * u
    cand = np.where(ctx["perf_sample"] <= target)[0]
    j = int(cand[np.argmin(np.linalg.norm(Z[cand] - ze, axis=1))])
    zs = Z[j]
    L2 = float(np.linalg.norm(zs - ze) / sq)

    def point(s):
        if s <= se:
            return leg1(s)
        lam = min((s - se) / L2, 1.0)
        return lo + W * (ze + lam * (zs - ze))
    s2 = se + np.linspace(0.0, L2, N_SCAN)
    p2 = np.array([perf(ctx, point(v))[0] for v in s2])
    k = int(np.where(p2 <= target)[0][0])
    path.update(point=point, s_end=se + L2, kind="leg2",
                s_perf=_bisect(lambda v: perf(ctx, point(v))[0] <= target, s2[k - 1], s2[k]),
                leg2={"z_exit": ze.tolist(), "sample_index": j, "z_sample": zs.tolist(), "length_diag": L2,
                      "perf_sample_point": float(ctx["perf_sample"][j]),
                      "n_sample_points_with_perf_le_target": int(len(cand))})
    return path


def knob_params(ctx, t):
    fam = ctx["fam"]
    if ctx.get("width_knob"):  # Hartmann-6 second knob: rho90 constants, every Gaussian narrowed by (1 + t - t90)
        p = FEL._params(fam, ctx["t90"], ctx["draws"])
        p["A"] = (np.asarray(p["A"]) * (1.0 + (t - ctx["t90"]))).tolist()
        return p
    p = FEL._params(fam, t, ctx["draws"])
    if fam == "ackley15":
        b90 = FEL._params(fam, ctx["t90"], ctx["draws"])["b"]
        p["b"] = float(b90 + 0.2 * abs(ctx["draws"]["db"]) * (t - ctx["t90"]))
    return p


def _st7_ystar(a, b):
    out = []
    for ai, bi in zip(a, b):
        r = np.roots([4.0, 0.0, -2.0 * ai, bi])
        r = r[np.abs(r.imag) < 1e-9].real
        out.append(float(r[np.argmin(r ** 4 - ai * r ** 2 + bi * r)]))
    return np.asarray(out)


def _h6_relocate(ctx, p, c, k=None):
    from scipy.optimize import minimize
    fam = "hartmann6"
    alpha, A, P = (np.array(p[key], float) for key in ("alpha", "A", "P"))
    M = lambda Z: FEL._eval(fam, {"alpha": alpha, "A": A, "P": P}, Z)  # noqa: E731
    f1 = lambda x: float(M(x[None, :])[0])  # noqa: E731
    Xs = ctx["Xs"][:H6_SCREEN]
    bnd = [(0.0, 1.0)] * 6

    def local_minima():
        mv = M(Xs)
        return [minimize(f1, Xs[i], method="L-BFGS-B", bounds=bnd) for i in np.argsort(mv)[:H6_STARTS]]
    if k is None:                                              # the Gaussian that carries the model optimum
        xs = min(local_minima(), key=lambda r: r.fun).x
        k = int(np.argmin(np.linalg.norm(P - xs[None, :], axis=1)))
        P[k] += c - xs
    else:
        P[k] = np.array(c, float)
    boost, second = 1.0, np.inf
    for _outer in range(200):
        for _ in range(100):                                     # pin: the model's local minimum at c exactly
            r = minimize(f1, c, method="BFGS", options={"gtol": 1e-12})
            P[k] += c - r.x
            if np.linalg.norm(c - r.x) < 1e-11:
                break
        others = [float(r.fun) for r in local_minima() if np.linalg.norm(r.x - c) / np.sqrt(6) > 1e-3]
        second = min(others) if others else np.inf
        if f1(c) <= second - WELL_H6_MARGIN:
            break
        if boost > H6_MAX_BOOST:
            break
        gk = alpha[k] * np.exp(-np.sum(A[k] * (c - P[k]) ** 2))
        need = (f1(c) - (second - WELL_H6_MARGIN)) / max(gk, 1e-12)
        fac = min(max(1.02, 1.0 + 1.05 * need), 1.01 * H6_MAX_BOOST / boost)
        alpha[k] *= fac
        boost *= fac
    return {"alpha": alpha.tolist(), "A": A.tolist(), "P": P.tolist()}, {"gaussian_moved": k, "alpha_boost": boost,
            "second_local_min": second, "valid": bool(f1(c) <= second - WELL_H6_MARGIN)}


def pinned(ctx, t, c):
    fam = ctx["fam"]
    p = knob_params(ctx, t)
    info = {}
    if fam == "hartmann6":
        c = np.asarray(c, float)
        cache = ctx.setdefault("_h6_k", {})
        key = tuple(np.round(c, 12))
        if key not in cache:  # the moved Gaussian is chosen once per c, at t90, and kept for every t
            cache[key] = _h6_relocate(ctx, knob_params(ctx, ctx["t90"]), c)[1]["gaussian_moved"]
        p, info = _h6_relocate(ctx, p, c, k=cache[key])
    elif fam == "styblinski_tang7":
        ys = _st7_ystar(p["a"], p["b"])
        p["s"] = (np.asarray(c) - ys / np.asarray(p["lam"])).tolist()
    elif fam == "levy10":
        p["o"] = list(map(float, c))
    else:
        p["c"] = list(map(float, c))
    return p, info


def pearson_of(ctx, p):
    return float(np.corrcoef(FEL._eval(ctx["fam"], p, ctx["X"]), ctx["Tv"])[0, 1])


def pear(ctx, t, c):
    """Pearson of the pinned card at (t, c); NaN when the pin is invalid (optimum not global within the boost cap)."""
    p, info = pinned(ctx, t, c)
    return pearson_of(ctx, p) if info.get("valid", True) else float("nan")


def calibrate(ctx, target, path, log=print):
    fam, t90 = ctx["fam"], ctx["t90"]
    s_p = path["s_perf"]
    c = path["point"](s_p)
    tg = t90 + (T_GRID_EXTRA if fam != "hartmann6" else T_GRID_H6)
    P = np.array([pear(ctx, t, c) for t in tg])
    n_invalid = int(np.isnan(P).sum())
    tg, P = tg[~np.isnan(P)], P[~np.isnan(P)]
    log(f"  {fam} {target}: Pearson(pinned at s_perf={s_p:.4f}) over t: t90 {P[0]:.3f}, min {P.min():.3f}, "
        f"max {P.max():.3f}")
    res = {"pearson_t_grid_at_s_perf": {"t": tg.tolist(), "pearson": P.tolist(), "n_invalid_dropped": n_invalid}}
    above = P >= target
    f = lambda t: pear(ctx, t, c) - target  # noqa: E731
    rejected = []
    for i in np.where(above[1:] != above[:-1])[0]:   # smallest t first; a jump (well switch) is not a root
        sgn = np.sign(P[i] - target)
        a, b = tg[i], tg[i + 1]
        for _ in range(30 if fam == "hartmann6" else 50):
            m = 0.5 * (a + b)
            if np.sign(f(m)) == sgn:
                a = m
            else:
                b = m
        t_star = 0.5 * (a + b)
        if abs(f(t_star)) <= TOL / 5:
            res.update(rule=path["kind"] if path["kind"] == "leg2" else "exact", s=s_p, t=float(t_star), c=c,
                       rejected_discontinuities=rejected)
            return res
        rejected.append({"t": float(t_star), "pearson_left": float(f(a) + target),
                         "pearson_right": float(f(b) + target)})
    res["rejected_discontinuities"] = rejected
    if not rejected and P.max() > target:
        res.update(rule="infeasible_pearson_too_high", s=s_p, t=None, c=c)
        return res
    s = np.linspace(0.0, s_p, 101 if fam == "hartmann6" else 401)
    Ps = np.array([pear(ctx, t90, path["point"](v)) for v in s])
    ok = np.where(Ps >= target)[0]
    res["pearson_t90_along_gamma"] = {"s": s.tolist(), "pearson": Ps.tolist()}
    if not len(ok) or ok[-1] == len(s) - 1:
        res.update(rule="infeasible_backoff", s=None, t=None, c=None)
        return res
    j = int(ok[-1])
    sb = _bisect(lambda v: not pear(ctx, t90, path["point"](v)) >= target, s[j], s[j + 1], n=40)
    log(f"  {fam} {target}: backoff to s={sb:.4f} (Pearson(t90) along Gamma max {Ps.max():.3f})")
    res.update(rule="backoff", s=sb, t=t90, c=path["point"](sb))
    return res


def well_card_text(fam, p):
    d = ST.BOUNDS[fam][0]
    return domain_text(fam, {"params": p, "family": fam, "dim": d})


def verify_card(ctx, text, c_target):
    from scipy import stats
    fam = ctx["fam"]
    pp = ST.card_params(fam, text)
    M = lambda X: FEL._eval(fam, pp, np.atleast_2d(np.asarray(X, float)))  # noqa: E731
    mv = M(ctx["X"])
    f_min, x_min = minimize_numeric(M, fam, ctx["X"], mv, extra_starts=[c_target])
    return {"card_params_parsed": pp, "model_opt_x": x_min.tolist(), "model_opt_f": float(f_min),
            "model_opt_dist_to_c_target_diag": diag(ctx, x_min, c_target),
            "pearson": float(np.corrcoef(mv, ctx["Tv"])[0, 1]),
            "spearman": float(stats.spearmanr(mv, ctx["Tv"]).statistic),
            "perf_at_model_opt": float(perf(ctx, x_min)[0]),
            "model_opt_dist_to_q_diag": diag(ctx, x_min, ctx["q"])}


def _js(v):
    if isinstance(v, dict):
        return {k: _js(x) for k, x in v.items() if not callable(x)}
    if isinstance(v, (list, tuple)):
        return [_js(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def build_well(fam, key, dry=False, log=print):
    target = KEYS[key]
    ctx = context(fam)
    path = build_path(ctx, target)
    log(f"{fam} {key}: path {path['kind']} s_exit={path['s_exit']:.4f} s_perf={path['s_perf']:.4f} "
        f"(leg1 perf min {path['leg1_perf_min']:.3f}, at exit {path['leg1_perf_at_exit']:.3f})")
    cal = calibrate(ctx, target, path, log)
    if cal["rule"] == "infeasible_pearson_too_high" and fam == "hartmann6":
        log(f"  {fam} {key}: knob cannot decorrelate enough -> width knob (A x (1 + t - t90))")
        ctx["width_knob"] = True
        cal = calibrate(ctx, target, path, log)
        cal["rule"] = cal["rule"] + "+width"
    rec = {"key": key, "family": fam, "target": target, "truth": "well", "rule": cal["rule"],
           "spec_exact": cal["rule"] == "exact",
           "q": ctx["q"].tolist(), "T_q": ctx["fq"], "Tmax_sample": ctx["Tmax"], "c90": ctx["c90"].tolist(),
           "ray_unit_dir": ctx["u"].tolist(), "path": {k: v for k, v in path.items() if k != "point"},
           "s_perf_c_target": path["s_perf"], "perf_c_at_s_perf": float(perf(ctx, path["point"](path["s_perf"]))[0]),
           "sample_perf_quantiles": dict(zip(["q01", "q05", "q25", "q50", "q75"], np.quantile(
               ctx["perf_sample"], [.01, .05, .25, .5, .75]).tolist())),
           "sample_frac_perf_le_target": float(np.mean(ctx["perf_sample"] <= target)),
           "t90": ctx["t90"], "draw_seed_string": f"formula:{fam}", "draws": ctx["draws"],
           "sample": "100k uniform points, seed sha256('agentic-fid:<fam>')",
           "calibration": {k: v for k, v in cal.items() if k not in ("c",)}}
    if cal["c"] is None or cal["t"] is None:
        rec["status"] = "NO CARD (infeasible)"
        log(f"  {fam} {key}: infeasible ({cal['rule']})")
        return rec, None
    c = np.asarray(cal["c"], float)
    p, info = pinned(ctx, cal["t"], c)
    text = well_card_text(fam, p)
    v = verify_card(ctx, text, c)
    rec.update(c_target=c.tolist(), s_c_target=cal["s"], t=cal["t"], pin_info=info, params_full=p,
               perf_c_target=float(perf(ctx, c)[0]), dist_c_target_q_diag=diag(ctx, c, ctx["q"]),
               dist_c_target_c90_diag=diag(ctx, c, ctx["c90"]), pearson_full_precision=pearson_of(ctx, p),
               verify_parsed_card=v,
               checks={"opt_within_1e-3_diag": v["model_opt_dist_to_c_target_diag"] <= 1e-3,
                       "pearson_within_tol": abs(v["pearson"] - target) <= TOL})
    rec["status"] = "OK" if all(rec["checks"].values()) else "CHECK FAILED"
    log(f"  {fam} {key}: rule={cal['rule']} |c-q|={rec['dist_c_target_q_diag']:.4f} perf={rec['perf_c_target']:.4f}"
        f" t={cal['t']:.4f} | parsed card: opt dist {v['model_opt_dist_to_c_target_diag']:.2e}, "
        f"P {v['pearson']:.4f}, S {v['spearman']:.4f} -> {rec['status']}")
    if not dry:
        d = LE / fam
        (d / f"card_formula_{key}_domain.txt").write_text(text, encoding="utf-8", newline="\n")
        (d / f"formula_spec_{key}.json").write_text(json.dumps(_js(rec), indent=1), encoding="utf-8", newline="\n")
    return rec, text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["specs", "cards", "well", "tasks"])
    ap.add_argument("--fams", default=",".join(FAMILIES))
    ap.add_argument("--keys", default="well50,well10", help="well step: card keys to build")
    ap.add_argument("--cards", default=",".join(TASK_CARDS), help="tasks step: cards to render")
    ap.add_argument("--dry", action="store_true", help="well step: report only, write nothing")
    a = ap.parse_args()
    fams = [f for f in a.fams.split(",") if f]
    assert set(fams) <= set(FAMS), fams
    if a.step == "specs":
        build_specs(fams)
    elif a.step == "cards":
        build_cards(fams)
    elif a.step == "well":
        for fam in fams:
            for key in a.keys.split(","):
                build_well(fam, key, dry=a.dry)
    else:
        build_tasks(fams, [c for c in a.cards.split(",") if c])


if __name__ == "__main__":
    main()
