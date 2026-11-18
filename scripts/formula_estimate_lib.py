#!/usr/bin/env python3
"""Formula card models of the synthetic panel: perturbed copies of the test functions.

The card model f_tilde is the same analytic family as the test function with perturbed
parameters (no additive field). A single knob t >= 0 scales all parameter perturbations
(random draws fixed per family); t is calibrated by bisection so that the Pearson correlation
of f_tilde and f over the uniform box equals the target rho = 0.9 (``RHOS``).

Perturbations per family (all draws fixed by a per-family seed, scaled linearly by t):
  hartmann6         Gaussian centers P displaced, depths alpha and widths A rescaled
  styblinski_tang7  per-axis affine y = lam * (x - s), well curvature a, tilt b (b can change
                    sign at large t, which moves the deeper well to the other side)
  levy10            optimum position o displaced, oscillation amplitude/frequency scaled
  ackley15          basin center c displaced, envelope decay b and cosine frequency
                    kappa rescaled

The spec JSON stores the draws, t and the achieved correlations; `estimate_fn(spec)`
reconstructs f_tilde; `formula_text(spec)` renders the model text, without the function's
name, that is printed on the elicitation card.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from estimate_lib import FAMS, RHOS  # true functions, bounds, optima, rho targets

SPEC_DIR = Path(__file__).resolve().parents[1] / "experiments" / "landscape_estimates"
N_FIT = 20_000
N_HELDOUT = 4_096


def _seed_of(family: str) -> int:
    return int(hashlib.sha256(("formula:" + family).encode()).hexdigest()[:8], 16) % (2 ** 31)


def _draws(family: str) -> dict:
    d = FAMS[family][0]
    rng = np.random.default_rng(_seed_of(family))
    if family == "hartmann6":
        return {"dP": rng.uniform(-0.35, 0.35, size=(4, 6)).tolist(),
                "dalpha": rng.uniform(-0.5, 0.5, size=4).tolist(),
                "dA": rng.uniform(-0.5, 0.5, size=(4, 6)).tolist()}
    if family == "styblinski_tang7":
        return {"s": rng.uniform(-1.5, 1.5, size=d).tolist(),
                "dlam": rng.uniform(-0.35, 0.35, size=d).tolist(),
                "da": rng.uniform(-0.4, 0.4, size=d).tolist(),
                "u": rng.uniform(0.2, 1.0, size=d).tolist()}
    if family == "levy10":
        return {"do": rng.uniform(-5.0, 5.0, size=d).tolist(),
                "damp": rng.uniform(-0.6, 0.6, size=d - 1).tolist(),
                "dkap": rng.uniform(-0.4, 0.4, size=d - 1).tolist()}
    if family == "ackley15":
        return {"dc": rng.uniform(-12.0, 12.0, size=d).tolist(),
                "db": rng.uniform(-0.5, 0.5),
                "dkap": rng.uniform(-0.5, 0.5, size=d).tolist()}
    raise KeyError(family)


def _params(family: str, t: float, draws: dict) -> dict:
    """Perturbed parameter values at knob t (what the formula card prints). The model's location
    parameters carry the hidden shift of the true function (synthetic_truth.shift_delta)."""
    p = _params_unshifted(family, t, draws)
    from synthetic_truth import shift_delta

    dl = shift_delta(family)
    if family == "hartmann6":
        p["P"] = np.clip(np.asarray(p["P"]) + dl[None, :], 0.02, 0.98).tolist()
    else:
        key = {"styblinski_tang7": "s", "levy10": "o", "ackley15": "c"}[family]
        p[key] = (np.asarray(p[key]) + dl).tolist()
    return p


def _params_unshifted(family: str, t: float, draws: dict) -> dict:
    if family == "hartmann6":
        from estimate_lib import _HA, _HM, _HP
        P = np.clip(_HP + t * np.asarray(draws["dP"]), 0.02, 0.98)
        alpha = _HA * np.clip(1.0 + t * np.asarray(draws["dalpha"]), 0.2, 5.0)
        A = _HM * np.clip(1.0 + t * np.asarray(draws["dA"]), 0.25, 4.0)
        return {"alpha": alpha.tolist(), "A": A.tolist(), "P": P.tolist()}
    if family == "styblinski_tang7":
        return {"s": (t * np.asarray(draws["s"])).tolist(),
                "lam": (1.0 + t * np.asarray(draws["dlam"])).clip(0.3, 3.0).tolist(),
                "a": (16.0 * (1.0 + t * np.asarray(draws["da"]))).clip(4.0, 40.0).tolist(),
                "b": (5.0 * (1.0 - 2.0 * t * np.asarray(draws["u"]))).tolist()}
    if family == "levy10":
        return {"o": (1.0 + t * np.asarray(draws["do"])).tolist(),
                "amp": (10.0 * (1.0 + t * np.asarray(draws["damp"]))).clip(1.0, 40.0).tolist(),
                "kap": (1.0 + t * np.asarray(draws["dkap"])).clip(0.2, 3.0).tolist()}
    if family == "ackley15":
        return {"c": (t * np.asarray(draws["dc"])).tolist(),
                "b": float(np.clip(0.2 * (1.0 + t * draws["db"]), 0.05, 0.6)),
                "kap": (1.0 + t * np.asarray(draws["dkap"])).clip(0.2, 3.0).tolist()}
    raise KeyError(family)


def _eval(family: str, params: dict, X: np.ndarray) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, float))
    if family == "hartmann6":
        alpha = np.asarray(params["alpha"])
        A = np.asarray(params["A"])
        P = np.asarray(params["P"])
        inner = np.sum(A[None, :, :] * (X[:, None, :] - P[None, :, :]) ** 2, axis=2)
        return -np.sum(alpha[None, :] * np.exp(-inner), axis=1)
    if family == "styblinski_tang7":
        s = np.asarray(params["s"])
        lam = np.asarray(params["lam"])
        a = np.asarray(params["a"])
        b = np.asarray(params["b"])
        y = lam[None, :] * (X - s[None, :])
        return 0.5 * np.sum(y ** 4 - a[None, :] * y ** 2 + b[None, :] * y, axis=1)
    if family == "levy10":
        o = np.asarray(params["o"])
        amp = np.asarray(params["amp"])
        kap = np.asarray(params["kap"])
        w = 1.0 + (X - o[None, :]) / 4.0
        t1 = np.sin(np.pi * w[:, 0]) ** 2
        t3 = (w[:, -1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[:, -1]) ** 2)
        mid = np.sum((w[:, :-1] - 1.0) ** 2
                     * (1.0 + amp[None, :] * np.sin(kap[None, :] * np.pi * w[:, :-1] + 1.0) ** 2),
                     axis=1)
        return t1 + mid + t3
    if family == "ackley15":
        c = np.asarray(params["c"])
        b = params["b"]
        kap = np.asarray(params["kap"])
        d = X.shape[1]
        Y = X - c[None, :]
        return (-20.0 * np.exp(-b * np.sqrt(np.sum(Y ** 2, axis=1) / d))
                - np.exp(np.sum(np.cos(kap[None, :] * 2.0 * np.pi * Y), axis=1) / d)
                + 20.0 + np.e)
    raise KeyError(family)


def _fit_sample(family: str):
    d, lo, hi, f, _, _ = FAMS[family]
    rng = np.random.default_rng(_seed_of(family) + 1)
    X = lo + (hi - lo) * rng.random((N_FIT, d))
    return X, f(X)


def calibrate_t(family: str, target_rho: float, draws: dict) -> float:
    X, fv = _fit_sample(family)

    def rho_of(t: float) -> float:
        return float(np.corrcoef(fv, _eval(family, _params(family, t, draws), X))[0, 1])

    t_hi = 1.0
    while rho_of(t_hi) > target_rho and t_hi < 64.0:
        t_hi *= 2.0
    if rho_of(t_hi) > target_rho:
        return t_hi  # family cannot decorrelate further; report achieved
    t_lo = 0.0
    for _ in range(40):
        mid = 0.5 * (t_lo + t_hi)
        if rho_of(mid) > target_rho:
            t_lo = mid
        else:
            t_hi = mid
    return 0.5 * (t_lo + t_hi)


def build_spec(family: str, rho_key: str) -> dict:
    d, lo, hi, f, f_opt, x_opt = FAMS[family]
    draws = _draws(family)
    t = calibrate_t(family, RHOS[rho_key], draws)
    params = _params(family, t, draws)
    spec = {"variant": "formula", "family": family, "rho_key": rho_key,
            "rho_target": RHOS[rho_key], "t": t, "dim": d, "lo": lo, "hi": hi,
            "draws": draws, "params": params, "f_opt": f_opt}
    ft = estimate_fn(spec)
    rng = np.random.default_rng(_seed_of(family) + 2)
    Xh = lo + (hi - lo) * rng.random((N_HELDOUT, d))
    from scipy.stats import spearmanr
    spec["achieved_pearson"] = float(np.corrcoef(f(Xh), ft(Xh))[0, 1])
    spec["achieved_spearman"] = float(spearmanr(f(Xh), ft(Xh)).statistic)
    scr = lo + (hi - lo) * np.random.default_rng(_seed_of(family) + 3).random((N_FIT, d))
    x0 = scr[int(np.argmin(ft(scr)))]
    from scipy.optimize import minimize
    r = minimize(lambda x: float(ft(x.reshape(1, -1))[0]), x0, method="L-BFGS-B",
                 bounds=[(lo, hi)] * d)
    spec["estimate_argmin"] = [float(v) for v in r.x]
    spec["estimate_min"] = float(r.fun)
    spec["true_f_at_estimate_argmin"] = float(f(r.x.reshape(1, -1))[0])
    return spec


def estimate_fn(spec: dict):
    family, params = spec["family"], spec["params"]

    def ft(X):
        return _eval(family, params, X)

    return ft


def _tab(rows: list[list[float]], headers: list[str]) -> str:
    lines = ["  ".join(headers)]
    for r in rows:
        lines.append("  ".join(f"{v:.4g}" for v in r))
    return "\n".join(lines)


def formula_text(spec: dict) -> str:
    """Analytic model text for the elicitation card, without the function's name."""
    p = spec["params"]
    fam = spec["family"]
    d = spec["dim"]
    if fam == "hartmann6":
        head = ("F(x) = - sum_{k=1..4} alpha_k * exp( - sum_{j=0..5} A[k][j] * "
                "(x_j - P[k][j])^2 )")
        alpha = "alpha = [" + ", ".join(f"{v:.4g}" for v in p["alpha"]) + "]"
        A = "A (4 rows x 6 cols):\n" + _tab(p["A"], [f"j={j}" for j in range(6)])
        P = "P (4 rows x 6 cols):\n" + _tab(p["P"], [f"j={j}" for j in range(6)])
        return f"{head}\n{alpha}\n{A}\n{P}"
    if fam == "styblinski_tang7":
        head = ("F(x) = 0.5 * sum_{i=0..6} [ y_i^4 - a_i*y_i^2 + b_i*y_i ],   "
                "where y_i = lam_i * (x_i - s_i)")
        rows = [[p["a"][i], p["b"][i], p["lam"][i], p["s"][i]] for i in range(d)]
        return head + "\nper-parameter constants:\n" + _tab(rows, ["a_i", "b_i", "lam_i", "s_i"])
    if fam == "levy10":
        head = ("F(x) = sin^2(pi*w_0) + sum_{i=0..8} (w_i - 1)^2 * "
                "[1 + amp_i * sin^2(kap_i*pi*w_i + 1)] + (w_9 - 1)^2 * "
                "[1 + sin^2(2*pi*w_9)],   where w_i = 1 + (x_i - o_i)/4")
        rows = [[p["o"][i], (p["amp"][i] if i < d - 1 else float("nan")),
                 (p["kap"][i] if i < d - 1 else float("nan"))] for i in range(d)]
        return head + "\nper-parameter constants (amp/kap unused for i=9):\n" + _tab(
            rows, ["o_i", "amp_i", "kap_i"])
    if fam == "ackley15":
        head = (f"F(x) = -20 * exp( -{p['b']:.4g} * sqrt( (1/15) * sum_i (x_i - c_i)^2 ) ) "
                "- exp( (1/15) * sum_i cos( kap_i * 2*pi * (x_i - c_i) ) ) + 20 + e")
        rows = [[p["c"][i], p["kap"][i]] for i in range(d)]
        return head + "\nper-parameter constants:\n" + _tab(rows, ["c_i", "kap_i"])
    raise KeyError(fam)


def spec_path(family: str, rho_key: str) -> Path:
    return SPEC_DIR / family / f"formula_spec_{rho_key}.json"
