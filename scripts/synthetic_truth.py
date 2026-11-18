"""Hidden objectives of the synthetic panel (Hartmann-6, Styblinski-Tang-7, Levy-10, Ackley-15).

Two worlds, selected with the environment variable ``SYNTH_TRUTH``:

``shift``
    The benchmark function evaluated at ``x - delta`` with a fixed per-dimension offset
    ``delta`` (``shift_delta``). ``f_opt`` is unchanged and ``x_opt`` becomes
    ``x_opt + delta``. Every coordinate of the shifted optimum lies inside the box with a
    10 % margin and at least 8 % of the box width from the box centre. The formula cards
    (``experiments/landscape_estimates/<family>/card_formula_*_domain.txt``) describe this
    shifted function, so their location parameters carry the same offset.

``well`` (default)
    The rho = 0.9 card model ``M`` (the constants printed on the rho90 card, evaluated with
    ``formula_estimate_lib._eval``) blended with a displaced copy of itself inside a
    Gaussian window, plus a quadratic term:

        g(x) = exp(-||z(x) - p||^2 / (2 ell^2)),  z(x) = (x - lo) / (hi - lo)
        T(x) = (1 - g) M(x) + g (M(x - d) - eps) + gamma * sum_j (z_j - c_j)^2

    so the true optimum lies near, but not at, the optimum of the card model, with
    Pearson(M, T) = 0.90 on a fixed uniform sample. The parameters (``p``, ``ell``, ``d``,
    ``eps``, ``gamma``, ``c``) and the optimum ``(f_opt, x_opt)`` are stored in
    ``experiments/landscape_estimates/well_truth.json``.

``apply(table)`` maps a runner's family table ``{family: (d, lo, hi, f, f_opt[, x_opt])}``
to the selected world; every synthetic runner builds its table through it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CARD_DIR = ROOT / "experiments" / "landscape_estimates"
WELL_TRUTH_PATH = CARD_DIR / "well_truth.json"
FAMILIES = ("hartmann6", "styblinski_tang7", "levy10", "ackley15")
WORLDS = ("well", "shift")

# Box bounds per family (dim, lo, hi).
BOUNDS = {
    "hartmann6": (6, 0.0, 1.0),
    "styblinski_tang7": (7, -5.0, 5.0),
    "levy10": (10, -10.0, 10.0),
    "ackley15": (15, -32.768, 32.768),
}

# Fixed per-dimension offsets of the shifted world.
SHIFT_DELTA = {
    "hartmann6": [0.13442504328933372, 0.02653074065223343, 0.13729490719759438, -0.12427565636103412,
                  0.02524937539897379, 0.013925925255556776],
    "styblinski_tang7": [0.2187494196928248, 1.0704505408121445, 1.1476965127701169, -0.46137739660260957,
                         1.2181446056184018, -0.8253882199634381, 1.3057760487435313],
    "levy10": [3.794477538163262, 4.391808840773745, 0.9226008971730781, 3.278218846966814, 3.6084914673132946,
               2.357812627807718, 3.973845967790247, -3.610365922514394, -4.38875116525753, 5.77120555620054],
    "ackley15": [18.172208865568386, -12.525930205462785, -16.55100413703897, 18.665864126291716,
                 -12.975255253846157, -14.937309081997512, 18.388854626347054, 14.921933453608986,
                 -17.463700366424572, -17.860456155344, 7.138587263297044, -6.278621531026904,
                 11.603229471788204, 15.48045220044741, 14.367807615098812],
}


def world() -> str:
    """The selected world (env ``SYNTH_TRUTH``, default ``well``)."""
    w = os.environ.get("SYNTH_TRUTH", "well").strip() or "well"
    if w not in WORLDS:
        raise ValueError(f"SYNTH_TRUTH must be one of {WORLDS}, got {w!r}")
    return w


def shift_delta(family: str) -> np.ndarray:
    """Per-dimension offset of the shifted world."""
    return np.asarray(SHIFT_DELTA[family], dtype=float)


def shift_fn(f, family: str):
    """``f_delta(x) = f(x - delta)`` for a single point (d,) or a batch (n, d)."""
    dl = shift_delta(family)

    def g(x, *a, **k):
        return f(np.asarray(x, float) - dl, *a, **k)

    return g


# ------------------------------------------------------------------------------------------------ card model
def card_text(family: str, card: str = "rho90") -> str:
    return (CARD_DIR / family / f"card_formula_{card}_domain.txt").read_text(encoding="utf-8")


def _num_rows(text: str):
    return [ln.split() for ln in text.splitlines() if re.match(r"^-?\d", ln.strip())]


def card_params(family: str, text: str | None = None) -> dict:
    """Model constants exactly as printed on a formula card (default: the rho90 card)."""
    t = text if text is not None else card_text(family)
    rows = _num_rows(t)
    if family == "hartmann6":
        alpha = [float(v) for v in re.search(r"alpha = \[([^\]]+)\]", t).group(1).split(",")]
        six = [list(map(float, r)) for r in rows if len(r) == 6]
        assert len(six) == 8, "hartmann6 card: expected 8 rows of 6"
        return {"alpha": alpha, "A": six[:4], "P": six[4:8]}
    if family == "styblinski_tang7":
        vals = [list(map(float, r)) for r in rows]
        assert len(vals) == 7 and all(len(r) == 4 for r in vals)
        a, b, lam, s = (list(c) for c in zip(*vals))
        return {"a": a, "b": b, "lam": lam, "s": s}
    if family == "levy10":
        assert len(rows) == 10
        return {"o": [float(r[0]) for r in rows], "amp": [float(r[1]) for r in rows[:9]],
                "kap": [float(r[2]) for r in rows[:9]]}
    if family == "ackley15":
        assert len(rows) == 15
        b = float(re.search(r"-20 \* exp\( -([\d.]+)", t).group(1))
        return {"c": [float(r[0]) for r in rows], "b": b, "kap": [float(r[1]) for r in rows]}
    raise KeyError(family)


def card_model(family: str, card: str = "rho90"):
    """Vectorized model of a formula card, X (n, d) -> (n,)."""
    import formula_estimate_lib as fel  # lazy: formula_estimate_lib imports estimate_lib, which imports this module

    p = card_params(family, card_text(family, card))
    return lambda X: fel._eval(family, p, np.atleast_2d(np.asarray(X, float)))


# ------------------------------------------------------------------------------------------------ well truth
def load_well_truth() -> dict:
    data = json.loads(WELL_TRUTH_PATH.read_text(encoding="utf-8"))
    for family, digest in data["card_sha256"].items():
        actual = hashlib.sha256(card_text(family).replace("\r\n", "\n").encode("utf-8")).hexdigest()
        if actual != digest:
            raise ValueError(f"rho90 card of {family} does not match well_truth.json (sha256 {actual})")
    return data


def unit(family: str, X) -> np.ndarray:
    _, lo, hi = BOUNDS[family]
    return (np.atleast_2d(np.asarray(X, float)) - lo) / (hi - lo)


def window_eval(family: str, X, t: dict, M=None) -> np.ndarray:
    """T(x) of the well world for parameters ``t`` (see the module docstring)."""
    X = np.atleast_2d(np.asarray(X, float))
    M = M if M is not None else card_model(family)
    z = unit(family, X)
    g = np.exp(-np.sum((z - np.asarray(t["far_point"], float)[None, :]) ** 2, axis=1) / (2.0 * t["ell"] ** 2))
    T = (1.0 - g) * M(X) + g * (M(X - np.asarray(t["delta"], float)[None, :]) - t["eps"])
    if t.get("coef"):
        T = T + float(t["coef"]) * np.sum((z - np.asarray(t["vector"], float)[None, :]) ** 2, axis=1)
    return T


def well_truth(family: str):
    """Vectorized well-world objective, X (n, d) or (d,) -> (n,)."""
    t = load_well_truth()["well"][family]
    M = card_model(family)
    return lambda X: window_eval(family, X, t, M=M)


def well_optimum(family: str) -> tuple[float, np.ndarray]:
    t = load_well_truth()["well"][family]
    return float(t["f_opt"]), np.asarray(t["x_opt"], float)


# ------------------------------------------------------------------------------------------------ runner tables
def apply(table: dict) -> dict:
    """Map a runner's family table ``{family: (d, lo, hi, f, f_opt[, x_opt])}`` to the selected world."""
    w = world()
    out = {}
    for fam, ent in table.items():
        ent = list(ent)
        if fam not in SHIFT_DELTA:
            out[fam] = tuple(ent)
            continue
        ent[3] = shift_fn(ent[3], fam)
        if len(ent) > 5 and ent[5] is not None:
            ent[5] = np.asarray(ent[5], float) + shift_delta(fam)
        if w == "well":
            params = json.loads(WELL_TRUTH_PATH.read_text(encoding="utf-8"))["well"][fam]
            cache: dict = {}

            def obj(x, *a, _fam=fam, _c=cache, **k):
                if "T" not in _c:
                    _c["T"] = well_truth(_fam)
                x = np.asarray(x, float)
                return float(_c["T"](x[None, :])[0]) if x.ndim == 1 else np.asarray(_c["T"](x), float)

            ent[3], ent[4] = obj, float(params["f_opt"])
            if len(ent) > 5:
                ent[5] = np.asarray(params["x_opt"], float)
        out[fam] = tuple(ent)
    return out


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    print("world:", world())
    for fam in FAMILIES:
        f_opt, x_opt = well_optimum(fam)
        print(fam, "delta", np.round(shift_delta(fam), 3).tolist(), "well f_opt", round(f_opt, 4))
