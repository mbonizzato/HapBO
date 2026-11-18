"""Shared helpers of the LGBO workers (``lgbo_worker`` and ``official_lgbo_worker``).

Self-contained Matern-5/2 GP, Sobol sampling, the region/point grid with lift weights,
the per-iteration guidance prompt and its parser, and a vectorized ``erf``. The module
runs in the isolated LLAMBO environment and needs no repository import.
"""

from __future__ import annotations

import json

import numpy as np


# --------------------------------------------------------------------------- #
# Self-contained Matern-5/2 GP (mirrors src/bayesian_optimization/surrogate.py so the
# worker needs no repo import; standardizes y with normalize_y semantics).
# --------------------------------------------------------------------------- #
class _GP:
    def __init__(self, length_scale=0.2, noise_std=1e-3, signal_variance=1.0):
        self.length_scale = float(length_scale)
        self.noise_variance = float(noise_std) ** 2
        self.signal_variance = float(signal_variance)

    def _kernel(self, X1, X2):
        X1 = np.atleast_2d(np.asarray(X1, dtype=float))
        X2 = np.atleast_2d(np.asarray(X2, dtype=float))
        diff = (X1[:, None, :] - X2[None, :, :]) / self.length_scale
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        s5 = np.sqrt(5.0) * dist
        return self.signal_variance * (1.0 + s5 + (5.0 / 3.0) * dist ** 2) * np.exp(-s5)

    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).reshape(-1)
        # Sanitize: drop any training row whose X has a non-finite entry or whose y is
        # non-finite, keeping X and y row-aligned. A single NaN/inf point (e.g. a NaN
        # continuous acquisition result, or a NaN-shifted proposal) would otherwise poison
        # the kernel matrix K -> Cholesky fails permanently (jitter cannot rescue NaN).
        # This is the primary guard that guarantees `fit` never raises on non-finite input;
        # the escalating jitter below is a secondary guard for genuine ill-conditioning.
        finite_rows = np.isfinite(X).all(axis=1) & np.isfinite(y)
        if not np.all(finite_rows):
            X = X[finite_rows]
            y = y[finite_rows]
        if X.shape[0] < 1:
            # Degrade gracefully to a constant-mean (zero-residual) GP rather than raise, so
            # a fully-corrupted history still yields a usable (flat) posterior mean/cov.
            self.X_train_ = np.zeros((0, X.shape[1] if X.ndim == 2 else 1), dtype=float)
            self.y_mean_ = 0.0
            self.y_scale_ = 1.0
            self.L_ = np.zeros((0, 0), dtype=float)
            self.alpha_ = np.zeros((0,), dtype=float)
            self._degenerate_ = True
            return self
        self._degenerate_ = False
        self.X_train_ = X.copy()
        self.y_mean_ = float(np.mean(y))
        centered = y - self.y_mean_
        scale = float(np.std(centered))
        self.y_scale_ = scale if scale > 1e-12 else 1.0
        centered = centered / self.y_scale_
        K = self._kernel(X, X)
        identity = np.eye(X.shape[0])
        # Escalating jitter. LGBO concentrates proposals near the guided region, so the
        # training set accumulates near-duplicate rows and K becomes near-singular; the
        # continuous panel needs a much larger jitter ceiling than the finite
        # discrete case (which never approaches this). Jitter only grows when K is
        # genuinely degenerate, so well-conditioned (finite) fits are unaffected.
        jitter = 1e-9
        for _ in range(14):
            try:
                self.L_ = np.linalg.cholesky(K + (self.noise_variance + jitter) * identity)
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        else:
            raise np.linalg.LinAlgError("Unable to factorize the GP covariance matrix")
        self.alpha_ = np.linalg.solve(self.L_.T, np.linalg.solve(self.L_, centered))
        return self

    def predict(self, X, return_std=True):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if getattr(self, "_degenerate_", False) or self.X_train_.shape[0] == 0:
            # No usable training data: flat prior mean, full prior std. Always finite.
            mean = np.full(X.shape[0], self.y_mean_, dtype=float)
            if not return_std:
                return mean
            std = np.full(X.shape[0], np.sqrt(self.signal_variance) * self.y_scale_)
            return mean, std
        Kx = self._kernel(X, self.X_train_)
        mean_z = Kx @ self.alpha_
        mean = mean_z * self.y_scale_ + self.y_mean_
        if not return_std:
            return mean
        v = np.linalg.solve(self.L_, Kx.T)
        var = np.maximum(self.signal_variance - np.sum(v ** 2, axis=0), 0.0)
        std = np.sqrt(var) * self.y_scale_
        return mean, std

    def posterior_cov(self, X):
        """Posterior covariance Sigma over rows of X, in *standardized* z-space.

        Sigma_ij = k(x_i, x_j) - k(x_i, X_tr) K^{-1} k(X_tr, x_j).  This is the object
        the LGBO lambda uses (a^T Sigma_GG a); returned in z-units so it is consistent
        with the standardized residual space the mean shift lives in.
        """

        X = np.atleast_2d(np.asarray(X, dtype=float))
        Kxx = self._kernel(X, X)
        if getattr(self, "_degenerate_", False) or self.X_train_.shape[0] == 0:
            # No conditioning data: posterior covariance == prior covariance (finite).
            return Kxx
        Kx = self._kernel(X, self.X_train_)
        v = np.linalg.solve(self.L_, Kx.T)          # (n_tr, m)
        return Kxx - v.T @ v

    def cross_kernel(self, X, G):
        """k(X, G) in z-space (same units as posterior_cov)."""

        return self._kernel(X, G)


# --------------------------------------------------------------------------- #
# Sobol grid + region-lift weights.
# --------------------------------------------------------------------------- #
def _sobol_unit(n, d, seed):
    import warnings

    from scipy.stats import qmc

    # A non-power-of-2 n only weakens Sobol balance slightly (the grid stays a valid
    # low-discrepancy sample); silence the informational warning for arbitrary n_grid.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return qmc.Sobol(d=d, scramble=True, seed=int(seed)).random(int(n))


def _grid_and_weights(kind, lb, ub, dim, n_grid, seed, decay):
    """Return (G, a): a Sobol grid inside the [lb, ub] sub-box and decay-weighted
    weights over the grid.

    * REGION: G fills the box uniformly (Sobol); a is uniform then L1-normalized.
    * POINT : the box is a small neighbourhood centred on the point; a is a
      distance-decayed peak (weight_g = exp(-decay * ||g - centre||^2 / r^2))
      then L1-normalized, so the lift concentrates the mean bump at the point.

    G is returned in the UNIT encoded cube (rows already clamped to [0, 1]).
    """

    lb = np.asarray(lb, dtype=float).reshape(-1)
    ub = np.asarray(ub, dtype=float).reshape(-1)
    U = _sobol_unit(n_grid, dim, seed)
    G = np.clip(lb[None, :] + (ub - lb)[None, :] * U, 0.0, 1.0)
    if kind == "point":
        centre = 0.5 * (lb + ub)
        span = np.maximum(ub - lb, 1e-9)
        d2 = np.sum(((G - centre[None, :]) / span[None, :]) ** 2, axis=1)
        w = np.exp(-float(decay) * d2)
    else:
        w = np.ones(G.shape[0], dtype=float)
    total = float(w.sum())
    a = (w / total) if total > 1e-12 else np.full(G.shape[0], 1.0 / G.shape[0])
    return G, a


# --------------------------------------------------------------------------- #
# LLM preference elicitation: parse a point/region + confidence.
# --------------------------------------------------------------------------- #
def _guidance_prompt(dim, config, history_lines):
    task = str(config.get("model", "a black-box objective"))
    metric = str(config.get("metric", "objective value"))
    lower_is_better = bool(config.get("lower_is_better", True))
    direction = "MINIMIZE" if lower_is_better else "MAXIMIZE"
    names = ", ".join("x%d" % j for j in range(dim))
    context = str(config.get("description", "")).strip()
    obs = "\n".join(history_lines) if history_lines else "(no observations yet)"
    return (
        "You are guiding a Bayesian optimization search over a %d-dimensional box; "
        "every coordinate (%s) is normalized to [0, 1].\n" % (dim, names)
        + "The objective is %s (%s); we want to %s it.\n" % (task, metric, direction)
        + (context + "\n" if context else "")
        + "Observed (x -> %s) history so far:\n%s\n\n" % (metric, obs)
        + "Emit ONE guidance suggestion pointing at where the optimum most likely is, "
        "as a JSON object on a single line, in EXACTLY one of these two forms:\n"
        + '  {"type": "point", "x": [<%d floats in [0,1]>], "confidence": <0..1>}\n' % dim
        + '  {"type": "region", "lb": [<%d floats>], "ub": [<%d floats>], '
        '"confidence": <0..1>}\n' % (dim, dim)
        + "confidence is your belief that the optimum lies at/inside the suggestion "
        "(0 = no idea, 1 = certain). Output ONLY the JSON object, nothing else."
    )


def _parse_guidance(raw, dim):
    """Parse an LLM reply into (kind, lb, ub, confidence) in the unit cube.

    Returns ``None`` if nothing parseable is found (caller degrades to lambda=0).
    A point is lifted to a small neighbourhood box of half-width ``_POINT_HALFWIDTH``.
    """

    try:
        start = raw.index("{")
        obj, _ = json.JSONDecoder().raw_decode(raw[start:])
    except (ValueError, KeyError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    kind = str(obj.get("type", "")).lower()
    try:
        c = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        c = 0.0
    c = min(max(c, 0.0), 1.0)
    try:
        if kind == "point":
            x = np.asarray(obj["x"], dtype=float).reshape(-1)
            if x.size != dim or not np.all(np.isfinite(x)):
                return None
            x = np.clip(x, 0.0, 1.0)
            lb = np.clip(x - _POINT_HALFWIDTH, 0.0, 1.0)
            ub = np.clip(x + _POINT_HALFWIDTH, 0.0, 1.0)
            return "point", lb, ub, c
        if kind == "region":
            lb = np.clip(np.asarray(obj["lb"], dtype=float).reshape(-1), 0.0, 1.0)
            ub = np.clip(np.asarray(obj["ub"], dtype=float).reshape(-1), 0.0, 1.0)
            if lb.size != dim or ub.size != dim:
                return None
            if not (np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))):
                return None
            ub = np.maximum(ub, lb + 1e-6)
            return "region", lb, ub, c
    except (KeyError, TypeError, ValueError):
        return None
    return None


_POINT_HALFWIDTH = 0.05  # neighbourhood half-width (unit cube) around a point suggestion


def _erf_vec(values):
    from math import erf

    values = np.asarray(values, dtype=float)
    return np.array([erf(float(v)) for v in values.reshape(-1)]).reshape(values.shape)
