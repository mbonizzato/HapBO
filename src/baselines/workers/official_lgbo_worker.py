# Portions of this file are ported from LGBO (https://github.com/Xinzhe309/LGBO,
# commit 01e56e1165a030ac74961f14d407bcc188f66ce6), distributed under the MIT License:
#
# Copyright (c) 2026 LGBO Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Official-release LGBO worker for a finite candidate set.

This sensitivity arm runs the *public author release* operator (``Xinzhe309/LGBO`` @
commit 01e56e1165a030ac74961f14d407bcc188f66ce6, pinned in ``external/PINNED_COMMITS.txt``) instead of the paper
equation used by ``lgbo_worker``.

Everything except the region-lift operator is reused verbatim from
``lgbo_worker`` (same Matérn-5/2 GP with per-iteration marginal-likelihood
refit, same instrumented Qwen chat, same 2 LLM-chosen initial points, same log-EI
finite-candidate selection), so the two arms differ ONLY in how the LLM confidence
becomes the lift.  The release operator, ported line-faithfully from ``decide.py`` and
``prior.py``:

    delta = guidance_scale * Phi^{-1}(c)          # guidance_scale = 3.0
    a     = sum-normalized (sigmoid) box mask on a Sobol grid of grid_size=512 over [0,1]^d
    lam   = delta / sqrt(a^T Sigma_GG a)          # posterior Sigma over the grid
    shift = lam * (Sigma_XG a)                    # covariance unchanged

Consequences of ``delta = 3*Phi^{-1}(c)`` (vs the paper's direct ``delta = c``):
``c < 0.5`` gives a NEGATIVE lift (suppresses the preferred region), ``c = 0.5`` gives
none, and ``c -> 1`` is unbounded.  A point preference is expanded to a soft region via
``choose_soft_radius_edge``; region density selects the mode/smoothing.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from statistics import NormalDist

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lgbo_worker as base  # noqa: E402  reuse GP, chat, init helpers
import lgbo_common as common  # noqa: E402  reuse guidance prompt + parser + erf
from finite import normalize_rows  # noqa: E402
from harness import finish, log, make_evaluator, read_spec  # noqa: E402


# Accumulator for the instrumented chat; written out on finish() if the parent set
# OFFICIAL_LGBO_TELEMETRY.  Reuses base._make_instrumented_chat, which appends here.
_TELEMETRY: list[dict] = []

# Official release defaults (decide.py / boo.py), overridable via config.
_GUIDANCE_SCALE = 3.0
_GRID_SIZE = 512
_E_SOFT_LOW = 2.0
_E_SOFT_HIGH = 6.0
_SMOOTH_SOFT = 0.08   # region-soft mode (decide.py)
_SMOOTH_REGION = 0.06  # hard-region mode (boo.py maps smooth=None -> 0.06)
_LOW_MARGIN = 0.05
_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Ported operator helpers (line-faithful to decide.py / prior.py @ pinned commit).
# --------------------------------------------------------------------------- #
_NORMAL = NormalDist()


def _norm_ppf(p):
    """Phi^{-1} with decide.py's clamp to [1e-12, 1 - 1e-12]."""

    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    return _NORMAL.inv_cdf(p)


def _confidence_to_delta(confidence, scale=_GUIDANCE_SCALE):
    """decide.py: confidence_to_delta(c, scale, two_sided=False) = scale * Phi^{-1}(c)."""

    return float(scale) * _norm_ppf(confidence)


def _sobol_grid(dim, n_points):
    """prior.py::_make_sobol_grid_norm: unscrambled Sobol over [0,1]^d, clamped."""

    from scipy.stats import qmc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        grid = qmc.Sobol(d=int(dim), scramble=False).random(int(n_points))
    return np.clip(np.asarray(grid, dtype=float), _EPS, 1.0 - _EPS)


def _build_a(grid, lb, ub, smooth):
    """prior.py::_build_a (smooth branch): sum-normalized sigmoid box mask."""

    s = float(smooth) + 1e-12
    z1 = (grid - lb[None, :]) / s
    z2 = (ub[None, :] - grid) / s
    sig1 = 1.0 / (1.0 + np.exp(-z1))
    sig2 = 1.0 / (1.0 + np.exp(-z2))
    a_raw = np.prod(sig1, axis=1) * np.prod(sig2, axis=1)
    total = float(a_raw.sum())
    if total > 1e-12:
        return a_raw / total
    return np.full(grid.shape[0], 1.0 / grid.shape[0], dtype=float)


def _box_width_with_clamp(center_i, r, eps=_EPS):
    lb = max(center_i - r, eps)
    ub = min(center_i + r, 1.0 - eps)
    return max(ub - lb, 0.0)


def _effective_volume(center, r, eps=_EPS):
    width = 1.0
    for value in np.asarray(center, dtype=float).reshape(-1):
        width *= _box_width_with_clamp(float(value), r, eps)
    return max(width, 0.0)


def _choose_soft_radius_edge(grid_size, center, e_soft_low=_E_SOFT_LOW,
                             low_margin=_LOW_MARGIN, eps=_EPS):
    """decide.py::choose_soft_radius_edge: smallest clipped box holding ~E_soft_low hits."""

    target_hits = float(e_soft_low) + float(low_margin)
    target_volume = max(target_hits / float(grid_size), 1e-12)

    lo, hi = 1e-8, 0.5 - eps
    if _effective_volume(center, hi, eps) < target_volume:
        return float(hi)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if _effective_volume(center, mid, eps) >= target_volume:
            hi = mid
        else:
            lo = mid
    r = 0.5 * (lo + hi)
    effective_hits = grid_size * _effective_volume(center, r, eps)
    if effective_hits < e_soft_low:
        for _ in range(10):
            r = min(r * 1.2, 0.5 - eps)
            effective_hits = grid_size * _effective_volume(center, r, eps)
            if effective_hits >= e_soft_low:
                break
    return float(r)


def _resolve_region(kind, lb, ub, dim, grid_size, e_low, e_high, smooth_soft,
                    smooth_region, low_margin):
    """Map a parsed (kind, lb, ub) preference to (region_lb, region_ub, smooth).

    Mirrors decide.py: points are expanded to a soft box (choose_soft_radius_edge);
    region density selects point-expanded / region-soft / region mode.
    """

    lb = np.asarray(lb, dtype=float).reshape(-1)
    ub = np.asarray(ub, dtype=float).reshape(-1)
    center = 0.5 * (lb + ub)

    def _soft_box():
        radius = _choose_soft_radius_edge(grid_size, center, e_low, low_margin)
        rlb = np.clip(center - radius, _EPS, 1.0 - _EPS)
        rub = np.clip(center + radius, _EPS, 1.0 - _EPS)
        return rlb, rub, smooth_soft

    if kind == "point":
        return _soft_box()

    rlb = np.clip(lb, _EPS, 1.0 - _EPS)
    rub = np.clip(ub, _EPS, 1.0 - _EPS)
    vol = float(np.prod(np.clip(rub - rlb, 0.0, 1.0)))
    expected_hits = grid_size * vol
    if expected_hits < e_low:
        return _soft_box()
    if expected_hits < e_high:
        return rlb, rub, smooth_soft
    return rlb, rub, smooth_region


def _official_regional_shift_z(gp, X, kind, lb, ub, confidence, config, eps=1e-12):
    """Return (shift_z, aSa, lambda, delta) for the release region-lift operator."""

    X = np.atleast_2d(np.asarray(X, dtype=float))
    dim = X.shape[1]
    grid_size = int(config.get("grid_size", _GRID_SIZE))
    scale = float(config.get("guidance_scale", _GUIDANCE_SCALE))
    e_low = float(config.get("E_soft_low", _E_SOFT_LOW))
    e_high = float(config.get("E_soft_high", _E_SOFT_HIGH))
    smooth_soft = float(config.get("smooth_soft", _SMOOTH_SOFT))
    smooth_region = float(config.get("smooth_region", _SMOOTH_REGION))
    low_margin = float(config.get("low_margin", _LOW_MARGIN))

    c = float(confidence)
    # decide.py raises for c not in (0,1); our parser can emit 0/1.
    if c <= 0.0:
        return np.zeros(X.shape[0]), 0.0, 0.0, 0.0
    if c >= 1.0:
        c = 1.0 - 1e-6
    delta = _confidence_to_delta(c, scale)

    rlb, rub, smooth = _resolve_region(kind, lb, ub, dim, grid_size, e_low, e_high,
                                       smooth_soft, smooth_region, low_margin)
    grid = _sobol_grid(dim, grid_size)
    a = _build_a(grid, rlb, rub, smooth)

    sigma_gg = gp.posterior_cov(grid)
    if not np.all(np.isfinite(sigma_gg)):
        return np.zeros(X.shape[0]), 0.0, 0.0, delta
    aSa = float(a @ sigma_gg @ a)
    if not np.isfinite(aSa) or aSa <= eps:
        return np.zeros(X.shape[0]), aSa, 0.0, delta
    lam = delta / np.sqrt(aSa)
    if not np.isfinite(lam):
        return np.zeros(X.shape[0]), aSa, 0.0, delta
    shift = lam * (gp.posterior_cross_cov(X, grid) @ a)
    if not np.all(np.isfinite(shift)):
        return np.zeros(X.shape[0]), aSa, 0.0, delta
    return shift, aSa, lam, delta


def _official_lgbo_step(gp, encoded, evaluated_mask, guidance, config, best_observed):
    """Select the finite candidate maximizing log-EI on the official-tilted mean."""

    shift_z = np.zeros(encoded.shape[0])
    aSa = lam = delta = 0.0
    if guidance is not None:
        kind, lb, ub, confidence = guidance
        shift_z, aSa, lam, delta = _official_regional_shift_z(
            gp, encoded, kind, lb, ub, confidence, config)

    mean, std = gp.predict(encoded)
    mean = mean + shift_z * gp.y_scale_
    std = np.maximum(std, 0.0)  # tilt leaves covariance unchanged
    improvement = mean - float(best_observed)
    ei = np.maximum(improvement, 0.0)
    positive = std > 1e-12
    if np.any(positive):
        z = improvement[positive] / std[positive]
        cdf = 0.5 * (1.0 + common._erf_vec(z / np.sqrt(2.0)))
        pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)
        ei[positive] = improvement[positive] * cdf + std[positive] * pdf
    log_ei = np.log(np.maximum(ei, np.finfo(float).tiny))
    log_ei = np.nan_to_num(log_ei, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    log_ei[evaluated_mask] = -np.inf
    return int(np.argmax(log_ei)), aSa, lam, delta


def _write_telemetry(config, budget, dim, seed):
    """Serialize accumulated LLM telemetry to OFFICIAL_LGBO_TELEMETRY (if set)."""

    path = os.environ.get("OFFICIAL_LGBO_TELEMETRY")
    if not path:
        return
    calls = list(_TELEMETRY)
    guidance = [c for c in calls if c["kind"] == "guidance"]
    n_ok = sum(1 for c in calls if c["ok"])

    def _sum(key, rows):
        return int(sum(c[key] for c in rows))

    def _mean(key, rows):
        return (float(sum(c[key] for c in rows)) / len(rows)) if rows else 0.0

    summary = {
        "model": str(config.get("llm_model") or os.environ.get("LLM_MODEL") or "local-model"),
        "budget": int(budget), "dim": int(dim), "seed": int(seed),
        "n_calls": len(calls), "n_calls_ok": n_ok, "n_calls_failed": len(calls) - n_ok,
        "n_initial_calls": sum(1 for c in calls if c["kind"] == "initial"),
        "n_guidance_calls": len(guidance),
        "prompt_tokens_total": _sum("prompt_tokens", calls),
        "completion_tokens_total": _sum("completion_tokens", calls),
        "total_tokens_total": _sum("total_tokens", calls),
        "llm_latency_seconds_total": float(sum(c["latency_seconds"] for c in calls)),
        "prompt_tokens_per_query_mean": _mean("prompt_tokens", guidance),
        "completion_tokens_per_query_mean": _mean("completion_tokens", guidance),
        "total_tokens_per_query_mean": _mean("total_tokens", guidance),
        "latency_seconds_per_query_mean": _mean("latency_seconds", guidance),
    }
    payload = {"artifact_format": "official-lgbo-llm-telemetry-v1",
               "summary": summary, "per_call": calls}
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    except OSError as exc:
        log("official-LGBO telemetry write failed (%s)" % exc)


def main():
    spec = read_spec()
    encoded = normalize_rows(spec["encoded"])
    budget = int(spec["budget"])
    seed = int(spec.get("seed") or 0)
    maximize = bool(spec.get("maximize", True))
    config = dict(spec.get("config") or {})
    evaluate = make_evaluator()
    n_total, dim = encoded.shape
    n_init = min(max(1, int(config.get("paper_n_initial", 2))), budget, n_total)
    config.setdefault("lower_is_better", not maximize)

    base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
    model = str(config.get("llm_model") or os.environ.get("LLM_MODEL") or "local-model")
    chat = base._make_instrumented_chat(
        base_url=base_url, api_key=os.environ.get("LLM_API_KEY", "none"),
        model=model, timeout=float(config.get("request_timeout", 300.0)),
        seed=seed, sink=_TELEMETRY)
    initial_indices = base._llm_initial_indices(chat, encoded, config, n_init, seed)
    log("official-LGBO endpoint=%s model=%s paper_n_init=%d budget=%d grid=%d scale=%.3g dim=%d"
        % (base_url, model, n_init, budget, int(config.get("grid_size", _GRID_SIZE)),
           float(config.get("guidance_scale", _GUIDANCE_SCALE)), dim))

    evaluated = np.zeros(n_total, dtype=bool)
    X_obs, y_obs, history = [], [], []

    def observe(index):
        raw = float(evaluate(index))
        evaluated[index] = True
        X_obs.append(encoded[index].tolist())
        y_obs.append(raw if maximize else -raw)
        history.append(("[" + ", ".join("%.4f" % v for v in encoded[index]) + "]", raw))

    for index in initial_indices:
        if len(y_obs) < budget:
            observe(index)

    while len(y_obs) < budget and not np.all(evaluated):
        index = None
        try:
            gp = base._GP(length_scale=float(config.get("length_scale", 0.2)),
                          noise_std=float(config.get("noise_std", 1e-3))).fit(
                              np.asarray(X_obs), np.asarray(y_obs))
            lines = ["  %s -> %.6g" % row for row in history[-30:]]
            guidance = None
            try:
                reply = chat([{"role": "user", "content": common._guidance_prompt(
                    dim, config, lines)}], temperature=0.0,
                    max_tokens=int(config.get("max_tokens", 512)),
                    kind="guidance", query_after=len(y_obs))
                guidance = common._parse_guidance(reply, dim)
                if guidance is None:
                    log("official-LGBO guidance parse failed; using unshifted log-EI")
            except Exception as exc:
                log("official-LGBO guidance call failed (%s); using unshifted log-EI" % exc)
            index, aSa, lam, delta = _official_lgbo_step(
                gp, encoded, evaluated, guidance, config, max(y_obs))
            if guidance is not None:
                log("official-LGBO posterior lift aSa=%.6g lambda=%.6g delta=%.6g confidence=%.4g"
                    % (aSa, lam, delta, guidance[3]))
        except Exception as exc:
            log("official-LGBO step failed (%s); using unseen fallback" % exc)
        if index is None or index < 0 or evaluated[index]:
            remaining = np.flatnonzero(~evaluated)
            if remaining.size == 0:
                break
            index = int(remaining[0])
        observe(index)
    _write_telemetry(config, budget, dim, seed)
    finish()


if __name__ == "__main__":
    main()
