"""Bayesian optimization over a continuous box (the synthetic panel).

:class:`~.bo.BayesianOptimization` optimizes over a finite candidate set; this module optimizes
over a continuous box with Expected Improvement: at each BO step it fits the GP, then maximizes
the acquisition over the box by a Sobol screen followed by L-BFGS-B restarts. Regret is measured
against the true optimum of the objective. The alpha-pi rule shapes the acquisition as on the
finite path:

    shaped(x) = max( acq_raw(x) * ((1-alpha) + alpha * pi_bar(x)), 0 )

where ``pi_bar`` is the prior preference min-max-normalized over a frozen reference sample of
the box (globally consistent, not per-batch).

With ``shape_acquisition=False`` the prior only supplies candidates: its peaks are located
and appended to every acquisition batch exactly as in the alpha-pi arm (same ``alpha > 0``
gate, screen, climbs and constant-prior rule), and the acquisition is scored unshaped
(vanilla BO plus the prior-peak candidates). The alpha-pi arm, this vanilla-peaks control and
vanilla BO share the seeded Sobol initial design.

Prior-injection arms (``prior_strategy`` other than ``"alpha-pi"``): the continuous counterparts of
the finite-grid arms of ``experiments.worker`` (pi-bo-ei, prior-to-mean, prior-warm-start,
prior-greedy). Each locates the prior peaks exactly as the alpha-pi arm does (same screen, climbs,
frozen box reference and constant-prior rule, but no ``alpha > 0`` gate) and scores the same
per-step candidate batches (``n_sobol`` Sobol points plus the prior peaks); only the injection
site differs. Their candidate set C0 is the alpha-pi arm's first acquisition batch,
``_sobol(n_sobol, d, seed + n_initial)`` plus the prior peaks, so every point any arm draws or scores
is one the alpha-pi arm also scores; points that lie within ``DISTINCT_TOL`` of a higher-prior point
in every unit-cube coordinate are dropped, so the climbs that converge on one peak count once
(``distinct_by_prior``):

  ``"pi-bo"``            piBO (Hvarfner et al. 2022) as in ``bo.py``: the initial design is
                         ``bo.pi_bo_initial_design`` on C0 (the mode of
                         ``p = max(pi, 0) / max_C0 pi + 1e-12``, the lowest C0 row on ties, then
                         ``n_initial - 1`` points drawn from p without the mode, the same seeded
                         generator), and every later acquisition is weighted by
                         ``p(x) ** (pi_bo_beta / t)``, ``t = 1`` at the first query after the initial
                         design, then clamped at zero (``acquisition.pi_bo_shape`` with the C0
                         maximum as reference). ``pi_bo_beta="auto"`` (the default) is
                         ``acquisition.pi_bo_beta``: ``N / 10`` with ``N = budget - n_initial`` BO
                         iterations (Sec. 4), as on the finite path.
  ``"prior-to-mean"``    GP prior mean ``m_z(x) = prior_mean_beta * (pi(x) - mean_C0 pi) / std_C0 pi``
                         in the GP's standardized target space (``worker._prior_mean_from_pi``),
                         the vanilla Sobol initial design and the unweighted acquisition.
  ``"prior-warm-start"`` initial design = the ``n_initial`` highest-prior points of C0, then the
                         unweighted acquisition.
  ``"prior-greedy"``     no GP and no acquisition: the ``budget`` highest-prior points of C0 in
                         descending prior order.

The default ``prior_strategy="alpha-pi"`` runs the alpha-pi arm, vanilla BO (no ``prior_fn``, or
``alpha = 0``) and the vanilla-peaks control (``shape_acquisition=False``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from . import acquisition as acq
from .acquisition import ExpectedImprovement
from .bo import pi_bo_initial_design
from .surrogate import GaussianProcessSurrogate

PRIOR_STRATEGIES = ("alpha-pi", "pi-bo", "prior-to-mean", "prior-warm-start", "prior-greedy")
# Two points of the injection arms' candidate set closer than this in every unit-cube coordinate are one
# candidate (prior-peak climbs that converge on the same peak end ~1e-6 apart; distinct peaks >= 0.1 apart).
DISTINCT_TOL = 1e-3


@dataclass(frozen=True)
class ContinuousBOResult:
    best_x: np.ndarray          # native coords of the best query
    best_min_value: float       # best f (minimization convention)
    X_native: np.ndarray        # all queried native coords, in order
    simple_regret: np.ndarray   # best-so-far regret vs the true optimum, per query
    prior_peaks: Optional[np.ndarray] = None  # unit-cube points appended to every candidate batch
    pi_bo_beta: Optional[float] = None        # the resolved beta of a "pi-bo" run


def _sobol(n: int, d: int, seed: int) -> np.ndarray:
    from scipy.stats import qmc

    return qmc.Sobol(d=d, scramble=True, seed=seed).random(n)


def _prior_peak_search(prior_fn, d: int, lo: np.ndarray, hi: np.ndarray, to_native, seed: int):
    """Locate the prior's maximizer(s): Sobol screen + L-BFGS ascent (unit cube).

    Returns ``(top8, peaks)``: the 8 best screen points and the climbed peaks. Serves twice:
    (1) the min-max reference must contain the prior max, else a localized bump in high-D is never
    sampled, pmax collapses to the plateau and the clip erases the belief;
    (2) the peaks are appended to the acquisition candidate batch each step, restoring the
    finite-candidate semantics where the prior max is always scored (a product bump's basin is
    ~width^d of the box, so Sobol candidates and their L-BFGS refinements otherwise never see it).
    """
    from scipy.optimize import minimize

    scr = _sobol(8192, d, seed + 777)
    sval = np.array([float(prior_fn(x)) for x in to_native(scr)])
    top8 = scr[np.argsort(-sval)[:8]]
    peaks = []
    for u0 in top8:
        res = minimize(lambda u: -float(prior_fn(to_native(u)[0])),
                       u0, method="L-BFGS-B", bounds=[(0.0, 1.0)] * d)
        peaks.append(np.clip(res.x, 0.0, 1.0))
    # (3) also climb from the document's own stated peaks (prior_fn.peak_candidates, native coords;
    #     landscape_generator.document_peak_points): on a multimodal prior (e.g. many periodic tokens) the
    #     Sobol-screen climbs can stop on local maxima and miss the location the document names as best.
    extra = getattr(prior_fn, "peak_candidates", None)
    if extra is not None and len(extra):
        for x0 in np.atleast_2d(np.asarray(extra, dtype=float)):
            u0 = np.clip((x0 - lo) / (hi - lo), 0.0, 1.0)
            res = minimize(lambda u: -float(prior_fn(to_native(u)[0])),
                           u0, method="L-BFGS-B", bounds=[(0.0, 1.0)] * d)
            peaks.append(np.clip(res.x, 0.0, 1.0))
    return top8, np.array(peaks)


def distinct_by_prior(U: np.ndarray, values: np.ndarray, tol: float = DISTINCT_TOL,
                      limit: Optional[int] = None) -> np.ndarray:
    """Row indices of ``U`` in descending ``values`` order (stable: ties keep row order), skipping a row
    that lies within ``tol`` of an already kept row in every coordinate; at most ``limit`` indices."""
    U = np.atleast_2d(np.asarray(U, dtype=float))
    order = np.argsort(-np.asarray(values, dtype=float).reshape(-1), kind="stable")
    kept: list[int] = []
    kept_rows = np.empty((0, U.shape[1]))
    for i in order:
        if len(kept_rows) and np.any(np.max(np.abs(kept_rows - U[i]), axis=1) <= tol):
            continue
        kept.append(int(i))
        kept_rows = np.vstack([kept_rows, U[i]])
        if limit is not None and len(kept) >= limit:
            break
    return np.asarray(kept, dtype=int)


def run_continuous_bo(
    objective: Callable[[np.ndarray], float],   # native-coord callable, MINIMIZED
    bounds: np.ndarray,                          # (d, 2) lower/upper
    optimum_value: float,                        # true min f* (for regret)
    *,
    budget: int,
    n_initial: int = 5,
    alpha: float = 0.0,
    prior_fn: Optional[Callable[[np.ndarray], float]] = None,  # native pref, higher=better
    xi: float = 0.0,
    length_scale: float = 0.2,
    noise_std: float = 1e-3,
    n_sobol: int = 2048,
    n_restarts: int = 8,
    seed: int = 0,
    shape_acquisition: bool = True,              # False: unshaped acquisition, prior peaks kept
    prior_strategy: str = "alpha-pi",            # or an injection arm (module docstring)
    pi_bo_beta: float | str = "auto",            # "pi-bo": beta of p ** (beta / t); auto = N / 10
    prior_mean_beta: Optional[float] = None,     # "prior-to-mean": scale of the standardized prior
) -> ContinuousBOResult:
    from scipy.optimize import minimize

    if prior_strategy not in PRIOR_STRATEGIES:
        raise ValueError(f"prior_strategy must be one of {PRIOR_STRATEGIES}")
    n_init = max(1, int(n_initial))
    injection = prior_strategy != "alpha-pi"
    if injection:
        if prior_fn is None:
            raise ValueError(f"prior_strategy {prior_strategy!r} needs a prior_fn")
        if not shape_acquisition:
            raise ValueError("shape_acquisition applies to the alpha-pi arm only")
        if prior_strategy == "prior-to-mean" and prior_mean_beta is None:
            raise ValueError("prior-to-mean needs prior_mean_beta (the compiled alpha)")
        if pi_bo_beta == "auto":
            pi_bo_beta = acq.pi_bo_beta(budget, n_init)
        elif isinstance(pi_bo_beta, str) or not float(pi_bo_beta) >= 0:
            raise ValueError("pi_bo_beta must be 'auto' or a non-negative number")
        pi_bo_beta = float(pi_bo_beta)

    bounds = np.asarray(bounds, dtype=float)
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    rng = np.random.default_rng(seed)

    def to_native(U: np.ndarray) -> np.ndarray:
        U = np.atleast_2d(np.asarray(U, dtype=float))
        return lo + (hi - lo) * U

    def g(U: np.ndarray) -> np.ndarray:                 # maximize g = -f
        return -np.array([float(objective(x)) for x in to_native(U)])

    # Frozen-reference normalized prior over the unit cube.
    pibar: Optional[Callable[[np.ndarray], np.ndarray]] = None
    prior_peaks: Optional[np.ndarray] = None
    if prior_fn is not None and (alpha > 0.0 or injection):
        # Prior peaks (_prior_peak_search): min-max reference and per-step candidates.
        top8, prior_peaks = _prior_peak_search(prior_fn, d, lo, hi, to_native, seed)
        ref = np.vstack([rng.random((4096, d)), top8, prior_peaks])
        rp = np.array([float(prior_fn(x)) for x in to_native(ref)])
        pmin, span = float(rp.min()), float(rp.max() - rp.min())
        if span > 1e-12:

            def pibar(U: np.ndarray) -> np.ndarray:
                U = np.atleast_2d(U)
                v = np.array([float(prior_fn(x)) for x in to_native(U)])
                return np.clip((v - pmin) / (span + 1e-12), 0.0, 1.0)

        # else: the prior is constant over the box reference (e.g. a narrow product-of-Gaussians
        # that numerically underflows to ~0 on a wide domain). pi_bar would be uniform, so the
        # multiplier M(x) = (1-alpha) + alpha*pi_bar is identically 1 and alpha-pi reduces to
        # vanilla. Leaving `pibar=None` makes this exact: the alpha-pi arm then runs the unshaped
        # acquisition (no multiply, no zero-clamp), identical to vanilla.

    ei = ExpectedImprovement(xi=xi)
    pi_bo_max: Optional[float] = None   # pi-bo: the C0 maximum of the prior (piBO's normalizer)
    prior_mean: Optional[Callable[[np.ndarray], np.ndarray]] = None
    U0: Optional[np.ndarray] = None
    if injection:
        # Candidate set C0 of the injection arms: the alpha-pi arm's first acquisition batch (the Sobol batch
        # of step len(y) = n_initial plus the prior peaks; no peaks for a constant prior, the constant-prior
        # rule of the alpha-pi arm), restricted to its distinct points below.
        cand0 = _sobol(n_sobol, d, seed + n_init)
        if pibar is not None:
            cand0 = np.vstack([cand0, prior_peaks])
        pc_all = np.array([float(prior_fn(x)) for x in to_native(cand0)])
        if prior_strategy in ("prior-warm-start", "prior-greedy"):
            # the n_initial (warm start) or budget (greedy) highest-prior distinct points
            count = budget if prior_strategy == "prior-greedy" else n_init
            U0 = cand0[distinct_by_prior(cand0, pc_all, limit=count)]
            if len(U0) < count:
                raise ValueError(f"{prior_strategy}: only {len(U0)} distinct candidates for {count} queries")
        else:
            keep = np.sort(distinct_by_prior(cand0, pc_all))   # C0 rows, in batch order
            C0, pc = cand0[keep], pc_all[keep]
            if prior_strategy == "pi-bo":
                U0 = C0[list(pi_bo_initial_design(pc, n_init, seed))]
                pi_bo_max = float(pc.max())
            else:  # prior-to-mean: m_z = beta * standardize_C0(pi) (worker._prior_mean_from_pi)
                mu, sd = float(pc.mean()), float(pc.std())
                if sd > 1e-12:
                    beta = float(prior_mean_beta)

                    def prior_mean(Umat: np.ndarray) -> np.ndarray:
                        v = np.array([float(prior_fn(x)) for x in to_native(Umat)])
                        return beta * ((v - mu) / sd)
    surro = GaussianProcessSurrogate(
        length_scale=length_scale, noise_variance=noise_std**2, normalize_y=True, prior_mean=prior_mean
    )
    # Initial design: the seeded Sobol design (alpha-pi, vanilla, vanilla-peaks, prior-to-mean),
    # or the prior-based design U0 of pi-bo, prior-warm-start and prior-greedy (whole run).
    if U0 is not None:
        U = U0
    else:
        U = _sobol(n_init, d, seed)
    y = g(U)

    def shaped(Umat: np.ndarray) -> np.ndarray:         # acquisition (higher=better)
        Umat = np.atleast_2d(Umat)
        mean, std = surro.predict(Umat)
        raw = ei.score(mean, std, best_value=float(y.max()), prior=None)
        raw = np.asarray(raw, dtype=float).reshape(-1)
        if pibar is not None and shape_acquisition and not injection:
            raw = np.maximum(raw * ((1.0 - alpha) + alpha * pibar(Umat)), 0.0)
        elif pi_bo_max is not None:
            # piBO's decay counter: t = 1 at the first query after the initial design
            v = np.array([float(prior_fn(x)) for x in to_native(Umat)])
            raw = acq.pi_bo_shape(raw, v, pi_bo_beta, len(y) - n_init + 1, reference_max=pi_bo_max)
        return raw

    while len(y) < budget:
        surro.fit(U, y)
        cand = _sobol(n_sobol, d, seed + len(y))
        if pibar is not None and prior_peaks is not None:
            cand = np.vstack([cand, prior_peaks])
        order = np.argsort(-shaped(cand))
        starts = cand[order[:n_restarts]]
        best_u = starts[0]
        best_val = float(shaped(starts[:1])[0])
        for x0 in starts:
            res = minimize(lambda u: -float(shaped(np.clip(u, 0.0, 1.0))[0]),
                           x0, method="L-BFGS-B", bounds=[(0.0, 1.0)] * d)
            val = -float(res.fun)
            if val > best_val:
                best_val, best_u = val, np.clip(res.x, 0.0, 1.0)
        U = np.vstack([U, best_u])
        y = np.append(y, g(best_u.reshape(1, -1))[0])

    # best-so-far regret vs the true optimum (minimization convention); clamp tiny
    # negative float noise (best_so_far can dip ~1e-9 below optimum_value numerically).
    fvals = -y
    best_so_far = np.minimum.accumulate(fvals)
    regret = np.maximum(best_so_far - float(optimum_value), 0.0)
    best_i = int(np.argmin(fvals))
    return ContinuousBOResult(
        best_x=to_native(U[best_i])[0],
        best_min_value=float(fvals[best_i]),
        X_native=to_native(U),
        simple_regret=regret,
        prior_peaks=prior_peaks if pibar is not None else None,
        pi_bo_beta=pi_bo_beta if prior_strategy == "pi-bo" else None,
    )


__all__ = ["ContinuousBOResult", "DISTINCT_TOL", "PRIOR_STRATEGIES", "distinct_by_prior", "run_continuous_bo"]
