"""Acquisition functions for discrete Bayesian optimization.

This module deliberately knows nothing about how a prior was created.  A prior
is simply an array containing one preference value per candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf
from typing import Optional

import numpy as np


# Upper bound of the alpha-pi trust weight: a low-prior candidate keeps a
# multiplier of at least 1 - ALPHA_MAX = 0.1.
ALPHA_MAX = 0.9
# piBO requires a strictly positive prior on the search space (Hvarfner et al.,
# ICLR 2022, Sec. 2 and Theorem 1); their implementation adds this constant to the
# prior everywhere (Appendix F.1).  It is added after the prior is scaled to a
# maximum of 1 (pi_bo_prior).
PI_BO_PRIOR_EPSILON = 1e-12


def _standard_normal_cdf(values: np.ndarray) -> np.ndarray:
    """Evaluate the standard normal CDF without adding a SciPy dependency."""

    values = np.asarray(values, dtype=float)
    flat = np.fromiter(
        (0.5 * (1.0 + erf(float(value) / np.sqrt(2.0))) for value in values.flat),
        dtype=float,
        count=values.size,
    )
    return flat.reshape(values.shape)


def upper_confidence_bound(
    mean: np.ndarray,
    std: np.ndarray,
    kappa: float = 2.0,
) -> np.ndarray:
    """Return UCB scores ``mean + kappa * std`` for a maximization problem."""
    mean = np.asarray(mean, dtype=float).reshape(-1)
    std = np.asarray(std, dtype=float).reshape(-1)
    if mean.shape != std.shape:
        raise ValueError("mean and std must have the same shape")
    if kappa < 0:
        raise ValueError("kappa must be non-negative")
    return mean + kappa * np.maximum(std, 0.0)


def probability_of_feasibility(
    mean: np.ndarray,
    std: np.ndarray,
    *,
    threshold: float = 0.0,
) -> np.ndarray:
    """Return ``P(g(x) <= threshold)`` for Gaussian constraint predictions."""

    mean = np.asarray(mean, dtype=float).reshape(-1)
    std = np.maximum(np.asarray(std, dtype=float).reshape(-1), 0.0)
    if mean.shape != std.shape:
        raise ValueError("mean and std must have the same shape")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    probability = np.empty_like(mean)
    uncertain = std > 1e-12
    probability[~uncertain] = (mean[~uncertain] <= threshold).astype(float)
    if np.any(uncertain):
        z = (float(threshold) - mean[uncertain]) / std[uncertain]
        probability[uncertain] = _standard_normal_cdf(z)
    return np.clip(probability, 0.0, 1.0)


def normalize_prior(prior: np.ndarray) -> np.ndarray:
    """Normalize a numeric candidate prior to [0, 1]."""
    prior = np.asarray(prior, dtype=float).reshape(-1)
    if prior.size == 0:
        raise ValueError("prior cannot be empty")
    if not np.all(np.isfinite(prior)):
        raise ValueError("prior must contain only finite values")

    prior_min = float(np.min(prior))
    prior_max = float(np.max(prior))
    if prior_max - prior_min <= 1e-12:
        # A uniform prior should not alter the acquisition function.
        return np.ones_like(prior)
    return (prior - prior_min) / (prior_max - prior_min)


def _valid_mask(values: np.ndarray, valid_mask: Optional[np.ndarray]) -> np.ndarray:
    """Return a finite, shape-checked mask for acquisition values."""

    if valid_mask is None:
        mask = np.ones(values.size, dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if mask.shape != values.shape:
            raise ValueError("valid_mask must have one entry per acquisition value")
    return mask & np.isfinite(values)


def _validate_alpha(alpha: float) -> None:
    if not 0.0 <= alpha <= ALPHA_MAX:
        raise ValueError(f"alpha must be between 0 and {ALPHA_MAX}")


def alpha_pi_shape(
    acquisition_values: np.ndarray,
    prior: Optional[np.ndarray],
    alpha: float,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply the alpha-pi rule.

    The prior multiplies the raw acquisition, which is not shifted or min-max
    normalized before weighting::

        weighted(x) = acquisition_raw(x) * (alpha * prior_norm(x) + 1 - alpha)
        shaped(x) = max(weighted(x), 0)

    Alpha is limited to 0.9. Low-prior candidates therefore retain a multiplier
    of at least 0.1, while the zero clamp removes negative weighted scores.
    """
    _validate_alpha(alpha)
    acquisition = np.asarray(acquisition_values, dtype=float).reshape(-1)
    mask = _valid_mask(acquisition, valid_mask)
    shaped = np.full(acquisition.shape, -np.inf, dtype=float)
    if not np.any(mask):
        return shaped

    if prior is None or alpha == 0.0:
        shaped[mask] = np.maximum(acquisition[mask], 0.0)
        return shaped

    prior_norm = normalize_prior(prior)
    if prior_norm.shape != acquisition.shape:
        raise ValueError("prior must have one entry per acquisition value")
    multiplier = alpha * prior_norm + (1.0 - alpha)
    shaped[mask] = np.maximum(acquisition[mask] * multiplier[mask], 0.0)
    return shaped


def pi_bo_beta(budget: int, n_initial: int) -> float:
    """piBO's beta, N / 10 with N = budget - n_initial BO iterations (Hvarfner et al. 2022, Sec. 4)."""
    return max(int(budget) - max(1, int(n_initial)), 0) / 10.0


def pi_bo_prior(prior: np.ndarray, reference_max: Optional[float] = None) -> np.ndarray:
    """Strictly positive prior weighted by piBO.

    ``p(x) = max(prior(x), 0) / max(prior) + PI_BO_PRIOR_EPSILON``.  The prior is
    scaled by its maximum without subtracting its minimum, so ``p`` does not depend
    on the scale of ``prior`` and no candidate has weight zero.  A prior with no
    positive value carries no preference and gives ``p = 1`` everywhere.

    ``reference_max`` replaces ``max(prior)`` by the maximum over a fixed reference
    set (the continuous path, which scores varying batches); values above it are
    capped at 1, so ``p`` stays in ``[eps, 1 + eps]``.
    """
    prior = np.asarray(prior, dtype=float).reshape(-1)
    if prior.size == 0:
        raise ValueError("prior cannot be empty")
    if not np.all(np.isfinite(prior)):
        raise ValueError("prior must contain only finite values")

    if reference_max is not None:
        prior_max = float(reference_max)
        if prior_max <= 0.0:
            return np.ones_like(prior)
        return np.minimum(np.maximum(prior, 0.0) / prior_max, 1.0) + PI_BO_PRIOR_EPSILON
    prior_max = float(np.max(prior))
    if prior_max <= 0.0:
        return np.ones_like(prior)
    return np.maximum(prior, 0.0) / prior_max + PI_BO_PRIOR_EPSILON


def pi_bo_shape(
    acquisition_values: np.ndarray,
    prior: Optional[np.ndarray],
    beta: float,
    t: float,
    valid_mask: Optional[np.ndarray] = None,
    reference_max: Optional[float] = None,
) -> np.ndarray:
    """piBO (Hvarfner et al. 2022) decayed prior weighting.

    ``weighted(x) = acquisition_raw(x) * p(x) ** (beta / max(t, 1))`` with the
    strictly positive prior ``p`` of :func:`pi_bo_prior`, followed by a zero clamp.
    ``t`` is piBO's iteration counter, 1 at the first query after the initial
    design (the mode of ``p``, then points sampled from it, :func:`~.bo.pi_bo_initial_design`).
    Unlike :func:`alpha_pi_shape` there is **no** ``(1 - alpha)`` floor: a candidate
    the prior rules out has weight ``PI_BO_PRIOR_EPSILON ** (beta / t)``, which
    grows toward 1 as ``t`` grows.  This is deliberately *not* routed through
    :func:`alpha_pi_shape`; it exists only for the piBO arm and must never be
    confused with the bounded alpha-pi rule.
    """

    if beta < 0:
        raise ValueError("beta must be non-negative")
    acquisition = np.asarray(acquisition_values, dtype=float).reshape(-1)
    mask = _valid_mask(acquisition, valid_mask)
    shaped = np.full(acquisition.shape, -np.inf, dtype=float)
    if not np.any(mask):
        return shaped

    if prior is None or beta == 0.0:
        shaped[mask] = np.maximum(acquisition[mask], 0.0)
        return shaped

    prior_weight = pi_bo_prior(prior, reference_max)
    if prior_weight.shape != acquisition.shape:
        raise ValueError("prior must have one entry per acquisition value")
    exponent = float(beta) / max(float(t), 1.0)
    weight = np.power(prior_weight, exponent)
    shaped[mask] = np.maximum(acquisition[mask] * weight[mask], 0.0)
    return shaped


@dataclass
class PriorShapedUCB:
    """UCB followed by optional alpha-pi prior shaping.

    ``ucb_center_incumbent`` selects the *incumbent-relative* UCB convention (the
    finite-path acquisition); it applies to vanilla BO and HapBO alike.  Raw UCB is an
    absolute score that can be negative, and the alpha-pi rule's ``max(., 0)`` clamp is
    not meaningful on a signed score: it maps every below-zero candidate to the same
    value.  Centering on the incumbent turns UCB into a confidence-bound
    *improvement over the current best*, an EI/PI-like quantity.  Without a prior
    (vanilla UCB) the centred score is returned unclipped, which preserves the ranking
    of below-incumbent candidates.  With a prior, the alpha-pi multiplier is applied to
    the clipped improvement ``max(UCB - incumbent, 0)``, falling back to the unshaped
    centred ordering when no valid candidate has a positive improvement, so the shaping
    acts consistently with the nonnegative EI and PI improvements.  Default ``False``
    uses the raw UCB score.
    """

    kappa: float = 2.0
    alpha: float = 0.0
    ucb_center_incumbent: bool = False

    def __post_init__(self) -> None:
        if self.kappa < 0:
            raise ValueError("kappa must be non-negative")
        _validate_alpha(self.alpha)

    def score(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        prior: Optional[np.ndarray] = None,
        excluded_mask: Optional[np.ndarray] = None,
        best_value: Optional[float] = None,
    ) -> np.ndarray:
        """Return shaped scores, with excluded candidates set to ``-inf``."""
        ucb = upper_confidence_bound(mean, std, self.kappa)
        if excluded_mask is None:
            valid_mask = np.ones(ucb.size, dtype=bool)
        else:
            excluded_mask = np.asarray(excluded_mask, dtype=bool).reshape(-1)
            if excluded_mask.shape != ucb.shape:
                raise ValueError("excluded_mask must have one entry per candidate")
            valid_mask = ~excluded_mask

        if self.ucb_center_incumbent:
            if best_value is None or not np.isfinite(best_value):
                raise ValueError("ucb_center_incumbent requires a finite best_value")
            centered = ucb - float(best_value)
            out = np.full(centered.shape, -np.inf, dtype=float)
            if prior is not None and self.alpha > 0.0:
                # Shape the nonnegative optimistic improvement max(UCB - incumbent, 0) * M with
                # M = (1 - alpha) + alpha * pi, so the multiplier can only promote high-prior
                # candidates (A >= 0, as for EI and PI).  When every valid shaped score is 0 (no
                # candidate's UCB exceeds the incumbent) the prior has nothing to reweight: use
                # the unshaped centred UCB ordering, which is the vanilla UCB choice.
                multiplier = self.alpha * normalize_prior(prior) + (1.0 - self.alpha)
                shaped = np.maximum(centered[valid_mask], 0.0) * multiplier[valid_mask]
                if np.any(shaped > 0.0):
                    out[valid_mask] = shaped
                else:
                    out[valid_mask] = centered[valid_mask]
            else:
                out[valid_mask] = centered[valid_mask]
            return out

        if prior is not None and self.alpha > 0.0:
            return alpha_pi_shape(ucb, prior, self.alpha, valid_mask)
        ucb[~valid_mask] = -np.inf
        return ucb


@dataclass
class ExpectedImprovement:
    """Expected Improvement followed by optional alpha-pi prior shaping."""

    xi: float = 0.0
    alpha: float = 0.0

    def __post_init__(self) -> None:
        if self.xi < 0:
            raise ValueError("xi must be non-negative")
        _validate_alpha(self.alpha)

    def score(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        best_value: float,
        excluded_mask: Optional[np.ndarray] = None,
        prior: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return optionally shaped EI scores with exclusions set to ``-inf``."""

        mean = np.asarray(mean, dtype=float).reshape(-1)
        std = np.maximum(np.asarray(std, dtype=float).reshape(-1), 0.0)
        if mean.shape != std.shape:
            raise ValueError("mean and std must have the same shape")
        if not np.isfinite(best_value):
            raise ValueError("best_value must be finite")

        improvement = mean - float(best_value) - self.xi
        positive_std = std > 1e-12
        scores = np.maximum(improvement, 0.0)
        if np.any(positive_std):
            z = improvement[positive_std] / std[positive_std]
            cdf = _standard_normal_cdf(z)
            pdf = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
            scores[positive_std] = (
                improvement[positive_std] * cdf + std[positive_std] * pdf
            )
        scores = np.maximum(scores, 0.0)

        if excluded_mask is None:
            valid_mask = np.ones(scores.size, dtype=bool)
        else:
            excluded_mask = np.asarray(excluded_mask, dtype=bool).reshape(-1)
            if excluded_mask.shape != scores.shape:
                raise ValueError("excluded_mask must have one entry per candidate")
            valid_mask = ~excluded_mask

        if prior is not None and self.alpha > 0.0:
            return alpha_pi_shape(scores, prior, self.alpha, valid_mask)

        scores[~valid_mask] = -np.inf
        return scores


@dataclass
class ProbabilityOfImprovement:
    """Probability of Improvement followed by optional alpha-pi prior shaping."""

    xi: float = 0.0
    alpha: float = 0.0

    def __post_init__(self) -> None:
        if self.xi < 0:
            raise ValueError("xi must be non-negative")
        _validate_alpha(self.alpha)

    def score(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        best_value: float,
        excluded_mask: Optional[np.ndarray] = None,
        prior: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return optionally shaped PI scores with exclusions set to ``-inf``."""

        mean = np.asarray(mean, dtype=float).reshape(-1)
        std = np.maximum(np.asarray(std, dtype=float).reshape(-1), 0.0)
        if mean.shape != std.shape:
            raise ValueError("mean and std must have the same shape")
        if not np.isfinite(best_value):
            raise ValueError("best_value must be finite")

        improvement = mean - float(best_value) - self.xi
        positive_std = std > 1e-12
        scores = (improvement > 0.0).astype(float)  # degenerate std -> 0/1 indicator
        if np.any(positive_std):
            scores[positive_std] = _standard_normal_cdf(
                improvement[positive_std] / std[positive_std]
            )
        scores = np.clip(scores, 0.0, 1.0)

        if excluded_mask is None:
            valid_mask = np.ones(scores.size, dtype=bool)
        else:
            excluded_mask = np.asarray(excluded_mask, dtype=bool).reshape(-1)
            if excluded_mask.shape != scores.shape:
                raise ValueError("excluded_mask must have one entry per candidate")
            valid_mask = ~excluded_mask

        if prior is not None and self.alpha > 0.0:
            return alpha_pi_shape(scores, prior, self.alpha, valid_mask)

        scores[~valid_mask] = -np.inf
        return scores
