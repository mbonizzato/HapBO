"""Incumbent-centred alpha-pi UCB shapes a nonnegative improvement.

``PriorShapedUCB(ucb_center_incumbent=True)`` multiplies ``max(UCB - incumbent, 0)`` by
``M = (1 - alpha) + alpha * pi``; when every valid shaped score is 0 it uses the unshaped
centred UCB ordering, so a candidate with a smaller ``M`` is never promoted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bayesian_optimization import PriorShapedUCB  # noqa: E402

KAPPA = 2.0


def _scores(centred: np.ndarray, prior, alpha: float, excluded=None) -> np.ndarray:
    """Score candidates whose centred UCB equals ``centred`` (std = 0, incumbent = 0)."""

    acquisition = PriorShapedUCB(kappa=KAPPA, alpha=alpha, ucb_center_incumbent=True)
    centred = np.asarray(centred, dtype=float)
    return acquisition.score(
        mean=centred,
        std=np.zeros_like(centred),
        prior=None if prior is None else np.asarray(prior, dtype=float),
        excluded_mask=excluded,
        best_value=0.0,
    )


def test_all_negative_picks_the_vanilla_candidate() -> None:
    centred = [-1.0, -1.0, -0.9]
    prior = [1.0, 0.0, 0.5]
    vanilla = _scores(centred, None, 0.0)
    shaped = _scores(centred, prior, 0.9)
    assert int(np.argmax(vanilla)) == 2
    # Multiplying the unclipped scores would pick the pi = 0 candidate: -1 * 0.1 > -0.9 * 0.55.
    assert int(np.argmax(shaped)) == int(np.argmax(vanilla))
    # Fallback is the unshaped centred ordering itself.
    np.testing.assert_array_equal(shaped, vanilla)


def test_mixed_sign_prefers_high_prior_among_positive_scores() -> None:
    centred = [0.30, 0.25, -0.5, -0.01]
    prior = [0.0, 1.0, 1.0, 0.0]
    vanilla = _scores(centred, None, 0.0)
    shaped = _scores(centred, prior, 0.9)
    assert int(np.argmax(vanilla)) == 0
    # M = [0.1, 1.0, 1.0, 0.1]: 0.25 * 1.0 > 0.30 * 0.1 -> the high-prior positive wins.
    assert int(np.argmax(shaped)) == 1
    # Negative candidates are clipped to 0, never above a positive candidate.
    np.testing.assert_allclose(shaped, [0.03, 0.25, 0.0, 0.0])
    # Negative candidates are 0 whatever their multiplier, below every positive score.
    assert shaped[2] == 0.0 and shaped[3] == 0.0


def test_alpha_zero_equals_vanilla() -> None:
    rng = np.random.default_rng(7)
    mean = rng.normal(size=25)
    std = rng.uniform(0.0, 0.5, size=25)
    prior = rng.uniform(size=25)
    excluded = np.zeros(25, dtype=bool)
    excluded[[3, 11]] = True
    best = float(np.max(mean[:5]))
    vanilla = PriorShapedUCB(kappa=KAPPA, alpha=0.0, ucb_center_incumbent=True)
    zero_alpha = PriorShapedUCB(kappa=KAPPA, alpha=0.0, ucb_center_incumbent=True)
    a = vanilla.score(mean, std, None, excluded, best)
    b = zero_alpha.score(mean, std, prior, excluded, best)
    np.testing.assert_array_equal(a, b)
    # Vanilla centred UCB is unchanged: unclipped UCB - incumbent on valid candidates.
    expected = mean + KAPPA * std - best
    expected[excluded] = -np.inf
    np.testing.assert_array_equal(a, expected)


def test_excluded_candidates_stay_minus_inf_and_fallback_ignores_them() -> None:
    centred = [5.0, -1.0, -0.2]
    prior = [0.0, 1.0, 0.0]
    excluded = np.array([True, False, False])
    shaped = _scores(centred, prior, 0.9, excluded=excluded)
    assert shaped[0] == -np.inf
    # Only valid candidates are all-negative -> unshaped centred ordering.
    assert int(np.argmax(shaped)) == 2
    np.testing.assert_array_equal(shaped[1:], [-1.0, -0.2])


def test_positive_scores_are_scaled_by_the_alpha_pi_multiplier() -> None:
    centred = np.array([0.4, 0.2, 0.1])
    prior = np.array([0.2, 0.6, 1.0])  # normalize_prior -> [0, 0.5, 1]
    alpha = 0.5
    shaped = _scores(centred, prior, alpha)
    multiplier = alpha * np.array([0.0, 0.5, 1.0]) + (1.0 - alpha)
    np.testing.assert_allclose(shaped, centred * multiplier)
