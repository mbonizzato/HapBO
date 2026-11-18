"""Probability of Improvement and native piBO decay shaping."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bayesian_optimization import (  # noqa: E402
    BayesianOptimization,
    ProbabilityOfImprovement,
    alpha_pi_shape,
    pi_bo_initial_design,
    pi_bo_prior,
    pi_bo_shape,
)
from bayesian_optimization.acquisition import (  # noqa: E402
    PI_BO_PRIOR_EPSILON,
    _standard_normal_cdf,
)


def test_probability_of_improvement_matches_closed_form() -> None:
    mean = np.array([0.2, 0.6, 0.9])
    std = np.array([0.1, 0.3, 0.2])
    best = 0.5
    scores = ProbabilityOfImprovement().score(mean, std, best_value=best)
    expected = _standard_normal_cdf((mean - best) / std)
    np.testing.assert_allclose(scores, expected)


def test_probability_of_improvement_handles_zero_std() -> None:
    mean = np.array([0.4, 0.7])
    std = np.array([0.0, 0.0])
    scores = ProbabilityOfImprovement().score(mean, std, best_value=0.5)
    np.testing.assert_array_equal(scores, np.array([0.0, 1.0]))


def test_pi_bo_prior_is_strictly_positive_and_not_min_subtracted() -> None:
    prior = np.array([-3.0, 0.0, 0.2, 0.5, 1.0])
    p = pi_bo_prior(prior)
    np.testing.assert_allclose(
        p, np.array([0.0, 0.0, 0.2, 0.5, 1.0]) + PI_BO_PRIOR_EPSILON, rtol=0, atol=1e-15
    )
    assert np.all(p > 0.0)
    assert p[2] > PI_BO_PRIOR_EPSILON  # the lowest positive prior is not mapped to zero


@pytest.mark.parametrize("prior", [np.zeros(4), np.full(4, -2.0), np.array([-1.0, 0.0, -5.0, 0.0])])
def test_pi_bo_prior_without_positive_value_is_uniform(prior) -> None:
    np.testing.assert_array_equal(pi_bo_prior(prior), np.ones(4))
    acq = np.array([-1.0, 0.5, 2.0, 1.0])
    np.testing.assert_array_equal(pi_bo_shape(acq, prior, beta=1.5, t=1), np.maximum(acq, 0.0))


def test_pi_bo_prior_rejects_empty_and_non_finite() -> None:
    with pytest.raises(ValueError):
        pi_bo_prior(np.array([]))
    with pytest.raises(ValueError):
        pi_bo_prior(np.array([0.5, np.nan]))


def test_pi_bo_hand_computed_case() -> None:
    # max prior 4: p = [eps, 0.25, 0.5, 1] (+ eps); t = 3, beta = 1.5 -> weight = sqrt(p)
    acq = np.array([1.0, 1.0, 1.0, 0.5])
    prior = np.array([0.0, 1.0, 2.0, 4.0])
    shaped = pi_bo_shape(acq, prior, beta=1.5, t=3)
    np.testing.assert_allclose(shaped, [1e-6, 0.5, np.sqrt(0.5), 0.5], rtol=1e-9)
    assert int(np.argmax(shaped)) == 2


def test_pi_bo_zero_prior_weight_is_positive_and_grows_with_t() -> None:
    acq = np.ones(3)
    prior = np.array([0.0, 0.5, 1.0])
    zero_weight = [pi_bo_shape(acq, prior, beta=1.5, t=t)[0] for t in range(1, 17)]
    assert all(w > 0.0 for w in zero_weight)
    assert all(b > a for a, b in zip(zero_weight, zero_weight[1:]))
    assert zero_weight[0] == pytest.approx(PI_BO_PRIOR_EPSILON ** 1.5, rel=1e-9)
    assert zero_weight[-1] == pytest.approx(PI_BO_PRIOR_EPSILON ** (1.5 / 16), rel=1e-9)


def test_pi_bo_decays_toward_unshaped_as_t_grows() -> None:
    # The exponent beta / t -> 0 sends every weight, zero-prior ones included, to 1.
    acq = np.array([1.0, 2.0, 3.0])
    prior = np.array([0.0, 0.5, 1.0])
    late = pi_bo_shape(acq, prior, beta=1.5, t=1_000_000)
    np.testing.assert_allclose(late, acq, rtol=1e-4)


@pytest.mark.parametrize("scale", [1e-6, 0.37, 1.0, 42.0, 1e6])
def test_pi_bo_argmax_is_invariant_to_prior_scale(scale) -> None:
    rng = np.random.default_rng(3)
    acq = rng.uniform(0.0, 1.0, 50)
    prior = rng.uniform(0.0, 1.0, 50)
    prior[:10] = 0.0
    for t in (1, 2, 5, 16):
        reference = pi_bo_shape(acq, prior, beta=1.5, t=t)
        scaled = pi_bo_shape(acq, scale * prior, beta=1.5, t=t)
        np.testing.assert_allclose(scaled, reference, rtol=1e-12)
        assert int(np.argmax(scaled)) == int(np.argmax(reference))


def test_pi_bo_has_no_floor_unlike_alpha_pi() -> None:
    acq = np.array([2.0, 2.0])
    prior = np.array([0.0, 1.0])  # first candidate has zero prior
    pib = pi_bo_shape(acq, prior, beta=1.5, t=1)
    api = alpha_pi_shape(acq, prior, alpha=0.5)
    assert pib[0] == pytest.approx(2.0 * PI_BO_PRIOR_EPSILON ** 1.5)  # piBO: weight eps ** beta at t = 1
    assert api[0] == pytest.approx(1.0)  # alpha-pi keeps the (1 - alpha) floor


def test_alpha_pi_keeps_the_min_max_prior() -> None:
    # alpha-pi rescales the prior to [0, 1] (minimum subtracted); piBO does not.
    acq = np.ones(3)
    prior = np.array([0.2, 0.6, 1.0])
    np.testing.assert_allclose(alpha_pi_shape(acq, prior, alpha=0.9), [0.1, 0.55, 1.0])
    np.testing.assert_allclose(
        pi_bo_shape(acq, prior, beta=1.5, t=1), (prior + PI_BO_PRIOR_EPSILON) ** 1.5, rtol=1e-12
    )


@pytest.mark.parametrize("t", [1, 3, 16])
def test_pi_bo_beta_zero_is_unshaped(t) -> None:
    acq = np.array([-1.0, 1.0, 2.0])
    prior = np.array([1.0, 0.0, 0.5])
    np.testing.assert_array_equal(
        pi_bo_shape(acq, prior, beta=0.0, t=t), np.maximum(acq, 0.0)
    )


def test_pi_bo_beta_zero_optimizer_matches_plain_ei_after_the_same_initial_design() -> None:
    candidates = np.linspace(0.0, 1.0, 21).reshape(-1, 1)
    prior = np.zeros(21)
    prior[15:] = 1.0  # rules out the optimum near 0.3

    def run(**kwargs):
        opt = BayesianOptimization(
            candidates, n_initial=2, random_state=4, **kwargs
        )
        return opt.run(lambda x: -float((x[0] - 0.3) ** 2), max_evaluations=8)

    shaped = run(prior=prior, alpha=0.0, prior_shaping="pi-bo", pi_bo_beta=0.0)
    design = pi_bo_initial_design(prior, 2, 4)
    plain = run(initial_indices=design)
    assert tuple(shaped.evaluated_indices[:2]) == design
    assert list(shaped.evaluated_indices) == list(plain.evaluated_indices)


def _record_pi_bo_calls(monkeypatch):
    """Wrap the pi_bo_shape bound in bayesian_optimization.bo; return the list of (t, acq, out) per call."""
    module = importlib.import_module("bayesian_optimization.bo")
    original = module.pi_bo_shape
    calls = []

    def recorder(acquisition_values, prior, beta, t, valid_mask=None):
        out = original(acquisition_values, prior, beta, t, valid_mask=valid_mask)
        calls.append((t, np.array(acquisition_values, dtype=float), out))
        return out

    monkeypatch.setattr(module, "pi_bo_shape", recorder)
    return calls


def _pi_bo_run(n_initial=1, budget=9, beta=0.8):
    candidates = np.linspace(0.0, 1.0, 31).reshape(-1, 1)
    prior = np.zeros(31)
    prior[20:] = np.linspace(0.2, 1.0, 11)  # rules out the optimum near 0.3
    opt = BayesianOptimization(
        candidates, prior=prior, alpha=0.0, prior_shaping="pi-bo", pi_bo_beta=beta,
        n_initial=n_initial, random_state=11,
    )
    opt.run(lambda x: -float((x[0] - 0.3) ** 2), max_evaluations=budget)
    return prior


def test_pi_bo_first_query_after_the_initial_design_uses_beta_over_one(monkeypatch) -> None:
    calls = _record_pi_bo_calls(monkeypatch)
    prior = _pi_bo_run(n_initial=1, budget=4, beta=2.0)
    assert [t for t, *_ in calls] == [1, 2, 3]
    t, acq, out = calls[0]
    zero = np.flatnonzero((prior == 0.0) & np.isfinite(out) & (acq > 0.0))
    assert zero.size
    # weight of a ruled-out candidate at the first BO query: eps ** (beta / 1)
    np.testing.assert_allclose(out[zero] / acq[zero], PI_BO_PRIOR_EPSILON ** 2.0, rtol=1e-9)


@pytest.mark.parametrize("n_initial", [1, 2, 3, 5])
def test_pi_bo_counter_counts_queries_after_the_initial_design(monkeypatch, n_initial):
    calls = _record_pi_bo_calls(monkeypatch)
    _pi_bo_run(n_initial=n_initial, budget=9)
    # t runs 1, 2, ... from the first query after the n_initial-point initial design
    assert [t for t, *_ in calls] == list(range(1, 9 - n_initial + 1))


def test_pi_bo_masks_excluded_candidates() -> None:
    acq = np.array([1.0, 2.0, 3.0])
    prior = np.array([0.0, 0.5, 1.0])
    shaped = pi_bo_shape(acq, prior, beta=1.5, t=2, valid_mask=np.array([True, False, True]))
    assert shaped[1] == -np.inf
    assert np.all(np.isfinite(shaped[[0, 2]]))


def test_optimizer_supports_pi_acquisition_and_pi_bo_shaping() -> None:
    candidates = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
    prior = np.linspace(0.0, 1.0, 8)
    opt = BayesianOptimization(
        candidates,
        prior=prior,
        acquisition="probability-of-improvement",
        prior_shaping="pi-bo",
        pi_bo_beta=0.3,
        n_initial=2,
        random_state=0,
    )
    result = opt.run(lambda x: -float((x[0] - 0.7) ** 2), max_evaluations=5)
    assert len(set(result.evaluated_indices)) == 5
