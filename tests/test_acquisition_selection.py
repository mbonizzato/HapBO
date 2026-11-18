from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bayesian_optimization import (  # noqa: E402
    ALPHA_MAX,
    BayesianOptimization,
    ExpectedImprovement,
    PriorShapedUCB,
    alpha_pi_shape,
)


class AcquisitionSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = np.linspace(0.0, 1.0, 6).reshape(-1, 1)

    def test_expected_improvement_is_the_default(self) -> None:
        optimizer = BayesianOptimization(self.candidates, random_state=3)

        self.assertEqual(optimizer.acquisition_name, "expected-improvement")
        self.assertIsInstance(optimizer.acquisition, ExpectedImprovement)

    def test_ucb_can_be_selected(self) -> None:
        optimizer = BayesianOptimization(
            self.candidates,
            acquisition="ucb",
            random_state=3,
        )

        self.assertEqual(optimizer.acquisition_name, "ucb")
        self.assertIsInstance(optimizer.acquisition, PriorShapedUCB)

    def test_expected_improvement_supports_alpha_pi_shaping(self) -> None:
        mean = np.array([0.2, 0.4, 0.6, 0.5])
        std = np.array([0.1, 0.2, 0.05, 0.3])
        prior = np.array([0.0, 0.2, 1.0, 0.5])
        excluded = np.array([False, True, False, False])

        raw_scores = ExpectedImprovement().score(
            mean,
            std,
            best_value=0.45,
            excluded_mask=excluded,
        )
        positional_scores = ExpectedImprovement().score(
            mean,
            std,
            0.45,
            excluded,
        )
        shaped_scores = ExpectedImprovement(alpha=0.6).score(
            mean,
            std,
            best_value=0.45,
            prior=prior,
            excluded_mask=excluded,
        )

        expected = alpha_pi_shape(raw_scores, prior, 0.6, ~excluded)
        np.testing.assert_allclose(positional_scores, raw_scores)
        np.testing.assert_allclose(shaped_scores, expected)

    def test_alpha_pi_multiplies_raw_values_then_zero_clamps(self) -> None:
        acquisition = np.array([-1.0, 1.0, 2.0])
        prior = np.array([1.0, 1.0, 0.0])

        shaped = alpha_pi_shape(acquisition, prior, alpha=0.5)

        np.testing.assert_allclose(shaped, [0.0, 1.0, 1.0])

    def test_alpha_pi_keeps_the_raw_acquisition_scale(self) -> None:
        # The raw acquisition is not min-max normalized, so a high-prior candidate
        # with a slightly lower raw score wins the argmax.
        acquisition = np.array([10.0, 11.0])
        prior = np.array([1.0, 0.0])

        shaped = alpha_pi_shape(acquisition, prior, alpha=0.5)

        np.testing.assert_allclose(shaped, [10.0, 5.5])
        self.assertEqual(int(np.argmax(shaped)), 0)

    def test_alpha_is_capped_at_point_nine(self) -> None:
        values = np.array([0.2, 0.8])
        prior = np.array([0.0, 1.0])

        self.assertEqual(ALPHA_MAX, 0.9)
        alpha_pi_shape(values, prior, alpha=0.9)
        with self.assertRaisesRegex(ValueError, "between 0 and 0.9"):
            alpha_pi_shape(values, prior, alpha=0.900001)
        with self.assertRaisesRegex(ValueError, "between 0 and 0.9"):
            BayesianOptimization(
                self.candidates,
                prior=np.linspace(0.0, 1.0, len(self.candidates)),
                alpha=0.95,
            )

    def test_ucb_uses_alpha_pi_shaping_too(self) -> None:
        prior = np.linspace(0.0, 1.0, len(self.candidates))
        optimizer = BayesianOptimization(
            self.candidates,
            acquisition="ucb",
            prior=prior,
            alpha=0.6,
        )
        mean = np.linspace(-0.5, 0.5, len(self.candidates))
        std = np.full(len(self.candidates), 0.1)

        np.testing.assert_allclose(
            optimizer.acquisition.score(mean, std, prior=prior),
            alpha_pi_shape(mean + 2.0 * std, prior, 0.6),
        )

    def test_prior_guided_ei_runs_a_complete_loop(self) -> None:
        prior = np.array([0.0, 0.2, 0.4, 1.0, 0.5, 0.1])
        optimizer = BayesianOptimization(
            self.candidates,
            prior=prior,
            alpha=0.6,
            random_state=3,
        )

        result = optimizer.run(
            lambda x: -float((x[0] - 0.7) ** 2),
            max_evaluations=4,
        )
        self.assertEqual(len(set(result.evaluated_indices)), 4)


if __name__ == "__main__":
    unittest.main()
