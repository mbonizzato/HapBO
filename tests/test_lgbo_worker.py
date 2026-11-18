from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "baselines" / "workers"))

import lgbo_worker as lgbo  # noqa: E402


class LGBOMathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gp = lgbo._GP(length_scale=0.31, noise_std=0.02).fit(
            np.array([[0.05], [0.35], [0.8]]),
            np.array([0.1, 0.9, 0.2]),
            optimize=False,
        )
        self.G = np.array([[0.2], [0.4], [0.6]])
        self.a = np.array([0.2, 0.5, 0.3])

    def test_direct_confidence_is_exact_standardized_functional_lift(self) -> None:
        confidence = 0.6
        shift, q, lam = lgbo._regional_shift_z(
            self.gp, self.G, self.G, self.a, confidence
        )
        self.assertGreater(q, 0.0)
        self.assertGreater(lam, 0.0)
        realized = float(self.a @ shift) / np.sqrt(q)
        self.assertAlmostEqual(realized, confidence, places=10)

    def test_regional_shift_uses_the_posterior_cross_covariance(self) -> None:
        posterior = self.gp.posterior_cross_cov(self.G, self.G)
        np.testing.assert_allclose(posterior, posterior.T, atol=1e-12)
        self.assertFalse(np.allclose(posterior, self.gp.cross_kernel(self.G, self.G)))

    def test_zero_confidence_has_no_shift(self) -> None:
        shift, _, lam = lgbo._regional_shift_z(
            self.gp, self.G, self.G, self.a, 0.0
        )
        np.testing.assert_array_equal(shift, np.zeros(len(self.G)))
        self.assertEqual(lam, 0.0)


if __name__ == "__main__":
    unittest.main()
