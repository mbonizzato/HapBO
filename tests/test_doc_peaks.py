"""Document peak points seed the prior-maximizer search of the continuous BO path."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bayesian_optimization.continuous_bo import run_continuous_bo  # noqa: E402
from prior.landscape.landscape_generator import document_peak_points  # noqa: E402

PRIOR = PROJECT_ROOT / "experiments/synthetic_priors/rho90/hartmann6/draw1/prior.json"


class DocumentPeakTests(unittest.TestCase):
    def test_points_are_complete_and_inside_the_box(self) -> None:
        record = json.loads(PRIOR.read_text(encoding="utf-8"))
        dims = record["schema"]["dimensions"]
        points = document_peak_points(record)
        self.assertGreaterEqual(len(points), 1)
        for p in points:
            self.assertEqual(set(p), {d["name"] for d in dims})
            for d in dims:
                lo, hi = map(float, d["range"])
                self.assertTrue(lo - 1e-9 <= p[d["name"]] <= hi + 1e-9)

    def test_narrow_planted_peak_is_reached(self) -> None:
        d = 6
        peak = np.array([0.83, 0.11, 0.67, 0.29, 0.91, 0.44])
        bounds = np.array([[0.0, 1.0]] * d)

        def objective(x):
            return float(np.sum((np.asarray(x) - peak) ** 2))

        def prior_fn(x):
            return float(np.exp(-np.sum((np.asarray(x) - peak) ** 2) / (2 * 0.004 ** 2)))

        def run(with_peaks: bool) -> float:
            fn = prior_fn
            if with_peaks:
                def fn(x):  # noqa: E306
                    return prior_fn(x)
                fn.peak_candidates = peak[None, :]
            res = run_continuous_bo(objective, bounds, 0.0, budget=3, n_initial=1, alpha=0.9, prior_fn=fn,
                                    n_sobol=256, n_restarts=2, seed=3)
            return float(res.simple_regret[-1])

        self.assertLess(run(True), 1e-4)
        self.assertGreater(run(False), 1e-3)


if __name__ == "__main__":
    unittest.main()
