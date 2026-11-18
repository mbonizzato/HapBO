"""Tests for the colabo-llm one-shot LLM point prior (BORA S C.2 construction)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.worker import _llm_point_gaussian_prior  # noqa: E402


def _write_artifact(path: Path, *, c: dict, s: float, dimensions: list) -> Path:
    sigma = 1.0 - s + 1e-6
    path.write_text(
        json.dumps(
            {
                "artifact_format": "colabo-llm-point-prior-v1",
                "c": c,
                "s": s,
                "sigma": sigma,
                "dimensions": dimensions,
            }
        ),
        encoding="utf-8",
    )
    return path


class _OrdinalGridProblem:
    """A finite 5x4 ordinal grid mimicking an HPOBench-style task."""

    def __init__(self):
        self.alpha_values = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
        self.eta_values = [1e-3, 1e-2, 1e-1, 1.0]
        self.native_candidates = tuple(
            {"alpha": a, "eta0": e} for a in self.alpha_values for e in self.eta_values
        )


class ColaBOLLMPriorTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _dims(self, problem):
        return [
            {"name": "alpha", "type": "ordinal", "values": problem.alpha_values, "scale": "log"},
            {"name": "eta0", "type": "ordinal", "values": problem.eta_values, "scale": "log"},
        ]

    def test_argmax_lands_on_candidate_nearest_c_and_confidence_is_sigma(self):
        problem = _OrdinalGridProblem()
        artifact = _write_artifact(
            self.tmp / "a.json",
            c={"alpha": 1e-3, "eta0": 1e-1},
            s=0.7,
            dimensions=self._dims(problem),
        )
        prior = _llm_point_gaussian_prior(problem, "hpobench-tabular", artifact)
        best = problem.native_candidates[int(np.argmax(prior["values"]))]
        self.assertEqual(best, {"alpha": 1e-3, "eta0": 1e-1})
        self.assertAlmostEqual(prior["confidence"], 1.0 - 0.7 + 1e-6, places=12)
        self.assertEqual(prior["values"].shape, (len(problem.native_candidates),))
        self.assertTrue(np.all(np.isfinite(prior["values"])))
        # the peak equals exp(0) because c sits exactly on a grid point
        self.assertAlmostEqual(float(np.max(prior["values"])), 1.0, places=12)

    def test_off_grid_c_snaps_to_nearest_candidate(self):
        problem = _OrdinalGridProblem()
        # c between grid values: nearest rank for alpha=2e-3 is 1e-3 (linear rank
        # snapping over the values list); eta0=0.5 is nearest 1.0 by linear distance.
        artifact = _write_artifact(
            self.tmp / "b.json",
            c={"alpha": 2e-3, "eta0": 0.5},
            s=0.2,
            dimensions=self._dims(problem),
        )
        prior = _llm_point_gaussian_prior(problem, "hpobench-tabular", artifact)
        best = problem.native_candidates[int(np.argmax(prior["values"]))]
        self.assertEqual(best["alpha"], 1e-3)
        self.assertEqual(best["eta0"], 1.0)

    def test_low_confidence_gives_flatter_prior_than_high_confidence(self):
        problem = _OrdinalGridProblem()
        dims = self._dims(problem)
        confident = _llm_point_gaussian_prior(
            problem,
            "hpobench-tabular",
            _write_artifact(self.tmp / "hi.json", c={"alpha": 1e-3, "eta0": 1e-1}, s=0.95, dimensions=dims),
        )
        unsure = _llm_point_gaussian_prior(
            problem,
            "hpobench-tabular",
            _write_artifact(self.tmp / "lo.json", c={"alpha": 1e-3, "eta0": 1e-1}, s=0.05, dimensions=dims),
        )
        self.assertLess(float(np.min(confident["values"])), float(np.min(unsure["values"])))
        self.assertGreater(unsure["confidence"], confident["confidence"])

    def test_rejects_wrong_artifact_format_and_missing_dimension(self):
        problem = _OrdinalGridProblem()
        bad_format = self.tmp / "bad.json"
        bad_format.write_text(json.dumps({"artifact_format": "other", "c": {}, "sigma": 0.3, "dimensions": []}))
        with self.assertRaises(ValueError):
            _llm_point_gaussian_prior(problem, "hpobench-tabular", bad_format)
        missing = _write_artifact(
            self.tmp / "missing.json",
            c={"alpha": 1e-3},
            s=0.5,
            dimensions=self._dims(problem),
        )
        with self.assertRaises(ValueError):
            _llm_point_gaussian_prior(problem, "hpobench-tabular", missing)

    def test_continuous_range_dimension_uses_log_min_max(self):
        class _ContinuousProblem:
            native_candidates = tuple({"width": w} for w in [16.0, 64.0, 256.0, 1024.0])

        dims = [{"name": "width", "type": "continuous", "range": [16.0, 1024.0], "scale": "log"}]
        artifact = _write_artifact(
            self.tmp / "c.json", c={"width": 200.0}, s=0.5, dimensions=dims
        )
        prior = _llm_point_gaussian_prior(_ContinuousProblem(), "hpobench-tabular", artifact)
        best = _ContinuousProblem.native_candidates[int(np.argmax(prior["values"]))]
        self.assertEqual(best["width"], 256.0)  # nearest in log space (200 is closer to 256 than 64)


if __name__ == "__main__":
    unittest.main()
