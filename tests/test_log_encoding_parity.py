"""Log-aware candidate encoding and the flat-prior parity of HapBO with vanilla BO."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bayesian_optimization import BayesianOptimization  # noqa: E402
from experiments.worker import _encode_mixed_candidates, _log_scaled_names  # noqa: E402


class LogEncodingTests(unittest.TestCase):
    def test_log_spaced_column_is_evenly_spread(self) -> None:
        values = np.logspace(-5, 0, 6)
        native = [{"alpha": float(v), "depth": d} for v in values for d in (1, 2, 3)]
        enc = _encode_mixed_candidates(native, ["alpha", "depth"], ("alpha",))
        col = sorted(set(np.round(enc[:, 0], 12)))
        np.testing.assert_allclose(np.diff(col), np.full(5, 0.2), atol=1e-12)
        lin = _encode_mixed_candidates(native, ["alpha", "depth"], ())
        self.assertLess(sorted(set(np.round(lin[:, 0], 12)))[3], 0.02)  # linear: crowded near 0
        np.testing.assert_allclose(sorted(set(np.round(enc[:, 1], 12))), [0.0, 0.5, 1.0])

    def test_only_hpobench_grids_have_log_scaled_columns(self) -> None:
        self.assertEqual(_log_scaled_names(object(), "neuromosaic-dataset1-rat-flexor"), ())


class FlatPriorParityTests(unittest.TestCase):
    def test_flat_prior_hapbo_matches_vanilla(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.random((60, 3))

        def f(x):
            return -float(np.sum((x - 0.3) ** 2))

        def drive(**kw):
            opt = BayesianOptimization(X, length_scale=0.2, n_initial=1, random_state=5, **kw)
            while opt.n_observations < 12:
                index, x = opt.ask()
                opt.tell(index, f(x))
            return list(opt.evaluated_indices)

        vanilla = drive()
        flat = drive(alpha=0.5, prior=np.ones(len(X)))
        zero = drive(alpha=0.0, prior=rng.random(len(X)) + 0.1)
        self.assertEqual(flat, vanilla)
        self.assertEqual(zero, vanilla)


if __name__ == "__main__":
    unittest.main()
