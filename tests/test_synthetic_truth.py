"""Hidden objectives of the synthetic panel (scripts/synthetic_truth.py)."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "scripts")]

import synthetic_truth as ST  # noqa: E402

CANONICAL_XOPT = {
    "hartmann6": np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]),
    "styblinski_tang7": np.full(7, -2.903534),
    "levy10": np.ones(10),
    "ackley15": np.zeros(15),
}
CARD_PEARSON = {"rho90": 0.9, "well50": 0.5, "well10": 0.1}


def _sample(family: str, n: int = 20000) -> np.ndarray:
    d, lo, hi = ST.BOUNDS[family]
    return lo + (hi - lo) * np.random.default_rng(11).random((n, d))


class ShiftWorldTests(unittest.TestCase):
    def test_shifted_optimum_inside_box_and_off_centre(self) -> None:
        for fam in ST.FAMILIES:
            d, lo, hi = ST.BOUNDS[fam]
            x = CANONICAL_XOPT[fam] + ST.shift_delta(fam)
            width, mid = hi - lo, 0.5 * (lo + hi)
            with self.subTest(family=fam):
                self.assertEqual(len(x), d)
                self.assertTrue(np.all(x >= lo + 0.1 * width - 1e-12) and np.all(x <= hi - 0.1 * width + 1e-12))
                self.assertTrue(np.all(np.abs(x - mid) >= 0.08 * width - 1e-12))


class WellWorldTests(unittest.TestCase):
    def test_frozen_optimum_is_the_minimum(self) -> None:
        for fam in ST.FAMILIES:
            f_opt, x_opt = ST.well_optimum(fam)
            _, lo, hi = ST.BOUNDS[fam]
            T = ST.well_truth(fam)
            with self.subTest(family=fam):
                self.assertTrue(np.all(x_opt >= lo) and np.all(x_opt <= hi))
                self.assertAlmostEqual(float(T(x_opt)[0]), f_opt, places=6)
                self.assertGreaterEqual(float(T(_sample(fam)).min()), f_opt - 1e-6)

    def test_card_fidelity(self) -> None:
        for fam in ST.FAMILIES:
            X = _sample(fam)
            t = ST.well_truth(fam)(X)
            for card, rho in CARD_PEARSON.items():
                m = ST.card_model(fam, card)(X)
                with self.subTest(family=fam, card=card):
                    self.assertAlmostEqual(float(np.corrcoef(m, t)[0, 1]), rho, delta=0.06)

    def test_runner_tables_agree(self) -> None:
        old = os.environ.get("SYNTH_TRUTH")
        os.environ["SYNTH_TRUTH"] = "well"
        try:
            import estimate_lib
            import lgbo_continuous
            import run_synthetic_arms
        except ImportError as exc:  # optional runner dependency missing
            self.skipTest(str(exc))
        finally:
            if old is None:
                os.environ.pop("SYNTH_TRUTH", None)
            else:
                os.environ["SYNTH_TRUTH"] = old
        tables = {"estimate_lib": estimate_lib.FAMS, "run_synthetic_arms": run_synthetic_arms.FAMILIES,
                  "lgbo_continuous": lgbo_continuous._FAMILIES}
        for fam in ST.FAMILIES:
            X = _sample(fam, 5)
            ref = ST.well_truth(fam)(X)
            f_opt, _ = ST.well_optimum(fam)
            for name, table in tables.items():
                with self.subTest(family=fam, table=name):
                    vals = np.array([float(table[fam][3](x)) for x in X])
                    np.testing.assert_allclose(vals, ref, rtol=0, atol=1e-12)
                    self.assertEqual(float(table[fam][4]), f_opt)


if __name__ == "__main__":
    unittest.main()
