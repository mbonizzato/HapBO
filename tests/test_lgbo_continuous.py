"""Continuous LGBO runner (scripts/lgbo_continuous.py) against the finite worker, with a mocked LLM."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "scripts")]

import lgbo_continuous as L  # noqa: E402

ST = L.SETTINGS


def mock_chat(dim, lb, ub, conf):
    pts = [[0.25] * dim, [0.75] * dim]

    def chat(messages, **kw):
        text = messages[-1]["content"]
        if text.startswith("Choose exactly"):
            return json.dumps({"points": pts[: int(text.split()[2])]})
        return json.dumps({"type": "region", "lb": list(lb), "ub": list(ub), "confidence": conf})

    return chat


class ContinuousLiftTests(unittest.TestCase):
    def test_lift_and_argmax_match_the_finite_worker(self) -> None:
        rng = np.random.default_rng(0)
        d = 6
        X = rng.random((30, d))
        y = -np.sum((X - 0.3) ** 2, axis=1) + 0.01 * rng.standard_normal(30)
        gp = L.base._GP(length_scale=0.2, noise_std=1e-3).fit(X, y, optimize=True)
        U = rng.random((300, d))
        for kind, lo_, hi_, c, gseed in (("region", .4, .6, .8, 1031), ("region", .1, .9, .3, 1032),
                                         ("point", .45, .55, .9, 1033), ("region", 0., .2, 1.0, 1034)):
            lb, ub = np.full(d, lo_), np.full(d, hi_)
            g = (kind, lb, ub, c)
            G, a = L.common._grid_and_weights(kind, lb, ub, d, ST["n_grid"], gseed, ST["decay"])
            ref, _, _ = L.base._regional_shift_z(gp, U, G, a, c)
            lift = L._PosteriorLift(gp, g, d, ST["n_grid"], gseed, ST["decay"])
            with self.subTest(kind=kind, confidence=c):
                rel = float(np.max(np.abs(lift.shift(U) - ref)) / max(np.max(np.abs(ref)), 1e-12))
                self.assertLess(rel, 1e-8)
                cand = rng.random((500, d))
                idx_w = L.base._lgbo_step(gp, cand, np.zeros(500, bool), g, ST["n_grid"], gseed, ST["decay"],
                                          float(np.max(y)))[0]
                idx_m = int(np.argmax(L._log_ei(gp, lift, cand, float(np.max(y)))))
                self.assertEqual(idx_w, idx_m)


class EndToEndTests(unittest.TestCase):
    def test_cli_output_and_resume(self) -> None:
        d, budget = 6, 6
        chat = mock_chat(d, [.2] * d, [.4] * d, .8)
        original = L.run_one

        def run_one(family, seed, budget, rho="90", **kw):
            return original(family, seed, budget, rho, chat=chat)

        L.run_one = run_one
        argv = sys.argv
        try:
            with tempfile.TemporaryDirectory() as tmp:
                for seeds in ("1", "2"):
                    sys.argv = ["x", "--family", "hartmann6", "--rho", "90", "--seeds", seeds, "--budget", str(budget),
                                "--out-root", tmp]
                    L.main()
                js = json.loads((Path(tmp) / "hartmann6-rho90-lgbo.json").read_text(encoding="utf-8"))
        finally:
            sys.argv = argv
            L.run_one = original
        self.assertEqual(len(js["trajectories"]), 2)
        for t in js["trajectories"]:
            self.assertEqual(len(t), budget)
            self.assertTrue(all(b <= a + 1e-12 for a, b in zip(t, t[1:])))
            self.assertTrue(all(v >= 0 for v in t))
        self.assertEqual(js["world"], {"truth": "well"})
        self.assertAlmostEqual(js["f_opt"], -3.2154, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
