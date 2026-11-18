"""Noisy single-trial polling for the cortical (dataset1) neuro benchmark.

Deterministic default is preserved (returns sorted_respMean); with a noise_seed
each query returns a random valid trial, means/regret-basis are unchanged.
Requires the rat .mat data (not shipped); skipped if absent.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets.neuromosaic_dataset1_rat import (
    NEUROMOSAIC_DATASET1_RAT_DIRECTORY,
    load_neuromosaic_dataset1_rat_problem,
)

_DATA_PRESENT = (NEUROMOSAIC_DATASET1_RAT_DIRECTORY / "rat1_M1_190716.mat").is_file()


@unittest.skipUnless(_DATA_PRESENT, "NeuroMOSAIC dataset1 rat .mat not present")
class NoisyPollingTests(unittest.TestCase):
    MUSCLE = "LeftKneeFlexor"

    def _best_candidate(self, problem):
        return problem.native_candidates[int(np.argmax(problem.ground_truth))]

    def test_deterministic_default_returns_mean(self) -> None:
        p = load_neuromosaic_dataset1_rat_problem(subject="rat1", muscle=self.MUSCLE)
        idx = int(np.argmax(p.ground_truth))
        self.assertAlmostEqual(p.objective(p.native_candidates[idx]), float(p.ground_truth[idx]))
        # repeated calls are identical (deterministic)
        c = p.native_candidates[idx]
        self.assertEqual(p.objective(c), p.objective(c))

    def test_noisy_varies_but_preserves_means(self) -> None:
        det = load_neuromosaic_dataset1_rat_problem(subject="rat1", muscle=self.MUSCLE)
        noi = load_neuromosaic_dataset1_rat_problem(subject="rat1", muscle=self.MUSCLE, noise_seed=1001)
        # ground-truth means (the regret basis) are untouched
        np.testing.assert_array_equal(det.ground_truth, noi.ground_truth)
        c = self._best_candidate(noi)
        samples = [noi.objective(c) for _ in range(12)]
        self.assertGreater(len(set(samples)), 1)  # noisy: not constant

    def test_noise_is_seed_reproducible(self) -> None:
        c = {"channel_x": 1, "channel_y": 1}
        a = [load_neuromosaic_dataset1_rat_problem(subject="rat1", muscle=self.MUSCLE, noise_seed=7).objective(c)
             for _ in range(4)]
        b = [load_neuromosaic_dataset1_rat_problem(subject="rat1", muscle=self.MUSCLE, noise_seed=7).objective(c)
             for _ in range(4)]
        self.assertEqual(a, b)

    def test_noisy_samples_are_drawn_from_valid_trials(self) -> None:
        import scipy.io as sio
        noi = load_neuromosaic_dataset1_rat_problem(subject="rat1", muscle=self.MUSCLE, noise_seed=3)
        path = noi.metadata["subject_path"]
        m = sio.loadmat(path)
        rec = m[[k for k in m if not k.startswith("__")][0]][0, 0]
        muscles = [str(np.asarray(x).reshape(-1)[0]) for x in np.asarray(rec["emgs"], dtype=object).reshape(-1)]
        mi = muscles.index(self.MUSCLE)
        idx = int(np.argmax(noi.ground_truth))
        resp = np.asarray(rec["sorted_resp"][idx, mi], dtype=float).reshape(-1)
        valid = set(np.round(resp[np.asarray(rec["sorted_isvalid"][idx, mi]).reshape(-1) != 0], 6))
        c = noi.native_candidates[idx]
        for _ in range(20):
            self.assertIn(round(noi.objective(c), 6), valid)


if __name__ == "__main__":
    unittest.main()
