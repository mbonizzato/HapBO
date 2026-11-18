"""Kriging-Believer batch selection on the discrete BO loop.

Guards two contracts: q_batch=1 equals the sequential ask/tell loop,
and q_batch>1 proposes distinct, diversified
candidates against a frozen surrogate without mutating observation state.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bayesian_optimization import BayesianOptimization  # noqa: E402


def _objective(x: np.ndarray) -> float:
    return -float((x[0] - 0.7) ** 2)


def _drive_sequential(opt: BayesianOptimization, budget: int) -> list[int]:
    while opt.n_observations < budget:
        index, x = opt.ask()
        opt.tell(index, _objective(x))
    return list(opt.evaluated_indices)


def _drive_batch(opt: BayesianOptimization, budget: int, q: int) -> list[int]:
    while opt.n_observations < budget:
        take = min(q, budget - opt.n_observations)
        for index in opt.ask_batch(take):
            opt.tell(index, _objective(opt.candidates[index]))
    return list(opt.evaluated_indices)


def _make(prior: np.ndarray | None = None, alpha: float = 0.0) -> BayesianOptimization:
    candidates = np.linspace(0.0, 1.0, 12).reshape(-1, 1)
    return BayesianOptimization(
        candidates,
        prior=prior,
        alpha=alpha,
        n_initial=1,
        length_scale=0.2,
        noise_variance=0.001 ** 2,
        normalize_y=True,
        random_state=1001,
    )


class BatchSelectionTests(unittest.TestCase):
    def test_q1_is_identical_to_the_sequential_loop(self) -> None:
        """ask_batch(1) selects and records exactly the sequential trajectory."""
        for prior, alpha in ((None, 0.0), (np.linspace(0.0, 1.0, 12), 0.6)):
            seq = _make(prior, alpha)
            bat = _make(None if prior is None else prior.copy(), alpha)
            seq_idx = _drive_sequential(seq, budget=6)
            bat_idx = _drive_batch(bat, budget=6, q=1)
            self.assertEqual(seq_idx, bat_idx)
            np.testing.assert_allclose(seq.observed_values, bat.observed_values)

    def test_ask_batch_does_not_mutate_observation_state(self) -> None:
        opt = _make()
        # advance past the initial design so a surrogate exists
        for _ in range(3):
            index, x = opt.ask()
            opt.tell(index, _objective(x))
        n_before = opt.n_observations
        mask_before = opt.queried_mask.copy()
        batch = opt.ask_batch(3)
        self.assertEqual(opt.n_observations, n_before)
        np.testing.assert_array_equal(opt.queried_mask, mask_before)
        self.assertEqual(len(batch), 3)
        self.assertEqual(len(set(batch)), 3, "batch picks must be distinct")
        self.assertFalse(
            any(mask_before[i] for i in batch), "batch must avoid queried candidates"
        )

    def test_kriging_believer_diversifies_within_a_batch(self) -> None:
        """With a frozen surrogate, the fantasy makes the 2nd/3rd picks differ from
        the greedy argmax that would repeat without belief injection."""
        opt = _make()
        for _ in range(3):
            index, x = opt.ask()
            opt.tell(index, _objective(x))
        # greedy argmax repeated with NO fantasy would return the same top index thrice
        greedy = opt.ask()[0]
        batch = opt.ask_batch(3)
        self.assertEqual(batch[0], greedy)
        self.assertNotEqual(batch[1], batch[0])
        self.assertEqual(len(set(batch)), 3)

    def test_batch_loop_yields_a_full_distinct_trajectory(self) -> None:
        idx = _drive_batch(_make(), budget=9, q=3)
        self.assertEqual(len(idx), 9)
        self.assertEqual(len(set(idx)), 9)

    def test_query_one_coincides_between_sequential_and_batch(self) -> None:
        """Shared seed + the same seeded initial design => same first query as the sequential arm,
        the property the figure's 'no batch vs batch' comparison relies on."""
        seq = _make()
        bat = _make()
        self.assertEqual(_drive_sequential(seq, 6)[0], _drive_batch(bat, 6, 3)[0])


if __name__ == "__main__":
    unittest.main()
