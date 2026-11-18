"""Initial designs in the BO loop: an explicit shared design, and the seeded random design."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bayesian_optimization import BayesianOptimization  # noqa: E402


CANDIDATES = np.linspace(0.0, 1.0, 10).reshape(-1, 1)
PRIOR = np.array([0.0, 0.1, 0.2, 0.9, 0.3, 1.0, 0.4, 0.2, 0.1, 0.0])


def test_shared_initial_indices_are_honoured_regardless_of_prior() -> None:
    shared = [7, 2, 5]
    first_three = []
    opt = BayesianOptimization(
        CANDIDATES,
        prior=PRIOR,
        alpha=0.6,
        initial_indices=shared,
        random_state=0,
    )
    for _ in range(3):
        index, _ = opt.ask()
        opt.tell(index, -float((opt.candidates[index][0] - 0.7) ** 2))
        first_three.append(index)
    assert first_three == shared


def test_two_arms_share_identical_initialisation() -> None:
    shared = [1, 8, 4]
    inits = {}
    for shaping, alpha in (("alpha-pi", 0.6), ("none", 0.0)):
        opt = BayesianOptimization(
            CANDIDATES,
            prior=PRIOR,
            alpha=alpha,
            prior_shaping=shaping,
            initial_indices=shared,
            random_state=123,
        )
        seen = []
        for _ in range(len(shared)):
            index, _ = opt.ask()
            opt.tell(index, -float((opt.candidates[index][0] - 0.7) ** 2))
            seen.append(index)
        inits[shaping] = seen
    assert inits["alpha-pi"] == inits["none"] == shared


def test_prior_arm_shares_the_random_initial_design_with_vanilla() -> None:
    # With the same seed the prior arm's initial design is the no-prior arm's seeded random design.
    for seed in range(8):
        shaped = BayesianOptimization(CANDIDATES, prior=PRIOR, alpha=0.6, n_initial=3, random_state=seed)
        vanilla = BayesianOptimization(CANDIDATES, n_initial=3, random_state=seed)
        for _ in range(3):
            index, _ = shaped.ask()
            assert vanilla.ask()[0] == index
            shaped.tell(index, 0.0)
            vanilla.tell(index, 0.0)


def test_random_first_query_is_not_tied_to_the_prior_max() -> None:
    # The first pick is the seeded random draw, so across seeds it is not always the prior argmax.
    picks = set()
    for seed in range(8):
        opt = BayesianOptimization(CANDIDATES, prior=PRIOR, alpha=0.6, random_state=seed)
        index, _ = opt.ask()
        picks.add(index)
    assert picks != {int(np.argmax(PRIOR))}
