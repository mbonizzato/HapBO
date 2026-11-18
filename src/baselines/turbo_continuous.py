"""TuRBO (uber-research/TuRBO, pinned de0db39) as a continuous baseline.

TuRBO is a trust-region BO method for continuous boxes; it is not run on the finite grids. It
is an arm of the synthetic-panel runner (scripts/run_synthetic_arms.py) rather than of the
finite subprocess bridge. Runs Turbo1 natively on the analytic test functions; regret is
measured against the true optimum.

Requires torch + gpytorch, which are not in the `hapbo` environment; run it under an
interpreter that has them (tested with torch 2.9 / gpytorch 1.14) or an isolated `turbo` env.

Use of the pinned upstream turbo/turbo_1.py:
  * every kwarg name/default matches Turbo1.__init__; `f` receives a 1-D real-space array and
    returns a scalar to be minimized; `turbo.fX` accumulates every evaluation across restarts,
    so its running minimum minus f* is the exact simple regret.
  * TuRBO's dominant randomness (Thompson sampling at turbo_1.py:216, GP Adam training) uses
    torch's global RNG, so both numpy and torch are seeded.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TURBO_ROOT = _REPO_ROOT / "external" / "turbo"


def run_turbo(objective: Callable[[np.ndarray], float], bounds: np.ndarray, *,
              budget: int, n_init: int, seed: int, optimum_value: float) -> np.ndarray:
    """Run TuRBO-1 (minimization) on a continuous box; return the best-so-far simple regret.

    objective: native-coord callable, minimized. bounds: (d,2). optimum_value: true min f*.
    Returns the per-query best-so-far regret array of length ``budget`` (fX accumulates every
    evaluation in query order across trust-region restarts).
    """
    if str(_TURBO_ROOT) not in sys.path:
        sys.path.insert(0, str(_TURBO_ROOT))
    import torch  # noqa: E402  (upstream dep; imported lazily)
    from turbo import Turbo1  # noqa: E402  (upstream; imported lazily)

    bounds = np.asarray(bounds, dtype=float)
    lb, ub = bounds[:, 0].copy(), bounds[:, 1].copy()
    # Turbo1 asserts max_evals/n_init are python ints and max_evals > n_init.
    budget = int(budget)
    n_init = int(max(2, min(n_init, budget - 1)))
    # Seed both RNGs: numpy (LHS/perturbation/Sobol seed) and torch (TS draw + GP training).
    np.random.seed(seed)
    torch.manual_seed(seed)

    class _F:  # TuRBO calls f(x)->scalar and minimizes
        def __init__(self):
            self.dim = bounds.shape[0]
            self.lb = lb
            self.ub = ub

        def __call__(self, x):
            return float(objective(np.asarray(x, dtype=float).reshape(-1)))

    turbo = Turbo1(f=_F(), lb=lb, ub=ub, n_init=n_init, max_evals=budget,
                   batch_size=1, verbose=False, use_ard=True, max_cholesky_size=2000,
                   n_training_steps=50, min_cuda=1024, device="cpu", dtype="float64")
    turbo.optimize()
    # fX accumulates every eval across all trust-region restarts. Trim to the stated budget
    # (a final restart can overshoot by <n_init) so accounting is exactly `budget` evals.
    fX = np.asarray(turbo.fX, dtype=float).reshape(-1)[:budget]
    traj = np.maximum(np.minimum.accumulate(fX) - float(optimum_value), 0.0)
    return traj


__all__ = ["run_turbo"]
