"""The SOLAR runner's optimizer override and initial design.

These tests pin the override and check, on a toy pool, that the alpha-pi arm shares
vanilla's seeded initial design. No SOLAR binary is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.worker import SOLAR_OPTIMIZER_EXTRA, _build_optimizer  # noqa: E402

ALPHA_PI = {"strategy": "alpha-pi", "acquisition": "expected-improvement"}
VANILLA = {"strategy": "vanilla-bo", "acquisition": "expected-improvement"}
RUN = {"seed": 2003, "budget": 12, "optimizer": {"n_initial": 3, "length_scale": 1.0, "noise_std": 0.05}}


def test_hidden_failure_model_is_the_only_override() -> None:
    # The same override applies to prior and prior-free arms alike.
    assert SOLAR_OPTIMIZER_EXTRA == {"model_hidden_failures": True}


def _pool(seed: int = 0):
    rng = np.random.default_rng(seed)
    encoded = rng.uniform(0.0, 1.0, size=(60, 4))
    prior = np.exp(-np.sum((encoded - 0.7) ** 2, axis=1) / 0.05)
    return encoded, prior


def _first_queries(optimizer, n: int, encoded: np.ndarray) -> list[int]:
    out = []
    for _ in range(n):
        index, _ = optimizer.ask()
        out.append(int(index))
        optimizer.tell(index, float(np.sum(encoded[index])), np.array([-1.0, -1.0]))
    return out


def test_alpha_pi_arm_shares_vanilla_initial_design() -> None:
    encoded, prior = _pool()
    n_init = RUN["optimizer"]["n_initial"]

    def build(method, with_prior):
        return _build_optimizer(
            encoded, RUN, method,
            prior=prior if with_prior else None,
            alpha=0.5 if with_prior else 0.0,
            maximize=False, n_constraints=2,
            extra=dict(SOLAR_OPTIMIZER_EXTRA),
        )

    shaped, vanilla = build(ALPHA_PI, True), build(VANILLA, False)
    assert shaped.model_hidden_failures and vanilla.model_hidden_failures
    q_shaped = _first_queries(shaped, n_init, encoded)
    q_vanilla = _first_queries(vanilla, n_init, encoded)
    assert q_shaped == q_vanilla, (q_shaped, q_vanilla)
