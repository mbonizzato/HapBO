"""Pure finite-domain behavior used by all isolated upstream workers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKERS = PROJECT_ROOT / "src" / "baselines" / "workers"
sys.path.insert(0, str(WORKERS))

from finite import nearest_index, normalize_rows, seeded_initial_indices  # noqa: E402


def test_normalize_rows_handles_mixed_scales_and_constant_columns() -> None:
    rows = normalize_rows([[10.0, -1.0, 8.0], [20.0, 1.0, 8.0]])
    np.testing.assert_allclose(rows, [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])


def test_nearest_index_is_deterministic_on_ties() -> None:
    rows = np.asarray([[0.0], [1.0]])
    assert nearest_index(rows, [0.5]) == 0
    assert nearest_index(rows, [0.9]) == 1


def test_seeded_initialization_is_unique_and_reproducible() -> None:
    first = seeded_initial_indices(20, 5, 42)
    second = seeded_initial_indices(20, 5, 42)
    assert first == second
    assert len(set(first)) == 5


def test_finite_helpers_reject_invalid_shapes() -> None:
    with pytest.raises(ValueError):
        normalize_rows([])
    with pytest.raises(ValueError):
        nearest_index(np.zeros((2, 3)), [0.0, 1.0])
