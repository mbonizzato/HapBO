"""SOLAR reader (paper/solar_data.py): best-feasible curve; a run shorter than the queries shown is not padded."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper"))

import solar_data as SD  # noqa: E402


def test_best_feasible_is_the_running_maximum_over_feasible_records() -> None:
    recs = [[0, -1.0], [1, -3.0], [0, -9.0], [1, -2.0]]
    assert SD.best_feasible(recs, 4).tolist() == [0.0, 3.0, 3.0, 3.0]
    assert SD.best_feasible(recs, 3).tolist() == [0.0, 3.0, 3.0]


def test_best_feasible_does_not_pad_a_short_run() -> None:
    with pytest.raises(ValueError, match="fewer than the 5 queries shown"):
        SD.best_feasible([[1, -3.0]] * 4, 5)


def test_every_committed_solar_run_covers_the_queries_shown() -> None:
    runs = SD.load()
    assert set(runs) >= {"HapBO", "vanilla BO", "random", "ColaBO", "LGBO", "LLAMBO", "BORA"}
    assert all(len(r) >= 20 for rs in runs.values() for r in rs)
