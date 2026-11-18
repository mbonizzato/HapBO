"""Subprocess baseline bridge protocol, exercised with the mock worker tests/mock_worker.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.baselines.base import config_key  # noqa: E402
from src.baselines.bridge import SubprocessBridgeOptimizer  # noqa: E402
from src.baselines.runner import run_ask_tell_loop  # noqa: E402

_MOCK = Path(__file__).resolve().parent / "mock_worker.py"


class MockBridge(SubprocessBridgeOptimizer):
    name = "mock"
    worker_command = [sys.executable, str(_MOCK)]


def _candidates(n: int) -> list[dict]:
    return [{"x": i} for i in range(n)]


def test_bridge_runs_a_finite_budget_with_unique_evaluations() -> None:
    calls: list[int] = []

    def objective(config):
        calls.append(config["x"])
        return -abs(config["x"] - 13)  # peak at x=13

    opt = MockBridge(
        search_space={"candidates": _candidates(20)},
        objective_meta={"maximize": True},
        seed=0,
        budget=10,
    )
    try:
        history = run_ask_tell_loop(opt, objective, budget=10, n_init=2)
    finally:
        opt.close()

    assert len(history) == 10  # exactly budget charged queries (repeats included)
    keys = [config_key(config) for config, _ in history]
    repeats = [row["repeat"] for row in opt.trace]
    assert sum(repeats) == len(keys) - len(set(keys))  # re-proposals are charged, flagged
    assert len(calls) == len(set(keys))  # the objective ran once per unique candidate


def test_bridge_reaches_full_enumeration_budget() -> None:
    def objective(config):
        return float(config["x"])

    opt = MockBridge(
        search_space={"candidates": _candidates(8)},
        objective_meta={"maximize": True},
        seed=1,
        budget=8,
    )
    try:
        history = run_ask_tell_loop(opt, objective, budget=8, n_init=1)
    finally:
        opt.close()
    assert len(history) == 8
    assert {config["x"] for config, _ in history} <= set(range(8))


def test_bridge_reports_worker_failure() -> None:
    class BrokenBridge(SubprocessBridgeOptimizer):
        name = "broken"
        worker_command = [sys.executable, "-c", "import sys; sys.exit(3)"]

    opt = BrokenBridge(
        search_space={"candidates": _candidates(5)},
        objective_meta={"maximize": True},
        seed=0,
        budget=3,
    )
    with pytest.raises(Exception):
        run_ask_tell_loop(opt, lambda c: 0.0, budget=3, n_init=1)
    opt.close()
