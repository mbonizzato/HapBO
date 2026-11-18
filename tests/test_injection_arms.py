"""Injection-point arms at the optimizer level (the config the worker builds)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bayesian_optimization import BayesianOptimization, pi_bo_initial_design  # noqa: E402


CANDIDATES = np.linspace(0.0, 1.0, 24).reshape(-1, 1)
PI = np.exp(-((CANDIDATES[:, 0] - 0.72) ** 2) / (2 * 0.05**2))
PI = PI / PI.max()


def _objective(x: np.ndarray) -> float:
    return -float((x[0] - 0.72) ** 2)  # maximize; optimum near 0.72


def _run(**kwargs):
    opt = BayesianOptimization(CANDIDATES, random_state=7, n_initial=2, **kwargs)
    return opt.run(_objective, max_evaluations=8)


def _prior_mean_callable(encoded: np.ndarray, pi: np.ndarray, beta: float):
    p = np.asarray(pi, dtype=float)
    sd = float(p.std())
    m_z = np.zeros_like(p) if sd <= 1e-12 else beta * (p - float(p.mean())) / sd
    table = {encoded[i].tobytes(): float(m_z[i]) for i in range(len(encoded))}

    def mean(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([table[X[i].tobytes()] for i in range(len(X))])

    return mean


def test_all_shared_init_arms_complete_share_init_and_diverge() -> None:
    shared = [3, 19]  # the same initial design for the alpha-pi, prior-to-mean and vanilla arms
    arms = {}
    arms["ours"] = _run(
        prior=PI, alpha=0.6, prior_shaping="alpha-pi", initial_indices=shared,
    )
    arms["prior-to-mean"] = _run(
        prior_mean=_prior_mean_callable(CANDIDATES, PI, beta=0.6), normalize_y=True,
        prior_shaping="none", initial_indices=shared,
    )
    arms["vanilla"] = _run(initial_indices=shared)

    for result in arms.values():
        assert len(set(result.evaluated_indices)) == 8  # complete run
        assert list(result.evaluated_indices[:2]) == shared  # identical initialisation

    traces = {name: tuple(int(i) for i in r.evaluated_indices) for name, r in arms.items()}
    assert len(set(traces.values())) > 1  # the prior injection changes the trajectory


def test_pi_bo_arm_starts_with_its_initial_design() -> None:
    result = _run(prior=PI, alpha=0.0, prior_shaping="pi-bo", pi_bo_beta=0.6)
    assert len(set(result.evaluated_indices)) == 8
    assert tuple(result.evaluated_indices[:2]) == pi_bo_initial_design(PI, 2, 7)


def test_prior_warm_start_initialisation_uses_top_prior_points() -> None:
    order = np.argsort(-PI, kind="stable")
    top2 = [int(order[0]), int(order[1])]
    result = _run(prior=None, prior_shaping="none", initial_indices=top2)
    assert list(result.evaluated_indices[:2]) == top2
