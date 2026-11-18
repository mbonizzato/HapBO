"""piBO (Hvarfner et al. 2022): initial design (prior mode, then points sampled from the prior) and beta = N / 10."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from bayesian_optimization import BayesianOptimization, pi_bo_beta, pi_bo_initial_design, pi_bo_prior  # noqa: E402
from experiments import worker  # noqa: E402
from experiments.catalog import load_catalog  # noqa: E402

CATALOG = load_catalog()
PRIOR = (
    PROJECT_ROOT / "experiments" / "priors" / "hpobench-svm-blood-transfusion" / "blood-transfusion"
    / "llm-card" / "prior.json"
)


def _svm_problem() -> SimpleNamespace:
    """The HPOBench SVM grid (21 x 21 powers of two) with a synthetic objective."""

    values = [float(2.0**k) for k in range(-10, 11)]
    native = tuple({"C": c, "gamma": g} for c in values for g in values)
    truth = np.asarray(
        [0.9 - 0.001 * ((np.log2(c["C"]) - 3) ** 2 + (np.log2(c["gamma"]) + 2) ** 2) for c in native]
    )
    row = {(c["C"], c["gamma"]): i for i, c in enumerate(native)}
    return SimpleNamespace(
        native_candidates=native, parameter_names=("C", "gamma"), model="svm",
        task_name="blood_transfusion", n_candidates=len(native), maximize=True,
        ground_truth=truth, optimum_value=float(truth.max()), dataset_id="hpobench",
        target_name="accuracy", objective=lambda c: float(truth[row[(c["C"], c["gamma"])]]),
    )


def _run(monkeypatch, tmp_path, method, seed=1001, n_initial=1, budget=6):
    import main as main_module

    monkeypatch.setattr(worker, "_load_offline_problem", lambda *a, **k: _svm_problem())
    monkeypatch.setattr(main_module, "save_performance_plot", lambda *a, **k: None)
    run = {
        "run_id": "t", "method_id": method["method_id"], "seed": seed, "budget": budget, "parameters": {},
        "optimizer": {"n_initial": n_initial, "length_scale": 0.2, "noise_std": 0.001, "improvement_xi": 0.0},
        "prior_path": None if not method.get("requires_prior") else str(PRIOR),
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    return worker._run_offline(tmp_path, run, {"adapter": "hpobench"}, method)


def _pi_values():
    _, _, pi_values, *_ = worker._offline_prior_context(
        _svm_problem(), "hpobench", PRIOR
    )
    return pi_values


# --------------------------------------------------------------------------- the design
def test_first_point_is_the_prior_mode_for_every_seed() -> None:
    pi = _pi_values()
    mode = int(np.argmax(pi))
    assert {pi_bo_initial_design(pi, 1, seed) for seed in range(1001, 1041)} == {(mode,)}
    assert {pi_bo_initial_design(pi, 3, seed)[0] for seed in range(1001, 1041)} == {mode}


def test_mode_ties_go_to_the_lowest_index() -> None:
    prior = np.array([0.1, 0.7, 0.3, 0.7, 0.7])
    assert pi_bo_initial_design(prior, 1, 5) == (1,)
    assert all(pi_bo_initial_design(prior, 2, s)[0] == 1 for s in range(20))


def test_the_other_points_are_deterministic_per_seed_and_depend_on_it() -> None:
    pi = _pi_values()
    for seed in (1001, 1002, 1010):
        assert pi_bo_initial_design(pi, 3, seed) == pi_bo_initial_design(pi, 3, seed)
    seconds = {pi_bo_initial_design(pi, 2, seed)[1] for seed in range(1001, 1041)}
    assert len(seconds) > 1 and int(np.argmax(pi)) not in seconds


def test_the_other_points_follow_the_pi_bo_prior_without_the_mode() -> None:
    from scipy.stats import chisquare

    prior = np.array([0.0, -1.0, 0.5, 1.0, 2.0, 4.5])  # mode: index 5
    w = pi_bo_prior(prior)
    draws = np.array([pi_bo_initial_design(prior, 2, seed) for seed in range(20_000)])
    assert np.all(draws[:, 0] == 5)
    counts = np.bincount(draws[:, 1], minlength=prior.size)
    assert counts[0] == 0 and counts[1] == 0 and counts[5] == 0  # weight 1e-12 / the mode
    expected = w[2:5] / w[2:5].sum() * len(draws)
    assert chisquare(counts[2:5], expected).pvalue > 1e-3


def test_a_constant_prior_has_no_mode_and_is_sampled() -> None:
    for prior in (np.full(6, 0.4), np.zeros(6)):
        firsts = {pi_bo_initial_design(prior, 1, seed)[0] for seed in range(40)}
        assert len(firsts) > 1


def test_design_indices_are_distinct_and_validated() -> None:
    prior = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    for seed in range(50):
        design = pi_bo_initial_design(prior, 6, seed)
        assert design[0] == 5 and len(set(design)) == 6
    with pytest.raises(ValueError):
        pi_bo_initial_design(prior, 7, 1)
    with pytest.raises(ValueError):
        pi_bo_initial_design(prior, 0, 1)


# --------------------------------------------------------------------------- the optimizer
@pytest.mark.parametrize("n_initial", [1, 3])
def test_optimizer_starts_with_the_prior_design(n_initial) -> None:
    candidates = np.linspace(0.0, 1.0, 31).reshape(-1, 1)
    prior = np.exp(-((candidates[:, 0] - 0.8) ** 2) / 0.01)
    opt = BayesianOptimization(
        candidates, prior=prior, prior_shaping="pi-bo", pi_bo_beta=(6 - n_initial) / 10, n_initial=n_initial,
        random_state=7,
    )
    result = opt.run(lambda x: -float((x[0] - 0.3) ** 2), max_evaluations=6)
    assert tuple(result.evaluated_indices[:n_initial]) == pi_bo_initial_design(prior, n_initial, 7)
    assert int(result.evaluated_indices[0]) == int(np.argmax(prior))


@pytest.mark.parametrize("prior", [np.linspace(0.0, 1.0, 8), None])
def test_optimizer_rejects_an_explicit_initial_design(prior) -> None:
    candidates = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
    with pytest.raises(ValueError, match="initial_indices"):
        BayesianOptimization(candidates, prior=prior, prior_shaping="pi-bo", pi_bo_beta=0.5, initial_indices=[0],
                             random_state=0)


# --------------------------------------------------------------------------- the worker arm
def test_catalog_method() -> None:
    method = CATALOG.methods["pi-bo-ei"]
    assert method["strategy"] == "pi-bo" and "pi_bo_beta" not in method
    assert not any(m != "pi-bo-ei" and e.get("strategy") == "pi-bo" for m, e in CATALOG.methods.items())


@pytest.mark.parametrize("budget, n_initial, beta", [(16, 1, 1.5), (12, 1, 1.1), (6, 3, 0.3), (1, 1, 0.0)])
def test_beta_is_a_tenth_of_the_bo_iterations(budget, n_initial, beta) -> None:
    run = {"budget": budget, "optimizer": {"n_initial": n_initial}}
    assert worker._pi_bo_beta(CATALOG.methods["pi-bo-ei"], run) == beta == pi_bo_beta(budget, n_initial)


def test_beta_is_not_a_method_field() -> None:
    run = {"budget": 16, "optimizer": {"n_initial": 1}}
    for value in (1.0, 2.0, "auto"):
        with pytest.raises(ValueError):
            worker._pi_bo_beta(dict(CATALOG.methods["pi-bo-ei"], pi_bo_beta=value), run)


def test_optimizer_requires_beta() -> None:
    candidates = np.linspace(0.0, 1.0, 5).reshape(-1, 1)
    for beta in (None, -0.5):
        with pytest.raises(ValueError):
            BayesianOptimization(candidates, prior=np.linspace(0.0, 1.0, 5), prior_shaping="pi-bo", pi_bo_beta=beta)


@pytest.mark.parametrize("n_initial", [1, 3])
def test_pi_bo_ei_run_uses_the_prior_design_and_beta_n_over_10(monkeypatch, tmp_path, n_initial) -> None:
    bo_module = importlib.import_module("bayesian_optimization.bo")
    original, calls = bo_module.pi_bo_shape, []

    def recorder(acquisition_values, prior, beta, t, valid_mask=None):
        calls.append((beta, t))
        return original(acquisition_values, prior, beta, t, valid_mask=valid_mask)

    monkeypatch.setattr(bo_module, "pi_bo_shape", recorder)
    out = _run(monkeypatch, tmp_path, CATALOG.methods["pi-bo-ei"], n_initial=n_initial, budget=6)
    want = list(pi_bo_initial_design(_pi_values(), n_initial, 1001))
    beta = (6 - n_initial) / 10
    assert out["pi_bo_initial_indices"] == want and want[0] == int(np.argmax(_pi_values()))
    assert out["metrics"]["evaluated_indices"][:n_initial] == want
    assert out["pi_bo_beta"] == beta
    assert calls == [(beta, t) for t in range(1, 6 - n_initial + 1)]
    assert out["strategy"] == "pi-bo" and out["alpha_effective"] == 0.0


# --------------------------------------------------------------------------- stored runs
def test_stored_pi_bo_runs_start_at_the_prior_mode() -> None:
    """results/paper pi-bo-ei runs: n_initial = 1, so every run starts at the prior mode and the rest is deterministic on
    the noiseless HPOBench tables (one trajectory per task)."""
    import json

    paper = PROJECT_ROOT / "results" / "paper"
    for study in ("injection", "injection-misleading"):
        methods = json.loads((paper / "hpo" / f"{study}.json").read_text(encoding="utf-8"))["methods"]
        units, greedy = methods["pi-bo-ei"]["budget-0016"], methods["prior-greedy"]["budget-0016"]
        assert len(units) == 40 and all(len(rows) == 10 for rows in units.values())
        assert all(r["status"] == "completed" and r["y"] == rows[0]["y"] for rows in units.values() for r in rows)
        # the first query is the prior mode, prior-greedy's first query
        assert all(r["y"][0] == g["y"][0] for u, rows in units.items() for r, g in zip(rows, greedy[u]))
