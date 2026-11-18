"""GP kernel option (Matérn-5/2 default, RBF, Matérn-1/2) of the surrogate and the BO loop.

Guards: (i) each kernel matches closed-form values and scikit-learn's kernels, for equal and
per-dimension lengthscales, with k(x, x) = signal variance; (ii) the default is the surrogate's
Matérn-5/2 expression, an explicit ``kernel="matern52"`` changes nothing, and the default surrogate
reproduces recorded posterior values; (iii) the setting
reaches every surrogate of the BO loop (objective, constraint, crash, Kriging-Believer refits) and a
study run from ``design.optimizer.kernel``; (iv) unknown names raise; (v) the kernel studies plan
the acquisition study's EI runs, and no other manifest sets a kernel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import bayesian_optimization.surrogate as S  # noqa: E402
from bayesian_optimization import BayesianOptimization, GaussianProcessSurrogate  # noqa: E402
from datasets.neuromosaic_common import OfflineOptimizationProblem  # noqa: E402
from experiments import worker  # noqa: E402
from experiments.catalog import RESULT_ROOT, STUDY_ROOT, StudyConfigurationError, load_catalog, load_study  # noqa: E402
from experiments.planner import build_study_plan  # noqa: E402

CATALOG = load_catalog()
FUNCTIONS = {"matern52": S._matern52, "rbf": S._rbf, "matern12": S._matern12}
CORTICAL_CARD = (
    PROJECT_ROOT / "experiments" / "priors" / "neuromosaic-dataset1-rat" / "left-knee-flexor"
    / "human-expert-map" / "prior.json"
)


def _closed_form(name: str, r: np.ndarray, s2: float) -> np.ndarray:
    if name == "matern52":
        return s2 * (1.0 + np.sqrt(5.0) * r + 5.0 / 3.0 * r**2) * np.exp(-np.sqrt(5.0) * r)
    if name == "rbf":
        return s2 * np.exp(-0.5 * r**2)
    return s2 * np.exp(-r)


def _reference_matern52(X1, X2, length_scale, signal_variance):
    """The Matérn-5/2 expression of the default path, written out here."""
    scaled_difference = (X1[:, None, :] - X2[None, :, :]) / length_scale[None, None, :]
    distance = np.sqrt(np.sum(scaled_difference**2, axis=2))
    sqrt_five_distance = np.sqrt(5.0) * distance
    return signal_variance * (
        1.0 + sqrt_five_distance + (5.0 / 3.0) * distance**2
    ) * np.exp(-sqrt_five_distance)


# --------------------------------------------------------------------------- formulas
@pytest.mark.parametrize("name", list(S.KERNELS))
def test_kernel_closed_form_isotropic_and_ard(name: str) -> None:
    X1 = np.array([[0.0, 0.0], [1.0, -2.0]])
    X2 = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, -2.0]])
    for length_scale in (np.array([1.0, 1.0]), np.array([0.5, 2.0])):
        diff = (X1[:, None, :] - X2[None, :, :]) / length_scale
        r = np.sqrt(np.sum(diff**2, axis=2))
        got = FUNCTIONS[name](X1, X2, length_scale, 1.7)
        np.testing.assert_allclose(got, _closed_form(name, r, 1.7), rtol=1e-13, atol=0.0)
    # r = 5 for (0, 0) vs (3, 4) at unit lengthscale.
    expected = {"matern52": (1 + 5 * np.sqrt(5) + 125 / 3) * np.exp(-5 * np.sqrt(5)),
                "rbf": np.exp(-12.5), "matern12": np.exp(-5.0)}[name]
    assert FUNCTIONS[name](X1[:1], X2[:1], np.ones(2), 1.0)[0, 0] == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize("name", list(S.KERNELS))
def test_kernel_matches_sklearn(name: str) -> None:
    kernels = pytest.importorskip("sklearn.gaussian_process.kernels")
    reference = {"matern52": lambda ls: kernels.Matern(length_scale=ls, nu=2.5),
                 "rbf": lambda ls: kernels.RBF(length_scale=ls),
                 "matern12": lambda ls: kernels.Matern(length_scale=ls, nu=0.5)}[name]
    rng = np.random.default_rng(0)
    for case in range(60):
        d = int(rng.integers(1, 7))
        X1 = rng.uniform(-1.0, 2.0, size=(int(rng.integers(1, 9)), d))
        X2 = rng.uniform(-1.0, 2.0, size=(int(rng.integers(1, 9)), d))
        s2 = float(rng.uniform(0.1, 4.0))
        ard = bool(case % 2)
        ls = rng.uniform(0.05, 3.0, size=d) if ard else np.full(d, float(rng.uniform(0.05, 3.0)))
        expected = s2 * reference(ls if ard else float(ls[0]))(X1, X2)
        np.testing.assert_allclose(FUNCTIONS[name](X1, X2, ls, s2), expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("name", list(S.KERNELS))
def test_prior_variance_is_signal_variance(name: str) -> None:
    X = np.random.default_rng(1).uniform(size=(6, 3))
    gp = GaussianProcessSurrogate(length_scale=0.3, signal_variance=2.5, kernel=name)
    assert np.array_equal(np.diag(gp.kernel(X, X)), np.full(6, 2.5))


@pytest.mark.parametrize("bad", [[0.3, 0.5], np.array([0.2]), 0.0, -1.0, np.inf])
def test_length_scale_must_be_a_positive_scalar(bad) -> None:
    with pytest.raises(ValueError, match="length_scale"):
        GaussianProcessSurrogate(length_scale=bad)


# --------------------------------------------------------------------------- default path
def _data(seed: int = 3):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(9, 2))
    return X, np.sin(4.0 * X[:, 0]) + X[:, 1] ** 2, rng.uniform(size=(7, 2))


def test_default_kernel_is_the_reference_matern52() -> None:
    X, y, Xtest = _data()
    default = GaussianProcessSurrogate(length_scale=0.2, noise_variance=1e-6, normalize_y=True)
    explicit = GaussianProcessSurrogate(length_scale=0.2, noise_variance=1e-6, normalize_y=True, kernel="matern52")
    assert S.DEFAULT_KERNEL == "matern52" and default.kernel_name == "matern52"
    assert np.array_equal(default.kernel(X, Xtest), _reference_matern52(X, Xtest, np.full(2, 0.2), 1.0))
    for a, b in zip(default.fit(X, y).predict(Xtest), explicit.fit(X, y).predict(Xtest)):
        assert np.array_equal(a, b)


def test_default_surrogate_matches_recorded_predictions() -> None:
    """The default surrogate reproduces recorded posterior means and standard deviations."""
    X = np.array([[0.0], [0.5], [1.0]])
    y = np.array([0.2, 0.8, 0.4])
    query = np.array([[0.25], [0.75]])
    surrogate = GaussianProcessSurrogate(length_scale=0.7, signal_variance=1.3, noise_variance=1e-5, normalize_y=True)
    mean, std = surrogate.fit(X, y).predict(query)
    np.testing.assert_allclose(mean, [0.55864109, 0.68099510], rtol=1e-7)
    np.testing.assert_allclose(std, [0.04835284, 0.04835284], rtol=1e-7)


def _trajectory(**kwargs) -> list[int]:
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 9), np.linspace(0, 1, 7)), -1).reshape(-1, 2)
    values = np.sin(5.0 * grid[:, 0]) * np.cos(3.0 * grid[:, 1])
    prior = np.exp(-((grid[:, 0] - 0.3) ** 2) / 0.1)
    bo = BayesianOptimization(grid, prior=prior, alpha=0.5, n_initial=1, length_scale=0.2,
                              noise_variance=1e-6, normalize_y=True, random_state=7, **kwargs)
    for _ in range(10):
        index, _ = bo.ask()
        bo.tell(index, float(values[index]))
    return list(bo.evaluated_indices)


def test_default_bo_loop_uses_only_matern52(monkeypatch) -> None:
    calls = {name: 0 for name in FUNCTIONS}

    def counting(name):
        def wrapped(*args):
            calls[name] += 1
            return FUNCTIONS[name](*args)
        return wrapped

    for name in FUNCTIONS:
        monkeypatch.setattr(S, f"_{name}", counting(name))
    default = _trajectory()
    assert calls["matern52"] > 0 and calls["rbf"] == calls["matern12"] == 0
    assert _trajectory(kernel="matern52") == default


# --------------------------------------------------------------------------- plumbing
def test_kernel_reaches_every_surrogate_of_the_bo_loop(monkeypatch) -> None:
    calls = {"rbf": 0, "matern52": 0}

    def rbf(*args):
        calls["rbf"] += 1
        return FUNCTIONS["rbf"](*args)

    def matern52(*args):
        calls["matern52"] += 1
        return FUNCTIONS["matern52"](*args)

    monkeypatch.setattr(S, "_rbf", rbf)
    monkeypatch.setattr(S, "_matern52", matern52)
    grid = np.random.default_rng(5).uniform(size=(40, 2))
    values = -np.sum((grid - 0.4) ** 2, axis=1)
    bo = BayesianOptimization(grid, n_initial=2, length_scale=0.3, random_state=3, n_constraints=1,
                              model_hidden_failures=True, kernel="rbf")
    gps = [bo.surrogate, *bo.constraint_surrogates, bo.crash_surrogate]
    assert bo.kernel == "rbf" and all(gp.kernel_name == "rbf" for gp in gps)
    while bo.n_observations < 10:
        index, _ = bo.ask()
        bo.tell(index, float(values[index]), np.array([-1.0]), crashed=bo.n_observations == 2)
    assert all(gp.is_fitted for gp in gps)
    sequential_calls = calls["rbf"]
    batch = BayesianOptimization(grid, n_initial=2, length_scale=0.3, random_state=3, kernel="rbf")
    while batch.n_observations < 10:
        for index in batch.ask_batch(3):  # Kriging Believer: fantasy refits of the objective GP
            batch.tell(index, float(values[index]))
    assert sequential_calls > 0 and calls["rbf"] > sequential_calls and calls["matern52"] == 0
    assert _trajectory(kernel="rbf") != _trajectory() != _trajectory(kernel="matern12")


def test_worker_settings_carry_the_kernel() -> None:
    method = {"acquisition": "expected-improvement", "strategy": "vanilla-bo"}
    base = {"seed": 1, "optimizer": {"n_initial": 1, "length_scale": 0.2, "noise_std": 0.001}}
    assert worker._optimizer_settings(base, method)["kernel"] == "matern52"
    run = {"seed": 1, "budget": 4, "optimizer": {**base["optimizer"], "kernel": "matern12"}}
    assert worker._optimizer_settings(run, method)["kernel"] == "matern12"
    optimizer = worker._build_optimizer(np.random.default_rng(0).uniform(size=(12, 2)), run, method,
                                        prior=None, alpha=0.0, maximize=True)
    assert optimizer.kernel == "matern12" and optimizer.surrogate.kernel_name == "matern12"


def _cortical_problem() -> OfflineOptimizationProblem:
    coords = [(x, y) for x in range(1, 9) for y in range(1, 5)]
    native = tuple({"channel_x": x, "channel_y": y} for x, y in coords)
    truth = np.random.default_rng(1).uniform(size=len(native))
    lookup = {(c["channel_x"], c["channel_y"]): float(v) for c, v in zip(native, truth)}
    return OfflineOptimizationProblem(
        dataset_id="neuromosaic-dataset1-rat", subject_id="synthetic", target_name="LeftKneeFlexor",
        native_candidates=native, encoded_candidates=np.asarray(coords, dtype=float), ground_truth=truth,
        response_standard_deviation=np.zeros_like(truth),
        objective=lambda c: lookup[(c["channel_x"], c["channel_y"])], metadata={}, maximize=True,
    )


@pytest.mark.parametrize("method_id", ["alpha-pi-raw-ei-matched", "vanilla-bo-ei"])
def test_study_run_with_rbf_calls_the_rbf_kernel(monkeypatch, tmp_path, method_id) -> None:
    counts = {"rbf": 0, "matern52": 0}

    def counted(name):
        def wrapped(*args):
            counts[name] += 1
            return FUNCTIONS[name](*args)
        return wrapped

    monkeypatch.setattr(S, "_rbf", counted("rbf"))
    monkeypatch.setattr(S, "_matern52", counted("matern52"))
    monkeypatch.setattr(worker, "_load_offline_problem", lambda *a, **k: _cortical_problem())
    run = {"run_id": "t", "method_id": method_id, "seed": 1001, "budget": 6, "parameters": {},
           "optimizer": {"n_initial": 1, "length_scale": 1.0, "noise_std": 0.05, "kernel": "rbf"},
           "prior_path": str(CORTICAL_CARD)}
    worker._run_offline(tmp_path, run, {"adapter": "neuromosaic-dataset1-rat"}, CATALOG.methods[method_id])
    assert counts["rbf"] > 0 and counts["matern52"] == 0
    assert len((tmp_path / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()) == 6


# --------------------------------------------------------------------------- unknown names
@pytest.mark.parametrize("bad", ["matern32", "RBF", "", None, 2.5])
def test_unknown_kernel_raises(bad) -> None:
    with pytest.raises(ValueError, match="unknown GP kernel"):
        GaussianProcessSurrogate(kernel=bad)
    with pytest.raises(ValueError, match="unknown GP kernel"):
        BayesianOptimization(np.eye(3), kernel=bad)
    with pytest.raises(ValueError, match="unknown GP kernel"):
        S._covariance_function(bad)


def test_unknown_kernel_in_a_manifest_fails_planning() -> None:
    study = load_study(STUDY_ROOT / "hpo" / "acquisition")
    study["design"]["optimizer"] = {**study["design"]["optimizer"], "kernel": "laplace"}
    with pytest.raises(StudyConfigurationError, match="design.optimizer.kernel"):
        build_study_plan(study, CATALOG)


@pytest.mark.parametrize("study,method", [("hpo/vanilla-random", "random-search"), ("hpo/colabo", "colabo-llm"),
                                          ("hpo/caliper", "caliper"), ("hpo/priorband", "priorband"),
                                          ("hpo/injection", "prior-greedy")])
def test_kernel_with_a_method_that_does_not_fit_the_gp_fails_planning(study: str, method: str) -> None:
    manifest = load_study(STUDY_ROOT / study)
    manifest["design"]["optimizer"] = {**manifest["design"]["optimizer"], "kernel": "rbf"}
    with pytest.raises(StudyConfigurationError, match=f"method '{method}'"):
        build_study_plan(manifest, CATALOG)


def test_kernel_is_accepted_by_every_gp_strategy() -> None:
    from experiments.planner import GP_KERNEL_STRATEGIES

    manifest = load_study(STUDY_ROOT / "hpo" / "kernel-rbf")
    gp_methods = sorted(m for m, spec in CATALOG.methods.items() if spec.get("strategy") in GP_KERNEL_STRATEGIES)
    assert {CATALOG.methods[m]["strategy"] for m in gp_methods} == set(GP_KERNEL_STRATEGIES)
    manifest["design"]["methods"] = gp_methods
    plan = build_study_plan(manifest, CATALOG)
    assert {r.method_id for r in plan.runs} == set(gp_methods)


# --------------------------------------------------------------------------- kernel studies
EI_ARMS = ["alpha-pi-raw-ei-matched", "vanilla-bo-ei"]


@pytest.mark.parametrize("group,units,budget", [("hpo", 40, 16), ("cortical", 40, 12)])
@pytest.mark.parametrize("kernel", ["rbf", "matern12"])
def test_kernel_studies_plan_the_acquisition_ei_runs(group: str, units: int, budget: int, kernel: str) -> None:
    plan = build_study_plan(load_study(STUDY_ROOT / group / f"kernel-{kernel}"), CATALOG)
    acquisition = build_study_plan(load_study(STUDY_ROOT / group / "acquisition"), CATALOG)
    assert len(plan.runs) == units * len(EI_ARMS) * 10 and plan.blocked_count == 0
    assert all(r.study_id == f"kernel-{kernel}" and r.budget == budget for r in plan.runs)
    assert all(r.optimizer == {**acquisition.runs[0].optimizer, "kernel": kernel} for r in plan.runs)

    def design(runs):
        return sorted((r.benchmark_id, r.problem_id, r.method_id, r.budget, r.seed, r.replicate, str(r.prior_path),
                       r.result_directory.relative_to(RESULT_ROOT / group / r.study_id).as_posix())
                      for r in runs)

    assert design(plan.runs) == design(r for r in acquisition.runs if r.method_id in EI_ARMS)


def test_make_studies_writes_the_kernel_manifests() -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import make_studies

    written = {path: text for path, text in make_studies.hpo_studies() + make_studies.cortical_studies()}
    for group in ("hpo", "cortical"):
        for kernel in ("rbf", "matern12"):
            path = STUDY_ROOT / group / f"kernel-{kernel}" / "study.yaml"
            assert written[path] == path.read_text(encoding="utf-8")


def test_only_the_kernel_studies_set_a_kernel() -> None:
    for path in sorted(STUDY_ROOT.glob("*/*/study.yaml")):
        optimizer = (load_study(path).get("design") or {}).get("optimizer", {})
        assert ("kernel" in optimizer) == path.parent.name.startswith("kernel-"), path
