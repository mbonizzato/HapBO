"""Fixed-alpha sweep: the alpha-pi method field ``alpha``, its catalog arms, manifests, stored runs and figure."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from experiments import worker  # noqa: E402
from experiments.catalog import load_catalog, load_study  # noqa: E402
from experiments.planner import build_study_plan  # noqa: E402
from prior.landscape.landscape_generator import compile_prior, load_record  # noqa: E402

CATALOG = load_catalog()
BASE = "alpha-pi-raw-ei-matched"
ALPHAS = (0.9, 0.7, 0.5, 0.3, 0.1)
ARMS = {a: f"{BASE}-alpha-{a:.1f}".replace(".", "-") for a in ALPHAS}
PRIOR = (
    PROJECT_ROOT / "experiments" / "priors" / "hpobench-svm-blood-transfusion" / "blood-transfusion"
    / "llm-card" / "prior.json"
)
STUDIES = PROJECT_ROOT / "experiments" / "studies" / "hpo"
PAPER = PROJECT_ROOT / "results" / "paper" / "hpo"


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


def _run(monkeypatch, tmp_path, method, seed=1001, budget=8):
    import main as main_module

    monkeypatch.setattr(worker, "_load_offline_problem", lambda *a, **k: _svm_problem())
    monkeypatch.setattr(main_module, "save_performance_plot", lambda *a, **k: None)
    run = {
        "run_id": "t", "method_id": method["method_id"], "seed": seed, "budget": budget, "parameters": {},
        "optimizer": {"n_initial": 1, "length_scale": 0.2, "noise_std": 0.001, "improvement_xi": 0.0},
        "prior_path": str(PRIOR),
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    return worker._run_offline(tmp_path, run, {"adapter": "hpobench"}, method)


# --------------------------------------------------------------------------- the method field
def test_method_alpha_is_validated() -> None:
    base = CATALOG.methods[BASE]
    assert worker._method_alpha(base) is None
    assert worker._method_alpha(dict(base, alpha=0.3)) == 0.3
    assert worker._method_alpha(dict(base, alpha=0)) == 0.0
    assert worker._method_alpha(dict(base, alpha=0.9)) == 0.9
    for bad in (-0.1, 0.95, 1.5, math.nan, math.inf):  # the alpha-pi rule caps alpha at 0.9
        with pytest.raises(ValueError, match="alpha"):
            worker._method_alpha(dict(base, alpha=bad))
    with pytest.raises(ValueError, match="alpha-pi"):
        worker._method_alpha(dict(CATALOG.methods["pi-bo-ei"], alpha=0.5))


@pytest.mark.parametrize("method_id", ["bora", "priorband"])
def test_external_baselines_reject_the_alpha_field_before_running(monkeypatch, tmp_path, method_id) -> None:
    method = dict(CATALOG.methods[method_id], alpha=0.5)
    assert method["strategy"] in ("external", "external-mf")
    called = []
    monkeypatch.setattr(worker, "_run_external_baseline", lambda *a, **k: called.append("external"))
    import baselines.neps_mf as neps_mf
    monkeypatch.setattr(neps_mf, "run_neps_mf_study_run", lambda *a, **k: called.append("external-mf"))
    with pytest.raises(ValueError, match="alpha-pi"):
        _run(monkeypatch, tmp_path, method)
    assert called == []


def test_fixed_alpha_replaces_the_compiled_alpha_and_is_recorded(monkeypatch, tmp_path) -> None:
    compiled = float(compile_prior(load_record(PRIOR)).alpha)
    out = _run(monkeypatch, tmp_path / "a", CATALOG.methods[ARMS[0.5]])
    assert out["alpha_requested"] == out["alpha_effective"] == out["alpha_fixed"] == 0.5
    assert out["alpha_compiled"] == compiled != 0.5
    reference = _run(monkeypatch, tmp_path / "b", CATALOG.methods[BASE])
    assert reference["alpha_requested"] == reference["alpha_effective"] == compiled
    assert "alpha_fixed" not in reference and "alpha_compiled" not in reference


def test_fixed_alpha_equal_to_the_compiled_alpha_gives_the_compiled_run(monkeypatch, tmp_path) -> None:
    compiled = float(compile_prior(load_record(PRIOR)).alpha)
    reference = _run(monkeypatch, tmp_path / "a", CATALOG.methods[BASE])
    pinned = _run(monkeypatch, tmp_path / "b", dict(CATALOG.methods[BASE], alpha=compiled))
    assert pinned["metrics"]["evaluated_indices"] == reference["metrics"]["evaluated_indices"]


# --------------------------------------------------------------------------- catalog and manifests
def test_catalog_arms_are_the_matched_hapbo_arm_with_alpha_fixed() -> None:
    skip = {"method_id", "label", "note", "role", "alpha", "_source_path"}
    base = {k: v for k, v in CATALOG.methods[BASE].items() if k not in skip}
    for a, mid in ARMS.items():
        arm = CATALOG.methods[mid]
        assert arm["alpha"] == a and arm["role"] == "ablation"
        assert {k: v for k, v in arm.items() if k not in skip} == base


@pytest.mark.parametrize("sweep, hapbo", [("alpha-sweep", "hapbo"), ("alpha-sweep-misleading", "hapbo-misleading")])
def test_sweep_manifests_share_tasks_priors_seeds_and_settings_with_hapbo(sweep, hapbo) -> None:
    def cells(name):
        plan = build_study_plan(load_study(STUDIES / name), CATALOG)
        return plan, {(r.problem_id, str(r.prior_path), r.budget, r.seed, json.dumps(r.optimizer, sort_keys=True))
                      for r in plan.runs}

    sweep_plan, sweep_cells = cells(sweep)
    hapbo_plan, hapbo_cells = cells(hapbo)
    assert sweep_cells == hapbo_cells and len(hapbo_cells) == 400
    assert sorted({r.method_id for r in sweep_plan.runs}) == sorted(ARMS.values())
    assert len(sweep_plan.runs) == 2000 and sweep_plan.blocked_count == 0


# --------------------------------------------------------------------------- stored runs and figure
@pytest.mark.parametrize("study", ["alpha-sweep", "alpha-sweep-misleading"])
def test_stored_sweep_runs_are_complete(study) -> None:
    data = json.loads((PAPER / f"{study}.json").read_text(encoding="utf-8"))
    assert sorted(data["methods"]) == sorted(ARMS.values())
    for budgets in data["methods"].values():
        units = budgets["budget-0016"]
        assert len(units) == 40
        assert all([r["seed"] for r in rows] == list(range(1001, 1011)) for rows in units.values())
        assert all(r["status"] == "completed" and len(r["y"]) == 16 for rows in units.values() for r in rows)


def test_alpha_sweep_figure(tmp_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    sys.path.insert(0, str(PROJECT_ROOT / "paper"))
    import fig_alpha_sweep

    summary = fig_alpha_sweep.build(tmp_path)["fig_alpha_sweep_hapbo"]
    assert (tmp_path / "fig_alpha_sweep_hapbo.pdf").is_file() and (tmp_path / "fig_alpha_sweep_hapbo.png").is_file()
    for cond, metric in (("card", "span"), ("misleading", "vq1")):
        s = summary[cond]
        assert s["metric"] == metric and s["n_tasks"] == 40 and s["n_pairs"] == 400
        assert sorted(s["q16"]) == sorted([str(a) for a in ALPHAS] + ["compiled", "vanilla"])
        assert 0.325 <= s["compiled_alpha_mean"] <= 0.455
