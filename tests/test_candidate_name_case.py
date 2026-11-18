"""Benchmark candidate keys match prior-schema names case-insensitively.

HPOBench's SVM table names the penalty ``C`` while the PSL priors name the dimension ``c``.
The unit tests use a synthetic SVM grid; the study runs need the HPOBench SVM tables and are
skipped when they are absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from main import _as_native_candidate_records, encode_candidates_for_gp, main  # noqa: E402
from experiments import worker  # noqa: E402
from experiments.catalog import load_catalog, load_study  # noqa: E402
from experiments.planner import build_study_plan, slug  # noqa: E402
from prior.landscape.landscape_generator import compile_prior, load_record  # noqa: E402

CATALOG = load_catalog()
SVM_PRIORS = PROJECT_ROOT / "experiments" / "priors" / "hpobench-svm-blood-transfusion" / "blood-transfusion"
SVM_TABLE = PROJECT_ROOT / "benchmark" / "HPOBench" / "svm" / "10101" / "svm_10101_data.parquet.gzip"
SVM_TASK = "svm-blood_transfusion"


# --------------------------------------------------------------------------- matching
def test_keys_are_matched_ignoring_case_and_rekeyed_in_schema_order() -> None:
    records = _as_native_candidate_records(
        [{"gamma": 0.5, "C": 2.0}, {"Gamma": 1.0, "c": 4.0}], ("c", "gamma")
    )
    assert records == ({"c": 2.0, "gamma": 0.5}, {"c": 4.0, "gamma": 1.0})
    assert [list(record) for record in records] == [["c", "gamma"], ["c", "gamma"]]


def test_exact_names_follow_the_schema_order() -> None:
    records = _as_native_candidate_records([{"alpha": 1e-3, "eta0": 0.1}], ("eta0", "alpha"))
    assert records == ({"eta0": 0.1, "alpha": 1e-3},)
    assert list(records[0]) == ["eta0", "alpha"]


@pytest.mark.parametrize(
    "candidates, names",
    [
        ([{"C": 1.0, "c": 2.0, "gamma": 3.0}], ("c", "gamma")),
        ([{"C": 1.0, "gamma": 3.0}, {"C": 1.0, "c": 2.0, "gamma": 3.0}], ("c", "gamma")),
        ([{"C": 1.0, "gamma": 3.0}], ("c", "C", "gamma")),
    ],
)
def test_names_equal_up_to_case_are_rejected(candidates, names) -> None:
    with pytest.raises(ValueError, match="ambiguous when case is ignored"):
        _as_native_candidate_records(candidates, names)


def test_missing_and_extra_keys_report_the_original_spelling() -> None:
    with pytest.raises(
        ValueError,
        match=r"native candidate 1 does not match the prior schema "
        r"\(missing=\['gamma'\], extra=\['Kernel'\]\)",
    ):
        _as_native_candidate_records(
            [{"C": 1.0, "gamma": 2.0}, {"C": 1.0, "Kernel": "rbf"}], ("c", "gamma")
        )


def test_matrix_candidates_map_columns_to_schema_names() -> None:
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected = ({"c": 1.0, "gamma": 2.0}, {"c": 3.0, "gamma": 4.0})
    assert _as_native_candidate_records(matrix, ("c", "gamma")) == expected
    assert _as_native_candidate_records(matrix.tolist(), ("c", "gamma")) == expected
    with pytest.raises(ValueError, match="one column per prior-schema dimension"):
        _as_native_candidate_records(matrix, ("c",))


# --------------------------------------------------------------------------- synthetic SVM grid
def _svm_problem() -> SimpleNamespace:
    """The HPOBench SVM grid (21 x 21 powers of two) with a table keyed by ``C``."""

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


def test_svm_benchmark_keys_encode_like_schema_keys() -> None:
    compiled = compile_prior(load_record(SVM_PRIORS / "llm-card" / "prior.json"))
    native = _svm_problem().native_candidates
    lowered = tuple({key.lower(): value for key, value in c.items()} for c in native)
    np.testing.assert_array_equal(
        encode_candidates_for_gp(native, compiled), encode_candidates_for_gp(lowered, compiled)
    )


def test_svm_prior_context() -> None:
    native, encoded, pi_values, *_ = worker._offline_prior_context(
        _svm_problem(), "hpobench", SVM_PRIORS / "llm-card" / "prior.json"
    )
    assert native[0] == {"C": 2.0**-10, "gamma": 2.0**-10}
    assert encoded.shape == (441, 2) and pi_values.shape == (441,)
    assert np.all(np.isfinite(pi_values))


@pytest.mark.parametrize(
    "method_id, card",
    [
        ("alpha-pi-raw-ei-matched", "llm-card"),
        ("alpha-pi-raw-ei-matched", "llm-card-misleading"),
        ("alpha-pi-raw-ei-matched", "inverted"),
        ("pi-bo-ei", "llm-card"),
        ("prior-to-mean", "llm-card"),
        ("prior-warm-start", "llm-card"),
        ("prior-greedy", "llm-card"),
        ("vanilla-bo-ei", "base"),
        ("vanilla-bo-ei", None),
    ],
)
def test_svm_offline_arms_complete(monkeypatch, tmp_path, method_id, card) -> None:
    import main as main_module

    monkeypatch.setattr(worker, "_load_offline_problem", lambda *a, **k: _svm_problem())
    monkeypatch.setattr(main_module, "save_performance_plot", lambda *a, **k: None)
    run = {
        "run_id": "t", "method_id": method_id, "seed": 1001, "budget": 6, "parameters": {},
        "optimizer": {"n_initial": 1, "length_scale": 0.2, "noise_std": 0.001, "improvement_xi": 0.0},
        "prior_path": None if card is None else str(SVM_PRIORS / card / "prior.json"),
    }
    outcome = worker._run_offline(tmp_path, run, {"adapter": "hpobench"}, CATALOG.methods[method_id])
    assert len(outcome["metrics"]["evaluated_indices"]) == 6
    assert set(outcome["metrics"]["best_configuration"]) == {"C", "gamma"}


# --------------------------------------------------------------------------- HPOBench SVM tables
@pytest.mark.skipif(not SVM_TABLE.is_file(), reason="HPOBench SVM table not present")
@pytest.mark.parametrize(
    "study, method_id",
    [("hapbo", "alpha-pi-raw-ei-matched"), ("vanilla-random", "vanilla-bo-ei"), ("injection", "pi-bo-ei"),
     ("injection-misleading", "pi-bo-ei"), ("alpha-sweep", "alpha-pi-raw-ei-matched-alpha-0-5")],
)
def test_svm_study_run_reproduces_the_paper_trajectory(tmp_path, study, method_id) -> None:
    plan = build_study_plan(load_study(PROJECT_ROOT / "experiments" / "studies" / "hpo" / study), CATALOG)
    run = next(
        r for r in plan.runs if r.problem_id == SVM_TASK and r.method_id == method_id and r.seed == 1001
    )
    record = run.to_dict()
    record["result_directory"] = str(tmp_path)
    (tmp_path / "study_run.json").write_text(json.dumps(record), encoding="utf-8")
    worker.execute_run_directory(tmp_path)

    lines = (tmp_path / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()
    trajectory = [json.loads(line)["ground_truth_objective"] for line in lines]
    paper = json.loads((PROJECT_ROOT / "results" / "paper" / "hpo" / f"{study}.json").read_text(encoding="utf-8"))
    stored = next(
        r for r in paper["methods"][method_id][f"budget-{run.budget:04d}"][slug(SVM_TASK)] if r["seed"] == 1001
    )
    assert stored["status"] == "completed"
    assert trajectory == stored["y"]


@pytest.mark.skipif(not SVM_TABLE.is_file(), reason="HPOBench SVM table not present")
def test_hpobench_command_runs_an_svm_task(tmp_path, capsys) -> None:
    assert main([
        "hpobench-bo", "--hpobench-model", "svm", "--hpobench-task", "blood_transfusion",
        "--queries", "4", "--plot", str(tmp_path / "svm.png"),
    ]) == 0
    out = capsys.readouterr().out
    assert "Evaluations: 4" in out
    assert out.count('{"C": ') == 5 and '"c": ' not in out  # 4 queries + best configuration


@pytest.mark.skipif(not SVM_TABLE.is_file(), reason="HPOBench SVM table not present")
def test_hpobench_command_runs_a_baseline_with_an_svm_prior(monkeypatch, tmp_path, capsys) -> None:
    import main as main_module

    args = main_module.build_parser().parse_args([
        "hpobench-bo", "--hpobench-model", "svm", "--hpobench-task", "blood_transfusion",
        "--method", "vanilla_bo", "--queries", "4",
        "--dataset-prior-file", str(SVM_PRIORS / "base" / "prior.json"),
    ])
    monkeypatch.setattr(main_module, "PROJECT_ROOT", tmp_path)  # the baseline writes under results/
    main_module.run_hpobench(args)
    assert "Evaluations: 4" in capsys.readouterr().out
    assert any((tmp_path / "results").rglob("*.json"))
