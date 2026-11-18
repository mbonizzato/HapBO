"""Baseline coordinate legend; bridge budget rule (every proposal is charged, repeats return
the cached value and are flagged, no random fill, a run whose worker stops early fails)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.baselines.bridge import BridgeWorkerError, SubprocessBridgeOptimizer, require_full_budget  # noqa: E402
from src.baselines.runner import run_ask_tell_loop  # noqa: E402

_WORKERS = PROJECT_ROOT / "src" / "baselines" / "workers"

# A worker that proposes 4, 7, 4 (a repeat), 9 and then stops early.
_STUB = f"""
import sys
sys.path.insert(0, {str(_WORKERS)!r})
from harness import finish, log, make_evaluator, read_spec
spec = read_spec()
evaluate = make_evaluator()
log("stub worker: stopping after four proposals")
sys.stderr.write("STUB-STDERR-LINE\\n"); sys.stderr.flush()
values = [evaluate(i) for i in (4, 7, 4, 9)]
log(f"values seen by the worker: {{values}}")
finish()
"""


def _stub_bridge(tmp_path: Path, budget: int = 8, **kwargs):
    script = tmp_path / "stub_worker.py"
    script.write_text(_STUB, encoding="utf-8")

    class StubBridge(SubprocessBridgeOptimizer):
        name = "stub"
        worker_command = [sys.executable, str(script)]

    return StubBridge(
        search_space={"candidates": [{"x": i} for i in range(30)]},
        objective_meta={"maximize": True}, seed=1001, budget=budget, **kwargs)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- bridge budget rule
def test_bridge_charges_repeats_and_never_fills(tmp_path: Path) -> None:
    log_path = tmp_path / "worker_stderr.log"
    opt = _stub_bridge(tmp_path, budget=8)
    opt.worker_log_path = str(log_path)
    calls: list[int] = []
    try:
        history = run_ask_tell_loop(opt, lambda c: calls.append(c["x"]) or float(c["x"]), budget=8, n_init=1)
    finally:
        opt.close()
    assert [c["x"] for c, _ in history] == [4, 7, 4, 9]  # every proposal is a query; nothing random
    assert [row["repeat"] for row in opt.trace] == [False, False, True, False]
    assert calls == [4, 7, 9]  # the repeat returned the cached value (objective not re-run)
    assert history[2][1] == history[0][1]
    acc = opt.bridge_accounting()
    assert acc == {"bridge_policy": "charge", "budget": 8, "queries_used": 4, "n_unique": 3, "n_repeats": 1,
                   "proposals_exhausted_at": 5, "worker_log": str(log_path)}
    text = log_path.read_text(encoding="utf-8")
    assert "STUB-STDERR-LINE" in text and "stopping after four proposals" in text
    assert "values seen by the worker: [4.0, 7.0, 4.0, 9.0]" in text
    assert "worker done after 4 charged queries" in text


def test_bridge_repeat_consumes_the_last_budget_unit(tmp_path: Path) -> None:
    opt = _stub_bridge(tmp_path, budget=3)
    try:
        history = run_ask_tell_loop(opt, lambda c: float(c["x"]), budget=3, n_init=1)
    finally:
        opt.close()
    assert [c["x"] for c, _ in history] == [4, 7, 4]
    acc = opt.bridge_accounting()
    assert acc["n_repeats"] == 1 and acc["queries_used"] == 3 and acc["proposals_exhausted_at"] is None


def test_bridge_validates_its_policy_and_runs_only_through_the_charged_loop(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _stub_bridge(tmp_path, bridge_policy="unknown")
    opt = _stub_bridge(tmp_path, bridge_policy="charge")
    with pytest.raises(BridgeWorkerError):
        opt.ask()  # bridges run only through the charged loop


def test_a_short_trace_is_a_failure() -> None:
    require_full_budget(8, 8, "stub")
    with pytest.raises(BridgeWorkerError, match="stub worker stopped after 4 of 8 queries"):
        require_full_budget(4, 8, "stub")


@pytest.mark.parametrize("budget", [4, 8])
def test_baseline_experiment_accepts_only_a_full_bridged_trace(tmp_path: Path, monkeypatch, budget: int) -> None:
    import src.baselines.runner as runner

    made = []

    def fake_get_baseline(name, **kwargs):
        made.append(_stub_bridge(tmp_path, budget=kwargs["budget"]))
        return made[-1]

    monkeypatch.setattr(runner, "get_baseline", fake_get_baseline)
    kwargs = dict(search_space={"candidates": [{"x": i} for i in range(30)]}, objective=lambda c: float(c["x"]),
                  objective_meta={"maximize": True}, dataset_path="stub-task", seed=1001, budget=budget,
                  results_root=tmp_path / "results")
    try:
        if budget == 4:  # the stub proposes 4 queries: a full run
            out = runner.run_baseline_experiment("stub", **kwargs)
            assert [c["x"] for c, _ in out.history] == [4, 7, 4, 9]
            assert len((out.output_directory / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()) == 4
        else:
            with pytest.raises(BridgeWorkerError, match="stub worker stopped after 4 of 8 queries"):
                runner.run_baseline_experiment("stub", **kwargs)
            assert not list((tmp_path / "results").rglob("evaluations.jsonl"))
    finally:
        for opt in made:
            opt.close()


def _harness_stub(tmp_path: Path, monkeypatch):
    import baselines
    from baselines.bridge import SubprocessBridgeOptimizer as HarnessBridge

    script = tmp_path / "stub_worker.py"
    script.write_text(_STUB, encoding="utf-8")

    class StubBridge(HarnessBridge):
        name = "stub"
        worker_command = [sys.executable, str(script)]

    seen = {}

    def fake_get_baseline(name, **kwargs):
        seen["objective_meta"] = kwargs["objective_meta"]
        return StubBridge(**kwargs)

    monkeypatch.setattr(baselines, "get_baseline", fake_get_baseline)
    problem, run, _, _ = _lr_kc1()
    run = dict(run, seed=1001, method_id="stub", optimizer={"n_initial": 1, "bridge_policy": "charge", "baseline_task_legend": True})
    return problem, run, seen


def test_worker_persists_charge_accounting_of_a_full_run(tmp_path: Path, monkeypatch) -> None:
    from experiments.worker import _run_external_baseline

    problem, run, seen = _harness_stub(tmp_path, monkeypatch)
    out = _run_external_baseline(tmp_path, run, {"adapter": "hpobench"}, {"baseline": "stub", "label": "stub"},
                                 problem, 4)
    acc = out["bridge_accounting"]
    assert (acc["queries_used"], acc["n_repeats"], acc["proposals_exhausted_at"]) == (4, 1, None)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))["methods"]["stub"]
    assert metrics["bridge_accounting"] == acc
    assert json.loads((tmp_path / "bridge_accounting.json").read_text(encoding="utf-8")) == acc
    records = [json.loads(line) for line in (tmp_path / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["query"] for r in records] == [1, 2, 3, 4]
    assert [r["candidate_index"] for r in records] == [4, 7, 4, 9]
    assert [r["repeat"] for r in records] == [False, False, True, False]
    assert records[2]["objective"] == records[0]["objective"]
    assert "STUB-STDERR-LINE" in (tmp_path / "worker_stderr.log").read_text(encoding="utf-8")
    text = json.loads((tmp_path / "baseline_task_text.json").read_text(encoding="utf-8"))
    assert text["coordinate_legend"].startswith("Search coordinates:")
    assert seen["objective_meta"]["coordinate_legend"] == text["coordinate_legend"]


def test_worker_fails_a_run_whose_worker_stops_early(tmp_path: Path, monkeypatch) -> None:
    from experiments.worker import _run_external_baseline

    problem, run, _ = _harness_stub(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="stub worker stopped after 4 of 8 queries"):
        _run_external_baseline(tmp_path, run, {"adapter": "hpobench"}, {"baseline": "stub", "label": "stub"},
                               problem, 8)
    acc = json.loads((tmp_path / "bridge_accounting.json").read_text(encoding="utf-8"))
    assert (acc["budget"], acc["queries_used"], acc["n_repeats"], acc["proposals_exhausted_at"]) == (8, 4, 1, 5)
    assert not (tmp_path / "evaluations.jsonl").exists() and not (tmp_path / "metrics.json").exists()


_SHORT_RUN = """
import sys
from pathlib import Path
sys.path[:0] = [{root!r}, {src!r}, {tests!r}]
import baselines
from baselines.bridge import SubprocessBridgeOptimizer
from experiments.worker import _run_external_baseline
from test_baseline_legend_bridge import _lr_kc1


class StubBridge(SubprocessBridgeOptimizer):
    name = "stub"
    worker_command = [sys.executable, sys.argv[2]]


baselines.get_baseline = lambda name, **kwargs: StubBridge(**kwargs)
problem, run, _, _ = _lr_kc1()
run = dict(run, seed=1001, method_id="stub", optimizer={{"n_initial": 1, "bridge_policy": "charge"}})
_run_external_baseline(Path(sys.argv[1]), run, {{"adapter": "hpobench"}}, {{"baseline": "stub", "label": "stub"}},
                       problem, 8)
"""


def test_a_short_bridge_run_ends_with_status_failed(tmp_path: Path) -> None:
    from experiments.execution import run_planned_run
    from experiments.planner import PlannedRun

    _lr_kc1()  # skips without the HPOBench data
    stub = tmp_path / "stub_worker.py"
    stub.write_text(_STUB, encoding="utf-8")
    script = tmp_path / "short_run.py"
    script.write_text(_SHORT_RUN.format(root=str(PROJECT_ROOT), src=str(PROJECT_ROOT / "src"),
                                        tests=str(PROJECT_ROOT / "tests")), encoding="utf-8")
    run_dir = tmp_path / "run"
    planned = PlannedRun(
        run_id="short", phase="test", study_id="test", study_source_path=tmp_path, benchmark_id="hpobench-tabular",
        problem_id="lr-kc1", method_id="stub", budget=8, seed=1001, replicate=1, parameters={}, optimizer={},
        prior_path=None, result_directory=run_dir, command=("{python}", str(script), str(run_dir), str(stub)),
        blocked_reasons=())
    status = run_planned_run(planned)
    assert status["status"] == "failed" and status["failure_stage"] == "optimization_process"
    assert json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["status"] == "failed"
    assert "stub worker stopped after 4 of 8 queries" in (run_dir / "stderr.log").read_text(encoding="utf-8")
    assert not (run_dir / "evaluations.jsonl").exists()


# ---------------------------------------------------------------- coordinate legend
def _lr_kc1():
    from experiments.worker import _encode_mixed_candidates, _load_offline_problem, _log_scaled_names

    try:
        problem = _load_offline_problem("hpobench", {"model": "lr", "task": "kc1"})
    except FileNotFoundError:
        pytest.skip("HPOBench tabular data not available locally")
    run = {"benchmark_id": "hpobench-tabular", "problem_id": "lr-kc1", "parameters": {"model": "lr", "task": "kc1"}}
    native = tuple(problem.native_candidates)
    encoded = _encode_mixed_candidates(native, problem.parameter_names, _log_scaled_names(problem, "hpobench"))
    return problem, run, native, encoded


def test_hpo_baseline_text_carries_the_card_legend() -> None:
    from experiments.worker import _external_objective_meta

    problem, run, native, encoded = _lr_kc1()
    meta = _external_objective_meta(dict(run, optimizer={"baseline_task_legend": True}), problem, "hpobench", native, encoded)
    legend = meta["coordinate_legend"]
    assert legend.startswith("Search coordinates:")
    assert "x0 = alpha (L2 regularization strength; values are log-spaced)" in legend
    assert "x1 = eta0 (initial SGD learning rate; values are log-spaced)" in legend
    assert "from 1e-05 (x0 = 0) to 1 (x0 = 1), log scale" in legend
    assert meta["description"].endswith(legend)  # LGBO / BORA text
    assert meta["description"].startswith("Hyperparameter tuning of a logistic-regression classifier")
    assert meta["task_description"] == legend  # LLAMBO model slot
    assert meta["task_card"].endswith("hpobench-lr-kc1/kc1/llm-card/prompt_card.md")
    # No benchmark / dataset identity in the text.
    for leak in ("kc1", "HPOBench", "hpobench", "OpenML"):
        assert leak not in meta["description"]
    # The legend's mapping is the encoding the baseline receives.
    alphas = np.array([c["alpha"] for c in native])
    assert np.allclose(encoded[:, 0], np.log(alphas / alphas.min()) / np.log(alphas.max() / alphas.min()))

    # Default (no switch): no legend.
    plain = _external_objective_meta(dict(run, optimizer={}), problem, "hpobench", native, encoded)
    assert "coordinate_legend" not in plain and "Search coordinates" not in plain["description"]
    assert "task_description" not in plain


def test_hpo_legend_follows_the_misleading_hint() -> None:
    from experiments.worker import _external_objective_meta

    problem, run, native, encoded = _lr_kc1()
    meta = _external_objective_meta(dict(run, optimizer={"misleading_description": True, "baseline_task_legend": True}),
                                    problem, "hpobench", native, encoded)
    head, _, tail = meta["description"].partition("\n\n")
    assert "Prefer that region" in head and "x0=" in head  # task sentence + hint, unchanged
    assert tail == meta["coordinate_legend"]
    assert meta["target_description"] in head


def test_coordinate_legend_handles_irregular_and_categorical_axes() -> None:
    from experiments.worker import _coordinate_legend

    native = [{"wd": wd, "kind": k} for wd in (0.0, 1e-4, 1e-2) for k in ("a", "b")]
    wd_x = {0.0: 0.0, 1e-4: 0.5, 1e-2: 1.0}
    encoded = np.array([[wd_x[c["wd"]], float(c["kind"] == "a"), float(c["kind"] == "b")] for c in native])
    card = {"wd": {"name": "wd", "role": "weight decay"}, "kind": {"name": "kind", "meaning": "variant"}}
    legend = _coordinate_legend(native, encoded, ["wd", "kind"], card)
    assert "x0 = wd (weight decay)" in legend and "monotone non-uniform" in legend
    assert "0.0001 -> 0.5" in legend
    assert "x1..x2 one-hot encode kind (variant): x1 = 1 means a, x2 = 1 means b" in legend
    with pytest.raises(ValueError):
        _coordinate_legend(native, encoded, ["wd", "kind"], {"wd": card["wd"]})  # card lacks a dimension


def test_cnn_baselines_get_card_domain_and_legend() -> None:
    from datasets import cnn_leak
    from experiments.worker import _external_objective_meta

    grid = cnn_leak.enumerate_grid()
    problem = SimpleNamespace(native_candidates=tuple(grid), encoded_candidates=cnn_leak.encode_candidates(grid),
                              parameter_names=cnn_leak.PARAMETER_NAMES, maximize=True, target_name="val_accuracy")
    run = {"benchmark_id": "cnn-leak-of", "problem_id": "fashion-of", "parameters": {"dataset": "fashion-of"},
           "optimizer": {"baseline_task_legend": True}}
    meta = _external_objective_meta(run, problem, "cnn-leak", grid,
                                    problem.encoded_candidates)
    assert meta["description"].startswith("A small convolutional image classifier")
    assert "x3 = weight_decay (L2 weight decay coefficient)" in meta["description"]
    assert meta["task_description"] == meta["description"]
    assert "fashion" not in meta["description"].lower() and "mnist" not in meta["description"].lower()
    assert meta["dataset_context"] == {"n_classes": "roughly ten"}  # class count as the card states it


def test_legend_numbers_use_four_significant_digits() -> None:
    from experiments.worker import _fmt_native

    assert [_fmt_native(v) for v in (1024.0, 8567.0, 2 ** -10, 9.999999747378752e-06, 0.1)] == [
        "1024", "8567", "0.0009766", "1e-05", "0.1"]


LEGEND_STUDIES = ("hpo/llambo", "hpo/llambo-misleading", "hpo/llambo-mode1", "hpo/llambo-mode1-misleading",
                  "hpo/lgbo", "hpo/lgbo-misleading", "hpo/bora", "hpo/bora-misleading", "hpo/lgbo-official",
                  "hpo/lgbo-official-misleading", "cnn-leak/cnn-leak-of-llambo-ws", "cnn-leak/cnn-leak-of-lgbo",
                  "cnn-leak/cnn-leak-of-bora")


def test_paper_manifests_turn_the_legend_on() -> None:
    import yaml

    for study in LEGEND_STUDIES:
        path = PROJECT_ROOT / "experiments" / "studies" / study / "study.yaml"
        optimizer = yaml.safe_load(path.read_text(encoding="utf-8"))["design"]["optimizer"]
        assert optimizer["baseline_task_legend"] is True, path
        assert optimizer.get("bridge_policy", "charge") == "charge", path


def test_solar_baselines_get_the_card_legend() -> None:
    from datasets.solar import resolve_solar_instance

    externals = _load_script("solar_externals")
    spec = resolve_solar_instance("MAXEFF_RE")
    meta = externals.solar_objective_meta(spec, legend=True)
    legend = meta["coordinate_legend"]
    assert meta["description"].endswith(legend) and meta["task_description"].endswith(legend)
    assert "x0 = receiver_aperture_height_m (vertical size of the rectangular cavity opening" in legend
    assert "x3 = receiver_tube_count" in legend and "range 1 (x3 = 0) to 8567 (x3 = 1)" in legend
    assert legend.count("\n- x") == len(spec.variables)
    # the legend's mapping is the encoding the external-baseline runner gives the baselines
    pool = [{v.name: v.sampling_upper for v in spec.variables}, {v.name: v.lower for v in spec.variables}]
    assert np.allclose(externals.encode(pool, spec), [[1.0] * 7, [0.0] * 7])
    plain = externals.solar_objective_meta(spec)  # default: no legend
    assert "coordinate_legend" not in plain and "Search coordinates" not in plain["description"]


def test_solar_bora_charges_repeats_and_flags_them() -> None:
    # Every BORA-SOLAR proposal is charged; repeats are flagged and counted (bookkeeping).
    text = (PROJECT_ROOT / "scripts" / "bora_solar_cached.py").read_text(encoding="utf-8")
    assert "bora.maximize(n_init_points=args.n_init, n_iter=args.budget - args.n_init)" in text
    assert '"repeat": repeat' in text and '"repeat_count"' in text


def test_bridge_accounting_uses_the_run_budget(tmp_path: Path) -> None:
    opt = _stub_bridge(tmp_path, budget=8)
    try:
        run_ask_tell_loop(opt, lambda c: float(c["x"]), budget=6, n_init=1)
    finally:
        opt.close()
    acc = opt.bridge_accounting()
    assert (acc["budget"], acc["queries_used"], acc["proposals_exhausted_at"]) == (6, 4, 5)


def test_solar_externals_raise_when_the_worker_proposes_nothing() -> None:
    text = (PROJECT_ROOT / "scripts" / "solar_externals.py").read_text(encoding="utf-8")
    assert "worker stopped before proposing any query" in text
    # a short run fails like the study worker; nothing is padded
    assert "require_full_budget(len(records), args.budget, args.baseline)" in text
