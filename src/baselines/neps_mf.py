"""PriorBand / PriMO (official NePS) on the HPOBench tables, native multi-fidelity, cost-charged.

The NePS optimizer runs in its own environment (``workers/neps_worker.py``, ask-and-tell); this
module is the parent side and owns EVALUATE, the cost ledger and the stop rule:

* Search space: each hyperparameter is a NePS ``Integer(0, n - 1)`` over the index of its
  sorted grid values (a bijection onto the table rows); the fidelity is the family's
  table fidelity (``datasets.hpobench_mf.MF_FIDELITY``) with its true bounds. NePS derives
  its rungs from (bounds, eta); every rung must coincide with one table fidelity
  (checked, rel. tol 1e-6) and every table fidelity must be a rung.
* Prior: the point prior (``neps-point-prior-v1``: one grid value + confidence per
  hyperparameter) is passed as grid indices + ``low``/``medium``/``high``.
* Cost: an evaluation at fidelity z costs z / z_max full-fidelity units. A promotion is a
  fresh evaluation (no continuation discount). A proposal is evaluated only if it fits in
  the remaining budget; the first proposal that would exceed the budget is refused (not
  evaluated, not charged, logged as ``refused``) and the run ends there, so the cost used
  is <= budget + 1e-9.
* Repeats: a proposal of an already evaluated (configuration, fidelity) is charged and
  served from the cache (flag ``repeat``); nothing is replaced. A run also stops after
  ``max_proposals`` proposals (a safety cap that charged runs cannot reach).
* Logging (``mf_evaluations.jsonl``): every evaluation with its trial id, table row, rung,
  fidelity, cost, cumulative cost, observed value, the TRUE full-fidelity value of the
  configuration and two incumbents:
  (a) the best full-fidelity evaluation so far (undefined before the first one), used in
      the paper;
  (b) the recommended incumbent = best observed value at any fidelity, ties -> the first
      one seen, scored by its true full-fidelity value.
  ``cost_curves.json`` samples both on the integer cost grid 1..budget; ``metrics.json``
  ``final_*`` fields report (a).
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER = _REPO_ROOT / "src" / "baselines" / "workers" / "neps_worker.py"
PRIOR_FORMAT = "neps-point-prior-v1"
CONFIDENCE_LEVELS = ("low", "medium", "high")
ALGORITHMS = ("priorband", "primo")
COST_TOL = 1e-9
DEFAULT_MAX_PROPOSALS = 2000  # charged runs stay below ~1340 evaluations (16 units at the cheapest rung)
EVALUATIONS_FILE = "mf_evaluations.jsonl"


class NePSWorkerError(RuntimeError):
    """The NePS worker failed or violated the protocol."""


# ---------------------------------------------------------------------------
# search space / prior / fidelity mapping
# ---------------------------------------------------------------------------


def grid_dimensions(
    parameter_names: Sequence[str], candidates: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Per parameter: its sorted unique grid values (index order seen by NePS)."""

    dims = []
    for name in parameter_names:
        values = sorted({candidate[name] for candidate in candidates})
        if len(values) < 2:
            raise ValueError(f"parameter {name!r} has fewer than two grid values")
        dims.append({"name": name, "n": len(values), "values": values})
    return dims


def index_to_row(
    dims: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]]
) -> dict[tuple[int, ...], int]:
    """Map each grid-index tuple to its table row; the grid must be complete (a product)."""

    position = [{value: i for i, value in enumerate(dim["values"])} for dim in dims]
    mapping: dict[tuple[int, ...], int] = {}
    for row, candidate in enumerate(candidates):
        key = tuple(position[d][candidate[dim["name"]]] for d, dim in enumerate(dims))
        if key in mapping:
            raise ValueError(f"duplicate table row for grid indices {key}")
        mapping[key] = row
    expected = int(np.prod([dim["n"] for dim in dims]))
    if len(mapping) != expected:
        raise ValueError(f"the table is not a full grid ({len(mapping)} rows, product {expected})")
    return mapping


def prior_for_neps(document: Mapping[str, Any], dims: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Grid indices + confidence levels from a ``neps-point-prior-v1`` artifact (validated)."""

    if document.get("artifact_format") != PRIOR_FORMAT:
        raise ValueError(f"prior artifact_format must be {PRIOR_FORMAT!r}")
    parameters = document.get("parameters") or {}
    names = [dim["name"] for dim in dims]
    if set(parameters) != set(names):
        raise ValueError(f"prior parameters {sorted(parameters)} != grid parameters {sorted(names)}")
    indices, confidence = {}, {}
    for dim in dims:
        entry = parameters[dim["name"]]
        if entry.get("confidence") not in CONFIDENCE_LEVELS:
            raise ValueError(f"{dim['name']}: invalid confidence {entry.get('confidence')!r}")
        matches = [i for i, v in enumerate(dim["values"]) if v == entry["value"]]
        if len(matches) != 1 or matches[0] != int(entry["index"]) or int(entry["n_values"]) != dim["n"]:
            raise ValueError(f"{dim['name']}: prior value {entry['value']!r} / index {entry['index']} "
                             "does not match the task grid")
        indices[dim["name"]] = matches[0]
        confidence[dim["name"]] = entry["confidence"]
    return {"indices": indices, "confidence": confidence}


def fidelity_spec(declared_values: Sequence[float], name: str) -> dict[str, Any]:
    """NePS fidelity bounds = the declared (metadata) table fidelity range."""

    low, high = float(min(declared_values)), float(max(declared_values))
    integral = all(float(v).is_integer() for v in declared_values)
    return {"name": name, "type": "int" if integral else "float",
            "lower": int(low) if integral else low, "upper": int(high) if integral else high}


def map_rungs(
    rung_to_fidelity: Mapping[int, float], declared_values: Sequence[float]
) -> dict[int, int]:
    """NePS rung -> table fidelity level (index into the ascending table fidelities)."""

    declared = sorted(float(v) for v in declared_values)
    mapping: dict[int, int] = {}
    for rung, fidelity in rung_to_fidelity.items():
        hits = [i for i, v in enumerate(declared) if np.isclose(float(fidelity), v, rtol=1e-6, atol=0.0)]
        if len(hits) != 1:
            raise ValueError(f"NePS rung {rung} fidelity {fidelity} matches no table fidelity {declared}")
        mapping[int(rung)] = hits[0]
    if sorted(mapping.values()) != list(range(len(declared))):
        raise ValueError(f"NePS rungs {dict(rung_to_fidelity)} do not cover the table fidelities {declared}")
    return mapping


# ---------------------------------------------------------------------------
# cost ledger
# ---------------------------------------------------------------------------

_META_FIRST = ("trial_id", "previous_trial_id", "phase")


class MFLedger:
    """Charges evaluations, enforces the budget, serves repeats from the cache, logs records."""

    def __init__(
        self,
        values: np.ndarray,
        costs: Sequence[float],
        ground_truth: np.ndarray,
        budget: float,
        *,
        fidelity_values: Sequence[float] | None = None,
        candidates: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self.values = np.asarray(values, dtype=float)
        self.costs = [float(c) for c in costs]
        self.ground_truth = np.asarray(ground_truth, dtype=float)
        if self.values.shape != (len(self.ground_truth), len(self.costs)):
            raise ValueError("values must be [n_candidates, n_fidelities]")
        self.budget = float(budget)
        self.fidelity_values = list(fidelity_values) if fidelity_values is not None else None
        self.candidates = candidates
        self.max_rung = len(self.costs) - 1
        self.cumulative_cost = 0.0
        self.records: list[dict[str, Any]] = []
        self.refused: dict[str, Any] | None = None
        self._cache: dict[tuple[int, int], float] = {}
        self._recommended: tuple[int, float] | None = None  # (row, observed); (b)
        self._best_full: tuple[int, float] | None = None  # (a)

    @property
    def remaining(self) -> float:
        return self.budget - self.cumulative_cost

    def fits(self, rung: int) -> bool:
        return self.cumulative_cost + self.costs[rung] <= self.budget + COST_TOL

    def charge(self, row: int, rung: int, **meta: Any) -> dict[str, Any] | None:
        """Evaluate (or serve from cache) and charge; ``None`` if it does not fit the budget."""

        row, rung = int(row), int(rung)
        cost = self.costs[rung]
        if not self.fits(rung):
            self.refused = {"candidate_index": row, "rung": rung, "cost": cost,
                            "cumulative_cost_before": self.cumulative_cost, "remaining": self.remaining,
                            **{k: v for k, v in meta.items() if k in _META_FIRST}}
            return None
        key = (row, rung)
        repeat = key in self._cache
        observed = self._cache[key] if repeat else float(self.values[row, rung])
        self._cache[key] = observed
        self.cumulative_cost += cost
        if self._recommended is None or observed > self._recommended[1]:  # strict: ties -> first seen
            self._recommended = (row, observed)
        if rung == self.max_rung and (self._best_full is None or observed > self._best_full[1]):
            self._best_full = (row, observed)
        record = {
            "query": len(self.records) + 1,
            **{k: v for k, v in meta.items() if k in _META_FIRST},
            "candidate_index": row,
            "candidate": None if self.candidates is None else dict(self.candidates[row]),
            "rung": rung,
            "fidelity": None if self.fidelity_values is None else self.fidelity_values[rung],
            "full_fidelity": rung == self.max_rung,
            "cost": cost,
            "cumulative_cost": self.cumulative_cost,
            "objective": observed,
            "true_full_fidelity_value": float(self.ground_truth[row]),
            "repeat": repeat,
            "best_full_fidelity_index": None if self._best_full is None else self._best_full[0],
            "best_full_fidelity_true_value": (None if self._best_full is None
                                              else float(self.ground_truth[self._best_full[0]])),
            "recommended_index": self._recommended[0],
            "recommended_observed": self._recommended[1],
            "recommended_true_value": float(self.ground_truth[self._recommended[0]]),
            **{k: v for k, v in meta.items() if k not in _META_FIRST},
        }
        self.records.append(record)
        return record

    def accounting(self) -> dict[str, Any]:
        rungs = Counter(r["rung"] for r in self.records)
        repeats = [r for r in self.records if r["repeat"]]
        repeat_units = float(sum(r["cost"] for r in repeats))
        return {
            "budget": self.budget,
            "cost_used": self.cumulative_cost,
            "unused_budget": self.remaining,
            "n_evaluations": len(self.records),
            "n_repeats": len(repeats),
            "repeat_units": repeat_units,
            "repeat_share_of_budget": repeat_units / self.budget if self.budget else None,
            "n_unique_configurations": len({r["candidate_index"] for r in self.records}),
            "n_promotions": sum(r.get("previous_trial_id") is not None for r in self.records),
            "per_rung": {
                str(rung): {"fidelity": None if self.fidelity_values is None else self.fidelity_values[rung],
                            "cost_each": self.costs[rung], "n": rungs.get(rung, 0),
                            "n_repeats": sum(r["repeat"] for r in self.records if r["rung"] == rung)}
                for rung in range(len(self.costs))
            },
            "refused": self.refused,
        }


def curve_from_records(records: Sequence[Mapping[str, Any]], budget: int, key: str) -> list[float | None]:
    """``key`` of the last record with cumulative cost <= c, for c = 1..budget (``None`` before)."""

    out: list[float | None] = []
    for c in range(1, int(budget) + 1):
        done = [r for r in records if r["cumulative_cost"] <= c + COST_TOL]
        out.append(None if not done else done[-1][key])
    return out


def cost_curves(records: Sequence[Mapping[str, Any]], budget: int, optimum: float) -> dict[str, Any]:
    """Both incumbent definitions on the integer cost grid 1..budget. ``None`` where undefined."""

    grid = list(range(1, int(budget) + 1))
    best_full = curve_from_records(records, budget, "best_full_fidelity_true_value")
    recommended = curve_from_records(records, budget, "recommended_true_value")

    def regret(values):
        return [None if v is None else 1.0 - v / optimum for v in values]

    return {
        "artifact_format": "neps-mf-cost-curves-v2",
        "cost_grid": grid,
        "optimum": optimum,
        "n_evaluations": [sum(r["cumulative_cost"] <= c + COST_TOL for r in records) for c in grid],
        "best_full_fidelity": {"true_value": best_full, "normalized_regret": regret(best_full),
                               "definition": "(a) best full-fidelity evaluation so far"},
        "recommended": {"true_value": recommended, "normalized_regret": regret(recommended),
                        "definition": "(b) true full-fidelity value of the recommended incumbent "
                                      "(best observed value at any fidelity, ties -> first seen)"},
    }


# ---------------------------------------------------------------------------
# worker process
# ---------------------------------------------------------------------------


def worker_command() -> list[str]:
    python = os.environ.get("NEPS_PYTHON")
    if python:
        return [python, str(_WORKER)]
    conda = os.environ.get("CONDA_EXE") or "conda"
    return [conda, "run", "--no-capture-output", "-n", os.environ.get("NEPS_ENV", "neps"), "python", str(_WORKER)]


class NePSWorkerProcess:
    """Line-JSON pipe to ``neps_worker.py``; stderr and log messages go to ``log_path``."""

    def __init__(self, spec: Mapping[str, Any], log_path: str | os.PathLike | None = None) -> None:
        self._log = open(log_path, "a", encoding="utf-8") if log_path is not None else None
        self._lock = threading.Lock()
        self._stderr: deque[str] = deque(maxlen=400)
        env = dict(os.environ)
        env.update({"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1", "PYTHONWARNINGS": "ignore"})
        self.proc = subprocess.Popen(worker_command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()
        self.send(dict(spec))

    def _write_log(self, text: str) -> None:
        self._stderr.append(text)
        if self._log is not None:
            with self._lock:
                self._log.write(text if text.endswith("\n") else text + "\n")
                self._log.flush()

    def _drain(self) -> None:
        try:
            for line in self.proc.stderr:  # type: ignore[union-attr]
                self._write_log(line)
        except (OSError, ValueError):
            return

    def tail(self) -> str:
        return "".join(self._stderr)[-4000:]

    def send(self, obj: Mapping[str, Any]) -> None:
        try:
            self.proc.stdin.write(json.dumps(obj) + "\n")  # type: ignore[union-attr]
            self.proc.stdin.flush()  # type: ignore[union-attr]
        except (BrokenPipeError, ValueError) as exc:
            raise NePSWorkerError(f"worker pipe closed: {self.tail()}") from exc

    def read(self) -> dict[str, Any]:
        """Next protocol message (``log`` messages are written to the log and skipped)."""

        for _ in range(10_000):
            line = self.proc.stdout.readline()  # type: ignore[union-attr]
            if not line:
                raise NePSWorkerError(f"worker exited unexpectedly: {self.tail()}")
            stripped = line.strip()
            if not stripped.startswith("{"):
                continue
            try:
                message = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if message.get("type") == "log":
                self._write_log(f"[log] {message.get('message', '')}")
                continue
            return message
        raise NePSWorkerError(f"no protocol message from worker: {self.tail()}")

    def close(self) -> None:
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            self.proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            self.proc.kill()
        self._thread.join(timeout=2)
        if self._log is not None:
            self._log.close()
            self._log = None


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------


def _phase(algorithm: str, trial_id: str, max_neps_rung: int, previous_trial_id: str | None = None) -> str:
    if algorithm == "primo":
        return "initial_design" if "_rung_" in trial_id else "bo"
    # sample_prior_first="highest_fidelity": NePS evaluates the prior mode first, as trial 1 at the top rung
    if trial_id == f"1_rung_{max_neps_rung}" and previous_trial_id is None:
        return "prior_mode"
    return "hyperband"


def run_neps_mf(
    mf_problem: Any,
    *,
    algorithm: str,
    prior: Mapping[str, Any] | None,
    seed: int,
    budget: float,
    optimizer_kwargs: Mapping[str, Any] | None = None,
    max_proposals: int = DEFAULT_MAX_PROPOSALS,
    log_path: str | os.PathLike | None = None,
    torch_threads: int = 1,
) -> dict[str, Any]:
    """Drive one NePS run to the cost budget; returns records, accounting and worker info."""

    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}")
    candidates = mf_problem.native_candidates
    dims = grid_dimensions(mf_problem.parameter_names, candidates)
    rows = index_to_row(dims, candidates)
    declared = mf_problem.metadata.get("declared_fidelity_values") or list(mf_problem.fidelity_values)
    fid = fidelity_spec(declared, mf_problem.fidelity_name)
    spec = {
        "algorithm": algorithm,
        "seed": int(seed),
        "dimensions": [{"name": d["name"], "n": d["n"]} for d in dims],
        "fidelity": fid,
        "prior": None if prior is None else dict(prior),
        "optimizer_kwargs": dict(optimizer_kwargs or {}),
        "torch_threads": int(torch_threads),
    }
    ledger = MFLedger(mf_problem.values, mf_problem.costs, mf_problem.ground_truth, budget,
                      fidelity_values=list(mf_problem.fidelity_values), candidates=candidates)
    started = time.perf_counter()
    worker = NePSWorkerProcess(spec, log_path)
    stop_reason = None
    n_proposals = 0
    try:
        info = worker.read()
        if info.get("type") != "info":
            raise NePSWorkerError(f"expected an info message, got {info}")
        neps_rungs = {int(k): float(v) for k, v in info["rung_to_fidelity"].items()}
        rung_to_level = map_rungs(neps_rungs, declared)
        while True:
            message = worker.read()
            if message.get("type") == "done":
                stop_reason = "worker_done"
                break
            if message.get("type") != "eval":
                raise NePSWorkerError(f"unexpected worker message: {message}")
            key = tuple(int(message["indices"][d["name"]]) for d in dims)
            if key not in rows:
                raise NePSWorkerError(f"proposal {message['indices']} is not a table row")
            fid_value = message["fidelity"]
            levels = [level for rung, level in rung_to_level.items()
                      if np.isclose(neps_rungs[rung], float(fid_value), rtol=1e-9, atol=0.0)]
            if len(levels) != 1:
                raise NePSWorkerError(f"proposal fidelity {fid_value} is not a NePS rung fidelity")
            if n_proposals >= int(max_proposals):
                stop_reason = "proposal_cap"
                record = None
            else:
                n_proposals += 1
                record = ledger.charge(
                    rows[key], levels[0],
                    trial_id=message["trial_id"], previous_trial_id=message.get("previous_trial_id"),
                    phase=_phase(algorithm, str(message["trial_id"]), max(neps_rungs),
                                 message.get("previous_trial_id")),
                    indices=dict(message["indices"]),
                    fidelity_requested=fid_value, ask_seconds=float(message.get("ask_seconds", 0.0)),
                )
                if record is None:
                    stop_reason = "budget"
            if record is None:
                worker.send({"stop": True})
                done = worker.read()
                if done.get("type") != "done":
                    raise NePSWorkerError(f"expected done after stop, got {done}")
                break
            worker.send({"value": record["objective"]})
    finally:
        worker.close()
    if worker.proc.returncode not in (0, None):
        raise NePSWorkerError(f"worker exited with code {worker.proc.returncode}: {worker.tail()}")
    accounting = ledger.accounting()
    accounting.update({
        "stop_reason": stop_reason,
        "n_proposals": n_proposals,
        "max_proposals": int(max_proposals),
        "algorithm": algorithm,
        "seed": int(seed),
        "wall_time_seconds": time.perf_counter() - started,
        "neps_version": info.get("neps_version"),
        "neps_info": info,
        "rung_to_table_level": {str(k): v for k, v in rung_to_level.items()},
        "fidelity": {"name": mf_problem.fidelity_name, "table_values": list(mf_problem.fidelity_values),
                     "declared_values": list(declared), "neps_bounds": [fid["lower"], fid["upper"]],
                     "fixed_fidelities": dict(mf_problem.fixed_fidelities)},
        "prior": None if prior is None else dict(prior),
        "optimizer_kwargs": dict(optimizer_kwargs or {}),
    })
    return {"records": ledger.records, "accounting": accounting, "spec": spec}


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def write_mf_results(
    directory: Path,
    *,
    method_id: str,
    label: str,
    records: Sequence[Mapping[str, Any]],
    accounting: Mapping[str, Any],
    ground_truth: np.ndarray,
    candidates: Sequence[Mapping[str, Any]],
    budget: int,
) -> dict[str, Any]:
    """mf_evaluations.jsonl, cost_curves.json, mf_accounting.json, metrics.json, best_candidate.json.

    No ``evaluations.jsonl`` is written: its readers assume one full-fidelity query per line."""

    directory.mkdir(parents=True, exist_ok=True)
    if not records:
        raise RuntimeError("no evaluation was charged")
    optimum = float(np.max(ground_truth))
    curves = cost_curves(records, budget, optimum)
    last = records[-1]
    best_full = last["best_full_fidelity_index"]
    recommended = int(last["recommended_index"])
    a_value = None if best_full is None else float(ground_truth[int(best_full)])
    metric_record = {
        "label": label,
        "cost_model": "fidelity-weighted: z / z_max per evaluation, budget in full-fidelity units",
        "budget": budget,
        "cost_used": float(accounting["cost_used"]),
        "n_evaluations": len(records),
        "n_repeats": int(accounting["n_repeats"]),
        "repeat_units": float(accounting["repeat_units"]),
        "best_index": None if best_full is None else int(best_full),
        "best_configuration": None if best_full is None else dict(candidates[int(best_full)]),
        "best_observed_value": a_value,
        "final_simple_regret": None if a_value is None else optimum - a_value,
        "final_normalized_regret": None if a_value is None else 1.0 - a_value / optimum,
        "recommended_index": recommended,
        "recommended_true_value": float(ground_truth[recommended]),
        "evaluated_indices": [int(r["candidate_index"]) for r in records],
        "evaluated_rungs": [int(r["rung"]) for r in records],
    }
    with (directory / EVALUATIONS_FILE).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
    _write_json(directory / "cost_curves.json", curves)
    _write_json(directory / "mf_accounting.json", accounting)
    _write_json(directory / "metrics.json", {"artifact_format": "llm-bo-study-metrics-v1",
                                             "methods": {method_id: metric_record}})
    _write_json(directory / "best_candidate.json", metric_record["best_configuration"])
    return metric_record


def method_settings(method: Mapping[str, Any], study_optimizer: Mapping[str, Any]) -> dict[str, Any]:
    """Effective NePS settings of one run: the method entry defines the arm; a study may restate its
    ``neps_kwargs`` (they must then agree) but cannot change them."""

    method_kwargs = dict(method.get("neps_kwargs") or {})
    study_kwargs = study_optimizer.get("neps_kwargs")
    if study_kwargs is not None and dict(study_kwargs) != method_kwargs:
        raise ValueError(f"study neps_kwargs {dict(study_kwargs)} disagree with method "
                         f"{method.get('method_id')!r} neps_kwargs {method_kwargs}")
    return {
        "algorithm": str(method["neps_algorithm"]),
        "neps_kwargs": method_kwargs,
        "max_proposals": int(method.get("max_proposals", DEFAULT_MAX_PROPOSALS)),
        "torch_threads": int(study_optimizer.get("torch_threads", 1)),
    }


def run_neps_mf_study_run(
    directory: Path,
    run: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    method: Mapping[str, Any],
    problem: Any,
    budget: int,
) -> dict[str, Any]:
    """Study-worker entry for ``strategy: external-mf`` methods (HPOBench only)."""

    from datasets.hpobench_mf import load_hpobench_mf_problem

    if str(benchmark.get("adapter")) != "hpobench":
        raise ValueError("external-mf methods run on the HPOBench tables only")
    params = run["parameters"]
    settings = method_settings(method, dict(run.get("optimizer", {})))
    started = time.perf_counter()
    mf = load_hpobench_mf_problem(model=str(params["model"]), task=str(params["task"]))
    load_seconds = time.perf_counter() - started
    # the multi-fidelity view must be the full-fidelity problem at z_max, row for row
    if tuple(mf.native_candidates) != tuple(problem.native_candidates):
        raise ValueError("multi-fidelity candidates differ from the full-fidelity problem")
    if not np.array_equal(mf.ground_truth, np.asarray(problem.ground_truth, dtype=float)):
        raise ValueError("multi-fidelity z_max values differ from the full-fidelity ground truth")
    prior = None
    if bool(method.get("requires_prior", False)):
        if run.get("prior_path") is None:
            raise ValueError(f"{method['method_id']} requires a prior_path")
        dims = grid_dimensions(mf.parameter_names, mf.native_candidates)
        prior = prior_for_neps(json.loads(Path(run["prior_path"]).read_text(encoding="utf-8")), dims)
    result = run_neps_mf(
        mf, algorithm=settings["algorithm"], prior=prior, seed=int(run["seed"]), budget=float(budget),
        optimizer_kwargs=settings["neps_kwargs"], max_proposals=settings["max_proposals"],
        log_path=directory / "worker_stderr.log", torch_threads=settings["torch_threads"],
    )
    accounting = {**result["accounting"], "table_load_seconds": load_seconds, "method_settings": settings}
    metric_record = write_mf_results(
        directory, method_id=str(run["method_id"]), label=str(method["label"]),
        records=result["records"], accounting=accounting, ground_truth=mf.ground_truth,
        candidates=mf.native_candidates, budget=int(budget),
    )
    return {
        "strategy": "external-mf",
        "baseline": settings["algorithm"],
        "metrics": metric_record,
        "mf_accounting": {k: v for k, v in accounting.items() if k not in ("neps_info",)},
        "outputs": {
            "mf_evaluations": str(directory / EVALUATIONS_FILE),
            "cost_curves": str(directory / "cost_curves.json"),
            "mf_accounting": str(directory / "mf_accounting.json"),
            "metrics": str(directory / "metrics.json"),
            "best_candidate": str(directory / "best_candidate.json"),
            "worker_log": str(directory / "worker_stderr.log"),
        },
    }


__all__ = [
    "EVALUATIONS_FILE",
    "MFLedger",
    "NePSWorkerError",
    "cost_curves",
    "curve_from_records",
    "fidelity_spec",
    "grid_dimensions",
    "index_to_row",
    "map_rungs",
    "method_settings",
    "prior_for_neps",
    "run_neps_mf",
    "run_neps_mf_study_run",
    "write_mf_results",
]
