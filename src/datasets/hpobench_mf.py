"""Multi-fidelity view of the HPOBench tabular tasks (for the multi-fidelity baselines).

``load_hpobench_problem`` reads one table at its MAXIMUM fidelity only. The official
tables also hold lower fidelities; this module exposes them for multi-fidelity
optimizers (PriorBand, PriMO) with the same objective and the same seed averaging:
the mean validation accuracy over the benchmark seeds available for each
(configuration, fidelity) cell.

One fidelity dimension varies per family; every other fidelity column is fixed at its
maximum (in the tables those columns are constant anyway):

====== ============== ============================== =========================
family fidelity        table values (= rungs, eta 3)  cost z / z_max
====== ============== ============================== =========================
lr     iter            12, 37, 111, 333, 1000         .012 .037 .111 .333 1
nn     iter            3, 9, 27, 81, 243              1/81 1/27 1/9 1/3 1
rf     n_estimators    18, 56, 170, 512               .0352 .1094 .3320 1
xgb    n_estimators    74, 222, 666, 2000             .037 .111 .333 1
svm    subsample       1/9, 1/3, 1                    1/9 1/3 1
====== ============== ============================== =========================

The candidate order is the one of ``load_hpobench_problem`` and the maximum-fidelity
column reproduces its ``ground_truth`` bit for bit (tests/test_neps_mf.py).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .hpobench import (
    DATASET_ID,
    HPOBENCH_DIRECTORY,
    TASKS,
    _configuration_space_document,
    _configuration_space_names,
    _maximum_fidelity_values,
    _python_scalar,
    _validation_accuracy,
    hpobench_task_path,
)

#: The one fidelity dimension used per family (the others stay at their maximum).
MF_FIDELITY = {
    "lr": "iter",
    "nn": "iter",
    "rf": "n_estimators",
    "svm": "subsample",
    "xgb": "n_estimators",
}


@dataclass(frozen=True)
class HPOBenchMFProblem:
    """One HPOBench model/task with its seed-averaged values at every fidelity."""

    dataset_id: str
    model: str
    task_name: str
    task_id: int
    parameter_names: tuple[str, ...]
    native_candidates: tuple[dict[str, Any], ...]
    fidelity_name: str
    #: table values of the varying fidelity, ascending (rung r <-> fidelity_values[r])
    fidelity_values: tuple[float, ...]
    #: the fixed (maximum) value of every other fidelity column
    fixed_fidelities: dict[str, Any]
    #: [n_candidates, n_fidelities] mean validation accuracy
    values: np.ndarray
    #: [n_candidates, n_fidelities] number of seeds averaged in each cell
    seed_counts: np.ndarray
    metadata: dict[str, Any]
    maximize: bool = True

    @property
    def n_candidates(self) -> int:
        return len(self.native_candidates)

    @property
    def n_fidelities(self) -> int:
        return len(self.fidelity_values)

    @property
    def max_rung(self) -> int:
        return self.n_fidelities - 1

    @property
    def ground_truth(self) -> np.ndarray:
        """Full-fidelity values (the column of the maximum fidelity)."""

        return self.values[:, -1]

    @property
    def costs(self) -> np.ndarray:
        """Cost of one evaluation per rung, in full-fidelity units (z / z_max).

        Uses the declared (metadata) fidelity values: svm stores subsample in float32
        (0.11111111194...), whose z / z_max would overcharge 1/9 and 1/3 by ~1e-8."""

        declared = self.metadata.get("declared_fidelity_values")
        z = np.asarray(declared if declared else self.fidelity_values, dtype=float)
        return z / z[-1]

    def value(self, index: int, rung: int) -> float:
        return float(self.values[int(index), int(rung)])


def _fidelity_sequence(serialized_space: Any, name: str) -> list[float]:
    space = _configuration_space_document(serialized_space, "fidelity")
    for hyperparameter in space.get("hyperparameters", []):
        if isinstance(hyperparameter, dict) and hyperparameter.get("name") == name:
            sequence = hyperparameter.get("sequence", hyperparameter.get("choices"))
            if not isinstance(sequence, list) or len(sequence) < 2:
                raise ValueError(f"fidelity {name!r} has fewer than two discrete values")
            return sorted(float(value) for value in sequence)
    raise ValueError(f"fidelity {name!r} is not in the z_discrete ConfigSpace")


def load_hpobench_mf_problem(
    *,
    model: str,
    task: str,
    benchmark_directory: str | Path = HPOBENCH_DIRECTORY,
) -> HPOBenchMFProblem:
    """Load every fidelity of one table; same objective and averaging as the full loader."""

    if model not in MF_FIDELITY:
        raise ValueError(f"no multi-fidelity mapping for HPOBench model {model!r}")
    parquet_path, metadata_path = hpobench_task_path(
        model, task, benchmark_directory=benchmark_directory
    )
    missing = [str(path) for path in (parquet_path, metadata_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"HPOBench task files are missing: {missing}")
    import pandas as pd

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config_spaces = metadata["config_spaces"]
    parameter_names = _configuration_space_names(config_spaces["x_discrete"], "configuration")
    fidelity_names = _configuration_space_names(config_spaces["z_discrete"], "fidelity")
    full_fidelity = _maximum_fidelity_values(config_spaces["z_discrete"])
    fidelity_name = MF_FIDELITY[model]
    if fidelity_name not in fidelity_names:
        raise ValueError(f"{model}: fidelity {fidelity_name!r} not in {fidelity_names}")
    declared = _fidelity_sequence(config_spaces["z_discrete"], fidelity_name)

    table = pd.read_parquet(parquet_path)
    required = {*parameter_names, *fidelity_names, "seed", "result"}
    absent = required - set(table.columns)
    if absent:
        raise ValueError(f"HPOBench table is missing columns {sorted(absent)}")

    # every other fidelity column is held at its maximum (exact match, as the full loader)
    keep = np.ones(len(table), dtype=bool)
    fixed = {name: value for name, value in full_fidelity.items() if name != fidelity_name}
    for name, value in fixed.items():
        keep &= (table[name] == value).to_numpy()
    table = table.loc[keep]

    # the table's own fidelity values (dtype as stored), matched to the declared sequence
    stored = sorted({_python_scalar(value) for value in table[fidelity_name].unique()})
    if len(stored) != len(declared) or not np.allclose(stored, declared, rtol=1e-6, atol=0.0):
        raise ValueError(
            f"{model}-{task}: table fidelity values {stored} do not match the declared "
            f"sequence {declared}"
        )
    if stored[-1] != full_fidelity[fidelity_name]:
        raise ValueError(f"{model}-{task}: maximum table fidelity {stored[-1]} != "
                         f"{full_fidelity[fidelity_name]}")

    columns = [*parameter_names, "seed", "result"]
    per_fidelity: list[dict[tuple[Any, ...], list[float]]] = []
    for value in stored:
        rows = table.loc[(table[fidelity_name] == value).to_numpy(), columns]
        scores: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        seeds: dict[tuple[Any, ...], set[Any]] = defaultdict(set)
        for row in rows.itertuples(index=False, name=None):
            key = tuple(_python_scalar(item) for item in row[: len(parameter_names)])
            seed = _python_scalar(row[len(parameter_names)])
            if seed in seeds[key]:
                raise ValueError(f"duplicate row for {key} at {fidelity_name}={value}, seed {seed}")
            seeds[key].add(seed)
            scores[key].append(_validation_accuracy(row[-1]))
        per_fidelity.append(scores)

    ordered_keys = sorted(per_fidelity[-1], key=lambda values: tuple(map(repr, values)))
    for level, scores in zip(stored, per_fidelity):
        if set(scores) != set(ordered_keys):
            raise ValueError(f"{model}-{task}: the configuration grid at {fidelity_name}={level} "
                             "differs from the maximum-fidelity grid")
    values = np.asarray(
        [[float(np.mean(scores[key])) for scores in per_fidelity] for key in ordered_keys],
        dtype=float,
    )
    counts = np.asarray(
        [[len(scores[key]) for scores in per_fidelity] for key in ordered_keys], dtype=int
    )
    native_candidates = tuple(
        {name: _python_scalar(value) for name, value in zip(parameter_names, key)}
        for key in ordered_keys
    )
    return HPOBenchMFProblem(
        dataset_id=DATASET_ID,
        model=model,
        task_name=task,
        task_id=TASKS[task],
        parameter_names=parameter_names,
        native_candidates=native_candidates,
        fidelity_name=fidelity_name,
        fidelity_values=tuple(float(value) for value in stored),
        fixed_fidelities=fixed,
        values=values,
        seed_counts=counts,
        metadata={
            "source": "HPOBench tabular ML benchmark (all fidelities)",
            "parquet_path": str(parquet_path),
            "metadata_path": str(metadata_path),
            "fidelity_name": fidelity_name,
            "fidelity_values": [float(value) for value in stored],
            "declared_fidelity_values": declared,
            "fixed_fidelities": fixed,
            "objective": "mean validation accuracy across available benchmark seeds",
        },
        maximize=True,
    )


__all__ = ["MF_FIDELITY", "HPOBenchMFProblem", "load_hpobench_mf_problem"]
