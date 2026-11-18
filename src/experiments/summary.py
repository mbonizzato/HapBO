"""Create study-level CSV summaries from immutable run directories."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_values(directory: Path, method_id: str) -> dict[str, Any]:
    metrics_path = directory / "metrics.json"
    if not metrics_path.is_file():
        return {}
    method = _load(metrics_path).get("methods", {}).get(method_id, {})
    return {
        name: method.get(name)
        for name in (
            "best_observed_value",
            "best_validation_accuracy",
            "final_simple_regret",
            "cumulative_regret",
            "best_normalized_performance",
            "first_optimum_query",
            "first_95_percent_query",
            "best_feasible_objective",
            "feasible_evaluation_count",
            "hidden_failure_count",
            "maximum_constraint_violation",
        )
    }


def collect_runs(study_result_directory: str | Path) -> list[dict[str, Any]]:
    """Collect one flat record per status artifact."""

    root = Path(study_result_directory).expanduser().resolve()
    records = []
    for status_path in sorted(root.glob("runs/**/status.json")):
        directory = status_path.parent
        status = _load(status_path)
        run = _load(directory / "study_run.json")
        record = {
            "run_id": run["run_id"],
            "benchmark_id": run["benchmark_id"],
            "problem_id": run["problem_id"],
            "method_id": run["method_id"],
            "budget": run["budget"],
            "seed": run["seed"],
            "replicate": run["replicate"],
            "status": status["status"],
            "return_code": status.get("return_code"),
            "wall_time_seconds": status.get("wall_time_seconds"),
            "failure_stage": status.get("failure_stage"),
            "result_directory": str(directory),
        }
        record.update(_metric_values(directory, run["method_id"]))
        records.append(record)
    return records


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_study(study_result_directory: str | Path) -> tuple[Path, Path, Path]:
    """Write run, failure, and conservative aggregate CSV files."""

    root = Path(study_result_directory).expanduser().resolve()
    summary_directory = root / "summaries"
    summary_directory.mkdir(parents=True, exist_ok=True)
    records = collect_runs(root)
    failures = [record for record in records if record["status"] != "completed"]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["status"] == "completed":
            grouped[
                (
                    record["benchmark_id"],
                    record["problem_id"],
                    record["method_id"],
                    record["budget"],
                )
            ].append(record)
    aggregates = []
    for key, group in sorted(grouped.items()):
        row: dict[str, Any] = {
            "benchmark_id": key[0],
            "problem_id": key[1],
            "method_id": key[2],
            "budget": key[3],
            "n_completed": len(group),
        }
        for metric in (
            "best_observed_value",
            "best_validation_accuracy",
            "final_simple_regret",
            "cumulative_regret",
            "best_normalized_performance",
            "first_optimum_query",
            "first_95_percent_query",
            "best_feasible_objective",
            "feasible_evaluation_count",
            "hidden_failure_count",
            "maximum_constraint_violation",
        ):
            values = [float(item[metric]) for item in group if item.get(metric) is not None]
            if values:
                row[f"{metric}_mean"] = statistics.fmean(values)
                row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else ""
        aggregates.append(row)
    runs_path = summary_directory / "runs.csv"
    failures_path = summary_directory / "failures.csv"
    aggregates_path = summary_directory / "aggregate.csv"
    _write_csv(runs_path, records)
    _write_csv(failures_path, failures)
    _write_csv(aggregates_path, aggregates)
    return runs_path, failures_path, aggregates_path
