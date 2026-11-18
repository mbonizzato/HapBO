"""Shared outer evaluation loop and result writer for baseline methods."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.bayesian_optimization.bo import OptimizationResult

from . import get_baseline
from .base import Config, Observation, config_key
from .bridge import require_full_budget


@dataclass(frozen=True)
class BaselineExperimentResult:
    """A completed baseline trace plus its on-disk artifact directory."""

    method: str
    optimization: OptimizationResult
    history: tuple[tuple[Config, Observation], ...]
    output_directory: Path


def _component(value: str) -> str:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", value)
    component = re.sub(r"[^a-z0-9]+", "-", separated.lower()).strip("-")
    if not component:
        raise ValueError(f"cannot construct a result path from {value!r}")
    return component


def _objective_value(observation: Observation) -> float:
    if isinstance(observation, Mapping):
        return float(observation["objective"])
    return float(observation)


def _constraints(observation: Observation, n_constraints: int) -> np.ndarray:
    if not n_constraints:
        return np.empty(0, dtype=float)
    if not isinstance(observation, Mapping):
        raise ValueError("constrained objectives must return a mapping observation")
    values = np.asarray(observation.get("constraints", ()), dtype=float).reshape(-1)
    if values.size != n_constraints:
        raise ValueError(
            f"observation constraints must contain {n_constraints} values"
        )
    return values


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def run_ask_tell_loop(
    optimizer: Any,
    objective: Callable[[Config], Observation],
    *,
    budget: int,
    n_init: int = 1,
    on_evaluation: Callable[[int, Config, Observation], None] | None = None,
) -> tuple[tuple[Config, Observation], ...]:
    """Execute the shared warmstart/ask/evaluate/tell lifecycle.

    Evaluation deliberately remains in this outer loop. ``on_evaluation`` is
    called after every successful ``tell`` so orchestration layers can persist
    crash-safe, incremental histories without duplicating optimizer logic.
    """

    if not 0 <= n_init <= budget:
        raise ValueError("n_init must be between 0 and budget")

    run_charged = getattr(optimizer, "run_charged", None)
    if callable(run_charged):
        # Subprocess-bridge baselines: every worker proposal is a charged query (repeats
        # included, flagged in optimizer.trace); the trace may end before the budget when
        # the worker stops; callers fail such a run (see baselines.bridge).
        return run_charged(objective, budget, on_evaluation)

    def evaluate(config: Config) -> None:
        observation = objective(dict(config))
        optimizer.tell(config, observation)
        if on_evaluation is not None:
            on_evaluation(len(optimizer.history), dict(config), observation)

    for config in optimizer.warmstart(n_init):
        evaluate(config)
    while len(optimizer.history) < budget:
        evaluate(optimizer.ask())
    return optimizer.history


def run_baseline_experiment(
    method: str,
    *,
    search_space: Mapping[str, Any],
    objective: Callable[[Config], Observation],
    objective_meta: Mapping[str, Any],
    dataset_path: str | Sequence[str],
    prior_id: str = "no-prior",
    prior: Any = None,
    llm_client: Any = None,
    seed: int | None = None,
    budget: int,
    n_init: int = 1,
    results_root: str | Path = "results",
    method_kwargs: Mapping[str, Any] | None = None,
) -> BaselineExperimentResult:
    """Run warmstart then ask/evaluate/tell, with evaluation owned here."""

    if not 0 <= n_init <= budget:
        raise ValueError("n_init must be between 0 and budget")
    parts = (dataset_path,) if isinstance(dataset_path, str) else tuple(dataset_path)
    seed_label = 0 if seed is None else int(seed)
    output_directory = Path(results_root).expanduser().resolve().joinpath(
        *(_component(part) for part in parts),
        method,
        _component(prior_id),
        f"seed-{seed_label:04d}",
    )
    if llm_client is not None and hasattr(llm_client, "set_run_directory"):
        llm_client.set_run_directory(output_directory)

    kwargs = dict(method_kwargs or {})
    kwargs.setdefault("n_initial", max(1, n_init))
    optimizer = get_baseline(
        method,
        search_space=search_space,
        objective_meta=objective_meta,
        prior=prior,
        llm_client=llm_client,
        seed=seed,
        budget=budget,
        **kwargs,
    )

    history = run_ask_tell_loop(
        optimizer,
        objective,
        budget=budget,
        n_init=n_init,
    )
    if callable(getattr(optimizer, "run_charged", None)):  # a bridged worker that stops early fails the run
        require_full_budget(len(history), budget, method)
    candidates = tuple(search_space["candidates"])
    key_to_index = {
        config_key(candidate): index for index, candidate in enumerate(candidates)
    }
    evaluated_indices = np.asarray(
        [key_to_index[config_key(config)] for config, _ in history], dtype=int
    )
    values = np.asarray([_objective_value(obs) for _, obs in history], dtype=float)
    maximize = bool(objective_meta.get("maximize", True))
    n_constraints = int(objective_meta.get("n_constraints", 0))
    constraint_values = np.asarray(
        [_constraints(obs, n_constraints) for _, obs in history], dtype=float
    ).reshape(len(history), n_constraints)
    if n_constraints:
        feasible = np.all(constraint_values <= 0.0, axis=1)
    else:
        feasible = np.ones(len(history), dtype=bool)
    eligible = np.flatnonzero(feasible)
    if eligible.size:
        local = int(np.argmax(values[eligible]) if maximize else np.argmin(values[eligible]))
        best_position = int(eligible[local])
    else:
        violations = np.sum(np.maximum(constraint_values, 0.0), axis=1)
        best_position = int(np.argmin(violations))
    best_index = int(evaluated_indices[best_position])
    encoded = np.asarray(optimizer.encoded_candidates, dtype=float)
    result = OptimizationResult(
        best_index=best_index,
        best_x=encoded[best_index].copy(),
        best_y=float(values[best_position]),
        evaluated_indices=evaluated_indices.copy(),
        X=encoded[evaluated_indices].copy(),
        y=values.copy(),
        constraint_values=constraint_values.copy(),
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    config_document = {
        "artifact_format": "llm-bo-baseline-run-v1",
        "generated_at_utc": generated_at,
        "dataset_path": list(parts),
        "method": method,
        "prior_id": prior_id,
        "seed": seed,
        "budget": budget,
        "n_init": n_init,
        "objective_meta": dict(objective_meta),
        "method_kwargs": kwargs,
    }
    metrics_document = {
        "artifact_format": "llm-bo-baseline-metrics-v1",
        "method": method,
        "evaluations": len(history),
        "best_index": best_index,
        "best_configuration": dict(candidates[best_index]),
        "best_observed_value": float(values[best_position]),
        "evaluated_indices": evaluated_indices.tolist(),
    }
    (output_directory / "run_config.json").write_text(
        json.dumps(config_document, indent=2, ensure_ascii=False, default=_json_value)
        + "\n",
        encoding="utf-8",
    )
    (output_directory / "metrics.json").write_text(
        json.dumps(metrics_document, indent=2, ensure_ascii=False, default=_json_value)
        + "\n",
        encoding="utf-8",
    )
    with (output_directory / "evaluations.jsonl").open("w", encoding="utf-8") as handle:
        for query, ((config, observation), candidate_index) in enumerate(
            zip(history, evaluated_indices), start=1
        ):
            record = {
                "query": query,
                "candidate_index": int(candidate_index),
                "config": config,
                "observation": observation,
            }
            handle.write(
                json.dumps(record, ensure_ascii=False, default=_json_value) + "\n"
            )
    return BaselineExperimentResult(method, result, history, output_directory)


__all__ = [
    "BaselineExperimentResult",
    "run_ask_tell_loop",
    "run_baseline_experiment",
]
