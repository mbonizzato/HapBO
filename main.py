"""
Main script for HapBO to launch either the synthetic example or a real world dataset of your choice with or without a natural language prior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import numpy as np
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIRECTORY = PROJECT_ROOT / "src"

if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from bayesian_optimization import (
    ALPHA_MAX,
    BayesianOptimization,
    OptimizationResult,
)
from src.baselines.runner import run_baseline_experiment
from datasets import (
    HPOBENCH_DATASET_ID,
    HPOBENCH_DEFAULT_MODEL,
    HPOBENCH_DEFAULT_TASK,
    HPOBENCH_DIRECTORY,
    HPOBENCH_MODELS,
    HPOBENCH_TASKS,
    HPOBenchProblem,
    NEUROMOSAIC_DATASET1_RAT_DATASET_ID,
    NEUROMOSAIC_DATASET1_RAT_DEFAULT_MUSCLE,
    NEUROMOSAIC_DATASET1_RAT_DEFAULT_SUBJECT,
    NEUROMOSAIC_DATASET1_RAT_DEFAULT_SUBJECT_PATH,
    NEUROMOSAIC_DATASET1_RAT_SUBJECT_FILES,
    OfflineOptimizationProblem,
    SOLAR_DEFAULT_INSTANCE,
    SOLAR_COMMIT,
    SOLAR_EXECUTABLE,
    SOLAR_INSTANCES,
    evaluate_solar,
    generate_solar_candidate_pool,
    load_hpobench_problem,
    load_neuromosaic_dataset1_rat_problem,
    resolve_solar_instance,
)
from prior.landscape.landscape_generator import (
    PRIOR_RESULTS_DIRECTORY,
    RECORD_PATH,
    VOCABULARY_CARD_PATH,
    CompiledPrior,
    compile_prior,
    finalize_prior_experiment,
    initialize_prior_experiment,
    load_record,
    resolve_prior_path,
    run_landscape_generation,
    validate_prior_compatibility,
)

# Generate path for the program.
SYNTHETIC_RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "synthetic-bo"
NEUROMOSAIC_RESULTS_DIRECTORY = PROJECT_ROOT / "results"
DEFAULT_PLOT_REQUEST = Path("__mode_default_plot__")
# Optimizer algorithms
VANILLA_BO_BASELINE = "vanilla-bo"
RANDOM_SEARCH_BASELINE = "random-search"
HAPBO_METHOD = "HapBO"
OURS_METHOD = "ours"
# registry baselines that run through main.py (the external ones need a study prior mapping)
MAIN_BASELINES = ("random", "vanilla_bo")
DEFAULT_ACQUISITION = "expected-improvement"
ACQUISITION_CHOICES = (DEFAULT_ACQUISITION, "ucb")
ACQUISITION_LABELS = {
    "expected-improvement": "Expected Improvement",
    "ucb": "UCB",
}
# Baselines list
BASELINE_LABELS = {
    VANILLA_BO_BASELINE: "Vanilla BO",
    RANDOM_SEARCH_BASELINE: "Random Search",
}
NEUROMOSAIC_DATASET_IDS = (NEUROMOSAIC_DATASET1_RAT_DATASET_ID,)
NEUROMOSAIC_SUBJECT_IDS = tuple(NEUROMOSAIC_DATASET1_RAT_SUBJECT_FILES)
# Synthetics example grid generation and optimal positioning.
GRID_ROWS = 4
GRID_COLUMNS = 8
TRUE_PEAK = np.array([7.0, 4.0])
DEFAULT_PRIOR_CENTER = np.array([8.0, 4.0])


@dataclass(frozen=True)
class PerformanceMetrics:
    """Noise-free optimization metrics evaluated against known ground truth."""

    expected_responses: np.ndarray
    instantaneous_regret: np.ndarray
    simple_regret: np.ndarray
    cumulative_regret: np.ndarray
    best_expected_response: np.ndarray
    normalized_best_performance: np.ndarray
    first_optimum_query: int | None
    first_95_percent_query: int | None


@dataclass(frozen=True)
class MethodRun:
    """One optimization trace and its plot-ready performance metrics."""

    key: str
    label: str
    result: OptimizationResult
    metrics: PerformanceMetrics


@dataclass(frozen=True)
class PriorGuidedOptimizationResult:
    """A BO result together with its compiled joint-prior context."""

    optimization: OptimizationResult
    best_native_candidate: dict[str, Any]
    evaluated_native_candidates: tuple[dict[str, Any], ...]
    encoded_candidates: np.ndarray
    pi_values: np.ndarray
    alpha: float
    parameter_names: tuple[str, ...]


PriorGuidedObjective = Callable[[Mapping[str, Any]], float]
PriorGuidedCallback = Callable[
    [int, Mapping[str, Any], float, BayesianOptimization],
    None,
]



# Prior version used by the dataset commands when --prior-id is not given.
DEFAULT_BENCHMARK_PRIOR_IDS = {"hpobench": "llm-card", "solar": "expert", "neuromosaic": "human-expert-map"}

def _path_component(value: str) -> str:
    """Convert a human-readable experiment label into a stable path segment."""

    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", value)
    component = re.sub(r"[^a-z0-9]+", "-", separated.lower()).strip("-")
    if not component:
        raise ValueError(f"cannot construct a result path from {value!r}")
    return component


def save_neuromosaic_run_records(
    directory: Path,
    problem: OfflineOptimizationProblem,
    runs: Sequence[MethodRun],
    args: argparse.Namespace,
    plot_path: Path,
) -> tuple[Path, Path]:
    """Save the configuration and final metrics beside one research result."""

    directory = directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    prior_path = Path(args.dataset_prior_file).expanduser().resolve()
    try:
        prior_source = prior_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        prior_source = str(prior_path)
    compiled_alpha = compile_prior(load_record(prior_path)).alpha

    configuration = {
        "artifact_format": "llm-bo-run-config-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": problem.dataset_id,
        "subject_id": problem.subject_id,
        "target_name": problem.target_name,
        "prior": {
            "id": prior_path.parent.name,
            "path": prior_source,
            "sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        },
        "optimization": {
            "acquisition": args.acquisition,
            "alpha": compiled_alpha,
            "improvement_xi": args.improvement_xi,
            "kappa": args.kappa,
            "queries": args.queries,
            "n_initial": args.n_initial,
            "length_scale": args.length_scale,
            "noise_std": args.noise_std,
            "seed": args.seed,
            "baselines": list(args.baselines),
        },
        "outputs": {"performance_plot": str(plot_path)},
    }
    metrics = {
        "artifact_format": "llm-bo-metrics-v1",
        "methods": {
            run.key: {
                "label": run.label,
                "best_observed_value": float(run.result.best_y),
                "best_index": int(run.result.best_index),
                "evaluated_indices": run.result.evaluated_indices.astype(int).tolist(),
                "final_simple_regret": float(run.metrics.simple_regret[-1]),
                "cumulative_regret": float(run.metrics.cumulative_regret[-1]),
                "best_normalized_performance": float(
                    run.metrics.normalized_best_performance[-1]
                ),
                "first_optimum_query": run.metrics.first_optimum_query,
                "first_95_percent_query": run.metrics.first_95_percent_query,
            }
            for run in runs
        },
    }

    config_path = directory / "run_config.json"
    metrics_path = directory / "metrics.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(configuration, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return config_path, metrics_path


def save_hpobench_run_records(
    directory: Path,
    problem: HPOBenchProblem,
    runs: Sequence[MethodRun],
    args: argparse.Namespace,
    plot_path: Path,
) -> tuple[Path, Path]:
    """Save one HPOBench experiment configuration and its final metrics."""

    directory = directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    prior_path = Path(args.dataset_prior_file).expanduser().resolve()
    try:
        prior_source = prior_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        prior_source = str(prior_path)
    compiled_alpha = compile_prior(load_record(prior_path)).alpha
    configuration = {
        "artifact_format": "llm-bo-run-config-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": problem.dataset_id,
        "task": {"name": problem.task_name, "openml_task_id": problem.task_id},
        "model": problem.model,
        "objective": problem.metadata["objective"],
        "full_fidelity": problem.metadata["full_fidelity"],
        "benchmark_seeds": list(problem.metadata["seeds"]),
        "candidate_count": problem.n_candidates,
        "prior": {
            "id": prior_path.parent.name,
            "path": prior_source,
            "sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        },
        "optimization": {
            "acquisition": args.acquisition,
            "alpha": compiled_alpha,
            "improvement_xi": args.improvement_xi,
            "kappa": args.kappa,
            "queries": args.queries,
            "n_initial": args.n_initial,
            "length_scale": args.length_scale,
            "noise_std": args.noise_std,
            "seed": args.seed,
            "baselines": list(args.baselines),
        },
        "outputs": {"performance_plot": str(plot_path)},
    }
    metrics = {
        "artifact_format": "llm-bo-metrics-v1",
        "methods": {
            run.key: {
                "label": run.label,
                "best_validation_accuracy": float(run.result.best_y),
                "best_index": int(run.result.best_index),
                "best_configuration": problem.native_candidates[
                    run.result.best_index
                ],
                "evaluated_indices": run.result.evaluated_indices.astype(int).tolist(),
                "final_simple_regret": float(run.metrics.simple_regret[-1]),
                "cumulative_regret": float(run.metrics.cumulative_regret[-1]),
                "best_normalized_performance": float(
                    run.metrics.normalized_best_performance[-1]
                ),
                "first_optimum_query": run.metrics.first_optimum_query,
                "first_95_percent_query": run.metrics.first_95_percent_query,
            }
            for run in runs
        },
    }
    config_path = directory / "run_config.json"
    metrics_path = directory / "metrics.json"
    config_path.write_text(
        json.dumps(configuration, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return config_path, metrics_path


def _names_by_lowercase(names: Sequence[str], label: str) -> dict[str, str]:
    """Map each lower-cased name to its spelling; reject names equal up to case."""

    folded: dict[str, str] = {}
    for name in names:
        key = str(name).lower()
        if key in folded:
            raise ValueError(
                f"{label} {folded[key]!r} and {name!r} are ambiguous when case is ignored"
            )
        folded[key] = name
    return folded


def _as_native_candidate_records(
    native_candidates: Sequence[Mapping[str, Any]] | np.ndarray,
    parameter_names: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    """Normalize native candidates to dictionaries in prior-schema order.

    Mapping candidates are matched to the schema names case-insensitively
    (HPOBench's SVM key ``C`` is the PSL dimension ``c``) and re-keyed by the
    schema names.
    """

    names = tuple(parameter_names)
    if isinstance(native_candidates, np.ndarray):
        candidate_rows: list[Any] = list(
            np.asarray(native_candidates, dtype=object)
        )
    else:
        candidate_rows = list(native_candidates)
    if not candidate_rows:
        raise ValueError("native_candidates must contain at least one candidate")

    if isinstance(candidate_rows[0], Mapping):
        records = []
        schema_names = _names_by_lowercase(names, "prior-schema dimensions")
        for index, candidate in enumerate(candidate_rows):
            if not isinstance(candidate, Mapping):
                raise ValueError(
                    "native candidates must all use the same representation"
                )
            keys = _names_by_lowercase(tuple(candidate), f"native candidate {index} keys")
            missing = set(schema_names) - set(keys)
            extra = set(keys) - set(schema_names)
            if missing or extra:
                raise ValueError(
                    f"native candidate {index} does not match the prior schema "
                    f"(missing={sorted(schema_names[key] for key in missing)}, "
                    f"extra={sorted(keys[key] for key in extra)})"
                )
            records.append({name: candidate[keys[name.lower()]] for name in names})
        return tuple(records)

    matrix = np.asarray(candidate_rows, dtype=object)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError(
            "native_candidates must be a two-dimensional matrix with one "
            "column per prior-schema dimension"
        )
    return tuple(
        {name: matrix[row, column] for column, name in enumerate(names)}
        for row in range(matrix.shape[0])
    )


def encode_candidates_for_gp(
    native_candidates: Sequence[Mapping[str, Any]] | np.ndarray,
    compiled_prior: CompiledPrior,
) -> np.ndarray:
    """Encode complete native configurations for the numeric GP geometry.

    Continuous dimensions use the compiler's linear or logarithmic unit
    transform, ordinal dimensions use normalized rank, and categorical
    dimensions use one-hot features so the GP does not infer a false ordering.
    """

    records = _as_native_candidate_records(
        native_candidates,
        compiled_prior.parameter_names,
    )
    feature_blocks = []
    for dimension in compiled_prior.dimensions:
        native_column = np.asarray(
            [record[dimension.name] for record in records],
            dtype=object,
        )
        unit_column = dimension.encode_column(native_column)
        if dimension.type != "categorical":
            feature_blocks.append(unit_column.reshape(-1, 1))
            continue

        assert dimension.values is not None
        n_categories = len(dimension.values)
        if n_categories == 1:
            category_indices = np.zeros(len(records), dtype=int)
        else:
            category_indices = np.rint(
                unit_column * (n_categories - 1)
            ).astype(int)
        feature_blocks.append(np.eye(n_categories)[category_indices])
    return np.column_stack(feature_blocks).astype(float, copy=False)


def run_prior_guided_optimization(
    native_candidates: Sequence[Mapping[str, Any]] | np.ndarray,
    objective: PriorGuidedObjective,
    max_evaluations: int,
    *,
    encoded_candidates: np.ndarray | None = None,
    prior_path: str | Path = RECORD_PATH,
    length_scale: float = 1.0,
    kappa: float = 2.0,
    acquisition: str = DEFAULT_ACQUISITION,
    improvement_xi: float = 0.0,
    n_initial: int = 1,
    noise_variance: float = 1e-6,
    signal_variance: float = 1.0,
    normalize_y: bool = False,
    random_state: int | None = None,
    callback: PriorGuidedCallback | None = None,
) -> PriorGuidedOptimizationResult:
    """Run alpha-pi BO using the joint prior compiled from ``prior.json``.

    ``native_candidates`` are complete configurations in the schema's native
    units and are used for joint ``pi(X)`` evaluation and objective calls.
    By default, the schema produces aligned numeric GP rows with log-aware
    scaling and one-hot categorical features. An application may instead pass
    ``encoded_candidates`` to use a custom numeric representation. Feasibility
    filtering remains an application responsibility.

    The initial design is random (seeded by ``random_state``), as for HapBO in
    the paper.
    """

    record = load_record(prior_path)
    compiled_prior: CompiledPrior = compile_prior(record)
    records = _as_native_candidate_records(
        native_candidates,
        compiled_prior.parameter_names,
    )
    if encoded_candidates is None:
        encoded_candidates = encode_candidates_for_gp(records, compiled_prior)
    encoded_candidates = np.asarray(encoded_candidates, dtype=float)
    if encoded_candidates.ndim == 1:
        encoded_candidates = encoded_candidates.reshape(-1, 1)
    if encoded_candidates.ndim != 2:
        raise ValueError(
            "encoded_candidates must be a two-dimensional numeric matrix"
        )
    if len(records) != encoded_candidates.shape[0]:
        raise ValueError(
            "native_candidates and encoded_candidates must contain the same "
            "number of aligned rows"
        )
    if not 1 <= max_evaluations <= len(records):
        raise ValueError(
            "max_evaluations must be between 1 and the number of candidates"
        )
    if not 1 <= n_initial <= max_evaluations:
        raise ValueError("n_initial must be between 1 and max_evaluations")

    # One complete native configuration produces one joint pi value. Alpha is
    # passed separately because the selected acquisition applies the multiplier.
    pi_values = np.asarray(compiled_prior.pi(records), dtype=float).reshape(-1)
    maximize = record["schema"]["objective"] == "maximize"
    optimizer = BayesianOptimization(
        encoded_candidates,
        length_scale=length_scale,
        kappa=kappa,
        acquisition=acquisition,
        improvement_xi=improvement_xi,
        alpha=compiled_prior.alpha,
        prior=pi_values,
        n_initial=n_initial,
        noise_variance=noise_variance,
        signal_variance=signal_variance,
        normalize_y=normalize_y,
        maximize=maximize,
        random_state=random_state,
    )

    while optimizer.n_observations < max_evaluations:
        index, _ = optimizer.ask()
        native_candidate = dict(records[index])
        value = float(objective(native_candidate))
        optimizer.tell(index, value)
        if callback is not None:
            callback(index, dict(native_candidate), value, optimizer)

    optimization = optimizer.result()
    return PriorGuidedOptimizationResult(
        optimization=optimization,
        best_native_candidate=dict(records[optimization.best_index]),
        evaluated_native_candidates=tuple(
            dict(records[index]) for index in optimization.evaluated_indices
        ),
        encoded_candidates=encoded_candidates.copy(),
        pi_values=pi_values.copy(),
        alpha=compiled_prior.alpha,
        parameter_names=tuple(compiled_prior.parameter_names),
    )


def build_synthetic_problem(
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Callable[[np.ndarray], float]]:
    """Create the 4x8 noisy synthetic objective used by HapBO."""
    candidates = np.array(
        [
            [float(x), float(y)]
            for y in range(1, GRID_ROWS + 1)
            for x in range(1, GRID_COLUMNS + 1)
        ]
    )

    distance_to_peak = np.linalg.norm(candidates - TRUE_PEAK, axis=1)
    ground_truth = np.exp(-0.3 * distance_to_peak)
    rng = np.random.default_rng(seed)

    def objective(candidate: np.ndarray) -> float:
        distance = float(np.linalg.norm(candidate - TRUE_PEAK))
        expected_response = float(np.exp(-0.3 * distance))
        return expected_response + float(rng.normal(0.0, noise_std))

    return candidates, ground_truth, objective


def build_gaussian_prior(
    candidates: np.ndarray,
    center: np.ndarray,
    spread: float,
) -> np.ndarray:
    """Create a numeric Gaussian prior from a user-configurable center."""
    center = np.asarray(center, dtype=float).reshape(-1)
    if center.shape != (candidates.shape[1],):
        raise ValueError("prior center must have one coordinate per candidate dimension")
    squared_distance = np.sum((candidates - center) ** 2, axis=1)
    return np.exp(-0.5 * squared_distance / spread**2)


def compute_performance_metrics(
    result: OptimizationResult,
    ground_truth: np.ndarray,
) -> PerformanceMetrics:
    """Compute regret and best-so-far metrics for one optimization run."""
    evaluated_indices = np.asarray(result.evaluated_indices, dtype=int)
    expected_responses = ground_truth[evaluated_indices]
    optimum_value = float(np.max(ground_truth))
    optimum_index = int(np.argmax(ground_truth))

    instantaneous_regret = optimum_value - expected_responses
    best_expected_response = np.maximum.accumulate(expected_responses)
    simple_regret = optimum_value - best_expected_response
    cumulative_regret = np.cumsum(instantaneous_regret)
    normalized_best_performance = best_expected_response / optimum_value

    optimum_hits = np.flatnonzero(evaluated_indices == optimum_index)
    threshold_hits = np.flatnonzero(normalized_best_performance >= 0.95)

    return PerformanceMetrics(
        expected_responses=expected_responses,
        instantaneous_regret=instantaneous_regret,
        simple_regret=simple_regret,
        cumulative_regret=cumulative_regret,
        best_expected_response=best_expected_response,
        normalized_best_performance=normalized_best_performance,
        first_optimum_query=int(optimum_hits[0] + 1) if optimum_hits.size else None,
        first_95_percent_query=int(threshold_hits[0] + 1) if threshold_hits.size else None,
    )


def build_method_run(
    key: str,
    label: str,
    result: OptimizationResult,
    ground_truth: np.ndarray,
) -> MethodRun:
    """Package one method result for shared reporting and plotting."""

    return MethodRun(
        key=key,
        label=label,
        result=result,
        metrics=compute_performance_metrics(result, ground_truth),
    )


def build_vanilla_bo_optimizer(
    candidates: np.ndarray,
    args: argparse.Namespace,
    *,
    normalize_y: bool,
    maximize: bool = True,
) -> BayesianOptimization:
    """Construct the registered prior-free vanilla GP baseline."""

    return BayesianOptimization(
        candidates,
        length_scale=args.length_scale,
        alpha=0.0,
        prior=None,
        n_initial=args.n_initial,
        noise_variance=max(args.noise_std**2, 1e-12),
        normalize_y=normalize_y,
        maximize=maximize,
        random_state=args.seed,
        acquisition=args.acquisition,
        kappa=args.kappa,
        improvement_xi=args.improvement_xi,
    )


def build_random_search_optimizer(
    candidates: np.ndarray,
    args: argparse.Namespace,
    *,
    maximize: bool = True,
) -> BayesianOptimization:
    """Construct uniform sampling without fitting a surrogate model."""

    return BayesianOptimization(
        candidates,
        alpha=0.0,
        prior=None,
        # Keeping the complete budget in the initialization phase makes every
        # ask uniformly random over the remaining candidates. The GP and its
        # acquisition function are therefore never fitted or evaluated.
        n_initial=args.queries,
        maximize=maximize,
        random_state=args.seed,
    )


def run_registered_baseline(
    args: argparse.Namespace,
    *,
    native_candidates: Sequence[Mapping[str, Any]],
    objective: Callable[[dict[str, Any]], Any],
    dataset_path: Sequence[str],
    maximize: bool,
    encoded_candidates: np.ndarray | None = None,
    prior: Any = None,
    prior_id: str = "no-prior",
    n_constraints: int = 0,
    normalize_y: bool = True,
) -> OptimizationResult:
    """Run one registered baseline through the shared outer evaluation loop."""

    search_space: dict[str, Any] = {"candidates": tuple(native_candidates)}
    if encoded_candidates is not None:
        search_space["encoded_candidates"] = np.asarray(
            encoded_candidates,
            dtype=float,
        )
    outcome = run_baseline_experiment(
        args.method,
        search_space=search_space,
        objective=objective,
        objective_meta={
            "maximize": maximize,
            "n_constraints": n_constraints,
        },
        dataset_path=dataset_path,
        prior_id=prior_id,
        prior=prior,
        seed=args.seed,
        budget=args.queries,
        n_init=args.n_initial,
        results_root=PROJECT_ROOT / "results",
        method_kwargs={
            "length_scale": args.length_scale,
            "kappa": args.kappa,
            "acquisition": args.acquisition,
            "improvement_xi": args.improvement_xi,
            "noise_variance": max(args.noise_std**2, 1e-12),
            "normalize_y": normalize_y,
        },
    )
    print(f"Baseline method: {args.method}")
    print(f"Best observed value: {outcome.optimization.best_y:.8g}")
    print(f"Evaluations: {len(outcome.optimization.y)}")
    print(f"Results: {outcome.output_directory}")
    return outcome.optimization


def adapt_neuromosaic_candidates_to_prior(
    problem: OfflineOptimizationProblem,
    dimension_names: Sequence[str],
) -> tuple[
    tuple[dict[str, int], ...],
    Callable[[Mapping[str, Any]], dict[str, int]],
]:
    """Map NeuroMOSAIC grid coordinates into a supported prior schema."""

    names = tuple(dimension_names)
    schema_to_dataset = {
        ("channel_x", "channel_y"): {
            "channel_x": "channel_x",
            "channel_y": "channel_y",
        },
        (
            "electrode_mediolateral_index",
            "electrode_rostrocaudal_index",
        ): {
            "electrode_mediolateral_index": "channel_y",
            "electrode_rostrocaudal_index": "channel_x",
        },
    }
    if names not in schema_to_dataset:
        raise ValueError(
            "the NeuroMOSAIC prior dimensions must be either "
            "['channel_x', 'channel_y'] or "
            "['electrode_mediolateral_index', "
            "'electrode_rostrocaudal_index'] in that order"
        )
    mapping = schema_to_dataset[names]
    prior_candidates = tuple(
        {
            schema_name: int(candidate[dataset_name])
            for schema_name, dataset_name in mapping.items()
        }
        for candidate in problem.native_candidates
    )

    def to_dataset_candidate(candidate: Mapping[str, Any]) -> dict[str, int]:
        return {
            dataset_name: int(candidate[schema_name])
            for schema_name, dataset_name in mapping.items()
        }

    return prior_candidates, to_dataset_candidate


def print_baseline_comparison(runs: Sequence[MethodRun]) -> None:
    """Print a compact final row for every requested baseline."""

    if len(runs) <= 1:
        return
    print("\nBaseline comparison")
    for run in runs[1:]:
        print(
            f"  {run.label}: best={run.result.best_y:.4f}, "
            f"simple regret={run.metrics.simple_regret[-1]:.4f}, "
            "normalized best="
            f"{100.0 * run.metrics.normalized_best_performance[-1]:.2f}%"
        )


def save_performance_plot(
    runs: Sequence[MethodRun],
    output_path: Path,
    *,
    title: str = "Synthetic BO method comparison",
    response_label: str = "Noise-free response",
    optimum_value: float = 1.0,
) -> Path:
    """Save dynamically overlaid finite-benchmark performance curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not runs:
        raise ValueError("runs must contain at least one optimization result")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    figure.suptitle(title)
    colors = plt.get_cmap("tab10").colors

    response_axis = axes[0, 0]
    response_axis.axhline(
        optimum_value,
        color="black",
        linestyle=":",
        label="True optimum",
    )
    response_axis.set_title("Expected response at each query")
    response_axis.set_xlabel("Query")
    response_axis.set_ylabel(response_label)

    regret_axis = axes[0, 1]
    regret_axis.set_title("Simple regret")
    regret_axis.set_xlabel("Query")
    regret_axis.set_ylabel("Regret")

    cumulative_axis = axes[1, 0]
    cumulative_axis.set_title("Cumulative regret")
    cumulative_axis.set_xlabel("Query")
    cumulative_axis.set_ylabel("Cumulative regret")

    performance_axis = axes[1, 1]
    performance_axis.axhline(95.0, color="black", linestyle=":", label="95% threshold")
    performance_axis.set_title("Best-so-far performance")
    performance_axis.set_xlabel("Query")
    performance_axis.set_ylabel("Percent of true optimum")
    performance_axis.set_ylim(0.0, 105.0)

    for run_index, run in enumerate(runs):
        queries = np.arange(1, len(run.result.y) + 1)
        color = colors[run_index % len(colors)]
        response_axis.plot(
            queries,
            run.metrics.expected_responses,
            "o-",
            color=color,
            label=run.label,
        )
        regret_axis.plot(
            queries,
            run.metrics.simple_regret,
            "o-",
            color=color,
            label=run.label,
        )
        cumulative_axis.plot(
            queries,
            run.metrics.cumulative_regret,
            "o-",
            color=color,
            label=run.label,
        )
        performance_axis.plot(
            queries,
            100.0 * run.metrics.normalized_best_performance,
            "o-",
            color=color,
            label=run.label,
        )

    regret_axis.set_ylim(bottom=0.0)
    cumulative_axis.set_ylim(bottom=0.0)

    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend()

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def save_offline_performance_plot(
    problem: OfflineOptimizationProblem,
    runs: Sequence[MethodRun],
    output_path: Path,
) -> Path:
    """Save every method's query map and dynamic offline comparison."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not runs:
        raise ValueError("runs must contain at least one optimization result")

    coordinates = np.asarray(
        [
            [candidate["channel_x"], candidate["channel_y"]]
            for candidate in problem.native_candidates
        ],
        dtype=int,
    )
    x_values = np.unique(coordinates[:, 0])
    y_values = np.unique(coordinates[:, 1])
    response_map = np.full((len(y_values), len(x_values)), np.nan)
    for index, (x, y) in enumerate(coordinates):
        x_position = int(np.flatnonzero(x_values == x)[0])
        y_position = int(np.flatnonzero(y_values == y)[0])
        response_map[y_position, x_position] = problem.ground_truth[index]

    figure, axes = plt.subplots(2, 2, figsize=(13, 9.5), constrained_layout=True)
    figure.suptitle(f"{problem.subject_id}: {problem.target_name} method comparison")
    colors = plt.get_cmap("tab10").colors

    map_axis = axes[0, 0]
    image = map_axis.imshow(response_map, origin="lower", cmap="viridis")
    map_axis.set_xticks(np.arange(len(x_values)), x_values)
    map_axis.set_yticks(np.arange(len(y_values)), y_values)
    map_axis.set_xticks(
        np.arange(-0.5, len(x_values), 1.0),
        minor=True,
    )
    map_axis.set_yticks(
        np.arange(-0.5, len(y_values), 1.0),
        minor=True,
    )
    map_axis.grid(which="minor", color="white", linewidth=0.8, alpha=0.35)
    map_axis.tick_params(which="minor", bottom=False, left=False)
    map_axis.set_xlabel("channel_x")
    map_axis.set_ylabel("channel_y")
    map_axis.set_title("Response map and query order by method")
    figure.colorbar(image, ax=map_axis, label="Mean response")
    query_offsets = (
        np.zeros(1)
        if len(runs) == 1
        else np.linspace(-0.20, 0.20, len(runs))
    )
    for run_index, run in enumerate(runs):
        query_positions = coordinates[run.result.evaluated_indices.astype(int)]
        map_x_positions = np.asarray(
            [
                int(np.flatnonzero(x_values == x)[0])
                for x in query_positions[:, 0]
            ],
            dtype=float,
        )
        map_y_positions = np.asarray(
            [
                int(np.flatnonzero(y_values == y)[0])
                for y in query_positions[:, 1]
            ],
            dtype=float,
        )
        shifted_x_positions = map_x_positions + query_offsets[run_index]
        color = colors[run_index % len(colors)]
        map_axis.scatter(
            shifted_x_positions,
            map_y_positions,
            marker="o",
            s=105,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            label=f"{run.label} queries",
            zorder=3,
        )
        for query, (x_position, y_position) in enumerate(
            zip(shifted_x_positions, map_y_positions),
            start=1,
        ):
            map_axis.text(
                x_position,
                y_position,
                str(query),
                color="white",
                ha="center",
                va="center",
                fontsize=6.5,
                fontweight="bold",
                zorder=4,
            )
    optimum_x, optimum_y = coordinates[problem.optimum_index]
    map_axis.scatter(
        int(np.flatnonzero(x_values == optimum_x)[0]),
        int(np.flatnonzero(y_values == optimum_y)[0]),
        marker="*",
        s=180,
        facecolors="none",
        edgecolors="red",
        linewidths=1.8,
        label="True optimum",
        zorder=2,
    )
    response_axis = axes[0, 1]
    response_axis.axhline(
        problem.optimum_value,
        color="black",
        linestyle=":",
        label="True optimum",
    )
    response_axis.set_title("Evaluated responses")
    response_axis.set_xlabel("Query")
    response_axis.set_ylabel("Mean valid-trial response")
    regret_axis = axes[1, 0]
    regret_axis.set_title("Offline simple regret")
    regret_axis.set_xlabel("Query")
    regret_axis.set_ylabel("Regret")

    performance_axis = axes[1, 1]
    performance_axis.axhline(95.0, color="black", linestyle=":")
    performance_axis.set_title("Best-so-far performance")
    performance_axis.set_xlabel("Query")
    performance_axis.set_ylabel("Percent of true optimum")
    performance_axis.set_ylim(0.0, 105.0)

    for run_index, run in enumerate(runs):
        queries = np.arange(1, len(run.result.y) + 1)
        color = colors[run_index % len(colors)]
        response_axis.plot(
            queries,
            run.result.y,
            "o-",
            color=color,
            label=run.label,
        )
        regret_axis.plot(
            queries,
            run.metrics.simple_regret,
            "o-",
            color=color,
            label=run.label,
        )
        performance_axis.plot(
            queries,
            100.0 * run.metrics.normalized_best_performance,
            "o-",
            color=color,
            label=run.label,
        )

    regret_axis.set_ylim(bottom=0.0)

    for axis in (response_axis, regret_axis, performance_axis):
        axis.grid(alpha=0.25)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=colors[run_index % len(colors)],
            marker="o",
            linewidth=2,
            label=run.label,
        )
        for run_index, run in enumerate(runs)
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=":",
            marker="*",
            markerfacecolor="none",
            markeredgecolor="red",
            markeredgewidth=1.5,
            markersize=11,
            label="True optimum / target threshold",
        )
    )
    figure.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncol=min(len(legend_handles), 4),
        fontsize=9,
    )

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def make_progress_callback(
    max_evaluations: int,
) -> Callable[[int, np.ndarray, float, BayesianOptimization], None]:
    """Create a callback that prints one compact line after every evaluation."""

    def show_progress(
        index: int,
        candidate: np.ndarray,
        value: float,
        optimizer: BayesianOptimization,
    ) -> None:
        _, best_candidate, best_value = optimizer.best_observation()
        print(
            f"Query {optimizer.n_observations:02d}/{max_evaluations}: "
            f"channel={index:02d}, point=({candidate[0]:.0f}, {candidate[1]:.0f}), "
            f"response={value:.4f}, best={best_value:.4f} "
            f"at ({best_candidate[0]:.0f}, {best_candidate[1]:.0f})"
        )

    return show_progress


def make_offline_progress_callback(
    problem: OfflineOptimizationProblem,
    max_evaluations: int,
) -> PriorGuidedCallback:
    """Create progress output using native electrode coordinates."""

    def show_progress(
        index: int,
        candidate: Mapping[str, Any],
        value: float,
        optimizer: BayesianOptimization,
    ) -> None:
        best_index, _, best_value = optimizer.best_observation()
        best_candidate = problem.native_candidates[best_index]
        print(
            f"Query {optimizer.n_observations:02d}/{max_evaluations}: "
            f"channel={index + 1:02d}, "
            f"point=({int(candidate['channel_x'])}, {int(candidate['channel_y'])}), "
            f"response={value:.4f}, best={best_value:.4f} at "
            f"({best_candidate['channel_x']}, {best_candidate['channel_y']})"
        )

    return show_progress


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run synthetic, NeuroMOSAIC, HPOBench, or SOLAR Bayesian optimization, "
            "or compile and inspect the expert prior landscape."
        )
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=(
            "synthetic-bo",
            "landscape-test",
            "prior-init",
            "prior-finalize",
            "neuromosaic-bo",
            "hpobench-bo",
            "solar-bo",
        ),
        default="synthetic-bo",
        help=(
            "Action to run (default: synthetic-bo). neuromosaic-bo optimizes "
            "one selected NeuroMOSAIC subject and muscle; hpobench-bo tunes "
            "one model on one selected tabular task; solar-bo runs the live "
            "SOLAR simulator."
        ),
    )
    parser.add_argument(
        "--dataset-id",
        default=None,
        help=(
            "Dataset identifier used by prior-init, such as "
            "neuromosaic-dataset1-rat or hpobench-rf-credit-g."
        ),
    )
    parser.add_argument(
        "--prior-id",
        default=None,
        help=(
            "Prior version identifier. Required when creating a prior with "
            "prior-init; optional for BO when more than one compatible prior "
            "version exists."
        ),
    )
    parser.add_argument(
        "--prior-directory",
        type=Path,
        default=None,
        help="Draft prior directory to validate and lock with prior-finalize.",
    )
    parser.add_argument(
        "--llm-provider",
        default=None,
        help="Optional LLM provider recorded by prior-init.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Optional LLM model recorded by prior-init.",
    )
    parser.add_argument(
        "--prior-subject-scope",
        default="dataset-level expert knowledge",
        help="Subject scope recorded in draft prior metadata.",
    )
    parser.add_argument(
        "--prior-target",
        default="unspecified optimization target",
        help="Optimization target recorded in draft prior metadata.",
    )
    parser.add_argument(
        "--prior-target-name",
        default=None,
        help=(
            "Exact output/target name used to scope a prior and create its path. "
            "For NeuroMOSAIC, --muscle is the preferred equivalent."
        ),
    )
    parser.add_argument(
        "--card-human-statement",
        action="store_true",
        help=(
            "prior-init: the prompt card will carry a human (expert) statement, so "
            "beliefs may be tagged user_stated / user_confirmed (recorded as "
            "card.human_statement in metadata.json)."
        ),
    )
    parser.add_argument(
        "--no-landscape",
        action="store_true",
        help="Finalize prior metadata without generating landscape artifacts.",
    )
    parser.add_argument(
        "--prior-file",
        type=Path,
        default=RECORD_PATH,
        help=(
            "PSL prior JSON used by landscape-test "
            f"(default: {RECORD_PATH.relative_to(PROJECT_ROOT)})."
        ),
    )
    parser.add_argument(
        "--prior-output-directory",
        type=Path,
        default=None,
        help=(
            "Directory for landscape-test artifacts. If omitted, it is derived "
            "from the dataset and prior IDs; the checked-in default resolves to "
            f"{PRIOR_RESULTS_DIRECTORY.relative_to(PROJECT_ROOT)}."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=NEUROMOSAIC_DATASET_IDS,
        default=NEUROMOSAIC_DATASET1_RAT_DATASET_ID,
        help=(
            "Dataset adapter for neuromosaic-bo "
            f"(default: {NEUROMOSAIC_DATASET1_RAT_DATASET_ID})."
        ),
    )
    parser.add_argument(
        "--subject",
        choices=NEUROMOSAIC_SUBJECT_IDS,
        default=None,
        help=(
            "Subject for neuromosaic-bo "
            f"(default: {NEUROMOSAIC_DATASET1_RAT_DEFAULT_SUBJECT})."
        ),
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=None,
        help=(
            "Optional .mat subject-file override for neuromosaic-bo. Normally "
            "use --subject. Default file: "
            f"{NEUROMOSAIC_DATASET1_RAT_DEFAULT_SUBJECT_PATH.relative_to(PROJECT_ROOT)}."
        ),
    )
    parser.add_argument(
        "--muscle",
        default=None,
        help=(
            "Target muscle for neuromosaic-bo, or exact prior target for a "
            "NeuroMOSAIC prior-init (BO default: "
            f"{NEUROMOSAIC_DATASET1_RAT_DEFAULT_MUSCLE})."
        ),
    )
    parser.add_argument(
        "--dataset-prior-file",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "Optional explicit override for the dataset- and exact-target-compatible "
            "prior JSON used by HapBO. By default, HapBO resolves the prior under "
            "experiments/priors from the selected benchmark and target."
        ),
    )
    parser.add_argument(
        "--solar-instance",
        choices=tuple(SOLAR_INSTANCES),
        default=SOLAR_DEFAULT_INSTANCE,
        help=f"SOLAR problem for solar-bo (default: {SOLAR_DEFAULT_INSTANCE}).",
    )
    parser.add_argument(
        "--solar-executable",
        type=Path,
        default=SOLAR_EXECUTABLE,
        help="Compiled SOLAR executable (default: benchmark/SOLAR/source/bin/solar).",
    )
    parser.add_argument(
        "--solar-candidate-pool-size",
        type=int,
        default=256,
        help="Finite mixed-variable candidate design size (default: 256).",
    )
    parser.add_argument(
        "--solar-fidelity",
        type=float,
        default=1.0,
        help="SOLAR fidelity in [0, 1] (default: truth fidelity 1.0).",
    )
    parser.add_argument(
        "--solar-replications",
        type=int,
        default=1,
        help="Simulator replications per stochastic query (default: 1).",
    )
    parser.add_argument(
        "--solar-seed",
        type=int,
        default=0,
        help="Simulator seed, separate from the BO design seed (default: 0).",
    )
    parser.add_argument(
        "--solar-timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Optional wall-clock timeout for each SOLAR query.",
    )
    parser.add_argument(
        "--hpobench-model",
        choices=HPOBENCH_MODELS,
        default=HPOBENCH_DEFAULT_MODEL,
        help=(
            "HPOBench model family for hpobench-bo "
            f"(default: {HPOBENCH_DEFAULT_MODEL})."
        ),
    )
    parser.add_argument(
        "--hpobench-task",
        choices=tuple(HPOBENCH_TASKS),
        default=HPOBENCH_DEFAULT_TASK,
        help=(
            "Classification task (the task subset used by LLAMBO) for hpobench-bo "
            f"(default: {HPOBENCH_DEFAULT_TASK})."
        ),
    )
    parser.add_argument(
        "--hpobench-benchmark-directory",
        type=Path,
        default=HPOBENCH_DIRECTORY,
        help=(
            "Root containing downloaded HPOBench model/task tables "
            f"(default: {HPOBENCH_DIRECTORY.relative_to(PROJECT_ROOT)})."
        ),
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=tuple(BASELINE_LABELS),
        default=(),
        metavar="BASELINE",
        help=(
            "Run one or more comparison baselines. Available: "
            f"{VANILLA_BO_BASELINE} (GP with the selected acquisition, no prior) "
            f"and {RANDOM_SEARCH_BASELINE} (uniform unqueried candidates)."
        ),
    )
    parser.add_argument(
        "--method",
        choices=(OURS_METHOD, *MAIN_BASELINES),
        default=OURS_METHOD,
        help=(
            "Primary optimizer (default: ours, the prior-guided HapBO method), or one of the "
            f"in-engine baselines {', '.join(MAIN_BASELINES)}. The external baselines run through "
            "the study system (scripts/run_study.py)."
        ),
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=16,
        help=(
            "Total objective evaluations (default: 16; maximum: 32 for "
            "Dataset 1 rat or synthetic data, or the number of full-fidelity "
            "configurations for HPOBench. SOLAR is limited by "
            "--solar-candidate-pool-size)."
        ),
    )
    parser.add_argument(
        "--length-scale",
        dest="length_scale",
        type=float,
        default=2.0,
        help="Matérn 5/2 kernel length scale (default: 2.0).",
    )
    parser.add_argument(
        "--acquisition",
        choices=ACQUISITION_CHOICES,
        default=DEFAULT_ACQUISITION,
        help=(
            "Acquisition function used by BO models "
            f"(default: {DEFAULT_ACQUISITION})."
        ),
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=2.0,
        help="UCB exploration parameter (default: 2.0).",
    )
    parser.add_argument(
        "--improvement-xi",
        dest="improvement_xi",
        type=float,
        default=0.0,
        help="EI improvement threshold (default: 0.0).",
    )
    parser.add_argument(
        "--initial",
        dest="n_initial",
        type=int,
        default=1,
        help="Number of initialization evaluations (default: 1).",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.05,
        help=(
            "Observation-noise standard deviation used by the GP and synthetic "
            "noise generator (default: 0.05)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization and response noise (default: 42).",
    )
    parser.add_argument(
        "--prior",
        action="store_true",
        help="Use a user-configurable Gaussian prior and alpha-pi shaping.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Prior influence when --prior is active (default: 0.7).",
    )
    parser.add_argument(
        "--prior-x",
        type=float,
        default=float(DEFAULT_PRIOR_CENTER[0]),
        help=(
            "X coordinate of the prior center "
            f"(default: {DEFAULT_PRIOR_CENTER[0]:.0f}, near but not at the optimum)."
        ),
    )
    parser.add_argument(
        "--prior-y",
        type=float,
        default=float(DEFAULT_PRIOR_CENTER[1]),
        help=(
            "Y coordinate of the prior center "
            f"(default: {DEFAULT_PRIOR_CENTER[1]:.0f})."
        ),
    )
    parser.add_argument(
        "--prior-spread",
        type=float,
        default=1.5,
        help="Width of the synthetic upper-right prior (default: 1.5).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hide per-query progress and print only the final result.",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        type=Path,
        const=DEFAULT_PLOT_REQUEST,
        default=None,
        metavar="FILE",
        help=(
            "Save performance and regret plots. If FILE is omitted, uses the "
            "result folder and filename for the selected experiment."
        ),
    )
    return parser


def run_synthetic(args: argparse.Namespace) -> OptimizationResult:
    """Configure and run one synthetic BO experiment."""
    candidates, ground_truth, objective = build_synthetic_problem(
        noise_std=args.noise_std,
        seed=args.seed,
    )
    if getattr(args, "method", OURS_METHOD) != OURS_METHOD:
        native_candidates = tuple(
            {"x": float(candidate[0]), "y": float(candidate[1])}
            for candidate in candidates
        )

        def baseline_objective(config: dict[str, Any]) -> float:
            return objective(np.asarray([config["x"], config["y"]], dtype=float))

        return run_registered_baseline(
            args,
            native_candidates=native_candidates,
            encoded_candidates=candidates,
            objective=baseline_objective,
            dataset_path=("synthetic-bo",),
            maximize=True,
            prior_id="gaussian-prior" if args.prior else "no-prior",
        )
    prior_center = np.array([args.prior_x, args.prior_y], dtype=float)
    prior = build_gaussian_prior(candidates, prior_center, args.prior_spread) if args.prior else None

    optimizer = BayesianOptimization(
        candidates,
        length_scale=args.length_scale,
        kappa=args.kappa,
        acquisition=args.acquisition,
        improvement_xi=args.improvement_xi,
        alpha=args.alpha if args.prior else 0.0,
        prior=prior,
        n_initial=args.n_initial,
        noise_variance=max(args.noise_std**2, 1e-12),
        random_state=args.seed,
    )

    print("HapBO-style synthetic Bayesian optimization")
    print(f"Grid: {GRID_ROWS}x{GRID_COLUMNS} = {len(candidates)} candidates")
    print(f"True optimum: ({TRUE_PEAK[0]:.0f}, {TRUE_PEAK[1]:.0f})")
    acquisition_label = ACQUISITION_LABELS[args.acquisition]
    if args.prior:
        acquisition_label = f"alpha-pi shaped {acquisition_label}"
    acquisition_parameter = (
        f"kappa={args.kappa:.2f}"
        if args.acquisition == "ucb"
        else f"xi={args.improvement_xi:.2f}"
    )
    print(
        f"Acquisition: {acquisition_label} "
        f"({acquisition_parameter}, alpha={optimizer.acquisition.alpha:.2f})"
    )
    if args.prior:
        print(
            f"Prior: Gaussian center=({args.prior_x:.2f}, {args.prior_y:.2f}), "
            f"spread={args.prior_spread:.2f}"
        )
    print()

    callback = None if args.quiet else make_progress_callback(args.queries)
    result = optimizer.run(objective, args.queries, callback=callback)
    primary_label = acquisition_label
    runs = [build_method_run("primary", primary_label, result, ground_truth)]

    for baseline_key in args.baselines:
        _, baseline_ground_truth, baseline_objective = build_synthetic_problem(
            noise_std=args.noise_std,
            seed=args.seed,
        )
        if baseline_key == VANILLA_BO_BASELINE:
            baseline_optimizer = build_vanilla_bo_optimizer(
                candidates,
                args,
                normalize_y=False,
            )
        elif baseline_key == RANDOM_SEARCH_BASELINE:
            baseline_optimizer = build_random_search_optimizer(
                candidates,
                args,
            )
        else:
            raise ValueError(f"unsupported baseline {baseline_key!r}")
        if not args.quiet:
            print(f"\nRunning baseline: {BASELINE_LABELS[baseline_key]}")
        baseline_result = baseline_optimizer.run(
            baseline_objective,
            args.queries,
            callback=callback,
        )
        runs.append(
            build_method_run(
                baseline_key,
                BASELINE_LABELS[baseline_key],
                baseline_result,
                baseline_ground_truth,
            )
        )

    expected_response = float(ground_truth[result.best_index])
    true_best_index = int(np.argmax(ground_truth))
    metrics = runs[0].metrics

    print("\nFinal result")
    print(f"  Best observed channel: {result.best_index}")
    print(f"  Best point: ({result.best_x[0]:.0f}, {result.best_x[1]:.0f})")
    print(f"  Best noisy response: {result.best_y:.4f}")
    print(f"  Expected response there: {expected_response:.4f}")
    print(f"  True best channel: {true_best_index}")
    print(f"  Evaluations: {len(result.y)}")
    print("\nPerformance metrics (using noise-free ground truth)")
    print(f"  Final simple regret: {metrics.simple_regret[-1]:.4f}")
    print(f"  Cumulative regret: {metrics.cumulative_regret[-1]:.4f}")
    print(f"  Mean instantaneous regret: {np.mean(metrics.instantaneous_regret):.4f}")
    print(
        "  Best normalized performance: "
        f"{100.0 * metrics.normalized_best_performance[-1]:.2f}%"
    )
    print(
        "  First true-optimum query: "
        f"{metrics.first_optimum_query if metrics.first_optimum_query is not None else 'not found'}"
    )
    print(
        "  First >=95% query: "
        f"{metrics.first_95_percent_query if metrics.first_95_percent_query is not None else 'not reached'}"
    )
    print_baseline_comparison(runs)

    if args.plot is not None:
        output_path = (
            SYNTHETIC_RESULTS_DIRECTORY / "bo_performance.png"
            if args.plot == DEFAULT_PLOT_REQUEST
            else args.plot
        )
        plot_path = save_performance_plot(
            runs,
            output_path=output_path,
        )
        print(f"\nPerformance plot saved to: {plot_path}")
    return result


def run_hpobench(args: argparse.Namespace) -> OptimizationResult:
    """Tune one model on one selected full-fidelity HPOBench task."""

    problem = load_hpobench_problem(
        model=args.hpobench_model,
        task=args.hpobench_task,
        benchmark_directory=args.hpobench_benchmark_directory,
    )
    if not 1 <= args.queries <= problem.n_candidates:
        raise ValueError(
            f"--queries must be between 1 and {problem.n_candidates} for "
            "the selected HPOBench table"
        )
    if not 1 <= args.n_initial <= args.queries:
        raise ValueError("--initial must be between 1 and --queries")
    benchmark_names = _names_by_lowercase(problem.parameter_names, "HPOBench parameters")

    method = getattr(args, "method", OURS_METHOD)
    if method == OURS_METHOD:
        prior_dataset_id = _path_component(
            f"hpobench-{problem.model}-{problem.task_name}"
        )
        args.dataset_prior_file = resolve_prior_path(
            prior_dataset_id,
            target_name=problem.task_name,
            prior_id=args.prior_id or DEFAULT_BENCHMARK_PRIOR_IDS["hpobench"],
            explicit_path=args.dataset_prior_file,
        )

    if method != OURS_METHOD:
        prior_path = (
            Path(args.dataset_prior_file).expanduser().resolve()
            if args.dataset_prior_file is not None
            else None
        )
        encoded_candidates = None
        if prior_path is not None:
            baseline_prior_record = load_record(prior_path)
            baseline_prior_names = tuple(
                dimension["name"]
                for dimension in baseline_prior_record["schema"]["dimensions"]
            )
            if {name.lower() for name in baseline_prior_names} != set(benchmark_names):
                raise ValueError(
                    "the HPOBench context prior dimensions must match the selected "
                    f"model parameters {list(problem.parameter_names)}"
                )
            encoded_candidates = encode_candidates_for_gp(
                problem.native_candidates,
                compile_prior(baseline_prior_record),
            )
        return run_registered_baseline(
            args,
            native_candidates=problem.native_candidates,
            encoded_candidates=encoded_candidates,
            objective=lambda config: problem.objective(config),
            dataset_path=(HPOBENCH_DATASET_ID, problem.task_name, problem.model),
            maximize=problem.maximize,
            prior=prior_path,
            prior_id=prior_path.parent.name if prior_path is not None else "no-prior",
        )

    prior_record = load_record(args.dataset_prior_file)
    if prior_record["schema"]["objective"] != "maximize":
        raise ValueError(
            "the HPOBench prior must use schema.objective='maximize' because "
            "this adapter optimizes mean validation accuracy"
        )
    prior_names = tuple(
        dimension["name"] for dimension in prior_record["schema"]["dimensions"]
    )
    if {name.lower() for name in prior_names} != set(benchmark_names):
        raise ValueError(
            "the HPOBench prior dimensions must match the selected model "
            f"parameters {list(problem.parameter_names)} up to case; got "
            f"{list(prior_names)}"
        )

    def objective(candidate: Mapping[str, Any]) -> float:
        # The optimizer passes prior-schema names; the table is keyed by its own.
        return problem.objective(
            {benchmark_names[name.lower()]: value for name, value in candidate.items()}
        )

    compiled_alpha = compile_prior(prior_record).alpha
    acquisition_parameter = (
        f"kappa={args.kappa:.2f}"
        if args.acquisition == "ucb"
        else f"xi={args.improvement_xi:.2f}"
    )
    print("HPOBench offline Bayesian optimization")
    print(f"Dataset: {problem.dataset_id}")
    print(f"Task: {problem.task_name} (OpenML task {problem.task_id})")
    print(f"Model: {problem.model}")
    print(f"Parameters: {', '.join(problem.parameter_names)}")
    print(f"Candidates: {problem.n_candidates} full-fidelity configurations")
    print(f"Full fidelity: {problem.metadata['full_fidelity']}")
    print(f"Benchmark seeds averaged: {list(problem.metadata['seeds'])}")
    print(
        "Algorithm: HapBO "
        f"(alpha-pi shaped {ACQUISITION_LABELS[args.acquisition]}, "
        f"{acquisition_parameter}, alpha={compiled_alpha:.3f})"
    )
    print(f"Prior: {Path(args.dataset_prior_file).expanduser().resolve()}")
    print(f"True best mean validation accuracy: {problem.optimum_value:.6f}")
    print()

    progress_callback: PriorGuidedCallback | None = None
    if not args.quiet:
        def show_progress(
            index: int,
            _candidate: Mapping[str, Any],
            value: float,
            optimizer: BayesianOptimization,
        ) -> None:
            _, _, best_value = optimizer.best_observation()
            print(
                f"Query {optimizer.n_observations:02d}/{args.queries}: "
                f"candidate={index}, validation_accuracy={value:.6f}, "
                f"best={best_value:.6f}, config="
                f"{json.dumps(problem.native_candidates[index], sort_keys=True)}"
            )

        progress_callback = show_progress

    guided_result = run_prior_guided_optimization(
        native_candidates=problem.native_candidates,
        objective=objective,
        max_evaluations=args.queries,
        prior_path=args.dataset_prior_file,
        length_scale=args.length_scale,
        kappa=args.kappa,
        acquisition=args.acquisition,
        improvement_xi=args.improvement_xi,
        n_initial=args.n_initial,
        noise_variance=max(args.noise_std**2, 1e-12),
        normalize_y=True,
        random_state=args.seed,
        callback=progress_callback,
    )
    result = guided_result.optimization
    runs = [
        build_method_run(
            HAPBO_METHOD,
            "HapBO",
            result,
            problem.ground_truth,
        )
    ]
    for baseline_key in args.baselines:
        if baseline_key == VANILLA_BO_BASELINE:
            optimizer = build_vanilla_bo_optimizer(
                guided_result.encoded_candidates,
                args,
                normalize_y=True,
                maximize=True,
            )
        elif baseline_key == RANDOM_SEARCH_BASELINE:
            optimizer = build_random_search_optimizer(
                guided_result.encoded_candidates,
                args,
                maximize=True,
            )
        else:
            raise ValueError(f"unsupported baseline {baseline_key!r}")
        if not args.quiet:
            print(f"\nRunning baseline: {BASELINE_LABELS[baseline_key]}")
        while optimizer.n_observations < args.queries:
            index, _ = optimizer.ask()
            value = problem.objective(problem.native_candidates[index])
            optimizer.tell(index, value)
        baseline_result = optimizer.result()
        runs.append(
            build_method_run(
                baseline_key,
                BASELINE_LABELS[baseline_key],
                baseline_result,
                problem.ground_truth,
            )
        )

    best_configuration = problem.native_candidates[result.best_index]
    metrics = runs[0].metrics
    print("\nFinal result")
    print(f"  Best candidate index: {result.best_index}")
    print(f"  Best configuration: {json.dumps(best_configuration, sort_keys=True)}")
    print(f"  Best mean validation accuracy: {result.best_y:.6f}")
    print(f"  True optimum index: {problem.optimum_index}")
    print(f"  Evaluations: {len(result.y)}")
    print("\nPerformance metrics (using the complete full-fidelity table)")
    print(f"  Final simple regret: {metrics.simple_regret[-1]:.6f}")
    print(f"  Cumulative regret: {metrics.cumulative_regret[-1]:.6f}")
    print(
        "  Best normalized performance: "
        f"{100.0 * metrics.normalized_best_performance[-1]:.2f}%"
    )
    print_baseline_comparison(runs)

    if args.plot is not None:
        output_path = (
            PROJECT_ROOT
            / "results"
            / HPOBENCH_DATASET_ID
            / _path_component(problem.task_name)
            / problem.model
            / _path_component(Path(args.dataset_prior_file).parent.name)
            / f"seed-{args.seed:04d}"
            / "bo_performance.png"
            if args.plot == DEFAULT_PLOT_REQUEST
            else args.plot
        )
        plot_path = save_performance_plot(
            runs,
            output_path,
            title=f"HPOBench: {problem.task_name} / {problem.model}",
            response_label="Mean validation accuracy",
            optimum_value=problem.optimum_value,
        )
        config_path, metrics_path = save_hpobench_run_records(
            plot_path.parent,
            problem,
            runs,
            args,
            plot_path,
        )
        print(f"\nPerformance plot saved to: {plot_path}")
        print(f"Run configuration saved to: {config_path}")
        print(f"Metrics saved to: {metrics_path}")
    return result


def _save_solar_plot(
    traces: Mapping[str, Sequence[Mapping[str, Any]]],
    labels: Mapping[str, str],
    output_path: Path,
    instance: str,
) -> Path:
    """Plot only quantities observed from live SOLAR calls (no fake regret)."""

    import matplotlib.pyplot as plt

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for key, records in traces.items():
        queries = np.arange(1, len(records) + 1)
        objectives = np.asarray([record["objective"] for record in records], dtype=float)
        feasible = np.asarray([record["feasible"] for record in records], dtype=bool)
        feasible_best = np.full(objectives.size, np.nan)
        current = np.inf
        for index, (value, is_feasible) in enumerate(zip(objectives, feasible)):
            if is_feasible:
                current = min(current, value)
            if np.isfinite(current):
                feasible_best[index] = current
        axes[0].plot(queries, objectives, marker="o", alpha=0.45, label=labels[key])
        axes[0].plot(queries, feasible_best, linewidth=2)
        violations = np.asarray(
            [record["maximum_constraint_violation"] for record in records],
            dtype=float,
        )
        axes[1].plot(queries, violations, marker="o", label=labels[key])
    axes[0].set_title("Observed objective (thin) / feasible best (thick)")
    axes[0].set_xlabel("Live simulator query")
    axes[0].set_ylabel("Raw SOLAR minimization objective")
    axes[1].set_title("Maximum constraint violation")
    axes[1].set_xlabel("Live simulator query")
    axes[1].set_ylabel("max(0, max constraint)")
    axes[1].set_yscale("symlog", linthresh=1e-6)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(f"SOLAR {instance}: live-evaluation trace")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def run_solar(args: argparse.Namespace) -> OptimizationResult:
    """Run constrained prior-guided BO against the live SOLAR executable."""

    spec = resolve_solar_instance(args.solar_instance)
    if args.solar_candidate_pool_size < 2:
        raise ValueError("--solar-candidate-pool-size must be at least 2")
    if not 1 <= args.queries <= args.solar_candidate_pool_size:
        raise ValueError(
            "--queries must be between 1 and --solar-candidate-pool-size"
        )
    if not 1 <= args.n_initial <= args.queries:
        raise ValueError("--initial must be between 1 and --queries")
    if args.solar_replications < 1:
        raise ValueError("--solar-replications must be positive")
    if args.solar_timeout is not None and args.solar_timeout <= 0:
        raise ValueError("--solar-timeout must be positive")

    expected_names = tuple(variable.name for variable in spec.variables)
    candidates = generate_solar_candidate_pool(
        spec.name,
        args.solar_candidate_pool_size,
        seed=args.seed,
        prescreen=True,
    )

    method = getattr(args, "method", OURS_METHOD)
    if method == OURS_METHOD:
        args.dataset_prior_file = resolve_prior_path(
            spec.dataset_id,
            prior_id=args.prior_id or DEFAULT_BENCHMARK_PRIOR_IDS["solar"],
            explicit_path=args.dataset_prior_file,
        )

    if method != OURS_METHOD:
        baseline_prior_path = (
            Path(args.dataset_prior_file).expanduser().resolve()
            if args.dataset_prior_file is not None
            else None
        )
        encoded_candidates = None
        if baseline_prior_path is not None:
            baseline_prior_record = load_record(baseline_prior_path)
            baseline_prior_names = tuple(
                dimension["name"]
                for dimension in baseline_prior_record["schema"]["dimensions"]
            )
            if baseline_prior_names != expected_names:
                raise ValueError(
                    "the SOLAR context prior dimensions must appear in native "
                    f"simulator order {list(expected_names)}"
                )
            encoded_candidates = encode_candidates_for_gp(
                candidates,
                compile_prior(baseline_prior_record),
            )

        def baseline_objective(config: dict[str, Any]) -> dict[str, Any]:
            evaluation = evaluate_solar(
                spec.name,
                config,
                executable=args.solar_executable,
                seed=args.solar_seed,
                fidelity=args.solar_fidelity,
                replications=args.solar_replications,
                timeout_seconds=args.solar_timeout,
            )
            return {
                "objective": 1e20 if evaluation.hidden_failure else evaluation.objective,
                "constraints": evaluation.constraints,
                "feasible": evaluation.feasible,
                "hidden_failure": evaluation.hidden_failure,
                "raw_output": evaluation.raw_output,
            }

        return run_registered_baseline(
            args,
            native_candidates=candidates,
            encoded_candidates=encoded_candidates,
            objective=baseline_objective,
            dataset_path=("solar", spec.name),
            maximize=False,
            prior=baseline_prior_path,
            prior_id=(
                baseline_prior_path.parent.name
                if baseline_prior_path is not None
                else "no-prior"
            ),
            n_constraints=spec.n_constraints,
            normalize_y=True,
        )

    prior_path = Path(args.dataset_prior_file).expanduser().resolve()
    prior_record = load_record(prior_path)
    # The simulator minimizes; instances whose objective is a negated quantity (e.g. MAXEFF_RE,
    # "minimize negative receiver efficiency") accept a prior stated as maximizing that quantity.
    stated = prior_record["schema"]["objective"]
    negated = spec.objective.startswith("minimize negative")
    if stated != "minimize" and not (stated == "maximize" and negated):
        raise ValueError(
            "the SOLAR prior must use schema.objective='minimize' "
            "('maximize' only for instances that minimize a negated quantity)"
        )
    prior_names = tuple(
        dimension["name"] for dimension in prior_record["schema"]["dimensions"]
    )
    if prior_names != expected_names:
        raise ValueError(
            "the SOLAR prior dimensions must appear in native simulator order: "
            f"{list(expected_names)}; got {list(prior_names)}"
        )

    compiler = compile_prior(prior_record)
    encoded_candidates = encode_candidates_for_gp(candidates, compiler)
    pi_values = np.asarray(compiler.pi(candidates), dtype=float).reshape(-1)
    method_keys = [HAPBO_METHOD, *args.baselines]
    method_labels = {HAPBO_METHOD: "HapBO", **BASELINE_LABELS}
    traces: dict[str, list[dict[str, Any]]] = {}
    results: dict[str, OptimizationResult] = {}

    acquisition_parameter = (
        f"kappa={args.kappa:.2f}"
        if args.acquisition == "ucb"
        else f"xi={args.improvement_xi:.2f}"
    )
    print("SOLAR live constrained Bayesian optimization")
    print(f"Instance: {spec.name} (official problem {spec.number})")
    print(f"Dimensions: {spec.n_dimensions}; constraints: {spec.n_constraints}")
    print(f"Candidate pool: {len(candidates)} (pre-screened on the documented a-priori input relations)")
    print(
        f"Algorithm: alpha-pi shaped {ACQUISITION_LABELS[args.acquisition]} "
        f"({acquisition_parameter}, alpha={compiler.alpha:.3f})"
    )
    if spec.n_constraints:
        print("Feasibility model: product of P(g_j(x) <= 0) across constraints")
    else:
        print("Feasibility model: not needed (unconstrained instance)")
    print(f"Executable: {Path(args.solar_executable).expanduser().resolve()}")
    print()

    for method_key in method_keys:
        if method_key == HAPBO_METHOD:
            method_prior = pi_values
            method_alpha = compiler.alpha
            method_initial = args.n_initial
        elif method_key == VANILLA_BO_BASELINE:
            method_prior = None
            method_alpha = 0.0
            method_initial = args.n_initial
        elif method_key == RANDOM_SEARCH_BASELINE:
            method_prior = None
            method_alpha = 0.0
            method_initial = args.queries
        else:
            raise ValueError(f"unsupported baseline {method_key!r}")
        optimizer = BayesianOptimization(
            encoded_candidates,
            length_scale=args.length_scale,
            kappa=args.kappa,
            acquisition=args.acquisition,
            improvement_xi=args.improvement_xi,
            alpha=method_alpha,
            prior=method_prior,
            n_initial=method_initial,
            noise_variance=max(args.noise_std**2, 1e-12),
            normalize_y=True,
            model_hidden_failures=True,
            maximize=False,
            random_state=args.seed,
            n_constraints=spec.n_constraints,
        )
        method_trace: list[dict[str, Any]] = []
        if len(method_keys) > 1:
            print(f"Running {method_labels[method_key]} ...")
        while optimizer.n_observations < args.queries:
            index, _ = optimizer.ask()
            candidate = candidates[index]
            started = time.perf_counter()
            evaluation = evaluate_solar(
                spec.name,
                candidate,
                executable=args.solar_executable,
                seed=args.solar_seed,
                fidelity=args.solar_fidelity,
                replications=args.solar_replications,
                timeout_seconds=args.solar_timeout,
            )
            elapsed = time.perf_counter() - started
            # Crashes are flagged and modeled by the crash surrogate; the placeholder
            # value never enters the objective or constraint GPs.
            optimizer.tell(
                index,
                0.0 if evaluation.hidden_failure else evaluation.objective,
                evaluation.constraints,
                crashed=evaluation.hidden_failure,
            )
            maximum_violation = (
                float(np.max(np.maximum(evaluation.constraints, 0.0)))
                if evaluation.constraints.size
                else 0.0
            )
            record = {
                "query": optimizer.n_observations,
                "candidate_index": index,
                "candidate": dict(candidate),
                "objective": evaluation.objective,
                "constraints": evaluation.constraints.tolist(),
                "feasible": evaluation.feasible,
                "hidden_failure": evaluation.hidden_failure,
                "maximum_constraint_violation": maximum_violation,
                "elapsed_seconds": elapsed,
                "raw_output": evaluation.raw_output,
            }
            method_trace.append(record)
            if not args.quiet:
                print(
                    f"  {method_labels[method_key]} query "
                    f"{optimizer.n_observations:02d}/{args.queries}: index={index}, "
                    f"objective={evaluation.objective:.8g}, "
                    f"feasible={evaluation.feasible}, elapsed={elapsed:.2f}s"
                )
        traces[method_key] = method_trace
        results[method_key] = optimizer.result()

    result = results[HAPBO_METHOD]
    best_position = int(np.flatnonzero(result.evaluated_indices == result.best_index)[0])
    best_record = traces[HAPBO_METHOD][best_position]
    prior_id = prior_path.parent.name
    solar_executable = Path(args.solar_executable).expanduser().resolve()
    output_directory = (
        PROJECT_ROOT
        / "results"
        / "solar"
        / spec.name.lower()
        / _path_component(prior_id)
        / f"seed-{args.seed:04d}"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    for method_key, records in traces.items():
        trace_path = output_directory / f"{method_key}_evaluations.jsonl"
        with trace_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    configuration = {
        "artifact_format": "llm-bo-solar-run-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "instance": spec.name,
        "problem_number": spec.number,
        "solar_source_commit": SOLAR_COMMIT,
        "solar_executable": str(solar_executable),
        "solar_executable_sha256": hashlib.sha256(
            solar_executable.read_bytes()
        ).hexdigest(),
        "dimension_count": spec.n_dimensions,
        "constraint_count": spec.n_constraints,
        "prior_path": str(prior_path),
        "prior_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        "candidate_pool_size": len(candidates),
        "candidate_pool_seed": args.seed,
        "simulator_seed": args.solar_seed,
        "fidelity": args.solar_fidelity,
        "replications": args.solar_replications,
        "queries": args.queries,
        "acquisition": args.acquisition,
        "alpha": compiler.alpha,
        "baselines": list(args.baselines),
        "best_hapbo_record": best_record,
    }
    (output_directory / "run_config.json").write_text(
        json.dumps(configuration, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nFinal HapBO result")
    print(f"  Best candidate index: {result.best_index}")
    print(f"  Best candidate: {json.dumps(candidates[result.best_index], sort_keys=True)}")
    print(f"  Best objective: {best_record['objective']:.8g}")
    print(f"  Feasible: {best_record['feasible']}")
    print(f"  Evaluation records: {output_directory}")
    if args.plot is not None:
        plot_path = (
            output_directory / "bo_performance.png"
            if args.plot == DEFAULT_PLOT_REQUEST
            else args.plot
        )
        saved_plot = _save_solar_plot(traces, method_labels, plot_path, spec.name)
        print(f"  Live trace plot: {saved_plot}")
    return result


def run_neuromosaic(args: argparse.Namespace) -> OptimizationResult:
    """Optimize one NeuroMOSAIC subject/muscle response map."""

    if args.dataset == NEUROMOSAIC_DATASET1_RAT_DATASET_ID:
        valid_subjects = NEUROMOSAIC_DATASET1_RAT_SUBJECT_FILES
        subject = args.subject or NEUROMOSAIC_DATASET1_RAT_DEFAULT_SUBJECT
        muscle = args.muscle or NEUROMOSAIC_DATASET1_RAT_DEFAULT_MUSCLE
        loader = load_neuromosaic_dataset1_rat_problem
        default_prior_path: Path | None = (
            RECORD_PATH
            if muscle == NEUROMOSAIC_DATASET1_RAT_DEFAULT_MUSCLE
            else None
        )
    else:
        raise ValueError(f"unsupported NeuroMOSAIC dataset {args.dataset!r}")

    if args.dataset_file is None and subject not in valid_subjects:
        raise ValueError(
            f"subject {subject!r} does not belong to {args.dataset}; "
            f"choose one of {list(valid_subjects)}"
        )
    problem = loader(
        subject_path=args.dataset_file,
        muscle=muscle,
        subject=subject,
    )

    if not 1 <= args.queries <= problem.n_candidates:
        raise ValueError(
            "--queries must be between 1 and "
            f"{problem.n_candidates} for neuromosaic-bo"
        )
    if not 1 <= args.n_initial <= args.queries:
        raise ValueError("--initial must be between 1 and --queries")

    method = getattr(args, "method", OURS_METHOD)
    if method == OURS_METHOD:
        args.dataset_prior_file = resolve_prior_path(
            problem.dataset_id,
            target_name=problem.target_name,
            prior_id=args.prior_id or DEFAULT_BENCHMARK_PRIOR_IDS["neuromosaic"],
            explicit_path=args.dataset_prior_file,
        )

    if method != OURS_METHOD:
        baseline_prior_path = (
            Path(args.dataset_prior_file).expanduser().resolve()
            if args.dataset_prior_file is not None
            else None
        )
        if (
            baseline_prior_path is None
            and default_prior_path is not None
            and problem.target_name == NEUROMOSAIC_DATASET1_RAT_DEFAULT_MUSCLE
        ):
            baseline_prior_path = Path(default_prior_path).expanduser().resolve()
        if baseline_prior_path is not None:
            validate_prior_compatibility(
                baseline_prior_path,
                dataset_id=problem.dataset_id,
                target_name=problem.target_name,
            )
        return run_registered_baseline(
            args,
            native_candidates=problem.native_candidates,
            encoded_candidates=problem.encoded_candidates,
            objective=lambda config: problem.objective(config),
            dataset_path=(
                problem.dataset_id,
                problem.subject_id,
                _path_component(problem.target_name),
            ),
            maximize=problem.maximize,
            prior=baseline_prior_path,
            prior_id=(
                baseline_prior_path.parent.name
                if baseline_prior_path is not None
                else "no-prior"
            ),
        )

    callback = (
        None
        if args.quiet
        else make_offline_progress_callback(problem, args.queries)
    )
    prior_record = load_record(args.dataset_prior_file)
    validate_prior_compatibility(
        args.dataset_prior_file,
        dataset_id=problem.dataset_id,
        target_name=problem.target_name,
    )
    schema = prior_record["schema"]
    dimension_names = [dimension["name"] for dimension in schema["dimensions"]]
    if schema["objective"] != "maximize":
        raise ValueError("the dataset prior must use schema.objective='maximize'")
    prior_candidates, to_dataset_candidate = adapt_neuromosaic_candidates_to_prior(
        problem,
        dimension_names,
    )

    valid_repetitions = problem.metadata["valid_repetitions_per_candidate"]
    optimum_candidate = problem.optimum_candidate
    compiled_alpha = compile_prior(prior_record).alpha
    print("NeuroMOSAIC offline Bayesian optimization")
    print(f"Dataset: {problem.dataset_id}")
    print(f"Subject: {problem.subject_id}")
    print(f"Target muscle: {problem.target_name}")
    candidate_coordinates = np.asarray(problem.encoded_candidates, dtype=float)
    x_span = int(np.ptp(candidate_coordinates[:, 0]) + 1)
    y_span = int(np.ptp(candidate_coordinates[:, 1]) + 1)
    print(
        f"Candidates: {problem.n_candidates} electrodes within a "
        f"{x_span}x{y_span} coordinate grid"
    )
    print(
        "Valid repetitions per electrode: "
        f"{int(np.min(valid_repetitions))}..{int(np.max(valid_repetitions))}"
    )
    acquisition_parameter = (
        f"kappa={args.kappa:.2f}"
        if args.acquisition == "ucb"
        else f"xi={args.improvement_xi:.2f}"
    )
    print(
        "Algorithm: HapBO "
        f"(alpha-pi shaped {ACQUISITION_LABELS[args.acquisition]}, "
        f"{acquisition_parameter}, alpha={compiled_alpha:.3f})"
    )
    print(f"Prior: {Path(args.dataset_prior_file).expanduser().resolve()}")
    print(
        "True optimum: "
        f"({optimum_candidate['channel_x']}, {optimum_candidate['channel_y']}), "
        f"mean response={problem.optimum_value:.4f}"
    )
    print()

    def llm_objective(candidate: Mapping[str, Any]) -> float:
        return problem.objective(to_dataset_candidate(candidate))

    guided_callback: PriorGuidedCallback | None = None
    if callback is not None:
        def show_guided_progress(
            index: int,
            candidate: Mapping[str, Any],
            value: float,
            optimizer: BayesianOptimization,
        ) -> None:
            callback(index, to_dataset_candidate(candidate), value, optimizer)

        guided_callback = show_guided_progress

    guided_result = run_prior_guided_optimization(
        native_candidates=prior_candidates,
        encoded_candidates=problem.encoded_candidates,
        objective=llm_objective,
        max_evaluations=args.queries,
        prior_path=args.dataset_prior_file,
        length_scale=args.length_scale,
        kappa=args.kappa,
        acquisition=args.acquisition,
        improvement_xi=args.improvement_xi,
        n_initial=args.n_initial,
        noise_variance=max(args.noise_std**2, 1e-12),
        normalize_y=True,
        random_state=args.seed,
        callback=guided_callback,
    )
    result = guided_result.optimization

    runs = [
        build_method_run(
            HAPBO_METHOD,
            "HapBO",
            result,
            problem.ground_truth,
        )
    ]
    for baseline_key in args.baselines:
        if baseline_key == VANILLA_BO_BASELINE:
            baseline_optimizer = build_vanilla_bo_optimizer(
                problem.encoded_candidates,
                args,
                normalize_y=True,
                maximize=problem.maximize,
            )
        elif baseline_key == RANDOM_SEARCH_BASELINE:
            baseline_optimizer = build_random_search_optimizer(
                problem.encoded_candidates,
                args,
                maximize=problem.maximize,
            )
        else:
            raise ValueError(f"unsupported baseline {baseline_key!r}")
        if not args.quiet:
            print(f"\nRunning baseline: {BASELINE_LABELS[baseline_key]}")
        while baseline_optimizer.n_observations < args.queries:
            index, _ = baseline_optimizer.ask()
            candidate = dict(problem.native_candidates[index])
            value = problem.objective(candidate)
            baseline_optimizer.tell(index, value)
            if callback is not None:
                callback(index, candidate, value, baseline_optimizer)
        baseline_result = baseline_optimizer.result()
        runs.append(
            build_method_run(
                baseline_key,
                BASELINE_LABELS[baseline_key],
                baseline_result,
                problem.ground_truth,
            )
        )

    best_candidate = problem.native_candidates[result.best_index]
    metrics = runs[0].metrics

    print("\nFinal result")
    print(f"  Best observed channel: {result.best_index + 1}")
    print(
        "  Best point: "
        f"({best_candidate['channel_x']}, {best_candidate['channel_y']})"
    )
    print(f"  Best mean valid-trial response: {result.best_y:.4f}")
    print(f"  True best channel: {problem.optimum_index + 1}")
    print(f"  Evaluations: {len(result.y)}")
    print("\nPerformance metrics (using the complete offline response map)")
    print(f"  Final simple regret: {metrics.simple_regret[-1]:.4f}")
    print(f"  Cumulative regret: {metrics.cumulative_regret[-1]:.4f}")
    print(f"  Mean instantaneous regret: {np.mean(metrics.instantaneous_regret):.4f}")
    print(
        "  Best normalized performance: "
        f"{100.0 * metrics.normalized_best_performance[-1]:.2f}%"
    )
    print(
        "  First true-optimum query: "
        f"{metrics.first_optimum_query if metrics.first_optimum_query is not None else 'not found'}"
    )
    print(
        "  First >=95% query: "
        f"{metrics.first_95_percent_query if metrics.first_95_percent_query is not None else 'not reached'}"
    )
    print_baseline_comparison(runs)
    if args.plot is not None:
        output_path = (
            NEUROMOSAIC_RESULTS_DIRECTORY
            / problem.dataset_id
            / problem.subject_id
            / _path_component(problem.target_name)
            / _path_component(Path(args.dataset_prior_file).parent.name)
            / f"seed-{args.seed:04d}"
            / "bo_performance.png"
            if args.plot == DEFAULT_PLOT_REQUEST
            else args.plot
        )
        plot_path = save_offline_performance_plot(
            problem,
            runs,
            output_path=output_path,
        )
        config_path, metrics_path = save_neuromosaic_run_records(
            plot_path.parent,
            problem,
            runs,
            args,
            plot_path,
        )
        print(f"\nPerformance plot saved to: {plot_path}")
        print(f"Run configuration saved to: {config_path}")
        print(f"Metrics saved to: {metrics_path}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if len(set(args.baselines)) != len(args.baselines):
        parser.error("--baselines cannot contain duplicate names")
    if args.method != OURS_METHOD and args.baselines:
        parser.error("--method cannot select a baseline together with --baselines")
    if args.method != OURS_METHOD and args.command in {
        "prior-init",
        "prior-finalize",
        "landscape-test",
    }:
        parser.error("--method is only available for BO commands")

    if args.command == "prior-init":
        if args.dataset_id is None or args.prior_id is None:
            parser.error("prior-init requires --dataset-id and --prior-id")
        dataset_id = args.dataset_id
        if args.prior_target_name is not None and args.muscle is not None:
            if args.prior_target_name != args.muscle:
                parser.error(
                    "--prior-target-name and --muscle must match when both are set"
                )
        target_name = args.prior_target_name or args.muscle
        if dataset_id == NEUROMOSAIC_DATASET1_RAT_DATASET_ID and target_name is None:
            parser.error(
                "NeuroMOSAIC prior-init requires --muscle so the prior is stored "
                "and validated for one exact muscle"
            )
        try:
            prior_directory = initialize_prior_experiment(
                dataset_id,
                args.prior_id,
                provider=args.llm_provider,
                model=args.llm_model,
                subject_scope=args.prior_subject_scope,
                target=args.prior_target,
                target_name=target_name,
                card_human_statement=args.card_human_statement,
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(f"Draft prior directory created: {prior_directory}")
        print(f"Edit prompt card: {prior_directory / 'prompt_card.md'}")
        print(f"Shared vocabulary: {VOCABULARY_CARD_PATH}")
        print(f"Save the generated prior to: {prior_directory / 'prior.json'}")
        print(
            "Then finalize it with: python main.py prior-finalize "
            f"--prior-directory {prior_directory}"
        )
        return 0

    if args.command == "prior-finalize":
        if args.prior_directory is None:
            parser.error("prior-finalize requires --prior-directory")
        try:
            finalize_prior_experiment(
                args.prior_directory,
                output_directory=args.prior_output_directory,
                generate_landscape=not args.no_landscape,
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(f"Prior finalized: {args.prior_directory.expanduser().resolve()}")
        return 0

    if args.command == "landscape-test":
        if args.baselines:
            parser.error("--baselines is only available for BO commands")
        run_landscape_generation(
            record_path=args.prior_file,
            output_directory=args.prior_output_directory,
        )
        return 0

    if args.command == "neuromosaic-bo":
        if args.length_scale <= 0:
            parser.error("--length-scale must be positive")
        if args.kappa < 0:
            parser.error("--kappa must be non-negative")
        if args.improvement_xi < 0:
            parser.error("--improvement-xi must be non-negative")
        if args.noise_std < 0:
            parser.error("--noise-std must be non-negative")
        try:
            run_neuromosaic(args)
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        return 0

    if args.command == "hpobench-bo":
        if args.length_scale <= 0:
            parser.error("--length-scale must be positive")
        if args.kappa < 0:
            parser.error("--kappa must be non-negative")
        if args.improvement_xi < 0:
            parser.error("--improvement-xi must be non-negative")
        if args.noise_std < 0:
            parser.error("--noise-std must be non-negative")
        try:
            run_hpobench(args)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        return 0

    if args.command == "solar-bo":
        if args.length_scale <= 0:
            parser.error("--length-scale must be positive")
        if args.kappa < 0:
            parser.error("--kappa must be non-negative")
        if args.improvement_xi < 0:
            parser.error("--improvement-xi must be non-negative")
        if args.noise_std < 0:
            parser.error("--noise-std must be non-negative")
        try:
            run_solar(args)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        return 0

    n_candidates = GRID_ROWS * GRID_COLUMNS
    if not 1 <= args.queries <= n_candidates:
        parser.error(f"--queries must be between 1 and {n_candidates}")
    if not 1 <= args.n_initial <= args.queries:
        parser.error("--initial must be between 1 and --queries")
    if args.length_scale <= 0:
        parser.error("--length-scale must be positive")
    if args.kappa < 0:
        parser.error("--kappa must be non-negative")
    if args.improvement_xi < 0:
        parser.error("--improvement-xi must be non-negative")
    if args.noise_std < 0:
        parser.error("--noise-std must be non-negative")
    if args.prior and not 0.0 <= args.alpha <= ALPHA_MAX:
        parser.error(f"--alpha must be between 0 and {ALPHA_MAX}")
    if args.prior and not 1.0 <= args.prior_x <= GRID_COLUMNS:
        parser.error(f"--prior-x must be between 1 and {GRID_COLUMNS}")
    if args.prior and not 1.0 <= args.prior_y <= GRID_ROWS:
        parser.error(f"--prior-y must be between 1 and {GRID_ROWS}")
    if args.prior_spread <= 0:
        parser.error("--prior-spread must be positive")

    run_synthetic(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
