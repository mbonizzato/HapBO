"""Expand a study manifest into reproducible, one-method optimization runs."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .catalog import Catalog, PROJECT_ROOT, RESULT_ROOT, StudyConfigurationError


def slug(value: Any) -> str:
    """Return a stable path component for a catalog value."""

    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", str(value))
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    if not text:
        raise StudyConfigurationError(f"cannot create a path component from {value!r}")
    return text


@dataclass(frozen=True)
class PlannedRun:
    """One fully specified optimizer trajectory."""

    run_id: str
    phase: str
    study_id: str
    study_source_path: Path
    benchmark_id: str
    problem_id: str
    method_id: str
    budget: int
    seed: int
    replicate: int
    parameters: dict[str, Any]
    optimizer: dict[str, Any]
    prior_path: Path | None
    result_directory: Path
    command: tuple[str, ...] | None
    blocked_reasons: tuple[str, ...]

    @property
    def executable(self) -> bool:
        return not self.blocked_reasons and self.command is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "phase": self.phase,
            "study_id": self.study_id,
            "study_source_path": str(self.study_source_path),
            "benchmark_id": self.benchmark_id,
            "problem_id": self.problem_id,
            "method_id": self.method_id,
            "budget": self.budget,
            "seed": self.seed,
            "replicate": self.replicate,
            "parameters": self.parameters,
            "optimizer": self.optimizer,
            "prior_path": None if self.prior_path is None else str(self.prior_path),
            "result_directory": str(self.result_directory),
            "command": None if self.command is None else list(self.command),
            "blocked_reasons": list(self.blocked_reasons),
        }


@dataclass(frozen=True)
class StudyPlan:
    """An expanded study and its planned runs."""

    study: dict[str, Any]
    runs: tuple[PlannedRun, ...]

    @property
    def blocked_count(self) -> int:
        return sum(not run.executable for run in self.runs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_format": "llm-bo-study-plan-v1",
            "study_id": self.study["study_id"],
            "phase": self.study["phase"],
            "status": self.study["status"],
            "execution_mode": self.study["execution_mode"],
            "run_count": len(self.runs),
            "blocked_count": self.blocked_count,
            "runs": [run.to_dict() for run in self.runs],
        }


def _catalog_cases(benchmark: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "cases" in benchmark:
        cases = benchmark["cases"]
        if not isinstance(cases, list) or not all(isinstance(case, Mapping) for case in cases):
            raise StudyConfigurationError(
                f"benchmark {benchmark['benchmark_id']}: cases must be mappings"
            )
        return [dict(case) for case in cases]
    axes = benchmark.get("axes", {})
    if not isinstance(axes, Mapping) or not axes:
        return [{}]
    names = list(axes)
    value_lists = []
    for name in names:
        values = axes[name]
        if not isinstance(values, list) or not values:
            raise StudyConfigurationError(
                f"benchmark {benchmark['benchmark_id']}: axis {name!r} is empty"
            )
        value_lists.append(values)
    return [dict(zip(names, values)) for values in itertools.product(*value_lists)]


def _selected_cases(
    benchmark: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    cases = _catalog_cases(benchmark)
    include = selection.get("include", "all")
    if include != "all":
        if not isinstance(include, list):
            raise StudyConfigurationError("benchmark include must be 'all' or a list")
        included = {str(value) for value in include}
        cases = [case for case in cases if _problem_id(benchmark, case) in included]

    variants = selection.get("variants", [{}])
    if not isinstance(variants, list) or not all(
        isinstance(variant, Mapping) for variant in variants
    ):
        raise StudyConfigurationError("benchmark variants must be a list of mappings")
    return [{**case, **dict(variant)} for case in cases for variant in variants]


def _problem_id(benchmark: Mapping[str, Any], parameters: Mapping[str, Any]) -> str:
    template = benchmark.get("problem_id_template", benchmark["benchmark_id"])
    try:
        return str(template).format(**parameters)
    except KeyError as exc:
        raise StudyConfigurationError(
            f"benchmark {benchmark['benchmark_id']}: problem template needs {exc.args[0]!r}"
        ) from exc


def _prior_path(selection: Mapping[str, Any], parameters: Mapping[str, Any]) -> Path | None:
    template = selection.get("prior_path")
    if template is None:
        return None
    try:
        rendered = str(template).format(**parameters)
    except KeyError as exc:
        raise StudyConfigurationError(f"prior_path needs {exc.args[0]!r}") from exc
    path = Path(rendered).expanduser()
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _command(
    benchmark: Mapping[str, Any],
    method: Mapping[str, Any],
    prior_path: Path | None,
    result_directory: Path,
) -> tuple[tuple[str, ...] | None, list[str]]:
    reasons: list[str] = []
    if benchmark.get("implementation_status") != "available":
        reasons.append(f"benchmark adapter is {benchmark.get('implementation_status', 'undefined')}")
    if method.get("implementation_status") != "available":
        reasons.append(f"method is {method.get('implementation_status', 'undefined')}")
    if not method.get("isolated_worker", False):
        reasons.append("method has no isolated study-worker adapter")
    if method.get("requires_prior", False):
        if prior_path is None:
            reasons.append("prior path is not assigned")
        elif not prior_path.is_file():
            reasons.append(f"prior file does not exist: {prior_path}")
    if not isinstance(benchmark.get("adapter"), str):
        reasons.append("benchmark has no study-worker adapter")
    if reasons:
        return None, reasons

    return (
        "{python}",
        "scripts/run_experiment.py",
        "--run-directory",
        str(result_directory),
    ), reasons


def _budget_values(study: Mapping[str, Any]) -> list[int]:
    values = study.get("design", {}).get("budgets")
    if not isinstance(values, list) or not values:
        raise StudyConfigurationError("design.budgets must be a non-empty list of positive budgets")
    budgets = [int(value) for value in values]
    if any(value < 1 for value in budgets):
        raise StudyConfigurationError("budgets must be positive")
    return budgets


# Method strategies whose runs fit the GP surrogate of BayesianOptimization (experiments.worker._build_optimizer).
# design.optimizer.kernel has no effect on the other strategies (external and external-mf baselines with their own
# models, random search and prior-greedy, which never fit the GP), so a study setting it may use only these.
GP_KERNEL_STRATEGIES = ("alpha-pi", "vanilla-bo", "pi-bo", "prior-to-mean", "prior-warm-start", "llm-ei")


def _validate_kernel(name: Any) -> str:
    """The GP kernel check of the surrogate (``ValueError`` on an unknown name)."""

    try:  # src/ on sys.path (scripts, study worker)
        from bayesian_optimization.surrogate import validate_kernel
    except ImportError:  # imported as the ``src.experiments`` package
        from src.bayesian_optimization.surrogate import validate_kernel
    return validate_kernel(name)


def build_study_plan(study: Mapping[str, Any], catalog: Catalog) -> StudyPlan:
    """Expand a study manifest into its planned runs."""

    study_document = dict(study)
    design = study_document.get("design")
    if not isinstance(design, Mapping):
        raise StudyConfigurationError("a study requires a design mapping")
    benchmark_selections = design.get("benchmarks")
    method_ids = design.get("methods")
    seeds = design.get("seeds")
    if not isinstance(benchmark_selections, list) or not benchmark_selections:
        raise StudyConfigurationError("design.benchmarks must be a non-empty list")
    if not isinstance(method_ids, list) or not method_ids:
        raise StudyConfigurationError("design.methods must be a non-empty list")
    if not isinstance(seeds, list) or not seeds:
        raise StudyConfigurationError("design.seeds must be a non-empty list")
    if len({int(seed) for seed in seeds}) != len(seeds):
        raise StudyConfigurationError("design.seeds must not contain duplicates")
    budgets = _budget_values(study_document)
    optimizer = design.get("optimizer", {})
    if not isinstance(optimizer, Mapping):
        raise StudyConfigurationError("design.optimizer must be a mapping")
    if "kernel" in optimizer:
        try:
            _validate_kernel(optimizer["kernel"])
        except ValueError as exc:
            raise StudyConfigurationError(f"design.optimizer.kernel: {exc}") from exc

    runs: list[PlannedRun] = []
    for raw_selection in benchmark_selections:
        selection = _ensure_mapping(raw_selection, "benchmark selection")
        selected_method_ids = selection.get("methods", method_ids)
        if not isinstance(selected_method_ids, list) or not selected_method_ids:
            raise StudyConfigurationError(
                "benchmark selection methods must be a non-empty list"
            )
        benchmark_id = selection.get("benchmark_id")
        if benchmark_id not in catalog.benchmarks:
            raise StudyConfigurationError(f"unknown benchmark {benchmark_id!r}")
        benchmark = catalog.benchmarks[benchmark_id]
        for parameters in _selected_cases(benchmark, selection):
            problem_id = _problem_id(benchmark, parameters)
            prior_path = _prior_path(selection, parameters)
            for method_id, budget, seed_position in itertools.product(
                selected_method_ids,
                budgets,
                range(len(seeds)),
            ):
                if method_id not in catalog.methods:
                    raise StudyConfigurationError(f"unknown method {method_id!r}")
                method = catalog.methods[method_id]
                if "kernel" in optimizer and method.get("strategy") not in GP_KERNEL_STRATEGIES:
                    raise StudyConfigurationError(
                        f"design.optimizer.kernel: method {method_id!r} (strategy {method.get('strategy')!r}) "
                        f"does not fit the GP surrogate; the kernel applies to strategies {list(GP_KERNEL_STRATEGIES)}"
                    )
                seed = int(seeds[seed_position])
                replicate = seed_position + 1
                result_directory = (
                    RESULT_ROOT
                    / slug(study_document["phase"])
                    / slug(study_document["study_id"])
                    / "runs"
                    / slug(benchmark_id)
                    / slug(problem_id)
                    / slug(method_id)
                    / f"budget-{budget:04d}"
                    / f"seed-{seed:04d}"
                )
                command, reasons = _command(benchmark, method, prior_path, result_directory)
                identity = {
                    "study": study_document["study_id"],
                    "benchmark": benchmark_id,
                    "problem": problem_id,
                    "method": method_id,
                    "budget": budget,
                    "seed": seed,
                    "replicate": replicate,
                }
                digest = hashlib.sha256(
                    json.dumps(identity, sort_keys=True).encode("utf-8")
                ).hexdigest()[:10]
                run_id = "__".join(
                    (
                        slug(benchmark_id),
                        slug(problem_id),
                        slug(method_id),
                        f"b{budget}",
                        f"s{seed:04d}",
                        digest,
                    )
                )
                runs.append(
                    PlannedRun(
                        run_id=run_id,
                        phase=study_document["phase"],
                        study_id=study_document["study_id"],
                        study_source_path=Path(study_document["_source_path"]),
                        benchmark_id=benchmark_id,
                        problem_id=problem_id,
                        method_id=method_id,
                        budget=budget,
                        seed=seed,
                        replicate=replicate,
                        parameters=dict(parameters),
                        optimizer=dict(optimizer),
                        prior_path=prior_path,
                        result_directory=result_directory,
                        command=command,
                        blocked_reasons=tuple(reasons),
                    )
                )
    return StudyPlan(study_document, tuple(runs))


def _ensure_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StudyConfigurationError(f"{label} must be a mapping")
    return dict(value)
