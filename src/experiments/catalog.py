"""Load and validate human-authored research catalogs and study manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"
CATALOG_ROOT = EXPERIMENT_ROOT / "catalog"
STUDY_ROOT = EXPERIMENT_ROOT / "studies"
RESULT_ROOT = PROJECT_ROOT / "results" / "studies"


class StudyConfigurationError(ValueError):
    """Raised when a catalog or study manifest is structurally invalid."""


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StudyConfigurationError(f"{label} must be a mapping")
    return dict(value)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load one YAML mapping with a path-aware validation error."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise StudyConfigurationError(f"invalid YAML in {source}: {exc}") from exc
    return _mapping(value, str(source))


@dataclass(frozen=True)
class Catalog:
    """Resolved benchmark and method entries keyed by stable identifiers."""

    benchmarks: dict[str, dict[str, Any]]
    methods: dict[str, dict[str, Any]]


def _load_entries(directory: Path, artifact_format: str, id_field: str) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    for path in sorted(directory.rglob("*.yaml")):
        document = load_yaml(path)
        if document.get("artifact_format") != artifact_format:
            raise StudyConfigurationError(
                f"{path}: artifact_format must be {artifact_format!r}"
            )
        identifier = document.get(id_field)
        if not isinstance(identifier, str) or not identifier:
            raise StudyConfigurationError(f"{path}: {id_field} must be a non-empty string")
        if identifier in entries:
            raise StudyConfigurationError(
                f"duplicate {id_field} {identifier!r} in {path}"
            )
        document["_source_path"] = str(path.resolve())
        entries[identifier] = document
    return entries


def load_catalog(root: str | Path = CATALOG_ROOT) -> Catalog:
    """Load the benchmark and method catalogs and reject duplicate identifiers."""

    root_path = Path(root).expanduser().resolve()
    return Catalog(
        benchmarks=_load_entries(
            root_path / "benchmarks",
            "llm-bo-benchmark-catalog-v1",
            "benchmark_id",
        ),
        methods=_load_entries(
            root_path / "methods",
            "llm-bo-method-catalog-v1",
            "method_id",
        ),
    )


def load_study(path: str | Path) -> dict[str, Any]:
    """Load the main ``study.yaml`` from a path or study directory."""

    source = Path(path).expanduser().resolve()
    if source.is_dir():
        source = source / "study.yaml"
    study = load_yaml(source)
    if study.get("artifact_format") != "llm-bo-study-v1":
        raise StudyConfigurationError(
            f"{source}: artifact_format must be 'llm-bo-study-v1'"
        )
    for field in ("study_id", "phase", "status", "execution_mode"):
        if not isinstance(study.get(field), str) or not study[field]:
            raise StudyConfigurationError(f"{source}: {field} is required")
    if study["status"] != "ready":
        raise StudyConfigurationError(f"{source}: status must be 'ready', got {study['status']!r}")
    if study["execution_mode"] != "optimization":
        raise StudyConfigurationError(
            f"{source}: execution_mode must be 'optimization', got {study['execution_mode']!r}"
        )
    study["_source_path"] = str(source)
    return study
