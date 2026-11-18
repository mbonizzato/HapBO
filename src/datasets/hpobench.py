"""Offline adapter and downloader for selected HPOBench tabular tasks."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HPOBENCH_DIRECTORY = PROJECT_ROOT / "benchmark" / "HPOBench"
DATASET_ID = "hpobench"
MODELS = ("lr", "nn", "rf", "svm", "xgb")
DEFAULT_MODEL = "rf"
TASKS = {
    "credit_g": 31,
    "vehicle": 53,
    "kc1": 3917,
    "phoneme": 9952,
    "blood_transfusion": 10101,
    "australian": 146818,
    "car": 146821,
    "segment": 146822,
}
DEFAULT_TASK = "credit_g"
DOWNLOAD_URLS = {
    "lr": "https://ndownloader.figshare.com/files/30379038",
    "nn": "https://ndownloader.figshare.com/files/30379005",
    "rf": "https://ndownloader.figshare.com/files/30469089",
    "svm": "https://ndownloader.figshare.com/files/30379359",
    "xgb": "https://ndownloader.figshare.com/files/30469920",
}


def _python_scalar(value: Any) -> Any:
    """Convert NumPy scalar values into stable dictionary/JSON values."""

    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hpobench_task_path(
    model: str,
    task: str,
    *,
    benchmark_directory: str | Path = HPOBENCH_DIRECTORY,
) -> tuple[Path, Path]:
    """Return the official parquet and metadata paths for one selected task."""

    if model not in MODELS:
        raise ValueError(f"unknown HPOBench model {model!r}; choose one of {MODELS}")
    if task not in TASKS:
        raise ValueError(
            f"unknown HPOBench task {task!r}; choose one of {tuple(TASKS)}"
        )
    root = Path(benchmark_directory).expanduser().resolve()
    task_id = TASKS[task]
    directory = root / model / str(task_id)
    return (
        directory / f"{model}_{task_id}_data.parquet.gzip",
        directory / f"{model}_{task_id}_metadata.json",
    )


def _configuration_space_document(serialized_space: Any, label: str) -> Mapping[str, Any]:
    if isinstance(serialized_space, str):
        try:
            space = json.loads(serialized_space)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid {label} ConfigSpace JSON") from exc
    elif isinstance(serialized_space, Mapping):
        space = serialized_space
    else:
        raise ValueError(f"{label} ConfigSpace must be JSON text or an object")
    return space


def _configuration_space_names(serialized_space: Any, label: str) -> tuple[str, ...]:
    """Read hyperparameter names from HPOBench's ConfigSpace JSON snapshot."""

    space = _configuration_space_document(serialized_space, label)
    hyperparameters = space.get("hyperparameters")
    if not isinstance(hyperparameters, list) or not hyperparameters:
        raise ValueError(f"{label} ConfigSpace has no hyperparameters")
    names = tuple(
        hyperparameter.get("name")
        for hyperparameter in hyperparameters
        if isinstance(hyperparameter, Mapping)
    )
    if len(names) != len(hyperparameters) or any(not name for name in names):
        raise ValueError(f"{label} ConfigSpace contains an unnamed hyperparameter")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} ConfigSpace contains duplicate names")
    return names


def _maximum_fidelity_values(serialized_space: Any) -> dict[str, Any]:
    """Recover HPOBench's maximum discrete fidelity without ConfigSpace."""

    space = _configuration_space_document(serialized_space, "fidelity")
    values: dict[str, Any] = {}
    for hyperparameter in space.get("hyperparameters", []):
        if not isinstance(hyperparameter, Mapping) or "name" not in hyperparameter:
            raise ValueError("fidelity ConfigSpace contains an invalid entry")
        name = hyperparameter["name"]
        sequence = hyperparameter.get("sequence", hyperparameter.get("choices"))
        if sequence is not None:
            if not isinstance(sequence, list) or not sequence:
                raise ValueError(f"fidelity {name!r} has no discrete values")
            try:
                value = max(sequence)
            except TypeError as exc:
                raise ValueError(f"fidelity {name!r} values are not ordered") from exc
        elif "value" in hyperparameter:
            value = hyperparameter["value"]
        elif "upper" in hyperparameter:
            value = hyperparameter["upper"]
        else:
            raise ValueError(f"cannot determine maximum fidelity for {name!r}")
        values[name] = _python_scalar(value)
    if not values:
        raise ValueError("fidelity ConfigSpace has no hyperparameters")
    return values


def hpobench_search_space_summary(
    model: str,
    task: str,
    *,
    benchmark_directory: str | Path = HPOBENCH_DIRECTORY,
) -> dict[str, Any]:
    """Build a prior-friendly finite search-space summary from task metadata."""

    _, metadata_path = hpobench_task_path(
        model,
        task,
        benchmark_directory=benchmark_directory,
    )
    if not metadata_path.is_file():
        raise FileNotFoundError(f"HPOBench metadata file not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config_spaces = metadata["config_spaces"]
    discrete_raw = config_spaces["x_discrete"]
    original_raw = config_spaces.get("x", discrete_raw)
    discrete = json.loads(discrete_raw) if isinstance(discrete_raw, str) else discrete_raw
    original = json.loads(original_raw) if isinstance(original_raw, str) else original_raw
    original_by_name = {
        item["name"]: item
        for item in original.get("hyperparameters", [])
        if isinstance(item, Mapping) and "name" in item
    }
    dimensions = []
    for hyperparameter in discrete.get("hyperparameters", []):
        if not isinstance(hyperparameter, Mapping) or "name" not in hyperparameter:
            raise ValueError("configuration ConfigSpace contains an invalid entry")
        name = hyperparameter["name"]
        values = hyperparameter.get("sequence", hyperparameter.get("choices"))
        if values is None and "value" in hyperparameter:
            values = [hyperparameter["value"]]
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"discrete ConfigSpace entry {name!r} has no finite values"
            )
        native_values = [_python_scalar(value) for value in values]
        categorical = any(isinstance(value, (str, bool)) for value in native_values)
        original_definition = original_by_name.get(name, {})
        dimensions.append(
            {
                "name": name,
                "type": "categorical" if categorical else "ordinal",
                "values": native_values,
                "scale": "log" if original_definition.get("log") else "linear",
            }
        )
    return {
        "artifact_format": "llm-bo-hpobench-search-space-v1",
        "dataset_id": DATASET_ID,
        "model": model,
        "task": task,
        "openml_task_id": TASKS[task],
        "objective": "maximize",
        "target": "mean validation accuracy on a held-out split, averaged over several random seeds",
        "dimensions": dimensions,
    }


def _validation_accuracy(result: Any) -> float:
    """Extract validation accuracy from one official HPOBench result cell."""

    if not isinstance(result, Mapping):
        raise ValueError("each HPOBench result cell must contain a mapping")
    try:
        accuracy = float(result["info"]["val_scores"]["acc"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "each HPOBench result must contain info.val_scores.acc"
        ) from exc
    if not np.isfinite(accuracy):
        raise ValueError("HPOBench validation accuracy must be finite")
    return accuracy


@dataclass(frozen=True)
class HPOBenchProblem:
    """One finite, full-fidelity HPOBench model/task optimization problem."""

    dataset_id: str
    model: str
    task_name: str
    task_id: int
    parameter_names: tuple[str, ...]
    native_candidates: tuple[dict[str, Any], ...]
    ground_truth: np.ndarray
    objective: Callable[[Mapping[str, Any]], float]
    metadata: dict[str, Any]
    maximize: bool = True

    @property
    def n_candidates(self) -> int:
        return len(self.native_candidates)

    @property
    def optimum_index(self) -> int:
        return int(np.argmax(self.ground_truth))

    @property
    def optimum_candidate(self) -> dict[str, Any]:
        return dict(self.native_candidates[self.optimum_index])

    @property
    def optimum_value(self) -> float:
        return float(self.ground_truth[self.optimum_index])


def load_hpobench_problem(
    *,
    model: str = DEFAULT_MODEL,
    task: str = DEFAULT_TASK,
    benchmark_directory: str | Path = HPOBENCH_DIRECTORY,
) -> HPOBenchProblem:
    """Load one selected table at maximum fidelity and average its seed scores."""

    parquet_path, metadata_path = hpobench_task_path(
        model,
        task,
        benchmark_directory=benchmark_directory,
    )
    missing = [path for path in (parquet_path, metadata_path) if not path.is_file()]
    if missing:
        relative = [str(path) for path in missing]
        raise FileNotFoundError(
            "HPOBench task files are missing: "
            f"{relative}. Run `python scripts/download_hpobench.py "
            f"--model {model} --task {task}` first."
        )
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "loading HPOBench tables requires pandas and pyarrow; update the "
            "environment with `conda env update -f environment.yaml --prune`"
        ) from exc

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read HPOBench metadata: {metadata_path}") from exc
    try:
        config_spaces = metadata["config_spaces"]
        parameter_names = _configuration_space_names(
            config_spaces["x_discrete"],
            "configuration",
        )
        fidelity_names = _configuration_space_names(
            config_spaces["z_discrete"], "fidelity"
        )
        full_fidelity = _maximum_fidelity_values(config_spaces["z_discrete"])
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "HPOBench metadata must contain x_discrete and z_discrete ConfigSpaces"
        ) from exc

    try:
        table = pd.read_parquet(
            parquet_path,
            filters=[(name, "==", value) for name, value in full_fidelity.items()],
        )
    except Exception as exc:
        raise ValueError(f"cannot read HPOBench parquet table: {parquet_path}") from exc
    required_columns = {*parameter_names, *fidelity_names, "seed", "result"}
    missing_columns = required_columns - set(table.columns)
    if missing_columns:
        raise ValueError(
            f"HPOBench table is missing columns {sorted(missing_columns)}"
        )

    if table.empty:
        raise ValueError("HPOBench table contains no maximum-fidelity rows")

    score_lists: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    seed_sets: dict[tuple[Any, ...], set[Any]] = defaultdict(set)
    selected_columns = [*parameter_names, "seed", "result"]
    for row in table.loc[:, selected_columns].itertuples(index=False, name=None):
        key = tuple(_python_scalar(value) for value in row[: len(parameter_names)])
        seed = _python_scalar(row[len(parameter_names)])
        if seed in seed_sets[key]:
            raise ValueError(
                f"duplicate full-fidelity HPOBench row for configuration {key} "
                f"and seed {seed}"
            )
        seed_sets[key].add(seed)
        score_lists[key].append(_validation_accuracy(row[-1]))
    if not score_lists:
        raise ValueError("HPOBench table produced no candidate configurations")

    ordered_keys = sorted(score_lists, key=lambda values: tuple(map(repr, values)))
    native_candidates = tuple(
        {
            name: _python_scalar(value)
            for name, value in zip(parameter_names, key)
        }
        for key in ordered_keys
    )
    ground_truth = np.asarray(
        [float(np.mean(score_lists[key])) for key in ordered_keys],
        dtype=float,
    )
    candidate_to_index = {
        key: index for index, key in enumerate(ordered_keys)
    }

    def objective(candidate: Mapping[str, Any]) -> float:
        missing_names = set(parameter_names) - set(candidate)
        extra_names = set(candidate) - set(parameter_names)
        if missing_names or extra_names:
            raise ValueError(
                "candidate does not match the HPOBench configuration space "
                f"(missing={sorted(missing_names)}, extra={sorted(extra_names)})"
            )
        key = tuple(_python_scalar(candidate[name]) for name in parameter_names)
        try:
            index = candidate_to_index[key]
        except KeyError as exc:
            raise ValueError(
                f"candidate {dict(candidate)!r} is not in the HPOBench table"
            ) from exc
        return float(ground_truth[index])

    seed_values = tuple(
        sorted({_python_scalar(value) for value in table["seed"].unique()})
    )
    problem_metadata = {
        "source": "HPOBench tabular ML benchmark",
        "parquet_path": str(parquet_path),
        "metadata_path": str(metadata_path),
        "full_fidelity": full_fidelity,
        "seeds": seed_values,
        "objective": "mean validation accuracy across available benchmark seeds",
        "official_function_value": "1 - validation accuracy",
    }
    return HPOBenchProblem(
        dataset_id=DATASET_ID,
        model=model,
        task_name=task,
        task_id=TASKS[task],
        parameter_names=parameter_names,
        native_candidates=native_candidates,
        ground_truth=ground_truth,
        objective=objective,
        metadata=problem_metadata,
        maximize=True,
    )


def _download_archive(
    url: str,
    destination: Path,
    *,
    opener: Callable[..., BinaryIO] = urlopen,
) -> None:
    """Download one archive atomically using only the Python standard library."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "hapbo-hpobench-downloader/1"})
    try:
        with opener(request) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _archive_member_for_task(
    archive: ZipFile,
    model: str,
    task_id: int,
    suffix: str,
) -> Any:
    """Find one official task file regardless of an optional archive prefix."""

    expected_name = f"{model}_{task_id}_{suffix}"
    matches = []
    for member in archive.infolist():
        parts = PurePosixPath(member.filename).parts
        if member.is_dir() or member.filename.startswith("/") or ".." in parts:
            continue
        if str(task_id) in parts and parts[-1] == expected_name:
            matches.append(member)
    if len(matches) != 1:
        raise ValueError(
            f"archive must contain exactly one {expected_name}; found {len(matches)}"
        )
    return matches[0]


def download_hpobench_data(
    *,
    models: Sequence[str] = MODELS,
    tasks: Sequence[str] = tuple(TASKS),
    benchmark_directory: str | Path = HPOBENCH_DIRECTORY,
    keep_archives: bool = False,
    force: bool = False,
    opener: Callable[..., BinaryIO] = urlopen,
) -> dict[str, Any]:
    """Download model bundles and extract only the task subset used by LLAMBO (Liu et al.)."""

    selected_models = tuple(dict.fromkeys(models))
    selected_tasks = tuple(dict.fromkeys(tasks))
    unknown_models = set(selected_models) - set(MODELS)
    unknown_tasks = set(selected_tasks) - set(TASKS)
    if unknown_models:
        raise ValueError(f"unknown HPOBench models {sorted(unknown_models)}")
    if unknown_tasks:
        raise ValueError(f"unknown HPOBench tasks {sorted(unknown_tasks)}")
    if not selected_models or not selected_tasks:
        raise ValueError("models and tasks must not be empty")

    root = Path(benchmark_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    extracted: list[dict[str, Any]] = []
    for model in selected_models:
        expected_paths = [
            hpobench_task_path(model, task, benchmark_directory=root)
            for task in selected_tasks
        ]
        all_present = all(path.is_file() for pair in expected_paths for path in pair)
        archive_path = root / model / f"{model}.zip"
        if not all_present or force:
            _download_archive(DOWNLOAD_URLS[model], archive_path, opener=opener)
            with ZipFile(archive_path) as archive:
                for task in selected_tasks:
                    task_id = TASKS[task]
                    parquet_path, metadata_path = hpobench_task_path(
                        model,
                        task,
                        benchmark_directory=root,
                    )
                    parquet_path.parent.mkdir(parents=True, exist_ok=True)
                    for suffix, destination in (
                        ("data.parquet.gzip", parquet_path),
                        ("metadata.json", metadata_path),
                    ):
                        if destination.is_file() and not force:
                            continue
                        member = _archive_member_for_task(
                            archive,
                            model,
                            task_id,
                            suffix,
                        )
                        temporary = destination.with_suffix(destination.suffix + ".part")
                        with archive.open(member) as source, temporary.open("wb") as output:
                            shutil.copyfileobj(source, output, length=1024 * 1024)
                        temporary.replace(destination)
        for task, paths in zip(selected_tasks, expected_paths):
            if not all(path.is_file() for path in paths):
                raise FileNotFoundError(
                    f"download did not produce the expected files for {model}/{task}"
                )
            search_space = hpobench_search_space_summary(
                model,
                task,
                benchmark_directory=root,
            )
            search_space_path = paths[1].parent / "search_space.json"
            search_space_path.write_text(
                json.dumps(search_space, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            extracted.append(
                {
                    "model": model,
                    "task": task,
                    "task_id": TASKS[task],
                    "files": {
                        path.name: {
                            "size_bytes": path.stat().st_size,
                            "sha256": _sha256_file(path),
                        }
                        for path in paths
                    },
                    "search_space": str(search_space_path),
                }
            )
        if not keep_archives:
            archive_path.unlink(missing_ok=True)

    manifest_path = root / "download_manifest.json"
    previous: Mapping[str, Any] = {}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if loaded.get("artifact_format") == "llm-bo-hpobench-download-v1":
                previous = loaded
        except (AttributeError, json.JSONDecodeError, OSError):
            previous = {}
    extracted_by_combination = {
        (item.get("model"), item.get("task")): item
        for item in previous.get("extracted", [])
        if (
            isinstance(item, Mapping)
            and item.get("model") in MODELS
            and item.get("task") in TASKS
        )
    }
    extracted_by_combination.update(
        {(item["model"], item["task"]): item for item in extracted}
    )
    cumulative_models = sorted(
        {
            *(
                model
                for model in previous.get("models", [])
                if model in MODELS
            ),
            *selected_models,
        },
        key=MODELS.index,
    )
    cumulative_tasks = sorted(
        {
            *(
                task
                for task in previous.get("tasks", [])
                if task in TASKS
            ),
            *selected_tasks,
        },
        key=tuple(TASKS).index,
    )
    manifest = {
        "artifact_format": "llm-bo-hpobench-download-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "HPOBench official Figshare tabular benchmark archives",
        "models": cumulative_models,
        "tasks": cumulative_tasks,
        "download_urls": {
            model: DOWNLOAD_URLS[model] for model in cumulative_models
        },
        "extracted": [
            extracted_by_combination[key]
            for key in sorted(
                extracted_by_combination,
                key=lambda item: (MODELS.index(item[0]), tuple(TASKS).index(item[1])),
            )
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "DATASET_ID",
    "DEFAULT_MODEL",
    "DEFAULT_TASK",
    "DOWNLOAD_URLS",
    "HPOBENCH_DIRECTORY",
    "HPOBenchProblem",
    "MODELS",
    "TASKS",
    "download_hpobench_data",
    "hpobench_task_path",
    "hpobench_search_space_summary",
    "load_hpobench_problem",
]
