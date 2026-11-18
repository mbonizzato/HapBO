"""Isolated execution adapter for one immutable study-run specification."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from bayesian_optimization import BayesianOptimization
from bayesian_optimization.acquisition import ALPHA_MAX, pi_bo_beta
from bayesian_optimization.surrogate import DEFAULT_KERNEL
from datasets import (
    SOLAR_COMMIT,
    SOLAR_EXECUTABLE,
    evaluate_solar,
    generate_solar_candidate_pool,
    load_hpobench_problem,
    load_neuromosaic_dataset1_rat_problem,
    resolve_solar_instance,
)
from prior.landscape.landscape_generator import (
    compile_prior,
    load_record,
    validate_prior_compatibility,
)

from .catalog import PROJECT_ROOT, load_catalog


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_jsonable) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _log_scaled_names(problem: Any, adapter: str) -> tuple[str, ...]:
    """Names of log-spaced grid dimensions, from the benchmark's own grid definition."""

    if adapter != "hpobench":
        return ()
    from datasets.hpobench import hpobench_search_space_summary

    summary = hpobench_search_space_summary(str(problem.model), str(problem.task_name))
    return tuple(str(d["name"]) for d in summary["dimensions"] if d.get("scale") == "log")


_CORTICAL_AXES = (
    "Search coordinates: x0 is the rostro-caudal index (1 = most rostral ... 8 = most caudal) and x1 is "
    "the medio-lateral index (1 = most medial ... 4 = most lateral); where coordinates are shown rescaled "
    "to [0, 1], 0 corresponds to index 1."
)
_CORTICAL_CHANNELS = (  # same mapping, in the native names LLAMBO Mode 2's zero-shot prompt shows
    "Configuration names: channel_x is the rostro-caudal index (1 = most rostral ... 8 = most caudal) and "
    "channel_y is the medio-lateral index (1 = most medial ... 4 = most lateral)."
)

# --- Coordinate legend for the HPOBench / CNN x-space baselines ---------------------------
#
# The x-space baselines (LGBO, BORA, in-loop LLAMBO) optimise anonymous coordinates
# x0..x{d-1} in [0, 1].  ``_coordinate_legend`` tells them which parameter each coordinate is,
# derived from the SAME task card HapBO elicits from (names + meanings from its
# "## 2. Search space" section) and from the encoded matrix the baseline actually receives
# (native range, log/linear mapping, so the legend can never disagree with the encoding).
# No dataset / benchmark identity is added beyond what the card shows.  Cortical runs carry
# their own axis text (_CORTICAL_AXES).  The legend is off by default; studies turn it on with
# ``optimizer.baseline_task_legend: true``.

_CNN_TASK_CARDS = {"fashion-of": "experiments/priors/cnn-leak/card-overfit/prompt_card.md"}


def _card_section(text: str, number: int) -> str:
    """Body of section ``## <number>. ...`` of a prompt card ('' when absent)."""

    match = re.search(rf"^## {number}\.[^\n]*\n(.*?)(?=^## \d+\.|\Z)", text, re.M | re.S)
    return match.group(1).strip() if match else ""


def _parse_card_dimensions(text: str) -> dict[str, dict[str, str]]:
    """``{lower-case name: {field: value}}`` from the card's search-space section."""

    dims: dict[str, dict[str, str]] = {}
    for block in re.split(r"^Parameter \d+\s*$", _card_section(text, 2), flags=re.M):
        fields: dict[str, str] = {}
        for line in block.strip().splitlines():
            key, sep, value = line.partition(":")
            if sep and key.strip() and " " not in key.strip():
                fields[key.strip().lower()] = value.strip()
        if "name" in fields:
            dims[fields["name"].lower()] = fields
    return dims


def _baseline_task_card(adapter: str, problem: Any, run: Mapping[str, Any]) -> Path:
    """Path of the task card HapBO elicits from for this task."""

    if adapter == "hpobench":
        slug_ = str(problem.task_name).replace("_", "-")
        return (PROJECT_ROOT / "experiments" / "priors" / f"hpobench-{problem.model}-{slug_}" / slug_
                / "llm-card" / "prompt_card.md")
    if adapter == "cnn-leak":
        dataset = str((run.get("parameters") or {}).get("dataset", "fashion-of"))
        if dataset in _CNN_TASK_CARDS:
            return PROJECT_ROOT / _CNN_TASK_CARDS[dataset]
        raise FileNotFoundError(
            f"no task card registered for cnn-leak dataset {dataset!r}; set optimizer.baseline_task_legend: false")
    raise ValueError(f"no task-card legend for adapter {adapter!r}")


def _fmt_native(value: float) -> str:
    return f"{float(value):.4g}"  # 4 significant digits: 1024, 8567, 0.0009766 print exactly enough


def _coordinate_legend(
    native: Sequence[Mapping[str, Any]],
    encoded: np.ndarray,
    names: Sequence[str],
    card_dims: Mapping[str, Mapping[str, str]],
) -> str:
    """Legend mapping each encoded coordinate to the card's parameter (see block comment)."""

    encoded = np.asarray(encoded, dtype=float)
    lines = [
        "Search coordinates: the search runs over x0..x%d, each rescaled to [0, 1]. They are the "
        "parameters of the task's search space, mapped as follows:" % (encoded.shape[1] - 1)
    ]
    column = 0
    for name in names:
        card = card_dims.get(str(name).lower())
        if card is None:
            raise ValueError(f"task card has no search-space entry for {name!r}")
        meaning = card.get("meaning") or card.get("role") or ""
        label = f"{card.get('name', name)}" + (f" ({meaning})" if meaning else "")
        values = [candidate[name] for candidate in native]
        try:
            numeric = np.asarray(values, dtype=float)
            if not np.all(np.isfinite(numeric)):
                raise ValueError
        except (TypeError, ValueError):
            categories = sorted(set(values), key=repr)
            width = len(categories)
            block = encoded[:, column:column + width]
            for k, category in enumerate(categories):
                rows = [i for i, v in enumerate(values) if v == category]
                if not np.allclose(block[rows, k], 1.0):
                    raise ValueError(f"encoded one-hot block for {name!r} does not match the grid")
            spans = ", ".join(f"x{column + k} = 1 means {c}" for k, c in enumerate(categories))
            lines.append(f"- x{column}..x{column + width - 1} one-hot encode {label}: {spans}.")
            column += width
            continue
        if column >= encoded.shape[1]:
            raise ValueError("encoded matrix has fewer columns than the search space")
        x = encoded[:, column]
        grid = sorted(set(float(v) for v in numeric))
        coord = {}
        for v in grid:
            xs = x[numeric == v]
            if np.ptp(xs) > 1e-9:
                raise ValueError(f"encoded column {column} is not a function of {name!r}")
            coord[v] = float(xs[0])
        lo, hi = grid[0], grid[-1]
        xv = np.asarray([coord[v] for v in grid])
        gv = np.asarray(grid)
        tag = f"x{column}"
        if len(grid) == 1:
            lines.append(f"- {tag} = {label}: fixed at {_fmt_native(lo)} ({tag} = 0).")
            column += 1
            continue
        linear = np.allclose(xv, (gv - lo) / (hi - lo), atol=1e-6)
        logfit = lo > 0 and np.allclose(xv, (np.log(gv) - np.log(lo)) / (np.log(hi) - np.log(lo)), atol=1e-6)
        head = (f"- {tag} = {label}: {len(grid)} grid values from {_fmt_native(lo)} ({tag} = 0) "
                f"to {_fmt_native(hi)} ({tag} = 1)")
        if logfit:
            text = (f"{head}, log scale: {tag} = (log({card.get('name', name)}) - log({_fmt_native(lo)})) / "
                    f"(log({_fmt_native(hi)}) - log({_fmt_native(lo)}))")
        elif linear:
            text = (f"{head}, linear scale: {tag} = ({card.get('name', name)} - {_fmt_native(lo)}) / "
                    f"({_fmt_native(hi)} - {_fmt_native(lo)})")
        else:
            text = f"{head}, monotone non-uniform scale"
        if len(grid) <= 8 or not (logfit or linear):
            text += "; grid value -> coordinate: " + ", ".join(
                f"{_fmt_native(v)} -> {coord[v]:.3g}" for v in grid)
        lines.append(text + ".")
        column += 1
    if column != encoded.shape[1]:
        raise ValueError(f"legend covers {column} columns, encoded matrix has {encoded.shape[1]}")
    return "\n".join(lines)


def _box_legend(
    bounds: Sequence[tuple[str, float, float]],
    card_dims: Mapping[str, Mapping[str, str]],
) -> str:
    """Legend for a continuous box encoded linearly, ``x_j = (v - lo) / (hi - lo)``.

    Used for SOLAR, whose external baselines receive the pool encoded linearly between each
    variable's bounds (``scripts/solar_externals.py``, ``encode``); names and meanings come from
    the task card, the mapping from the bounds actually used by the encoding."""

    lines = [
        "Search coordinates: the search runs over x0..x%d, each rescaled to [0, 1]. They are the "
        "parameters of the task's search space, mapped as follows:" % (len(bounds) - 1)
    ]
    for j, (name, lo, hi) in enumerate(bounds):
        card = card_dims.get(str(name).lower())
        if card is None:
            raise ValueError(f"task card has no search-space entry for {name!r}")
        meaning = card.get("meaning") or card.get("role") or ""
        label = f"{card.get('name', name)}" + (f" ({meaning})" if meaning else "")
        lines.append(f"- x{j} = {label}: range {_fmt_native(lo)} (x{j} = 0) to {_fmt_native(hi)} (x{j} = 1), "
                     f"linear in the value: x{j} = ({card.get('name', name)} - {_fmt_native(lo)}) / "
                     f"({_fmt_native(hi)} - {_fmt_native(lo)}).")
    return "\n".join(lines)


def _baseline_text_from_card(
    adapter: str, problem: Any, run: Mapping[str, Any], native: Sequence[Mapping[str, Any]],
    encoded: np.ndarray,
) -> tuple[str, str, str, Path]:
    """``(legend, domain, objective_text, card_path)`` for an HPO / CNN external baseline."""

    card_path = _baseline_task_card(adapter, problem, run)
    if not card_path.is_file():
        raise FileNotFoundError(f"task card for the baseline legend is missing: {card_path}")
    text = card_path.read_text(encoding="utf-8")
    legend = _coordinate_legend(native, encoded, problem.parameter_names, _parse_card_dimensions(text))
    objective_lines = [line for line in _card_section(text, 3).splitlines()
                       if line.lower().startswith(("objective:", "direction:"))]
    return legend, _card_section(text, 1), "\n".join(objective_lines), card_path


_HPO_MODEL_PHRASE = {
    "lr": "logistic-regression classifier",
    "rf": "random-forest classifier",
    "xgb": "gradient-boosted-tree classifier",  # same family word as the cards/hints
    "nn": "multilayer-perceptron classifier",
    "svm": "support-vector classifier with an RBF kernel",
}


def _misleading_in_encoded_space(
    text: str,
    native: Sequence[Mapping[str, Any]],
    encoded: np.ndarray,
    names: Sequence[str],
) -> str:
    """Restate "near the configuration k=v, ..." in the encoded coordinates x0..x{d-1}.

    The configuration is snapped to its grid candidate (log-nearest for positive values)
    and replaced by that candidate's encoded row, so baselines that only see x-coordinates
    receive the same (misleading) location in their own space. Text without a parsable
    configuration is returned unchanged."""
    m = re.search(r"near the configuration (.*?)\. Prefer", text)
    if m is None:
        return text
    pairs = {}
    for item in m.group(1).split(","):
        key, _, value = item.partition("=")
        pairs[key.strip().lower()] = float(value)
    lower = {str(n).lower(): n for n in names}
    if set(pairs) != set(lower):
        return text
    target = {}
    for key, value in pairs.items():
        grid = sorted({float(c[lower[key]]) for c in native})
        if value > 0 and all(g > 0 for g in grid):
            target[key] = min(grid, key=lambda g: abs(math.log(g) - math.log(value)))
        else:
            target[key] = min(grid, key=lambda g: abs(g - value))
    for index, candidate in enumerate(native):
        if all(float(candidate[lower[k]]) == v for k, v in target.items()):
            row = encoded[index]
            coords = ", ".join(f"x{j}={float(row[j]):.3g}" for j in range(len(row)))
            return text[: m.start(1)] + coords + text[m.end(1):]
    return text


def _encode_mixed_candidates(
    candidates: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    log_names: Sequence[str] = (),
) -> np.ndarray:
    """Encode numeric columns by range and categorical columns one-hot.

    Columns named in ``log_names`` (log-spaced grids, all values positive) are min-max
    scaled in log space, so a grid spanning several decades is spread over [0, 1] instead of
    being squashed against 0."""

    log_set = {str(n) for n in log_names}
    blocks: list[np.ndarray] = []
    for name in names:
        values = [candidate[name] for candidate in candidates]
        try:
            numeric = np.asarray(values, dtype=float)
            if not np.all(np.isfinite(numeric)):
                raise ValueError
        except (TypeError, ValueError):
            categories = sorted(set(values), key=repr)
            indices = np.asarray([categories.index(value) for value in values])
            blocks.append(np.eye(len(categories), dtype=float)[indices])
            continue
        if name in log_set and np.all(numeric > 0):
            numeric = np.log(numeric)
        span = float(np.ptp(numeric))
        normalized = np.zeros_like(numeric) if span <= 1e-12 else (
            numeric - float(np.min(numeric))
        ) / span
        blocks.append(normalized.reshape(-1, 1))
    return np.column_stack(blocks).astype(float, copy=False)


# --- One GP input encoding per finite task ------------------------------------------------
#
# Prior arms encode candidates with ``main.encode_candidates_for_gp`` under the prior's
# schema (unit / log / rank / one-hot).  The no-prior BO arms use the same encoding, so
# every BO arm of a task runs its GP in the same input geometry:
#
# * when the run carries the task's prior_path (set at the benchmark selection level and
#   shared by all arms of the task), the encoding is computed by the calls the prior arms
#   make (``load_record`` -> ``compile_prior`` -> NeuroMOSAIC schema adaptation ->
#   ``encode_candidates_for_gp``), so the array is bit-identical to the prior arms'
#   ``encoded``.  Only the schema enters; the prior's beliefs, pi and alpha are not used.
#   All cards of a task share one encoding schema.
# * without a prior_path, a task-level reference schema is derived from the grid
#   (NeuroMOSAIC: ordinal electrode indices in the cards' naming; HPOBench: ordinal rank of
#   each grid axis) or read from experiments/priors/solar-<instance>/schema.json.
# * a prior_path that is set but missing raises.
#
# Random search and the external baselines keep their own candidate encodings.

class _SchemaOnly:
    """Duck-typed stand-in for ``CompiledPrior`` exposing only the encoding schema."""

    def __init__(self, dimensions: Sequence[Any]) -> None:
        self.dimensions = list(dimensions)

    @property
    def parameter_names(self) -> list[str]:
        return [dimension.name for dimension in self.dimensions]


def _native_offline_encoding(problem: Any, adapter: str) -> np.ndarray:
    """Candidate encoding of random search and the external baselines."""

    native = tuple(problem.native_candidates)
    if adapter.startswith("neuromosaic-"):
        return np.asarray(problem.encoded_candidates, dtype=float)
    return _encode_mixed_candidates(native, problem.parameter_names, _log_scaled_names(problem, adapter))


def _grid_schema_dimensions(problem: Any, adapter: str) -> tuple[list[Any], tuple[dict[str, Any], ...]] | None:
    """Task-level reference schema for a finite grid, mirroring the cards' schema.

    Returns ``(dimensions, native_in_schema_names)`` or ``None`` when the adapter has no
    card schema to mirror (then the native encoding is used)."""

    from main import adapt_neuromosaic_candidates_to_prior
    from prior.landscape.landscape_generator import DimensionSpec

    if adapter.startswith("neuromosaic-"):
        names = ("electrode_mediolateral_index", "electrode_rostrocaudal_index")
        native, _ = adapt_neuromosaic_candidates_to_prior(problem, names)
        raw = [
            {"name": name, "type": "ordinal", "values": sorted({int(c[name]) for c in native})}
            for name in names
        ]
        return [DimensionSpec.from_json(r, i) for i, r in enumerate(raw)], tuple(native)
    if adapter == "hpobench":
        native = tuple(problem.native_candidates)
        raw = []
        for name in problem.parameter_names:
            values = [candidate[name] for candidate in native]
            try:
                numeric = sorted({float(v) for v in values})
                if not all(np.isfinite(numeric)):
                    raise ValueError
                raw.append({"name": str(name).lower(), "type": "ordinal", "values": numeric})
            except (TypeError, ValueError):
                raw.append({"name": str(name).lower(), "type": "categorical",
                            "values": sorted(set(values), key=repr)})
        return [DimensionSpec.from_json(r, i) for i, r in enumerate(raw)], native
    return None


def _task_gp_encoding(
    problem: Any,
    adapter: str,
    prior_path: Path | None,
) -> tuple[np.ndarray, str]:
    """GP input encoding of a finite task for the no-prior BO arms: ``(encoded, provenance)``."""

    from main import adapt_neuromosaic_candidates_to_prior, encode_candidates_for_gp

    if prior_path is not None and not Path(prior_path).is_file():
        raise FileNotFoundError(f"prior_path set but missing: {prior_path}")
    if prior_path is not None:
        raw = json.loads(Path(prior_path).read_text(encoding="utf-8"))
        if isinstance(raw, Mapping) and "schema" in raw:
            # The same calls as _offline_prior_context -> bit-identical array.
            compiler = compile_prior(load_record(prior_path))
            native = problem.native_candidates
            if adapter.startswith("neuromosaic-"):
                native, _ = adapt_neuromosaic_candidates_to_prior(problem, compiler.parameter_names)
            return encode_candidates_for_gp(native, compiler), f"prior-schema:{Path(prior_path).as_posix()}"
    derived = _grid_schema_dimensions(problem, adapter)
    if derived is None:
        return _native_offline_encoding(problem, adapter), "native"
    dimensions, native = derived
    return encode_candidates_for_gp(native, _SchemaOnly(dimensions)), "task-grid-schema"


def _solar_task_schema_path(instance: str) -> Path:
    slug_ = str(instance).strip().lower().replace("_", "-")
    return PROJECT_ROOT / "experiments" / "priors" / f"solar-{slug_}" / "schema.json"


def _solar_task_gp_encoding(
    candidates: Sequence[Mapping[str, Any]],
    spec: Any,
    instance: str,
    prior_path: Path | None,
) -> tuple[np.ndarray, str]:
    """SOLAR analogue of :func:`_task_gp_encoding` for the no-prior BO arm."""

    from main import encode_candidates_for_gp
    from prior.landscape.landscape_generator import DimensionSpec

    expected = tuple(variable.name for variable in spec.variables)
    if prior_path is not None and not Path(prior_path).is_file():
        raise FileNotFoundError(f"prior_path set but missing: {prior_path}")
    if prior_path is not None:
        raw = json.loads(Path(prior_path).read_text(encoding="utf-8"))
        if isinstance(raw, Mapping) and "schema" in raw:
            # The same calls as the alpha-pi branch of _run_solar -> bit-identical array.
            compiler = compile_prior(load_record(prior_path))
            if tuple(compiler.parameter_names) != expected:
                raise ValueError(f"SOLAR prior dimensions must be {list(expected)}")
            return encode_candidates_for_gp(candidates, compiler), f"prior-schema:{Path(prior_path).as_posix()}"
    schema_path = _solar_task_schema_path(instance)
    if schema_path.is_file():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        dimensions = [DimensionSpec.from_json(r, i) for i, r in enumerate(schema["dimensions"])]
        if tuple(d.name for d in dimensions) != expected:
            raise ValueError(f"{schema_path} dimensions must be {list(expected)}")
        return encode_candidates_for_gp(candidates, _SchemaOnly(dimensions)), f"task-schema:{schema_path.name}"
    return _encode_solar(candidates, spec), "native"


def _optimizer_settings(run: Mapping[str, Any], method: Mapping[str, Any]) -> dict[str, Any]:
    configured = run.get("optimizer", {})
    return {
        "length_scale": float(configured.get("length_scale", 1.0)),
        "kappa": float(configured.get("kappa", 2.0)),
        "acquisition": method.get("acquisition", "expected-improvement"),
        "improvement_xi": float(configured.get("improvement_xi", 0.0)),
        # Incumbent-relative, unclipped UCB weighting.
        # Off by default; UCB acquisition methods opt in via the catalog field.
        "ucb_center_incumbent": bool(method.get("ucb_center_incumbent", False)),
        "n_initial": int(configured.get("n_initial", 1)),
        "noise_variance": max(float(configured.get("noise_std", 0.05)) ** 2, 1e-12),
        "signal_variance": float(configured.get("signal_variance", 1.0)),
        "normalize_y": bool(configured.get("normalize_y", True)),
        # GP covariance of every surrogate of the run (matern52, rbf or matern12);
        # manifests without the key use the default Matern-5/2.
        "kernel": configured.get("kernel", DEFAULT_KERNEL),
        "random_state": int(run["seed"]),
    }


def _method_alpha(method: Mapping[str, Any]) -> float | None:
    """Fixed trust weight of an alpha-pi method (catalog field ``alpha``), or None to use the compiled alpha.

    The value must lie in [0, 0.9] (``ALPHA_MAX``); any other strategy rejects the field."""

    if method.get("alpha") is None:
        return None
    if str(method["strategy"]) != "alpha-pi":
        raise ValueError("the method field 'alpha' applies to the alpha-pi strategy only")
    value = float(method["alpha"])
    if not (math.isfinite(value) and 0.0 <= value <= ALPHA_MAX):
        raise ValueError(f"method alpha must lie in [0, {ALPHA_MAX}], got {method['alpha']!r}")
    return value


def _pi_bo_beta(method: Mapping[str, Any], run: Mapping[str, Any]) -> float:
    """piBO's beta of a run (acquisition.pi_bo_beta: N / 10). It is not a method field."""

    if "pi_bo_beta" in method:
        raise ValueError("pi_bo_beta is not a method field: piBO uses beta = N / 10")
    return pi_bo_beta(run["budget"], run.get("optimizer", {}).get("n_initial", 1))


def _build_optimizer(
    encoded: np.ndarray,
    run: Mapping[str, Any],
    method: Mapping[str, Any],
    *,
    prior: np.ndarray | None,
    alpha: float,
    maximize: bool,
    n_constraints: int = 0,
    extra: Mapping[str, Any] | None = None,
) -> BayesianOptimization:
    settings = _optimizer_settings(run, method)
    if method["strategy"] == "random-search":
        settings["n_initial"] = int(run["budget"])
    kwargs = dict(settings)
    if extra:
        kwargs.update(extra)
    return BayesianOptimization(
        encoded,
        prior=prior,
        alpha=alpha,
        maximize=maximize,
        n_constraints=n_constraints,
        **kwargs,
    )


def _load_offline_problem(
    adapter: str, parameters: Mapping[str, Any], *, noise_seed: int | None = None
) -> Any:
    if adapter == "neuromosaic-dataset1-rat":
        return load_neuromosaic_dataset1_rat_problem(
            subject=str(parameters["subject"]), muscle=str(parameters["muscle"]),
            noise_seed=noise_seed,
        )
    if adapter == "hpobench":
        return load_hpobench_problem(
            model=str(parameters["model"]), task=str(parameters["task"])
        )
    if adapter == "cnn-leak":
        from datasets.cnn_leak import load_cnn_leak_problem

        return load_cnn_leak_problem(
            dataset=str(parameters.get("dataset", "fashion-of")),
        )
    raise ValueError(f"unsupported offline study adapter {adapter!r}")


def _standardize_pi(pi_values: np.ndarray) -> np.ndarray:
    """Candidate-set z-score of the prior; all-zeros for a flat prior."""

    pi = np.asarray(pi_values, dtype=float).reshape(-1)
    std = float(pi.std())
    if std <= 1e-12:
        return np.zeros_like(pi)
    return (pi - float(pi.mean())) / std


def _prior_mean_from_pi(encoded: np.ndarray, pi_values: np.ndarray, beta: float):
    """Callable ``m_z(X) = beta * standardize(pi)`` for the prior-to-mean arm.

    ``m_z`` lives in standardized target space; values are looked up by encoded
    candidate row so the GP can query it at train and test points alike.
    """

    m_z = float(beta) * _standardize_pi(pi_values)
    rows = np.asarray(encoded, dtype=float)
    table = {rows[i].tobytes(): float(m_z[i]) for i in range(rows.shape[0])}

    def mean(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.empty(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            key = X[i].tobytes()
            if key not in table:
                raise KeyError("prior-to-mean received an unknown candidate row")
            out[i] = table[key]
        return out

    return mean


def _top_pi_indices(pi_values: np.ndarray, count: int) -> list[int]:
    """The ``count`` distinct highest-prior candidate indices (prior-based initial design)."""

    pi = np.asarray(pi_values, dtype=float).reshape(-1)
    order = np.argsort(-pi, kind="stable")
    return [int(i) for i in order[: max(1, int(count))]]


def _offline_prior_context(
    problem: Any,
    adapter: str,
    prior_path: Path,
) -> tuple[tuple[dict[str, Any], ...], np.ndarray, np.ndarray, float]:
    """Compile the prior over the problem grid: native candidates, GP encoding, pi and alpha."""

    from main import adapt_neuromosaic_candidates_to_prior, encode_candidates_for_gp

    record = load_record(prior_path)
    if adapter.startswith("neuromosaic-"):
        validate_prior_compatibility(
            prior_path,
            dataset_id=str(problem.dataset_id),
            target_name=str(problem.target_name),
        )
    compiler = compile_prior(record)
    native = problem.native_candidates
    if adapter.startswith("neuromosaic-"):
        native, _ = adapt_neuromosaic_candidates_to_prior(
            problem, compiler.parameter_names
        )
    encoded = encode_candidates_for_gp(native, compiler)
    pi_values = np.asarray(compiler.pi(native), dtype=float).reshape(-1)
    alpha = float(compiler.alpha)
    return tuple(native), encoded, pi_values, alpha


def _llm_point_gaussian_prior(
    problem: Any, adapter: str, prior_path: Path
) -> dict[str, Any]:
    """Build the ``colabo-llm`` prior from a one-shot LLM (c, s) artifact.

    BORA supplement C.2 construction: ``pi(x) = N(c, sigma)`` with
    ``sigma = 1 - s + 1e-6``. The returned ``values`` are candidate-aligned so the
    ColaBO worker takes ``argmax(values)`` = the candidate nearest ``c``
    (isotropic Gaussian in the per-dimension normalized space, mirroring
    ``DimensionSpec.encode_column`` conventions), and ``confidence`` = ``sigma``
    lands directly on upstream ``DefaultPrior``'s float confidence, which IS the
    per-dimension Gaussian sigma in its normalized [0,1] space.
    """

    import math

    artifact = json.loads(Path(prior_path).read_text(encoding="utf-8"))
    if artifact.get("artifact_format") != "colabo-llm-point-prior-v1":
        raise ValueError(
            f"{prior_path}: expected a colabo-llm-point-prior-v1 artifact, "
            f"got {artifact.get('artifact_format')!r}"
        )
    dims = list(artifact["dimensions"])
    c = dict(artifact["c"])
    sigma = float(artifact["sigma"])
    if not (sigma > 0.0 and np.isfinite(sigma)):
        raise ValueError(f"{prior_path}: sigma must be positive and finite, got {sigma}")
    if problem.native_candidates:
        native_case = {str(k).lower(): str(k) for k in problem.native_candidates[0]}
        dims = [dict(d, name=native_case.get(str(d["name"]).lower(), str(d["name"]))) for d in dims]
        c = {native_case.get(str(k).lower(), str(k)): v for k, v in c.items()}
    names = tuple(str(d["name"]) for d in dims)
    missing = [n for n in names if n not in c]
    if missing:
        raise ValueError(f"{prior_path}: artifact 'c' is missing dimensions {missing}")

    native = problem.native_candidates
    if adapter.startswith("neuromosaic-"):
        from main import adapt_neuromosaic_candidates_to_prior

        native, _ = adapt_neuromosaic_candidates_to_prior(problem, names)

    def _normalize(dim: Mapping[str, Any], value: Any) -> float:
        values = dim.get("values")
        if values is not None:
            span = max(len(values) - 1, 1)
            if any(isinstance(v, str) for v in values):
                if value not in values:
                    raise ValueError(
                        f"{prior_path}: value {value!r} not allowed for {dim['name']!r}"
                    )
                return float(values.index(value)) / span
            numeric = [float(v) for v in values]
            x = float(value)
            if (
                str(dim.get("scale", "linear")) == "log"
                and x > 0.0
                and all(v > 0.0 for v in numeric)
            ):
                index = min(
                    range(len(numeric)),
                    key=lambda i: abs(math.log(numeric[i]) - math.log(x)),
                )
            else:
                index = min(range(len(numeric)), key=lambda i: abs(numeric[i] - x))
            return float(index) / span
        low, high = (float(dim["range"][0]), float(dim["range"][1]))
        x = min(max(float(value), low), high)
        if str(dim.get("scale", "linear")) == "log" and low > 0.0:
            return (math.log(x) - math.log(low)) / (math.log(high) - math.log(low))
        return (x - low) / (high - low)

    center = np.array([_normalize(d, c[d["name"]]) for d in dims], dtype=float)
    grid = np.array(
        [[_normalize(d, candidate[d["name"]]) for d in dims] for candidate in native],
        dtype=float,
    )
    squared_distance = np.sum((grid - center[None, :]) ** 2, axis=1)
    values = np.exp(-0.5 * squared_distance / (sigma**2))
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{prior_path}: non-finite prior values")
    return {"values": values, "confidence": sigma}


def _llm_hypothesis_boxes(
    problem: Any, prior_path: Path, native: Sequence[Mapping[str, Any]], encoded: np.ndarray
) -> dict[str, Any]:
    """Build the ``hypbo-llm`` box hypotheses from a one-shot LLM artifact.

    BORA supplement C.3 construction: each hypothesis ``(c, s)`` becomes a box centred at the
    encoded coordinates of ``c`` with side ``(ub - lb) * (1 - s + 1e-6)``, clipped to ``[lb, ub]``,
    where ``lb`` and ``ub`` bound the encoded candidates the HypBO worker optimizes over. The
    encoded boxes recorded in the artifact at elicitation time must agree with these. A later box
    is dropped only when it and a kept box with the same centre are both narrower than 1e-4
    (``select_hypotheses``); dropped boxes are listed.
    """

    from baselines.workers.box_hypotheses import hypothesis_box, select_hypotheses

    artifact = json.loads(Path(prior_path).read_text(encoding="utf-8"))
    if artifact.get("artifact_format") != "hypbo-llm-hypotheses-v1":
        raise ValueError(
            f"{prior_path}: expected a hypbo-llm-hypotheses-v1 artifact, "
            f"got {artifact.get('artifact_format')!r}"
        )
    names = [str(name) for name in problem.parameter_names]
    encoded = np.asarray(encoded, dtype=float)
    if encoded.shape != (len(native), len(names)):
        raise ValueError(f"{prior_path}: HypBO boxes need one numeric encoded column per parameter")
    lower, upper = encoded.min(axis=0), encoded.max(axis=0)
    # native grid value -> encoded value, per parameter
    columns = []
    for j, name in enumerate(names):
        mapping: dict[float, float] = {}
        for row, candidate in enumerate(native):
            mapping.setdefault(float(candidate[name]), float(encoded[row, j]))
        columns.append(mapping)
    folded = [name.lower() for name in names]
    boxes = []
    for k, hypothesis in enumerate(artifact.get("hypotheses") or []):
        c = {str(key).lower(): value for key, value in dict(hypothesis["c"]).items()}
        if sorted(c) != sorted(folded):
            raise ValueError(f"{prior_path}: hypothesis {k + 1} names {sorted(c)}, expected {sorted(folded)}")
        center = np.empty(len(names), dtype=float)
        for j, key in enumerate(folded):
            value = float(c[key])
            grid = np.asarray(sorted(columns[j]), dtype=float)
            nearest = float(grid[int(np.argmin(np.abs(grid - value)))])
            if not np.isclose(nearest, value, rtol=1e-6, atol=0.0):
                raise ValueError(f"{prior_path}: hypothesis {k + 1}: {names[j]}={value!r} is not a grid value")
            center[j] = columns[j][nearest]
        score = float(hypothesis["s"])
        lo, hi = hypothesis_box(center, score, lower, upper)
        recorded = hypothesis.get("box_encoded")
        if recorded is not None and not (
            np.allclose(recorded["lb"], lo, rtol=0.0, atol=1e-6)
            and np.allclose(recorded["ub"], hi, rtol=0.0, atol=1e-6)
        ):
            raise ValueError(f"{prior_path}: hypothesis {k + 1}: the recorded encoded box disagrees "
                             "with the task encoding")
        boxes.append({"name": f"H{k + 1}", "center": center.tolist(), "score": score,
                      "lb": lo.tolist(), "ub": hi.tolist()})
    if not boxes:
        raise ValueError(f"{prior_path}: no hypotheses")
    kept, dropped = select_hypotheses([box["center"] for box in boxes], [box["lb"] for box in boxes],
                                      [box["ub"] for box in boxes])
    return {
        "hypotheses": [boxes[k] for k in kept],
        "dropped": [{"name": boxes[k]["name"], "score": boxes[k]["score"],
                     "same_center_as": boxes[first]["name"]} for k, first in sorted(dropped.items())],
        "source": Path(prior_path).as_posix(),
    }


def _external_objective_meta(
    run: Mapping[str, Any],
    problem: Any,
    adapter: str,
    native: Sequence[Mapping[str, Any]],
    encoded: np.ndarray,
) -> dict[str, Any]:
    """Every LLM-visible text field an external baseline receives."""

    optimizer = run.get("optimizer") or {}
    objective_meta = {
        "maximize": bool(problem.maximize),
        "n_constraints": 0,
        "model": str(getattr(problem, "model", getattr(problem, "name", "black-box system"))),
        "task": str(getattr(problem, "task_name", adapter)),
        "metric": str(getattr(problem, "target_name", "objective value")),
        "description": "finite candidate optimization task",
    }
    if adapter.startswith("neuromosaic-dataset1-rat"):
        # The cards' Domain text (ICMS of rat M1, array geometry, stereotaxic placement); the
        # metric is the response amplitude, never the recorded muscle's name.
        _dom = (PROJECT_ROOT / "experiments" / "elicitation" / "cortical-flexor" / "domain.txt")
        objective_meta["model"] = "rat motor-cortex microstimulation array (32 electrodes)"
        objective_meta["task"] = "regression"
        objective_meta["metric"] = "evoked motor response amplitude"
        _ct = str((run.get("optimizer") or {}).get("cortical_text", "human"))
        if _ct != "human":
            raise ValueError(f"optimizer.cortical_text must be 'human', got {_ct!r}")
        # The shared text (Domain + goal + expert map), i.e. the same warm-start context LGBO and
        # ColaBO receive, for the in-loop baselines.
        _ws = (PROJECT_ROOT / "experiments" / "priors" / "neuromosaic-dataset1-rat" / "left-knee-flexor"
               / "human-expert-map-shared" / "warmstart_context.md").read_text(encoding="utf-8")
        _goal = _ws.split("## 3. Optimization description")[1].strip()
        objective_meta["description"] = _dom.read_text(encoding="utf-8").strip() + "\n\n" + _goal
        # x-space baselines see anonymous coordinates; say which axis each one is (all cortical runs)
        objective_meta["description"] += "\n\n" + _CORTICAL_AXES
        objective_meta["task_description"] = objective_meta["description"]  # LLAMBO: model slot
        objective_meta["prompt_setting"] = "partial_context"  # no tabular-dataset sentence
    if adapter == "hpobench":
        descriptors = json.loads(
            (PROJECT_ROOT / "experiments" / "hpo_task_descriptors.json").read_text(encoding="utf-8")
        )["tasks"]
        objective_meta["task"] = "classification"
        objective_meta["model"] = _HPO_MODEL_PHRASE.get(str(problem.model), "classifier")
        objective_meta["metric"] = "validation accuracy"
        task_facts = descriptors[str(problem.task_name).replace("_", "-")]
        objective_meta["dataset_context"] = dict(task_facts["llambo"])
        objective_meta["description"] = (
            f"Hyperparameter tuning of a {objective_meta['model']} on {task_facts['descriptor']} "
            f"({task_facts['size']}, {task_facts['features']}, {task_facts['classes']}); maximize validation accuracy."
        )
    # Coordinate legend from the task card (HPOBench, CNN leak control); see _coordinate_legend.
    legend = None
    if bool(optimizer.get("baseline_task_legend", False)) and adapter in ("hpobench", "cnn-leak"):
        legend, card_domain, card_objective, card_path = _baseline_text_from_card(
            adapter, problem, run, native, encoded)
        objective_meta["task_card"] = card_path.relative_to(PROJECT_ROOT).as_posix() if \
            card_path.is_relative_to(PROJECT_ROOT) else str(card_path)
        if adapter == "cnn-leak":
            # The CNN baselines get the card's Domain and objective lines (the same facts
            # HapBO's elicitor reads), as on the cortical maps.
            objective_meta["model"] = "small convolutional image classifier"
            objective_meta["task"] = "classification"
            objective_meta["metric"] = "validation accuracy"
            objective_meta["description"] = card_domain + ("\n\n" + card_objective if card_objective else "")
            objective_meta["prompt_setting"] = "partial_context"  # no tabular-dataset sentence (LLAMBO)
            # LLAMBO's classification template has an n_classes slot (default 2): take the class
            # count from the card's Domain text when it states one ("roughly ten balanced classes").
            classes = re.search(r"((?:roughly|about|approximately)\s+)?(\w+)\s+(?:balanced\s+)?classes\b",
                                card_domain)
            if classes:
                objective_meta["dataset_context"] = {"n_classes": (classes.group(1) or "") + classes.group(2)}
    # Misleading domain description for the baselines that read task text. Active only in
    # studies that set optimizer.misleading_description: true.
    hint_path = (
        PROJECT_ROOT / "experiments" / "misleading_descriptions"
        / str(run.get("benchmark_id", "")) / (str(run.get("problem_id", "")) + ".txt")
    )
    # Explicit per-study switch (optimizer.misleading_description: true), never file presence
    # alone: a standard study can never pick up a hint, and an adversarial study whose hint
    # file is missing fails loudly instead of silently running un-misled.
    misleading = bool(optimizer.get("misleading_description", False))
    if misleading and not hint_path.is_file():
        raise FileNotFoundError(f"misleading_description requested but {hint_path} is missing")
    if misleading:
        hint = hint_path.read_text(encoding="utf-8").strip()
        hint_encoded = _misleading_in_encoded_space(hint, native, encoded, problem.parameter_names)
        objective_meta["description"] = (
            f"{objective_meta['description']} {hint_encoded}"
            if objective_meta.get("description") and adapter == "hpobench" else hint_encoded
        )
        objective_meta["target_description"] = hint_encoded
        objective_meta["misleading_native"] = hint
    if legend is not None:
        # Appended after the (optional) hint, so the task sentence + hint wording is unchanged.
        objective_meta["description"] = f"{objective_meta['description']}\n\n{legend}"
        objective_meta["coordinate_legend"] = legend
        # LLAMBO carries free text only in its model slot: HPO keeps its dataset-facts template
        # and adds the legend; CNN gets the full text.
        objective_meta["task_description"] = (
            legend if adapter == "hpobench" else objective_meta["description"])
    return objective_meta


def _run_external_baseline(
    directory: Path,
    run: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    method: Mapping[str, Any],
    problem: Any,
    budget: int,
) -> dict[str, Any]:
    """Run a faithful external baseline (subprocess bridge) over the finite grid."""

    from main import build_method_run, save_offline_performance_plot, save_performance_plot
    from bayesian_optimization.bo import OptimizationResult
    from baselines import get_baseline
    from baselines.base import config_key
    from baselines.runner import run_ask_tell_loop

    adapter = str(benchmark["adapter"])
    native = tuple(problem.native_candidates)
    encoded = (
        np.asarray(problem.encoded_candidates, dtype=float)
        if getattr(problem, "encoded_candidates", None) is not None
        else _encode_mixed_candidates(native, problem.parameter_names, _log_scaled_names(problem, adapter))
    )
    key_to_index = {config_key(candidate): index for index, candidate in enumerate(native)}
    search_space = {"candidates": [dict(c) for c in native], "encoded_candidates": encoded}
    objective_meta = _external_objective_meta(run, problem, adapter, native, encoded)
    legend = objective_meta.get("coordinate_legend")
    n_init = min(int(run.get("optimizer", {}).get("n_initial", 1)), budget)
    prior = None
    if bool(method.get("requires_prior", False)):
        prior_path = None if run.get("prior_path") is None else Path(run["prior_path"])
        if prior_path is None:
            raise ValueError(f"{method['baseline']} requires a prior_path")
        prior_source = method.get("prior_source")
        if prior_source == "llm-point-gaussian":
            # colabo-llm: one-shot LLM (c, s) -> N(c, sigma).
            prior = _llm_point_gaussian_prior(problem, adapter, prior_path)
        elif prior_source == "llm-hypotheses":
            # hypbo-llm: one-shot LLM hypotheses (c, s) -> encoded boxes (BORA supplement C.3).
            prior = _llm_hypothesis_boxes(problem, prior_path, native, encoded)
            _write_json(directory / "hypbo_hypotheses.json", prior)
        elif prior_source == "warmstart-card":
            # llambo-ws (LLAMBO Mode 2, zero-shot warmstarting): no numeric prior;
            # the card text becomes the warmstart context. A sibling
            # warmstart_context.md overrides the raw card (e.g. cortical expert-map
            # where the information lives in the anchor table, not the card text).
            # The adversarial hint, when the misleading-description injection fired
            # above, is appended so Mode 2 reads the same hint as the in-loop path.
            card_dir = Path(prior_path).parent
            override = card_dir / "warmstart_context.md"
            card = override if override.is_file() else (card_dir / "prompt_card.md")
            context = card.read_text(encoding="utf-8").strip()
            if "## 1. Domain" not in context:
                raise ValueError(f"warm-start context {card} has no '## 1. Domain' task text")
            context = context[context.index("## 1. Domain"):]  # the PSL-writing instructions are not LLAMBO's task
            if adapter.startswith("neuromosaic-dataset1-rat"):  # warm-start text replaces the description
                # LLAMBO Mode 2's zero-shot prompt lists the native names (channel_x, channel_y), not x0/x1
                context += "\n\n" + (_CORTICAL_CHANNELS if str(method.get("baseline")) == "llambo" else _CORTICAL_AXES)
            elif legend is not None and str(method.get("baseline")) != "llambo":
                # x-space warm-start users (LGBO-prior) read the card with native names but
                # answer in x0..x{d-1}; LLAMBO Mode 2's zero-shot prompt uses native names.
                context += "\n\n" + legend
            if objective_meta.get("misleading_native"):  # warm-start shows native names
                context += "\n" + str(objective_meta["misleading_native"])
            objective_meta["warmstart_context"] = context
            prior = None
        elif prior_source == "caliper-multirole":
            # CALIPER consumes a per-role artifact (mu, c over candidates) and
            # aligns it to the finite grid itself; pass the raw cached JSON through.
            if Path(prior_path).suffix == ".gz":
                import gzip

                with gzip.open(prior_path, "rt", encoding="utf-8") as handle:
                    prior = json.load(handle)
            else:
                prior = json.loads(Path(prior_path).read_text(encoding="utf-8"))
        else:
            raise ValueError(
                f"{method['baseline']} requires a prior but its prior_source {prior_source!r} is not one of "
                "'llm-point-gaussian', 'llm-hypotheses', 'warmstart-card', 'caliper-multirole'"
            )
    # LGBO / official-LGBO write per-call token+latency telemetry to this path;
    # the bridge copies os.environ into the isolated worker, so setting it here (in the
    # per-run process) routes each run's telemetry to its own result directory.
    if str(method.get("baseline")) == "lgbo":
        os.environ["LGBO_TELEMETRY"] = str(directory / "llm_telemetry.json")
    if str(method.get("baseline")) == "official_lgbo":
        os.environ["OFFICIAL_LGBO_TELEMETRY"] = str(directory / "llm_telemetry.json")
    if str(method.get("baseline")) == "cake":
        # CAKE builds its own LLMClient in-process; route its per-call
        # cache and token/latency telemetry into this run's directory.
        os.environ["CAKE_RUN_DIR"] = str(directory)

    optimizer = get_baseline(
        str(method["baseline"]),
        search_space=search_space,
        objective_meta=objective_meta,
        prior=prior,
        seed=int(run["seed"]),
        budget=budget,
        **dict(run.get("optimizer", {})),
    )
    # Audit trail of every LLM-visible text field this baseline received.
    _write_json(directory / "baseline_task_text.json", {
        key: objective_meta[key] for key in (
            "model", "task", "metric", "description", "task_description", "target_description",
            "warmstart_context", "dataset_context", "prompt_setting", "coordinate_legend", "task_card")
        if objective_meta.get(key) not in (None, "", {})
    })

    # Subprocess bridges: every proposal is a charged query (repeats flagged), no random fill;
    # a run whose worker stops before the budget fails. Worker stderr goes to worker_stderr.log.
    bridged = callable(getattr(optimizer, "bridge_accounting", None))
    if bridged:
        optimizer.worker_log_path = str(directory / "worker_stderr.log")

    records: list[dict[str, Any]] = []

    def objective(config: Mapping[str, Any]):
        return float(problem.objective(dict(config)))

    def _record_for(count: int, index: int, value: float) -> dict[str, Any]:
        return {
            "query": count,
            "candidate_index": index,
            "candidate": dict(native[index]),
            "objective": value,
            "ground_truth_objective": float(problem.ground_truth[index]),
        }

    def on_evaluation(count: int, config: Mapping[str, Any], observation: Any) -> None:
        index = key_to_index[config_key(config)]
        value = float(observation["objective"] if isinstance(observation, Mapping) else observation)
        record = _record_for(count, index, value)
        if bridged:
            record["repeat"] = bool(optimizer.trace[count - 1]["repeat"])
        records.append(record)

    try:
        history = run_ask_tell_loop(
            optimizer, objective, budget=budget, n_init=n_init, on_evaluation=on_evaluation
        )
    finally:
        close = getattr(optimizer, "close", None)
        if callable(close):
            close()
    bridge_accounting = optimizer.bridge_accounting() if bridged else None

    evaluated_indices = [key_to_index[config_key(c)] for c, _ in history]
    values = [float(o["objective"] if isinstance(o, Mapping) else o) for _, o in history]
    if not evaluated_indices:
        raise RuntimeError(f"{method['baseline']} worker stopped before proposing any query")
    if bridged and len(evaluated_indices) < budget:  # the worker stopped early: the run fails
        from baselines.bridge import require_full_budget

        _write_json(directory / "bridge_accounting.json", bridge_accounting)
        require_full_budget(len(evaluated_indices), budget, str(method["baseline"]))
    evaluated_indices = np.asarray(evaluated_indices, dtype=int)
    values = np.asarray(values, dtype=float)
    best_position = int(np.argmax(values) if problem.maximize else np.argmin(values))
    result = OptimizationResult(
        best_index=int(evaluated_indices[best_position]),
        best_x=encoded[int(evaluated_indices[best_position])].copy(),
        best_y=float(values[best_position]),
        evaluated_indices=evaluated_indices.copy(),
        X=encoded[evaluated_indices].copy(),
        y=values.copy(),
        constraint_values=np.empty((len(history), 0), dtype=float),
    )
    method_run = build_method_run(
        str(run["method_id"]), str(method["label"]), result, problem.ground_truth
    )
    metrics = method_run.metrics
    metric_record = {
        "label": method["label"],
        "best_observed_value": float(result.best_y),
        "best_index": int(result.best_index),
        "best_configuration": dict(problem.native_candidates[result.best_index]),
        "evaluated_indices": result.evaluated_indices.astype(int).tolist(),
        "final_simple_regret": float(metrics.simple_regret[-1]),
        "cumulative_regret": float(metrics.cumulative_regret[-1]),
        "best_normalized_performance": float(metrics.normalized_best_performance[-1]),
        "first_optimum_query": metrics.first_optimum_query,
        "first_95_percent_query": metrics.first_95_percent_query,
    }
    if bridge_accounting is not None:
        metric_record["bridge_accounting"] = bridge_accounting
    _write_json(
        directory / "metrics.json",
        {"artifact_format": "llm-bo-study-metrics-v1", "methods": {run["method_id"]: metric_record}},
    )
    _write_json(directory / "best_candidate.json", metric_record["best_configuration"])
    with (directory / "evaluations.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=_jsonable) + "\n")
    if bridge_accounting is not None:
        _write_json(directory / "bridge_accounting.json", bridge_accounting)
    if adapter.startswith("neuromosaic-"):
        save_offline_performance_plot(problem, [method_run], directory / "performance.png")
    else:
        save_performance_plot(
            [method_run],
            directory / "performance.png",
            title=f"{getattr(problem, 'task_name', adapter)} / {getattr(problem, 'model', 'external')}",
            response_label="Objective",
            optimum_value=getattr(problem, "optimum_value", None),
        )
    llm_telemetry = None
    telemetry_path = directory / "llm_telemetry.json"
    if telemetry_path.is_file():
        try:
            llm_telemetry = json.loads(telemetry_path.read_text(encoding="utf-8")).get("summary")
        except (ValueError, OSError):
            llm_telemetry = None
    outcome = {
        "alpha_requested": 0.0,
        "alpha_effective": 0.0,
        "strategy": "external",
        "baseline": str(method["baseline"]),
        "metrics": metric_record,
        "llm_telemetry": llm_telemetry,
    }
    if bridge_accounting is not None:  # provenance -> run_config.json
        outcome["bridge_accounting"] = bridge_accounting
    return outcome


def _run_offline(
    directory: Path,
    run: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    method: Mapping[str, Any],
) -> dict[str, Any]:
    from main import build_method_run, save_offline_performance_plot, save_performance_plot

    adapter = str(benchmark["adapter"])
    # Noisy polling (OptimizeNeurostim emulator): when the benchmark is flagged
    # noisy, each query samples one valid trial via per-channel common-random-numbers
    # seeded from the run seed. Regret stays defined against the true means
    # (best-queried-by-true-mean), so this only adds measurement noise the optimizer
    # must work through. The noise seed is namespaced distinctly from the optimizer
    # seed (init/tie-break) so the two random streams are unambiguously separate,
    # while remaining paired across arms within a (task, seed) cell.
    noise_seed = (int(run["seed"]) ^ 0x9E3779B9) if bool(benchmark.get("noisy", False)) else None
    problem = _load_offline_problem(adapter, run["parameters"], noise_seed=noise_seed)
    # compute_performance_metrics (main.py) and the best-index selection below assume a
    # maximization objective (np.max/argmax over ground_truth). Every offline adapter is
    # maximize=True (neuromosaic / HPOBench / cnn-leak). Guard
    # so a future minimize dataset fails loudly instead of silently inverting the regret.
    if not bool(problem.maximize):
        raise ValueError(
            f"_run_offline assumes a maximization objective but adapter {adapter!r} is minimize"
        )
    budget = int(run["budget"])
    if not 1 <= budget <= problem.n_candidates:
        raise ValueError(
            f"budget {budget} exceeds the {problem.n_candidates} available candidates"
        )
    fixed_alpha = _method_alpha(method)  # rejects the field for every strategy but alpha-pi

    if str(method["strategy"]) == "external":
        return _run_external_baseline(directory, run, benchmark, method, problem, budget)
    if str(method["strategy"]) == "external-mf":
        # PriorBand / PriMO (official NePS, isolated env): native multi-fidelity on the HPOBench
        # tables, every evaluation charged z/z_max, run stops at cumulative cost `budget`.
        from baselines.neps_mf import run_neps_mf_study_run

        return run_neps_mf_study_run(directory, run, benchmark, method, problem, budget)

    strategy = str(method["strategy"])
    requires_prior = bool(method.get("requires_prior", False))
    prior_path = None if run.get("prior_path") is None else Path(run["prior_path"])
    compiled_alpha = 0.0
    pi_values: np.ndarray | None = None
    gp_encoding: str | None = None
    llm_ei = None
    if strategy == "llm-ei":
        # LLM-EI = the vanilla-bo arm (same encoding, GP, EI, n_initial) whose candidates are
        # gated by the LLM utility; prior_path is the utility artifact (src/experiments/llm_ei.py).
        from experiments.llm_ei import load_llm_ei_context

        llm_ei = load_llm_ei_context(problem, adapter, method, prior_path)
        native = tuple(problem.native_candidates)
        encoded, gp_encoding = _task_gp_encoding(problem, adapter, llm_ei.encoding_prior_path)
    elif requires_prior:
        if prior_path is None:
            raise ValueError(f"{strategy} requires a prior_path")
        native, encoded, pi_values, compiled_alpha = _offline_prior_context(problem, adapter, prior_path)
    elif strategy == "vanilla-bo":
        # No-prior BO arms share the prior arms' (task-level) GP input encoding.
        native = tuple(problem.native_candidates)
        encoded, gp_encoding = _task_gp_encoding(problem, adapter, prior_path)
    else:
        native = tuple(problem.native_candidates)
        encoded = _native_offline_encoding(problem, adapter)

    # Injection-point dispatch: every prior arm consumes the SAME compiled prior;
    # only how the prior enters the optimizer differs.
    prior: np.ndarray | None = None
    acquisition_alpha = 0.0
    extra: dict[str, Any] = {}
    if strategy == "alpha-pi":
        # The method's fixed alpha, when set, replaces the compiled one (fixed-alpha sweep).
        prior, acquisition_alpha = pi_values, compiled_alpha if fixed_alpha is None else fixed_alpha
        extra["prior_shaping"] = "alpha-pi"
        # The arm shares the seeded random initial design with vanilla BO and prior-to-mean.
    elif strategy == "pi-bo":
        # piBO (Hvarfner et al. 2022): BayesianOptimization starts at the prior mode, samples
        # the other initial points from the prior and decays the prior weighting.
        prior = pi_values
        extra["prior_shaping"] = "pi-bo"
        extra["pi_bo_beta"] = _pi_bo_beta(method, run)
    elif strategy == "prior-to-mean":
        extra["prior_shaping"] = "none"
        extra["prior_mean"] = _prior_mean_from_pi(encoded, pi_values, beta=compiled_alpha)
    elif strategy == "prior-warm-start":
        extra["prior_shaping"] = "none"
        n_init = int(run.get("optimizer", {}).get("n_initial", 1))
        extra["initial_indices"] = _top_pi_indices(pi_values, n_init)
    elif strategy == "prior-greedy":
        # Prior-only control: no GP, no acquisition. Query candidates
        # in descending compiled-pi order. A `budget`-length initial design
        # (`initial_indices`) answers every `ask()` (BayesianOptimization.ask), so the
        # surrogate is never fit and the run follows the prior's ranking.
        # Deterministic given pi (stable argsort tie-breaks by candidate index) and
        # paired across arms (same compiled prior); no optimizer randomness needed.
        extra["prior_shaping"] = "none"
        extra["initial_indices"] = _top_pi_indices(pi_values, budget)
    elif strategy == "llm-ei":
        extra["candidate_gate"] = llm_ei.gate
    elif strategy in ("vanilla-bo", "random-search"):
        pass
    else:
        raise ValueError(f"unsupported offline strategy {strategy!r}")

    optimizer = _build_optimizer(
        encoded,
        run,
        method,
        prior=prior,
        alpha=acquisition_alpha,
        maximize=problem.maximize,
        extra=extra,
    )
    # Batch selection: q_batch>1 chooses q candidates per round via
    # Kriging Believer against a frozen surrogate, then evaluates them for real -- the
    # parallel-stimulation setting. Default 1 keeps the sequential ask/tell loop below
    # unchanged.
    q_batch = int(run.get("optimizer", {}).get("q_batch", 1))
    if q_batch < 1:
        raise ValueError(f"q_batch must be positive, got {q_batch}")
    records = []

    def _evaluate_and_tell(index: int) -> None:
        candidate = native[index]
        if adapter.startswith("neuromosaic-") and requires_prior:
            dataset_candidate = problem.native_candidates[index]
        else:
            dataset_candidate = candidate
        observed = float(problem.objective(dataset_candidate))
        optimizer.tell(index, observed)
        records.append(
            {
                "query": optimizer.n_observations,
                "candidate_index": index,
                "candidate": dict(problem.native_candidates[index]),
                "objective": observed,
                "ground_truth_objective": float(problem.ground_truth[index]),
            }
        )

    while optimizer.n_observations < budget:
        if q_batch == 1:
            index, _ = optimizer.ask()
            _evaluate_and_tell(index)
        else:
            # Choose up to q_batch candidates against the frozen surrogate, then
            # evaluate them for real. The trajectory is still recorded query-by-query
            # (best-so-far vs #evaluations), so it stays directly comparable to the
            # sequential (q_batch=1) arms of the same study.
            take = min(q_batch, budget - optimizer.n_observations)
            for index in optimizer.ask_batch(take):
                _evaluate_and_tell(index)
    result = optimizer.result()
    method_run = build_method_run(
        str(run["method_id"]), str(method["label"]), result, problem.ground_truth
    )
    metrics = method_run.metrics
    effective_alpha = float(optimizer.acquisition.alpha)
    metric_record = {
        "label": method["label"],
        "best_observed_value": float(result.best_y),
        "best_validation_accuracy": float(result.best_y) if adapter == "hpobench" else None,
        "best_index": int(result.best_index),
        "best_configuration": dict(problem.native_candidates[result.best_index]),
        "evaluated_indices": result.evaluated_indices.astype(int).tolist(),
        "final_simple_regret": float(metrics.simple_regret[-1]),
        "cumulative_regret": float(metrics.cumulative_regret[-1]),
        "best_normalized_performance": float(metrics.normalized_best_performance[-1]),
        "first_optimum_query": metrics.first_optimum_query,
        "first_95_percent_query": metrics.first_95_percent_query,
    }
    _write_json(
        directory / "metrics.json",
        {"artifact_format": "llm-bo-study-metrics-v1", "methods": {run["method_id"]: metric_record}},
    )
    _write_json(directory / "best_candidate.json", metric_record["best_configuration"])
    with (directory / "evaluations.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=_jsonable) + "\n")
    if adapter.startswith("neuromosaic-"):
        save_offline_performance_plot(problem, [method_run], directory / "performance.png")
    else:
        if hasattr(problem, "task_name") and hasattr(problem, "model"):
            plot_title = f"HPOBench: {problem.task_name} / {problem.model}"
        else:  # generic offline-finite problem (e.g. cnn-leak): no task_name/model
            plot_subject = (
                getattr(problem, "target_name", None)
                or getattr(problem, "subject_id", "")
            )
            plot_title = f"{getattr(problem, 'dataset_id', adapter)}: {plot_subject}"
        save_performance_plot(
            [method_run],
            directory / "performance.png",
            title=plot_title,
            response_label="Mean validation accuracy",
            optimum_value=problem.optimum_value,
        )
    outcome = {
        "alpha_requested": compiled_alpha if fixed_alpha is None else fixed_alpha,
        "alpha_effective": effective_alpha,
        "strategy": strategy,
        "metrics": metric_record,
    }
    if fixed_alpha is not None:  # the compiled alpha that the method's alpha replaced
        outcome["alpha_fixed"] = fixed_alpha
        outcome["alpha_compiled"] = compiled_alpha
    if gp_encoding is not None:  # GP input encoding of the no-prior BO arms
        outcome["gp_encoding"] = gp_encoding
    if strategy == "pi-bo":  # the resolved beta and the initial design built from the prior
        outcome["pi_bo_beta"] = float(optimizer.pi_bo_beta)
        outcome["pi_bo_initial_indices"] = [int(i) for i in optimizer.initial_indices]
    if llm_ei is not None:
        outcome["llm_ei"] = {**llm_ei.summary(), "gate_fallbacks": list(optimizer.gate_fallbacks_)}
    return outcome


def _encode_solar(candidates: Sequence[Mapping[str, Any]], spec: Any) -> np.ndarray:
    blocks = []
    for variable in spec.variables:
        values = np.asarray([candidate[variable.name] for candidate in candidates], dtype=float)
        if variable.kind == "categorical":
            indices = values.astype(int) - int(variable.lower)
            blocks.append(np.eye(int(variable.sampling_upper - variable.lower + 1))[indices])
        else:
            blocks.append(
                ((values - variable.lower) / (variable.sampling_upper - variable.lower)).reshape(-1, 1)
            )
    return np.column_stack(blocks)


# Optimizer override of the SOLAR path, the same for every method: the hidden-failure (crash)
# model. As on the offline path, the alpha-pi arm shares the seeded random initial design with
# vanilla BO.
SOLAR_OPTIMIZER_EXTRA: dict[str, Any] = {"model_hidden_failures": True}


def _run_solar(
    directory: Path,
    run: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    method: Mapping[str, Any],
) -> dict[str, Any]:
    from main import _save_solar_plot, encode_candidates_for_gp

    instance = str(run["parameters"]["instance"])
    spec = resolve_solar_instance(instance)
    defaults = benchmark.get("defaults", {})
    pool_size = int(run["parameters"].get("candidate_pool_size", defaults.get("candidate_pool_size", 256)))
    budget = int(run["budget"])
    if not 1 <= budget <= pool_size:
        raise ValueError("SOLAR budget must not exceed candidate_pool_size")
    # The pool is pre-screened on the simulator's documented a-priori input relations (no
    # simulator queries; see datasets.solar.APRIORI_CHECKS), and simulator crashes are
    # modeled as a hidden constraint (crash surrogate) instead of entering the GPs.
    candidates = generate_solar_candidate_pool(
        instance, pool_size, seed=int(run["seed"]), prescreen=True)
    prior_path = None if run.get("prior_path") is None else Path(run["prior_path"])
    gp_encoding: str | None = None
    fixed_alpha = _method_alpha(method)
    compiled_alpha: float | None = None
    if method.get("requires_prior") and str(method["strategy"]) != "alpha-pi":
        # SOLAR supports the alpha-pi injection arm only; the other injection arms
        # run on the offline benchmarks.
        raise NotImplementedError(
            f"SOLAR supports the alpha-pi injection arm only, not {method['strategy']!r}"
        )
    if method["strategy"] == "alpha-pi":
        if prior_path is None:
            raise ValueError("alpha-pi requires a prior_path")
        compiler = compile_prior(load_record(prior_path))
        expected = tuple(variable.name for variable in spec.variables)
        if tuple(compiler.parameter_names) != expected:
            raise ValueError(f"SOLAR prior dimensions must be {list(expected)}")
        encoded = encode_candidates_for_gp(candidates, compiler)
        prior = np.asarray(compiler.pi(candidates), dtype=float)
        compiled_alpha = float(compiler.alpha)
        alpha = compiled_alpha if fixed_alpha is None else fixed_alpha
    elif method["strategy"] == "vanilla-bo":
        # The no-prior BO arm shares the alpha-pi arm's GP input encoding.
        encoded, gp_encoding = _solar_task_gp_encoding(candidates, spec, instance, prior_path)
        prior, alpha = None, 0.0
    else:
        encoded = _encode_solar(candidates, spec)
        prior, alpha = None, 0.0
    optimizer = _build_optimizer(
        encoded,
        run,
        method,
        prior=prior,
        alpha=alpha,
        maximize=False,
        n_constraints=spec.n_constraints,
        extra=dict(SOLAR_OPTIMIZER_EXTRA),
    )
    executable = Path(run["parameters"].get("solar_executable", SOLAR_EXECUTABLE)).resolve()
    fidelity = float(run["parameters"].get("fidelity", defaults.get("fidelity", 1.0)))
    replications = run["parameters"].get("replications", defaults.get("replications", 1))
    records = []
    while optimizer.n_observations < budget:
        # Query 1 comes from ask() like every subsequent step (and like the offline
        # path), so the seeded initial design applies.
        index = optimizer.ask()[0]
        started = time.perf_counter()
        evaluation = evaluate_solar(
            instance,
            candidates[index],
            executable=executable,
            seed=int(run["seed"]),
            fidelity=fidelity,
            replications=replications,
            timeout_seconds=run["parameters"].get("timeout_seconds"),
        )
        elapsed = time.perf_counter() - started
        # Crashes are flagged, excluded from the objective/constraint GPs, and modeled by
        # the crash surrogate; the numeric placeholders never enter a fit.
        optimizer.tell(
            index,
            0.0 if evaluation.hidden_failure else evaluation.objective,
            evaluation.constraints,
            crashed=evaluation.hidden_failure,
        )
        records.append(
            {
                "query": optimizer.n_observations,
                "candidate_index": index,
                "candidate": dict(candidates[index]),
                "objective": evaluation.objective,
                "constraints": evaluation.constraints.tolist(),
                "feasible": evaluation.feasible,
                "hidden_failure": evaluation.hidden_failure,
                "maximum_constraint_violation": float(np.max(np.maximum(evaluation.constraints, 0.0))) if evaluation.constraints.size else 0.0,
                "elapsed_seconds": elapsed,
                "raw_output": evaluation.raw_output,
            }
        )
    feasible = [record for record in records if record["feasible"]]
    best = min(feasible, key=lambda item: item["objective"]) if feasible else min(
        records, key=lambda item: item["maximum_constraint_violation"]
    )
    metric_record = {
        "best_observed_value": best["objective"],
        "best_feasible_objective": min((item["objective"] for item in feasible), default=None),
        "feasible_evaluation_count": len(feasible),
        "hidden_failure_count": sum(item["hidden_failure"] for item in records),
        "maximum_constraint_violation": best["maximum_constraint_violation"],
    }
    _write_json(
        directory / "metrics.json",
        {"artifact_format": "llm-bo-study-metrics-v1", "methods": {run["method_id"]: metric_record}},
    )
    _write_json(directory / "best_candidate.json", best)
    with (directory / "evaluations.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=_jsonable) + "\n")
    _save_solar_plot(
        {run["method_id"]: records},
        {run["method_id"]: method["label"]},
        directory / "performance.png",
        instance,
    )
    outcome = {
        "alpha_requested": alpha,
        "alpha_effective": float(optimizer.acquisition.alpha),
        "metrics": metric_record,
        "solar_source_commit": SOLAR_COMMIT,
        "solar_executable": str(executable),
        "solar_executable_sha256": _sha256(executable),
    }
    if gp_encoding is not None:  # GP input encoding of the no-prior BO arm
        outcome["gp_encoding"] = gp_encoding
    if fixed_alpha is not None:  # the compiled alpha that the method's alpha replaced
        outcome["alpha_fixed"] = fixed_alpha
        outcome["alpha_compiled"] = compiled_alpha
    return outcome


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def execute_run_directory(run_directory: str | Path) -> dict[str, Any]:
    """Execute exactly one ``study_run.json`` and write standardized artifacts."""

    directory = Path(run_directory).expanduser().resolve()
    run_path = directory / "study_run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    catalog = load_catalog()
    benchmark = catalog.benchmarks[run["benchmark_id"]]
    method = catalog.methods[run["method_id"]]
    if benchmark["adapter"] == "solar":
        outcome = _run_solar(directory, run, benchmark, method)
    else:
        outcome = _run_offline(directory, run, benchmark, method)

    prior_path = None if run.get("prior_path") is None else Path(run["prior_path"])
    study_source = Path(run["study_source_path"])
    configuration = {
        "artifact_format": "llm-bo-study-run-config-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run["run_id"],
        "study_id": run["study_id"],
        "study_source_path": str(study_source),
        "study_source_sha256": _sha256(study_source),
        "benchmark_id": run["benchmark_id"],
        "benchmark_catalog": benchmark["_source_path"],
        "method_id": run["method_id"],
        "method_catalog": method["_source_path"],
        "budget": run["budget"],
        "seed": run["seed"],
        "replicate": run["replicate"],
        "parameters": run["parameters"],
        "optimizer": run.get("optimizer", {}),
        "prior_path": None if prior_path is None else str(prior_path),
        "prior_sha256": None if prior_path is None else _sha256(prior_path),
        "git": _git_state(),
        "outputs": {
            "evaluations": str(directory / "evaluations.jsonl"),
            "metrics": str(directory / "metrics.json"),
            "best_candidate": str(directory / "best_candidate.json"),
            "performance_plot": str(directory / "performance.png"),
        },
        **outcome,
    }
    _write_json(directory / "run_config.json", configuration)
    return configuration
