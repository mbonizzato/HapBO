"""Deterministically compile a HapBO PSL belief document into a prior landscape.

This module defines the semantics of PSL v1.0 (``PSL_VERSION``): the closed form of
every token, the normalization reference set and the anchor blend.

The input is the JSON stored in the sibling ``expert_knowledge`` directory and
produced from a versioned ``prompt_card.md`` under ``experiments/priors`` and
the shared vocabulary referenced by its metadata. No LLM is called here: the
JSON is validated and every PSL token is translated into a fixed numerical
shape.

The compiler exposes two related quantities:

``pi(X)``
    The normalized expert landscape in ``[0, 1]``.

``multiplier(X)``
    The trust-adjusted factor applied to the raw acquisition by alpha-pi BO:
    ``(1 - alpha) + alpha * pi(X)``.  Since alpha is clamped to ``[0.1, 0.9]``,
    a mistaken prior never makes any candidate unreachable.

Coordinates supplied to the public API are in the native units from
``schema.dimensions``.  Categorical values may be labels or zero-based indices.
Log-scale dimensions are transformed internally; callers must not pre-log them.

Example
-------

    record = load_record(
        "experiments/priors/hpobench-nn-car/car/llm-card/prior.json"
    )
    prior = compile_prior(record)
    values = prior.multiplier([
        [1e-4, 32, 12, 48, 0.1, 1e-4, "adamw"],
        [1e-2, 128, 24, 128, 0.5, 1e-2, "sgd"],
    ])

``build_landscapes`` returns one-dimensional slices for plots; every slice shares
the same joint compiler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats


SHAPING_TOKENS = {
    "bump",
    "trend",
    "threshold",
    "periodic",
    "categorical",
    "joint_region",
    "ridge",
    "conditional",
    "plateau",
    "avoid",
}
META_TOKENS = {"smoothness", "noise", "modality"}
VALID_TOKENS = SHAPING_TOKENS | META_TOKENS | {"baseline", "anchors"}
TOKEN_FIELDS = {
    "baseline": {"token", "level", "tilt"},
    "bump": {
        "token",
        "dim",
        "mode",
        "ci",
        "skew",
        "sign",
        "weight",
        "provenance",
        "note",
    },
    "trend": {
        "token",
        "dim",
        "direction",
        "knee",
        "weight",
        "provenance",
        "note",
    },
    "threshold": {
        "token",
        "dim",
        "at",
        "active",
        "sharpness",
        "weight",
        "provenance",
        "note",
    },
    "periodic": {
        "token",
        "dim",
        "period",
        "phase",
        "weight",
        "provenance",
        "note",
    },
    "categorical": {
        "token",
        "dim",
        "preferences",
        "weight",
        "provenance",
        "note",
    },
    "joint_region": {
        "token",
        "dims",
        "mode",
        "ci",
        "weight",
        "provenance",
        "note",
    },
    "ridge": {
        "token",
        "dims",
        "relation",
        "width",
        "through",
        "weight",
        "provenance",
        "note",
    },
    "conditional": {
        "token",
        "when",
        "then",
        "mode",
        "weight",
        "provenance",
        "note",
    },
    "plateau": {"token", "region", "weight", "provenance", "note"},
    "avoid": {
        "token",
        "region",
        "dim",
        "mode",
        "ci",
        "weight",
        "provenance",
        "note",
    },
    "anchors": {"token", "bandwidth", "provenance", "note", "points"},
    "smoothness": {"token", "level", "per_dim"},
    "noise": {"token", "level", "repeats_recommended"},
    "modality": {"token", "expectation", "n_optima"},
}
PROVENANCE_TRUST = {
    "user_stated": 1.00,
    "user_confirmed": 0.90,
    "literature": 0.85,
    "llm_domain": 0.65,
    "llm_default": 0.40,
}
# Provenance values that name a human source. A belief may carry one only when its task card
# carries a human (expert) statement, declared as card.human_statement in the prior's metadata.
HUMAN_PROVENANCE = ("user_stated", "user_confirmed")
OVERALL_TRUST = {
    "very_low": 0.20,
    "low": 0.35,
    "medium": 0.50,
    "high": 0.70,
    "very_high": 0.90,
}

PSL_VERSION = "1.0"
DEFAULT_BASELINE = 0.1
DEFAULT_ANCHOR_BANDWIDTH = 0.15
CI_Z_90 = 1.645
NORMALIZATION_POINTS = 8192


def _fail(message: str) -> None:
    raise ValueError(message)


def _is_trough(belief: Mapping[str, Any]) -> bool:
    return belief.get("token") == "bump" and belief.get("sign", "peak") == "trough"


def _sha256_text_artifact(path: Path) -> str:
    """SHA-256 of a text artifact with CRLF/CR normalized to LF before hashing.

    Every hashed prior artifact (prior.json, prompt_card.md, metadata.json, the shared
    vocabulary card) is text. Recorded digests are LF-normalized, so a checkout that
    materializes CRLF line endings (e.g. git on Windows overriding the repo
    .gitattributes) still verifies byte-exactly. Recording and validation must both go
    through this helper.
    """
    data = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{where} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        _fail(f"{where} must be finite, got {value!r}")
    return value


@dataclass(frozen=True)
class DimensionSpec:
    """One typed search-space dimension and its native/normalized geometry."""

    name: str
    type: str
    range: tuple[float, float] | None = None
    values: tuple[Any, ...] | None = None
    scale: str = "linear"
    meaning: str | None = None

    @classmethod
    def from_json(cls, raw: Mapping[str, Any], position: int) -> "DimensionSpec":
        where = f"schema.dimensions[{position}]"
        if not isinstance(raw, Mapping):
            _fail(f"{where} must be an object")
        allowed = {"name", "type", "range", "values", "scale", "meaning"}
        extra = set(raw) - allowed
        if extra:
            _fail(f"{where} has unknown fields {sorted(extra)}")

        name = raw.get("name")
        dim_type = raw.get("type")
        if not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
            _fail(f"{where}.name must be a lowercase snake_case identifier")
        if dim_type not in {"continuous", "ordinal", "categorical"}:
            _fail(
                f"{where}.type must be 'continuous', 'ordinal', or 'categorical'"
            )

        meaning = raw.get("meaning")
        if meaning is not None and not isinstance(meaning, str):
            _fail(f"{where}.meaning must be a string")

        if dim_type == "continuous":
            raw_range = raw.get("range")
            if not isinstance(raw_range, list) or len(raw_range) != 2:
                _fail(f"{where}.range must contain [low, high]")
            low = _number(raw_range[0], f"{where}.range[0]")
            high = _number(raw_range[1], f"{where}.range[1]")
            if low >= high:
                _fail(f"{where}.range must satisfy low < high")
            scale = raw.get("scale", "linear")
            if scale not in {"linear", "log"}:
                _fail(f"{where}.scale must be 'linear' or 'log'")
            if scale == "log" and low <= 0:
                _fail(f"{where}: log scale requires a positive lower bound")
            return cls(name, dim_type, (low, high), None, scale, meaning)

        raw_values = raw.get("values")
        if not isinstance(raw_values, list) or not raw_values:
            _fail(f"{where}.values must be a non-empty array")
        if len({json.dumps(v, sort_keys=True) for v in raw_values}) != len(raw_values):
            _fail(f"{where}.values must not contain duplicates")
        if "scale" in raw:
            _fail(f"{where}.scale is only valid for continuous dimensions")
        return cls(name, dim_type, None, tuple(raw_values), "linear", meaning)

    @property
    def size(self) -> int | None:
        return len(self.values) if self.values is not None else None

    def _continuous_to_unit(self, values: np.ndarray) -> np.ndarray:
        assert self.range is not None
        low, high = self.range
        values = np.asarray(values, dtype=float)
        if np.any(~np.isfinite(values)):
            _fail(f"dimension {self.name!r} contains a non-finite coordinate")
        # Unit/native round trips at logarithmic bounds can differ by a few
        # floating-point ulps (for example 9.999999999999997e-06 vs 1e-05).
        tolerance = 16 * np.finfo(float).eps * max(1.0, abs(low), abs(high))
        if np.any((values < low - tolerance) | (values > high + tolerance)):
            _fail(
                f"dimension {self.name!r} coordinates must lie in [{low}, {high}]"
            )
        values = np.clip(values, low, high)
        if self.scale == "log":
            return (np.log(values) - math.log(low)) / (math.log(high) - math.log(low))
        return (values - low) / (high - low)

    def to_unit_scalar(self, value: Any, where: str | None = None) -> float:
        where = where or self.name
        if self.type == "continuous":
            try:
                return float(self._continuous_to_unit(np.asarray([value]))[0])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{where} is not a valid coordinate: {value!r}") from exc

        assert self.values is not None
        try:
            index = self.values.index(value)
        except ValueError:
            _fail(f"{where} must be one of {list(self.values)!r}, got {value!r}")
        if len(self.values) == 1:
            return 0.0
        return index / (len(self.values) - 1)

    def encode_column(self, values: np.ndarray) -> np.ndarray:
        if self.type == "continuous":
            return self._continuous_to_unit(values)

        assert self.values is not None
        out = np.empty(len(values), dtype=float)
        for row, value in enumerate(values):
            try:
                native_index = self.values.index(value)
            except ValueError:
                native_index = None
            # Numeric zero-based category indices remain accepted for the BO
            # representation of the landscape generator.
            if native_index is not None:
                index = native_index
            elif self.type == "categorical" and isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                numeric = float(value)
                if numeric.is_integer() and 0 <= int(numeric) < len(self.values):
                    index = int(numeric)
                else:
                    _fail(
                        f"dimension {self.name!r} category index {value!r} is invalid"
                    )
            else:
                try:
                    index = self.values.index(value)
                except ValueError:
                    _fail(
                        f"dimension {self.name!r} must be one of "
                        f"{list(self.values)!r}, got {value!r}"
                    )
            out[row] = 0.0 if len(self.values) == 1 else index / (len(self.values) - 1)
        return out

    def unit_to_native(self, unit: np.ndarray) -> np.ndarray:
        unit = np.clip(np.asarray(unit, dtype=float), 0.0, 1.0)
        if self.type == "continuous":
            assert self.range is not None
            low, high = self.range
            if self.scale == "log":
                return np.exp(math.log(low) + unit * (math.log(high) - math.log(low)))
            return low + unit * (high - low)

        assert self.values is not None
        if len(self.values) == 1:
            indices = np.zeros(unit.shape, dtype=int)
        else:
            indices = np.rint(unit * (len(self.values) - 1)).astype(int)
        return np.asarray(self.values, dtype=object)[indices]

    def plotting_grid(self, n_points: int) -> np.ndarray:
        if self.type == "continuous":
            assert self.range is not None
            low, high = self.range
            if self.scale == "log":
                return np.geomspace(low, high, n_points)
            return np.linspace(low, high, n_points)
        assert self.values is not None
        if self.type == "categorical":
            return np.arange(len(self.values))
        return np.asarray(self.values)


def _dimensions(record: Mapping[str, Any]) -> list[DimensionSpec]:
    raw_schema = record.get("schema")
    if not isinstance(raw_schema, Mapping):
        _fail("record must contain a 'schema' object")
    if set(raw_schema) - {"problem", "objective", "dimensions"}:
        _fail(
            "schema has unknown fields "
            f"{sorted(set(raw_schema) - {'problem', 'objective', 'dimensions'})}"
        )
    if raw_schema.get("objective") not in {"maximize", "minimize"}:
        _fail("schema.objective must be 'maximize' or 'minimize'")
    raw_dims = raw_schema.get("dimensions")
    if not isinstance(raw_dims, list) or not raw_dims:
        _fail("schema.dimensions must be a non-empty array")
    dims = [DimensionSpec.from_json(raw, i) for i, raw in enumerate(raw_dims)]
    names = [dim.name for dim in dims]
    if len(set(names)) != len(names):
        _fail("schema dimension names must be unique")
    return dims


def _validate_interval(
    interval: Any, dim: DimensionSpec, where: str
) -> tuple[float, float]:
    if not isinstance(interval, list) or len(interval) != 2:
        _fail(f"{where} must contain [low, high]")
    low = dim.to_unit_scalar(interval[0], f"{where}[0]")
    high = dim.to_unit_scalar(interval[1], f"{where}[1]")
    if low > high:
        _fail(f"{where} must satisfy low <= high")
    return low, high


def _validate_belief(
    belief: Mapping[str, Any],
    position: int,
    dims_by_name: Mapping[str, DimensionSpec],
) -> None:
    where = f"beliefs[{position}]"
    if not isinstance(belief, Mapping):
        _fail(f"{where} must be an object")
    token = belief.get("token")
    if token not in VALID_TOKENS:
        _fail(f"{where}.token {token!r} is not in the PSL vocabulary")
    extra = set(belief) - TOKEN_FIELDS[token]
    if extra:
        _fail(f"{where} has fields not allowed for {token!r}: {sorted(extra)}")
    if "note" in belief and not isinstance(belief["note"], str):
        _fail(f"{where}.note must be a string")

    if token in SHAPING_TOKENS:
        weight = _number(belief.get("weight"), f"{where}.weight")
        if not 0 <= weight <= 1:
            _fail(f"{where}.weight must lie in [0, 1]")
        if belief.get("provenance") not in PROVENANCE_TRUST:
            _fail(f"{where}.provenance is missing or invalid")

    def named_dim(key: str = "dim") -> DimensionSpec:
        name = belief.get(key)
        if name not in dims_by_name:
            _fail(f"{where}.{key} refers to unknown dimension {name!r}")
        return dims_by_name[str(name)]

    if token == "baseline":
        if "level" in belief:
            level = _number(belief["level"], f"{where}.level")
            if not 0 <= level <= 1:
                _fail(f"{where}.level must lie in [0, 1]")
        tilt = belief.get("tilt", {})
        if not isinstance(tilt, Mapping):
            _fail(f"{where}.tilt must be an object")
        for name, slope in tilt.items():
            if name not in dims_by_name:
                _fail(f"{where}.tilt refers to unknown dimension {name!r}")
            if dims_by_name[name].type == "categorical":
                _fail(f"{where}.tilt cannot target a categorical dimension")
            if not -1 <= _number(slope, f"{where}.tilt.{name}") <= 1:
                _fail(f"{where}.tilt slopes must lie in [-1, 1]")

    elif token in {"bump", "trend", "threshold", "periodic"}:
        dim = named_dim()
        if dim.type == "categorical":
            _fail(f"{where}: {token!r} cannot target a categorical dimension")

    if token == "bump":
        dim.to_unit_scalar(belief.get("mode"), f"{where}.mode")
        if "ci" in belief:
            _validate_interval(belief["ci"], dim, f"{where}.ci")
        if "skew" in belief and not -1 <= _number(
            belief["skew"], f"{where}.skew"
        ) <= 1:
            _fail(f"{where}.skew must lie in [-1, 1]")
        if belief.get("sign", "peak") not in {"peak", "trough"}:
            _fail(f"{where}.sign must be 'peak' or 'trough'")

    elif token == "trend":
        if belief.get("direction") not in {"increase", "decrease"}:
            _fail(f"{where}.direction must be 'increase' or 'decrease'")
        if "knee" in belief:
            named_dim().to_unit_scalar(belief["knee"], f"{where}.knee")

    elif token == "threshold":
        named_dim().to_unit_scalar(belief.get("at"), f"{where}.at")
        if belief.get("active") not in {"above", "below"}:
            _fail(f"{where}.active must be 'above' or 'below'")
        if belief.get("sharpness", "soft") not in {"soft", "sharp"}:
            _fail(f"{where}.sharpness must be 'soft' or 'sharp'")

    elif token == "periodic":
        period = _number(belief.get("period"), f"{where}.period")
        if period <= 0:
            _fail(f"{where}.period must be positive")
        if dim.type == "ordinal" and not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in dim.values
        ):
            _fail(f"{where}: periodic requires numeric ordinal values")
        if dim.type == "continuous":
            span = dim.range[1] - dim.range[0]
        else:
            span = float(max(dim.values)) - float(min(dim.values))
        if span > 0 and period < 1e-9 * span:
            _fail(f"{where}.period is too small for the range of {dim.name!r}")
        if "phase" in belief:
            # Any finite native value: the phase is a reference point of the oscillation,
            # not a coordinate of the search space.
            _number(belief["phase"], f"{where}.phase")

    elif token == "categorical":
        dim = named_dim()
        if dim.type != "categorical":
            _fail(f"{where}: categorical token requires a categorical dimension")
        preferences = belief.get("preferences")
        if not isinstance(preferences, Mapping):
            _fail(f"{where}.preferences must be an object")
        assert dim.values is not None
        valid_levels = {str(v) for v in dim.values}
        unknown = set(preferences) - valid_levels
        if unknown:
            _fail(f"{where}.preferences contains unknown levels {sorted(unknown)}")
        for level, preference in preferences.items():
            if not 0 <= _number(
                preference, f"{where}.preferences[{level!r}]"
            ) <= 1:
                _fail(f"{where}.preferences values must lie in [0, 1]")

    elif token in {"joint_region", "ridge"}:
        names = belief.get("dims")
        required_count = 2 if token == "ridge" else None
        if (
            not isinstance(names, list)
            or len(names) < 2
            or (required_count is not None and len(names) != required_count)
            or len(set(names)) != len(names)
        ):
            _fail(f"{where}.dims must contain distinct valid dimension names")
        for name in names:
            if name not in dims_by_name:
                _fail(f"{where}.dims refers to unknown dimension {name!r}")
            if dims_by_name[name].type == "categorical":
                _fail(f"{where}: {token!r} requires geometric dimensions")
        if token == "joint_region":
            mode = belief.get("mode")
            if not isinstance(mode, Mapping) or set(mode) != set(names):
                _fail(f"{where}.mode must give one coordinate per listed dimension")
            for name, value in mode.items():
                dims_by_name[name].to_unit_scalar(value, f"{where}.mode.{name}")
            if "ci" in belief:
                ci = belief["ci"]
                if not isinstance(ci, Mapping) or set(ci) != set(names):
                    _fail(f"{where}.ci must give one interval per listed dimension")
                for name, interval in ci.items():
                    _validate_interval(interval, dims_by_name[name], f"{where}.ci.{name}")
        else:
            if belief.get("relation") not in {"negative", "positive"}:
                _fail(f"{where}.relation must be 'negative' or 'positive'")
            if belief.get("width", "medium") not in {"narrow", "medium", "wide"}:
                _fail(f"{where}.width must be 'narrow', 'medium', or 'wide'")
            through = belief.get("through")
            if through is not None:
                if not isinstance(through, Mapping) or set(through) != set(names):
                    _fail(f"{where}.through must give one coordinate per listed dimension")
                for name, value in through.items():
                    dims_by_name[name].to_unit_scalar(value, f"{where}.through.{name}")

    elif token == "conditional":
        when = belief.get("when")
        then = belief.get("then")
        if not isinstance(when, Mapping) or set(when) != {"dim", "region"}:
            _fail(f"{where}.when must contain exactly 'dim' and 'region'")
        gate_name = when["dim"]
        if gate_name not in dims_by_name:
            _fail(f"{where}.when.dim refers to unknown dimension {gate_name!r}")
        _validate_interval(
            when["region"], dims_by_name[gate_name], f"{where}.when.region"
        )
        if not isinstance(then, Mapping) or then.get("token") not in {
            "bump",
            "trend",
            "threshold",
            "plateau",
        }:
            _fail(f"{where}.then must be a bump, trend, threshold, or plateau")
        _validate_nested_shape(then, where + ".then", dims_by_name)
        if belief.get("mode", "gate") not in {"gate", "synergy"}:
            _fail(f"{where}.mode must be 'gate' or 'synergy'")

    elif token in {"plateau", "avoid"}:
        if token == "avoid" and ("region" in belief) == ("dim" in belief):
            _fail(
                f"{where}: avoid must use either 'region' or 'dim'/'mode', not both"
            )
        if token == "avoid" and "region" not in belief:
            dim = named_dim()
            if dim.type == "categorical":
                _fail(f"{where}: bump-shaped avoid requires a geometric dimension")
            dim.to_unit_scalar(belief.get("mode"), f"{where}.mode")
            if "ci" in belief:
                _validate_interval(belief["ci"], dim, f"{where}.ci")
        else:
            _validate_region(belief.get("region"), where + ".region", dims_by_name)

    elif token == "anchors":
        if "provenance" in belief and belief["provenance"] not in PROVENANCE_TRUST:
            _fail(f"{where}.provenance is invalid")
        points = belief.get("points")
        if not isinstance(points, list) or not points:
            _fail(f"{where}.points must be a non-empty array")
        if "bandwidth" in belief and _number(
            belief["bandwidth"], f"{where}.bandwidth"
        ) <= 0:
            _fail(f"{where}.bandwidth must be positive")
        for p, point in enumerate(points):
            point_where = f"{where}.points[{p}]"
            if not isinstance(point, Mapping) or set(point) != {"coords", "value"}:
                _fail(f"{point_where} must contain exactly 'coords' and 'value'")
            coords = point["coords"]
            if not isinstance(coords, Mapping) or set(coords) != set(dims_by_name):
                _fail(f"{point_where}.coords must give every schema dimension")
            for name, value in coords.items():
                dims_by_name[name].to_unit_scalar(value, f"{point_where}.coords.{name}")
            value = _number(point["value"], f"{point_where}.value")
            if not 0 <= value <= 1:
                _fail(f"{point_where}.value must lie in [0, 1]")

    elif token == "smoothness":
        levels = {"very_smooth", "smooth", "medium", "rough", "very_rough"}
        if "level" in belief and belief["level"] not in levels:
            _fail(f"{where}.level is invalid")
        per_dim = belief.get("per_dim", {})
        if not isinstance(per_dim, Mapping):
            _fail(f"{where}.per_dim must be an object")
        for name, level in per_dim.items():
            if name not in dims_by_name:
                _fail(f"{where}.per_dim refers to unknown dimension {name!r}")
            if level not in levels:
                _fail(f"{where}.per_dim.{name} is invalid")

    elif token == "noise":
        if belief.get("level") not in {"low", "medium", "high", "very_high"}:
            _fail(f"{where}.level is missing or invalid")
        if "repeats_recommended" in belief and not isinstance(
            belief["repeats_recommended"], bool
        ):
            _fail(f"{where}.repeats_recommended must be a boolean")

    elif token == "modality":
        has_expectation = "expectation" in belief
        has_count = "n_optima" in belief
        if has_expectation and belief["expectation"] not in {"unimodal", "few", "many"}:
            _fail(f"{where}.expectation is invalid")
        if has_count and (
            isinstance(belief["n_optima"], bool)
            or not isinstance(belief["n_optima"], int)
            or belief["n_optima"] < 1
        ):
            _fail(f"{where}.n_optima must be a positive integer")


def _validate_nested_shape(
    shape: Mapping[str, Any],
    where: str,
    dims_by_name: Mapping[str, DimensionSpec],
) -> None:
    token = shape["token"]
    extra = set(shape) - TOKEN_FIELDS[token]
    if extra:
        _fail(f"{where} has fields not allowed for a nested {token!r}: {sorted(extra)}")
    if "note" in shape and not isinstance(shape["note"], str):
        _fail(f"{where}.note must be a string")
    # A nested shape may restate a weight and a provenance; the conditional's own weight
    # and provenance are the ones compiled, so these are only checked.
    if "weight" in shape and not 0 <= _number(shape["weight"], where + ".weight") <= 1:
        _fail(f"{where}.weight must lie in [0, 1]")
    if "provenance" in shape and shape["provenance"] not in PROVENANCE_TRUST:
        _fail(f"{where}.provenance is invalid")
    if token == "plateau":
        _validate_region(shape.get("region"), where + ".region", dims_by_name)
        return
    name = shape.get("dim")
    if name not in dims_by_name or dims_by_name[name].type == "categorical":
        _fail(f"{where}.dim must name a geometric dimension")
    dim = dims_by_name[name]
    if token == "bump":
        dim.to_unit_scalar(shape.get("mode"), where + ".mode")
        if "ci" in shape:
            _validate_interval(shape["ci"], dim, where + ".ci")
        if "skew" in shape and not -1 <= _number(shape["skew"], where + ".skew") <= 1:
            _fail(f"{where}.skew must lie in [-1, 1]")
        if shape.get("sign", "peak") not in {"peak", "trough"}:
            _fail(f"{where}.sign must be 'peak' or 'trough'")
    elif token == "trend":
        if shape.get("direction") not in {"increase", "decrease"}:
            _fail(f"{where}.direction must be 'increase' or 'decrease'")
        if "knee" in shape:
            dim.to_unit_scalar(shape["knee"], where + ".knee")
    elif token == "threshold":
        dim.to_unit_scalar(shape.get("at"), where + ".at")
        if shape.get("active") not in {"above", "below"}:
            _fail(f"{where}.active must be 'above' or 'below'")
        if shape.get("sharpness", "soft") not in {"soft", "sharp"}:
            _fail(f"{where}.sharpness must be 'soft' or 'sharp'")


def _validate_region(
    region: Any,
    where: str,
    dims_by_name: Mapping[str, DimensionSpec],
) -> None:
    if not isinstance(region, Mapping) or not region:
        _fail(f"{where} must be a non-empty object")
    for name, interval in region.items():
        if name not in dims_by_name:
            _fail(f"{where} refers to unknown dimension {name!r}")
        if dims_by_name[name].type == "categorical":
            _fail(f"{where}: box regions require geometric dimensions")
        _validate_interval(interval, dims_by_name[name], f"{where}.{name}")


def load_record(path: str | Path) -> dict[str, Any]:
    """Load and semantically validate one HapBO PSL JSON document."""

    path = Path(path)
    metadata = validate_prior_metadata(path)
    with path.open(encoding="utf-8") as handle:
        record = json.load(handle)
    validate_record(
        record,
        source=str(path),
        card_human_statement=(
            None if metadata is None else declares_human_statement(metadata)
        ),
    )
    return record


def declares_human_statement(metadata: Mapping[str, Any]) -> bool:
    """Whether prior metadata declares a human (expert) statement in its task card.

    The declaration is ``card.human_statement: true``; a prior without it is treated as
    having none, so it cannot carry ``user_stated`` / ``user_confirmed`` provenance.
    """

    card = metadata.get("card")
    return isinstance(card, Mapping) and card.get("human_statement") is True


def validate_prior_metadata(prior_path: str | Path) -> dict[str, Any] | None:
    """Validate companion provenance metadata when a versioned prior has it."""

    prior_path = Path(prior_path).expanduser().resolve()
    metadata_path = prior_path.parent / "metadata.json"
    if not metadata_path.exists():
        try:
            prior_path.relative_to(EXPERIMENT_PRIORS_DIRECTORY.resolve())
        except ValueError:
            return None
        _fail(f"{prior_path}: versioned priors require companion metadata.json")

    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, Mapping):
        _fail(f"{metadata_path}: top level must be an object")
    artifact_format = metadata.get("artifact_format")
    if artifact_format not in {
        "llm-bo-prior-metadata-v1",
        "llm-bo-prior-metadata-v2",
    }:
        _fail(f"{metadata_path}: unsupported or missing artifact_format")
    if metadata.get("status") != "finalized":
        _fail(
            f"{metadata_path}: prior is not finalized; run prior-finalize first"
        )
    if metadata.get("prior_id") != prior_path.parent.name:
        _fail(f"{metadata_path}: prior_id must match its directory name")
    if artifact_format == "llm-bo-prior-metadata-v2":
        scope = metadata.get("scope")
        if not isinstance(scope, Mapping):
            _fail(f"{metadata_path}: scope must be an object")
        target_name = scope.get("target_name")
        target_slug = scope.get("target_slug")
        if not isinstance(target_name, str) or not target_name.strip():
            _fail(f"{metadata_path}: scope.target_name must be a non-empty string")
        expected_slug = target_path_component(target_name)
        if target_slug != expected_slug:
            _fail(
                f"{metadata_path}: scope.target_slug must be {expected_slug!r} "
                "for scope.target_name"
            )
        if prior_path.parent.parent.name != target_slug:
            _fail(f"{metadata_path}: target_slug must match its directory name")
        dataset_directory = prior_path.parent.parent.parent
    else:
        dataset_directory = prior_path.parent.parent
    if metadata.get("dataset_id") != dataset_directory.name:
        _fail(f"{metadata_path}: dataset_id must match its directory name")
    if "card" in metadata and not (
        isinstance(metadata["card"], Mapping)
        and isinstance(metadata["card"].get("human_statement"), bool)
    ):
        _fail(f"{metadata_path}: card.human_statement must be true or false")

    files = metadata.get("files")
    required_files = {"prompt_card.md", "prior.json"}
    if not isinstance(files, Mapping) or not required_files <= set(files):
        _fail(
            f"{metadata_path}: files must include {sorted(required_files)}"
        )
    for filename in sorted(required_files):
        file_metadata = files[filename]
        expected_hash = (
            file_metadata.get("sha256")
            if isinstance(file_metadata, Mapping)
            else None
        )
        if not isinstance(expected_hash, str) or re.fullmatch(
            r"[0-9a-f]{64}", expected_hash
        ) is None:
            _fail(f"{metadata_path}: files.{filename}.sha256 is invalid")
        artifact_path = prior_path.parent / filename
        if not artifact_path.is_file():
            _fail(f"{metadata_path}: missing artifact {filename}")
        actual_hash = _sha256_text_artifact(artifact_path)
        if actual_hash != expected_hash:
            _fail(
                f"{metadata_path}: {filename} does not match its recorded sha256"
            )

    vocabulary = metadata.get("vocabulary")
    vocabulary_source = (
        vocabulary.get("path") if isinstance(vocabulary, Mapping) else None
    )
    vocabulary_hash = (
        vocabulary.get("sha256") if isinstance(vocabulary, Mapping) else None
    )
    if not isinstance(vocabulary_source, str) or not isinstance(
        vocabulary_hash, str
    ):
        _fail(f"{metadata_path}: vocabulary path and sha256 are required")
    if re.fullmatch(r"[0-9a-f]{64}", vocabulary_hash) is None:
        _fail(f"{metadata_path}: vocabulary.sha256 is invalid")
    vocabulary_path = (PROJECT_ROOT / vocabulary_source).resolve()
    try:
        vocabulary_path.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        _fail(f"{metadata_path}: vocabulary path must remain inside the project")
    if not vocabulary_path.is_file():
        _fail(f"{metadata_path}: missing vocabulary artifact {vocabulary_source}")
    actual_vocabulary_hash = _sha256_text_artifact(vocabulary_path)
    if actual_vocabulary_hash != vocabulary_hash:
        _fail(
            f"{metadata_path}: shared vocabulary does not match its recorded sha256"
        )
    return dict(metadata)


def validate_prior_compatibility(
    prior_path: str | Path,
    *,
    dataset_id: str,
    target_name: str,
) -> dict[str, Any] | None:
    """Reject a versioned prior scoped to another dataset or exact target."""

    metadata = validate_prior_metadata(prior_path)
    if metadata is None:
        return None
    metadata_path = Path(prior_path).expanduser().resolve().parent / "metadata.json"
    if metadata.get("dataset_id") != dataset_id:
        _fail(
            f"{metadata_path}: prior dataset {metadata.get('dataset_id')!r} does not "
            f"match selected dataset {dataset_id!r}"
        )
    scope = metadata.get("scope")
    scoped_target = scope.get("target_name") if isinstance(scope, Mapping) else None
    if not isinstance(scoped_target, str):
        _fail(
            f"{metadata_path}: this two-level prior has no exact "
            "scope.target_name; initialize a target-aware prior before using it"
        )
    if scoped_target != target_name:
        _fail(
            f"{metadata_path}: prior target {scoped_target!r} does not match "
            f"selected target {target_name!r}"
        )
    return metadata


def validate_record(
    record: Mapping[str, Any],
    source: str = "record",
    *,
    card_human_statement: bool | None = None,
) -> None:
    """Validate PSL structure plus constraints JSON Schema cannot express.

    ``card_human_statement`` says whether the task card the document was written from
    carries a human (expert) statement. When it is False, no belief may name a human
    source (``user_stated`` / ``user_confirmed``). None (a bare document, no card known)
    skips that check.
    """

    if not isinstance(record, Mapping):
        _fail(f"{source}: top level must be an object")
    required = {"schema", "beliefs", "trust"}
    if set(record) != required:
        _fail(
            f"{source}: top level must contain exactly {sorted(required)}; "
            f"got {sorted(record)}"
        )
    dims = _dimensions(record)
    dims_by_name = {dim.name: dim for dim in dims}

    beliefs = record.get("beliefs")
    if not isinstance(beliefs, list):
        _fail(f"{source}: beliefs must be an array")
    baselines = 0
    unit_weights = 0
    for position, belief in enumerate(beliefs):
        _validate_belief(belief, position, dims_by_name)
        baselines += belief["token"] == "baseline"
        if belief["token"] in SHAPING_TOKENS:
            unit_weights += math.isclose(float(belief["weight"]), 1.0)
    if baselines > 1:
        _fail(f"{source}: at most one baseline token is allowed")
    if any(b["token"] in SHAPING_TOKENS for b in beliefs) and unit_weights != 1:
        _fail(f"{source}: exactly one shaping belief must have weight 1.0")
    if card_human_statement is False:
        claimed = [
            f"beliefs[{position}]"
            for position, belief in enumerate(beliefs)
            if belief.get("provenance") in HUMAN_PROVENANCE
        ]
        if claimed:
            _fail(
                f"{source}: {', '.join(claimed)} name a human source (provenance "
                "'user_stated' or 'user_confirmed'), but the task card carries no human "
                "or expert statement. Tag every belief that comes from your own knowledge "
                "'llm_domain' ('literature' with the citation in its note, 'llm_default' "
                "for a structural guess)"
            )

    trust = record.get("trust")
    if not isinstance(trust, Mapping):
        _fail(f"{source}: trust must be an object")
    if set(trust) - {"overall_confidence", "notes"}:
        _fail(f"{source}: trust has unknown fields")
    if trust.get("overall_confidence") not in OVERALL_TRUST:
        _fail(f"{source}: trust.overall_confidence is missing or invalid")
    if "notes" in trust and not isinstance(trust["notes"], str):
        _fail(f"{source}: trust.notes must be a string")


@dataclass
class CompiledPrior:
    """A compiled, joint expert prior over all schema dimensions."""

    record: Mapping[str, Any]
    dimensions: list[DimensionSpec]
    alpha: float

    def __post_init__(self) -> None:
        self._index = {dim.name: i for i, dim in enumerate(self.dimensions)}
        self._beliefs = list(self.record["beliefs"])
        self._reference_unit = self._make_reference_point()
        self._normalization_min, self._normalization_max = self._normalization_range()

    @property
    def parameter_names(self) -> list[str]:
        return [dim.name for dim in self.dimensions]

    @property
    def reference_point(self) -> dict[str, Any]:
        """Native-unit point used for one-dimensional visualization slices."""

        point: dict[str, Any] = {}
        for i, dim in enumerate(self.dimensions):
            value = dim.unit_to_native(np.asarray([self._reference_unit[i]]))[0]
            point[dim.name] = value.item() if isinstance(value, np.generic) else value
        return point

    def _encode(self, X: Any) -> tuple[np.ndarray, bool]:
        single = False
        # Parameter names are matched case-insensitively so a benchmark whose native
        # key is not lowercase snake_case (e.g. SVM's ``C``) still aligns with its
        # PSL dimension (``c``); lowercase families are unaffected.
        param_lc = {name.lower(): name for name in self.parameter_names}
        if isinstance(X, Mapping):
            xkeys = {k.lower(): k for k in X}
            missing = set(param_lc) - set(xkeys)
            extra = set(xkeys) - set(param_lc)
            if missing or extra:
                _fail(
                    f"point keys mismatch schema (missing={sorted(param_lc[m] for m in missing)}, "
                    f"extra={sorted(xkeys[e] for e in extra)})"
                )
            native = np.asarray([[X[xkeys[name.lower()]] for name in self.parameter_names]], dtype=object)
            single = True
        elif (
            isinstance(X, Sequence)
            and not isinstance(X, (str, bytes, np.ndarray))
            and X
            and isinstance(X[0], Mapping)
        ):
            rows = []
            for row, point in enumerate(X):
                pk = {k.lower(): k for k in point}
                if set(pk) != set(param_lc):
                    _fail(f"point {row} keys do not match schema dimensions")
                rows.append([point[pk[name.lower()]] for name in self.parameter_names])
            native = np.asarray(rows, dtype=object)
        else:
            native = np.asarray(X, dtype=object)
            if native.ndim == 1:
                native = native.reshape(1, -1)
                single = True
            if native.ndim != 2:
                _fail("X must be a point, a list of points, or a two-dimensional array")

        if native.shape[1] != len(self.dimensions):
            _fail(
                f"X has {native.shape[1]} columns but schema has "
                f"{len(self.dimensions)} dimensions"
            )
        unit = np.column_stack(
            [dim.encode_column(native[:, j]) for j, dim in enumerate(self.dimensions)]
        )
        return unit, single

    def pi(self, X: Any) -> np.ndarray | float:
        """Evaluate the normalized expert landscape at native-unit points."""

        unit, single = self._encode(X)
        values = self._pi_unit(unit)
        return float(values[0]) if single else values

    def multiplier(self, X: Any) -> np.ndarray | float:
        """Evaluate ``(1-alpha) + alpha*pi(X)`` for alpha-pi BO."""

        values = self.pi(X)
        return (1.0 - self.alpha) + self.alpha * values

    def _pi_unit(self, unit: np.ndarray) -> np.ndarray:
        raw = self._raw_score(unit)
        low, high = self._normalization_min, self._normalization_max
        if math.isclose(low, high, rel_tol=1e-12, abs_tol=1e-12):
            # Constant shaping score (e.g. a document with only baseline, anchors and
            # meta tokens): min-max normalization is undefined, so the pre-anchor field
            # is the baseline level (candidates not covered by any anchor take the
            # baseline value).
            values = np.full(len(unit), self._baseline_level())
        else:
            values = np.clip((raw - low) / (high - low), 0.0, 1.0)
        return self._apply_anchors(unit, values)

    def _baseline_level(self) -> float:
        """Declared ``baseline.level`` (default 0.1), clipped to [0, 1]."""

        baseline = next(
            (b for b in self._beliefs if b["token"] == "baseline"), {}
        )
        level = float(baseline.get("level", DEFAULT_BASELINE))
        return float(np.clip(level, 0.0, 1.0))

    def _raw_score(self, unit: np.ndarray) -> np.ndarray:
        baseline = next(
            (b for b in self._beliefs if b["token"] == "baseline"), {}
        )
        score = np.full(len(unit), float(baseline.get("level", DEFAULT_BASELINE)))
        for name, slope in baseline.get("tilt", {}).items():
            score += float(slope) * (unit[:, self._index[name]] - 0.5)

        gates: list[np.ndarray] = []
        for belief in self._beliefs:
            token = belief["token"]
            if token in {"baseline", "anchors"} | META_TOKENS:
                continue
            weight = float(belief["weight"])
            if token == "categorical":
                gates.append(1.0 - weight + weight * self._categorical(belief, unit))
            elif token == "conditional" and belief.get("mode", "gate") == "synergy":
                region = self._box(
                    {belief["when"]["dim"]: belief["when"]["region"]}, unit
                )
                nested = self._shape(belief["then"], unit)
                if _is_trough(belief["then"]):
                    # A nested trough inside a multiplicative gate is the dip 1 - bump.
                    nested = 1.0 - nested
                gates.append(1.0 - weight * region * (1.0 - nested))
            else:
                shape = self._shape(belief, unit)
                negative = (
                    token == "avoid"
                    or _is_trough(belief)
                    or (token == "conditional" and _is_trough(belief["then"]))
                )
                score += (-weight if negative else weight) * shape

        score = np.maximum(score, 0.0)
        for gate in gates:
            score *= gate
        return score

    def _shape(self, belief: Mapping[str, Any], unit: np.ndarray) -> np.ndarray:
        token = belief["token"]
        if token == "bump":
            return self._bump(belief, unit)
        if token == "trend":
            return self._trend(belief, unit)
        if token == "threshold":
            return self._threshold(belief, unit)
        if token == "periodic":
            return self._periodic(belief, unit)
        if token == "joint_region":
            result = np.ones(len(unit))
            for name in belief["dims"]:
                component = {
                    "token": "bump",
                    "dim": name,
                    "mode": belief["mode"][name],
                }
                if "ci" in belief:
                    component["ci"] = belief["ci"][name]
                result *= self._bump(component, unit)
            return result
        if token == "ridge":
            return self._ridge(belief, unit)
        if token == "conditional":
            region = self._box(
                {belief["when"]["dim"]: belief["when"]["region"]}, unit
            )
            return region * self._shape(belief["then"], unit)
        if token == "plateau":
            return self._box(belief["region"], unit)
        if token == "avoid":
            if "region" in belief:
                return self._box(belief["region"], unit)
            component = dict(belief)
            component["token"] = "bump"
            component["sign"] = "peak"
            return self._bump(component, unit)
        _fail(f"cannot compile shape token {token!r}")

    def _bump(self, belief: Mapping[str, Any], unit: np.ndarray) -> np.ndarray:
        dim = self.dimensions[self._index[belief["dim"]]]
        column = unit[:, self._index[dim.name]]
        mode = dim.to_unit_scalar(belief["mode"])
        if "ci" in belief:
            ci_low, ci_high = _validate_interval(belief["ci"], dim, "ci")
            fallback = max(ci_high - ci_low, 0.05) / (2 * CI_Z_90)
            sigma_left = (mode - ci_low) / CI_Z_90
            sigma_right = (ci_high - mode) / CI_Z_90
            sigma_left = sigma_left if sigma_left > 0 else fallback
            sigma_right = sigma_right if sigma_right > 0 else fallback
        else:
            sigma_left = sigma_right = 0.15 / CI_Z_90
        skew = float(belief.get("skew", 0.0))
        sigma_left *= math.exp(-skew)
        sigma_right *= math.exp(skew)
        sigma = np.where(column < mode, sigma_left, sigma_right)
        return np.exp(-0.5 * ((column - mode) / sigma) ** 2)

    def _trend(self, belief: Mapping[str, Any], unit: np.ndarray) -> np.ndarray:
        dim = self.dimensions[self._index[belief["dim"]]]
        column = unit[:, self._index[dim.name]]
        direction = belief["direction"]
        if "knee" not in belief:
            return column if direction == "increase" else 1.0 - column
        knee = dim.to_unit_scalar(belief["knee"])
        if direction == "increase":
            return np.ones(len(unit)) if knee <= 0 else np.clip(column / knee, 0, 1)
        return (
            np.ones(len(unit))
            if knee >= 1
            else np.clip((1.0 - column) / (1.0 - knee), 0, 1)
        )

    def _threshold(self, belief: Mapping[str, Any], unit: np.ndarray) -> np.ndarray:
        dim = self.dimensions[self._index[belief["dim"]]]
        column = unit[:, self._index[dim.name]]
        at = dim.to_unit_scalar(belief["at"])
        width = 0.015 if belief.get("sharpness", "soft") == "sharp" else 0.06
        signed = (column - at) / width
        if belief["active"] == "below":
            signed = -signed
        return stats.logistic.cdf(signed)

    def _periodic(self, belief: Mapping[str, Any], unit: np.ndarray) -> np.ndarray:
        dim = self.dimensions[self._index[belief["dim"]]]
        column = unit[:, self._index[dim.name]]
        period_native = float(belief["period"])
        phase_native = float(belief.get("phase", 0.0))
        if (
            dim.type == "continuous"
            and dim.scale == "linear"
            and dim.range[0] <= phase_native
            and phase_native + period_native <= dim.range[1]
        ):
            # Linear range with the phase and a full period inside it: evaluated on the
            # encoded coordinate, which equals the native-unit form below.
            phase = dim.to_unit_scalar(phase_native)
            period = abs(dim.to_unit_scalar(phase_native + period_native) - phase)
            period = max(period, np.finfo(float).eps)
            return 0.5 * (1.0 + np.cos(2 * np.pi * (column - phase) / period))
        # Otherwise the period is applied in native units: a linear range whose phase lies
        # outside it or whose first period extends past the upper bound, a log range, or
        # numeric ordinal levels (interpolated linearly between levels on the relaxed cube).
        if dim.type == "ordinal":
            assert dim.values is not None
            values = np.asarray(dim.values, dtype=float)
            native = np.interp(column * (len(values) - 1), np.arange(len(values)), values)
        else:
            assert dim.range is not None
            native = dim.unit_to_native(column)
        return 0.5 * (1.0 + np.cos(2 * np.pi * (native - phase_native) / period_native))

    def _categorical(
        self, belief: Mapping[str, Any], unit: np.ndarray
    ) -> np.ndarray:
        dim = self.dimensions[self._index[belief["dim"]]]
        assert dim.values is not None
        if len(dim.values) == 1:
            indices = np.zeros(len(unit), dtype=int)
        else:
            indices = np.rint(
                unit[:, self._index[dim.name]] * (len(dim.values) - 1)
            ).astype(int)
        preferences = belief["preferences"]
        return np.asarray(
            [float(preferences.get(str(dim.values[i]), 1.0)) for i in indices]
        )

    def _ridge(self, belief: Mapping[str, Any], unit: np.ndarray) -> np.ndarray:
        first, second = belief["dims"]
        x = unit[:, self._index[first]]
        y = unit[:, self._index[second]]
        if "through" in belief:
            x0 = self.dimensions[self._index[first]].to_unit_scalar(
                belief["through"][first]
            )
            y0 = self.dimensions[self._index[second]].to_unit_scalar(
                belief["through"][second]
            )
        else:
            x0 = y0 = 0.5
        slope = 1.0 if belief["relation"] == "positive" else -1.0
        perpendicular_distance = np.abs((y - y0) - slope * (x - x0)) / math.sqrt(2)
        sigma = {"narrow": 0.05, "medium": 0.12, "wide": 0.25}[
            belief.get("width", "medium")
        ]
        return np.exp(-0.5 * (perpendicular_distance / sigma) ** 2)

    def _box(
        self, region: Mapping[str, Sequence[Any]], unit: np.ndarray
    ) -> np.ndarray:
        inside = np.ones(len(unit), dtype=bool)
        for name, interval in region.items():
            dim = self.dimensions[self._index[name]]
            low, high = _validate_interval(interval, dim, f"region.{name}")
            column = unit[:, self._index[name]]
            inside &= (column >= low) & (column <= high)
        return inside.astype(float)

    def _make_reference_point(self) -> np.ndarray:
        reference = np.full(len(self.dimensions), 0.5)
        strongest: dict[str, float] = {}
        for belief in self._beliefs:
            token = belief["token"]
            weight = float(belief.get("weight", 0.0))
            if token == "bump" and belief.get("sign", "peak") == "peak":
                name = belief["dim"]
                if weight >= strongest.get(name, -1):
                    reference[self._index[name]] = self.dimensions[
                        self._index[name]
                    ].to_unit_scalar(belief["mode"])
                    strongest[name] = weight
            elif token == "categorical":
                name = belief["dim"]
                if weight >= strongest.get(name, -1):
                    dim = self.dimensions[self._index[name]]
                    assert dim.values is not None
                    best = max(
                        range(len(dim.values)),
                        key=lambda i: float(
                            belief["preferences"].get(str(dim.values[i]), 1.0)
                        ),
                    )
                    reference[self._index[name]] = (
                        0.0 if len(dim.values) == 1 else best / (len(dim.values) - 1)
                    )
                    strongest[name] = weight
        return reference

    def _normalization_range(self) -> tuple[float, float]:
        d = len(self.dimensions)
        sampler = stats.qmc.Halton(d=d, scramble=False)
        points = sampler.random(NORMALIZATION_POINTS)
        special = [
            np.zeros(d),
            np.ones(d),
            np.full(d, 0.5),
            self._reference_unit,
        ]
        for belief in self._beliefs:
            if belief["token"] == "anchors":
                for anchor in belief["points"]:
                    special.append(
                        np.asarray(
                            [
                                dim.to_unit_scalar(anchor["coords"][dim.name])
                                for dim in self.dimensions
                            ]
                        )
                    )
        # Belief-informed extremum candidates. A joint product bump in d dimensions
        # occupies ~width^d of the unit cube, so the Halton sample almost surely misses
        # its peak. Without these candidates the normalization maximum would be the
        # off-mode plateau, and pi() would clip a saturated, gradient-free disk around
        # the mode and lose the belief's location content.
        base = np.full(d, 0.5)
        combined = base.copy()

        def _unit_of(name: str, value: Any) -> float:
            return float(self.dimensions[self._index[name]].to_unit_scalar(value))

        for belief in self._beliefs:
            token = belief["token"]
            point = base.copy()
            if token in {"joint_region", "ridge"} and isinstance(
                belief.get("mode"), Mapping
            ):
                for name, value in belief["mode"].items():
                    point[self._index[name]] = _unit_of(name, value)
                    combined[self._index[name]] = _unit_of(name, value)
            elif "dim" in belief and "mode" in belief:
                point[self._index[belief["dim"]]] = _unit_of(
                    belief["dim"], belief["mode"]
                )
                combined[self._index[belief["dim"]]] = point[
                    self._index[belief["dim"]]
                ]
            elif token == "avoid" and isinstance(belief.get("region"), Mapping):
                for name, interval in belief["region"].items():
                    point[self._index[name]] = 0.5 * (
                        _unit_of(name, interval[0]) + _unit_of(name, interval[1])
                    )
            else:
                continue
            special.append(point)
        # A ridge peaks along its line through ``through`` (default: the centre, already
        # in R); that point is a declared extremum candidate too.
        for belief in self._beliefs:
            if belief["token"] == "ridge" and isinstance(belief.get("through"), Mapping):
                point = base.copy()
                for name, value in belief["through"].items():
                    point[self._index[name]] = _unit_of(name, value)
                special.append(point)
        special.append(combined)
        points = np.vstack([points, *special])
        raw = self._raw_score(points)
        return float(np.min(raw)), float(np.max(raw))

    def _apply_anchors(
        self, unit: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        result = values.copy()
        for belief in self._beliefs:
            if belief["token"] != "anchors":
                continue
            bandwidth = float(belief.get("bandwidth", DEFAULT_ANCHOR_BANDWIDTH))
            anchor_points = np.asarray(
                [
                    [
                        dim.to_unit_scalar(point["coords"][dim.name])
                        for dim in self.dimensions
                    ]
                    for point in belief["points"]
                ]
            )
            anchor_values = np.asarray(
                [float(point["value"]) for point in belief["points"]]
            )
            categorical = [
                i for i, dim in enumerate(self.dimensions) if dim.type == "categorical"
            ]
            if not categorical:
                distances = np.linalg.norm(
                    unit[:, None, :] - anchor_points[None, :, :], axis=2
                ) / math.sqrt(len(self.dimensions))
            else:
                # Unordered levels have no geometry: an anchor acts only on its own
                # categorical stratum, with distance over the geometric dimensions.
                geometric = [
                    i for i in range(len(self.dimensions)) if i not in categorical
                ]
                if geometric:
                    distances = np.linalg.norm(
                        unit[:, None, geometric] - anchor_points[None, :, geometric],
                        axis=2,
                    ) / math.sqrt(len(geometric))
                else:
                    distances = np.zeros((len(unit), len(anchor_points)))
                levels = [
                    np.rint(
                        unit[:, i] * max(len(self.dimensions[i].values) - 1, 1)
                    )
                    for i in categorical
                ]
                anchor_levels = [
                    np.rint(
                        anchor_points[:, i] * max(len(self.dimensions[i].values) - 1, 1)
                    )
                    for i in categorical
                ]
                same = np.ones(distances.shape, dtype=bool)
                for lv, av in zip(levels, anchor_levels):
                    same &= lv[:, None] == av[None, :]
                distances = np.where(same, distances, np.inf)

            exact = distances <= 1e-12
            for row in np.flatnonzero(np.any(exact, axis=1)):
                result[row] = float(np.mean(anchor_values[exact[row]]))

            scaled = distances / bandwidth
            kernels = np.where(
                scaled < 1,
                (1 - np.minimum(scaled, 1)) ** 4
                * (1 + 4 * np.minimum(scaled, 1)),
                0.0,
            )
            totals = kernels.sum(axis=1)
            affected = (totals > 0) & ~np.any(exact, axis=1)
            if np.any(affected):
                targets = (
                    kernels[affected] @ anchor_values / totals[affected]
                )
                blend = np.minimum(totals[affected], 1.0)
                result[affected] = (
                    (1 - blend) * result[affected] + blend * targets
                )
        return np.clip(result, 0.0, 1.0)


def _compile_alpha(record: Mapping[str, Any]) -> float:
    overall = OVERALL_TRUST[record["trust"]["overall_confidence"]]
    weighted_trust = 0.0
    total_weight = 0.0
    for belief in record["beliefs"]:
        token = belief["token"]
        if token in SHAPING_TOKENS:
            weight = float(belief["weight"])
        elif token == "anchors" and "provenance" in belief:
            weight = 1.0
        else:
            continue
        weighted_trust += weight * PROVENANCE_TRUST[belief["provenance"]]
        total_weight += weight
    provenance = weighted_trust / total_weight if total_weight else 1.0
    return float(np.clip(overall * provenance, 0.1, 0.9))


def document_peak_points(record: Mapping[str, Any]) -> list[dict[str, float]]:
    """The document's own stated peak locations, in native units (one dict per point, every dimension set):
    each anchor point, each joint_region / ridge mode (other dimensions at the range midpoint), and the point made of
    every dimension's strongest peak-bump mode (midpoint where a dimension has none). Used as starting points for the
    prior-maximizer search of the continuous BO path, so a multimodal prior (e.g. many periodic tokens) still offers
    the location the document itself names as best."""
    dims = record["schema"]["dimensions"]
    names = [d["name"] for d in dims]
    mid = {d["name"]: (float(sorted(float(v) for v in d["values"])[len(d["values"]) // 2]) if "values" in d
                       else 0.5 * (float(d["range"][0]) + float(d["range"][1]))) for d in dims}  # ordinal: an allowed value
    points: list[dict[str, float]] = []
    combined, strongest = dict(mid), {}
    beliefs = list(record["beliefs"])
    beliefs += [b["then"] for b in record["beliefs"] if b["token"] == "conditional" and isinstance(b.get("then"), Mapping)]
    for b in beliefs:
        token = b.get("token")
        if token == "anchors":
            for p in b["points"]:
                points.append({n: float(p["coords"].get(n, mid[n])) for n in names})
        elif token in ("joint_region", "ridge") and isinstance(b.get("mode") or b.get("through"), Mapping):
            m = b.get("mode") or b.get("through")
            points.append({n: float(m.get(n, mid[n])) for n in names})
        elif token == "bump" and b.get("sign", "peak") == "peak" and "dim" in b and "mode" in b:
            w = float(b.get("weight", 1.0))
            if w >= strongest.get(b["dim"], -1.0):
                combined[b["dim"]], strongest[b["dim"]] = float(b["mode"]), w
    if strongest:
        points.append(combined)
    # The document's own maximum when it states no point (e.g. only trend/periodic tokens): coordinate ascent on the
    # UNCLIPPED raw score (pi itself is clipped at 1 against a reference maximum that can miss the true peak, so it is
    # flat around the top). 3 sweeps, 801-point axis grids (geometric on log dimensions, the values on ordinal ones).
    try:
        compiled = compile_prior(record)
    except Exception:  # noqa: BLE001 -- an invalid document has no peaks to offer
        return points
    axes = []
    for d in dims:
        if "values" in d:
            axes.append(np.asarray([float(v) for v in d["values"]]))
        elif d.get("scale") == "log" and float(d["range"][0]) > 0:
            axes.append(np.geomspace(float(d["range"][0]), float(d["range"][1]), 801))
        else:
            axes.append(np.linspace(float(d["range"][0]), float(d["range"][1]), 801))
    x = np.asarray([combined[n] for n in names], dtype=float)
    for _ in range(3):
        for i, grid in enumerate(axes):
            X = np.tile(x, (len(grid), 1))
            X[:, i] = grid
            x[i] = grid[int(np.argmax(compiled._raw_score(compiled._encode(X)[0])))]
    points.append({n: float(x[i]) for i, n in enumerate(names)})
    return points


def compile_prior(record: Mapping[str, Any]) -> CompiledPrior:
    """Validate and compile a PSL document into a deterministic joint prior."""

    validate_record(record)
    dimensions = _dimensions(record)
    return CompiledPrior(
        record=record,
        dimensions=dimensions,
        alpha=_compile_alpha(record),
    )


@dataclass
class PriorLandscape:
    """A one-dimensional slice through a shared joint ``CompiledPrior``."""

    dimension: int
    parameter_name: str
    spec: DimensionSpec
    compiler: CompiledPrior

    @property
    def alpha(self) -> float:
        return self.compiler.alpha

    @property
    def bounds(self) -> tuple[float, float]:
        if self.spec.type == "continuous":
            assert self.spec.range is not None
            return self.spec.range
        assert self.spec.values is not None
        return 0.0, float(max(0, len(self.spec.values) - 1))

    @property
    def is_discrete(self) -> bool:
        return self.spec.type != "continuous"

    @property
    def categories(self) -> list[Any] | None:
        if self.spec.type == "categorical":
            assert self.spec.values is not None
            return list(self.spec.values)
        return None

    def _points(self, x: np.ndarray | float) -> tuple[np.ndarray, bool]:
        values = np.atleast_1d(np.asarray(x, dtype=object))
        reference = self.compiler.reference_point
        points = []
        for value in values:
            point = dict(reference)
            if self.spec.type == "categorical":
                assert self.spec.values is not None
                if value in self.spec.values:
                    pass
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    index = int(value)
                    if float(value) != index or not 0 <= index < len(self.spec.values):
                        _fail(f"invalid category index {value!r}")
                    value = self.spec.values[index]
            point[self.parameter_name] = value
            points.append(point)
        return np.asarray(points, dtype=object), np.ndim(x) == 0

    def effective_prior(self, x: np.ndarray | float) -> np.ndarray | float:
        points, single = self._points(x)
        values = np.asarray(self.compiler.pi(list(points)), dtype=float)
        return float(values[0]) if single else values

    def landscape(self, n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
        grid = self.spec.plotting_grid(n_points)
        return grid, np.asarray(self.effective_prior(grid), dtype=float)


def build_landscapes(record: Mapping[str, Any]) -> list[PriorLandscape]:
    """Build plotting slices for every dimension in a PSL belief document.

    The search bounds come from ``record['schema']``.
    """

    compiler = compile_prior(record)
    return [
        PriorLandscape(i + 1, dim.name, dim, compiler)
        for i, dim in enumerate(compiler.dimensions)
    ]


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PRIOR_DATASET_ID = "neuromosaic-dataset1-rat"
DEFAULT_PRIOR_TARGET_NAME = "LeftKneeFlexor"
DEFAULT_PRIOR_TARGET_SLUG = "left-knee-flexor"
DEFAULT_PRIOR_ID = "human-expert-map"
EXPERIMENT_PRIORS_DIRECTORY = PROJECT_ROOT / "experiments" / "priors"
DEFAULT_PRIOR_DIRECTORY = (
    EXPERIMENT_PRIORS_DIRECTORY
    / DEFAULT_PRIOR_DATASET_ID
    / DEFAULT_PRIOR_TARGET_SLUG
    / DEFAULT_PRIOR_ID
)
PROMPT_CARD_TEMPLATE_PATH = (
    PROJECT_ROOT / "src" / "prior" / "expert_knowledge" / "prompt_card_template.md"
)
# HAPBO_VOCABULARY_CARD (project-relative) selects an alternative card for new elicitations; existing priors
# keep validating against the card path recorded in their own metadata.
VOCABULARY_CARD_PATH = PROJECT_ROOT / os.environ.get(
    "HAPBO_VOCABULARY_CARD", "src/prior/expert_knowledge/vocabulary_card.md")
PRIOR_RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "prior"
    / DEFAULT_PRIOR_DATASET_ID
    / DEFAULT_PRIOR_TARGET_SLUG
    / DEFAULT_PRIOR_ID
)
RECORD_PATH = DEFAULT_PRIOR_DIRECTORY / "prior.json"
PLOT_PATH = PRIOR_RESULTS_DIRECTORY / "prior_landscape.png"
COMBINED_PLOT_PATH = PRIOR_RESULTS_DIRECTORY / "combined_prior_landscape.png"
SUMMARY_PATH = PRIOR_RESULTS_DIRECTORY / "prior_summary.json"
DIAGNOSTIC_SAMPLES_PATH = PRIOR_RESULTS_DIRECTORY / "prior_diagnostic_samples.npz"
ARTIFACT_FORMAT_VERSION = "hapbo-prior-artifact-v1"
N_GRID_POINTS = 200
N_COMBINED_PLOT_SAMPLES = 600


def _validate_identifier(value: str, label: str) -> str:
    """Validate one filesystem-safe dataset or prior identifier."""

    if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value) is None:
        _fail(f"{label} must use lowercase letters, numbers, and single hyphens")
    return value


def target_path_component(value: str) -> str:
    """Convert an exact optimization target name into a stable path component."""

    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", value)
    component = re.sub(r"[^a-z0-9]+", "-", separated.lower()).strip("-")
    if not component:
        _fail(f"cannot construct a target path component from {value!r}")
    return component


def resolve_prior_path(
    dataset_id: str,
    *,
    target_name: str | None = None,
    prior_id: str | None = None,
    explicit_path: str | Path | None = None,
    experiments_directory: str | Path = EXPERIMENT_PRIORS_DIRECTORY,
) -> Path:
    """Resolve one versioned prior using the repository's dataset layout.

    Target-aware datasets use ``<dataset>/<target>/<prior-id>/prior.json``;
    benchmark instances whose dataset ID already identifies the target use
    ``<dataset>/<prior-id>/prior.json``. An explicit path always takes
    precedence. Without ``prior_id``, exactly one prior version must exist.
    """

    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            _fail(f"explicit prior file does not exist: {path}")
        return path

    dataset_id = _validate_identifier(dataset_id, "dataset_id")
    prior_root = Path(experiments_directory).expanduser().resolve() / dataset_id
    if target_name is not None:
        prior_root /= target_path_component(target_name)

    if prior_id is not None:
        prior_id = _validate_identifier(prior_id, "prior_id")
        path = prior_root / prior_id / "prior.json"
        if not path.is_file():
            _fail(
                f"no HapBO prior exists at {path}; create and finalize it, "
                "choose another --prior-id, or pass --dataset-prior-file"
            )
        return path

    candidates = sorted(prior_root.glob("*/prior.json"))
    if not candidates:
        expected = prior_root / "<prior-id>" / "prior.json"
        target_context = (
            f" and target {target_name!r}" if target_name is not None else ""
        )
        _fail(
            f"no HapBO prior found for dataset {dataset_id!r}{target_context}; "
            f"expected {expected}. Create and finalize a prior, or pass "
            "--dataset-prior-file"
        )
    if len(candidates) > 1:
        versions = ", ".join(path.parent.name for path in candidates)
        _fail(
            f"multiple HapBO priors found under {prior_root}: {versions}; "
            "select one with --prior-id or --dataset-prior-file"
        )
    return candidates[0].resolve()


def initialize_prior_experiment(
    dataset_id: str,
    prior_id: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    subject_scope: str = "dataset-level expert knowledge",
    target: str = "unspecified optimization target",
    target_name: str | None = None,
    experiments_directory: str | Path = EXPERIMENT_PRIORS_DIRECTORY,
    card_human_statement: bool = False,
) -> Path:
    """Create a new draft directory for a manually generated expert prior.

    ``card_human_statement`` declares that the task card will carry a human (expert)
    statement (metadata ``card.human_statement``); only then may beliefs be tagged
    ``user_stated`` / ``user_confirmed``.
    """

    dataset_id = _validate_identifier(dataset_id, "dataset_id")
    prior_id = _validate_identifier(prior_id, "prior_id")
    if dataset_id.startswith("neuromosaic-") and target_name is None:
        _fail(
            "NeuroMOSAIC priors require an exact target_name so they cannot be "
            "reused for the wrong muscle"
        )
    prior_root = Path(experiments_directory).expanduser().resolve() / dataset_id
    target_slug = None
    if target_name is not None:
        target_name = target_name.strip()
        if not target_name:
            _fail("target_name must not be empty")
        target_slug = target_path_component(target_name)
        prior_root /= target_slug
    prior_directory = prior_root / prior_id
    if prior_directory.exists():
        _fail(
            f"{prior_directory} already exists; choose a new prior_id instead "
            "of overwriting a prior version"
        )

    prior_directory.mkdir(parents=True)
    shutil.copy2(PROMPT_CARD_TEMPLATE_PATH, prior_directory / "prompt_card.md")
    metadata = {
        "artifact_format": (
            "llm-bo-prior-metadata-v2"
            if target_name is not None
            else "llm-bo-prior-metadata-v1"
        ),
        "status": "draft",
        "prior_id": prior_id,
        "dataset_id": dataset_id,
        "scope": {
            "subjects": subject_scope,
            "target": target,
            **(
                {"target_name": target_name, "target_slug": target_slug}
                if target_name is not None
                else {}
            ),
            "uses_observed_objective_values": False,
        },
        "card": {"human_statement": bool(card_human_statement)},
        "generation": {
            "provider": provider,
            "model": model,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "generated_at": None,
            "raw_llm_response_available": False,
            "notes": "",
        },
        "files": {
            "prompt_card.md": {"sha256": None},
            "prior.json": {"sha256": None},
        },
        "vocabulary": {
            "path": VOCABULARY_CARD_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": None,
        },
    }
    metadata_path = prior_directory / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return prior_directory


def finalize_prior_experiment(
    prior_directory: str | Path,
    *,
    output_directory: str | Path | None = None,
    generate_landscape: bool = True,
) -> CompiledPrior:
    """Validate, hash, and finalize one manually generated prior version."""

    prior_directory = Path(prior_directory).expanduser().resolve()
    metadata_path = prior_directory / "metadata.json"
    prompt_path = prior_directory / "prompt_card.md"
    prior_path = prior_directory / "prior.json"
    for required_path in (metadata_path, prompt_path, prior_path):
        if not required_path.is_file():
            _fail(f"cannot finalize prior: missing {required_path}")

    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, Mapping):
        _fail(f"{metadata_path}: top level must be an object")
    artifact_format = metadata.get("artifact_format")
    if artifact_format not in {
        "llm-bo-prior-metadata-v1",
        "llm-bo-prior-metadata-v2",
    }:
        _fail(f"{metadata_path}: unsupported or missing artifact_format")
    if metadata.get("status") == "finalized":
        _fail(
            f"{prior_directory} is already finalized; create a new prior_id "
            "for any changed prompt or prior"
        )
    if metadata.get("status") != "draft":
        _fail(f"{metadata_path}: status must be 'draft' before finalization")
    if metadata.get("prior_id") != prior_directory.name:
        _fail(f"{metadata_path}: prior_id must match its directory name")
    if artifact_format == "llm-bo-prior-metadata-v2":
        scope = metadata.get("scope")
        if not isinstance(scope, Mapping):
            _fail(f"{metadata_path}: scope must be an object")
        target_name = scope.get("target_name")
        target_slug = scope.get("target_slug")
        if not isinstance(target_name, str) or not target_name.strip():
            _fail(f"{metadata_path}: scope.target_name must be a non-empty string")
        expected_slug = target_path_component(target_name)
        if target_slug != expected_slug:
            _fail(
                f"{metadata_path}: scope.target_slug must be {expected_slug!r} "
                "for scope.target_name"
            )
        if prior_directory.parent.name != target_slug:
            _fail(f"{metadata_path}: target_slug must match its directory name")
        dataset_directory = prior_directory.parent.parent
    else:
        dataset_directory = prior_directory.parent
    if metadata.get("dataset_id") != dataset_directory.name:
        _fail(f"{metadata_path}: dataset_id must match its directory name")

    with prior_path.open(encoding="utf-8") as handle:
        record = json.load(handle)
    validate_record(
        record,
        source=str(prior_path),
        card_human_statement=declares_human_statement(metadata),
    )
    compiler = compile_prior(record)

    metadata["status"] = "finalized"
    metadata["finalized_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["files"] = {
        "prompt_card.md": {
            "sha256": _sha256_text_artifact(prompt_path),
        },
        "prior.json": {
            "sha256": _sha256_text_artifact(prior_path),
        },
    }
    metadata["vocabulary"] = {
        "path": VOCABULARY_CARD_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": _sha256_text_artifact(VOCABULARY_CARD_PATH),
    }
    generation = metadata.setdefault("generation", {})
    if isinstance(generation, dict):
        generation["raw_llm_response_available"] = (
            prior_directory / "raw_llm_response.txt"
        ).is_file()
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    # Validate the completed provenance record before generating derivatives.
    validate_prior_metadata(prior_path)
    if generate_landscape:
        return run_landscape_generation(
            record_path=prior_path,
            output_directory=output_directory,
        )
    return compiler


def results_directory_for_prior(record_path: str | Path) -> Path:
    """Derive a stable diagnostic directory from a versioned prior path."""

    record_path = Path(record_path).expanduser().resolve()
    try:
        relative = record_path.relative_to(EXPERIMENT_PRIORS_DIRECTORY.resolve())
    except ValueError:
        # External priors remain separated by their containing directory. The
        # caller can always override this fallback with --prior-output-directory.
        prior_id = record_path.parent.name or record_path.stem
        return PROJECT_ROOT / "results" / "prior" / "external" / prior_id

    if relative.name != "prior.json" or len(relative.parts) not in {3, 4}:
        _fail(
            "versioned priors must use "
            "experiments/priors/<dataset-id>/<target-slug>/<prior-id>/prior.json "
            "(or the two-level <dataset-id>/<prior-id>/prior.json layout)"
        )
    if len(relative.parts) == 4:
        dataset_id, target_slug, prior_id, _ = relative.parts
        return (
            PROJECT_ROOT / "results" / "prior" / dataset_id / target_slug / prior_id
        )
    dataset_id, prior_id, _ = relative.parts
    return PROJECT_ROOT / "results" / "prior" / dataset_id / prior_id


def plot_landscapes(
    landscapes: Sequence[PriorLandscape],
    n_points: int = N_GRID_POINTS,
    out_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot one native-unit cross-section per schema dimension."""

    import matplotlib.pyplot as plt

    n = len(landscapes)
    ncols = min(n, 3)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax, landscape in zip(axes.flat, landscapes):
        grid, density = landscape.landscape(n_points)
        if landscape.is_discrete:
            positions = np.arange(len(grid))
            ax.bar(positions, density, width=0.6)
            ax.set_xticks(positions)
            labels = (
                landscape.categories
                if landscape.categories is not None
                else [str(value) for value in grid]
            )
            ax.set_xticklabels(labels, rotation=30)
        else:
            ax.plot(grid, density)
            if landscape.spec.scale == "log":
                ax.set_xscale("log")
        ax.set_title(
            f"dim {landscape.dimension}: {landscape.parameter_name}\n"
            f"joint slice, alpha={landscape.alpha:.3f}"
        )
        ax.set_ylabel("normalized expert landscape π")
        ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"saved plot to {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _short_native_value(value: Any) -> str:
    """Compact native-coordinate label for the combined-prior figure."""

    if isinstance(value, (float, np.floating)):
        value = float(value)
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e4):
            return f"{value:.1e}"
        return f"{value:g}"
    return str(value)


def _combined_axis_label(dimension: DimensionSpec) -> str:
    if dimension.type == "continuous":
        assert dimension.range is not None
        low, high = dimension.range
        scale = " · log" if dimension.scale == "log" else ""
        extent = f"{_short_native_value(low)} → {_short_native_value(high)}{scale}"
    else:
        assert dimension.values is not None
        labels = [_short_native_value(value) for value in dimension.values]
        extent = " | ".join(labels)
    return f"{dimension.name}\n{extent}"


def _two_dimensional_prior_grid(
    compiler: CompiledPrior,
    n_points: int = N_GRID_POINTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate joint ``pi(x1, x2)`` on a native-coordinate plotting grid."""

    if len(compiler.dimensions) != 2:
        _fail("two-dimensional prior plotting requires exactly two dimensions")
    if n_points < 2:
        _fail("two-dimensional prior plotting requires at least two grid points")

    axis_values: list[np.ndarray] = []
    for dimension in compiler.dimensions:
        if dimension.type == "continuous":
            values = dimension.plotting_grid(n_points)
        else:
            assert dimension.values is not None
            values = np.asarray(dimension.values, dtype=object)
        axis_values.append(values)

    x_values, y_values = axis_values
    x_grid, y_grid = np.meshgrid(x_values, y_values, indexing="xy")
    native_points = np.column_stack(
        [
            x_grid.reshape(-1),
            y_grid.reshape(-1),
        ]
    )
    pi_values = np.asarray(compiler.pi(native_points), dtype=float).reshape(
        len(y_values),
        len(x_values),
    )
    return x_values, y_values, pi_values


def _two_dimensional_prior_display_grid(
    compiler: CompiledPrior,
    n_points: int = N_GRID_POINTS,
) -> tuple[DimensionSpec, DimensionSpec, np.ndarray, np.ndarray, np.ndarray]:
    """Orient a joint prior grid like the corresponding BO response map.

    NeuroMOSAIC anatomical priors are stored in ``(mediolateral,
    rostrocaudal)`` schema order.  The offline BO figures display
    ``channel_x=rostrocaudal`` horizontally and
    ``channel_y=mediolateral`` vertically, so the visualization must transpose
    that one supported alias pair.  The compiled prior and its schema order are
    left unchanged.
    """

    first_dimension, second_dimension = compiler.dimensions
    first_values, second_values, pi_values = _two_dimensional_prior_grid(
        compiler,
        n_points=n_points,
    )
    if (
        first_dimension.name,
        second_dimension.name,
    ) == (
        "electrode_mediolateral_index",
        "electrode_rostrocaudal_index",
    ):
        return (
            second_dimension,
            first_dimension,
            second_values,
            first_values,
            pi_values.T,
        )
    return (
        first_dimension,
        second_dimension,
        first_values,
        second_values,
        pi_values,
    )


def _heatmap_ticks(
    values: np.ndarray,
    max_ticks: int = 9,
) -> tuple[np.ndarray, list[str]]:
    """Return readable index positions and native-value labels for a heatmap."""

    if len(values) <= max_ticks:
        positions = np.arange(len(values), dtype=int)
    else:
        positions = np.unique(
            np.rint(np.linspace(0, len(values) - 1, max_ticks)).astype(int)
        )
    return positions, [_short_native_value(values[index]) for index in positions]


def _prior_axis_label(dimension: DimensionSpec) -> str:
    """Describe one native-coordinate axis in the joint 2D prior plot."""

    if dimension.type == "continuous" and dimension.scale == "log":
        return f"{dimension.name} (continuous, log scale)"
    return f"{dimension.name} ({dimension.type})"


def plot_two_dimensional_prior(
    compiler: CompiledPrior,
    n_points: int = N_GRID_POINTS,
    out_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot the compiled joint prior for exactly two dimensions.

    Discrete or mixed spaces use a native-labelled heatmap. Two continuous
    dimensions use a filled contour plot. In either representation, color is
    the fully compiled joint ``pi(x1, x2)`` rather than an independent slice.
    """

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    (
        x_dimension,
        y_dimension,
        x_values,
        y_values,
        pi_values,
    ) = _two_dimensional_prior_display_grid(
        compiler,
        n_points=n_points,
    )
    both_continuous = (
        x_dimension.type == "continuous" and y_dimension.type == "continuous"
    )

    fig, ax = plt.subplots(figsize=(11, 7.5))
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0.0, vmax=1.0)
    best_row, best_column = np.unravel_index(
        int(np.argmax(pi_values)),
        pi_values.shape,
    )
    best_pi = float(pi_values[best_row, best_column])

    if both_continuous:
        x_numeric = np.asarray(x_values, dtype=float)
        y_numeric = np.asarray(y_values, dtype=float)
        image = ax.contourf(
            x_numeric,
            y_numeric,
            pi_values,
            levels=np.linspace(0.0, 1.0, 21),
            cmap=cmap,
            norm=norm,
        )
        ax.contour(
            x_numeric,
            y_numeric,
            pi_values,
            levels=np.linspace(0.1, 0.9, 9),
            colors="white",
            linewidths=0.45,
            alpha=0.55,
        )
        ax.scatter(
            [x_numeric[best_column]],
            [y_numeric[best_row]],
            marker="*",
            s=180,
            color="#ef476f",
            edgecolor="white",
            linewidth=0.9,
            label=f"highest grid π = {best_pi:.3f}",
            zorder=5,
        )
        if x_dimension.scale == "log":
            ax.set_xscale("log")
        if y_dimension.scale == "log":
            ax.set_yscale("log")
    else:
        image = ax.imshow(
            pi_values,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
        )
        x_positions, x_labels = _heatmap_ticks(x_values)
        y_positions, y_labels = _heatmap_ticks(y_values)
        ax.set_xticks(x_positions, labels=x_labels)
        ax.set_yticks(y_positions, labels=y_labels)
        ax.add_patch(
            Rectangle(
                (best_column - 0.5, best_row - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="#ef476f",
                linewidth=3.0,
                label=f"highest grid π = {best_pi:.3f}",
            )
        )
        # Exact values remain readable for small discrete candidate grids such
        # as NeuroMOSAIC's 8 x 4 electrode array.
        if (
            x_dimension.type != "continuous"
            and y_dimension.type != "continuous"
            and pi_values.size <= 100
        ):
            for row, column in np.ndindex(pi_values.shape):
                value = float(pi_values[row, column])
                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="black" if value >= 0.65 else "white",
                    fontsize=8.5,
                )

    ax.set_xlabel(_prior_axis_label(x_dimension))
    ax.set_ylabel(_prior_axis_label(y_dimension))
    ax.set_title(
        "Compiled joint expert prior π(x₁, x₂)\n"
        f"color = normalized π; trust α = {compiler.alpha:.3f}",
        pad=14,
    )
    ax.legend(loc="upper right", frameon=True)
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("fully combined normalized prior π(x₁, x₂)")
    fig.text(
        0.5,
        0.015,
        "Every color represents a complete two-parameter configuration. "
        f"The raw-acquisition alpha-pi multiplier spans "
        f"[{1.0 - compiler.alpha:.3f}, 1.000].",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.98))

    if out_path is not None:
        out_path = Path(out_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_path,
            dpi=150,
            bbox_inches="tight",
            pil_kwargs={"optimize": True},
        )
        print(f"saved two-dimensional prior plot to {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _combined_plot_points(
    compiler: CompiledPrior, n_samples: int
) -> np.ndarray:
    """Deterministic full configurations in the compiler's unit geometry."""

    if n_samples < 2:
        _fail("combined-prior plot requires at least two samples")
    n_dimensions = len(compiler.dimensions)
    sampler = stats.qmc.Halton(d=n_dimensions, scramble=False)
    points = sampler.random(n_samples)

    # Ordinal and categorical dimensions are finite sets, not continuous
    # coordinates. Snap the Halton sequence to their valid unit locations.
    for column, dimension in enumerate(compiler.dimensions):
        if dimension.type == "continuous":
            continue
        assert dimension.values is not None
        n_values = len(dimension.values)
        indices = np.minimum((points[:, column] * n_values).astype(int), n_values - 1)
        points[:, column] = (
            0.0 if n_values == 1 else indices / float(n_values - 1)
        )

    special_points = [compiler._reference_unit]
    for belief in compiler._beliefs:
        if belief["token"] == "anchors":
            for anchor in belief["points"]:
                special_points.append(
                    np.asarray(
                        [
                            dimension.to_unit_scalar(
                                anchor["coords"][dimension.name]
                            )
                            for dimension in compiler.dimensions
                        ]
                    )
                )
    return np.vstack([points, *special_points])


def _json_scalar(value: Any) -> Any:
    """Convert one NumPy/native scalar into a JSON-compatible scalar."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _native_values_as_json(
    compiler: CompiledPrior,
    unit_points: np.ndarray,
) -> np.ndarray:
    """Return lossless JSON scalars for mixed-type native coordinates.

    The two-dimensional Unicode array can be loaded from NPZ without pickle.
    Calling ``json.loads`` on each cell restores numbers, strings, and other
    JSON scalar category values with their original types.
    """
    columns = [
        dimension.unit_to_native(unit_points[:, column])
        for column, dimension in enumerate(compiler.dimensions)
    ]
    rows = [
        [json.dumps(_json_scalar(columns[column][row])) for column in range(len(columns))]
        for row in range(len(unit_points))
    ]
    return np.asarray(rows, dtype=str)


def save_prior_artifacts(
    record: Mapping[str, Any],
    compiler: CompiledPrior,
    source_path: str | Path = RECORD_PATH,
    summary_path: str | Path = SUMMARY_PATH,
    samples_path: str | Path = DIAGNOSTIC_SAMPLES_PATH,
    n_samples: int = N_COMBINED_PLOT_SAMPLES,
) -> tuple[Path, Path]:
    """Save reproducibility metadata and deterministic diagnostic samples.

    These artifacts document one compilation but do not replace ``prior.json``.
    A future optimizer should recompile the source document and evaluate
    ``pi(X)`` on its exact candidate set.
    """
    source_path = Path(source_path).expanduser().resolve()
    summary_path = Path(summary_path).expanduser().resolve()
    samples_path = Path(samples_path).expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path.parent.mkdir(parents=True, exist_ok=True)

    unit_points = _combined_plot_points(compiler, n_samples)
    pi_values = np.asarray(compiler._pi_unit(unit_points), dtype=float)
    multiplier_values = (1.0 - compiler.alpha) + compiler.alpha * pi_values
    native_values_json = _native_values_as_json(compiler, unit_points)
    dimension_names = np.asarray(compiler.parameter_names, dtype=str)

    np.savez_compressed(
        samples_path,
        unit_points=unit_points,
        native_values_json=native_values_json,
        dimension_names=dimension_names,
        pi=pi_values,
        multiplier=multiplier_values,
    )

    try:
        source_relative = source_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        source_relative = str(source_path)
    try:
        samples_relative = samples_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        samples_relative = str(samples_path)
    prior_metadata = validate_prior_metadata(source_path)

    summary = {
        "artifact_format": ARTIFACT_FORMAT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": source_relative,
            "sha256": _sha256_text_artifact(source_path),
        },
        "schema": record["schema"],
        "dimension_order": compiler.parameter_names,
        "alpha": compiler.alpha,
        "reference_point": {
            name: _json_scalar(value)
            for name, value in compiler.reference_point.items()
        },
        "normalization": {
            "raw_score_min": compiler._normalization_min,
            "raw_score_max": compiler._normalization_max,
            "pi_range": [0.0, 1.0],
        },
        "diagnostic_samples": {
            "path": samples_relative,
            "sampling": "deterministic unscrambled Halton plus reference and anchor points",
            "requested_halton_samples": n_samples,
            "saved_points": int(len(unit_points)),
            "fields": {
                "unit_points": "float matrix in [0, 1], columns follow dimension_order",
                "native_values_json": (
                    "JSON scalar strings aligned with unit_points; decode each cell "
                    "with json.loads"
                ),
                "dimension_names": "column names in schema order",
                "pi": "raw normalized expert prior pi(x)",
                "multiplier": "diagnostic (1 - alpha) + alpha * pi(x)",
            },
            "observed_pi_min": float(np.min(pi_values)),
            "observed_pi_max": float(np.max(pi_values)),
        },
        "reconstruction": {
            "authoritative_artifact": source_relative,
            "instruction": (
                "Load and compile the source prior JSON, then evaluate pi(X) on "
                "the optimizer's exact native-unit candidates. Do not use the "
                "diagnostic samples as a candidate-specific prior vector."
            ),
        },
    }
    if prior_metadata is not None:
        metadata_path = source_path.parent / "metadata.json"
        try:
            metadata_relative = metadata_path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            metadata_relative = str(metadata_path)
        summary["prior_identity"] = {
            "dataset_id": prior_metadata["dataset_id"],
            "prior_id": prior_metadata["prior_id"],
            "metadata_path": metadata_relative,
            "metadata_sha256": _sha256_text_artifact(metadata_path),
        }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"saved prior summary to {summary_path}")
    print(f"saved diagnostic samples to {samples_path}")
    return summary_path, samples_path


def plot_combined_prior(
    compiler: CompiledPrior,
    n_samples: int = N_COMBINED_PLOT_SAMPLES,
    out_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot the fully combined expert prior with a dimension-aware view.

    Exactly two dimensions use a joint native-coordinate heatmap or contour.
    Other dimensionalities use parallel coordinates, where deterministic full
    configurations are colored by the *joint* ``pi(x)`` after the exact PSL
    compile order: additive shapes, penalties, multiplicative gates,
    normalization, then anchors.
    """

    if len(compiler.dimensions) == 2:
        plot_two_dimensional_prior(
            compiler,
            out_path=out_path,
            show=show,
        )
        return

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    unit_points = _combined_plot_points(compiler, n_samples)
    combined_pi = compiler._pi_unit(unit_points)
    order = np.argsort(combined_pi)
    unit_points = unit_points[order]
    combined_pi = combined_pi[order]

    n_dimensions = len(compiler.dimensions)
    x_positions = np.arange(n_dimensions)
    segments = np.stack(
        [
            np.broadcast_to(x_positions, unit_points.shape),
            unit_points,
        ],
        axis=2,
    )

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0.0, vmax=1.0)
    colors = cmap(norm(combined_pi))
    # Low-prior configurations provide context without obscuring the high-prior
    # paths, which are deliberately drawn last and more opaquely.
    colors[:, 3] = 0.015 + 0.50 * combined_pi**2

    fig, ax = plt.subplots(figsize=(18, 9))
    collection = LineCollection(
        segments,
        colors=colors,
        linewidths=0.30 + 0.70 * combined_pi**2,
        zorder=2,
    )
    ax.add_collection(collection)

    best_index = int(np.argmax(combined_pi))
    ax.plot(
        x_positions,
        unit_points[best_index],
        color="#ef476f",
        linewidth=2.4,
        marker="o",
        markersize=4,
        label=f"highest sampled joint π = {combined_pi[best_index]:.3f}",
        zorder=4,
    )

    for x_position in x_positions:
        ax.axvline(x_position, color="#7f8c8d", linewidth=0.75, alpha=0.55, zorder=1)
    ax.set_xlim(-0.1, n_dimensions - 0.9)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [_combined_axis_label(dimension) for dimension in compiler.dimensions],
        rotation=16,
        ha="right",
    )
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel(
        "normalized coordinate within each dimension\n"
        "(native bounds/categories are shown below each axis)"
    )
    ax.set_title(
        "Combined HapBO expert prior across all dimensions\n"
        f"line color = joint π(x); trust α = {compiler.alpha:.3f}; "
        "pink line = highest deterministic sample",
        pad=18,
    )
    ax.grid(axis="y", color="#d5d8dc", linewidth=0.7, alpha=0.65)
    ax.legend(loc="upper right", frameon=True)

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        pad=0.015,
        fraction=0.025,
    )
    colorbar.set_label("fully combined normalized prior π(x)")
    fig.text(
        0.5,
        0.015,
        "Each path is one complete configuration. Colors use the joint PSL score—not "
        "a product of independent one-dimensional curves. "
        f"The raw-acquisition multiplier spans [{1.0 - compiler.alpha:.3f}, 1.000].",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.02, 0.055, 0.98, 0.98))

    if out_path is not None:
        out_path = Path(out_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_path,
            dpi=140,
            bbox_inches="tight",
            pil_kwargs={"optimize": True},
        )
        print(f"saved combined-prior plot to {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def run_landscape_generation(
    record_path: str | Path = RECORD_PATH,
    output_directory: str | Path | None = None,
) -> CompiledPrior:
    """Compile one prior document and save all diagnostic artifacts.

    This is the reusable application boundary shared by this module's
    standalone command and the repository-level ``main.py`` entry point.  The
    returned compiler remains the authoritative callable joint prior; the
    saved samples and figures are diagnostics only.
    """

    record_path = Path(record_path).expanduser().resolve()
    output_directory = (
        results_directory_for_prior(record_path)
        if output_directory is None
        else Path(output_directory).expanduser().resolve()
    )
    record = load_record(record_path)
    landscapes = build_landscapes(record)
    compiler = landscapes[0].compiler

    schema = record["schema"]
    print(
        f"{schema.get('problem', '(prior)')} (objective={schema['objective']}, "
        f"alpha={compiler.alpha:.3f})"
    )
    print(f"reference point: {compiler.reference_point}")
    for landscape in landscapes:
        grid, density = landscape.landscape(N_GRID_POINTS)
        peak_index = int(np.argmax(density))
        peak = (
            landscape.categories[peak_index]
            if landscape.categories is not None
            else grid[peak_index]
        )
        print(
            f"  dim {landscape.dimension} | {landscape.parameter_name:<24} | "
            f"{landscape.spec.type:<11} | scale={landscape.spec.scale:<6} | "
            f"slice mode={peak}"
        )
    save_prior_artifacts(
        record,
        compiler,
        source_path=record_path,
        summary_path=output_directory / SUMMARY_PATH.name,
        samples_path=output_directory / DIAGNOSTIC_SAMPLES_PATH.name,
    )
    # Both figures are saved on every run in the selected result directory.
    # Keeping plotting non-interactive also makes the generator safe in
    # headless environments.
    plot_landscapes(
        landscapes,
        out_path=output_directory / PLOT_PATH.name,
        show=False,
    )
    plot_combined_prior(
        compiler,
        out_path=output_directory / COMBINED_PLOT_PATH.name,
        show=False,
    )
    return compiler


def main(argv: list[str] | None = None) -> None:
    """Run prior compilation directly with optional versioned input paths."""

    parser = argparse.ArgumentParser(description="Compile and plot one expert prior.")
    parser.add_argument(
        "--prior-file",
        type=Path,
        default=RECORD_PATH,
        help=f"Validated prior JSON (default: {RECORD_PATH}).",
    )
    parser.add_argument(
        "--prior-output-directory",
        type=Path,
        default=None,
        help="Optional artifact directory; otherwise derive it from the prior path.",
    )
    args = parser.parse_args(argv)
    run_landscape_generation(
        record_path=args.prior_file,
        output_directory=args.prior_output_directory,
    )


if __name__ == "__main__":
    main()
