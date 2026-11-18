"""Live adapter and search-space registry for the SOLAR C++ simulator."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOLAR_DIRECTORY = PROJECT_ROOT / "benchmark" / "SOLAR"
SOLAR_SOURCE_DIRECTORY = SOLAR_DIRECTORY / "source"
SOLAR_EXECUTABLE = SOLAR_SOURCE_DIRECTORY / "bin" / "solar"
DATASET_ID = "solar"
SOLAR_REPOSITORY = "https://github.com/bbopt/solar.git"
SOLAR_COMMIT = "dda8fd7f98d4392aa07b3e3056888624a00712a0"
HIDDEN_FAILURE_VALUE = 1e20

VariableKind = Literal["continuous", "integer", "categorical"]


@dataclass(frozen=True)
class SolarVariable:
    """One native simulator input and the finite bound used for pool sampling."""

    name: str
    kind: VariableKind
    lower: float
    upper: float | None
    sampling_upper: float
    initial: float | int
    description: str
    units: str = ""

    @property
    def prior_type(self) -> str:
        # The PSL compiler supports continuous and categorical coordinates.
        # Integer simulator inputs use bounded continuous priors and are rounded
        # after candidate generation, avoiding enormous ordinal value arrays.
        return "categorical" if self.kind == "categorical" else "continuous"


@dataclass(frozen=True)
class SolarInstance:
    """Static definition of one selected official single-objective instance."""

    number: int
    name: str
    objective: str
    variables: tuple[SolarVariable, ...]
    constraints: tuple[str, ...]
    stochastic: bool
    fidelity_fixed: bool = False

    @property
    def n_dimensions(self) -> int:
        return len(self.variables)

    @property
    def n_constraints(self) -> int:
        return len(self.constraints)

    @property
    def dataset_id(self) -> str:
        return f"solar-{self.name.lower().replace('_', '-')}"


@dataclass(frozen=True)
class SolarEvaluation:
    """One live simulator response in the native SOLAR minimization convention."""

    objective: float
    constraints: np.ndarray
    feasible: bool
    hidden_failure: bool
    raw_output: str


def _v(
    name: str,
    kind: VariableKind,
    lower: float,
    upper: float | None,
    initial: float | int,
    description: str,
    units: str = "",
    *,
    sampling_upper: float | None = None,
) -> SolarVariable:
    if upper is None and sampling_upper is None:
        raise ValueError(f"unbounded variable {name} needs a finite sampling cap")
    return SolarVariable(
        name=name,
        kind=kind,
        lower=float(lower),
        upper=None if upper is None else float(upper),
        sampling_upper=float(upper if sampling_upper is None else sampling_upper),
        initial=initial,
        description=description,
        units=units,
    )


FIELD_CONSTRAINTS = (
    "Field surface area is below the instance limit",
    "Compliance to the demand requirement",
    "Tower height is at least twice the heliostat length",
    "Minimum tower distance does not exceed maximum tower distance",
    "The requested heliostats fit in the field",
)
RECEIVER_CONSTRAINTS = (
    "Receiver-tube pressure does not exceed yield pressure",
    "Molten-salt melting point is below the hot-storage minimum temperature",
    "Molten-salt melting point is below the cold-storage minimum temperature",
    "Molten-salt melting point is below the steam-generator outlet temperature",
    "Receiver tube inner diameter does not exceed outer diameter",
    "Receiver tubes fit inside the receiver aperture",
    "Receiver outlet temperature is at least the turbine inlet temperature",
)
STEAM_CONSTRAINTS = (
    "Parasitic losses remain below the allowed fraction",
    "Steam-generator tube outer diameter does not exceed tube spacing",
    "Steam-generator tube inner diameter does not exceed outer diameter",
    "Steam-generator tube pressure does not exceed yield pressure",
)


def _field_variables(initial: Sequence[float | int]) -> tuple[SolarVariable, ...]:
    return (
        _v("heliostat_length_m", "continuous", 1, 40, initial[0], "Heliostat length", "m"),
        _v("heliostat_width_m", "continuous", 1, 40, initial[1], "Heliostat width", "m"),
        _v("tower_height_m", "continuous", 20, 250, initial[2], "Tower height", "m"),
        _v("receiver_aperture_height_m", "continuous", 1, 30, initial[3], "Receiver aperture height", "m"),
        _v("receiver_aperture_width_m", "continuous", 1, 30, initial[4], "Receiver aperture width", "m"),
        _v(
            "heliostat_count",
            "integer",
            1,
            None,
            initial[5],
            "Number of heliostats fitted in the field",
            sampling_upper=50_000,
        ),
        _v("field_angular_width_deg", "continuous", 1, 89, initial[6], "Field angular width", "degree"),
        _v("minimum_tower_distance_ratio", "continuous", 0, 20, initial[7], "Minimum distance from tower as a tower-height multiple"),
        _v("maximum_tower_distance_ratio", "continuous", 1, 20, initial[8], "Maximum distance from tower as a tower-height multiple"),
    )


def _storage_variables(initial: Sequence[float]) -> tuple[SolarVariable, ...]:
    return (
        _v("receiver_outlet_temperature_k", "continuous", 793, 995, initial[0], "Receiver outlet temperature", "K"),
        _v("hot_storage_height_m", "continuous", 2, 50, initial[1], "Hot-storage tank height", "m"),
        _v("hot_storage_diameter_m", "continuous", 2, 30, initial[2], "Hot-storage tank diameter", "m"),
        _v("hot_storage_insulation_m", "continuous", 0.01, 5, initial[3], "Hot-storage insulation thickness", "m"),
        _v("cold_storage_insulation_m", "continuous", 0.01, 5, initial[4], "Cold-storage insulation thickness", "m"),
    )


def _cost_loop_variables(initial: Sequence[float | int]) -> tuple[SolarVariable, ...]:
    return (
        _v("receiver_outlet_temperature_k", "continuous", 793, 995, initial[0], "Receiver outlet temperature", "K"),
        _v("hot_storage_height_m", "continuous", 1, 50, initial[1], "Hot-storage tank height", "m"),
        _v("hot_storage_diameter_m", "continuous", 1, 30, initial[2], "Hot-storage tank diameter", "m"),
        _v("hot_storage_insulation_m", "continuous", 0.01, 5, initial[3], "Hot-storage insulation thickness", "m"),
        _v("cold_storage_insulation_m", "continuous", 0.01, 5, initial[4], "Cold-storage insulation thickness", "m"),
        _v("minimum_cold_storage_temperature_k", "continuous", 495, 650, initial[5], "Minimum cold-storage temperature", "K"),
        _v("receiver_tube_count", "integer", 1, 7_853, initial[6], "Number of receiver tubes"),
        _v("receiver_insulation_m", "continuous", 0.01, 5, initial[7], "Receiver insulation thickness", "m"),
        _v("receiver_tube_inner_diameter_m", "continuous", 0.005, 0.1, initial[8], "Receiver tube inner diameter", "m"),
        _v("receiver_tube_outer_diameter_m", "continuous", 0.006, 0.1, initial[9], "Receiver tube outer diameter", "m"),
    )


def _steam_variables(initial: Sequence[float | int]) -> tuple[SolarVariable, ...]:
    return (
        _v("steam_tube_spacing_m", "continuous", 0.007, 0.2, initial[0], "Steam-generator tube spacing", "m"),
        _v("steam_tube_length_m", "continuous", 0.5, 10, initial[1], "Steam-generator tube length", "m"),
        _v("steam_tube_inner_diameter_m", "continuous", 0.005, 0.1, initial[2], "Steam-generator tube inner diameter", "m"),
        _v("steam_tube_outer_diameter_m", "continuous", 0.006, 0.1, initial[3], "Steam-generator tube outer diameter", "m"),
        _v("steam_baffle_cut", "continuous", 0.15, 0.4, initial[4], "Steam-generator baffle cut"),
        _v("steam_baffle_count", "integer", 2, None, initial[5], "Number of steam-generator baffles", sampling_upper=100),
        _v("steam_tube_count", "integer", 1, None, initial[6], "Number of steam-generator tubes", sampling_upper=50_000),
        _v("steam_shell_pass_count", "integer", 1, 10, initial[7], "Number of steam-generator shell passes"),
        _v("steam_tube_pass_count", "integer", 1, 9, initial[8], "Number of steam-generator tube passes"),
        _v("turbine_type", "categorical", 1, 8, initial[9], "Turbine model index"),
    )


MAXNRG_VARIABLES = _field_variables((8, 8, 150, 7, 7, 250, 45, 0.5, 5))

MINSURF_VARIABLES = _field_variables((11, 11, 140, 10, 10, 2650, 89, 0.5, 5)) + (
    _v("receiver_outlet_temperature_k", "continuous", 793, 995, 838, "Receiver outlet temperature", "K"),
    _v("receiver_tube_count", "integer", 1, 9_424, 36, "Number of receiver tubes"),
    _v("receiver_insulation_m", "continuous", 0.01, 5, 0.30, "Receiver insulation thickness", "m"),
    _v("receiver_tube_inner_diameter_m", "continuous", 0.005, 0.1, 0.020, "Receiver tube inner diameter", "m"),
    _v("receiver_tube_outer_diameter_m", "continuous", 0.005, 0.1, 0.0216, "Receiver tube outer diameter", "m"),
)

MINCOST_C1_VARIABLES = _field_variables((8, 8, 150, 7, 7, 250, 45, 0.5, 5)) + (
    _v("receiver_outlet_temperature_k", "continuous", 793, 995, 900, "Receiver outlet temperature", "K"),
    _v("hot_storage_height_m", "continuous", 1, 50, 9, "Hot-storage tank height", "m"),
    _v("hot_storage_diameter_m", "continuous", 1, 30, 9, "Hot-storage tank diameter", "m"),
    _v("hot_storage_insulation_m", "continuous", 0.01, 5, 0.30, "Hot-storage insulation thickness", "m"),
    _v("cold_storage_insulation_m", "continuous", 0.01, 5, 0.20, "Cold-storage insulation thickness", "m"),
    _v("minimum_cold_storage_temperature_k", "continuous", 495, 650, 560, "Minimum cold-storage temperature", "K"),
    _v("receiver_tube_count", "integer", 1, 9_424, 40, "Number of receiver tubes"),
    _v("receiver_insulation_m", "continuous", 0.01, 5, 0.30, "Receiver insulation thickness", "m"),
    _v("receiver_tube_inner_diameter_m", "continuous", 0.005, 0.1, 0.015, "Receiver tube inner diameter", "m"),
    _v("receiver_tube_outer_diameter_m", "continuous", 0.005, 0.1, 0.017, "Receiver tube outer diameter", "m"),
    _v("turbine_type", "categorical", 1, 8, 3, "Turbine model index"),
)

MINCOST_C2_VARIABLES = (
    _field_variables((9, 9, 150, 6, 8, 1000, 45, 0.5, 5))
    + _cost_loop_variables((900, 9, 9, 0.30, 0.20, 560, 500, 0.30, 0.0165, 0.018))
    + _steam_variables((0.017, 10, 0.0155, 0.016, 0.20, 3, 12000, 1, 2, 2))
)

MAXCOMP_VARIABLES = (
    _v("receiver_outlet_temperature_k", "continuous", 793, 995, 900, "Receiver outlet temperature", "K"),
    _v("hot_storage_height_m", "continuous", 1, 30, 10, "Hot-storage tank height", "m"),
    _v("hot_storage_diameter_m", "continuous", 1, 30, 12, "Hot-storage tank diameter", "m"),
    _v("hot_storage_insulation_m", "continuous", 0.01, 2, 0.15, "Hot-storage insulation thickness", "m"),
    _v("cold_storage_insulation_m", "continuous", 0.01, 2, 0.10, "Cold-storage insulation thickness", "m"),
    _v("minimum_cold_storage_temperature_k", "continuous", 495, 650, 560, "Minimum cold-storage temperature", "K"),
    _v("receiver_tube_count", "integer", 1, 1_884, 24, "Number of receiver tubes"),
    _v("receiver_insulation_m", "continuous", 0.1, 2, 0.35, "Receiver insulation thickness", "m"),
    _v("receiver_tube_inner_diameter_m", "continuous", 0.005, 0.1, 0.020, "Receiver tube inner diameter", "m"),
    _v("receiver_tube_outer_diameter_m", "continuous", 0.005, 0.1, 0.023, "Receiver tube outer diameter", "m"),
) + tuple(
    # MAXCOMP uses the same steam-generator concepts with a 0.006 m spacing bound.
    _v(
        variable.name,
        variable.kind,
        0.006 if variable.name == "steam_tube_spacing_m" else variable.lower,
        variable.upper,
        variable.initial,
        variable.description,
        variable.units,
        sampling_upper=variable.sampling_upper,
    )
    for variable in _steam_variables((0.050, 8, 0.020, 0.023, 0.20, 2, 5000, 5, 5, 1))
)

MINCOST_TS_VARIABLES = _storage_variables((900, 10, 12, 0.20, 0.20))

MAXEFF_VARIABLES = (
    _v("receiver_aperture_height_m", "continuous", 1, 30, 7, "Receiver aperture height", "m"),
    _v("receiver_aperture_width_m", "continuous", 1, 30, 7, "Receiver aperture width", "m"),
    _v("receiver_outlet_temperature_k", "continuous", 793, 995, 850, "Receiver outlet temperature", "K"),
    _v("receiver_tube_count", "integer", 1, 8_567, 40, "Number of receiver tubes"),
    _v("receiver_insulation_m", "continuous", 0.01, 5, 0.20, "Receiver insulation thickness", "m"),
    _v("receiver_tube_inner_diameter_m", "continuous", 0.005, 0.1, 0.010, "Receiver tube inner diameter", "m"),
    _v("receiver_tube_outer_diameter_m", "continuous", 0.0055, 0.1, 0.011, "Receiver tube outer diameter", "m"),
)

MINCOST_CH_VARIABLES = MINCOST_C2_VARIABLES + (
    _v("cold_storage_height_m", "continuous", 1, 50, 10.8, "Cold-storage tank height", "m"),
    _v("cold_storage_diameter_m", "continuous", 1, 30, 9, "Cold-storage tank diameter", "m"),
)


SOLAR_INSTANCES = {
    "MAXNRG_H1": SolarInstance(
        1,
        "MAXNRG_H1",
        "minimize negative total daily solar energy concentrated on the receiver (kWh)",
        MAXNRG_VARIABLES,
        (
            "Plant cost does not exceed the $50M budget",
            "Field surface area does not exceed 195 hectares",
            *FIELD_CONSTRAINTS[2:],
        ),
        stochastic=True,
        fidelity_fixed=True,
    ),
    "MINSURF_H1": SolarInstance(
        2,
        "MINSURF_H1",
        "minimize total heliostat-field surface (square metres)",
        MINSURF_VARIABLES,
        (
            "Field surface area does not exceed 400 hectares",
            "Compliance to demand",
            "Plant cost does not exceed the $300M budget",
            *FIELD_CONSTRAINTS[2:],
            *RECEIVER_CONSTRAINTS[:3],
            *RECEIVER_CONSTRAINTS[4:],
        ),
        stochastic=True,
    ),
    "MINCOST_C1": SolarInstance(
        3,
        "MINCOST_C1",
        "minimize total investment cost ($)",
        MINCOST_C1_VARIABLES,
        (
            "Field surface area does not exceed 80 hectares",
            "Compliance to demand",
            *FIELD_CONSTRAINTS[2:],
            *RECEIVER_CONSTRAINTS,
            "Storage returns at least to its original conditions",
        ),
        stochastic=True,
    ),
    "MINCOST_C2": SolarInstance(
        4,
        "MINCOST_C2",
        "minimize total power-plant investment cost ($)",
        MINCOST_C2_VARIABLES,
        (
            "Field surface area does not exceed 200 hectares",
            "Compliance to demand",
            *FIELD_CONSTRAINTS[2:],
            *RECEIVER_CONSTRAINTS,
            *STEAM_CONSTRAINTS,
        ),
        stochastic=True,
    ),
    "MAXCOMP_HTF1": SolarInstance(
        5,
        "MAXCOMP_HTF1",
        "minimize negative compliance with the demand profile",
        MAXCOMP_VARIABLES,
        (
            "Plant cost does not exceed the $100M budget",
            *RECEIVER_CONSTRAINTS,
            *STEAM_CONSTRAINTS,
        ),
        stochastic=False,
        fidelity_fixed=True,
    ),
    "MINCOST_TS": SolarInstance(
        6,
        "MINCOST_TS",
        "minimize thermal-storage cost ($)",
        MINCOST_TS_VARIABLES,
        (
            "Compliance to demand",
            "Receiver-tube pressure does not exceed yield pressure",
            "Molten-salt melting point is below the hot-storage minimum temperature",
            "Molten-salt melting point is below the cold-storage minimum temperature",
            "Receiver outlet temperature is at least the turbine inlet temperature",
            "Storage returns at least to its original conditions",
        ),
        stochastic=False,
        fidelity_fixed=True,
    ),
    "MAXEFF_RE": SolarInstance(
        7,
        "MAXEFF_RE",
        "minimize negative receiver efficiency (energy transferred to molten salt)",
        MAXEFF_VARIABLES,
        (
            "Plant cost does not exceed the $45M budget",
            "Receiver-tube pressure does not exceed yield pressure",
            "Receiver tube inner diameter does not exceed outer diameter",
            "Receiver outlet temperature is at least the turbine inlet temperature",
            "Receiver tubes fit inside the receiver aperture",
            "Parasitic losses remain below 3% of generated output",
        ),
        stochastic=True,
    ),
    "MINCOST_UNCONSTRAINED": SolarInstance(
        10,
        "MINCOST_UNCONSTRAINED",
        "minimize storage cost plus squared penalties from MINCOST_TS",
        MINCOST_TS_VARIABLES,
        (),
        stochastic=False,
    ),
    "MINCOST_CH": SolarInstance(
        11,
        "MINCOST_CH",
        "minimize corrected total power-plant investment cost ($)",
        MINCOST_CH_VARIABLES,
        (
            "Field surface area does not exceed 200 hectares",
            "Compliance to demand",
            *FIELD_CONSTRAINTS[2:],
            *RECEIVER_CONSTRAINTS,
            *STEAM_CONSTRAINTS,
        ),
        stochastic=True,
    ),
}
DEFAULT_INSTANCE = "MAXEFF_RE"


def resolve_solar_instance(instance: str) -> SolarInstance:
    """Resolve and validate one selected official instance name."""

    key = instance.upper()
    try:
        return SOLAR_INSTANCES[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown SOLAR instance {instance!r}; choose one of {tuple(SOLAR_INSTANCES)}"
        ) from exc


def solar_search_space_summary(instance: str) -> dict[str, Any]:
    """Return a JSON-friendly exact domain and prior-schema description."""

    spec = resolve_solar_instance(instance)
    dimensions = []
    for position, variable in enumerate(spec.variables, start=1):
        prior_dimension: dict[str, Any] = {
            "name": variable.name,
            "type": variable.prior_type,
            "meaning": variable.description,
        }
        if variable.kind == "categorical":
            assert variable.upper is not None
            prior_dimension["values"] = list(
                range(int(variable.lower), int(variable.upper) + 1)
            )
        else:
            prior_dimension["range"] = [
                variable.lower,
                variable.sampling_upper,
            ]
            prior_dimension["scale"] = "linear"
        dimensions.append(
            {
                "position": position,
                "name": variable.name,
                "simulator_type": variable.kind,
                "official_lower": variable.lower,
                "official_upper": variable.upper,
                "candidate_pool_upper": variable.sampling_upper,
                "initial": variable.initial,
                "units": variable.units,
                "description": variable.description,
                "prior_schema": prior_dimension,
            }
        )
    return {
        "artifact_format": "llm-bo-solar-search-space-v1",
        "dataset_id": spec.dataset_id,
        "source_commit": SOLAR_COMMIT,
        "instance": spec.name,
        "problem_number": spec.number,
        "objective_direction": "minimize",
        "objective": spec.objective,
        "stochastic": spec.stochastic,
        "truth_fidelity_required": spec.fidelity_fixed,
        "dimension_count": spec.n_dimensions,
        "constraint_count": spec.n_constraints,
        "feasibility": "all returned constraint values must be <= 0",
        "dimensions": dimensions,
        "constraints": list(spec.constraints),
    }


# A-priori constraints: input-only validity relations the SIMULATOR ITSELF documents
# as "A priori" in its help text (source/src/helpFunctions.cpp; formulas quoted
# verbatim in each entry). Violating one makes the binary refuse the query and return
# 1e20 sentinels ("hidden failure"), so most rows of an unfiltered Latin-hypercube pool
# over the raw box crash. Pre-screening the pool on these relations never queries the
# simulator: the documented input relations are evaluated in Python.
APRIORI_CHECKS: dict[str, tuple[tuple[str, Any], ...]] = {
    "MAXEFF_RE": (
        (
            "c3: x6 <= x7 (tube inner diameter <= outer diameter)",
            lambda c: c["receiver_tube_inner_diameter_m"]
            <= c["receiver_tube_outer_diameter_m"],
        ),
        (
            "c5: x4*x7 <= x2*PI/2 (tubes fit inside receiver aperture)",
            lambda c: c["receiver_tube_count"] * c["receiver_tube_outer_diameter_m"]
            <= c["receiver_aperture_width_m"] * math.pi / 2.0,
        ),
    ),
}


def apriori_valid(instance: str, candidate: Mapping[str, Any]) -> bool:
    """True when the candidate satisfies every documented a-priori input relation."""

    checks = APRIORI_CHECKS.get(resolve_solar_instance(instance).name, ())
    return all(fn(candidate) for _, fn in checks)


def generate_solar_candidate_pool(
    instance: str,
    size: int,
    *,
    seed: int = 42,
    prescreen: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Generate a reproducible mixed-variable Latin-hypercube candidate pool.

    The live simulator has a continuous/mixed domain, whereas this repository's
    BO engine selects from a finite set. The official starting point is row 0;
    the remaining rows form a seeded stratified design. Officially unbounded
    integer variables use the documented adapter caps in the search summary.

    With ``prescreen=True`` the stratified rows are rejection-filtered on the
    simulator's DOCUMENTED a-priori input relations (:data:`APRIORI_CHECKS` —
    evaluated in Python; the simulator is never queried) and generation batches
    continue until ``size`` valid rows exist. Requires the instance to have
    documented checks.
    """

    spec = resolve_solar_instance(instance)
    if size < 2:
        raise ValueError("SOLAR candidate-pool size must be at least 2")
    if prescreen and spec.name not in APRIORI_CHECKS:
        raise ValueError(
            f"no documented a-priori input checks for {spec.name}; the candidate pool can be "
            f"pre-screened for {sorted(APRIORI_CHECKS)} only"
        )
    rng = np.random.default_rng(seed)

    def batch(n_rows: int) -> list[dict[str, Any]]:
        unit = np.empty((n_rows, spec.n_dimensions), dtype=float)
        for column in range(spec.n_dimensions):
            unit[:, column] = (rng.permutation(n_rows) + rng.random(n_rows)) / n_rows
        rows: list[dict[str, Any]] = []
        for row in range(n_rows):
            candidate: dict[str, Any] = {}
            for column, variable in enumerate(spec.variables):
                low = variable.lower
                high = variable.sampling_upper
                value = low + unit[row, column] * (high - low)
                if variable.kind in {"integer", "categorical"}:
                    value = int(np.clip(np.rint(value), math.ceil(low), math.floor(high)))
                else:
                    value = float(value)
                candidate[variable.name] = value
            rows.append(candidate)
        return rows

    initial = {variable.name: variable.initial for variable in spec.variables}
    if prescreen and not apriori_valid(instance, initial):
        raise RuntimeError(f"official initial point violates a-priori checks ({spec.name})")
    records: list[dict[str, Any]] = [initial]
    if not prescreen:
        records.extend(batch(size - 1))
        return tuple(records)
    while len(records) < size:
        for candidate in batch(size - 1):
            if apriori_valid(instance, candidate):
                records.append(candidate)
                if len(records) == size:
                    break
    return tuple(records)


def _candidate_values(
    candidate: Mapping[str, Any],
    spec: SolarInstance,
) -> list[float | int]:
    expected = {variable.name for variable in spec.variables}
    missing = expected - set(candidate)
    extra = set(candidate) - expected
    if missing or extra:
        raise ValueError(
            f"SOLAR candidate keys mismatch (missing={sorted(missing)}, "
            f"extra={sorted(extra)})"
        )
    values: list[float | int] = []
    for variable in spec.variables:
        try:
            numeric = float(candidate[variable.name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{variable.name} must be numeric") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"{variable.name} must be finite")
        upper = variable.upper
        if numeric < variable.lower or (upper is not None and numeric > upper):
            interval = f"[{variable.lower}, {upper}]" if upper is not None else f">= {variable.lower}"
            raise ValueError(f"{variable.name} must be in {interval}")
        if variable.kind in {"integer", "categorical"}:
            if not numeric.is_integer():
                raise ValueError(f"{variable.name} must be an integer")
            values.append(int(numeric))
        else:
            values.append(numeric)
    return values


def evaluate_solar(
    instance: str,
    candidate: Mapping[str, Any],
    *,
    executable: str | Path = SOLAR_EXECUTABLE,
    seed: int = 0,
    fidelity: float = 1.0,
    replications: int | float = 1,
    timeout_seconds: float | None = None,
) -> SolarEvaluation:
    """Run one trusted SOLAR executable query and parse objective/constraints."""

    spec = resolve_solar_instance(instance)
    executable_path = Path(executable).expanduser().resolve()
    if not executable_path.is_file():
        raise FileNotFoundError(
            f"SOLAR executable not found: {executable_path}. Run "
            "`python scripts/setup_solar.py` first."
        )
    if seed < 0:
        raise ValueError("SOLAR seed must be non-negative")
    if not 0 <= fidelity <= 1:
        raise ValueError("SOLAR fidelity must lie in [0, 1]")
    if spec.fidelity_fixed and fidelity != 1.0:
        raise ValueError(f"{spec.name} requires fidelity=1.0")
    if not spec.stochastic and replications != 1:
        raise ValueError(f"{spec.name} is deterministic and requires replications=1")
    values = _candidate_values(candidate, spec)

    with tempfile.TemporaryDirectory(prefix="hapbo-solar-") as directory:
        input_path = Path(directory) / "candidate.txt"
        input_path.write_text(" ".join(map(str, values)) + "\n", encoding="utf-8")
        command = [
            str(executable_path),
            str(spec.number),
            str(input_path),
            f"-seed={seed}",
            f"-fid={fidelity}",
            f"-rep={replications}",
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"SOLAR {spec.name} evaluation failed with exit code "
            f"{completed.returncode}: {detail}"
        )
    raw = completed.stdout.strip()
    try:
        outputs = np.asarray([float(value) for value in raw.split()], dtype=float)
    except ValueError as exc:
        raise RuntimeError(f"SOLAR returned non-numeric output: {raw!r}") from exc
    expected = 1 + spec.n_constraints
    if outputs.size != expected:
        raise RuntimeError(
            f"SOLAR {spec.name} returned {outputs.size} values; expected {expected}. "
            "Re-run `python scripts/setup_solar.py` to apply the pinned "
            "compatibility patches and rebuild."
        )
    if not np.all(np.isfinite(outputs)):
        raise RuntimeError("SOLAR returned a non-finite output")
    constraints = outputs[1:].copy()
    hidden_failure = bool(np.any(np.abs(outputs) >= HIDDEN_FAILURE_VALUE * 0.99))
    feasible = bool(not hidden_failure and np.all(constraints <= 0))
    return SolarEvaluation(
        objective=float(outputs[0]),
        constraints=constraints,
        feasible=feasible,
        hidden_failure=hidden_failure,
        raw_output=raw,
    )


def write_solar_search_spaces(
    root: str | Path = SOLAR_DIRECTORY,
) -> tuple[Path, ...]:
    """Write one machine-readable search-space summary per instance (``<root>/<instance>/search_space.json``)."""

    output_root = Path(root).expanduser().resolve()
    paths = []
    for name in SOLAR_INSTANCES:
        directory = output_root / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "search_space.json"
        path.write_text(
            json.dumps(solar_search_space_summary(name), indent=2) + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    return tuple(paths)
