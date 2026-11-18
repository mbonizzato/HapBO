"""Every BO arm of a finite task uses the same GP input encoding.

The tests drive the worker's dispatch (``_run_offline`` / ``_run_solar``) up to the optimizer
construction and assert that the ``encoded`` array handed to the optimizer by ``vanilla-bo-*``
is bit-identical to the one handed over by ``alpha-pi-raw-*-matched``, for a cortical map, an
HPOBench task and SOLAR.  Unavailable datasets (NeuroMOSAIC .mat, HPOBench tables, the SOLAR
binary) are replaced by minimal synthetic problems that go through the same code path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.neuromosaic_common import OfflineOptimizationProblem  # noqa: E402
from experiments import worker  # noqa: E402
from experiments.catalog import load_catalog  # noqa: E402

CATALOG = load_catalog()
CORTICAL_CARD = (
    PROJECT_ROOT / "experiments" / "priors" / "neuromosaic-dataset1-rat" / "left-knee-flexor"
    / "human-expert-map" / "prior.json"
)
HPO_CARD = (
    PROJECT_ROOT / "experiments" / "priors" / "hpobench-lr-australian" / "australian"
    / "base" / "prior.json"
)
SOLAR_SCHEMA = PROJECT_ROOT / "experiments" / "priors" / "solar-maxeff-re" / "schema.json"
PRIOR_ARMS = ("alpha-pi-raw-ei-matched", "alpha-pi-raw-pi-matched", "alpha-pi-raw-ucb-matched")
VANILLA_ARMS = ("vanilla-bo-ei", "vanilla-bo-pi", "vanilla-bo-ucb")


class _Captured(Exception):
    pass


def _capture_encoded(monkeypatch) -> dict:
    box: dict = {}

    def fake_build_optimizer(encoded, *args, **kwargs):
        box["encoded"] = np.array(encoded, copy=True)
        raise _Captured

    monkeypatch.setattr(worker, "_build_optimizer", fake_build_optimizer)
    return box


def _offline_encoded(monkeypatch, tmp_path, problem, adapter, method_id, prior_path, optimizer=None):
    box = _capture_encoded(monkeypatch)
    monkeypatch.setattr(worker, "_load_offline_problem", lambda *a, **k: problem)
    run = {
        "run_id": "t", "method_id": method_id, "seed": 1001, "budget": 4, "parameters": {},
        "optimizer": dict(optimizer or {"n_initial": 1, "length_scale": 1.0}),
        "prior_path": None if prior_path is None else str(prior_path),
    }
    with pytest.raises(_Captured):
        worker._run_offline(tmp_path, run, {"adapter": adapter}, CATALOG.methods[method_id])
    return box["encoded"]


# --------------------------------------------------------------------------- cortical
def _cortical_problem() -> OfflineOptimizationProblem:
    """8 x 4 electrode grid in a scrambled channel order (as ch2xy lists channels)."""

    coords = [(x, y) for x in range(1, 9) for y in range(1, 5)]
    order = np.random.default_rng(0).permutation(len(coords))
    native = tuple({"channel_x": coords[i][0], "channel_y": coords[i][1]} for i in order)
    encoded = np.asarray([[c["channel_x"], c["channel_y"]] for c in native], dtype=float)
    truth = np.random.default_rng(1).uniform(size=len(native))
    return OfflineOptimizationProblem(
        dataset_id="neuromosaic-dataset1-rat", subject_id="synthetic", target_name="LeftKneeFlexor",
        native_candidates=native, encoded_candidates=encoded, ground_truth=truth,
        response_standard_deviation=np.zeros_like(truth),
        objective=lambda c: 0.0, metadata={}, maximize=True,
    )


def _real_cortical_problem():
    try:
        return worker._load_offline_problem(
            "neuromosaic-dataset1-rat", {"subject": "rat1", "muscle": "LeftKneeFlexor"}
        )
    except Exception:  # noqa: BLE001 - .mat files are not shipped with the repo
        return None


@pytest.mark.parametrize("prior_arm, vanilla_arm", list(zip(PRIOR_ARMS, VANILLA_ARMS)))
def test_cortical_vanilla_encoding_equals_alpha_pi(monkeypatch, tmp_path, prior_arm, vanilla_arm) -> None:
    for problem in filter(None, (_cortical_problem(), _real_cortical_problem())):
        adapter = "neuromosaic-dataset1-rat"
        prior_enc = _offline_encoded(monkeypatch, tmp_path, problem, adapter, prior_arm, CORTICAL_CARD)
        vanilla_enc = _offline_encoded(monkeypatch, tmp_path, problem, adapter, vanilla_arm, CORTICAL_CARD)
        np.testing.assert_array_equal(vanilla_enc, prior_enc)
        # Task-level reference schema (no prior_path) gives the same array.
        no_path = _offline_encoded(monkeypatch, tmp_path, problem, adapter, vanilla_arm, None)
        np.testing.assert_array_equal(no_path, prior_enc)
        # The native candidate encoding is the raw 1-8 x 1-4 electrode coordinates.
        assert not np.array_equal(vanilla_enc, np.asarray(problem.encoded_candidates, dtype=float))
        assert vanilla_enc.min() == 0.0 and vanilla_enc.max() == 1.0


def test_cortical_random_search_keeps_the_native_encoding(monkeypatch, tmp_path) -> None:
    problem = _cortical_problem()
    adapter = "neuromosaic-dataset1-rat"
    random_enc = _offline_encoded(monkeypatch, tmp_path, problem, adapter, "random-search", CORTICAL_CARD)
    np.testing.assert_array_equal(random_enc, problem.encoded_candidates)


def test_missing_prior_path_raises(monkeypatch, tmp_path) -> None:
    problem = _cortical_problem()
    with pytest.raises(FileNotFoundError, match="prior_path set but missing"):
        _offline_encoded(monkeypatch, tmp_path, problem, "neuromosaic-dataset1-rat", "vanilla-bo-ei",
                         tmp_path / "absent" / "prior.json")


# --------------------------------------------------------------------------- HPOBench
def _real_hpo_problem():
    try:
        return worker._load_offline_problem("hpobench", {"model": "lr", "task": "australian"})
    except Exception:  # noqa: BLE001 - tabular data may be absent
        return None


def _synthetic_hpo(tmp_path: Path):
    """Two-axis log-spaced HPO grid plus a schema-matching PSL card (rank encoding)."""

    alpha = [float(v) for v in np.logspace(-5, 0, 5)]
    eta0 = [float(v) for v in np.logspace(-4, 0, 4)]
    native = tuple({"alpha": a, "eta0": e} for e in eta0 for a in alpha)
    problem = SimpleNamespace(
        native_candidates=native, parameter_names=["alpha", "eta0"], model="lr",
        task_name="synthetic", n_candidates=len(native), maximize=True,
        ground_truth=np.linspace(0.0, 1.0, len(native)), optimum_value=1.0,
        objective=lambda c: 0.0, dataset_id="hpobench", target_name="accuracy",
    )
    card = json.loads(HPO_CARD.read_text(encoding="utf-8"))
    card["schema"] = {"objective": "maximize", "dimensions": [
        {"name": "alpha", "type": "ordinal", "values": alpha},
        {"name": "eta0", "type": "ordinal", "values": eta0},
    ]}
    card["beliefs"] = [{"token": "baseline", "level": 0.05}]
    path = tmp_path / "card" / "prior.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card), encoding="utf-8")
    return problem, path


@pytest.mark.parametrize("prior_arm, vanilla_arm", list(zip(PRIOR_ARMS, VANILLA_ARMS)))
def test_hpobench_vanilla_encoding_equals_alpha_pi(monkeypatch, tmp_path, prior_arm, vanilla_arm) -> None:
    cases = [_synthetic_hpo(tmp_path)]
    real = _real_hpo_problem()
    if real is not None:
        cases.append((real, HPO_CARD))
    for problem, card in cases:
        opt = {"n_initial": 1, "length_scale": 0.2}
        prior_enc = _offline_encoded(monkeypatch, tmp_path, problem, "hpobench", prior_arm, card, opt)
        vanilla_enc = _offline_encoded(monkeypatch, tmp_path, problem, "hpobench", vanilla_arm, card, opt)
        np.testing.assert_array_equal(vanilla_enc, prior_enc)
        no_path = _offline_encoded(monkeypatch, tmp_path, problem, "hpobench", vanilla_arm, None, opt)
        np.testing.assert_array_equal(no_path, prior_enc)


def test_hpobench_real_task_differs_from_native_min_max(monkeypatch, tmp_path) -> None:
    real = _real_hpo_problem()
    if real is None:
        pytest.skip("HPOBench tabular data not available locally")
    opt = {"n_initial": 1, "length_scale": 0.2}
    task = _offline_encoded(monkeypatch, tmp_path, real, "hpobench", "vanilla-bo-ei", HPO_CARD, opt)
    native = _offline_encoded(monkeypatch, tmp_path, real, "hpobench", "random-search", HPO_CARD, opt)
    assert task.shape == native.shape
    assert not np.array_equal(task, native)


# --------------------------------------------------------------------------- SOLAR
def _solar_card(tmp_path: Path) -> Path:
    schema = json.loads(SOLAR_SCHEMA.read_text(encoding="utf-8"))
    card = {"schema": schema, "beliefs": [{"token": "baseline", "level": 0.05}],
            "trust": {"overall_confidence": "medium", "notes": "synthetic test card"}}
    path = tmp_path / "solar-card" / "prior.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card), encoding="utf-8")
    return path


def _solar_encoded(monkeypatch, tmp_path, method_id: str, prior_path, optimizer=None) -> np.ndarray:
    box = _capture_encoded(monkeypatch)
    run = {
        "run_id": "t", "method_id": method_id, "seed": 2001, "budget": 5,
        "parameters": {"instance": "MAXEFF_RE", "candidate_pool_size": 32},
        "optimizer": dict(optimizer or {"n_initial": 3, "length_scale": 1.0}),
        "prior_path": None if prior_path is None else str(prior_path),
    }
    benchmark = {"adapter": "solar", "defaults": {}}
    with pytest.raises(_Captured):
        worker._run_solar(tmp_path, run, benchmark, CATALOG.methods[method_id])
    return box["encoded"]


def test_solar_vanilla_encoding_equals_alpha_pi(monkeypatch, tmp_path) -> None:
    card = _solar_card(tmp_path)
    prior_enc = _solar_encoded(monkeypatch, tmp_path, "alpha-pi-raw-ei-matched", card)
    vanilla_enc = _solar_encoded(monkeypatch, tmp_path, "vanilla-bo-ei", card)
    np.testing.assert_array_equal(vanilla_enc, prior_enc)
    # Task-level schema.json (no prior_path) reproduces the card's encoding.
    no_path = _solar_encoded(monkeypatch, tmp_path, "vanilla-bo-ei", None)
    np.testing.assert_array_equal(no_path, prior_enc)
    # Random search keeps the linear _encode_solar encoding, which differs on the
    # log-scaled dimensions.
    random_enc = _solar_encoded(monkeypatch, tmp_path, "random-search", card)
    assert not np.array_equal(random_enc, prior_enc)
