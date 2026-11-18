"""Cortical paper figures evaluate every run over 12 queries (paper/cortical_data.HORIZON)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")

import cortical_data as CD  # noqa: E402
import yaml  # noqa: E402

DATA = ROOT / "results" / "paper" / "cortical"


def test_horizon_and_budget_key() -> None:
    assert CD.HORIZON == 12
    assert CD.budget_key(["budget-0016"]) == "budget-0016"
    assert CD.budget_key(["budget-0016", "budget-0012"]) == "budget-0012"
    assert CD.budget_key(["budget-0008"]) is None


def test_cortical_manifests_use_the_horizon_as_budget() -> None:
    manifests = sorted((ROOT / "experiments" / "studies" / "cortical").glob("*/study.yaml"))
    assert manifests
    for m in manifests:
        assert yaml.safe_load(m.read_text(encoding="utf-8"))["design"]["budgets"] == [CD.HORIZON], m


def test_committed_cortical_runs_hold_the_horizon() -> None:
    """Every committed cortical run is a budget-0012 run of 12 queries (trajectories and exploration scores)."""
    studies = sorted(p for p in DATA.glob("*.json") if "methods" in json.loads(p.read_text(encoding="utf-8")))
    assert [p.stem for p in studies] == ["acquisition", "batch", "human-map-all", "human-map", "kernel-matern12",
                                         "kernel-rbf"]
    for p in studies:
        for method, budgets in json.loads(p.read_text(encoding="utf-8"))["methods"].items():
            assert list(budgets) == ["budget-0012"], (p.stem, method)
            assert all(len(r["y"]) == CD.HORIZON for rows in budgets["budget-0012"].values() for r in rows)
    human_map = json.loads((DATA / "human-map.json").read_text(encoding="utf-8"))["methods"]
    assert sorted(human_map) == ["alpha-pi-raw-ei-matched", "random-search", "vanilla-bo-ei"]
    for p in sorted((DATA / "exploration").glob("*.json")):
        scores = json.loads(p.read_text(encoding="utf-8"))
        assert scores and all(k.split("/")[2] == "budget-0012" and len(v) == CD.HORIZON for k, v in scores.items())
    assert len(CD.exploration("human-map")) == 3 * 160
    assert len(CD.exploration("human-map-all")) == 3 * 400


def _independent_q12(study: str, method: str, unit: str = "rat") -> float:
    """Mean over rats (or maps) of the best-so-far / map optimum at query 12, straight from the compact data."""
    raw = json.loads((DATA / f"{study}.json").read_text(encoding="utf-8"))["methods"][method]
    opt = json.loads((DATA / "map_optimum.json").read_text(encoding="utf-8"))
    groups: dict = {}
    for prob, runs in raw[CD.budget_key(raw)].items():
        vals = [max(r["y"][:12]) / opt[prob] for r in runs if r["status"] == "completed"]
        if vals:
            groups.setdefault(prob.split("-")[0] if unit == "rat" else prob, []).append(np.mean(vals))
    return float(np.mean([np.mean(v) for v in groups.values()]))


@pytest.mark.parametrize("study,method", [("human-map", "alpha-pi-raw-ei-matched"), ("human-map", "vanilla-bo-ei"),
                                          ("acquisition", "alpha-pi-raw-ucb-matched"), ("batch", "vanilla-bo-ei")])
def test_normalized_performance_endpoint_is_query_12(study: str, method: str) -> None:
    rows = CD.cort_curve(study, method)
    assert rows and all(len(r) == 12 for r in rows)
    assert np.mean(rows, axis=0)[-1] == pytest.approx(_independent_q12(study, method), abs=1e-12)


_LLAMBO_FAILED = {  # upstream raised: no candidate points generated; a candidate key outside the search space
    ("rat2-left-gluteus", 1002), ("rat6-right-medialis-gastrocnemius", 1002)}


def test_failed_llambo_runs_are_absent() -> None:
    maps = json.loads((DATA / "map_optimum.json").read_text(encoding="utf-8"))
    key = "{}/llambo-ws/budget-0012/seed-{}".format
    scores = CD.exploration("human-map-llambo")
    planned = {key(m, s) for m in maps for s in range(1001, 1011)}
    assert len(planned) == 400 and set(scores) <= planned
    assert planned - set(scores) == {key(m, s) for m, s in _LLAMBO_FAILED}
    assert sorted({len(v) for v in scores.values()}) == [CD.HORIZON]
    assert sum(len(v) for v in CD._explore_runs("human-map-llambo", "llambo-ws").values()) == 398


def test_generator_endpoints_are_query_12(tmp_path: Path) -> None:
    import fig_acquisition
    import fig_prior_sources
    import fig_realworld

    real = fig_realworld.build(tmp_path)["fig3_realworld"]["cortical_q12"]
    mu, *_ = CD.explore_mean_sem("human-map-all", "alpha-pi-raw-ei-matched", unit="map")
    assert len(mu) == 12 and real["HapBO"][0] == round(float(mu[11]), 4)

    src = fig_prior_sources.build(tmp_path)
    van, *_ = CD.explore_mean_sem("human-map", "vanilla-bo-ei")
    assert src["fig_prior_sources_main"]["vanilla"]["q12"] == round(float(van[11]), 4)
    assert src["figS2_prior_sources_and_trajectories"]["vanilla"] == round(float(van[11]), 4)

    acq = fig_acquisition.build(tmp_path)
    assert acq["fig_acq_across"]["cortical EI vanilla"][1] == round(
        _independent_q12("acquisition", "vanilla-bo-ei", "map"), 4)
    assert acq["fig_batch_bo"]["cortical E vanilla"][1] == round(_independent_q12("batch", "vanilla-bo-ei", "map"), 4)
