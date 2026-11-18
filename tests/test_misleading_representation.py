"""Representation variants of the misleading HPOBench prior: priors and the three study manifests."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.catalog import load_catalog, load_study  # noqa: E402
from experiments.planner import build_study_plan  # noqa: E402
from prior.landscape.landscape_generator import compile_prior, load_record  # noqa: E402

CATALOG = load_catalog()
PRIORS = PROJECT_ROOT / "experiments" / "priors"
STUDIES = PROJECT_ROOT / "experiments" / "studies" / "hpo"
VARIANTS = ("pointonly", "gaussian", "directllm")
SCRIPTED = ("pointonly", "gaussian")   # scripts/make_representation_variants.py (no LLM)
SOURCES = sorted(PRIORS.glob("hpobench-*/*/llm-card-misleading"))


def test_every_misleading_prior_has_all_variants_with_its_alpha() -> None:
    assert len(SOURCES) == 40
    for src in SOURCES:
        alpha = compile_prior(load_record(src / "prior.json")).alpha
        for v in VARIANTS:
            dst = src.parent / f"llm-card-misleading-{v}"
            record = load_record(dst / "prior.json")                       # validates the finalized hashes
            assert abs(compile_prior(record).alpha - alpha) < 1e-9          # as the script checks
            assert record["trust"] == load_record(src / "prior.json")["trust"]
            md = json.loads((dst / "metadata.json").read_text(encoding="utf-8"))
            assert md["prior_id"] == dst.name and md["status"] == "finalized"
            assert (dst / "prompt_card.md").read_bytes() == (src / "prompt_card.md").read_bytes()


@pytest.mark.parametrize("task", ["hpobench-lr-australian/australian", "hpobench-svm-vehicle/vehicle"])
def test_variants_are_the_script_output(tmp_path, task) -> None:
    """scripts/make_representation_variants.py on a copy of the misleading prior gives the committed priors."""
    work = tmp_path / task
    work.mkdir(parents=True)
    shutil.copytree(PRIORS / task / "llm-card-misleading", work / "llm-card-misleading")
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "make_representation_variants.py"),
                    str(work / "llm-card-misleading")], check=True, cwd=PROJECT_ROOT, capture_output=True)
    for v in SCRIPTED:
        name = f"llm-card-misleading-{v}"
        assert (work / name / "prior.json").read_bytes() == (PRIORS / task / name / "prior.json").read_bytes()


@pytest.mark.parametrize("variant", VARIANTS)
def test_manifests_share_tasks_seeds_and_settings_with_the_card_variant(variant) -> None:
    def cells(name):
        plan = build_study_plan(load_study(STUDIES / name), CATALOG)
        assert plan.blocked_count == 0
        return plan, {(r.problem_id, r.method_id, r.budget, r.seed, json.dumps(r.optimizer, sort_keys=True)): r
                      for r in plan.runs}

    mis_plan, mis = cells(f"representation-{variant}-misleading")
    card_plan, card = cells(f"representation-{variant}")
    assert set(mis) == set(card) and len(mis) == 400
    assert {r.method_id for r in mis_plan.runs} == {"alpha-pi-raw-ei-matched"}
    for key, run in mis.items():
        card_prior = Path(card[key].prior_path)
        assert card_prior.parent.name == f"llm-card-{variant}"
        assert Path(run.prior_path) == card_prior.parent.parent / f"llm-card-misleading-{variant}" / "prior.json"
