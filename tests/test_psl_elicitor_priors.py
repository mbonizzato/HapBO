"""The PSL documents of the elicitor comparison validate and match the compact figure data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prior.landscape.landscape_generator import load_record  # noqa: E402

DATA = PROJECT_ROOT / "results" / "paper" / "psl_elicitors" / "psl_elicitors.json"


def test_every_document_validates_and_is_listed() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    listed = {PROJECT_ROOT / p for routes in data["documents"].values() for fams in routes.values()
              for p in fams.values()}
    shipped = set((PROJECT_ROOT / "experiments" / "psl_elicitor_priors").rglob("prior.json"))
    assert listed == shipped
    assert len(shipped) == 5 * 4 * 4
    for path in sorted(shipped):
        load_record(path)


def test_compact_data_covers_every_elicitor_route_and_function() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    for agent in data["elicitors"]:
        for route in ("one-call", "agentic"):
            for fam in data["families"]:
                assert data["metrics"][agent][route][fam]["spearman_model"]
                assert data["regret"][agent][route][fam]
    for fam in data["families"]:
        assert data["regret"]["vanilla"][fam]
