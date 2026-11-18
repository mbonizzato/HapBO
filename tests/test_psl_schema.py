"""The committed PSL schema stays in sync with the vocabulary card."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prior.expert_knowledge import psl_schema  # noqa: E402


def test_committed_schema_matches_the_card() -> None:
    psl_schema.assert_schema_matches_card()  # raises on drift


def test_extract_equals_committed_file() -> None:
    assert psl_schema.extract_schema_from_card() == psl_schema.load_psl_schema()


def test_schema_shape() -> None:
    schema = psl_schema.load_psl_schema()
    assert schema["required"] == ["schema", "beliefs", "trust"]
    # All 15 PSL tokens are representable as belief branches.
    assert len(schema["$defs"]["belief"]["oneOf"]) == 15


def test_drift_is_detected(tmp_path) -> None:
    card = psl_schema.VOCABULARY_CARD_PATH.read_text(encoding="utf-8")
    mutated = card.replace('"title": "HapBO PSL belief document"', '"title": "changed"', 1)
    bad_card = tmp_path / "vocabulary_card.md"
    bad_card.write_text(mutated, encoding="utf-8")
    try:
        psl_schema.assert_schema_matches_card(card_path=bad_card)
    except ValueError:
        return
    raise AssertionError("expected a drift ValueError when the card changes")
