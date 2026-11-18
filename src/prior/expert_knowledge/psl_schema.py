"""The PSL belief-document JSON Schema, kept in sync with the vocabulary card.

The canonical schema lives inside ``vocabulary_card.md`` section 1 (so the card is
the single human-facing source of truth).  For structured-output elicitation the
LLM must be decoded against a real JSON object, so we extract that fenced block
into a committed ``psl_schema.json`` and guard against drift with
:func:`assert_schema_matches_card` (run in tests and at elicitation startup).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
# HAPBO_VOCABULARY_CARD (project-relative) selects an alternative card (same schema section).
VOCABULARY_CARD_PATH = (_HERE.parents[2] / os.environ["HAPBO_VOCABULARY_CARD"]
                        if os.environ.get("HAPBO_VOCABULARY_CARD") else _HERE / "vocabulary_card.md")
PSL_SCHEMA_PATH = _HERE / "psl_schema.json"

_SECTION_MARKER = "## 1."
_FENCE = "```json"


def extract_schema_from_card(card_path: Path = VOCABULARY_CARD_PATH) -> dict[str, Any]:
    """Parse the first ```json fence under section 1 of the vocabulary card."""

    text = Path(card_path).read_text(encoding="utf-8")
    section = text.find(_SECTION_MARKER)
    if section == -1:
        raise ValueError(f"{card_path}: could not find the section-1 machine-contract heading")
    fence = text.find(_FENCE, section)
    if fence == -1:
        raise ValueError(f"{card_path}: could not find a ```json schema block in section 1")
    body_start = text.index("\n", fence) + 1
    body_end = text.index("```", body_start)
    return json.loads(text[body_start:body_end])


def load_psl_schema(schema_path: Path = PSL_SCHEMA_PATH) -> dict[str, Any]:
    """Load the committed standalone PSL schema."""

    return json.loads(Path(schema_path).read_text(encoding="utf-8"))


def dump_schema_json(schema: dict[str, Any]) -> str:
    """Canonical serialization used both for the committed file and hashing."""

    return json.dumps(schema, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def assert_schema_matches_card(
    schema_path: Path = PSL_SCHEMA_PATH,
    card_path: Path = VOCABULARY_CARD_PATH,
) -> None:
    """Fail loudly if the committed schema drifted from the vocabulary card."""

    from_card = extract_schema_from_card(card_path)
    from_file = load_psl_schema(schema_path)
    if from_card != from_file:
        raise ValueError(
            f"{schema_path} is out of sync with {card_path} section 1; "
            f"regenerate it with `python -m prior.expert_knowledge.psl_schema`"
        )


def _regenerate() -> None:
    schema = extract_schema_from_card()
    PSL_SCHEMA_PATH.write_text(dump_schema_json(schema), encoding="utf-8", newline="\n")
    print(f"wrote {PSL_SCHEMA_PATH}")


if __name__ == "__main__":
    _regenerate()
