"""Elicitation validate-and-retry loop, exercised with a mock client (no network)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llm.psl_elicitor import ElicitationError, _strip_to_json, elicit_belief_document  # noqa: E402
from prior.expert_knowledge.psl_schema import VOCABULARY_CARD_PATH, load_psl_schema  # noqa: E402

VALID_PRIOR = (
    PROJECT_ROOT
    / "experiments/priors/neuromosaic-dataset1-rat/left-knee-flexor/human-expert-map/prior.json"
)
_VALID_TEXT = VALID_PRIOR.read_text(encoding="utf-8")
_VOCAB = VOCABULARY_CARD_PATH.read_text(encoding="utf-8")
_CARD = "## Task\nProduce a belief document for the test problem."


class _MockClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def complete(self, prompt, *, system=None, response_schema=None, seed=None, elicitation_replicate=0, **kwargs):
        self.calls += 1
        return self._responses.pop(0)


def test_strip_to_json_handles_leaked_thinking_trace() -> None:
    wrapped = "let me think about this...\n" + '{"ok": true}'
    assert json.loads(_strip_to_json(wrapped)) == {"ok": True}


def test_retries_until_valid_then_returns_document_and_raw() -> None:
    client = _MockClient(["not json at all", '{"bad": true}', _VALID_TEXT])
    result = elicit_belief_document(  # the fixture is the human expert map (user_stated anchors)
        client, _CARD, schema=load_psl_schema(), vocabulary_card_text=_VOCAB, max_attempts=3,
        card_human_statement=True,
    )
    assert client.calls == 3  # two invalid attempts, then the valid one
    assert result.document == json.loads(_VALID_TEXT)
    assert result.raw == _VALID_TEXT


def test_raises_after_exhausting_attempts() -> None:
    client = _MockClient(["nope", "{}"])
    with pytest.raises(ElicitationError):
        elicit_belief_document(
            client, _CARD, schema=load_psl_schema(), vocabulary_card_text=_VOCAB, max_attempts=2
        )
    assert client.calls == 2
