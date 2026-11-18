"""Provenance that names a human source (user_stated / user_confirmed) requires a human statement in the card.

The card declares it in metadata (card.human_statement); the compiler (load_record, finalization) and the
elicitor's validate-and-retry loop enforce the rule.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm.psl_elicitor import elicit_belief_document  # noqa: E402
from prior.expert_knowledge.psl_schema import VOCABULARY_CARD_PATH, load_psl_schema  # noqa: E402
from prior.landscape.landscape_generator import (  # noqa: E402
    HUMAN_PROVENANCE,
    _sha256_text_artifact,
    compile_prior,
    declares_human_statement,
    finalize_prior_experiment,
    initialize_prior_experiment,
    load_record,
    validate_record,
)

PRIORS = ROOT / "experiments" / "priors"
MUSCLES = ("left-knee-flexor", "left-tibialis-anterioris", "right-tibialis-anterioris")
GEOMETRY = PRIORS / "neuromosaic-dataset1-rat" / "left-knee-flexor" / "llm-geometry"
RULE_TEXT = "the task card carries no human or expert statement"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _geometry_documents(directory: Path = GEOMETRY) -> tuple[dict, dict]:
    """(shipped llm-geometry document, the document with the ``original`` values of its metadata ``corrections``).

    The ``corrections`` record of the shipped metadata names belief 1 with original provenance user_stated.
    """
    shipped = json.loads((directory / "prior.json").read_text(encoding="utf-8"))
    human = copy.deepcopy(shipped)
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    for fix in meta["corrections"]:
        human["beliefs"][int(fix["belief"].strip("beliefs[]"))][fix["field"]] = fix["original"]
    return shipped, human


def test_validator_rejects_human_provenance_without_a_card_statement() -> None:
    shipped, human = _geometry_documents()
    for tag in HUMAN_PROVENANCE:
        document = copy.deepcopy(human)
        document["beliefs"][1]["provenance"] = tag
        with pytest.raises(ValueError, match=r"beliefs\[1\] name a human source.*" + RULE_TEXT):
            validate_record(document, card_human_statement=False)
        validate_record(document, card_human_statement=True)
        validate_record(document)  # bare document: no card known, no check
    validate_record(shipped, card_human_statement=False)


class _MockClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self._responses.pop(0)


def test_elicitor_feeds_the_rule_back_and_accepts_the_corrected_document() -> None:
    shipped, human = _geometry_documents()
    card = (GEOMETRY / "prompt_card.md").read_text(encoding="utf-8")
    kwargs = dict(schema=load_psl_schema(), vocabulary_card_text=VOCABULARY_CARD_PATH.read_text(encoding="utf-8"))

    client = _MockClient([json.dumps(human), json.dumps(shipped)])
    result = elicit_belief_document(client, card, **kwargs)
    assert result.document == shipped
    assert len(client.prompts) == 2
    assert "Validator error" in client.prompts[1] and RULE_TEXT in client.prompts[1]

    client = _MockClient([json.dumps(human)])  # a card that does carry a human statement
    assert elicit_belief_document(client, card, card_human_statement=True, **kwargs).document == human
    assert len(client.prompts) == 1


def test_versioned_prior_needs_the_declaration(tmp_path: Path) -> None:
    _, human = _geometry_documents()
    text = json.dumps(human, indent=2) + "\n"

    undeclared = initialize_prior_experiment("fixture-dataset", "undeclared", experiments_directory=tmp_path)
    assert json.loads((undeclared / "metadata.json").read_text())["card"] == {"human_statement": False}
    (undeclared / "prior.json").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=RULE_TEXT):
        finalize_prior_experiment(undeclared, generate_landscape=False)

    declared = initialize_prior_experiment(
        "fixture-dataset", "declared", experiments_directory=tmp_path, card_human_statement=True
    )
    (declared / "prior.json").write_text(text, encoding="utf-8")
    finalize_prior_experiment(declared, generate_landscape=False)
    assert load_record(declared / "prior.json") == human

    meta_path = declared / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["card"]["human_statement"] = False
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=RULE_TEXT):
        load_record(declared / "prior.json")
    meta["card"] = {"human_statement": "yes"}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="card.human_statement must be true or false"):
        load_record(declared / "prior.json")


@pytest.mark.parametrize("muscle", MUSCLES)
def test_llm_geometry_corrections_record_matches_the_shipped_prior(muscle: str, tmp_path: Path) -> None:
    """The shipped prior.json and the document rebuilt from the ``corrections`` record hash as recorded."""
    directory = PRIORS / "neuromosaic-dataset1-rat" / muscle / "llm-geometry"
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    (fix,) = meta["corrections"]
    assert (fix["belief"], fix["field"], fix["original"], fix["corrected"]) == (
        "beliefs[1]", "provenance", "user_stated", "llm_domain"
    )
    shipped, human = _geometry_documents(directory)
    assert shipped["beliefs"][1]["provenance"] == fix["corrected"]
    assert _sha256_text_artifact(directory / "prior.json") == meta["files"]["prior.json"]["sha256"]
    for document, digest in ((shipped, meta["files"]["prior.json"]["sha256"]), (human, fix["prior_sha256_before"])):
        path = tmp_path / "prior.json"
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
        assert _sha256_text_artifact(path) == digest


def _builder_human_text(card: str) -> bool:
    """Whether the card holds human text written by the card builders (expert map card, quote, map table)."""
    cortical = _script("build_cortical_solar_cards")
    expert = _script("build_cortical_expert_prior")
    return card == expert._CARD or cortical.EXPERT_QUOTE in card or "## Human expert map\n" in card


def test_shipped_priors_claim_no_human_source_their_card_lacks() -> None:
    declared = []
    for prior in sorted(PRIORS.rglob("prior.json")):
        record = load_record(prior)  # enforces the rule against metadata card.human_statement
        meta = json.loads((prior.parent / "metadata.json").read_text(encoding="utf-8"))
        card = (prior.parent / "prompt_card.md").read_text(encoding="utf-8")
        assert declares_human_statement(meta) == _builder_human_text(card), prior.parent
        if declares_human_statement(meta):
            declared.append(prior.parent.relative_to(PRIORS).as_posix())
        else:
            assert not any(b.get("provenance") in HUMAN_PROVENANCE for b in record["beliefs"]), prior
    # 8 muscles x (human-expert-map, human-expert-map-shared, llm-expert-hint)
    assert len(declared) == 24 and all("/human-expert-map" in d or "/llm-expert-hint" in d for d in declared)
    # PSL documents shipped without a card or metadata carry no human-source provenance at all.
    for root in ("synthetic_priors", "psl_elicitor_priors"):
        for prior in sorted((ROOT / "experiments" / root).rglob("prior.json")):
            beliefs = json.loads(prior.read_text(encoding="utf-8"))["beliefs"]
            assert not any(b.get("provenance") in HUMAN_PROVENANCE for b in beliefs), prior


def test_llm_geometry_alpha_and_provenance() -> None:
    """The shipped llm-geometry priors compile to alpha 0.455; provenance enters alpha, not pi."""
    shipped, human = _geometry_documents()
    for muscle in MUSCLES:
        prior = PRIORS / "neuromosaic-dataset1-rat" / muscle / "llm-geometry" / "prior.json"
        assert compile_prior(load_record(prior)).alpha == pytest.approx(0.455)
    assert compile_prior(human).alpha == pytest.approx(0.630)  # belief 1 tagged user_stated
    grid = [{"electrode_mediolateral_index": ml, "electrode_rostrocaudal_index": rc}
            for rc in range(1, 9) for ml in range(1, 5)]
    assert list(compile_prior(shipped).pi(grid)) == list(compile_prior(human).pi(grid))
