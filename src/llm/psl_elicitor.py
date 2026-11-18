"""Elicit a schema-valid PSL belief document from an LLM, with a validate-and-retry
fallback so correctness never depends on server-side guided decoding."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple


class ElicitationError(RuntimeError):
    """Raised when no valid belief document could be elicited within the budget."""


@dataclass
class Elicitation:
    """A validated belief document plus the raw model text that produced it."""

    document: dict[str, Any]
    raw: str


# Semantic constraints the JSON Schema / guided decoding cannot enforce but the
# compiler (validate_record) does. Front-loading them greatly improves first-pass
# validity; the validate-and-retry loop still guarantees correctness.
HARD_RULES = (
    "## Hard rules (the compiler enforces these; any violation is rejected)\n"
    "- Output ONE JSON object with exactly the keys: schema, beliefs, trust.\n"
    "- Be CONCISE: summarize your belief with a FEW high-level shaping tokens "
    "(typically 3-8 total: e.g. one `bump`/`joint_region` for the most likely "
    "region, `trend`s, and the meta tokens `smoothness`/`noise`/`modality`). "
    "Do NOT enumerate an `anchor` for every grid point; do NOT restate the same "
    "belief many times. The whole document must stay under a few hundred lines.\n"
    "- beliefs: include AT MOST ONE `baseline` token.\n"
    "- schema.dimensions: continuous dims use `range` [low, high]; ordinal and "
    "categorical dims use `values` (the exact allowed values) and NEVER `range`.\n"
    "- Exactly ONE belief must have weight = 1.0; every other weight is in (0, 1].\n"
    "- Every non-meta belief needs a `provenance` (use `llm_domain`).\n"
    "- All belief coordinates must lie within their dimension's declared range/values.\n"
    "- Use ONLY the closed PSL vocabulary; never invent a token.\n"
)


def build_elicitation_prompt(
    prompt_card_text: str, vocabulary_card_text: str
) -> Tuple[str, str]:
    """Return ``(system, user)`` -- the vocabulary spec and the filled task card."""

    rules = HARD_RULES
    # HAPBO_PROVENANCE_RULE=card replaces the fixed "use llm_domain" instruction with the
    # vocabulary card's own provenance rule (used when the card states facts from several sources).
    if os.environ.get("HAPBO_PROVENANCE_RULE") == "card":
        fixed = "- Every non-meta belief needs a `provenance` (use `llm_domain`).\n"
        rules = rules.replace(fixed, "- Every non-meta belief needs a `provenance`: use the value that matches the "
                                     "belief's source, as defined in the vocabulary card.\n")
    return vocabulary_card_text, prompt_card_text + "\n\n" + rules


def _strip_to_json(text: str) -> str:
    """Extract the belief-document JSON object, tolerating a leaked thinking trace.

    Fast path: the span from the first ``{`` to the last ``}``.
    When a reasoning model (thinking mode) emits prose that itself contains braces
    before the final answer, that span is not parseable; in that case fall back to a
    brace-matched scan from the last ``}`` to recover the FINAL balanced object (the
    actual answer). The fallback only fires when the fast path fails to parse, so every
    well-formed response is extracted unchanged.
    """

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("model response did not contain a JSON object")
    candidate = text[start : end + 1]
    try:
        json.loads(candidate)
        return candidate  # fast path: clean or thinking-free response
    except ValueError:
        pass
    # Fallback: walk back from the last '}' with brace counting to find the matching '{'
    # of the last top-level object, then keep trying earlier opens until one parses.
    depth = 0
    for i in range(end, start - 1, -1):
        char = text[i]
        if char == "}":
            depth += 1
        elif char == "{":
            depth -= 1
            if depth == 0:
                span = text[i : end + 1]
                try:
                    json.loads(span)
                    return span
                except ValueError:
                    continue
    return candidate  # let the caller's json.loads raise its informative error


def elicit_belief_document(
    client: Any,
    prompt_card_text: str,
    *,
    schema: Mapping[str, Any],
    vocabulary_card_text: str,
    max_attempts: int = 4,
    seed: Optional[int] = None,
    elicitation_replicate: int = 0,
    use_structured_output: bool = True,
    card_human_statement: bool = False,
) -> Elicitation:
    """Elicit and validate a PSL belief document.

    Validation is two-layer: ``jsonschema`` (shape) then the compiler's
    ``validate_record`` + ``compile_prior`` (coordinates in range,
    exactly-one-weight-1.0, compilability).  On failure the exact error is fed
    back as a corrective turn; because that changes the prompt, each retry has a
    distinct cache key and is itself reproducible.

    ``card_human_statement`` says whether the card carries a human (expert)
    statement; unless it does, a belief tagged ``user_stated`` / ``user_confirmed``
    is rejected and corrected through the same loop.
    """

    import jsonschema  # lazy

    from prior.landscape.landscape_generator import compile_prior, validate_record

    system, base_user = build_elicitation_prompt(prompt_card_text, vocabulary_card_text)
    validator = jsonschema.Draft202012Validator(dict(schema))
    user = base_user
    last_error: Optional[Exception] = None
    # With use_structured_output, only the first attempt uses guided decoding over the PSL
    # oneOf schema; retries are free-form, which follows the prompt's conciseness rules
    # (few shaping tokens, one baseline) more closely. The validate-and-retry loop below
    # enforces correctness either way.
    use_structured = bool(use_structured_output)
    for attempt in range(max_attempts):
        raw = client.complete(
            user,
            system=system,
            response_schema=dict(schema) if use_structured else None,
            seed=seed,
            elicitation_replicate=elicitation_replicate,
        )
        note: Optional[str] = None
        try:
            document = json.loads(_strip_to_json(raw))  # ValueError / JSONDecodeError
        except ValueError as exc:  # malformed or truncated JSON
            last_error = exc
            use_structured = False  # fall back to free-form
            note = (
                "Your previous JSON was incomplete or malformed (likely cut off "
                "because it was too long). Produce a COMPLETE, MORE CONCISE belief "
                "document with far fewer tokens (only a handful of shaping tokens)."
            )
        else:
            try:
                validator.validate(document)  # syntactic
                validate_record(  # semantic
                    document, card_human_statement=card_human_statement
                )
                compile_prior(document)  # compilable
                return Elicitation(document=document, raw=raw)
            except Exception as exc:  # noqa: BLE001 -- semantic/compile failure
                last_error = exc
                use_structured = False  # free-form follows the hard rules better
                note = (
                    f"Validator error: {exc}\nReturn ONLY a corrected JSON belief "
                    "document that fixes this exact error and satisfies every hard rule."
                )
        user = (
            base_user
            + f"\n\n## Your previous attempt (#{attempt + 1}) was REJECTED\n"
            + note
            + "\nReturn ONLY the JSON object, no prose."
        )
    raise ElicitationError(
        f"no valid belief document after {max_attempts} attempts: {last_error}"
    )


__all__ = [
    "Elicitation",
    "ElicitationError",
    "HARD_RULES",
    "build_elicitation_prompt",
    "elicit_belief_document",
]
