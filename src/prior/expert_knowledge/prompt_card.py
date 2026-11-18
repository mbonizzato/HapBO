"""Render a filled elicitation prompt card from a problem description.

The card is the *task* handed to the LLM (the vocabulary card is the language).
It is committed alongside the elicited prior and hash-tracked, so it must be a
deterministic function of its inputs.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def render_prompt_card(
    *,
    domain: str,
    search_space: str,
    optimization_description: str,
) -> str:
    """Return a filled prompt card (Prompt A -- LLM as an expert)."""

    return (
        "# Prompt A -- LLM as an expert.\n\n"
        "## 0. Prompt\n"
        "You are an expert in the domain described below. Using your knowledge of this\n"
        "field, generate a prior belief over the location of the optimum for the\n"
        "Bayesian optimization problem defined by the search space and optimization\n"
        "description below. Write the belief only in the closed PSL vocabulary you were\n"
        "given; never invent tokens.\n\n"
        "## 1. Domain\n"
        f"{domain.strip()}\n\n"
        "## 2. Search space\n"
        f"{search_space.strip()}\n\n"
        "## 3. Optimization description\n"
        f"{optimization_description.strip()}\n"
    )


def search_space_from_schema(dimensions: Sequence[Mapping[str, Any]]) -> str:
    """Render PSL/benchmark schema dimensions into the template's parameter blocks."""

    blocks: list[str] = []
    for index, dimension in enumerate(dimensions, start=1):
        dimension_type = dimension.get("type", "continuous")
        lines = [f"Parameter {index}", f"name: {dimension['name']}", f"type: {dimension_type}"]
        # Match the PSL schema: continuous dims use `range` [low, high]; ordinal
        # and categorical dims use the exact `values` list (never `range`).
        if dimension_type == "continuous":
            if dimension.get("range") is not None:
                lines.append(f"range: {list(dimension['range'])}  # [low, high]")
        else:
            if dimension.get("values") is not None:
                lines.append(f"values: {list(dimension['values'])}  # exact allowed values")
        if dimension_type == "continuous":
            lines.append(f"scale: {dimension.get('scale', 'linear')}")
        meaning = dimension.get("meaning") or dimension.get("role")
        if dimension_type != "continuous" and dimension.get("scale") == "log":
            meaning = (meaning + "; " if meaning else "") + "values are log-spaced"
        if meaning:
            lines.append(f"meaning: {meaning}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


__all__ = ["render_prompt_card", "search_space_from_schema"]
