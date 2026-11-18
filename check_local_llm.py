"""Smoke-test the OpenAI-compatible LLM endpoint at LLM_BASE_URL used for elicitation.

Runs three probes: a trivial completion, a tiny structured-output request, and
the full PSL schema (to surface any oneOf / guided-decoding limitation early).
If the endpoint is unreachable it says which variables configure it.

    python check_local_llm.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(__file__).resolve().parent / ".env")


def _guidance(base_url: str) -> str:
    return (
        f"\nCannot reach the LLM endpoint at {base_url}.\n"
        "Start an OpenAI-compatible server (e.g. vLLM or SGLang) or set LLM_BASE_URL,\n"
        "LLM_MODEL and LLM_API_KEY, then re-run: python check_local_llm.py\n"
    )


def main() -> int:
    _load_env()
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
    model = os.getenv("LLM_MODEL", "Qwen3.5-122B-A10B-GPTQ-Int4")
    api_key = os.getenv("LLM_API_KEY", "none")

    try:
        from openai import OpenAI
    except ImportError:
        print("openai is not installed in this environment (pip install openai).")
        return 2

    client = OpenAI(base_url=base_url, api_key=api_key)
    thinking = {"chat_template_kwargs": {"enable_thinking": False}}
    print(f"endpoint: {base_url}\nmodel:    {model}")

    # 1) trivial completion
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
            max_tokens=16,
            temperature=0.0,
            extra_body=thinking,
        )
    except Exception as exc:  # noqa: BLE001
        if type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}:
            print(_guidance(base_url))
            return 1
        print(f"trivial completion failed: {type(exc).__name__}: {exc}")
        return 1
    print(f"[1/3] trivial completion OK -> {r.choices[0].message.content!r}")

    # 2) tiny structured output
    tiny = {
        "type": "object",
        "additionalProperties": False,
        "required": ["ok"],
        "properties": {"ok": {"type": "boolean"}},
    }
    guided_available = True
    try:
        r2 = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": 'Return JSON {"ok": true}'}],
            max_tokens=64,
            temperature=0.0,
            response_format={"type": "json_schema", "json_schema": {"name": "probe", "schema": tiny}},
            extra_body=thinking,
        )
        print(f"[2/3] structured (tiny) OK -> {r2.choices[0].message.content!r}")
    except Exception as exc:  # noqa: BLE001
        guided_available = False
        print(f"[2/3] structured (tiny) NOT available: {type(exc).__name__}: {exc}")

    # 3) full PSL schema
    try:
        from prior.expert_knowledge.psl_schema import load_psl_schema

        schema = load_psl_schema()
        r3 = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Return a minimal valid PSL belief document."}],
            max_tokens=512,
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "hapbo_psl_belief_document", "schema": schema},
            },
            extra_body=thinking,
        )
        print(f"[3/3] structured (full PSL) OK ({len(r3.choices[0].message.content or '')} chars)")
    except Exception as exc:  # noqa: BLE001
        guided_available = False
        print(f"[3/3] structured (full PSL) NOT available: {type(exc).__name__}: {exc}")

    if guided_available:
        print(
            "\nGuided decoding is available. NOTE: for the full PSL schema this model "
            "tends to over-generate under guided decoding, so elicitation defaults to "
            "structured-first then falls back to free-form; pass --no-structured to "
            "elicit_prior.py to use free-form directly (recommended for this model)."
        )
    else:
        print(
            "\nGuided decoding is unavailable on this server; elicitation uses free-form "
            "generation + the validate-and-retry loop (still correct)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
