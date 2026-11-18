"""Shared runtime glue for the standalone BORA drivers (scripts/bora_continuous.py,
scripts/bora_solar_cached.py).

The pinned BORA checkout targets OpenAI's hosted models. The drivers run the same BORA
implementation against a locally served OpenAI-compatible endpoint (LLM_BASE_URL, LLM_MODEL).
:func:`configure_qwen` applies the client shims listed in external/PINNED_COMMITS.txt (BORA
entry):

* the served model is registered in BORA's ``MODEL_INFOS`` (zero prices, 32k tokens) with a
  ``cl100k_base`` tokenizer fallback;
* BORA's captured OpenAI client is pointed at the endpoint (300 s timeout);
* every chat call disables thinking, clamps n>1 sampling to n=1, and sends any system
  message after the first position with the user role (the chat template accepts a system
  message only at the start).

BORA's optimizer, policy and acquisition function are unchanged, and so is the text of every
prompt; only the role of non-leading system messages and the sample count change.
"""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path


TOKEN_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
_PATCHED = False


def configure_qwen(*, llm_model: str, base_url: str, api_key: str) -> None:
    """Point BORA's captured OpenAI client at the endpoint and apply the shims of the module docstring."""

    global _PATCHED
    if _PATCHED:
        return

    from bora import assistant
    import tiktoken

    assistant.MODEL_INFOS.setdefault(
        llm_model,
        {
            "prompt_tokens_price_per_1k": 0.0,
            "completion_tokens_price_per_1k": 0.0,
            "max_tokens": 32_000,
        },
    )
    original_get_model_info = assistant.Assistant._get_model_info

    def patched_get_model_info(self):
        if self._model in assistant.MODEL_INFOS:
            info = dict(assistant.MODEL_INFOS[self._model])
            try:
                info["encoding"] = tiktoken.encoding_for_model(self._model)
            except Exception:
                info["encoding"] = tiktoken.get_encoding("cl100k_base")
            return info
        return original_get_model_info(self)

    assistant.Assistant._get_model_info = patched_get_model_info
    original_openai = assistant.OpenAI

    def patched_openai(*args, **kwargs):
        kwargs.setdefault("base_url", base_url)
        kwargs.setdefault("timeout", 300.0)
        client = original_openai(*args, **kwargs)
        original_create = client.chat.completions.create

        def create(*call_args, **call_kwargs):
            extra = dict(call_kwargs.get("extra_body") or {})
            template = dict(extra.get("chat_template_kwargs") or {})
            template.setdefault("enable_thinking", False)
            extra["chat_template_kwargs"] = template
            call_kwargs["extra_body"] = extra
            if int(call_kwargs.get("n", 1) or 1) > 1:
                call_kwargs["n"] = 1
            messages = call_kwargs.get("messages")
            if messages:
                call_kwargs["messages"] = [
                    ({**message, "role": "user"}
                     if position > 0 and message.get("role") == "system" else message)
                    for position, message in enumerate(messages)
                ]
            response = original_create(*call_args, **call_kwargs)
            usage = getattr(response, "usage", None)
            if usage is not None:
                TOKEN_USAGE["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
                TOKEN_USAGE["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
                TOKEN_USAGE["total_tokens"] += int(getattr(usage, "total_tokens", 0) or 0)
                TOKEN_USAGE["calls"] += 1
            return response

        client.chat.completions.create = create
        return client

    assistant.OpenAI = patched_openai
    _PATCHED = True


def install_token_dump(path: str | Path | None) -> None:
    """Write aggregate token usage at normal interpreter shutdown."""

    if path is None:
        return
    target = Path(path)

    def dump() -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(TOKEN_USAGE, indent=2), encoding="utf-8")

    atexit.register(dump)


def runtime_settings() -> tuple[str, str, str]:
    """Return ``(model, base_url, api_key)`` using the project environment names."""

    model = os.environ.get("LLM_MODEL", "Qwen3.5-122B-A10B-GPTQ-Int4")
    base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("LLM_API_KEY", "none")
    os.environ.setdefault("OPENAI_API_KEY", api_key)
    os.environ.setdefault("OPENAI_BASE_URL", base_url)
    os.environ.setdefault("OPENAI_API_TYPE", "open_ai")
    os.environ.setdefault("OPENAI_API_VERSION", "")
    return model, base_url, api_key

