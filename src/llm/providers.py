"""LLM provider strategies used by :class:`LLMClient`.

The default provider is OpenAI-compatible: a locally served OpenAI-compatible endpoint
(e.g. SGLang/vLLM). An Anthropic provider is also available. Each provider is imported
lazily so importing :class:`LLMClient` never requires an SDK.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass
class Completion:
    """One provider response plus usage metadata."""

    text: str
    tokens: int = 0
    cost: float = 0.0


_STRUCTURED_NAME = "belief_document"

# First-party list prices ($ per million input / output tokens); used only to log the cost of each call.
# Haiku 5.5: rate for prompts up to 100K tokens (elicitation prompts are far below).
ANTHROPIC_PRICE_PER_MTOK = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50),
}


class OpenAIProvider:
    """OpenAI-compatible chat completions (SGLang / vLLM / OpenAI)."""

    def __init__(self, base_url: Optional[str], api_key: Optional[str]) -> None:
        self.base_url = base_url
        self.api_key = api_key or "none"
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI  # lazy import

            kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        response_schema: Optional[Mapping[str, Any]] = None,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        enable_thinking: bool = False,
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> Completion:
        import openai

        client = self._ensure_client()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        base_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            # Qwen thinking toggle passes through SGLang's extra_body.
            "extra_body": {"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}},
        }
        if top_p is not None:
            base_kwargs["top_p"] = top_p
        if seed is not None:
            base_kwargs["seed"] = seed

        def _call(use_schema: bool) -> Completion:
            kwargs = dict(base_kwargs)
            if use_schema and response_schema is not None:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": _STRUCTURED_NAME, "schema": dict(response_schema)},
                }
            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0
            return Completion(text=text, tokens=tokens)

        transient = tuple(
            error
            for error in (
                getattr(openai, "APIConnectionError", None),
                getattr(openai, "APITimeoutError", None),
                getattr(openai, "RateLimitError", None),
                getattr(openai, "InternalServerError", None),
            )
            if error is not None
        )
        schema_rejected = tuple(
            error
            for error in (
                getattr(openai, "BadRequestError", None),
                getattr(openai, "UnprocessableEntityError", None),
            )
            if error is not None
        )

        use_schema = response_schema is not None
        last_error: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                return _call(use_schema)
            except schema_rejected as exc:  # guided decoding unavailable -> fall back
                if use_schema:
                    use_schema = False
                    last_error = exc
                    continue
                raise
            except transient as exc:
                last_error = exc
                time.sleep(min(2.0 * (attempt + 1), 5.0))
        raise RuntimeError(f"OpenAI-compatible provider failed after retries: {last_error}")


class AnthropicProvider:
    """Anthropic Messages API."""

    def __init__(self, api_key: Optional[str]) -> None:
        self.api_key = api_key
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic  # lazy import

            self._client = (
                anthropic.Anthropic(api_key=self.api_key)
                if self.api_key
                else anthropic.Anthropic()
            )
        return self._client

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        response_schema: Optional[Mapping[str, Any]] = None,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,  # unused; kept for a uniform interface
        enable_thinking: bool = False,  # unused
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> Completion:
        import anthropic

        client = self._ensure_client()
        system_text = system or ""
        if response_schema is not None:
            system_text = (
                system_text
                + "\n\nReturn a single JSON object that validates against this JSON Schema:\n"
                + json.dumps(dict(response_schema))
            ).strip()

        def _create(include_temperature: bool):
            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "system": system_text or None,
                "messages": [{"role": "user", "content": prompt}],
            }
            if include_temperature:
                kwargs["temperature"] = temperature
            if top_p is not None:
                kwargs["top_p"] = top_p
            if max_tokens > 16000:
                # large output budgets (thinking models): the SDK requires streaming; same final Message
                with client.messages.stream(**kwargs) as stream:
                    return stream.get_final_message()
            return client.messages.create(**kwargs)

        # Some Claude models reject the `temperature` control with a 400; the call is then
        # retried once without it, and the model samples at its default temperature.
        transient = tuple(
            error
            for error in (
                getattr(anthropic, "APIConnectionError", None),
                getattr(anthropic, "APITimeoutError", None),
                getattr(anthropic, "RateLimitError", None),
                getattr(anthropic, "InternalServerError", None),
            )
            if error is not None
        )
        include_temperature = True
        last_error: Optional[Exception] = None
        response = None
        for attempt in range(max_retries + 2):
            try:
                response = _create(include_temperature)
                break
            except transient as exc:
                last_error = exc
                time.sleep(min(2.0 * (attempt + 1), 5.0))
            except Exception as exc:  # noqa: BLE001 -- inspect message for the temperature 400
                if include_temperature and "temperature" in str(exc).lower():
                    include_temperature = False
                    last_error = exc
                    continue
                raise
        if response is None:
            raise RuntimeError(f"Anthropic provider failed after retries: {last_error}")
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        usage = getattr(response, "usage", None)
        n_in = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
        n_out = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0  # includes thinking tokens
        price = ANTHROPIC_PRICE_PER_MTOK.get(model)
        cost = (n_in * price[0] + n_out * price[1]) / 1e6 if price else 0.0
        return Completion(text=text, tokens=n_in + n_out, cost=cost)


def build_provider(name: str, *, base_url: Optional[str], api_key: Optional[str]):
    if name == "openai":
        return OpenAIProvider(base_url, api_key)
    if name == "anthropic":
        return AnthropicProvider(api_key)
    raise ValueError(f"unknown LLM provider {name!r}")


__all__ = ["Completion", "OpenAIProvider", "AnthropicProvider", "build_provider"]
