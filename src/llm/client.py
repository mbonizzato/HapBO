"""Provider-neutral LLM client with reproducibility controls (per-run cache, call budget, usage log).

Used by the HapBO prior elicitor, the LLM-EI survey, the NePS point-prior and CALIPER
elicitations, ``scripts/make_directllm_priors.py`` and the CAKE baseline.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LLMClient:
    """One run's model settings, cache, call budget, and usage log."""

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        cache_dir: str | Path,
        budget: int,
        *,
        provider: str = "openai",
        base_url: str | None = None,
        api_key: str | None = None,
        enable_thinking: bool = False,
        top_p: float | None = None,
        request_timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if budget < 0:
            raise ValueError("budget must be non-negative")
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.budget = int(budget)
        self._call_count = 0
        self._log_path = self.cache_dir / "llm_calls.jsonl"
        # Provider / deployment settings. The api_key is NEVER hashed or logged.
        self.provider_name = str(provider)
        self.base_url = base_url
        self._api_key = api_key
        self.enable_thinking = bool(enable_thinking)
        self.top_p = top_p
        self.request_timeout = float(request_timeout)
        self.max_retries = int(max_retries)
        self._provider = None

    @property
    def call_count(self) -> int:
        """Return the number of uncached provider calls reserved this run."""

        return self._call_count

    def set_run_directory(self, directory: str | Path) -> Path:
        """Place subsequent JSONL usage records beside one run's results."""

        run_directory = Path(directory).expanduser().resolve()
        run_directory.mkdir(parents=True, exist_ok=True)
        self._log_path = run_directory / "llm_calls.jsonl"
        return self._log_path

    def _request_document(self, prompt: str, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt": prompt,
            "params": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **params,
            },
        }

    def prompt_hash(self, prompt: str, **params: Any) -> str:
        """Hash model, prompt, and canonical request parameters."""

        payload = json.dumps(
            self._request_document(prompt, params),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _cache_path(self, prompt_hash: str) -> Path:
        return self.cache_dir / f"{prompt_hash}.json"

    def _cached_response(self, prompt_hash: str) -> dict[str, Any] | None:
        path = self._cache_path(prompt_hash)
        if not path.is_file():
            return None
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("prompt_hash") != prompt_hash or "response" not in document:
            raise ValueError(f"invalid LLM cache entry: {path}")
        return document

    def _store_response(
        self,
        prompt_hash: str,
        response: str,
        *,
        tokens: int = 0,
        cost: float = 0.0,
    ) -> Path:
        """Atomically store one provider response and its usage metadata."""

        document = {
            "artifact_format": "llm-bo-llm-cache-v1",
            "prompt_hash": prompt_hash,
            "model": self.model,
            "response": response,
            "tokens": int(tokens),
            "cost": float(cost),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        destination = self._cache_path(prompt_hash)
        temporary = destination.with_suffix(".json.part")
        temporary.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    def _log_call(
        self,
        prompt_hash: str,
        *,
        tokens: int,
        cost: float,
        cached: bool,
        status: str,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt_hash": prompt_hash,
            "model": self.model,
            "tokens": int(tokens),
            "cost": float(cost),
            "cached": bool(cached),
            "status": status,
        }
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _get_provider(self):
        if self._provider is None:
            from .providers import build_provider

            self._provider = build_provider(
                self.provider_name, base_url=self.base_url, api_key=self._api_key
            )
        return self._provider

    def _identity(
        self,
        *,
        system: str | None,
        response_schema: dict[str, Any] | None,
        seed: int | None,
        elicitation_replicate: int,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        """Full cache identity (everything that changes the sampled output).

        Excludes the api_key. Includes the deployment (provider + complete
        canonical base_url incl. host/port/path), decoding parameters, the
        system prompt, the schema hash, the sampling seed, and the elicitation
        replicate so a re-elicitation never collapses onto a stale cache entry.
        """

        schema_hash = None
        if response_schema is not None:
            schema_hash = hashlib.sha256(
                json.dumps(response_schema, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
        identity: dict[str, Any] = {
            "provider": self.provider_name,
            "base_url": self.base_url,
            "enable_thinking": self.enable_thinking,
            "top_p": self.top_p,
            "seed": seed,
            "elicitation_replicate": int(elicitation_replicate),
            "system": system,
            "response_format": response_schema is not None,
            "response_schema_sha256": schema_hash,
        }
        identity.update(extra)
        return identity

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        response_schema: dict[str, Any] | None = None,
        seed: int | None = None,
        elicitation_replicate: int = 0,
        **kwargs: Any,
    ) -> str:
        """Return a cached response or execute exactly one provider call.

        Order: cache lookup -> budget check -> provider call -> _store_response ->
        _log_call; failed calls are logged (status ``error:...``) and never cached.
        """

        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty string")
        identity = self._identity(
            system=system,
            response_schema=response_schema,
            seed=seed,
            elicitation_replicate=elicitation_replicate,
            extra=dict(kwargs),
        )
        prompt_hash = self.prompt_hash(prompt, **identity)
        cached = self._cached_response(prompt_hash)
        if cached is not None:
            self._log_call(
                prompt_hash,
                tokens=int(cached.get("tokens", 0)),
                cost=float(cached.get("cost", 0.0)),
                cached=True,
                status="cache_hit",
            )
            return str(cached["response"])
        if self._call_count >= self.budget:
            raise RuntimeError(
                f"LLM call budget exhausted ({self._call_count}/{self.budget})"
            )
        self._call_count += 1
        try:
            completion = self._get_provider().generate(
                prompt,
                system=system,
                response_schema=response_schema,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                seed=seed,
                enable_thinking=self.enable_thinking,
                timeout=self.request_timeout,
                max_retries=self.max_retries,
            )
        except Exception as exc:  # noqa: BLE001 -- log then re-raise, never cache
            self._log_call(
                prompt_hash, tokens=0, cost=0.0, cached=False, status=f"error:{type(exc).__name__}"
            )
            raise
        self._store_response(
            prompt_hash, completion.text, tokens=completion.tokens, cost=completion.cost
        )
        self._log_call(
            prompt_hash,
            tokens=completion.tokens,
            cost=completion.cost,
            cached=False,
            status="ok",
        )
        return completion.text


__all__ = ["LLMClient"]
