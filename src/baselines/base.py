"""Shared ask/tell contract for baseline optimizers."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import numpy as np


Config = dict[str, Any]
Observation = float | Mapping[str, Any]


def _native_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def config_key(config: Mapping[str, Any]) -> str:
    """Return a stable identity for a native configuration dictionary."""

    normalized = {str(key): _native_value(value) for key, value in config.items()}
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)


def _encode_candidates(candidates: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Create deterministic numeric rows when an adapter does not supply them."""

    names = tuple(candidates[0])
    blocks: list[np.ndarray] = []
    for name in names:
        values = [_native_value(candidate[name]) for candidate in candidates]
        numeric = all(
            isinstance(value, (int, float, bool)) and not isinstance(value, complex)
            for value in values
        )
        if numeric:
            column = np.asarray(values, dtype=float)
            span = float(np.ptp(column))
            blocks.append(
                ((column - float(np.min(column))) / span if span else np.zeros_like(column))
                .reshape(-1, 1)
            )
            continue
        categories = tuple(dict.fromkeys(values))
        category_index = {value: index for index, value in enumerate(categories)}
        indices = np.asarray([category_index[value] for value in values], dtype=int)
        blocks.append(np.eye(len(categories), dtype=float)[indices])
    return np.column_stack(blocks)


class BaselineOptimizer(ABC):
    """Contract shared by every baseline optimizer.

    ``ask`` proposes one native configuration dictionary and ``tell`` records
    the observation returned for that exact configuration. The EVALUATE step is
    always owned by the outer experiment loop, never by the baseline. This
    guarantees that every method sees identical objective calls, budgets, and
    seeds. ``search_space`` follows the repository's finite-dataset convention:
    it contains ``candidates`` and may contain aligned ``encoded_candidates``
    for numeric GP fitting.
    """

    name = "baseline"

    def __init__(
        self,
        search_space: Mapping[str, Any],
        objective_meta: Mapping[str, Any],
        prior: Any = None,
        llm_client: Any = None,
        seed: int | None = None,
        budget: int | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            raw_candidates = search_space["candidates"]
        except (KeyError, TypeError) as exc:
            raise ValueError("search_space must contain finite 'candidates'") from exc
        if not isinstance(raw_candidates, Sequence) or not raw_candidates:
            raise ValueError("search_space candidates must be a non-empty sequence")
        if not all(isinstance(candidate, Mapping) for candidate in raw_candidates):
            raise ValueError("every baseline candidate must be a configuration mapping")

        self.candidates = tuple(
            {str(key): _native_value(value) for key, value in candidate.items()}
            for candidate in raw_candidates
        )
        expected_names = tuple(self.candidates[0])
        if any(tuple(candidate) != expected_names for candidate in self.candidates):
            raise ValueError("all candidate dictionaries must use the same key order")
        keys = tuple(config_key(candidate) for candidate in self.candidates)
        if len(set(keys)) != len(keys):
            raise ValueError("search_space candidates must be unique")

        encoded = search_space.get("encoded_candidates")
        if encoded is None:
            encoded = _encode_candidates(self.candidates)
        encoded_array = np.asarray(encoded, dtype=float)
        if encoded_array.ndim == 1:
            encoded_array = encoded_array.reshape(-1, 1)
        if encoded_array.ndim != 2 or encoded_array.shape[0] != len(self.candidates):
            raise ValueError(
                "encoded_candidates must be a 2D array aligned with candidates"
            )
        if not np.all(np.isfinite(encoded_array)):
            raise ValueError("encoded_candidates must contain only finite values")

        if budget is None:
            budget = len(self.candidates)
        if not 1 <= int(budget) <= len(self.candidates):
            raise ValueError("budget must be between 1 and the candidate count")

        self.search_space = dict(search_space)
        self.objective_meta = dict(objective_meta)
        self.prior = prior
        self.llm_client = llm_client
        self.seed = seed
        self.budget = int(budget)
        self.kwargs = dict(kwargs)
        self.encoded_candidates = encoded_array.copy()
        self._candidate_index = {key: index for index, key in enumerate(keys)}
        self._rng = np.random.default_rng(seed)
        self._history: list[tuple[Config, Observation]] = []
        self._evaluated: set[str] = set()
        self._pending: set[str] = set()

    @abstractmethod
    def ask(self) -> Config:
        """Propose the next unevaluated native configuration."""

    @abstractmethod
    def tell(self, config: Mapping[str, Any], observation: Observation) -> None:
        """Record an externally evaluated configuration and observation."""

    def warmstart(self, n_init: int) -> list[Config]:
        """Propose ``n_init`` uniform configurations without evaluating them."""

        if not 0 <= int(n_init) <= self.budget - len(self._history):
            raise ValueError("n_init exceeds the remaining evaluation budget")
        return [self._random_unqueried_config() for _ in range(int(n_init))]

    @property
    def history(self) -> tuple[tuple[Config, Observation], ...]:
        """Return defensive copies of evaluated configuration/observation pairs."""

        return tuple(
            (dict(config), deepcopy(observation))
            for config, observation in self._history
        )

    def _index_for(self, config: Mapping[str, Any]) -> int:
        try:
            return self._candidate_index[config_key(config)]
        except KeyError as exc:
            raise ValueError("configuration is not in the finite search space") from exc

    def _random_unqueried_config(self) -> Config:
        if len(self._history) + len(self._pending) >= self.budget:
            raise RuntimeError("the baseline evaluation budget is exhausted")
        available = [
            index
            for index, candidate in enumerate(self.candidates)
            if config_key(candidate) not in self._evaluated | self._pending
        ]
        if not available:
            raise RuntimeError("all candidates have already been proposed")
        index = int(self._rng.choice(np.asarray(available, dtype=int)))
        config = dict(self.candidates[index])
        self._pending.add(config_key(config))
        return config

    def _record(self, config: Mapping[str, Any], observation: Observation) -> None:
        key = config_key(config)
        self._index_for(config)
        if key in self._evaluated:
            raise ValueError("configuration has already been evaluated")
        if len(self._history) >= self.budget:
            raise RuntimeError("the baseline evaluation budget is exhausted")
        if isinstance(observation, Mapping):
            if "objective" not in observation:
                raise ValueError("mapping observations must contain 'objective'")
            value = float(observation["objective"])
            if not np.isfinite(value):
                raise ValueError("observation objective must be finite")
            stored: Observation = deepcopy(dict(observation))
        else:
            value = float(observation)
            if not np.isfinite(value):
                raise ValueError("observation must be finite")
            stored = value
        self._pending.discard(key)
        self._evaluated.add(key)
        self._history.append((dict(self.candidates[self._index_for(config)]), stored))


__all__ = [
    "BaselineOptimizer",
    "Config",
    "Observation",
    "config_key",
]
