"""Uniform random search over the repository's finite candidate sets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import BaselineOptimizer, Config, Observation


class RandomSearch(BaselineOptimizer):
    """Sample uniformly from configurations that have not been evaluated."""

    name = "random"

    def ask(self) -> Config:
        return self._random_unqueried_config()

    def tell(self, config: Mapping[str, Any], observation: Observation) -> None:
        self._record(config, observation)


__all__ = ["RandomSearch"]
