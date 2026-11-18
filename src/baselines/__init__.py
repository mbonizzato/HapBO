"""Registry of the baseline optimizers.

Adapter modules are safe to import because they never import their optional
upstream repositories at module import time.
"""

from __future__ import annotations

from typing import Any

from .base import BaselineOptimizer
from .bora import BORA
from .caliper import CALIPER
from .colabo_adapter import ColaBOAdapter
from .hypbo import HypBO
from .lgbo import LGBO
from .llambo_adapter import LLAMBOAdapter
from .official_lgbo import OfficialLGBO
from .random_search import RandomSearch
from .vanilla_bo import VanillaBO

try:  # CAKE needs torch/botorch/gpytorch; it is unavailable where that stack is not installed.
    from .cake import CAKE
except Exception:  # pragma: no cover - optional heavy dependency
    CAKE = None


BASELINE_REGISTRY: dict[str, type[BaselineOptimizer]] = {
    "random": RandomSearch,
    "vanilla_bo": VanillaBO,
    "llambo": LLAMBOAdapter,
    "colabo": ColaBOAdapter,
    "bora": BORA,
    "hypbo": HypBO,
    "caliper": CALIPER,
    "lgbo": LGBO,
    "official_lgbo": OfficialLGBO,
}

if CAKE is not None:  # registered only where the GP stack is installed
    BASELINE_REGISTRY["cake"] = CAKE


def get_baseline(name: str, **kwargs: Any) -> BaselineOptimizer:
    """Instantiate a registered baseline or report every accepted key."""

    try:
        baseline_class = BASELINE_REGISTRY[name]
    except KeyError as exc:
        valid = ", ".join(BASELINE_REGISTRY)
        raise ValueError(f"unknown baseline {name!r}; valid baselines: {valid}") from exc
    return baseline_class(**kwargs)


__all__ = ["BASELINE_REGISTRY", "BaselineOptimizer", "get_baseline"]
