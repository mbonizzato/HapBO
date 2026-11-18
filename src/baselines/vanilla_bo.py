"""Prior-free wrapper around the repository's existing BO engine."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.bayesian_optimization.bo import BayesianOptimization

from .base import BaselineOptimizer, Config, Observation, config_key


class VanillaBO(BaselineOptimizer):
    """Reuse :class:`BayesianOptimization` with alpha zero and no prior."""

    name = "vanilla_bo"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        n_initial = int(self.kwargs.get("n_initial", 1))
        if not 1 <= n_initial <= self.budget:
            raise ValueError("n_initial must be between 1 and budget")
        maximize = bool(self.objective_meta.get("maximize", True))
        n_constraints = int(self.objective_meta.get("n_constraints", 0))
        self._optimizer = BayesianOptimization(
            self.encoded_candidates,
            length_scale=float(self.kwargs.get("length_scale", 1.0)),
            kappa=float(self.kwargs.get("kappa", 2.0)),
            alpha=0.0,
            prior=None,
            n_initial=n_initial,
            noise_variance=float(self.kwargs.get("noise_variance", 1e-6)),
            signal_variance=float(self.kwargs.get("signal_variance", 1.0)),
            normalize_y=bool(self.kwargs.get("normalize_y", False)),
            maximize=maximize,
            random_state=self.seed,
            acquisition=str(self.kwargs.get("acquisition", "expected-improvement")),
            improvement_xi=float(self.kwargs.get("improvement_xi", 0.0)),
            n_constraints=n_constraints,
        )
        self._asked_indices: dict[str, int] = {}

    def ask(self) -> Config:
        index, _ = self._optimizer.ask()
        config = dict(self.candidates[index])
        key = config_key(config)
        self._pending.add(key)
        self._asked_indices[key] = index
        return config

    def tell(self, config: Mapping[str, Any], observation: Observation) -> None:
        key = config_key(config)
        index = self._asked_indices.pop(key, self._index_for(config))
        if isinstance(observation, Mapping):
            value = float(observation["objective"])
            constraints = np.asarray(observation.get("constraints", ()), dtype=float)
        else:
            value = float(observation)
            constraints = None
        self._optimizer.tell(index, value, constraints)
        self._record(config, observation)


__all__ = ["VanillaBO"]
