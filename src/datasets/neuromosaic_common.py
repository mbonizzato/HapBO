"""Shared finite-problem types used by all NeuroMOSAIC adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class OfflineOptimizationProblem:
    """A finite candidate set backed by a fully observed offline response map."""

    dataset_id: str
    subject_id: str
    target_name: str
    native_candidates: tuple[dict[str, int], ...]
    encoded_candidates: np.ndarray
    ground_truth: np.ndarray
    response_standard_deviation: np.ndarray
    objective: Callable[[Mapping[str, Any]], float]
    metadata: dict[str, Any]
    maximize: bool = True

    @property
    def n_candidates(self) -> int:
        return len(self.native_candidates)

    @property
    def parameter_names(self) -> list[str]:
        """Ordered dimension names, taken from the first native candidate.

        Used by the study worker to encode non-neuromosaic candidates
        (``_encode_mixed_candidates``); every candidate shares the same keys.
        """

        return list(self.native_candidates[0].keys()) if self.native_candidates else []

    @property
    def optimum_index(self) -> int:
        if self.maximize:
            return int(np.argmax(self.ground_truth))
        return int(np.argmin(self.ground_truth))

    @property
    def optimum_candidate(self) -> dict[str, int]:
        return dict(self.native_candidates[self.optimum_index])

    @property
    def optimum_value(self) -> float:
        return float(self.ground_truth[self.optimum_index])


__all__ = ["OfflineOptimizationProblem"]
