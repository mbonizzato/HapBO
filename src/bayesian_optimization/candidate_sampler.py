"""Selection helpers for a finite set of BO candidates."""

from __future__ import annotations

import numpy as np


class CandidateSampler:
    """Select random or highest-scoring candidates without replacement."""

    def __init__(self, candidates: np.ndarray, random_state: int | None = None) -> None:
        candidates = np.asarray(candidates, dtype=float)
        if candidates.ndim == 1:
            candidates = candidates.reshape(-1, 1)
        if candidates.ndim != 2 or candidates.shape[0] == 0:
            raise ValueError("candidates must be a non-empty 2D array")
        if not np.all(np.isfinite(candidates)):
            raise ValueError("candidates must contain only finite values")
        self.candidates = candidates
        self.rng = np.random.default_rng(random_state)

    def _available_indices(self, queried_mask: np.ndarray) -> np.ndarray:
        queried_mask = np.asarray(queried_mask, dtype=bool).reshape(-1)
        if queried_mask.size != self.candidates.shape[0]:
            raise ValueError("queried_mask must have one entry per candidate")
        available = np.flatnonzero(~queried_mask)
        if available.size == 0:
            raise RuntimeError("All candidates have already been evaluated")
        return available

    def random(self, queried_mask: np.ndarray) -> int:
        """Return a random unqueried candidate index."""
        available = self._available_indices(queried_mask)
        return int(self.rng.choice(available))

    def best(self, scores: np.ndarray, queried_mask: np.ndarray) -> int:
        """Return the unqueried candidate with the largest finite score."""
        scores = np.asarray(scores, dtype=float).reshape(-1)
        if scores.size != self.candidates.shape[0]:
            raise ValueError("scores must have one entry per candidate")
        available = self._available_indices(queried_mask)
        available_scores = scores[available]
        if not np.any(np.isfinite(available_scores)):
            return self.random(queried_mask)
        safe_scores = np.where(np.isfinite(available_scores), available_scores, -np.inf)
        return int(available[np.argmax(safe_scores)])
