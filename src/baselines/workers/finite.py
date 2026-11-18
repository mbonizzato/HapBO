"""Finite-domain helpers shared by isolated upstream baseline workers."""

from __future__ import annotations

import numpy as np


def normalize_rows(encoded) -> np.ndarray:
    """Normalize each encoded feature to [0, 1], preserving constant columns."""

    values = np.asarray(encoded, dtype=float)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("encoded candidates must be a non-empty 2D matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("encoded candidates must be finite")
    lower = values.min(axis=0)
    span = values.max(axis=0) - lower
    safe_span = np.where(span > 0.0, span, 1.0)
    normalized = (values - lower) / safe_span
    normalized[:, span == 0.0] = 0.0
    return normalized


def nearest_index(encoded: np.ndarray, point) -> int:
    """Return the deterministic nearest finite row in Euclidean encoded space."""

    values = np.asarray(encoded, dtype=float)
    query = np.asarray(point, dtype=float).reshape(-1)
    if values.ndim != 2 or query.shape != (values.shape[1],):
        raise ValueError("proposal dimension does not match encoded candidates")
    distances = np.square(values - query).sum(axis=1)
    return int(np.argmin(distances))


def seeded_initial_indices(n_candidates: int, count: int, seed: int) -> list[int]:
    """Choose a reproducible initialization without replacement."""

    if not 0 <= count <= n_candidates:
        raise ValueError("initialization count is outside the candidate set")
    rng = np.random.default_rng(seed)
    return [int(index) for index in rng.permutation(n_candidates)[:count]]


__all__ = ["nearest_index", "normalize_rows", "seeded_initial_indices"]
