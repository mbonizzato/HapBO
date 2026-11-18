"""Shared loader for NeuroMOSAIC Dataset 1 cortical-stimulation files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.io import loadmat

from .neuromosaic_common import OfflineOptimizationProblem


REQUIRED_FIELDS = {
    "emgs",
    "nChan",
    "ch2xy",
    "sorted_resp",
    "sorted_isvalid",
    "sorted_respMean",
    "sorted_respSD",
}


def _matlab_strings(values: np.ndarray) -> list[str]:
    """Decode the one-row MATLAB cell array used for muscle names."""

    names = []
    for value in np.asarray(values, dtype=object).reshape(-1):
        scalar = np.asarray(value).reshape(-1)
        if scalar.size != 1:
            raise ValueError("each entry in emgs must contain one muscle name")
        names.append(str(scalar[0]))
    return names


def load_neuromosaic_dataset1_problem(
    subject_path: str | Path,
    muscle: str,
    *,
    dataset_id: str,
    source: str,
    noise_seed: int | None = None,
) -> OfflineOptimizationProblem:
    """Load one Dataset 1 subject/muscle response map from a MATLAB file.

    The species-specific module (Dataset 1 rat) owns only its registry,
    defaults, and path resolution. Candidate rows follow ``ch2xy`` exactly.

    ``objective`` polling:
      * ``noise_seed is None`` (default): deterministic — returns the aligned
        valid-trial mean ``sorted_respMean``.
      * ``noise_seed`` set: **noisy** — each query returns one uniformly-random
        VALID single trial from ``sorted_resp`` (matching OptimizeNeurostim's
        emulator), drawn from a run-scoped RNG seeded by ``noise_seed``. The
        ground-truth means are unchanged, so regret (best-queried-by-true-mean)
        stays defined against ``sorted_respMean``.
    """

    path = Path(subject_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"NeuroMOSAIC subject file not found: {path} "
            "(download Dataset 1 from https://osf.io/54vhx/overview)"
        )
    if path.suffix.lower() != ".mat":
        raise ValueError("NeuroMOSAIC Dataset 1 subject_path must be a .mat file")

    subject_id = path.stem
    contents = loadmat(path)
    if subject_id not in contents:
        raise ValueError(
            f"{path.name} does not contain the expected {subject_id!r} struct"
        )
    root = contents[subject_id]
    if root.shape != (1, 1) or root.dtype.names is None:
        raise ValueError(f"{path.name} has an unsupported MATLAB structure")
    missing = REQUIRED_FIELDS - set(root.dtype.names)
    if missing:
        raise ValueError(f"{path.name} is missing required fields {sorted(missing)}")
    record = root[0, 0]

    muscle_names = _matlab_strings(record["emgs"])
    if muscle not in muscle_names:
        raise ValueError(
            f"muscle {muscle!r} is unavailable for {subject_id}; "
            f"choose one of {muscle_names}"
        )
    muscle_index = muscle_names.index(muscle)

    coordinates = np.asarray(record["ch2xy"], dtype=int)
    n_channels = int(np.asarray(record["nChan"]).reshape(-1)[0])
    if coordinates.shape != (n_channels, 2):
        raise ValueError(
            "ch2xy must have shape (n_channels, 2), got "
            f"{coordinates.shape} for n_channels={n_channels}"
        )
    if len({tuple(row) for row in coordinates}) != n_channels:
        raise ValueError("ch2xy must contain one unique coordinate per channel")

    response_means = np.asarray(record["sorted_respMean"], dtype=float)
    response_sds = np.asarray(record["sorted_respSD"], dtype=float)
    expected_shape = (n_channels, len(muscle_names))
    if response_means.shape != expected_shape or response_sds.shape != expected_shape:
        raise ValueError(
            "sorted response statistics must align with channels and muscles"
        )
    ground_truth = response_means[:, muscle_index].copy()
    ground_truth_sd = response_sds[:, muscle_index].copy()
    if not np.all(np.isfinite(ground_truth)):
        raise ValueError("selected response map contains non-finite mean values")

    response_cells = record["sorted_resp"]
    validity_cells = record["sorted_isvalid"]
    valid_trials: list[np.ndarray] = []
    for row in range(n_channels):
        responses = np.asarray(response_cells[row, muscle_index], dtype=float).reshape(-1)
        validity = np.asarray(validity_cells[row, muscle_index]).reshape(-1)
        # keep only trials that are both flagged valid AND finite, so noisy polling
        # can never return a NaN/inf that would crash the run (and be silently dropped)
        valid_trials.append(responses[(validity != 0) & np.isfinite(responses)])
    valid_repetitions = np.asarray([trials.size for trials in valid_trials], dtype=int)
    total_repetitions = np.asarray(
        [np.asarray(response_cells[row, muscle_index]).size for row in range(n_channels)],
        dtype=int,
    )
    if np.any(valid_repetitions == 0):
        raise ValueError("every candidate must have at least one valid response trial")

    native_candidates = tuple(
        {"channel_x": int(x), "channel_y": int(y)} for x, y in coordinates
    )
    coordinate_to_index = {
        (candidate["channel_x"], candidate["channel_y"]): index
        for index, candidate in enumerate(native_candidates)
    }
    # Per-channel common-random-numbers (CRN): each channel gets its own reproducible
    # stream (spawned from noise_seed), so the k-th query to a channel returns the same
    # trial regardless of which arm/strategy asks -- maximal paired-noise cancellation
    # across arms, and unbiased. Independent of the optimizer's init/tie-break RNG.
    if noise_seed is None:
        channel_rngs: list[np.random.Generator] | None = None
    else:
        channel_rngs = [
            np.random.default_rng(seed)
            for seed in np.random.SeedSequence(int(noise_seed)).spawn(n_channels)
        ]

    def _index_of(candidate: Mapping[str, Any]) -> int:
        try:
            x = float(candidate["channel_x"])
            y = float(candidate["channel_y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("candidate must provide numeric channel_x and channel_y") from exc
        if not x.is_integer() or not y.is_integer():
            raise ValueError("channel_x and channel_y must be integer coordinates")
        coordinate = (int(x), int(y))
        if coordinate not in coordinate_to_index:
            raise ValueError(f"candidate coordinate {coordinate} is not in ch2xy")
        return coordinate_to_index[coordinate]

    def objective(candidate: Mapping[str, Any]) -> float:
        index = _index_of(candidate)
        if channel_rngs is None:
            return float(ground_truth[index])
        trials = valid_trials[index]
        return float(trials[int(channel_rngs[index].integers(trials.size))])

    metadata = {
        "source": source,
        "subject_path": str(path),
        "available_muscles": tuple(muscle_names),
        "muscle_index": muscle_index,
        "response_mode": (
            "sorted_respMean (mean of valid repeated trials)"
            if noise_seed is None
            else f"noisy single valid trial (sorted_resp), seed={int(noise_seed)}"
        ),
        "total_repetitions_per_candidate": total_repetitions,
        "valid_repetitions_per_candidate": valid_repetitions,
    }
    return OfflineOptimizationProblem(
        dataset_id=dataset_id,
        subject_id=subject_id,
        target_name=muscle,
        native_candidates=native_candidates,
        encoded_candidates=coordinates.astype(float),
        ground_truth=ground_truth,
        response_standard_deviation=ground_truth_sd,
        objective=objective,
        metadata=metadata,
        maximize=True,
    )


__all__ = ["REQUIRED_FIELDS", "load_neuromosaic_dataset1_problem"]
