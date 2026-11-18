"""Leak-control benchmark: a bespoke convolutional encoder trained from scratch.

This is the contamination-clean control against benchmark memorization (the LLM
cannot have seen this bespoke grid's optimum). The optimization target is a small,
deliberately custom CNN ("convolutional encoder + linear head") whose
hyperparameter-response surface is computed by us on a subsampled image dataset.
No published leaderboard, paper, or repository contains the accuracy of every
config in this specific grid, so the location of the optimum cannot have been in
any LLM's training data. An LLM can still form a prior from general
machine-learning knowledge, so the benchmark tests prior shaping without the
possibility of recall.

The grid is a finite 4-D discrete space (750 configs). Each config is trained from
scratch for ``SEEDS`` seeds; the seed-averaged validation accuracy is the ground
truth and the per-config seed spread is stored as the response SD. The loader returns
an ``OfflineOptimizationProblem``, so the benchmark plugs into the offline-finite
harness path.

Top-level imports are numpy-only so this module is importable without torch or
scikit-learn (which are needed only to *build* the table, not to load it).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from datasets.neuromosaic_common import OfflineOptimizationProblem

# --------------------------------------------------------------------------- #
# Search space: 4-D discrete grid = 6 x 5 x 5 x 5 = 750 configs.
# Every axis carries transferable ML meaning (so a prior can be elicited from
# general ML knowledge) but the joint optimum is a property of *our* net + *our*
# grid + *our* protocol.
# --------------------------------------------------------------------------- #
LEARNING_RATES: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)   # 6, log
BASE_CHANNELS: tuple[int, ...] = (8, 16, 24, 32, 48)                        # 5, linear
DROPOUTS: tuple[float, ...] = (0.0, 0.1, 0.25, 0.4, 0.5)                    # 5, linear
WEIGHT_DECAYS: tuple[float, ...] = (0.0, 1e-5, 1e-4, 1e-3, 1e-2)           # 5, log(floor)

PARAMETER_NAMES: tuple[str, ...] = (
    "learning_rate",
    "base_channels",
    "dropout",
    "weight_decay",
)
LOG_SCALED = {"learning_rate", "weight_decay"}
WD_FLOOR = 1e-6  # weight_decay == 0 maps here so log spacing stays one step below 1e-5

# Training protocol of the benchmark regime (single source of truth).
# "fashion-of": Fashion-MNIST subsample of 2,000 training, 2,000 validation and
# 2,000 test images; each configuration is trained for 35 epochs with batch size
# 64.
DEFAULT_DATASET = "fashion-of"
SEEDS: tuple[int, ...] = (0, 1, 2)
DATA_SPLIT_SEED = 0  # fixes the train/val/test subsample; separate from train seeds

REGIMES: dict[str, dict] = {
    "fashion-of": {"source": "fashion", "n_train": 2_000, "n_val": 2_000, "n_test": 2_000, "epochs": 35, "batch": 64},
}


def regime_for(dataset: str) -> dict:
    if dataset not in REGIMES:
        raise ValueError(f"unknown cnn-leak dataset/regime {dataset!r}; known: {sorted(REGIMES)}")
    return REGIMES[dataset]


def enumerate_grid(
    learning_rates: Sequence[float] = LEARNING_RATES,
    base_channels: Sequence[int] = BASE_CHANNELS,
    dropouts: Sequence[float] = DROPOUTS,
    weight_decays: Sequence[float] = WEIGHT_DECAYS,
) -> list[dict[str, Any]]:
    """Return the finite candidate grid as native config dicts (Cartesian product)."""

    grid: list[dict[str, Any]] = []
    for lr in learning_rates:
        for channels in base_channels:
            for dropout in dropouts:
                for weight_decay in weight_decays:
                    grid.append(
                        {
                            "learning_rate": float(lr),
                            "base_channels": int(channels),
                            "dropout": float(dropout),
                            "weight_decay": float(weight_decay),
                        }
                    )
    return grid


def _transform_column(name: str, values: np.ndarray) -> np.ndarray:
    if name == "weight_decay":
        return np.log10(np.maximum(values, WD_FLOOR))
    if name in LOG_SCALED:
        return np.log10(values)
    return values


def encode_candidates(candidates: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Encode the 4 columns to [0, 1] (log where appropriate, min-max per column)."""

    columns = []
    for name in PARAMETER_NAMES:
        raw = np.array([float(candidate[name]) for candidate in candidates], dtype=float)
        transformed = _transform_column(name, raw)
        span = float(np.ptp(transformed))
        normalized = (
            np.zeros_like(transformed)
            if span <= 1e-12
            else (transformed - float(np.min(transformed))) / span
        )
        columns.append(normalized.reshape(-1, 1))
    return np.column_stack(columns).astype(float, copy=False)


def config_key(config: Mapping[str, Any]) -> str:
    """Stable string key; identical between the trainer, builder, and loader."""

    return "|".join(f"{name}={config[name]}" for name in PARAMETER_NAMES)


def table_path(dataset: str = DEFAULT_DATASET) -> "Path":  # noqa: F821
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "benchmark" / "cnn_leak"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"table_{dataset}.json"


def load_cnn_leak_problem(
    *,
    dataset: str = DEFAULT_DATASET,
) -> OfflineOptimizationProblem:
    """Load the precomputed leak-control CNN grid as a finite known-optimum benchmark.

    The table is produced by the training sweep (``leak_control_cnn/``): each config
    is trained from scratch for ``SEEDS`` seeds and aggregated by
    ``build_table.py`` into ``benchmark/cnn_leak/table_<dataset>.json`` as
    ``{config_key: {mean, sd, seeds, n_seeds}}``.
    """

    import json

    path = table_path(dataset)
    candidates = enumerate_grid()
    if not path.is_file():
        raise FileNotFoundError(
            f"cnn-leak table missing: {path}. Build it first by running the "
            f"training sweep and `python leak_control_cnn/build_table.py "
            f"--dataset {dataset}` (see leak_control_cnn/README.md)."
        )

    table = json.loads(path.read_text(encoding="utf-8"))
    missing = [config_key(c) for c in candidates if config_key(c) not in table]
    if missing:
        raise ValueError(
            f"cnn-leak table {path} is incomplete: {len(missing)}/{len(candidates)} "
            f"configs absent (e.g. {missing[0]}). Re-run the sweep / build_table."
        )

    ground_truth = np.array(
        [float(table[config_key(c)]["mean"]) for c in candidates], dtype=float
    )
    response_sd = np.array(
        [float(table[config_key(c)].get("sd", 0.0)) for c in candidates], dtype=float
    )
    lookup = {config_key(c): float(table[config_key(c)]["mean"]) for c in candidates}

    def objective(config: Mapping[str, Any]) -> float:
        return lookup[config_key(config)]

    return OfflineOptimizationProblem(
        dataset_id="cnn-leak",
        subject_id=str(dataset),
        target_name="val_accuracy",
        native_candidates=tuple(candidates),
        encoded_candidates=encode_candidates(candidates),
        ground_truth=ground_truth,
        response_standard_deviation=response_sd,
        objective=objective,
        metadata={
            "dataset": dataset,
            "n_seeds": len(SEEDS),
            "epochs": REGIMES.get(dataset, {}).get("epochs"),
            "n_train": REGIMES.get(dataset, {}).get("n_train"),
            "architecture": "conv-encoder-2block+linear-head",
            "leak_control": True,
        },
        maximize=True,
    )


__all__ = [
    "LEARNING_RATES",
    "BASE_CHANNELS",
    "DROPOUTS",
    "WEIGHT_DECAYS",
    "PARAMETER_NAMES",
    "LOG_SCALED",
    "SEEDS",
    "DATA_SPLIT_SEED",
    "DEFAULT_DATASET",
    "REGIMES",
    "regime_for",
    "enumerate_grid",
    "encode_candidates",
    "config_key",
    "table_path",
    "load_cnn_leak_problem",
]
