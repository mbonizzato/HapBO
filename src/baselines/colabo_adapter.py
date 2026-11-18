"""Faithful ColaBO-LogEI baseline via the subprocess bridge.

The worker imports the pinned repository's patched Ax and BoTorch trees and
reconstructs its ``colabo_logei`` GenerationStrategy without importing the
optional PD1/mf-prior-bench benchmark mapping.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .bridge import SubprocessBridgeOptimizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COLABO_ROOT = _REPO_ROOT / "external" / "colabo"
_WORKER = _REPO_ROOT / "src" / "baselines" / "workers" / "colabo_worker.py"


class ColaBOAdapter(SubprocessBridgeOptimizer):
    """ColaBO with the upstream MC prior-weighted LogEI acquisition."""

    name = "colabo"

    def resolved_command(self) -> list[str]:
        python = os.environ.get("COLABO_PYTHON")
        if python:
            return [python, str(_WORKER)]
        conda = os.environ.get("CONDA_EXE") or "conda"
        env_name = os.environ.get("COLABO_ENV", "colabo")
        return [conda, "run", "--no-capture-output", "-n", env_name, "python", str(_WORKER)]

    def worker_config(self) -> dict[str, Any]:
        if not isinstance(self.prior, Mapping) or "values" not in self.prior:
            raise ValueError(
                "ColaBO requires a candidate-aligned prior mapping with a 'values' field"
            )
        prior_values = np.asarray(self.prior["values"], dtype=float).reshape(-1)
        if prior_values.shape != (len(self.candidates),) or not np.all(np.isfinite(prior_values)):
            raise ValueError("ColaBO prior values must be finite and candidate-aligned")
        confidence = float(
            self.kwargs.get(
                "colabo_confidence",
                self.prior.get("confidence", os.environ.get("COLABO_CONFIDENCE", 0.2)),
            )
        )
        if confidence <= 0.0:
            raise ValueError("ColaBO confidence/spread must be positive")
        return {
            "prior_values": prior_values.tolist(),
            "confidence": confidence,
            "num_samples": int(self.kwargs.get("colabo_num_samples", 76_800)),
            "resampling_fraction": float(
                self.kwargs.get("colabo_resampling_fraction", 0.0067)
            ),
            "raw_samples": int(self.kwargs.get("colabo_raw_samples", 1_024)),
            "num_restarts": int(self.kwargs.get("colabo_num_restarts", 20)),
            "max_internal_trials": int(
                self.kwargs.get("colabo_max_internal_trials", 3 * self.budget)
            ),
        }

    def worker_env(self) -> dict[str, str]:
        return {"COLABO_ROOT": str(_COLABO_ROOT)}


__all__ = ["ColaBOAdapter"]
