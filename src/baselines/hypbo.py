"""HypBO (Cisse et al., IJCAI 2024) via the subprocess bridge, with LLM hypotheses (BORA C.3).

The worker runs the pinned upstream checkout (external/hypbo) in its own environment; see
src/baselines/workers/hypbo_worker.py and external/PINNED_COMMITS.txt.  The prior is a list of
box hypotheses in the encoded space, built by ``src/experiments/worker.py`` from the one-shot LLM
artifact of scripts/elicit_hypbo_llm.py.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .bridge import SubprocessBridgeOptimizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HYPBO_ROOT = _REPO_ROOT / "external" / "hypbo"
_WORKER = _REPO_ROOT / "src" / "baselines" / "workers" / "hypbo_worker.py"

#: Upstream HypBO defaults (hypbo.py, main_continuous.py) used by the worker.
HYPBO_DEFAULTS = {
    "global_failure_limit": 5,
    "local_failure_limit": 2,
    "gamma": 0.0,
    "batch": 1,
    # HypBO's docstring default; one process keeps the run deterministic and serves the
    # objective callback from a single process.
    "n_processes": 1,
}


class HypBO(SubprocessBridgeOptimizer):
    """HypBO with multiple static box hypotheses."""

    name = "hypbo"

    def resolved_command(self) -> list[str]:
        python = os.environ.get("HYPBO_PYTHON")
        if python:
            return [python, str(_WORKER)]
        conda = os.environ.get("CONDA_EXE") or "conda"
        env_name = os.environ.get("HYPBO_ENV", "hypbo")
        return [conda, "run", "--no-capture-output", "-n", env_name, "python", str(_WORKER)]

    def worker_config(self) -> dict[str, Any]:
        if not isinstance(self.prior, Mapping) or not isinstance(self.prior.get("hypotheses"), list):
            raise ValueError("HypBO requires a prior mapping with a 'hypotheses' list")
        dim = self.encoded_candidates.shape[1]
        lower = self.encoded_candidates.min(axis=0)
        upper = self.encoded_candidates.max(axis=0)
        boxes = []
        for k, hypothesis in enumerate(self.prior["hypotheses"]):
            lo = np.asarray(hypothesis["lb"], dtype=float).reshape(-1)
            hi = np.asarray(hypothesis["ub"], dtype=float).reshape(-1)
            if lo.shape != (dim,) or hi.shape != (dim,):
                raise ValueError(f"hypothesis {k} has {lo.size} bounds for {dim} encoded dimensions")
            if np.any(hi < lo) or np.any(lo < lower - 1e-12) or np.any(hi > upper + 1e-12):
                raise ValueError(f"hypothesis {k} is not a box inside the encoded bounds")
            boxes.append({"name": str(hypothesis.get("name", f"H{k + 1}")),
                          "lb": lo.tolist(), "ub": hi.tolist()})
        config = {key: self.kwargs.get(f"hypbo_{key}", value) for key, value in HYPBO_DEFAULTS.items()}
        config["hypotheses"] = boxes
        return config

    def worker_env(self) -> dict[str, str]:
        return {"HYPBO_ROOT": os.environ.get("HYPBO_ROOT", str(_HYPBO_ROOT))}


__all__ = ["HYPBO_DEFAULTS", "HypBO"]
