"""Faithful BORA baseline (Ablatif6c/bora-the-explorer) via the subprocess bridge.

BORA runs in its own `bora` conda env (external/bora, pinned in
external/PINNED_COMMITS.txt) and its LLM assistant calls a locally served
OpenAI-compatible endpoint. See src/baselines/workers/bora_worker.py.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .bridge import SubprocessBridgeOptimizer
from .workers.llm_defaults import DEFAULT_LLM_MODEL

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BORA_ROOT = _REPO_ROOT / "external" / "bora"
_WORKER = _REPO_ROOT / "src" / "baselines" / "workers" / "bora_worker.py"
_CONDA = os.environ.get("CONDA_EXE") or "conda"
_ENV_NAME = os.environ.get("BORA_ENV", "bora")


class BORA(SubprocessBridgeOptimizer):
    """BORA (IJCAI 2025) run faithfully from its pinned upstream checkout."""

    name = "bora"
    worker_command = [
        _CONDA, "run", "--no-capture-output", "-n", _ENV_NAME, "python", str(_WORKER)
    ]

    def worker_config(self) -> dict[str, Any]:
        return {
            "llm_model": os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL),
            "name": "finite-candidate",
            # Upstream BORA renders experiment.description / experiment.domain / target.name in
            # its prompts; Target.description is never shown. The harness description carries
            # the task line and, in misleading-hint studies, the hint.
            "description": str(self.objective_meta.get("description", "finite candidate optimization task")),
            "domain": str(self.objective_meta.get("description", "encoded real-valued search space")),
            "target_name": str(self.objective_meta.get("metric", "objective")),
            "target_description": str(self.objective_meta.get("target_description", "objective value")),
        }

    def worker_env(self) -> dict[str, str]:
        return {
            "BORA_ROOT": str(_BORA_ROOT),
            # BORA's OpenAI client calls the configured endpoint (documented deviation).
            "OPENAI_API_KEY": os.environ.get("LLM_API_KEY", "none"),
            "OPENAI_BASE_URL": os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
            "LLM_MODEL": os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL),
        }


__all__ = ["BORA"]
