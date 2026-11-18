"""Faithful discriminative LLAMBO baseline via the subprocess bridge.

The pinned upstream optimizer owns candidate generation, LLM surrogate ranking,
and its internal optimization loop. The worker projects its continuous encoded
proposals onto this repository's finite candidate set and asks the parent to
evaluate the resulting candidate. LLAMBO stays in an isolated Python 3.8 env.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .bridge import SubprocessBridgeOptimizer
from .workers.llm_defaults import DEFAULT_LLM_MODEL

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LLAMBO_ROOT = _REPO_ROOT / "external" / "LLAMBO"
_WORKER = _REPO_ROOT / "src" / "baselines" / "workers" / "llambo_worker.py"


class LLAMBOAdapter(SubprocessBridgeOptimizer):
    """LLAMBO-discriminative at the pinned upstream commit (external/PINNED_COMMITS.txt)."""

    name = "llambo"

    def resolved_command(self) -> list[str]:
        python = os.environ.get("LLAMBO_PYTHON")
        if python:
            return [python, str(_WORKER)]
        conda = os.environ.get("CONDA_EXE") or "conda"
        env_name = os.environ.get("LLAMBO_ENV", "llambo")
        return [conda, "run", "--no-capture-output", "-n", env_name, "python", str(_WORKER)]

    def worker_config(self) -> dict[str, Any]:
        return {
            "llm_model": os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL),
            "model": str(self.objective_meta.get("model", "black-box system")),
            "task": str(self.objective_meta.get("task", "optimization")),
            "metric": str(self.objective_meta.get("metric", "objective value")),
            "description": str(
                self.objective_meta.get(
                    "description", "finite candidate optimization task"
                )
            ),
            # The misleading description of the adversarial studies lands here; the worker folds
            # it into the `model` descriptor so it reaches LLAMBO's in-loop surrogate and
            # acquisition prompts (the same text BORA gets via its experiment description).
            # Set only in studies with optimizer.misleading_description: true.
            "target_description": str(self.objective_meta.get("target_description", "")),
            # Mode 2 (paper 'zero-shot warmstarting'): init_mode=zero-shot makes the
            # worker ask the LLM for the n_init initial configurations, with the
            # card text as context (set by the study harness).
            "init_mode": str(self.kwargs.get("llambo_init_mode", "random")),
            "warmstart_context": str(self.objective_meta.get("warmstart_context", "")),
            "dataset_context": dict(self.objective_meta.get("dataset_context") or {}),
            "prompt_setting": self.objective_meta.get("prompt_setting"),
            "task_description": str(self.objective_meta.get("task_description", "")),
            "n_candidates": int(self.kwargs.get("llambo_candidates", 10)),
            "n_templates": int(self.kwargs.get("llambo_templates", 2)),
            "n_generations": int(self.kwargs.get("llambo_generations", 10)),
            "alpha": float(self.kwargs.get("llambo_alpha", 0.1)),
            "request_timeout": float(self.kwargs.get("llm_timeout", 300.0)),
        }

    def worker_env(self) -> dict[str, str]:
        base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
        return {
            "LLAMBO_ROOT": str(_LLAMBO_ROOT),
            "LLM_BASE_URL": base_url,
            "LLM_API_KEY": os.environ.get("LLM_API_KEY", "none"),
            "LLM_MODEL": os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL),
            # Upstream reads all four at module import. Calls are replaced by the
            # worker's OpenAI-compatible chat shim immediately after import.
            "OPENAI_API_KEY": os.environ.get("LLM_API_KEY", "none"),
            "OPENAI_API_BASE": base_url,
            "OPENAI_API_TYPE": "open_ai",
            "OPENAI_API_VERSION": "",
        }


__all__ = ["LLAMBOAdapter"]
