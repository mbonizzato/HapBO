"""Official-release LGBO baseline (public author code operator).

``official_lgbo`` runs the *public author release* operator (``Xinzhe309/LGBO`` @ pinned
commit ``01e56e1165a030ac74961f14d407bcc188f66ce6``; see ``external/PINNED_COMMITS.txt``): the confidence maps to the
lift by ``delta = guidance_scale * Phi^{-1}(c)`` (``guidance_scale = 3.0``), the region
weights live on a Sobol grid of ``grid_size`` points, and points are expanded to soft
regions.  It is the baseline-implementation sensitivity arm.

It differs from ``lgbo`` only in the region-lift operator
(``delta = 3*Phi^{-1}(c)`` vs the paper's direct ``delta = c``); the GP, LLM prompts, initial
design, and finite-candidate log-EI selection are otherwise standard.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .bridge import SubprocessBridgeOptimizer
from .workers.llm_defaults import DEFAULT_LLM_MODEL

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER = _REPO_ROOT / "src" / "baselines" / "workers" / "official_lgbo_worker.py"


class OfficialLGBO(SubprocessBridgeOptimizer):
    """LGBO using the public author release's guidance-scale posterior lift."""

    name = "official_lgbo"

    def resolved_command(self) -> list[str]:
        python = (os.environ.get("OFFICIAL_LGBO_PYTHON")
                  or os.environ.get("LGBO_PYTHON")
                  or os.environ.get("LLAMBO_PYTHON"))
        if python:
            return [python, str(_WORKER)]
        conda = os.environ.get("CONDA_EXE") or "conda"
        env_name = os.environ.get(
            "OFFICIAL_LGBO_ENV", os.environ.get("LGBO_ENV", os.environ.get("LLAMBO_ENV", "llambo"))
        )
        return [conda, "run", "--no-capture-output", "-n", env_name, "python", str(_WORKER)]

    def worker_config(self) -> dict[str, Any]:
        return {
            "llm_model": os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL),
            "model": str(self.objective_meta.get("model", "a black-box objective")),
            "metric": str(self.objective_meta.get("metric", "objective value")),
            "description": str(self.objective_meta.get("warmstart_context")
                               or self.objective_meta.get("description", "")),
            # Official-release operator parameters (decide.py / prior.py / boo.py defaults).
            "guidance_scale": float(self.kwargs.get("lgbo_guidance_scale", 3.0)),
            "grid_size": int(self.kwargs.get("lgbo_grid_size", 512)),
            "E_soft_low": float(self.kwargs.get("lgbo_e_soft_low", 2.0)),
            "E_soft_high": float(self.kwargs.get("lgbo_e_soft_high", 6.0)),
            "smooth_soft": float(self.kwargs.get("lgbo_smooth_soft", 0.08)),
            "smooth_region": float(self.kwargs.get("lgbo_smooth_region", 0.06)),
            "low_margin": float(self.kwargs.get("lgbo_low_margin", 0.05)),
            # Standard GP / init settings.
            "length_scale": float(self.kwargs.get("lgbo_length_scale", 0.2)),
            "noise_std": float(self.kwargs.get("lgbo_noise_std", 1e-3)),
            "paper_n_initial": 2,
            "max_tokens": int(self.kwargs.get("lgbo_max_tokens", 512)),
            "request_timeout": float(self.kwargs.get("llm_timeout", 300.0)),
        }

    def worker_env(self) -> dict[str, str]:
        base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
        return {
            "LLM_BASE_URL": base_url,
            "LLM_API_KEY": os.environ.get("LLM_API_KEY", "none"),
            "LLM_MODEL": os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL),
            "OPENAI_API_KEY": os.environ.get("LLM_API_KEY", "none"),
            "OPENAI_API_BASE": base_url,
            "OPENAI_API_TYPE": "open_ai",
            "OPENAI_API_VERSION": "",
        }


__all__ = ["OfficialLGBO"]
