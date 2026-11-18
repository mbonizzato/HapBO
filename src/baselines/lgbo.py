"""LGBO baseline (LLM-guided BO with a region-lifted posterior) via the subprocess bridge.

From-paper implementation of LGBO (ICLR 2026, arXiv 2605.17976), not a pinned upstream
checkout (see external/PINNED_COMMITS.txt). LGBO queries the LLM in the loop each
iteration for a preference (a point or a region plus a confidence ``c``), discretizes
it on a Sobol grid, and lifts the current GP posterior so that the preferred regional
functional moves by ``c`` posterior standard deviations: the mean is shifted by
``c * Sigma(X, G) a / sqrt(a^T Sigma(G, G) a)`` with posterior covariances, and the
covariance is unchanged. The worker refits the Matern-5/2 hyperparameters by marginal
likelihood at every iteration and asks the LLM for the two initial locations. This is
distinct from HapBO's bounded acquisition multiplier.

``requires_prior: false``: LGBO generates its own guidance in the loop from the task text
and the observed data; it does not consume a compiled prior. The parent only serves
objective evaluations over the finite candidate set through the JSON bridge.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .bridge import SubprocessBridgeOptimizer
from .workers.llm_defaults import DEFAULT_LLM_MODEL

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER = _REPO_ROOT / "src" / "baselines" / "workers" / "lgbo_worker.py"


class LGBO(SubprocessBridgeOptimizer):
    """From-paper LGBO run through the finite-candidate subprocess bridge."""

    name = "lgbo"

    def resolved_command(self) -> list[str]:
        # Prefer a direct interpreter (LGBO_PYTHON, like LLAMBO_PYTHON) to avoid the
        # per-run `conda run` startup cost; LGBO reuses LLAMBO's Python 3.8 env (it
        # needs only numpy/scipy/aiohttp, all present there). Falls back to
        # `conda run -n <env>` where <env> defaults to the llambo env.
        python = os.environ.get("LGBO_PYTHON") or os.environ.get("LLAMBO_PYTHON")
        if python:
            return [python, str(_WORKER)]
        conda = os.environ.get("CONDA_EXE") or "conda"
        env_name = os.environ.get("LGBO_ENV", os.environ.get("LLAMBO_ENV", "llambo"))
        return [conda, "run", "--no-capture-output", "-n", env_name, "python", str(_WORKER)]

    def worker_config(self) -> dict[str, Any]:
        return {
            "llm_model": os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL),
            # Shared context fed to the in-loop guidance prompt (card text).
            "model": str(self.objective_meta.get("model", "a black-box objective")),
            "metric": str(self.objective_meta.get("metric", "objective value")),
            # For a warmstart-card (requires_prior) study, worker.py fills
            # objective_meta["warmstart_context"] with the rendered expert prior
            # (e.g. the cortical human expert map) -> feed it as LGBO's in-loop task
            # description so its LLM reasons on the SAME expert knowledge the other
            # prior-consuming methods get (LGBO's native protocol has no cortical
            # prior format). Prior-free studies leave it unset and fall back to the
            # plain description.
            "description": str(self.objective_meta.get("warmstart_context")
                               or self.objective_meta.get("description", "")),
            # Region-lift discretization + decay weighting (paper Sec. 1).
            "n_grid": int(self.kwargs.get("lgbo_n_grid", 64)),
            "decay": float(self.kwargs.get("lgbo_decay", 4.0)),
            # GP hyperparameters (initial values; refit by marginal likelihood each iteration).
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
            # Parity with the other LLM workers' OpenAI-compatible endpoint variables.
            "OPENAI_API_KEY": os.environ.get("LLM_API_KEY", "none"),
            "OPENAI_API_BASE": base_url,
            "OPENAI_API_TYPE": "open_ai",
            "OPENAI_API_VERSION": "",
        }


__all__ = ["LGBO"]
