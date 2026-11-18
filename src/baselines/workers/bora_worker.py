"""BORA worker (runs in the isolated `bora` env).

Reproduces BORA (Ablatif6c/bora-the-explorer) faithfully on our finite candidate
set: BORA optimises the continuous encoded box; each continuous proposal is
snapped to the nearest finite candidate (Euclidean, in encoded space) and its
objective value is fetched from the parent via the bridge callback. Documented
deviations (external/PINNED_COMMITS.txt): (1) continuous->finite snapping; (2) the
LLM assistant calls a locally served OpenAI-compatible endpoint (OPENAI_BASE_URL).
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # harness.py, llm_defaults.py

BORA_ROOT = os.environ.get("BORA_ROOT")
if BORA_ROOT:
    sys.path.insert(0, BORA_ROOT)
    # BORA loads its prompt templates via paths relative to its repo root.
    os.chdir(BORA_ROOT)

import numpy as np  # noqa: E402

from harness import finish, log, make_evaluator, read_spec  # noqa: E402
from llm_defaults import DEFAULT_LLM_MODEL  # noqa: E402


def main() -> None:
    spec = read_spec()
    encoded = np.asarray(spec["encoded"], dtype=float)
    budget = int(spec["budget"])
    n_init = int(spec["n_init"])
    seed = int(spec["seed"] or 0)
    maximize = bool(spec.get("maximize", True))
    config = spec.get("config", {})
    evaluate = make_evaluator()
    n_candidates, dim = encoded.shape

    from bora.bora import Bora
    from bora.experiment import Experiment, Parameter, Target, Type

    parameters = []
    for j in range(dim):
        parameter = Parameter(name=f"x{j}", type=Type.continuous, description=f"encoded dim {j}")
        parameter.set_bounds(float(encoded[:, j].min()), float(encoded[:, j].max()))
        parameters.append(parameter)
    target = Target(name=config.get("target_name", "objective"),
                    description=config.get("target_description", "objective value"))

    def objective_function(input_data):
        # BORA calls this with one positional array of encoded coords (in key
        # order) and expects a length-1 sequence back (it uses res[0]).
        point = np.asarray(input_data, dtype=float).reshape(-1)
        index = int(np.argmin(((encoded - point) ** 2).sum(axis=1)))  # snap to nearest candidate
        value = evaluate(index)
        return np.array([value if maximize else -value])  # BORA maximises

    experiment = Experiment(
        name=config.get("name", "finite-candidate"),
        description=config.get("description", "finite candidate objective via bridge"),
        parameters=parameters,
        target=target,
        objective_function=objective_function,
        domain=config.get("domain", "encoded real-valued search space"),
        default_precision=6,
    )

    llm_model = config.get("llm_model", os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL))
    api_key = os.environ.get("OPENAI_API_KEY", "none")
    log_path = tempfile.mkdtemp(prefix="bora_log_")

    # Register the served model in BORA's model registry (token/cost bookkeeping)
    # and fall back to a generic tokenizer for unknown models. OPENAI_BASE_URL is
    # read by the openai SDK so BORA's client talks to the configured endpoint.
    from bora import assistant as _assistant
    import tiktoken as _tiktoken

    _assistant.MODEL_INFOS.setdefault(
        llm_model,
        {"prompt_tokens_price_per_1k": 0.0, "completion_tokens_price_per_1k": 0.0, "max_tokens": 32_000},
    )
    _orig_get_model_info = _assistant.Assistant._get_model_info

    def _patched_get_model_info(self):
        if self._model in _assistant.MODEL_INFOS:
            info = dict(_assistant.MODEL_INFOS[self._model])
            try:
                info["encoding"] = _tiktoken.encoding_for_model(self._model)
            except Exception:
                info["encoding"] = _tiktoken.get_encoding("cl100k_base")
            return info
        return _orig_get_model_info(self)

    _assistant.Assistant._get_model_info = _patched_get_model_info

    # Force BORA's OpenAI client at the configured endpoint (it captured `OpenAI` at
    # import, so setting OPENAI_BASE_URL alone is not enough).
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL")
    if base_url and hasattr(_assistant, "OpenAI"):
        _OrigOpenAI = _assistant.OpenAI

        def _PatchedOpenAI(*args, **kwargs):
            kwargs.setdefault("base_url", base_url)
            client = _OrigOpenAI(*args, **kwargs)
            # Qwen has thinking mode ON by default; its reasoning trace breaks
            # BORA's response parser, so disable it on every call.
            _orig_create = client.chat.completions.create

            def _create(*call_args, **call_kwargs):
                extra = dict(call_kwargs.get("extra_body") or {})
                template = dict(extra.get("chat_template_kwargs") or {})
                template.setdefault("enable_thinking", False)
                extra["chat_template_kwargs"] = template
                call_kwargs["extra_body"] = extra
                if int(call_kwargs.get("n", 1) or 1) > 1:
                    call_kwargs["n"] = 1  # SGLang may reject n>1 (documented deviation)
                # Qwen's chat template requires a system message only at the start;
                # BORA appends system messages mid-conversation, so coerce any
                # non-leading system message to a user message (documented deviation).
                messages = call_kwargs.get("messages")
                if messages:
                    fixed = []
                    for position, message in enumerate(messages):
                        if position > 0 and message.get("role") == "system":
                            message = {**message, "role": "user"}
                        fixed.append(message)
                    call_kwargs["messages"] = fixed
                try:
                    return _orig_create(*call_args, **call_kwargs)
                except Exception as exc:  # surface BORA's swallowed LLM errors
                    sys.stderr.write(f"[bora_worker] LLM create failed: {exc!r}\n")
                    sys.stderr.flush()
                    raise

            client.chat.completions.create = _create
            return client

        _assistant.OpenAI = _PatchedOpenAI
    log(f"BORA LLM endpoint: {base_url}")

    bora = Bora(
        experiment=experiment,
        api_key=api_key,
        log_path=log_path,
        llm_model=llm_model,
        verbose=False,
        random_seed=seed,
        allow_duplicate_points=False,
    )
    # BORA's natural budget = n_init + n_iter probes. Snapping collisions are
    # served from cache by the bridge and charged as repeat queries; the bridge
    # stops the worker once the budget is used. We run past `budget` so the
    # worker does not stop early.
    n_init_points = max(1, min(n_init, budget - 1))
    n_iter = max(1, min(n_candidates, 2 * budget) - n_init_points)
    log(f"BORA maximize n_init={n_init_points} n_iter={n_iter} model={llm_model}")
    bora.maximize(n_init_points=n_init_points, n_iter=n_iter)
    finish()


if __name__ == "__main__":
    main()
