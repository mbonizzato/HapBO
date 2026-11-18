"""LLAMBO-discriminative worker for a finite HapBO candidate set.

This process runs in LLAMBO's isolated Python 3.8 environment. Upstream owns
its complete optimization loop. Continuous proposals are snapped to the nearest
finite encoded candidate and evaluated through the line-delimited JSON bridge.
"""

from __future__ import annotations

import asyncio
import itertools
import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness import finish, log, make_evaluator, read_spec  # noqa: E402
from finite import nearest_index, normalize_rows, seeded_initial_indices  # noqa: E402


def _install_chat_shim(openai_module, *, base_url: str, api_key: str, model: str,
                       timeout: float, seed: int) -> None:
    """Adapt the asynchronous OpenAI 0.x ``ChatCompletion.acreate`` calls to SGLang chat completions."""

    import aiohttp

    endpoint = base_url.rstrip("/") + "/chat/completions"
    request_ids = itertools.count()

    async def one_completion(payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=client_timeout, trust_env=True) as session:
            async with session.post(endpoint, headers=headers, json=payload) as response:
                text = await response.text()
                if response.status >= 400:
                    raise RuntimeError(
                        f"LLAMBO local LLM returned HTTP {response.status}: {text[:1000]}"
                    )
                try:
                    return await response.json()
                except Exception as exc:
                    raise RuntimeError(f"LLAMBO local LLM returned invalid JSON: {text[:1000]}") from exc

    async def chat_acreate(**kwargs):
        count = max(1, int(kwargs.get("n", 1) or 1))
        ordinal = next(request_ids)
        requested_max_tokens = int(kwargs.get("max_tokens", 500))
        messages = [dict(message) for message in kwargs["messages"]]
        if requested_max_tokens <= 8 and messages:
            messages[-1]["content"] += (
                "\nThe text between the ## markers must be a numeric decimal; "
                "do not write the word performance."
            )
        base_payload = {
            "model": kwargs.get("model") or kwargs.get("engine") or model,
            "messages": messages,
            "temperature": float(kwargs.get("temperature", 0.7)),
            # Eight GPT tokens are often too short under Qwen's tokenizer.
            "max_tokens": max(requested_max_tokens, 24),
            "top_p": float(kwargs.get("top_p", 0.95)),
            "n": 1,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        payloads = []
        for generation in range(count):
            payload = dict(base_payload)
            payload["seed"] = int(seed + ordinal * 10_000 + generation)
            payloads.append(payload)
        responses = await asyncio.gather(*(one_completion(payload) for payload in payloads))
        choices: list[dict[str, Any]] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for index, response in enumerate(responses):
            response_choices = response.get("choices") or []
            if not response_choices:
                raise RuntimeError("LLAMBO local LLM response contained no choices")
            choice = dict(response_choices[0])
            if requested_max_tokens <= 8:
                message = dict(choice.get("message") or {})
                content = str(message.get("content") or "")
                strict = re.search(r"##\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*##", content)
                if strict:
                    message["content"] = f"## {strict.group(1)} ##"
                else:
                    numbers = re.findall(r"-?(?:\d+(?:\.\d*)?|\.\d+)", content)
                    if numbers:
                        message["content"] = f"## {numbers[-1]} ##"
                choice["message"] = message
            choice["index"] = index
            choices.append(choice)
            response_usage = response.get("usage") or {}
            for key in usage:
                usage[key] += int(response_usage.get(key, 0) or 0)
        return {"choices": choices, "usage": usage}

    openai_module.ChatCompletion.acreate = staticmethod(chat_acreate)


def _task_context(
    dim: int, n_samples: int, maximize: bool, config: dict[str, Any]
) -> dict[str, Any]:
    constraints = {
        f"x{column}": ["float", "linear", [0.0, 1.0]] for column in range(dim)
    }
    requested_task = str(config.get("task", "regression")).lower()
    task = "classification" if "class" in requested_task else "regression"
    # Domain description. Upstream LLAMBO's surrogate- and acquisition-prompt builders read
    # only the fixed task_context fields (model/task/metric/feature counts); there is no
    # free-text description slot. The task description and any domain guidance (the
    # misleading description of the adversarial studies) are therefore appended to the
    # `model` descriptor, which every prompt prefix prints verbatim ("...for a {model}...").
    # The synthetic-panel runner (scripts/llambo_continuous.py) uses the same channel, and
    # BORA receives the same text through its experiment description. `target_description`
    # is set only in studies with optimizer.misleading_description: true
    # (src/experiments/worker.py); without it `model` is unchanged.
    model = str(config.get("model", "black-box system"))
    task_text = str(config.get("task_description", "")).strip()
    if task_text:  # non-HPO problems: the task description (the card's facts, no benchmark names)
        model = f"{model} (task description: {task_text})"
    description = str(config.get("target_description", "")).strip()
    if description:
        model = f"{model} (domain guidance: {description.rstrip('.')})"
    # Upstream's full-context prefix describes the ML DATASET ("contains N samples and F
    # features (C categorical, M numerical)", "label contains K classes"). When the harness
    # supplies coarse dataset facts (tabular HPO tasks) they fill those slots; otherwise the
    # generic stand-ins below are used.
    ds = dict(config.get("dataset_context") or {})
    return {
        "model": model,
        # Upstream prompt construction supports exactly these two labels.
        "task": task,
        "tot_feats": ds.get("tot_feats", dim),
        "cat_feats": ds.get("cat_feats", 0),
        "num_feats": ds.get("num_feats", dim),
        "n_classes": ds.get("n_classes", 2 if task == "classification" else 0),
        "metric": str(config.get("metric", "objective value")),
        "lower_is_better": not maximize,
        "num_samples": ds.get("num_samples", n_samples),
        "hyperparameter_constraints": constraints,
    }


def _bound_acquisition_scores(llambo) -> None:
    """Avoid upstream's accuracy-only [0, 1] guard for arbitrary objectives.

    Canonical LLAMBO is unchanged for score histories strictly inside (0, 1).
    Otherwise only the inverse-model candidate sampler sees an affine [0.1, 0.9]
    view; the discriminative surrogate still ranks candidates using raw scores.
    """

    import pandas as pd

    original = llambo.acq_func.get_candidate_points
    announced = [False]

    def bounded(observed_configs, observed_fvals, *args, **kwargs):
        values = observed_fvals.iloc[:, 0].astype(float)
        if float(values.min()) > 0.0 and float(values.max()) < 1.0:
            return original(observed_configs, observed_fvals, *args, **kwargs)
        span = float(values.max() - values.min())
        if span <= 0.0:
            scaled = values * 0.0 + 0.5
        else:
            scaled = 0.1 + 0.8 * (values - float(values.min())) / span
        if not announced[0]:
            log("LLAMBO acquisition prompts rescaled non-[0,1] scores to [0.1,0.9]")
            announced[0] = True
        frame = pd.DataFrame({observed_fvals.columns[0]: scaled}, index=observed_fvals.index)
        return original(observed_configs, frame, *args, **kwargs)

    llambo.acq_func.get_candidate_points = bounded


def _zero_shot_initial_indices(spec, config, n_init, seed, *, base_url, api_key, model):
    """LLAMBO Mode 2: zero-shot warmstarting (paper Sec. 'warmstarting'; upstream
    exp_warmstarting FullTemplate skeleton). The LLM proposes the n_init initial
    configurations from the problem context; proposals are snapped to the nearest
    finite candidates (log-aware per dimension) and de-duplicated, any shortfall
    filled from the seeded random permutation (logged). The context is the HapBO
    prompt-card text in place of upstream's dataset meta-features (external/PINNED_COMMITS.txt)."""

    import json as _json
    import math
    import urllib.request

    natives = [dict(c) for c in spec["candidates"]]
    names = sorted(natives[0].keys())
    per_dim = {n: sorted({float(c[n]) for c in natives}) for n in names}
    ranges = ", ".join(
        f"{n} (float between [{min(per_dim[n]):g}, {max(per_dim[n]):g}])" for n in names
    )
    context = str(config.get("warmstart_context", "")).strip()
    prompt = (
        "You are assisting me with automated machine learning.\n"
        + (context + "\n" if context else "")
        + f"I'm exploring a subset of hyperparameters detailed as: {ranges}.\n"
        + f"Provide {n_init} diverse and efficient configuration suggestions to commence a "
        + "Bayesian Optimization procedure aimed at fine-tuning hyperparameters.\n"
        + "You mustn't include 'None' in the configurations.\n"
        + "Your response should include only a list of dictionaries, where each dictionary "
        + "describes one recommended configuration. Do not enumerate the dictionaries."
    )

    def call(text):
        payload = _json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "temperature": 0.0,
            "max_tokens": 2048,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode("utf-8")
        request = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions", data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(request, timeout=float(config.get("request_timeout", 300.0))) as resp:
            return _json.loads(resp.read())["choices"][0]["message"]["content"]

    def snap_index(proposal):
        snapped = {}
        folded = {str(k).lower(): v for k, v in proposal.items()}
        for n in names:
            values = per_dim[n]
            x = float(proposal[n] if n in proposal else folded[n.lower()])
            if x > 0 and all(v > 0 for v in values):
                snapped[n] = min(values, key=lambda v: abs(math.log(v) - math.log(x)))
            else:
                snapped[n] = min(values, key=lambda v: abs(v - x))
        key = tuple(snapped[n] for n in names)
        lookup = {tuple(float(c[n]) for n in names): i for i, c in enumerate(natives)}
        index = lookup.get(key)
        if index is None:
            # Candidate sets that are not complete grids (SOLAR's Latin-hypercube pool) almost never
            # contain the per-dimension snapped tuple -> nearest candidate row jointly, in the same per-dimension
            # log-aware space, each dimension scaled by its range. Complete grids (HPO, cortical) never reach this.
            def coord(n, v):
                vals = per_dim[n]
                return math.log(max(v, min(vals))) if all(u > 0 for u in vals) else float(v)  # clamp <=0 on log dims
            spans = {n: (max(coord(n, v) for v in per_dim[n]) - min(coord(n, v) for v in per_dim[n])) or 1.0
                     for n in names}
            want = {n: coord(n, float(proposal[n] if n in proposal else folded[n.lower()])) for n in names}
            index = min(range(len(natives)), key=lambda i: sum(
                ((coord(n, float(natives[i][n])) - want[n]) / spans[n]) ** 2 for n in names))
        return index

    indices, attempt_prompt, error = [], prompt, ""
    for _ in range(4):
        raw = call(attempt_prompt)
        try:
            start = raw.index("[")
            proposals, _ = _json.JSONDecoder().raw_decode(raw[start:])
            assert isinstance(proposals, list) and proposals
            for proposal in proposals:
                index = snap_index(proposal)
                if index is not None and index not in indices:
                    indices.append(index)
            if len(indices) >= n_init:
                break
            error = f"only {len(indices)} distinct valid configurations after snapping"
        except (ValueError, KeyError, AssertionError, TypeError) as exc:
            error = f"unparseable reply ({exc})"
        attempt_prompt = prompt + (
            f"\n\nYour previous answer was insufficient: {error}. Answer again with "
            f"{n_init} DISTINCT configurations as a plain JSON list of dictionaries."
        )
    log(f"zero-shot warmstart proposals -> candidate indices {indices[:n_init]}")
    if len(indices) < n_init:
        fill = [i for i in seeded_initial_indices(len(natives), len(natives), seed)
                if i not in indices]
        needed = n_init - len(indices)
        log(f"warmstart shortfall: filling {needed} of {n_init} from the seeded permutation")
        indices.extend(fill[:needed])
    return indices[:n_init]


def main() -> None:
    spec = read_spec()
    encoded = normalize_rows(spec["encoded"])
    budget = int(spec["budget"])
    n_init = max(1, min(int(spec["n_init"]), budget))
    seed = int(spec.get("seed") or 0)
    maximize = bool(spec.get("maximize", True))
    config = dict(spec.get("config") or {})
    evaluate = make_evaluator()
    n_total, dim = encoded.shape

    root = os.environ.get("LLAMBO_ROOT")
    if not root or not os.path.isdir(root):
        raise RuntimeError("LLAMBO_ROOT must point to the pinned upstream checkout")
    sys.path.insert(0, root)
    os.chdir(root)

    # Every upstream LLM module reads these variables at import time.
    base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("LLM_API_KEY", "none")
    model = str(config.get("llm_model") or os.environ.get("LLM_MODEL") or "local-model")
    os.environ.setdefault("OPENAI_API_KEY", api_key)
    os.environ.setdefault("OPENAI_API_BASE", base_url)
    os.environ.setdefault("OPENAI_API_TYPE", "open_ai")
    os.environ.setdefault("OPENAI_API_VERSION", "")

    import openai  # noqa: E402
    from llambo.llambo import LLAMBO  # noqa: E402
    import llambo.acquisition_function as _llambo_af  # noqa: E402
    import llambo.discriminative_sm_utils as _llambo_ds  # noqa: E402
    import llambo.generative_sm_utils as _llambo_gs  # noqa: E402

    # Upstream picks the printed precision from each bound's decimals; our coordinates live in
    # [0, 1], whose lower bound 0.0 would print every value with 0 decimals. Fix it at 4.
    _llambo_af.LLM_ACQ._count_decimal_places = lambda self, n: 4
    _llambo_ds._count_decimal_places = lambda n: 4
    _llambo_gs._count_decimal_places = lambda n: 2  # upstream adds 2 here

    _install_chat_shim(
        openai,
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=float(config.get("request_timeout", 300.0)),
        seed=seed,
    )

    init_mode = str(config.get("init_mode", "random"))
    if init_mode == "zero-shot":
        initial_indices = _zero_shot_initial_indices(
            spec, config, n_init, seed, base_url=base_url, api_key=api_key, model=model
        )
    else:
        initial_indices = seeded_initial_indices(n_total, n_init, seed)

    def initialization(count: int) -> list[dict[str, float]]:
        if count != n_init:
            raise ValueError(f"LLAMBO requested {count} initial points; expected {n_init}")
        return [
            {f"x{column}": float(encoded[index, column]) for column in range(dim)}
            for index in initial_indices
        ]

    def objective(candidate: dict[str, float]):
        point = [candidate[f"x{column}"] for column in range(dim)]
        index = nearest_index(encoded, point)
        value = float(evaluate(index))
        snapped = {f"x{column}": float(encoded[index, column]) for column in range(dim)}
        return snapped, {"score": value, "generalization_score": value}

    n_candidates = int(config.get("n_candidates", 10))
    n_templates = int(config.get("n_templates", 2))
    if n_candidates < 5 or n_templates < 1 or n_candidates % n_templates:
        raise ValueError("LLAMBO candidates must be >=5 and divisible by templates")
    n_trials = max(0, budget - n_init)
    optimizer = LLAMBO(
        _task_context(dim, n_total, maximize, config),
        sm_mode="discriminative",
        n_candidates=n_candidates,
        n_templates=n_templates,
        n_gens=int(config.get("n_generations", 10)),
        alpha=float(config.get("alpha", 0.1)),
        n_initial_samples=n_init,
        n_trials=n_trials,
        init_f=initialization,
        bbox_eval_f=objective,
        chat_engine=model,
        prompt_setting=config.get("prompt_setting"),
    )
    optimizer.seed = seed
    _bound_acquisition_scores(optimizer)
    log(
        f"LLAMBO-discriminative endpoint={base_url} model={model} "
        f"n_init={n_init} n_trials={n_trials}"
    )
    optimizer.optimize()
    finish()


if __name__ == "__main__":
    main()
