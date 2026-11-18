"""One-shot CALIPER prior elicitation: 3 expert personas -> caliper_prior.json.

Runs ONCE against a local LLM endpoint serving Qwen (OpenAI-compatible).  For each
HPOBench task and each of three personas the LLM returns a COMPACT belief about
the validation-accuracy landscape (a best-config region, the accuracy it expects
there and away from it, how sharply performance falls off, and a confidence).
That belief is expanded DETERMINISTICALLY to a per-candidate predicted objective
``mu(x)`` on the accuracy scale [0, 1] and a confidence ``c(x)`` in [0, 1].

The task text is the task's prompt card from "## 1. Domain" to the end (the text HapBO's
elicitor receives): ``--card llm-card`` for the standard condition, ``--card llm-card-misleading``
for the misleading-hint condition (the same card + the task's misleading hint). One artifact per
card is written next to it, gzip-compressed:

* ``caliper-card/caliper_prior.json.gz``             -- from llm-card
* ``caliper-card-misleading/caliper_prior.json.gz``  -- from llm-card-misleading

The artifact is consumed with NO LLM at BO time: ``src/baselines/caliper.py`` aligns it to the
run's candidates by ``config_key``.  The artifact
schema is dictated by ``CALIPER._load_prior``::

    {"expert_roles": [...], "candidate_keys": [config_key(cand), ...],
     "mu": [[...]]*n_roles, "c": [[...]]*n_roles}   # mu,c shape [n_roles, n_candidates]

Usage (local LLM endpoint up)::

    python -m src.experiments.caliper_elicitation --card llm-card --tasks lr-australian       # one task
    python -m src.experiments.caliper_elicitation --card llm-card --all                        # all 40 tasks
    python -m src.experiments.caliper_elicitation --card llm-card-misleading --all
    python -m src.experiments.caliper_elicitation --card llm-card --tasks lr-australian --mock # offline check
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datasets.hpobench import (  # noqa: E402  (path set above)
    MODELS,
    TASKS,
    load_hpobench_problem,
)
from baselines.base import config_key  # noqa: E402


# The three expert roles that populate the reputation market (single-objective
# adaptation of CALIPER's per-objective experts -> per-persona experts).
ROLES: tuple[dict[str, str], ...] = (
    {
        "id": "accuracy_optimist",
        "system": (
            "You are a pragmatic AutoML practitioner whose ONLY goal is to maximize "
            "validation accuracy. You reason directly about which hyperparameter "
            "settings tend to give the highest accuracy for this model and dataset."
        ),
    },
    {
        "id": "regularization_expert",
        "system": (
            "You are a regularization and generalization expert. You favor "
            "well-regularized hyperparameter settings that generalize robustly and "
            "you are wary of configurations prone to overfitting."
        ),
    },
    {
        "id": "capacity_expert",
        "system": (
            "You are a model-capacity expert. You ensure the model has enough "
            "capacity and effective learning to avoid underfitting, and you reason "
            "about which settings give the model sufficient expressiveness."
        ),
    },
)

# JSON the LLM is asked to return, per role.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "best_config",
        "expected_best_accuracy",
        "expected_baseline_accuracy",
        "relative_sharpness",
        "confidence",
    ],
    "properties": {
        "best_config": {"type": "object"},
        "expected_best_accuracy": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "expected_baseline_accuracy": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "relative_sharpness": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


# ---------------------------------------------------------------------------
# candidate encoding (scale-free, derived from the native grid itself)
# ---------------------------------------------------------------------------


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _column_encoders(native: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Rank-normalize each parameter column to [0, 1] (scale-free: works for log grids)."""

    names = list(native[0])
    encoders: dict[str, Any] = {}
    for name in names:
        values = [candidate[name] for candidate in native]
        if all(_numeric(v) for v in values):
            uniques = sorted(set(float(v) for v in values))
            rank = {v: (i / (len(uniques) - 1) if len(uniques) > 1 else 0.0) for i, v in enumerate(uniques)}
            encoders[name] = {"kind": "numeric", "uniques": uniques, "rank": rank}
        else:
            cats = list(dict.fromkeys(str(v) for v in values))
            encoders[name] = {"kind": "categorical", "cats": cats}
    return encoders


def _encode_value(name: str, value: Any, encoders: dict[str, Any]) -> tuple[float, str]:
    """Return (normalized-position, kind) for one parameter value."""

    enc = encoders[name]
    if enc["kind"] == "numeric":
        if not _numeric(value):
            return 0.5, "numeric"  # unusable LLM value -> neutral
        v = float(value)
        # snap to the nearest grid rank so an off-grid LLM value still lands sensibly
        nearest = min(enc["uniques"], key=lambda u: abs(u - v))
        return enc["rank"][nearest], "numeric"
    cat = str(value)
    return (0.0 if cat in enc["cats"] else -1.0), "categorical"


def _expand_belief(
    native: tuple[dict[str, Any], ...],
    encoders: dict[str, Any],
    belief: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Expand one persona's compact belief to (mu, c) over ALL candidates."""

    names = list(native[0])
    # case-insensitive: the cards name the svm penalty "c", the grid "C"
    best = {str(k).lower(): v for k, v in dict(belief.get("best_config") or {}).items()}
    best_pos: dict[str, tuple[float, str]] = {}
    for name in names:
        if name.lower() in best:
            best_pos[name] = _encode_value(name, best[name.lower()], encoders)
        else:
            best_pos[name] = (0.5, encoders[name]["kind"])  # unspecified -> neutral

    lo = float(np.clip(belief.get("expected_baseline_accuracy", 0.5), 0.0, 1.0))
    hi = float(np.clip(belief.get("expected_best_accuracy", 0.8), 0.0, 1.0))
    hi = max(hi, lo)  # best must not sit below baseline
    sharpness = float(np.clip(belief.get("relative_sharpness", 0.5), 0.0, 1.0))
    confidence = float(np.clip(belief.get("confidence", 0.5), 0.0, 1.0))
    k = 1.0 + 9.0 * sharpness  # falloff rate in [1, 10]

    mu = np.empty(len(native), dtype=float)
    for row, candidate in enumerate(native):
        sq = 0.0
        for name in names:
            pos, kind = _encode_value(name, candidate[name], encoders)
            bpos, _ = best_pos[name]
            if kind == "categorical":
                sq += 0.0 if (pos == 0.0 and bpos == 0.0) else 1.0
            else:
                sq += (pos - bpos) ** 2
        d2 = sq / max(len(names), 1)
        mu[row] = lo + (hi - lo) * float(np.exp(-k * d2))
    mu = np.clip(mu, 0.0, 1.0)
    c = np.full(len(native), confidence, dtype=float)
    return mu, c


# ---------------------------------------------------------------------------
# domain / prompt
# ---------------------------------------------------------------------------


CARD_IDS = {"llm-card": "caliper-card", "llm-card-misleading": "caliper-card-misleading"}


def _card_path(model: str, task: str, card: str) -> Path:
    task_slug = task.replace("_", "-")
    return _ROOT / "experiments" / "priors" / f"hpobench-{model}-{task_slug}" / task_slug / card / "prompt_card.md"


def _card_domain_text(model: str, task: str, card: str) -> str:
    """The prompt card from "## 1. Domain" to the end, verbatim -- the same text
    scripts/elicit_hpo_priors.py hands HapBO's elicitor (Domain / Search space / Optimization
    description, incl. the questions and, for llm-card-misleading, the domain-guidance hint).
    Section 0 (the PSL-vocabulary instruction, HapBO-specific) is not sent."""

    text = _card_path(model, task, card).read_text(encoding="utf-8").replace("\r\n", "\n")
    start = text.find("## 1. Domain\n")
    if start < 0 or "\n## 2. Search space\n" not in text or "\n## 3. Optimization description\n" not in text:
        raise SystemExit(f"{model}-{task}: card {card} lacks the Domain / Search space / Optimization sections")
    return text[start:].strip()


def _prompt(domain: str, param_names: list[str]) -> str:
    return (
        f"{domain}\n\n"
        "Predict how validation accuracy varies over these hyperparameters for THIS "
        "model and dataset, from your prior knowledge (you have seen no evaluations).\n"
        "Return ONLY a JSON object with these fields:\n"
        f"  best_config: an object giving your single best guess value for each of "
        f"{param_names} (use values inside each parameter's stated range/set),\n"
        "  expected_best_accuracy: the validation accuracy (0..1) you expect AT that best config,\n"
        "  expected_baseline_accuracy: the accuracy (0..1) you expect for a typical/poor config,\n"
        "  relative_sharpness: 0..1, how sharply accuracy falls off away from the best config "
        "(0 = nearly flat landscape, 1 = a very peaked optimum),\n"
        "  confidence: 0..1, how confident you are in this prediction.\n"
        "Respond with the JSON object and nothing else."
    )


def _extract_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in response")
    return json.loads(text[start : end + 1])


# ---------------------------------------------------------------------------
# elicitation
# ---------------------------------------------------------------------------


def _mock_belief(model: str, task: str, role_idx: int, native, encoders) -> dict[str, Any]:
    """Deterministic offline belief (for --mock): pick a grid point as the 'best'."""

    rng = np.random.default_rng(abs(hash((model, task, role_idx))) % (2**32))
    pick = native[int(rng.integers(len(native)))]
    return {
        "best_config": dict(pick),
        "expected_best_accuracy": float(0.75 + 0.15 * rng.random()),
        "expected_baseline_accuracy": float(0.45 + 0.1 * rng.random()),
        "relative_sharpness": float(0.3 + 0.5 * rng.random()),
        "confidence": float(0.4 + 0.5 * rng.random()),
    }


def _elicit_belief(client, system: str, prompt: str, role_idx: int, max_attempts: int = 3) -> dict[str, Any]:
    last_error = ""
    for attempt in range(max_attempts):
        text = client.complete(
            prompt if not last_error else f"{prompt}\n\nYour previous answer was invalid: {last_error}\nReturn corrected JSON only.",
            system=system,
            response_schema=RESPONSE_SCHEMA,
            elicitation_replicate=role_idx,
            seed=1000 + role_idx,
        )
        try:
            belief = _extract_json(text)
            for key in ("expected_best_accuracy", "expected_baseline_accuracy", "relative_sharpness", "confidence"):
                float(belief[key])
            if not isinstance(belief.get("best_config"), dict):
                raise ValueError("best_config must be an object")
            return belief
        except Exception as exc:  # noqa: BLE001 - corrective retry
            last_error = str(exc)
    # Fall back to an uninformative, low-confidence prior rather than crash; the fallback flag
    # is recorded in the provenance so flat priors can be counted.
    return {
        "best_config": {},
        "expected_best_accuracy": 0.6,
        "expected_baseline_accuracy": 0.6,
        "relative_sharpness": 0.0,
        "confidence": 0.05,
        "fallback": True,
    }


def _build_client(out_dir: Path):
    from llm.client import LLMClient

    client = LLMClient(
        model=os.getenv("LLM_MODEL", "Qwen3.5-122B-A10B-GPTQ-Int4"),
        temperature=float(os.getenv("CALIPER_ELICIT_TEMPERATURE", "1.0")),
        max_tokens=int(os.getenv("CALIPER_ELICIT_MAX_TOKENS", "4096")),
        cache_dir=out_dir / "llm_cache",
        budget=int(os.getenv("CALIPER_ELICIT_BUDGET", "32")),
        provider=os.getenv("LLM_PROVIDER", "openai"),
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("LLM_API_KEY", "none"),
        enable_thinking=str(os.getenv("LLM_ENABLE_THINKING", "false")).lower() in {"1", "true", "yes"},
    )
    client.set_run_directory(out_dir)
    return client


def _write_artifact(path: Path, roles: list[str], keys: list[str], mu: np.ndarray, c: np.ndarray, provenance: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_format": "caliper-llm-prior-v1",
        "expert_roles": roles,
        "candidate_keys": keys,
        "mu": np.clip(mu, 0.0, 1.0).tolist(),
        "c": np.clip(c, 0.0, 1.0).tolist(),
        "provenance": provenance,
    }
    data = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
        handle.write(data)


def elicit_task(model: str, task: str, *, mock: bool, card: str) -> Path:
    problem = load_hpobench_problem(model=model, task=task)
    native = tuple(dict(candidate) for candidate in problem.native_candidates)
    keys = [config_key(candidate) for candidate in native]
    encoders = _column_encoders(native)
    param_names = list(native[0])
    domain = _card_domain_text(model, task, card)

    # one artifact per card, next to it; the misleading condition is its own card, as for every
    # other method
    out_path = _card_path(model, task, card).parent.parent / CARD_IDS[card] / "caliper_prior.json.gz"
    client = None if mock else _build_client(out_path.parent)
    shown = re.findall(r"^name: (\S+)$", domain, re.M) or param_names  # the card's own parameter names
    mu_rows, c_rows, raw = [], [], {}
    for role_idx, role in enumerate(ROLES):
        if mock:
            belief = _mock_belief(model, task, role_idx, native, encoders)
        else:
            belief = _elicit_belief(client, role["system"], _prompt(domain, shown), role_idx)
        raw[role["id"]] = belief
        mu, c = _expand_belief(native, encoders, belief)
        mu_rows.append(mu)
        c_rows.append(c)
    _write_artifact(out_path, [role["id"] for role in ROLES], keys, np.vstack(mu_rows), np.vstack(c_rows), {
        "model": model, "task": task, "card": card,
        "llm_model": None if mock else os.getenv("LLM_MODEL", "Qwen3.5-122B-A10B-GPTQ-Int4"),
        "mock": bool(mock), "beliefs": raw,
    })
    return out_path


def _parse_tasks(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.all:
        return list(itertools.product(MODELS, TASKS))
    pairs: list[tuple[str, str]] = []
    for token in args.tasks:
        model, _, task = token.partition("-")
        if model not in MODELS or task not in TASKS:
            raise SystemExit(f"unknown task {token!r}; models={MODELS} tasks={sorted(TASKS)}")
        pairs.append((model, task))
    if not pairs:
        raise SystemExit("specify --all or --tasks <model-task> [...]")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="*", default=[], help="e.g. lr-australian rf-kc1")
    parser.add_argument("--all", action="store_true", help="all 40 HPOBench tasks")
    parser.add_argument("--mock", action="store_true", help="offline: synthetic beliefs, no LLM")
    parser.add_argument("--card", choices=sorted(CARD_IDS), required=True,
                        help="task text = this prompt card; writes caliper-card[-misleading]/caliper_prior.json.gz")
    args = parser.parse_args()

    for model, task in _parse_tasks(args):
        path = elicit_task(model, task, mock=args.mock, card=args.card)
        print(f"{model}-{task}: wrote {path}")


if __name__ == "__main__":
    main()
