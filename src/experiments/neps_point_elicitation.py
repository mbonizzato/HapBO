"""One-shot point-prior elicitation for the NePS multi-fidelity baselines (PriorBand, PriMO).

NePS consumes a user prior as ONE value per hyperparameter plus a confidence level
``low`` / ``medium`` / ``high``. This module asks the LLM for exactly that, from the same
task card HapBO's elicitor and CALIPER receive (``--card llm-card`` for the standard
condition, ``--card llm-card-misleading`` for the card plus the misleading hint).

LLM-visible text (rendered with ``--render``):

* system: the card's section-0 framing without its PSL-vocabulary sentence
  (:data:`SYSTEM_PROMPT`, checked against every card);
* user: the card from ``## 1. Domain`` to the end, verbatim, then
  :data:`OUTPUT_FORMAT` (output format only).

Sampling as the HapBO elicitation (temperature 0, seed 7, thinking off, free-form
generation). Validation: at most 4 attempts; on a malformed or invalid answer the exact
validator error is fed back as a corrective turn. A task that still fails after 4 attempts
is written as ``elicitation_failed.json``; no prior is substituted (its study runs stay
blocked: the prior file does not exist).

A value must lie inside the parameter's listed range; an off-grid value is mapped to
the nearest grid value on the parameter's scale (log for log-spaced grids) and the
mapping is recorded. The grid index is what NePS receives (``src/baselines/neps_mf.py``).

Output, next to the card's directory::

    experiments/priors/hpobench-<fam>-<task>/<task>/neps-point-card/neps_prior.json             (llm-card)
    experiments/priors/hpobench-<fam>-<task>/<task>/neps-point-card-misleading/neps_prior.json  (llm-card-misleading)

plus ``prompt.json``, ``raw_llm_response_attempt<k>.txt``, ``llm_calls.jsonl`` and ``llm_cache/``.

Usage (LLM endpoint up, see REPRODUCING.md)::

    python -m src.experiments.neps_point_elicitation --card llm-card --tasks xgb-kc1 svm-car
    python -m src.experiments.neps_point_elicitation --card llm-card-misleading --all
    python -m src.experiments.neps_point_elicitation --card llm-card --all --render   # prompts only, no LLM
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datasets.hpobench import MODELS, TASKS, hpobench_search_space_summary, load_hpobench_problem  # noqa: E402

#: card directory -> output directory (next to the card)
CARD_IDS = {"llm-card": "neps-point-card", "llm-card-misleading": "neps-point-card-misleading"}
CONFIDENCE_LEVELS = ("low", "medium", "high")
MAX_ATTEMPTS = 4
ARTIFACT_FORMAT = "neps-point-prior-v1"

SYSTEM_PROMPT = (
    "You are an expert in the domain described below. Using your knowledge of this field, "
    "generate a prior belief over the location of the optimum for the Bayesian optimization "
    "problem defined by the search space and optimization description below."
)

OUTPUT_FORMAT = (
    "## Output format\n"
    "Write the prior as one value and one confidence level per hyperparameter:\n"
    "- \"value\": your single best guess for the hyperparameter, taken from its listed values;\n"
    "- \"confidence\": how confident you are in that value: \"low\", \"medium\" or \"high\".\n"
    "Return ONLY a JSON object keyed by the hyperparameter names of the search space:\n"
    "{\"<name>\": {\"value\": <number>, \"confidence\": \"<low|medium|high>\"}, ...}"
)


class PointElicitationError(RuntimeError):
    """No valid point prior after the maximum number of attempts."""


# ---------------------------------------------------------------------------
# card / grid
# ---------------------------------------------------------------------------


def card_path(model: str, task: str, card: str) -> Path:
    slug = task.replace("_", "-")
    return _ROOT / "experiments" / "priors" / f"hpobench-{model}-{slug}" / slug / card / "prompt_card.md"


def output_dir(model: str, task: str, card: str) -> Path:
    return card_path(model, task, card).parent.parent / CARD_IDS[card]


def card_task_text(text: str) -> str:
    """The card from ``## 1. Domain`` to the end, verbatim (as for CALIPER and LLAMBO Mode 2)."""

    text = text.replace("\r\n", "\n")
    start = text.find("## 1. Domain\n")
    if start < 0 or "\n## 2. Search space\n" not in text or "\n## 3. Optimization description\n" not in text:
        raise ValueError("card lacks the Domain / Search space / Optimization sections")
    return text[start:].strip()


def card_parameter_names(task_text: str) -> list[str]:
    names = re.findall(r"^name: (\S+)$", task_text, re.M)
    if not names:
        raise ValueError("card lists no parameter names")
    return names


def check_system_prompt(card_text: str) -> None:
    """SYSTEM_PROMPT must equal the card's section 0 without its PSL-vocabulary sentence."""

    text = card_text.replace("\r\n", "\n")
    match = re.search(r"## 0\. Prompt\n(.*?)\n## 1\. Domain\n", text, re.S)
    if match is None:
        raise ValueError("card has no '## 0. Prompt' section")
    section = " ".join(match.group(1).split())
    psl = "Write the belief only in the closed PSL vocabulary you were given; never invent tokens."
    if section.replace(psl, "").strip() != " ".join(SYSTEM_PROMPT.split()):
        raise ValueError("SYSTEM_PROMPT no longer matches the card's section 0 (minus the PSL sentence)")


def render_prompt(task_text: str) -> tuple[str, str]:
    """(system, user) for the first attempt."""

    return SYSTEM_PROMPT, f"{task_text}\n\n{OUTPUT_FORMAT}"


def grid_dimensions(model: str, task: str, *, benchmark_directory: Path | None = None) -> dict[str, dict[str, Any]]:
    """Per grid parameter: sorted unique values (the NePS index order) and scale."""

    kwargs = {} if benchmark_directory is None else {"benchmark_directory": benchmark_directory}
    problem = load_hpobench_problem(model=model, task=task, **kwargs)
    summary = hpobench_search_space_summary(model, task, **kwargs)
    scales = {dim["name"]: dim.get("scale", "linear") for dim in summary["dimensions"]}
    dims: dict[str, dict[str, Any]] = {}
    for name in problem.parameter_names:
        values = sorted({candidate[name] for candidate in problem.native_candidates})
        dims[name] = {"values": values, "scale": scales.get(name, "linear")}
    return dims


# ---------------------------------------------------------------------------
# parsing / validation
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> Any:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the answer")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"the JSON object is malformed ({exc.msg} at char {exc.pos})") from exc


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def snap_to_grid(value: float, grid: list[float], scale: str) -> tuple[int, bool]:
    """Index of the grid value nearest to ``value`` on the grid's scale; (index, exact)."""

    arr = np.asarray(grid, dtype=float)
    exact = np.flatnonzero(np.isclose(arr, value, rtol=1e-6, atol=0.0))
    if exact.size:
        return int(exact[0]), True
    if scale == "log" and value > 0 and np.all(arr > 0):
        distance = np.abs(np.log(arr) - math.log(value))
    else:
        distance = np.abs(arr - value)
    return int(np.argmin(distance)), False


def validate_answer(
    document: Any,
    card_names: list[str],
    dims: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Validate one parsed answer; return the per-grid-parameter prior or raise ValueError."""

    if not isinstance(document, dict):
        raise ValueError("the answer must be a JSON object keyed by hyperparameter name")
    by_lower = {str(key).strip().lower(): key for key in document}
    if len(by_lower) != len(document):
        raise ValueError("duplicate hyperparameter names (case-insensitive)")
    expected = {name.lower(): name for name in card_names}
    missing = [expected[k] for k in expected if k not in by_lower]
    extra = [str(by_lower[k]) for k in by_lower if k not in expected]
    if missing or extra:
        raise ValueError(f"the keys must be exactly {card_names}; missing {missing}, unexpected {extra}")
    grid_by_lower = {name.lower(): name for name in dims}
    prior: dict[str, dict[str, Any]] = {}
    for lower, card_name in expected.items():
        entry = document[by_lower[lower]]
        if not isinstance(entry, dict) or not {"value", "confidence"} <= set(entry):
            raise ValueError(f"{card_name}: expected an object with the fields 'value' and 'confidence'")
        value, confidence = entry["value"], entry["confidence"]
        if not _numeric(value):
            raise ValueError(f"{card_name}: 'value' must be a number, got {value!r}")
        if not isinstance(confidence, str) or confidence.strip().lower() not in CONFIDENCE_LEVELS:
            raise ValueError(f"{card_name}: 'confidence' must be one of {list(CONFIDENCE_LEVELS)}, got {confidence!r}")
        grid_name = grid_by_lower.get(lower)
        if grid_name is None:
            raise ValueError(f"{card_name}: not a parameter of the task grid")
        grid = dims[grid_name]["values"]
        low, high = float(min(grid)), float(max(grid))
        v = float(value)
        tol = 1e-6 * max(abs(low), abs(high))
        if not (low - tol <= v <= high + tol):
            raise ValueError(f"{card_name}: value {value} lies outside the listed values [{low}, {high}]")
        index, exact = snap_to_grid(v, grid, dims[grid_name]["scale"])
        prior[grid_name] = {
            "card_name": card_name,
            "llm_value": value,
            "value": grid[index],
            "index": index,
            "n_values": len(grid),
            "on_grid": exact,
            "scale": dims[grid_name]["scale"],
            "confidence": confidence.strip().lower(),
        }
    return prior


# ---------------------------------------------------------------------------
# elicitation
# ---------------------------------------------------------------------------


def _llm_settings() -> dict[str, Any]:
    return {
        "model": os.getenv("LLM_MODEL", "Qwen3.5-122B-A10B-GPTQ-Int4"),
        "provider": os.getenv("LLM_PROVIDER", "openai"),
        "base_url": os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
        "temperature": float(os.getenv("NEPS_ELICIT_TEMPERATURE", "0.0")),
        "seed": int(os.getenv("NEPS_ELICIT_SEED", "7")),
        "max_tokens": int(os.getenv("NEPS_ELICIT_MAX_TOKENS", "4096")),
        "enable_thinking": str(os.getenv("LLM_ENABLE_THINKING", "false")).lower() in {"1", "true", "yes"},
    }


def _build_client(out_dir: Path, settings: dict[str, Any]):
    from llm.client import LLMClient

    client = LLMClient(
        model=settings["model"],
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
        cache_dir=out_dir / "llm_cache",
        budget=MAX_ATTEMPTS,
        provider=settings["provider"],
        base_url=settings["base_url"],
        api_key=os.getenv("LLM_API_KEY", "none"),
        enable_thinking=settings["enable_thinking"],
        request_timeout=600.0,
    )
    client.set_run_directory(out_dir)
    return client


def elicit_point_prior(
    client: Any,
    task_text: str,
    dims: dict[str, dict[str, Any]],
    *,
    seed: int,
    max_attempts: int = MAX_ATTEMPTS,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Validate-and-retry loop. Returns (prior, attempts) or raises PointElicitationError
    (its ``attempts`` attribute holds every raw answer and error)."""

    card_names = card_parameter_names(task_text)
    system, base_user = render_prompt(task_text)
    user = base_user
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        raw = client.complete(user, system=system, response_schema=None, seed=seed)
        try:
            prior = validate_answer(_extract_json(raw), card_names, dims)
        except ValueError as exc:
            attempts.append({"attempt": attempt, "raw": raw, "error": str(exc)})
            user = (
                base_user
                + f"\n\n## Your previous answer (#{attempt}) was REJECTED\n"
                + f"Validator error: {exc}\nReturn ONLY the corrected JSON object."
            )
            continue
        attempts.append({"attempt": attempt, "raw": raw, "error": None})
        return prior, attempts
    error = PointElicitationError(f"no valid point prior after {max_attempts} attempts: {attempts[-1]['error']}")
    error.attempts = attempts  # type: ignore[attr-defined]
    raise error


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _call_log(out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / "llm_calls.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _usage(out_dir: Path, start: int) -> dict[str, int]:
    """Calls / tokens logged by this invocation (call-log lines from ``start`` on)."""

    calls = _call_log(out_dir)[start:]
    return {"calls": len(calls), "cached_calls": sum(bool(c.get("cached")) for c in calls),
            "tokens": sum(int(c.get("tokens", 0)) for c in calls)}


def elicit_task(model: str, task: str, card: str, *, render_only: bool = False, force: bool = False) -> dict[str, Any]:
    out = output_dir(model, task, card)
    prior_path = out / "neps_prior.json"
    card_file = card_path(model, task, card)
    card_text = card_file.read_text(encoding="utf-8")
    task_text = card_task_text(card_text)
    system, user = render_prompt(task_text)
    check_system_prompt(card_text)
    if prior_path.is_file() and not force and not render_only:
        stored = json.loads(prior_path.read_text(encoding="utf-8")).get("provenance", {})
        same_input = stored.get("prompt_sha256") == _sha256_text(system + "\n\n" + user)
        if same_input and stored.get("llm") == _llm_settings():  # same LLM input and settings: keep it
            return {"task": f"{model}-{task}", "card": card, "status": "exists", "path": str(prior_path)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt.json").write_text(json.dumps({
        "system": system, "user": user, "card": card,
        "card_sha256": _sha256_text(card_text),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if render_only:
        return {"task": f"{model}-{task}", "card": card, "status": "rendered", "path": str(out / "prompt.json")}

    dims = grid_dimensions(model, task)
    settings = _llm_settings()
    client = _build_client(out, settings)
    (out / "elicitation_failed.json").unlink(missing_ok=True)
    for stale in out.glob("raw_llm_response_attempt*.txt"):
        stale.unlink()
    log_start = len(_call_log(out))
    provenance = {
        "model": model, "task": task, "card": card, "card_sha256": _sha256_text(card_text),
        "prompt_sha256": _sha256_text(system + "\n\n" + user), "llm": settings, "max_attempts": MAX_ATTEMPTS,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        prior, attempts = elicit_point_prior(client, task_text, dims, seed=settings["seed"])
    except PointElicitationError as exc:
        attempts = getattr(exc, "attempts", [])
        for item in attempts:
            (out / f"raw_llm_response_attempt{item['attempt']}.txt").write_text(item["raw"], encoding="utf-8")
        usage = _usage(out, log_start)
        (out / "elicitation_failed.json").write_text(json.dumps({
            **provenance, "status": "failed", "error": str(exc),
            "attempts": [{"attempt": a["attempt"], "error": a["error"]} for a in attempts],
            "usage": usage,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        prior_path.unlink(missing_ok=True)
        return {"task": f"{model}-{task}", "card": card, "status": "failed", "error": str(exc),
                "attempts": len(attempts), **usage}
    for item in attempts:
        (out / f"raw_llm_response_attempt{item['attempt']}.txt").write_text(item["raw"], encoding="utf-8")
    usage = _usage(out, log_start)
    document = {
        "artifact_format": ARTIFACT_FORMAT,
        "model": model,
        "task": task,
        "card": card,
        "parameters": prior,
        "provenance": {
            **provenance,
            "attempts": len(attempts),
            "rejected": [{"attempt": a["attempt"], "error": a["error"]} for a in attempts if a["error"]],
            "usage": usage,
        },
    }
    prior_path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"task": f"{model}-{task}", "card": card, "status": "ok", "attempts": len(attempts),
            **usage, "path": str(prior_path)}


def _parse_tasks(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.all:
        return list(itertools.product(MODELS, TASKS))
    pairs = []
    for token in args.tasks:
        model, _, task = token.partition("-")
        task = task.replace("-", "_")
        if model not in MODELS or task not in TASKS:
            raise SystemExit(f"unknown task {token!r}")
        pairs.append((model, task))
    if not pairs:
        raise SystemExit("specify --all or --tasks <model-task> [...]")
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--card", choices=sorted(CARD_IDS), required=True)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--render", action="store_true", help="write prompt.json only (no LLM call)")
    parser.add_argument("--force", action="store_true", help="re-elicit even if neps_prior.json exists")
    parser.add_argument("--summary", type=Path, default=None, help="write the per-task summary JSON here")
    args = parser.parse_args()
    rows = []
    for model, task in _parse_tasks(args):
        row = elicit_task(model, task, args.card, render_only=args.render, force=args.force)
        rows.append(row)
        print(json.dumps(row), flush=True)
    counts = {status: sum(r["status"] == status for r in rows) for status in sorted({r["status"] for r in rows})}
    summary = {"card": args.card, "counts": counts, "tasks": rows,
               "tokens": sum(int(r.get("tokens", 0)) for r in rows)}
    print(f"SUMMARY card={args.card} {counts} tokens={summary['tokens']}", flush=True)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 1 if counts.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
