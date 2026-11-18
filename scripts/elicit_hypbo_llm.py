#!/usr/bin/env python3
"""Elicit HypBO hypotheses from the local LLM for the hypbo-llm baseline (BORA supplement C.3).

For every HPOBench task the LLM is asked ONCE, offline, from the SAME prompt card as the
colabo-llm (c, s) elicitation (scripts/elicit_colabo_llm_priors.py), for any number of
hypotheses, each a guessed optimum location ``c`` with a confidence score ``s`` in [0, 1].
Only the answer format differs from the ColaBO ask: a list of hypotheses instead of a single
guess. Each hypothesis becomes a box centred at ``c`` with side ``L = (ub - lb) * (1 - s + 1e-6)``,
clipped to the bounds; ``lb`` and ``ub`` bound the encoded space HypBO optimizes over
(per-dimension min-max over the grid values, log-scaled dimensions in log space). Every parsed
hypothesis is recorded; ``"kept"`` marks the boxes passed to HypBO (a later box is dropped only
when it and a kept box with the same centre are both narrower than 1e-4).

Modalities: ``llm-card`` (the card of the experiments/studies/hpo/hapbo HapBO arm) and, with
--adversarial, ``llm-card-misleading`` (the same card plus the task's misleading hint as a
"Domain guidance" section). Inference settings are those of the colabo-llm elicitation:
temperature 0.0, thinking disabled, up to --max-attempts attempts with the validation error
fed back. Artifacts: experiments/hypbo_llm_hypotheses/hpobench-tabular/<task>/<modality>.json
(model, prompt hash, parsed hypotheses, native and encoded boxes, token usage) with the prompt
next to them. The raw reply of every attempt goes to
results/elicitation/hypbo/hpobench-tabular/<task>/<modality>.replies.json (not shipped), which the
gate scripts/check_hypbo_hypotheses.py re-reads.

Endpoint: LLM_BASE_URL, LLM_MODEL, LLM_API_KEY (OpenAI-compatible).
    python scripts/elicit_hypbo_llm.py [--dry-run] [--only SUBSTR] [--overwrite] [--adversarial] [--workers N]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src" / "baselines" / "workers"))

from elicit_colabo_llm_priors import (  # noqa: E402
    ADVERSARIAL_MODALITY,
    _parse_card_dims,
    _sha256_text,
    build_adversarial_jobs,
    build_jobs,
    validate_answer,
)
from box_hypotheses import EPSILON, hypothesis_box, select_hypotheses  # noqa: E402

OUT_ROOT = _ROOT / "experiments" / "hypbo_llm_hypotheses"
REPLIES_ROOT = _ROOT / "results" / "elicitation" / "hypbo"
MODEL = os.environ.get("LLM_MODEL", "Qwen3.5-122B-A10B-GPTQ-Int4")
BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
TEMPERATURE = 0.0
MAX_TOKENS = 4096
TIMEOUT = 300.0
MAX_WORKERS = 4
CONSTRUCTION = ("BORA supplement C.3: one box per hypothesis, centred at c, side "
                "L = (ub - lb) * (1 - s + 1e-6) (C.2 floor), clipped to [lb, ub]; a later box is "
                "dropped only when it and a kept same-centre box are both narrower than 1e-4")


def _ask_block(parameter_names: list[str], example: str) -> str:
    return (
        "## 0. Prompt\n"
        "You are an expert in the domain described below. Using your knowledge of this\n"
        "field, give one or more hypotheses about the location of the optimum of the\n"
        "optimization problem defined by the search space and optimization description\n"
        "below. Each hypothesis is a guess for the location of the optimum together with\n"
        "your confidence in that guess. The number of hypotheses is up to you.\n\n"
        "You may reason briefly first, in plain prose. Then, on the final line of your\n"
        "answer, output a single JSON object of this exact form and nothing after it:\n"
        f"{example}\n\n"
        '- "hypotheses" lists your hypotheses, one object per hypothesis.\n'
        '- "c" is the guessed optimum location of a hypothesis: one value for every parameter\n'
        f"  ({', '.join(parameter_names)}), using only values the search space allows\n"
        "  (an exact listed value for ordinal/categorical parameters; any value inside\n"
        "  the range for continuous parameters).\n"
        '- "s" is the confidence score of that hypothesis in [0, 1], where 0 means no\n'
        "  confidence and 1 means certain.\n"
    )


def build_prompt(card_text: str, card_path: Path, hint: str | None = None) -> str:
    """The colabo-llm HPO prompt with the hypotheses ask: section 0 replaced, the card from
    '1. Domain' on verbatim, the misleading hint (if any) as a 'Domain guidance' section."""

    m = re.search(r"(?m)^(?:## )?1\. Domain\s*$", card_text)
    if m is None:
        raise SystemExit(f"no '1. Domain' section found in {card_path} - refusing to guess")
    dims = _parse_card_dims(card_text)
    if not dims:
        raise SystemExit(f"no search-space dims parsed from {card_path}")
    names = [d["name"] for d in dims]
    example = ('{"hypotheses": [{"c": {' + ", ".join(f'"{n}": <allowed value>' for n in names)
               + '}, "s": <number in [0,1]>}, ...]}')
    prompt = "# Prompt B\n\n" + _ask_block(names, example) + "\n" + card_text[m.start():]
    if hint is not None:
        prompt = prompt.rstrip() + "\n\n## Domain guidance\n" + hint + "\n"
    return prompt.replace("then write the prior.", "then give your answer.")


def extract_answer(text: str) -> dict | None:
    """The last JSON object in the reply that carries a "hypotheses" key."""

    decoder = json.JSONDecoder()
    for start in reversed([m.start() for m in re.finditer(r"\{", text)]):
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "hypotheses" in obj:
            return obj
    return None


def validate_hypotheses(obj: dict, dims: list[dict]) -> tuple[list[dict] | None, str]:
    """Validate every hypothesis with the colabo-llm (c, s) rules (values snapped to the grid)."""

    hypotheses = obj.get("hypotheses")
    if not isinstance(hypotheses, list) or not hypotheses:
        return None, '"hypotheses" must be a non-empty list of {"c": ..., "s": ...} objects'
    parsed = []
    for k, entry in enumerate(hypotheses, start=1):
        if not isinstance(entry, dict):
            return None, f'hypothesis {k} must be a JSON object with keys "c" and "s"'
        c_raw, c, s, error = validate_answer(entry, dims)
        if error:
            return None, f"hypothesis {k}: {error}"
        parsed.append({"c_raw": c_raw, "c": c, "s": s})
    return parsed, ""


def _scale(dim: dict):
    """Per-dimension encoding of the harness: min-max over the grid, log dims in log space."""

    if dim.get("values") is not None:
        if dim["type"] == "categorical" or any(isinstance(v, str) for v in dim["values"]):
            raise SystemExit(f"categorical dimension {dim['name']!r}: boxes need numeric dimensions")
        values = [float(v) for v in dim["values"]]
        low, high = min(values), max(values)
    else:
        low, high = (float(dim["range"][0]), float(dim["range"][1]))
    log = dim.get("scale") == "log" and low > 0.0
    forward = math.log if log else (lambda x: x)
    backward = math.exp if log else (lambda x: x)
    span = forward(high) - forward(low)

    def encode(x: float) -> float:
        return 0.0 if span <= 1e-12 else (forward(float(x)) - forward(low)) / span

    def decode(e: float) -> float:
        return backward(forward(low) + float(e) * span)

    return encode, decode


def derive_boxes(hypotheses: list[dict], dims: list[dict]) -> list[dict]:
    """Attach the encoded centre and the encoded and native boxes to each parsed hypothesis."""

    scales = [_scale(d) for d in dims]
    lower, upper = [0.0] * len(dims), [1.0] * len(dims)
    centers = [[encode(h["c"][d["name"]]) for d, (encode, _) in zip(dims, scales)] for h in hypotheses]
    boxes = [hypothesis_box(center, h["s"], lower, upper) for h, center in zip(hypotheses, centers)]
    _, dropped = select_hypotheses(centers, [lo for lo, _ in boxes], [hi for _, hi in boxes])
    out = []
    for index, (hypothesis, center, (lo, hi)) in enumerate(zip(hypotheses, centers, boxes)):
        out.append({
            **hypothesis,
            "kept": index not in dropped,
            **({"same_center_as": f"H{dropped[index] + 1}"} if index in dropped else {}),
            "side_encoded": 1.0 - float(hypothesis["s"]) + EPSILON,
            "center_encoded": center,
            "box_encoded": {"lb": lo.tolist(), "ub": hi.tolist()},
            "box_native": {d["name"]: [decode(a), decode(b)]
                           for d, (_, decode), a, b in zip(dims, scales, lo, hi)},
        })
    return out


def call_llm(prompt: str) -> tuple[str, dict, str | None]:
    from openai import OpenAI

    client = OpenAI(base_url=BASE_URL, api_key=os.environ.get("LLM_API_KEY", "none"), timeout=TIMEOUT)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    usage = getattr(response, "usage", None)
    tokens = {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }
    choice = response.choices[0]
    return choice.message.content or "", tokens, getattr(choice, "finish_reason", None)


def elicit(prompt: str, dims: list[dict], max_attempts: int) -> dict:
    """Ask, validate and retry with the validation error (colabo-llm policy)."""

    attempt_prompt = prompt
    attempts = []
    hypotheses, error = None, ""
    for attempt in range(1, max_attempts + 1):
        raw, tokens, finish_reason = call_llm(attempt_prompt)
        obj = extract_answer(raw)
        if obj is None:
            hypotheses, error = None, 'no JSON object with a "hypotheses" key found in the reply'
        else:
            hypotheses, error = validate_hypotheses(obj, dims)
        attempts.append({"attempt": attempt, "raw_reply": raw, "usage": tokens,
                         "finish_reason": finish_reason, "error": error or None})
        if not error:
            break
        attempt_prompt = (
            prompt
            + f"\n\nYour previous answer was invalid: {error}. "
            "Answer again; the final line must be exactly one JSON object of the requested form."
        )
    return {"hypotheses": hypotheses, "error": error, "attempts": attempts}


def hpo_jobs(adversarial: bool) -> list[dict]:
    jobs = build_adversarial_jobs() if adversarial else build_jobs()
    return [job for job in jobs if job["benchmark_id"] == "hpobench-tabular"]


def run_job(job: dict, args) -> tuple[str, str, dict | None]:
    label = f'{job["benchmark_id"]}/{job["task"]}/{job["modality"]}'
    out_dir = Path(args.out_root) / job["benchmark_id"] / job["task"]
    artifact_path = out_dir / f'{job["modality"]}.json'
    prompt_path = out_dir / f'{job["modality"]}.prompt.md'
    if artifact_path.exists() and not args.overwrite and not args.dry_run:
        return label, "SKIP (exists)", None
    card_text = job["card_path"].read_text(encoding="utf-8")
    hint = (job["hint_path"].read_text(encoding="utf-8").strip()
            if job["modality"] == ADVERSARIAL_MODALITY else None)
    prompt = build_prompt(card_text, job["card_path"], hint)
    dims = _parse_card_dims(card_text)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8", newline="\n")
    if args.dry_run:
        return label, f"DRY prompt {len(prompt)} chars -> {prompt_path.name}", None

    result = elicit(prompt, dims, args.max_attempts)
    usage = {key: sum(a["usage"][key] for a in result["attempts"])
             for key in ("prompt_tokens", "completion_tokens")}
    summary = {"label": label, "attempts": len(result["attempts"]), "usage": usage,
               "error": result["error"] or None}
    if result["error"]:
        summary["attempt_log"] = result["attempts"]
        return label, f'FAIL after {len(result["attempts"])} attempts: {result["error"]}', summary

    hypotheses = derive_boxes(result["hypotheses"], dims)
    artifact = {
        "artifact_format": "hypbo-llm-hypotheses-v1",
        "benchmark_id": job["benchmark_id"],
        "task": job["task"],
        "modality": job["modality"],
        "construction": CONSTRUCTION,
        "input_space": {
            "encoding": "per-dimension min-max over the grid values; log-scaled dimensions in log space",
            "lb": [0.0] * len(dims),
            "ub": [1.0] * len(dims),
        },
        "hypotheses": hypotheses,
        "dimensions": dims,
        "model": MODEL,
        "base_url": BASE_URL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "enable_thinking": False,
        "attempts": len(result["attempts"]),
        "usage": usage,
        "source_prompt_card": str(job["card_path"].relative_to(_ROOT)).replace("\\", "/"),
        "source_prompt_card_sha256": _sha256_text(card_text),
        **({"adversarial_hint": str(job["hint_path"].relative_to(_ROOT)).replace("\\", "/"),
            "adversarial_hint_sha256": _sha256_text(job["hint_path"].read_text(encoding="utf-8"))}
           if job.get("hint_path") else {}),
        "prompt_sha256": _sha256_text(prompt),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    artifact_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8", newline="\n")
    replies_path = Path(args.replies_root) / job["benchmark_id"] / job["task"] / f'{job["modality"]}.replies.json'
    replies_path.parent.mkdir(parents=True, exist_ok=True)
    replies_path.write_text(json.dumps({"attempt_log": result["attempts"]}, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8", newline="\n")
    summary.update(n_hypotheses=len(hypotheses), n_kept=sum(h["kept"] for h in hypotheses),
                   scores=[h["s"] for h in hypotheses])
    scores = ", ".join(f'{h["s"]:.2f}' for h in hypotheses)
    return label, f"OK   {len(hypotheses)} hypotheses, s=[{scores}] (attempts={summary['attempts']})", summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="write prompts only; no LLM calls")
    parser.add_argument("--only", default=None, help="substring filter on benchmark/task/modality")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--adversarial", action="store_true",
                        help="elicit the llm-card-misleading set instead of llm-card")
    parser.add_argument("--workers", type=int, default=1, help=f"concurrent requests (at most {MAX_WORKERS})")
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    parser.add_argument("--replies-root", default=str(REPLIES_ROOT),
                        help="raw replies of every attempt, read by scripts/check_hypbo_hypotheses.py")
    parser.add_argument("--summary-json", default=None, help="write per-job attempts, tokens and errors here")
    args = parser.parse_args()

    jobs = hpo_jobs(args.adversarial)
    if args.only:
        jobs = [j for j in jobs if args.only in f'{j["benchmark_id"]}/{j["task"]}/{j["modality"]}']
    print(f"{len(jobs)} elicitation jobs", flush=True)
    workers = max(1, min(int(args.workers), MAX_WORKERS))

    def guarded(job: dict) -> tuple[str, str, dict | None]:
        label = f'{job["benchmark_id"]}/{job["task"]}/{job["modality"]}'
        try:
            return run_job(job, args)
        except Exception as exc:  # transport or server error: no artifact, rerun later
            return label, f"ERROR {exc!r}", {"label": label, "error": repr(exc)}

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for label, status, summary in pool.map(guarded, jobs):
            print(f"{label}: {status}", flush=True)
            results.append((label, status, summary))
    failures = [(label, status) for label, status, _ in results if status.startswith(("FAIL", "ERROR"))]
    if args.summary_json:
        Path(args.summary_json).write_text(
            json.dumps([s for _, _, s in results if s is not None], indent=2) + "\n", encoding="utf-8")
    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for label, status in failures:
            print(f"  {label}: {status}")
        return 1
    print("\nall elicitations complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
