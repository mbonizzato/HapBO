#!/usr/bin/env python3
"""Elicit the colabo-llm (c, s) prior from the local LLM (BORA supplement Sec. C.2).

Given the SAME expert-intuition prompt card (no benchmark names) that HapBO's elicitor consumes,
ask the LLM ONCE, offline, for exactly two things:

* ``c`` -- a d-dimensional guessed optimum location (native parameter space), and
* ``s`` -- a scalar confidence in [0, 1].

These become the ColaBO expert prior N(c, sigma) with ``sigma = 1 - s + eps`` (eps=1e-6),
exactly as the BORA paper built their ColaBO baseline. No PSL, no landscape -- deliberately
simpler than HapBO's compiler. The LLM then disappears; ColaBO consumes only this artifact.

Writes ``experiments/colabo_llm_priors/<dataset_id>/<slug>.json`` with provenance (model id,
prompt hash, c, s); the raw reply is written to results/elicitation/colabo/<dataset_id>/<slug>.txt.
One-shot, offline, local open-weights model.

Usage:
    python scripts/elicit_colabo_llm.py \
        --dataset-id synthetic-continuous-hartmann6 --slug hartmann6 \
        --domain-file d.txt --optimization-file o.txt --schema-json dims.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

_INSTRUCTION = """

## 4. Your task (single expert guess)
You are a domain expert. Based ONLY on the qualitative intuition and structure above, give your
single best guess for the location of the optimum, and how confident you are.

Respond with ONE JSON object and nothing else, in exactly this form:
{"c": [<one number per parameter, in the order and native range listed above>],
 "s": <a single confidence between 0 and 1>}

Rules:
- "c" must have exactly one entry per parameter, each within that parameter's stated range.
- "s" is your overall confidence that the optimum is near "c" (0 = no idea, 1 = certain).
- Do not add any other keys, comments, or text. No exact optimum was given; this is your guess.
"""

# Synthetic analytic-model cards: the card carries an approximate model, not qualitative intuition.
_INSTRUCTION_FORMULA = """

## 4. Your task (single expert guess)
Based ONLY on the information above, including the approximate analytic model, give your single best guess
for the location of the optimum of the TRUE objective, and how confident you are.

Respond with ONE JSON object and nothing else, in exactly this form:
{"c": [<one number per parameter, in the order and native range listed above>],
 "s": <a single confidence between 0 and 1>}

Rules:
- "c" must have exactly one entry per parameter, each within that parameter's stated range.
- "s" is your overall confidence that the optimum is near "c" (0 = no idea, 1 = certain).
- Do not add any other keys, comments, or text.
"""


def _text(inline, file):
    if file:
        return Path(file).read_text(encoding="utf-8")
    return inline or ""


def _extract_json(raw: str) -> dict:
    # Prefer a fenced block, else the last {...} span.
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    cand = m.group(1) if m else None
    if cand is None:
        spans = re.findall(r"\{.*?\}", raw, re.DOTALL)
        cand = spans[-1] if spans else raw
    return json.loads(cand)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--domain-file", default=None)
    ap.add_argument("--optimization-file", default=None)
    ap.add_argument("--schema-json", required=True)
    ap.add_argument("--model", default=os.getenv("LLM_MODEL", "Qwen3.5-122B-A10B-GPTQ-Int4"))
    ap.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--api-key", default=os.getenv("LLM_API_KEY", "none"))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--request-timeout", type=float, default=180.0)
    ap.add_argument("--enable-thinking", action="store_true",
                    help="turn Qwen thinking mode on (needed for formula-analysis cards)")
    ap.add_argument("--out-root", default=str(_ROOT / "experiments" / "colabo_llm_priors"))
    ap.add_argument("--prompt-card-file", default=None,
                    help="use this rendered card (from its '## 1. Domain' on) instead of rendering "
                         "domain/optimization files; the PSL-writing instruction is not sent")
    ap.add_argument("--instruction", default="default", choices=["default", "formula"],
                    help="'formula' for analytic-model (synthetic) cards")
    args = ap.parse_args()
    base_instruction = _INSTRUCTION_FORMULA if args.instruction == "formula" else _INSTRUCTION

    from prior.expert_knowledge.prompt_card import render_prompt_card, search_space_from_schema

    raw_schema = json.loads(Path(args.schema_json).read_text(encoding="utf-8"))
    dims = raw_schema if isinstance(raw_schema, list) else (
        raw_schema.get("dimensions") or raw_schema.get("schema", {}).get("dimensions", [])
    )
    d = len(dims)
    bounds = [tuple(map(float, dd["range"])) for dd in dims]

    if args.prompt_card_file:
        full = Path(args.prompt_card_file).read_text(encoding="utf-8")
        card = "# Prompt B" + chr(10) + chr(10) + full[full.index("## 1. Domain"):]
        card = card.replace("then write the prior.", "then give your answer.")
        card = card.replace("Questions to consider before writing the prior", "Questions to consider")
        instruction = (base_instruction.replace("## 4. Your task", "## 5. Your task") if "## 4." in card
                       else base_instruction)
    else:
        domain = _text(None, args.domain_file)
        optimization = _text(None, args.optimization_file)
        search_space = search_space_from_schema(dims)
        card = render_prompt_card(
            domain=domain, search_space=search_space, optimization_description=optimization
        )
        instruction = base_instruction
    prompt = card + instruction
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    resp = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=args.temperature,
        **({"top_p": args.top_p} if args.top_p is not None else {}),
        max_tokens=args.max_tokens,
        timeout=args.request_timeout,
        extra_body={"chat_template_kwargs": {"enable_thinking": bool(args.enable_thinking)}},
    )
    raw = resp.choices[0].message.content or ""
    parsed = _extract_json(raw)

    c = [float(v) for v in parsed["c"]]
    s = float(parsed["s"])
    if len(c) != d:
        raise ValueError(f"expected {d} coords, got {len(c)}")
    # Clamp c to each parameter's stated range (guard against small out-of-range slips).
    c = [min(max(ci, lo), hi) for ci, (lo, hi) in zip(c, bounds)]
    s = min(max(s, 0.0), 1.0)
    eps = 1e-6
    sigma = 1.0 - s + eps  # BORA Sec. C.2 formula (in native units, per-dimension)

    out_dir = Path(args.out_root) / args.dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "artifact_format": "colabo-llm-prior-v1",
        "dataset_id": args.dataset_id,
        "target_name": args.slug,
        "construction": "BORA supplement Sec. C.2: pi(x)=N(c, sigma=1-s+eps), eps=1e-6",
        "dim": d,
        "bounds": bounds,
        "c": c,
        "s": s,
        "sigma": sigma,
        "eps": eps,
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "enable_thinking": bool(args.enable_thinking),
        "prompt_hash": prompt_hash,
    }
    raw_path = _ROOT / "results" / "elicitation" / "colabo" / args.dataset_id / f"{args.slug}.txt"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(raw, encoding="utf-8", newline="\n")
    out_path = out_dir / f"{args.slug}.json"
    out_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"colabo-llm prior for {args.slug}: c={c} s={s:.3f} sigma={sigma:.6f}")
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
