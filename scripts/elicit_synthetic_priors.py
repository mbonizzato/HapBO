#!/usr/bin/env python3
"""Elicit the HapBO priors of the synthetic panel: one LLM call per (card, function, draw), no tools.

Prompt: experiments/synthetic_cards/ELICITATION_PROMPT.md with {TASK}, {VOCAB} and {SCHEMA} filled from
experiments/synthetic_cards/<card>/<fam>/TASK.md, experiments/synthetic_cards/PSL_VOCABULARY.md and
experiments/synthetic_cards/psl_schema.json. The model may reason; the document is the last ```json block of
the answer. It is accepted only if it parses, passes validate_record, lists TASK.md's parameters with the same
names and order, and has at most 60 tokens (every belief counts 1, an anchors belief counts its points).
Otherwise the correction turn is appended (up to 4 attempts in total).

Elicitors (--elicitor):
  qwen (default)  OpenAI-compatible endpoint LLM_BASE_URL, model LLM_MODEL, key LLM_API_KEY: thinking on,
                  temperature 0.6, top_p 0.95, seed 1000 * draw + attempt, max_tokens 32000.
  claude-haiku-5-5, claude-sonnet-5-5, claude-opus-5-5, claude-fable-5-1
                  Anthropic Messages API (key ANTHROPIC_API_KEY from the environment), streaming, max_tokens 64000
                  including thinking, each model's default sampling and thinking settings. The cost of a call is
                  computed from the list prices in PRICE_PER_MTOK (US$ per million input / output tokens).

Output: <out-root>/<card>/<fam>/draw<d>/prior.json, with out-root experiments/synthetic_priors for qwen and
results/elicitation/one-call/<elicitor> otherwise (idempotent: an existing prior.json is skipped; a draw that
fails 4 attempts writes failed.json there). --usage writes usage.json (tokens and cost of every call) next to the
prior. With --save-transcript the prompt and answers are written to results/elicitation/synthetic/<elicitor>/
<card>/<fam>/draw<d>/. The point-only and Gaussian-only variants of the rho90 documents are derived with
scripts/make_representation_variants.py --psl-docs.

usage: python scripts/elicit_synthetic_priors.py --cards rho90,well50,well10 [--draws 1-3] [--fams hartmann6,...]
       python scripts/elicit_synthetic_priors.py --elicitor claude-opus-5-5 --cards rho90 --usage
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from prior.landscape.landscape_generator import validate_record  # noqa: E402

CARDS_DIR = ROOT / "experiments" / "synthetic_cards"
OUT = ROOT / "experiments" / "synthetic_priors"
ONE_CALL = ROOT / "results" / "elicitation" / "one-call"
TRANSCRIPTS = ROOT / "results" / "elicitation" / "synthetic"
FAMS = ["hartmann6", "styblinski_tang7", "levy10", "ackley15"]
CAP = 60
CORRECTION = ("Your answer is not a valid PSL document for this task: {error}. End your answer with the corrected "
              "document as a single JSON object inside one ```json code block; only that block is read.")
# list prices, US$ per million (input, output) tokens
PRICE_PER_MTOK = {"claude-fable-5-1": (10.0, 50.0), "claude-opus-5-5": (4.0, 20.0),
                  "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}
ELICITORS = ["qwen", *PRICE_PER_MTOK]


def prompt_for(card: str, fam: str) -> tuple[str, str]:
    task = (CARDS_DIR / card / fam / "TASK.md").read_text(encoding="utf-8")
    tpl = (CARDS_DIR / "ELICITATION_PROMPT.md").read_text(encoding="utf-8")
    text = (tpl.replace("{TASK}", task)
               .replace("{VOCAB}", (CARDS_DIR / "PSL_VOCABULARY.md").read_text(encoding="utf-8"))
               .replace("{SCHEMA}", (CARDS_DIR / "psl_schema.json").read_text(encoding="utf-8")))
    return text, task


def task_dims(task_md: str) -> list[str]:
    return re.findall(r"^name:\s*(\S+)\s*$", task_md, re.M)


def lenient_tokens(doc: dict) -> int:
    return sum(len(b.get("points", [])) if b.get("token") == "anchors" else 1 for b in doc.get("beliefs", []))


def extract_json(text: str) -> dict:
    """The document is the last ```json code block (the prompt says only that block is read)."""
    blocks = re.findall(r"```json\s*(.*?)```", text, re.S)
    if not blocks:
        raise ValueError("no ```json code block found at the end of the answer")
    return json.loads(blocks[-1])


def check(doc: dict, dims: list[str]) -> None:
    validate_record(doc)
    got = [d.get("name") for d in doc.get("schema", {}).get("dimensions", [])]
    if got != dims:
        raise ValueError(f"schema.dimensions must list {dims} in this order (got {got})")
    n = lenient_tokens(doc)
    if n > CAP:
        raise ValueError(f"it has {n} tokens; the limit is {CAP}")


def call_llm(messages: list[dict], seed: int) -> tuple[str, dict]:
    url = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
    payload = json.dumps({"model": os.getenv("LLM_MODEL", "Qwen3.5-122B-A10B-GPTQ-Int4"), "messages": messages,
                          "max_tokens": 32000, "temperature": 0.6, "top_p": 0.95, "seed": seed,
                          "chat_template_kwargs": {"enable_thinking": True}}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + os.getenv("LLM_API_KEY", "none")})
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read())
    u = d.get("usage", {})
    return d["choices"][0]["message"]["content"] or "", {"input": u.get("prompt_tokens", 0),
                                                          "output": u.get("completion_tokens", 0),
                                                          "cache_read": 0, "cache_creation": 0, "cost_usd": 0.0,
                                                          "stop_reason": d["choices"][0].get("finish_reason")}


def call_claude(model: str, messages: list[dict]) -> tuple[str, dict]:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    with client.messages.stream(model=model, max_tokens=64000, messages=messages) as s:
        msg = s.get_final_message()
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    u = msg.usage
    usage = {"input": u.input_tokens, "output": u.output_tokens,
             "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
             "cache_creation": getattr(u, "cache_creation_input_tokens", 0) or 0, "stop_reason": msg.stop_reason}
    pin, pout = PRICE_PER_MTOK[model]
    usage["cost_usd"] = (usage["input"] * pin + usage["output"] * pout) / 1e6
    return text, usage


def run_one(elicitor: str, card: str, fam: str, draw: int, out_root: Path, save_usage: bool,
            save_transcript: bool) -> None:
    out = out_root / card / fam / f"draw{draw}"
    if (out / "prior.json").exists():
        print(f"skip {elicitor} {card}/{fam}/draw{draw}")
        return
    out.mkdir(parents=True, exist_ok=True)
    text_prompt, task = prompt_for(card, fam)
    dims = task_dims(task)
    messages = [{"role": "user", "content": text_prompt}]
    attempts, usage_all, t0 = [], [], time.time()
    for k in range(4):
        if elicitor == "qwen":
            text, usage = call_llm(messages, seed=1000 * draw + k)
        else:
            text, usage = call_claude(elicitor, messages)
        usage_all.append(usage)
        try:
            doc = extract_json(text)
            check(doc, dims)
            err = None
        except Exception as e:  # noqa: BLE001 -- any failure becomes the correction turn
            doc, err = None, str(e).splitlines()[0][:600]
        attempts.append({"answer": text, "error": err})
        if err is None:
            (out / "prior.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
            break
        messages += [{"role": "assistant", "content": text}, {"role": "user", "content": CORRECTION.format(error=err)}]
    accepted = (out / "prior.json").exists()
    if not accepted:
        (out / "failed.json").write_text(json.dumps({"error": attempts[-1]["error"]}, indent=1), encoding="utf-8")
    tot = {k: sum(u.get(k, 0) or 0 for u in usage_all) for k in ("input", "output", "cache_read", "cache_creation")}
    tot["cost_usd"] = round(sum(u.get("cost_usd", 0) or 0 for u in usage_all), 4)
    summary = {"elicitor": elicitor, "card": card, "family": fam, "draw": draw, "attempts": len(attempts),
               "accepted": accepted, "wall_s": round(time.time() - t0, 1), "usage": tot, "per_call": usage_all}
    if save_usage:
        (out / "usage.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if save_transcript:
        t = TRANSCRIPTS / elicitor / card / fam / f"draw{draw}"
        t.mkdir(parents=True, exist_ok=True)
        (t / "transcript.json").write_text(json.dumps({"prompt": text_prompt, "attempts": attempts}, indent=1),
                                           encoding="utf-8")
    print(f"{elicitor} {card}/{fam}/draw{draw}: accepted={accepted} attempts={len(attempts)} tokens={tot} "
          f"{summary['wall_s']} s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--elicitor", default="qwen", choices=ELICITORS)
    ap.add_argument("--cards", default="rho90,well50,well10")
    ap.add_argument("--draws", default="1-3")
    ap.add_argument("--fams", default=",".join(FAMS))
    ap.add_argument("--out-root", type=Path, default=None,
                    help="default experiments/synthetic_priors (qwen) or results/elicitation/one-call/<elicitor>")
    ap.add_argument("--usage", action="store_true", help="write usage.json (tokens, cost) next to every prior")
    ap.add_argument("--save-transcript", action="store_true")
    a = ap.parse_args()
    out_root = a.out_root or (OUT if a.elicitor == "qwen" else ONE_CALL / a.elicitor)
    lo, _, hi = a.draws.partition("-")
    for card in a.cards.split(","):
        for fam in a.fams.split(","):
            for d in range(int(lo), int(hi or lo) + 1):
                try:
                    run_one(a.elicitor, card, fam, d, out_root, a.usage, a.save_transcript)
                except Exception as e:  # noqa: BLE001 -- keep going (e.g. a transient endpoint error); calling again resumes
                    print(f"ERROR {a.elicitor} {card}/{fam}/draw{d}: {type(e).__name__}: {str(e)[:300]}", flush=True)


if __name__ == "__main__":
    main()
