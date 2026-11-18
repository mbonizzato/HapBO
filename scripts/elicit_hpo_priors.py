#!/usr/bin/env python3
"""Elicit the HapBO priors of the 40 HPOBench tasks from their prompt cards.

For each task the card experiments/priors/hpobench-<fam>-<task>/<task>/<card>/prompt_card.md (written by
scripts/build_hpo_cards.py) is split into its Domain / Search space / Optimization sections and sent
through scripts/elicit_prior.py (seed 7, temperature 0, thinking off, no structured output), writing
into a staging tree results/elicitation/priors/. The card actually sent must equal the card file
(whitespace-normalized) and the compiled prior must be finite on the benchmark's candidate grid;
accepted priors (prior.json, metadata.json, prompt_card.md) are then copied next to their card, where
the studies read them. Elicitation transcripts stay in the staging tree. Sequential and idempotent.

Cards: base (input of scripts/make_inverted_priors.py), llm-card, llm-card-misleading.
Endpoint: LLM_BASE_URL, LLM_MODEL, LLM_API_KEY (OpenAI-compatible).

usage: python scripts/elicit_hpo_priors.py --card llm-card [--only lr-car,svm-kc1] [--seed 7]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMS = ["lr", "rf", "xgb", "nn", "svm"]
TASKS = ["australian", "blood-transfusion", "car", "credit-g", "kc1", "phoneme", "segment", "vehicle"]
CARDS = ["base", "llm-card", "llm-card-misleading"]
STAGE = ROOT / "results" / "elicitation" / "priors"
LOG = ROOT / "results" / "elicitation" / "logs"


def split(card: str):
    m = re.search(r"## 1\. Domain\n(.*?)\n## 2\. Search space\n(.*?)\n## 3\. Optimization description\n(.*)$",
                  card, re.S)
    return None if m is None else [s.strip() for s in m.groups()]


def norm(t: str) -> str:
    return "\n".join(line.rstrip() for line in t.replace("\r\n", "\n").strip().splitlines())


def check_on_grid(fam: str, task: str, prior_json: Path) -> None:
    sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
    import numpy as np
    from experiments.worker import _load_offline_problem
    from prior.landscape.landscape_generator import compile_prior
    problem = _load_offline_problem("hpobench", {"model": fam, "task": task.replace("-", "_")})
    compiled = compile_prior(json.loads(prior_json.read_text(encoding="utf-8")))
    pi = np.asarray(compiled.pi(list(problem.native_candidates)), float)
    if pi.shape != (len(problem.native_candidates),) or not np.all(np.isfinite(pi)):
        raise ValueError("pi not finite on the candidate grid")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--card", required=True, choices=CARDS)
    ap.add_argument("--only", default="", help="comma list of <fam>-<task> tags (default: all 40)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--request-timeout", type=float, default=900.0)
    args = ap.parse_args()
    only = {s for s in args.only.split(",") if s}
    LOG.mkdir(parents=True, exist_ok=True)
    summary = {}
    for fam in FAMS:
        for task in TASKS:
            tag = f"{fam}-{task}"
            if only and tag not in only:
                continue
            home = ROOT / "experiments" / "priors" / f"hpobench-{fam}-{task}" / task / args.card
            card = (home / "prompt_card.md").read_text(encoding="utf-8")
            parts = split(card)
            if parts is None:
                summary[tag] = "FAIL card not parsable"
                continue
            stage = STAGE / f"hpobench-{fam}-{task}" / task / args.card
            if stage.is_dir() and not (stage / "prior.json").is_file():
                shutil.rmtree(stage)  # partial staging from an interrupted attempt
            if not (stage / "prior.json").is_file():
                work = LOG / "inputs" / args.card / tag
                work.mkdir(parents=True, exist_ok=True)
                for name, txt in zip(("domain.txt", "search_space.txt", "opt.txt"), parts):
                    (work / name).write_text(txt, encoding="utf-8")
                cmd = [sys.executable, "scripts/elicit_prior.py",
                       "--dataset-id", f"hpobench-{fam}-{task}", "--prior-id", args.card,
                       "--target-name", task,
                       "--domain-file", str(work / "domain.txt"),
                       "--search-space-file", str(work / "search_space.txt"),
                       "--optimization-file", str(work / "opt.txt"),
                       "--provider", "openai",
                       "--model", os.getenv("LLM_MODEL", "Qwen3.5-122B-A10B-GPTQ-Int4"),
                       "--base-url", os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
                       "--api-key", os.getenv("LLM_API_KEY", "none"),
                       "--seed", str(args.seed), "--temperature", "0.0", "--no-structured",
                       "--request-timeout", str(args.request_timeout), "--experiments-directory", str(STAGE)]
                print(f"[run ] {tag}", flush=True)
                p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
                (LOG / f"{args.card}_{tag}.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
                if p.returncode != 0:
                    summary[tag] = f"FAIL rc={p.returncode}: {p.stderr.strip()[-300:]}"
                    print(f"[FAIL] {tag}: {p.stderr.strip()[-300:]}", flush=True)
                    continue
            sent = (stage / "prompt_card.md").read_text(encoding="utf-8")
            if norm(sent) != norm(card):
                summary[tag] = "FAIL card sent != card file"
                print(f"[MISM] {tag}", flush=True)
                continue
            try:
                check_on_grid(fam, task, stage / "prior.json")
            except Exception as exc:  # noqa: BLE001
                summary[tag] = f"FAIL grid check: {type(exc).__name__}: {str(exc)[:200]}"
                print(f"[GRID] {tag}: {exc}", flush=True)
                continue
            for name in ("prompt_card.md", "prior.json", "metadata.json"):
                shutil.copy2(stage / name, home / name)
            summary[tag] = "ok"
            print(f"[ok  ] {tag}", flush=True)
    (LOG / f"summary_{args.card}.json").write_text(json.dumps(summary, indent=2))
    bad = {k: v for k, v in summary.items() if v != "ok"}
    print(f"SUMMARY ok={len(summary) - len(bad)} bad={len(bad)} {json.dumps(bad, indent=1)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
