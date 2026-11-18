#!/usr/bin/env python3
"""Elicit the cortical and SOLAR priors and ColaBO beliefs (run after scripts/build_cortical_solar_cards.py).

Steps (each skips outputs that already exist):

  hapbo-cortical   HapBO priors llm-geometry and llm-expert-hint, elicited once on left-knee-flexor from the
                   card verbatim (scripts/elicit_prior.py --prompt-card-file; seed 7, temperature 0, thinking off,
                   free-form + validate) with the neutral-example vocabulary card
                   (HAPBO_VOCABULARY_CARD=src/prior/expert_knowledge/vocabulary_card_neutral_example.md) and the
                   card's own provenance rule (HAPBO_PROVENANCE_RULE=card); only the llm-expert-hint card carries a
                   human statement (--card-human-statement), so only its beliefs may be tagged user_stated. The
                   human-expert-map prior is the human-made map (scripts/build_cortical_expert_prior.py, no LLM).
  colabo-cortical  ColaBO (c, s) belief of human-expert-map-shared on left-knee-flexor
                   (scripts/elicit_colabo_llm_priors.py).
  replicate        copy the left-knee-flexor priors and beliefs to the other muscle targets
                   (scripts/replicate_cortical_priors.py: one prior per condition across muscles).
  hapbo-solar      HapBO prior solar-maxeff-re/expert from its card (same settings, default vocabulary card).
  colabo-solar     ColaBO belief experiments/colabo_llm_priors/solar-maxeff-re/maxeff-re.json
                   (scripts/elicit_colabo_llm.py, thinking on).

Elicitation transcripts stay in the staging tree results/elicitation/. Endpoint: LLM_BASE_URL, LLM_MODEL,
LLM_API_KEY (OpenAI-compatible).

usage: python scripts/elicit_cortical_solar_priors.py [--steps hapbo-cortical,colabo-cortical,...]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "results" / "elicitation" / "priors"
LOG = ROOT / "results" / "elicitation" / "logs"
CORT = ROOT / "experiments" / "priors" / "neuromosaic-dataset1-rat"
MUSCLES = ["left-knee-flexor", "left-tibialis-anterioris", "right-tibialis-anterioris"]
SOLAR = ROOT / "experiments" / "priors" / "solar-maxeff-re" / "expert"
NEUTRAL_CARD = "src/prior/expert_knowledge/vocabulary_card_neutral_example.md"
STEPS = ["hapbo-cortical", "colabo-cortical", "replicate", "hapbo-solar", "colabo-solar"]


def endpoint_args() -> list[str]:
    return ["--model", os.getenv("LLM_MODEL", "Qwen3.5-122B-A10B-GPTQ-Int4"),
            "--base-url", os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
            "--api-key", os.getenv("LLM_API_KEY", "none")]


def run(cmd: list[str], log_name: str, env: dict | None = None) -> int:
    LOG.mkdir(parents=True, exist_ok=True)
    print("[run ]", " ".join(cmd[1:3]), "...", flush=True)
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={**os.environ, **(env or {})})
    (LOG / f"{log_name}.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr, encoding="utf-8")
    print(f"[{'ok' if p.returncode == 0 else 'FAIL'}] {log_name}", flush=True)
    return p.returncode


def elicit_card(dataset_id: str, prior_dir: Path, target_name: str | None, env: dict | None, tag: str,
                human_statement: bool = False) -> bool:
    """Elicit one prior from prior_dir/prompt_card.md in the staging tree; copy it next to its card."""
    if (prior_dir / "prior.json").is_file():
        print("[skip]", prior_dir.relative_to(ROOT).as_posix())
        return True
    stage = STAGE / prior_dir.relative_to(ROOT / "experiments" / "priors")
    if stage.is_dir():
        shutil.rmtree(stage)
    cmd = [sys.executable, "scripts/elicit_prior.py", "--dataset-id", dataset_id, "--prior-id", prior_dir.name,
           "--prompt-card-file", str(prior_dir / "prompt_card.md"), "--provider", "openai", *endpoint_args(),
           "--seed", "7", "--temperature", "0.0", "--no-structured", "--request-timeout", "1800",
           "--experiments-directory", str(STAGE)]
    if target_name:
        cmd += ["--target-name", target_name]
    if human_statement:
        cmd += ["--card-human-statement"]
    if run(cmd, tag, env) != 0 or not (stage / "prior.json").is_file():
        return False
    if (stage / "prompt_card.md").read_bytes() != (prior_dir / "prompt_card.md").read_bytes():
        print("[MISM] card sent != card file:", prior_dir)
        return False
    for name in ("prior.json", "metadata.json"):
        shutil.copy2(stage / name, prior_dir / name)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--steps", default=",".join(STEPS), help="comma list of " + ",".join(STEPS))
    a = ap.parse_args()
    steps = [s for s in a.steps.split(",") if s]
    unknown = set(steps) - set(STEPS)
    if unknown:
        ap.error(f"unknown steps {sorted(unknown)}")
    ok = True
    cortical_env = {"HAPBO_VOCABULARY_CARD": NEUTRAL_CARD, "HAPBO_PROVENANCE_RULE": "card"}
    if "hapbo-cortical" in steps:
        for pid in ("llm-geometry", "llm-expert-hint"):
            ok &= elicit_card("neuromosaic-dataset1-rat", CORT / MUSCLES[0] / pid, "LeftKneeFlexor", cortical_env,
                              f"hapbo_{pid}", human_statement=pid == "llm-expert-hint")
    if "colabo-cortical" in steps:
        for mod in ("human-expert-map-shared",):
            ok &= run([sys.executable, "scripts/elicit_colabo_llm_priors.py", "--only",
                       f"neuromosaic-dataset1-rat/{MUSCLES[0]}/{mod}"], f"colabo_{mod}") == 0
    if "replicate" in steps:
        ok &= run([sys.executable, "scripts/replicate_cortical_priors.py"], "replicate", cortical_env) == 0
    if "hapbo-solar" in steps:
        ok &= elicit_card("solar-maxeff-re", SOLAR, None, None, "hapbo_solar")
    if "colabo-solar" in steps:
        out = ROOT / "experiments" / "colabo_llm_priors" / "solar-maxeff-re" / "maxeff-re.json"
        if out.is_file():
            print("[skip]", out.relative_to(ROOT).as_posix())
        else:
            ok &= run([sys.executable, "scripts/elicit_colabo_llm.py", "--dataset-id", "solar-maxeff-re",
                       "--slug", "maxeff-re", "--prompt-card-file", str(SOLAR / "prompt_card.md"),
                       "--schema-json", str(SOLAR.parent / "schema.json"), "--enable-thinking",
                       "--max-tokens", "8192", *endpoint_args()], "colabo_solar") == 0
    print("done" if ok else "some steps failed (see results/elicitation/logs)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
