#!/usr/bin/env python3
"""Run the synthetic panel: every arm x card x function x seed (and draw for HapBO).

Functions: Hartmann-6 (budget 40), Styblinski-Tang-7 (50), Levy-10 (50), Ackley-15 (150); seeds 0-9.
Cards: rho90, well50, well10 (experiments/landscape_estimates/<fam>/card_formula_<card>_domain.txt).
The hidden objective is the well world of scripts/synthetic_truth.py: every runner is started with
SYNTH_TRUTH=well.

Arms and runners (interpreter of each from the environment; default: this python):
  hapbo     scripts/run_synthetic_arms.py --arms hapbo --prior-file
            experiments/synthetic_priors/<card>/<fam>/draw<d>/prior.json
            --alpha-override 0.325 (3 draws x 10 seeds)                                     HAPBO_PYTHON
  hapbo-pointonly, hapbo-gaussian
            the same with experiments/synthetic_priors/rho90/<fam>/draw<d>-{pointonly,gaussian}/prior.json:
            the representation ablation, card rho90 only (other cards are skipped)          HAPBO_PYTHON
  vanilla, random   scripts/run_synthetic_arms.py --arms vanilla,random (card-independent)   HAPBO_PYTHON
  turbo     scripts/run_synthetic_arms.py --arms turbo (card-independent; torch + gpytorch) TURBO_PYTHON
  colabo    scripts/colabo_llm_continuous.py --cs-json
            experiments/colabo_llm_priors/synthetic-continuous-<fam>/<fam>-<card>.json, one process per seed
            (ColaBO's memory grows across in-process runs)                                    COLABO_PYTHON
  llambo    scripts/llambo_continuous.py --mode zero-shot (LLAMBO Mode 2)                     LLAMBO_PYTHON
  bora      scripts/bora_continuous.py (default_precision 4)                                  BORA_PYTHON
  lgbo      scripts/lgbo_continuous.py (paper-equation LGBO)                                  LGBO_PYTHON

Control arms (run only when named in --arms):
  vanilla-peaks
            scripts/run_synthetic_arms.py --arms vanilla-peaks with the hapbo prior files and flags: vanilla EI
            whose candidate batch includes the prior peaks of the hapbo arm (3 draws x 10 seeds)  HAPBO_PYTHON
  pi-bo, prior-to-mean, prior-warm-start, prior-greedy
            scripts/run_synthetic_arms.py --arms <arm> with the hapbo prior files: the prior-injection arms of
            fig_ablations on the synthetic panel (3 draws x 10 seeds)                          HAPBO_PYTHON

The LLM arms (colabo beliefs are precomputed; llambo, bora, lgbo call the LLM in the loop) need the endpoint
LLM_BASE_URL / LLM_MODEL / LLM_API_KEY; LLAMBO, BORA and ColaBO need their pinned checkouts (external/README.md).

Outputs: results/synthetic/well/<card|none>/<arm>/... (each runner's own resumable JSON format).
Every runner skips completed seeds, so the driver can be rerun after an interruption.

usage: python scripts/run_synthetic_panel.py [--arms hapbo,vanilla,...] [--cards rho90,well50,well10]
                                             [--families hartmann6,...] [--seeds 10] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ["hartmann6", "styblinski_tang7", "levy10", "ackley15"]
SLUG = {"hartmann6": "hartmann6", "styblinski_tang7": "styblinski-tang7", "levy10": "levy10", "ackley15": "ackley15"}
CARDS = ["rho90", "well50", "well10"]
ARMS = ["hapbo", "hapbo-pointonly", "hapbo-gaussian", "vanilla", "random", "turbo", "colabo", "llambo", "bora",
        "lgbo"]
REPRESENTATION_ARMS = ["hapbo-pointonly", "hapbo-gaussian"]
REPRESENTATION_CARD = "rho90"  # card of the representation ablation
INJECTION_ARMS = ["pi-bo", "prior-to-mean", "prior-warm-start", "prior-greedy"]
CONTROL_ARMS = ["vanilla-peaks", *INJECTION_ARMS]
DRAWS = (1, 2, 3)
ALPHA = "0.325"
TRUTH = "well"  # hidden objective of the panel (scripts/synthetic_truth.py)


def py(var: str) -> str:
    return os.environ.get(var) or sys.executable


def rho_arg(card: str) -> str:
    """Card key in the --rho form of llambo_continuous / lgbo_continuous ('90', 'well50', 'well10')."""
    return card[3:] if card.startswith("rho") else card


def commands(arm: str, cards, families, seeds: int):
    out = f"results/synthetic/{TRUTH}"
    if arm in ("vanilla", "random", "turbo"):
        if arm == "turbo":
            yield [py("TURBO_PYTHON"), "scripts/run_synthetic_arms.py", "--label", "turbo", "--out-root",
                   f"{out}/none", "--arms", "turbo", "--families", ",".join(families), "--seeds", str(seeds)]
        else:
            yield [py("HAPBO_PYTHON"), "scripts/run_synthetic_arms.py", "--label", arm, "--out-root",
                   f"{out}/none", "--arms", arm, "--families", ",".join(families), "--seeds", str(seeds)]
        return
    for card in cards:
        if arm in REPRESENTATION_ARMS and card != REPRESENTATION_CARD:
            continue
        for fam in families:
            if arm.startswith("hapbo"):
                suffix = arm[len("hapbo"):]
                for d in DRAWS:
                    prior = f"experiments/synthetic_priors/{card}/{fam}/draw{d}{suffix}/prior.json"
                    yield [py("HAPBO_PYTHON"), "scripts/run_synthetic_arms.py", "--label", f"{fam}-draw{d}",
                           "--out-root", f"{out}/{card}/{arm}", "--families", fam, "--arms", "hapbo",
                           "--prior-file", prior, "--prior-id", f"{card}/{fam}/draw{d}{suffix}",
                           "--alpha-override", ALPHA, "--seeds", str(seeds)]
            elif arm == "vanilla-peaks":
                for d in DRAWS:
                    prior = f"experiments/synthetic_priors/{card}/{fam}/draw{d}/prior.json"
                    yield [py("HAPBO_PYTHON"), "scripts/run_synthetic_arms.py", "--label", f"{fam}-draw{d}",
                           "--out-root", f"{out}/{card}/{arm}", "--families", fam, "--arms", arm,
                           "--prior-file", prior, "--prior-id", f"{card}/{fam}/draw{d}",
                           "--alpha-override", ALPHA, "--seeds", str(seeds)]
            elif arm in INJECTION_ARMS:
                for d in DRAWS:
                    prior = f"experiments/synthetic_priors/{card}/{fam}/draw{d}/prior.json"
                    yield [py("HAPBO_PYTHON"), "scripts/run_synthetic_arms.py", "--label", f"{fam}-draw{d}",
                           "--out-root", f"{out}/{card}/{arm}", "--families", fam, "--arms", arm,
                           "--prior-file", prior, "--prior-id", f"{card}/{fam}/draw{d}", "--seeds", str(seeds)]
            elif arm == "colabo":
                cs = f"experiments/colabo_llm_priors/synthetic-continuous-{SLUG[fam]}/{SLUG[fam]}-{card}.json"
                for s in range(seeds):
                    yield [py("COLABO_PYTHON"), "scripts/colabo_llm_continuous.py", "--family", fam, "--cs-json", cs,
                           "--only-seed", str(s), "--out-root", f"{out}/{card}/colabo", "--out-name", f"{fam}-{card}"]
            elif arm == "llambo":
                yield [py("LLAMBO_PYTHON"), "scripts/llambo_continuous.py", "--family", fam, "--mode", "zero-shot",
                       "--rho", rho_arg(card), "--seeds", str(seeds), "--out-root", f"{out}/{card}/llambo"]
            elif arm == "bora":
                yield [py("BORA_PYTHON"), "scripts/bora_continuous.py", "--family", fam, "--rho", card,
                       "--seeds", str(seeds), "--out-root", f"{out}/{card}/bora"]
            elif arm == "lgbo":
                yield [py("LGBO_PYTHON"), "scripts/lgbo_continuous.py", "--family", fam, "--rho", rho_arg(card),
                       "--seeds", str(seeds), "--out-root", f"{out}/{card}/lgbo"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--cards", default=",".join(CARDS))
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x]
    cards = [x for x in a.cards.split(",") if x]
    fams = [x for x in a.families.split(",") if x]
    bad = (set(arms) - set(ARMS + CONTROL_ARMS)) | (set(cards) - set(CARDS)) | (set(fams) - set(FAMILIES))
    if bad:
        ap.error(f"unknown values {sorted(bad)}")
    env = dict(os.environ, SYNTH_TRUTH=TRUTH)
    failed = []
    for arm in arms:
        for cmd in commands(arm, cards, fams, a.seeds):
            line = " ".join(shlex.quote(c) for c in cmd[1:])
            print(f"[{arm}] {line}", flush=True)
            if a.dry_run:
                continue
            if subprocess.run(cmd, cwd=ROOT, env=env).returncode != 0:
                failed.append(line)
    if failed:
        print(f"{len(failed)} command(s) failed:")
        for line in failed:
            print("  ", line)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
