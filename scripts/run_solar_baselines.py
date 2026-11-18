#!/usr/bin/env python3
"""Run the external SOLAR baselines (MAXEFF_RE, budget 50, seeds 2001-2020, prescreened candidate pools).

  colabo   scripts/solar_externals.py --baseline colabo --cs-json
           experiments/colabo_llm_priors/solar-maxeff-re/maxeff-re.json (peak memory about 75 GB per seed)
  lgbo     scripts/solar_externals.py --baseline lgbo --task-legend (paper-equation LGBO)
  llambo   scripts/solar_externals.py --baseline llambo --task-legend --llambo-mode zero-shot (LLAMBO Mode 2)
           -> results/diagnostics/solar_external/<stem>_penalized_MAXEFF_RE_s<seed>_prescreen.jsonl
  bora     scripts/cache_solar_pools.py (frozen candidate pools, Linux SOLAR binary), then
           scripts/bora_solar_cached.py in the BORA environment -> results/diagnostics/solar_bora/

The task text of every baseline is experiments/priors/solar-maxeff-re/expert/prompt_card.md. The HapBO,
vanilla and random arms run through the study harness (experiments/studies/solar/maxeff-re). Interpreters:
HAPBO_PYTHON (external-baseline runs, pool cache; default this python) and BORA_PYTHON; the external-baseline
runner reaches the ColaBO, LLAMBO and LGBO workers through COLABO_PYTHON / LLAMBO_PYTHON / LGBO_PYTHON. LLM
endpoint: LLM_BASE_URL, LLM_MODEL, LLM_API_KEY. Completed seeds are skipped.

usage: python scripts/run_solar_baselines.py [--baselines colabo,lgbo,llambo,bora] [--seeds 2001-2020] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTANCE = "MAXEFF_RE"
EXTERNAL_OUT = ROOT / "results" / "diagnostics" / "solar_external"
POOLS = ROOT / "results" / "diagnostics" / "solar_pool_cache"
CS_JSON = "experiments/colabo_llm_priors/solar-maxeff-re/maxeff-re.json"
BASELINES = ["colabo", "lgbo", "llambo", "bora"]


def py(var: str) -> str:
    return os.environ.get(var) or sys.executable


def external_cmd(baseline: str, seed: int, budget: int) -> tuple[list[str], Path]:
    cmd = [py("HAPBO_PYTHON"), "scripts/solar_externals.py", "--instance", INSTANCE,
           "--baseline", baseline, "--seed", str(seed), "--budget", str(budget), "--prescreen"]
    stem = baseline
    if baseline == "colabo":
        cmd += ["--cs-json", CS_JSON]
    if baseline in ("lgbo", "llambo"):  # x-space baselines: coordinate legend from the SOLAR card
        cmd += ["--task-legend"]
    if baseline == "llambo":
        cmd += ["--llambo-mode", "zero-shot"]
        stem = "llambo-ws"
    return cmd, EXTERNAL_OUT / f"{stem}_penalized_{INSTANCE}_s{seed}_prescreen.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--baselines", default=",".join(BASELINES))
    ap.add_argument("--seeds", default="2001-2020")
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    a = ap.parse_args()
    baselines = [b for b in a.baselines.split(",") if b]
    if set(baselines) - set(BASELINES):
        ap.error(f"unknown baselines {sorted(set(baselines) - set(BASELINES))}")
    lo, _, hi = a.seeds.partition("-")
    seeds = list(range(int(lo), int(hi or lo) + 1))

    jobs: list[list[str]] = []
    for b in baselines:
        if b == "bora":
            if not POOLS.is_dir() or not any(POOLS.iterdir()):
                jobs.append([py("HAPBO_PYTHON"), "scripts/cache_solar_pools.py", "--instance", INSTANCE,
                             "--seed-start", str(seeds[0]), "--seed-end", str(seeds[-1])])
            jobs.append([py("BORA_PYTHON"), "scripts/bora_solar_cached.py", "--instance", INSTANCE,
                         "--seed-start", str(seeds[0]), "--seed-end", str(seeds[-1]), "--budget", str(a.budget)])
            continue
        for s in seeds:
            cmd, out = external_cmd(b, s, a.budget)
            if out.is_file():
                print("[skip]", out.relative_to(ROOT).as_posix())
                continue
            jobs.append(cmd)

    failed = []
    for cmd in jobs:
        line = " ".join(shlex.quote(c) for c in cmd[1:])
        print("[run ]", line, flush=True)
        if a.dry_run:
            continue
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            failed.append(line)
    for line in failed:
        print("failed:", line)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
