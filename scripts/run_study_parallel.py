#!/usr/bin/env python3
"""Execute a study's runs in parallel (thread pool over run_planned_run).

run_study.py --execute is serial; each PlannedRun spawns its own isolated
run_experiment.py subprocess, so a thread pool of N gives N concurrent subprocess
runs.  Idempotent: completed runs (status.json == "completed") are skipped, so this
is safe to re-run / resume.

    python scripts/run_study_parallel.py \
        experiments/studies/cortical/acquisition --workers 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from experiments.catalog import load_catalog, load_study  # noqa: E402
from experiments.execution import run_planned_run  # noqa: E402
from experiments.planner import build_study_plan  # noqa: E402


def _completed(run) -> bool:
    p = run.result_directory / "status.json"
    if not p.is_file():
        return False
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("status") == "completed"
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("study", type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="cap #runs (smoke test)")
    args = ap.parse_args()

    plan = build_study_plan(load_study(args.study), load_catalog())
    runs = [r for r in plan.runs if r.executable and not _completed(r)]
    if args.limit:
        runs = runs[: args.limit]
    total = len(runs)
    print(f"study={args.study.name}: {total} runs to execute, workers={args.workers}",
          flush=True)
    if not total:
        print("nothing to do (all completed).")
        return 0

    done = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_planned_run, r): r for r in runs}
        for fut in as_completed(futs):
            run = futs[fut]
            try:
                ok = fut.result().get("status") == "completed"
            except Exception as exc:  # noqa: BLE001
                ok = False
                print(f"  ERROR {run.run_id}: {exc}", flush=True)
            done += ok
            fail += (not ok)
            n = done + fail
            if n % 50 == 0 or n == total:
                rate = n / max(time.time() - t0, 1e-9)
                eta = (total - n) / max(rate, 1e-9)
                print(f"  {n}/{total} done={done} fail={fail} "
                      f"({rate:.1f}/s, eta {eta/60:.1f} min)", flush=True)
    print(f"DONE study={args.study.name} completed={done} failed={fail} "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
