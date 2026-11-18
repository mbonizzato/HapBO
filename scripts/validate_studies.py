#!/usr/bin/env python3
"""Validate all research catalogs and study manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from experiments.catalog import STUDY_ROOT, load_catalog, load_study
from experiments.planner import build_study_plan


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    catalog = load_catalog()
    paths = sorted(STUDY_ROOT.glob("**/study.yaml"))
    total_runs = 0
    blocked = 0
    for path in paths:
        plan = build_study_plan(load_study(path), catalog)
        total_runs += len(plan.runs)
        blocked += plan.blocked_count
        print(f"OK {path.relative_to(PROJECT_ROOT)} ({len(plan.runs)} runs, {plan.blocked_count} blocked)")
    print(f"Validated {len(paths)} studies: {total_runs} runs, {blocked} blocked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

