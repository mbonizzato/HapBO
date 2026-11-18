#!/usr/bin/env python3
"""Validate, plan, or execute a versioned research study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from experiments.catalog import RESULT_ROOT, load_catalog, load_study
from experiments.execution import run_planned_run
from experiments.planner import build_study_plan, slug


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path, help="study directory or study.yaml")
    parser.add_argument("--execute", action="store_true", help="run instead of plan only")
    parser.add_argument("--run-id", action="append", default=[], help="execute only this exact run ID; repeatable")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--write-plan", action="store_true")
    parser.add_argument(
        "--skip-plan-write",
        action="store_true",
        help="do not (re)write plan.json (when several processes execute runs of one study)",
    )
    args = parser.parse_args()

    study = load_study(args.study)
    plan = build_study_plan(study, load_catalog())
    root = RESULT_ROOT / slug(study["phase"]) / slug(study["study_id"])
    if (args.write_plan or args.execute) and not args.skip_plan_write:
        _write_json(root / "plan.json", plan.to_dict())

    executable = [run for run in plan.runs if run.executable]
    blocked = [run for run in plan.runs if not run.executable]
    print(
        f"Study {study['study_id']}: {len(plan.runs)} planned, "
        f"{len(executable)} executable, {len(blocked)} blocked"
    )
    if blocked:
        for run in blocked[:10]:
            print(f"  BLOCKED {run.run_id}: {'; '.join(run.blocked_reasons)}")
        if len(blocked) > 10:
            print(f"  ... and {len(blocked) - 10} more (see plan.json with --write-plan)")
    if not args.execute:
        return 0

    selected = executable
    if args.run_id:
        requested = set(args.run_id)
        selected = [run for run in executable if run.run_id in requested]
        missing = requested - {run.run_id for run in selected}
        if missing:
            parser.error(f"unknown or blocked run IDs: {sorted(missing)}")
    if not selected:
        parser.error("no executable runs selected")

    failures = 0
    for position, run in enumerate(selected, start=1):
        print(f"[{position}/{len(selected)}] {run.run_id}")
        status_path = run.result_directory / "status.json"
        if status_path.is_file():
            existing = json.loads(status_path.read_text(encoding="utf-8"))
            if existing.get("status") == "completed":
                print("  SKIPPED: completed result already exists")
                continue
            failures += 1
            print(
                f"  EXISTING {existing.get('status', 'unknown').upper()}: "
                f"inspect {status_path} before assigning a new study ID or seed"
            )
            if not args.continue_on_error:
                break
            continue
        try:
            status = run_planned_run(run)
        except FileExistsError as exc:
            print(f"  SKIPPED: {exc}")
            continue
        if status["status"] != "completed":
            failures += 1
            print(f"  FAILED: see {run.result_directory / 'stderr.log'}")
            if not args.continue_on_error:
                break
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
