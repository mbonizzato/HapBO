"""Execute isolated planned runs and preserve complete success/failure records."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .catalog import PROJECT_ROOT
from .planner import PlannedRun


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_planned_run(run: PlannedRun) -> dict[str, Any]:
    """Execute one unblocked run without overwriting a completed attempt."""

    if not run.executable or run.command is None:
        raise ValueError(f"run {run.run_id} is blocked: {list(run.blocked_reasons)}")
    directory = run.result_directory
    status_path = directory / "status.json"
    if status_path.exists():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        raise FileExistsError(
            f"run directory already contains status={existing.get('status')!r}: {directory}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable if value == "{python}" else value for value in run.command]
    run_record = run.to_dict()
    run_record["resolved_command"] = command
    run_record["environment"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": str(PROJECT_ROOT),
    }
    _write_json(directory / "study_run.json", run_record)
    started_at = datetime.now(timezone.utc).isoformat()
    _write_json(
        status_path,
        {
            "artifact_format": "llm-bo-study-run-status-v1",
            "run_id": run.run_id,
            "status": "running",
            "started_at_utc": started_at,
        },
    )
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            check=False,
        )
    except OSError as exc:
        elapsed = time.perf_counter() - started
        (directory / "stdout.log").write_text("", encoding="utf-8")
        (directory / "stderr.log").write_text(str(exc) + "\n", encoding="utf-8")
        status = {
            "artifact_format": "llm-bo-study-run-status-v1",
            "run_id": run.run_id,
            "status": "failed",
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_time_seconds": elapsed,
            "return_code": None,
            "failure_stage": "process_start",
        }
        _write_json(status_path, status)
        return status
    elapsed = time.perf_counter() - started
    (directory / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (directory / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    status = {
        "artifact_format": "llm-bo-study-run-status-v1",
        "run_id": run.run_id,
        "status": "completed" if completed.returncode == 0 else "failed",
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_time_seconds": elapsed,
        "return_code": completed.returncode,
        "failure_stage": None if completed.returncode == 0 else "optimization_process",
    }
    _write_json(status_path, status)
    return status
