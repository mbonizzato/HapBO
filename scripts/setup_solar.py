#!/usr/bin/env python3
"""Fetch, patch, build, and verify the pinned SOLAR black-box simulator."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from datasets.solar import (  # noqa: E402
    SOLAR_COMMIT,
    SOLAR_DIRECTORY,
    SOLAR_REPOSITORY,
    SOLAR_SOURCE_DIRECTORY,
    evaluate_solar,
    generate_solar_candidate_pool,
    write_solar_search_spaces,
)


PATCHES = (
    {
        "id": "qualified-std-isnan",
        "path": "src/HeatExchanger.cpp",
        "old": "if ( isnan(T_in_ms) || isnan(T_out_ms) )",
        "new": "if ( std::isnan(T_in_ms) || std::isnan(T_out_ms) )",
    },
    {
        "id": "mincost-ch-output-formatter",
        "path": "src/Evaluator.cpp",
        "old": 'else if ( pb_id == "MINCOST_C2" )',
        "new": 'else if ( pb_id == "MINCOST_C2" || pb_id == "MINCOST_CH" )',
    },
)


def _run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=True, text=True)


def _clone_sparse(destination: Path) -> None:
    if shutil.which("git") is None:
        raise RuntimeError("git is required to download SOLAR")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--no-checkout",
            SOLAR_REPOSITORY,
            str(destination),
        ]
    )
    _run(["git", "sparse-checkout", "init", "--no-cone"], cwd=destination)
    _run(
        [
            "git",
            "sparse-checkout",
            "set",
            "/README.md",
            "/LICENSE",
            "/src/",
            "/bin/README.md",
            "/tests/*/param.txt",
            "/tests/*/x0.txt",
        ],
        cwd=destination,
    )
    _run(["git", "checkout", "--detach", SOLAR_COMMIT], cwd=destination)


def _verify_checkout(source: Path) -> None:
    if not (source / ".git").exists():
        raise RuntimeError(
            f"{source} exists but is not the managed SOLAR checkout; move it aside "
            "and run this command again"
        )
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )
    actual = completed.stdout.strip()
    if actual != SOLAR_COMMIT:
        raise RuntimeError(
            f"SOLAR checkout is at {actual}, not the supported commit {SOLAR_COMMIT}. "
            "Move benchmark/SOLAR/source aside and rerun setup."
        )


def _apply_compatibility_patches(source: Path) -> list[str]:
    applied: list[str] = []
    for patch in PATCHES:
        path = source / patch["path"]
        # Preserve upstream CRLF/LF choices so the managed patch changes one
        # source expression instead of mechanically rewriting whole files.
        with path.open("r", encoding="utf-8", newline="") as handle:
            content = handle.read()
        if patch["new"] in content:
            applied.append(patch["id"])
            continue
        if patch["old"] not in content:
            raise RuntimeError(
                f"cannot apply compatibility patch {patch['id']}: expected source "
                f"text was not found in {path}"
            )
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content.replace(patch["old"], patch["new"], 1))
        applied.append(patch["id"])
    return applied


def _smoke_test(executable: Path) -> dict[str, object]:
    outputs: dict[str, object] = {}
    for instance in ("MAXNRG_H1", "MINCOST_CH"):
        candidate = generate_solar_candidate_pool(instance, 2, seed=0)[0]
        evaluation = evaluate_solar(instance, candidate, executable=executable)
        outputs[instance] = {
            "objective": evaluation.objective,
            "constraint_count": int(evaluation.constraints.size),
            "feasible": evaluation.feasible,
        }
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install the pinned, sparse SOLAR source checkout and compile it."
    )
    parser.add_argument(
        "--source-directory",
        type=Path,
        default=SOLAR_SOURCE_DIRECTORY,
        help="Managed source checkout (default: benchmark/SOLAR/source)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="Parallel make jobs (default: up to 4)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not clone or build; verify the existing executable with two queries",
    )
    parser.add_argument(
        "--full-check",
        action="store_true",
        help="Also run SOLAR's complete 10-20 minute validation suite",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    source = args.source_directory.expanduser().resolve()
    executable = source / "bin" / "solar"

    if not source.exists():
        if args.verify_only:
            raise FileNotFoundError(f"SOLAR checkout does not exist: {source}")
        print(f"Fetching sparse SOLAR checkout at {SOLAR_COMMIT} ...")
        _clone_sparse(source)
    _verify_checkout(source)

    patches: list[str] = []
    if not args.verify_only:
        patches = _apply_compatibility_patches(source)
        if shutil.which("make") is None:
            raise RuntimeError("make is required to compile SOLAR")
        print(f"Building SOLAR with {args.jobs} job(s) ...")
        _run(["make", f"-j{args.jobs}"], cwd=source / "src")
    if not executable.is_file():
        raise FileNotFoundError(f"SOLAR executable was not produced: {executable}")

    print("Running focused smoke tests ...")
    smoke_tests = _smoke_test(executable)
    if args.full_check:
        _run([str(executable), "-check"], cwd=source)

    search_spaces = write_solar_search_spaces(SOLAR_DIRECTORY)
    manifest = {
        "artifact_format": "llm-bo-solar-installation-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": SOLAR_REPOSITORY,
        "commit": SOLAR_COMMIT,
        "source_directory": str(source),
        "executable": str(executable),
        "compatibility_patches": patches or [patch["id"] for patch in PATCHES],
        "smoke_tests": smoke_tests,
        "full_check_run": bool(args.full_check),
        "search_spaces": [str(path) for path in search_spaces],
    }
    manifest_path = SOLAR_DIRECTORY / "installation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"SOLAR is ready: {executable}")
    print(f"Installation record: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
