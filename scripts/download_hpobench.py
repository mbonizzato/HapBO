#!/usr/bin/env python3
"""Download selected official HPOBench tabular benchmark tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from zipfile import BadZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = PROJECT_ROOT / "src"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from datasets.hpobench import (  # noqa: E402
    HPOBENCH_DIRECTORY,
    MODELS,
    TASKS,
    download_hpobench_data,
    hpobench_task_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download official HPOBench tabular archives and extract only the "
            "eight classification tasks selected by LLaMBO."
        )
    )
    parser.add_argument(
        "--model",
        nargs="+",
        choices=MODELS,
        default=list(MODELS),
        help="Model bundles to download (default: all five).",
    )
    parser.add_argument(
        "--task",
        nargs="+",
        choices=tuple(TASKS),
        default=list(TASKS),
        help="Tasks to extract (default: all eight selected tasks).",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=HPOBENCH_DIRECTORY,
        help=(
            "Benchmark root (default: "
            f"{HPOBENCH_DIRECTORY.relative_to(PROJECT_ROOT)})."
        ),
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep the large model ZIP archives after selected files are extracted.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and replace files that are already present.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check expected files without downloading anything.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify_only:
        missing = []
        for model in args.model:
            for task in args.task:
                for path in hpobench_task_path(
                    model,
                    task,
                    benchmark_directory=args.output_directory,
                ):
                    if not path.is_file():
                        missing.append(path)
        if missing:
            print("Missing HPOBench files:", file=sys.stderr)
            for path in missing:
                print(f"  {path}", file=sys.stderr)
            return 1
        print(
            f"Verified {len(args.model) * len(args.task)} "
            "HPOBench model/task combinations."
        )
        return 0

    print(
        "Preparing HPOBench model bundles. The upstream archives are large; "
        "only the selected task files will be retained."
    )
    try:
        manifest = download_hpobench_data(
            models=args.model,
            tasks=args.task,
            benchmark_directory=args.output_directory,
            keep_archives=args.keep_archives,
            force=args.force,
        )
    except (BadZipFile, OSError, ValueError) as exc:
        print(f"HPOBench download failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Prepared {len(manifest['extracted'])} combinations under "
        f"{args.output_directory.expanduser().resolve()}"
    )
    print(
        "Manifest: "
        f"{args.output_directory.expanduser().resolve() / 'download_manifest.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
