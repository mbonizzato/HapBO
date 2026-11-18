#!/usr/bin/env python3
"""Execute exactly one planned study run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from experiments.worker import execute_run_directory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path)
    args = parser.parse_args()
    configuration = execute_run_directory(args.run_directory)
    print(f"Completed isolated run {configuration['run_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
