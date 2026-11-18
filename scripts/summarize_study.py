#!/usr/bin/env python3
"""Generate run-level, failure, and aggregate CSVs for one study result tree."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from experiments.summary import summarize_study


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_result_directory", type=Path)
    args = parser.parse_args()
    for output in summarize_study(args.study_result_directory):
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

