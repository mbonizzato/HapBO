from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets import (  # noqa: E402
    HPOBENCH_DIRECTORY,
    NEUROMOSAIC_DATASET1_RAT_DEFAULT_SUBJECT_PATH,
)


class BenchmarkLayoutTests(unittest.TestCase):
    def test_all_default_payloads_resolve_under_benchmark(self) -> None:
        benchmark_root = PROJECT_ROOT / "benchmark"
        self.assertTrue(HPOBENCH_DIRECTORY.is_relative_to(benchmark_root))
        self.assertTrue(
            NEUROMOSAIC_DATASET1_RAT_DEFAULT_SUBJECT_PATH.is_relative_to(benchmark_root)
        )
        self.assertIn(
            "benchmark/NeuroMOSAIC/dataset1/rat",
            NEUROMOSAIC_DATASET1_RAT_DEFAULT_SUBJECT_PATH.as_posix(),
        )


if __name__ == "__main__":
    unittest.main()
