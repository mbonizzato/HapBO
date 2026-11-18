#!/usr/bin/env python3
"""Train every grid configuration of the leak-control CNN (750 configs x 3 seeds), one process per config.

Each worker runs `train_cnn.py --task-id <i> --threads 1` (single-threaded convolutions; parallelism comes from
running configs side by side). Configs whose result file already exists are skipped by train_cnn.py, so an
interrupted sweep resumes where it stopped. Afterwards aggregate with build_table.py.

usage (from the repository root):
  python leak_control_cnn/prepare_data.py --dataset fashion-of
  python leak_control_cnn/sweep_cnn.py --dataset fashion-of --workers 8
  python leak_control_cnn/build_table.py --dataset fashion-of
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
N_CONFIGS = 750


def run_one(task_id: int, dataset: str) -> int:
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    cmd = [sys.executable, str(HERE / "train_cnn.py"), "--task-id", str(task_id), "--dataset", dataset, "--threads", "1"]
    return subprocess.run(cmd, env=env).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="fashion-of")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--start", type=int, default=0, help="first config index")
    ap.add_argument("--stop", type=int, default=N_CONFIGS, help="one past the last config index")
    a = ap.parse_args()
    ids = range(a.start, a.stop)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        codes = list(ex.map(lambda i: run_one(i, a.dataset), ids))
    failed = [i for i, c in zip(ids, codes) if c != 0]
    print(f"{len(ids) - len(failed)}/{len(ids)} configs finished" + (f"; failed: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
