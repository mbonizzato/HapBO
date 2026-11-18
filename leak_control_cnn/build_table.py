"""Aggregate per-config sweep results into the offline benchmark table.

Reads ``results/result_*.json`` (one per config, written by train_cnn.py) and
writes ``benchmark/cnn_leak/table_<dataset>.json`` as
``{config_key: {mean, sd, seeds, n_seeds}}`` -- the exact structure the
``cnn_leak`` loader expects. Reports coverage and the discovered optimum.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))

from datasets.cnn_leak import (  # noqa: E402
    DEFAULT_DATASET,
    config_key,
    enumerate_grid,
    table_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--resultdir", default=None, help="default: results/<dataset>/ (per-regime)")
    parser.add_argument("--allow-partial", action="store_true",
                        help="write the table even if some configs are missing")
    args = parser.parse_args()

    grid = enumerate_grid()
    resultdir = Path(args.resultdir) if args.resultdir else (_HERE / "results" / args.dataset)
    records: dict[str, dict] = {}
    for f in sorted(resultdir.glob("result_*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if rec.get("dataset", args.dataset) != args.dataset:
            continue
        records[rec["config_key"]] = rec

    table: dict[str, dict] = {}
    missing = []
    for c in grid:
        key = config_key(c)
        rec = records.get(key)
        if rec is None:
            missing.append(key)
            continue
        table[key] = {
            "mean": float(rec["mean"]),
            "sd": float(rec["sd"]),
            "seeds": list(rec.get("seed_accuracies", [])),
            "n_seeds": len(rec.get("seeds", [])),
        }

    print(f"coverage: {len(table)}/{len(grid)} configs; {len(missing)} missing")
    if missing and not args.allow_partial:
        print("  first missing:", missing[:5])
        print("  (pass --allow-partial to write anyway)")
        return 1

    out = table_path(args.dataset)
    out.write_text(json.dumps(table, indent=0), encoding="utf-8")

    means = np.array([v["mean"] for v in table.values()])
    best_key = max(table, key=lambda k: table[k]["mean"])
    print(f"wrote {out}")
    print(f"  val-accuracy range [{means.min():.4f}, {means.max():.4f}], "
          f"mean {means.mean():.4f}, sd-across-grid {means.std():.4f}")
    print(f"  optimum: {best_key} -> {table[best_key]['mean']:.4f} "
          f"(seed sd {table[best_key]['sd']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
