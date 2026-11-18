#!/usr/bin/env python3
"""Evaluate and checkpoint complete prescreened SOLAR pools on Linux.

The resulting files are simulator-frozen objective/constraint tables.  They let a
finite-candidate external optimizer run on another host without porting the SOLAR
C++ executable, while preserving the exact pool, seed, fidelity, and replication
settings used by the SOLAR study.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets.solar import (  # noqa: E402
    SOLAR_EXECUTABLE,
    evaluate_solar,
    generate_solar_candidate_pool,
    resolve_solar_instance,
)


def encode(candidates, spec) -> np.ndarray:
    return np.column_stack([
        (np.asarray([candidate[v.name] for candidate in candidates], dtype=float) - v.lower)
        / (v.sampling_upper - v.lower)
        for v in spec.variables
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default="MAXEFF_RE")
    parser.add_argument("--seed-start", type=int, default=2001)
    parser.add_argument("--seed-end", type=int, default=2020)
    parser.add_argument("--pool-size", type=int, default=256)
    parser.add_argument("--fidelity", type=float, default=1.0)
    parser.add_argument("--replications", type=int, default=1)
    parser.add_argument("--out-root", default="results/diagnostics/solar_pool_cache")
    args = parser.parse_args()

    executable = Path(SOLAR_EXECUTABLE)
    if not executable.is_file():
        raise SystemExit(f"SOLAR binary missing at {executable}")
    spec = resolve_solar_instance(args.instance)
    out_dir = ROOT / args.out_root
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in range(args.seed_start, args.seed_end + 1):
        out_path = out_dir / f"{args.instance}_s{seed}_prescreen.json"
        partial_path = out_path.with_suffix(".partial.json")
        if out_path.is_file():
            print(f"skip completed seed {seed}", flush=True)
            continue
        candidates = generate_solar_candidate_pool(
            args.instance, args.pool_size, seed=seed, prescreen=True)
        encoded = encode(candidates, spec)
        rows = []
        if partial_path.is_file():
            rows = json.loads(partial_path.read_text(encoding="utf-8")).get("rows", [])
            print(f"resume seed {seed} at {len(rows)}/{len(candidates)}", flush=True)
        started = time.time()
        for index in range(len(rows), len(candidates)):
            evaluation = evaluate_solar(
                args.instance,
                candidates[index],
                executable=executable,
                seed=seed,
                fidelity=args.fidelity,
                replications=args.replications,
            )
            rows.append({
                "index": index,
                "candidate": dict(candidates[index]),
                "encoded": [float(v) for v in encoded[index]],
                "objective": float(evaluation.objective),
                "constraints": [float(v) for v in evaluation.constraints],
                "feasible": bool(evaluation.feasible),
                "hidden_failure": bool(evaluation.hidden_failure),
            })
            if (index + 1) % 8 == 0 or index + 1 == len(candidates):
                checkpoint = {
                    "artifact_format": "hapbo-solar-frozen-pool-v1",
                    "instance": args.instance,
                    "seed": seed,
                    "pool_size": len(candidates),
                    "prescreen": True,
                    "fidelity": args.fidelity,
                    "replications": args.replications,
                    "rows": rows,
                }
                partial_path.write_text(json.dumps(checkpoint), encoding="utf-8")
                print(f"seed {seed}: {index + 1}/{len(candidates)}", flush=True)
        payload = json.loads(partial_path.read_text(encoding="utf-8"))
        payload["elapsed_seconds_last_session"] = time.time() - started
        payload["feasible_count"] = sum(bool(row["feasible"]) for row in rows)
        payload["hidden_failure_count"] = sum(bool(row["hidden_failure"]) for row in rows)
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        partial_path.unlink()
        print(f"completed {out_path}", flush=True)


if __name__ == "__main__":
    main()

