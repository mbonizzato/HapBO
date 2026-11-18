#!/usr/bin/env python3
"""Build the inverted-map HapBO prior of every HPOBench task from its base prior (no LLM).

The inverted prior is a baseline plus one joint_region bump at the task's empirical anti-optimum (argmin of
the maximized ground truth, the configuration the misleading descriptions name), with a +-2-grid-step interval,
written in the base prior's own schema (so SVM keeps its lowercase `c`). The card is a copy of the base card;
the metadata is the base metadata with prior_id `inverted`, generation model null, a note and fresh sha256
digests.

Reads experiments/priors/hpobench-<fam>-<task>/<task>/base/{prior.json,prompt_card.md,metadata.json};
writes .../inverted/. Requires the HPOBench data (scripts/download_hpobench.py).

usage: python scripts/make_inverted_priors.py [--root <repo>] [--benchmark-dir benchmark/HPOBench]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from datasets.hpobench import load_hpobench_problem  # noqa: E402
from prior.landscape.landscape_generator import compile_prior, validate_prior_metadata  # noqa: E402

FAMS = ["lr", "rf", "xgb", "nn", "svm"]
TASKS = ["australian", "blood_transfusion", "car", "credit_g", "kc1", "phoneme", "segment", "vehicle"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT, help="repository root holding experiments/priors")
    ap.add_argument("--benchmark-dir", type=Path, default=None, help="default <repo>/benchmark/HPOBench")
    ap.add_argument("--no-validate", action="store_true", help="skip validate_prior_metadata (outside the repo)")
    args = ap.parse_args()
    priors = args.root.resolve() / "experiments" / "priors"
    bench = args.benchmark_dir or ROOT / "benchmark" / "HPOBench"
    done = []
    for fam in FAMS:
        for task in TASKS:
            slug = task.replace("_", "-")
            base = priors / f"hpobench-{fam}-{slug}" / slug / "base"
            if not (base / "prior.json").is_file():
                raise SystemExit(f"missing base prior {base}")
            problem = load_hpobench_problem(model=fam, task=task, benchmark_directory=bench)
            gt = np.asarray(problem.ground_truth, float)
            anti = dict(problem.native_candidates[int(np.argmin(gt))])
            anti_lc = {k.lower(): (k, v) for k, v in anti.items()}
            doc = json.loads((base / "prior.json").read_text(encoding="utf-8"))
            dims = [d["name"] for d in doc["schema"]["dimensions"]]
            mode, ci = {}, {}
            for name in dims:
                native_key, value = anti_lc[name.lower()]
                grid = sorted({float(c[native_key]) for c in problem.native_candidates})
                i = int(np.argmin([abs(g - float(value)) for g in grid]))
                mode[name] = grid[i]
                ci[name] = [grid[max(0, i - 2)], grid[min(len(grid) - 1, i + 2)]]
            doc["beliefs"] = [
                {"token": "baseline", "level": 0.05},
                {"token": "joint_region", "dims": dims, "mode": mode, "ci": ci, "weight": 1.0,
                 "provenance": "llm_domain",
                 "note": "ADVERSARIAL: bump at the empirical anti-optimum (worst config)."},
            ]
            doc["trust"] = {"overall_confidence": doc.get("trust", {}).get("overall_confidence", "medium")}
            compile_prior(doc)  # validates
            inv = base.parent / "inverted"
            inv.mkdir(parents=True, exist_ok=True)
            (inv / "prior.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
            if not (inv / "prompt_card.md").is_file() or \
                    (base / "prompt_card.md").read_bytes() != (inv / "prompt_card.md").read_bytes():
                shutil.copy(base / "prompt_card.md", inv / "prompt_card.md")
            meta = json.loads((base / "metadata.json").read_text(encoding="utf-8"))
            meta["prior_id"] = "inverted"
            gen = meta.setdefault("generation", {})
            gen["model"] = None  # no LLM
            gen["notes"] = "Inverted-map prior at the empirical anti-optimum; no LLM."
            meta["files"] = {"prompt_card.md": {"sha256": sha(inv / "prompt_card.md")},
                             "prior.json": {"sha256": sha(inv / "prior.json")}}
            (inv / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
            if not args.no_validate:
                validate_prior_metadata(inv / "prior.json")
            done.append(f"{fam}-{task}")
    print(f"built {len(done)} inverted priors")


if __name__ == "__main__":
    main()
