"""Readers of the SOLAR MAXEFF_RE compact data in results/paper/solar/.

  maxeff-re.json   HapBO / vanilla BO / random runs of experiments/studies/solar/maxeff-re (y = [feasible, objective])
  external.json    {"ColaBO" | "LGBO" | "LLAMBO" | "BORA": [[[feasible, objective], ...] per seed]}

Metric: running maximum of the receiver efficiency (= -objective) over feasible evaluations, 0 until the first
feasible one; mean +- SEM over seeds.
"""
from __future__ import annotations

import functools
import json

import numpy as np

from hpo_data import DATA, load_study

INTERNAL = {"HapBO": "alpha-pi-raw-ei-matched", "vanilla BO": "vanilla-bo-ei", "random": "random-search"}


@functools.lru_cache(maxsize=1)
def load() -> dict:
    """Per arm label: list over seeds of evaluation records [feasible, objective] (completed runs only)."""
    out = {}
    st = load_study("solar", "maxeff-re")["methods"]
    for label, method in INTERNAL.items():
        rows = st.get(method, {}).get("budget-0050", {}).get("maxeff-re", [])
        out[label] = [r["y"] for r in rows if r["status"] == "completed" and r["y"] is not None]
    out.update(json.loads((DATA / "solar" / "external.json").read_text(encoding="utf-8")))
    return out


def best_feasible(recs, qmax: int):
    """Best-feasible curve over the first qmax records; a run shorter than qmax is not padded (callers drop it)."""
    if len(recs) < qmax:
        raise ValueError(f"run has {len(recs)} records, fewer than the {qmax} queries shown")
    out, best = np.zeros(qmax), 0.0
    for j in range(qmax):
        if recs[j][0]:
            best = max(best, -float(recs[j][1]))
        out[j] = best
    return out
