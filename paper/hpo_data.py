"""Readers of the compact study data in results/paper/ (HPOBench part + the shared run reader).

results/paper/<group>/<study>.json:
  {"benchmark": ..., "fields": [...], "methods": {method: {"budget-XXXX": {unit: [{"seed", "status", "y"}]}}}}
  y = the recorded field of every evaluation (ground-truth objective per query for HPOBench and cortical runs).
results/paper/hpo/task_stats.json: {"<fam>-<task>": {"max", "min", "mean", "optimum"}} of the full HPOBench grid.

HPOBench metrics:
  simple regret          = grid maximum - best ground-truth accuracy so far (seed mean per task)
  normalized regret      = 1 - best / grid maximum
  grid-span normalized   = regret / (optimum - grid mean)
  vanilla-q1 normalized  = regret / vanilla BO's seed-mean regret at query 1
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "paper"
FAMS = ["lr", "rf", "xgb", "nn", "svm"]
TASKS = ["australian", "blood_transfusion", "car", "credit_g", "kc1", "phoneme", "segment", "vehicle"]
BUDGET = 16


@functools.lru_cache(maxsize=None)
def load_study(group: str, name: str) -> dict:
    return json.loads((DATA / group / f"{name}.json").read_text(encoding="utf-8"))


def runs(group: str, study: str, method: str, unit: str, budget: int = BUDGET) -> list:
    """Run records [{seed, status, y}] of one (study, method, unit, budget), sorted by seed; [] if absent."""
    return load_study(group, study)["methods"].get(method, {}).get(f"budget-{budget:04d}", {}).get(unit, [])


def has_unit(group: str, study: str, method: str, unit: str, budget: int = BUDGET) -> bool:
    return unit in load_study(group, study)["methods"].get(method, {}).get(f"budget-{budget:04d}", {})


def best_curve(run: dict, budget: int):
    """Best-so-far objective over the first `budget` queries of a completed run, else None."""
    if run["status"] != "completed" or run["y"] is None or len(run["y"]) < budget:
        return None
    return np.maximum.accumulate(np.asarray(run["y"][:budget], float))


def unit(fam: str, task: str) -> str:
    return f"{fam}-{task.replace('_', '-')}"


@functools.lru_cache(maxsize=None)
def task_stats() -> dict:
    return json.loads((DATA / "hpo" / "task_stats.json").read_text(encoding="utf-8"))


def grid_max(fam: str, task: str) -> float:
    return task_stats()[unit(fam, task)]["max"]


def grid_span(fam: str, task: str) -> float:
    """optimum - mean grid accuracy."""
    s = task_stats()[unit(fam, task)]
    return s["optimum"] - s["mean"]


def grid_max_mean(fam: str, task: str):
    s = task_stats()[unit(fam, task)]
    return s["max"], s["mean"]


def seed_regret(studies, method: str, fam: str, task: str, budget: int = BUDGET):
    """Seed-mean simple-regret curve of one task (first study holding the task), or None; also the run count."""
    studies = [studies] if isinstance(studies, str) else list(studies)
    st = next((s for s in studies if has_unit("hpo", s, method, unit(fam, task), budget)), None)
    if st is None:
        return None, 0
    opt = grid_max(fam, task)
    curves = [opt - c for c in (best_curve(r, budget) for r in runs("hpo", st, method, unit(fam, task), budget))
              if c is not None]
    return (np.mean(curves, axis=0), len(curves)) if curves else (None, 0)


def hpo_curve(study: str, method: str, fams, budget: int = BUDGET) -> list:
    """Per task (fams x TASKS): seed-mean normalized regret 1 - best / grid maximum."""
    rows = []
    for f in fams:
        for t in TASKS:
            opt = grid_max(f, t)
            cs = [c for c in (best_curve(r, budget) for r in runs("hpo", study, method, unit(f, t), budget))
                  if c is not None]
            if cs:
                rows.append(np.mean([1.0 - c / opt for c in cs], axis=0))
    return rows


def task_regret(study: str, method: str, fams, budget: int = BUDGET) -> dict:
    """{(fam, task): seed-mean simple-regret curve}."""
    out = {}
    for f in fams:
        for t in TASKS:
            opt = grid_max(f, t)
            cs = [c for c in (best_curve(r, budget) for r in runs("hpo", study, method, unit(f, t), budget))
                  if c is not None]
            if cs:
                out[(f, t)] = np.mean([opt - np.asarray(c, float) for c in cs], axis=0)
    return out


def collect(arms: dict, condition: str, fams=FAMS, budget: int = BUDGET):
    """Seed-mean simple-regret curve per (fam, task) per arm.

    arms: label -> ({"std": study or [studies], "adv": ...}, method). Returns (data, run counts)."""
    data, counts = {}, {}
    for label, (studies, method) in arms.items():
        per_task, n = {}, 0
        for f in fams:
            for t in TASKS:
                c, k = seed_regret(studies[condition], method, f, t, budget)
                if c is not None:
                    per_task[(f, t)] = c
                    n += k
        if n:
            data[label], counts[label] = per_task, n
    return data, counts


def mean_sem(rows):
    a = np.asarray(rows, float)
    return a.mean(0), (a.std(0, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.zeros(a.shape[1]))
