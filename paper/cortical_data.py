"""Readers of the cortical (NeuroMOSAIC rat M1) compact data in results/paper/cortical/.

  <study>.json                 run trajectories (ground-truth mean response per query), as hpo_data.load_study
  map_optimum.json             best true mean response per map
  exploration/<study>.json     exploration performance per run: {"<map>/<method>/budget-<B>/<seed>": [score per query]}
                               (scripts/score_exploration.py: true mean of the GP-recommended tested electrode / map optimum)
  prior_sources.json           compiled prior fields pi(x) and three normalized example truth maps
                               (supp/figS2_prior_sources_and_trajectories)

Every cortical run has budget HORIZON = 12; the figures read the `budget-0012` runs (budget_key) and evaluate
their 12 queries.

Metrics, averaged over seeds within each rat-muscle map; unit="rat" then averages maps within rat (mean +- SEM over
the 6 rats), unit="map" keeps one curve per map (mean +- SEM over the maps, technical replicates):
  normalized performance   = best true mean so far / map optimum
  exploration performance  = exploration score (every run scored to HORIZON is averaged; a run that fails at or
                             before HORIZON is a failure)
"""
from __future__ import annotations

import functools
import json

import numpy as np

from hpo_data import DATA, best_curve, load_study

HORIZON = 12


def budget_key(keys) -> str | None:
    """The smallest `budget-XXXX` key among `keys` whose budget is at least HORIZON, else None."""
    ok = sorted(int(k.split("-")[1]) for k in keys if int(k.split("-")[1]) >= HORIZON)
    return f"budget-{ok[0]:04d}" if ok else None


@functools.lru_cache(maxsize=None)
def map_optimum() -> dict:
    return json.loads((DATA / "cortical" / "map_optimum.json").read_text(encoding="utf-8"))


def _group(prob: str, unit: str) -> str:
    return prob.split("-")[0] if unit == "rat" else prob


def cort_curve(study: str, method: str, unit: str = "rat") -> list:
    """Per-rat (or per-map) normalized-performance curves (best / map optimum) over the HORIZON queries."""
    budgets = load_study("cortical", study)["methods"].get(method, {})
    units = budgets.get(budget_key(budgets), {})
    groups: dict = {}
    for u in sorted(units):
        opt = map_optimum()[u]
        cs = [c for c in (best_curve(r, HORIZON) for r in units[u]) if c is not None]
        if cs:
            groups.setdefault(_group(u, unit), []).append(np.mean([c / opt for c in cs], axis=0))
    return [np.mean(v, axis=0) for v in groups.values()]


@functools.lru_cache(maxsize=None)
def exploration(study: str) -> dict:
    f = DATA / "cortical" / "exploration" / f"{study}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}


def _explore_runs(study: str, method: str) -> dict:
    """{map: [exploration curve over the HORIZON queries, one per seed]} of one method."""
    runs = {tuple(k.split("/")): sc for k, sc in exploration(study).items()}
    bud = budget_key({b for _, m, b, _ in runs if m == method})
    by_map: dict = {}
    for (prob, meth, b, _seed), sc in runs.items():
        if meth == method and b == bud and len(sc) >= HORIZON:
            by_map.setdefault(prob, []).append(np.asarray(sc[:HORIZON], float))
    return by_map


def explore_curve(study: str, method: str, unit: str = "rat") -> list:
    """Per-rat (seeds and muscles averaged within rat) or per-map (seeds averaged) exploration-performance curves
    (maps with a scored seed)."""
    groups: dict = {}
    for prob, curves in _explore_runs(study, method).items():
        groups.setdefault(_group(prob, unit), []).append(np.mean(curves, axis=0))
    return [np.mean(v, axis=0) for v in groups.values()]


def n_maps(study: str, method: str) -> int:
    """Maps with at least one scored seed."""
    return len(_explore_runs(study, method))


def explore_mean_sem(study: str, method: str, unit: str = "rat"):
    """(mean, SEM over rats or maps, number of rats or maps, maps) of the exploration curve over the HORIZON
    queries."""
    from style import mean_sem
    rows = explore_curve(study, method, unit)
    mu, se = mean_sem(np.asarray(rows))
    return mu, se, len(rows), n_maps(study, method)


@functools.lru_cache(maxsize=None)
def prior_sources() -> dict:
    return json.loads((DATA / "cortical" / "prior_sources.json").read_text(encoding="utf-8"))
