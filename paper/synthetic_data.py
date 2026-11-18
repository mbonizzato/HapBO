"""Readers of the synthetic-panel compact data in results/paper/synthetic/.

  <card>.json   card = rho90 | well50 | well10: {arm: {family: [best-so-far simple-regret trajectory per run]}}
                arms hapbo (3 elicitation draws x 10 seeds), colabo, lgbo, llambo, bora, vanilla-peaks (vanilla
                BO with the hapbo prior-peak candidates; 3 draws x 10 seeds); rho90 also pointonly, gaussian
                (representation ablation, fig_representation; 3 draws x 10 seeds); well50 also pi-bo,
                prior-to-mean, prior-warm-start, prior-greedy (fig_ablations panel C; the hapbo documents, 3 draws
                x 10 seeds)
  none.json     card-independent arms vanilla, random, turbo (10 seeds)

Normalization (fig4_synthetic_fidelity, fig_ablations panel C, fig_representation): every trajectory is divided by vanilla BO's mean first-query regret on the
same function and interpolated onto a 101-point grid over the fraction of the function's budget; a method's curve
is the mean over the four functions.
"""
from __future__ import annotations

import functools
import json

import numpy as np

from hpo_data import DATA

FAMS = {"hartmann6": ("Hartmann-6", 40), "styblinski_tang7": ("Styblinski-Tang-7", 50), "levy10": ("Levy-10", 50),
        "ackley15": ("Ackley-15", 150)}
GRID = np.linspace(0, 1, 101)
NONE_ARMS = ("vanilla", "random", "turbo")


@functools.lru_cache(maxsize=None)
def _load(name: str) -> dict:
    return json.loads((DATA / "synthetic" / f"{name}.json").read_text(encoding="utf-8"))


def trajs(arm: str, fam: str, card: str) -> list:
    src = _load("none") if arm in NONE_ARMS else _load(card)
    return src.get(arm, {}).get(fam, [])


def run_rows(arm: str, fam: str, card: str) -> np.ndarray:
    """Normalized curves of every run of `arm` on `fam` (array [runs x 101], in stored run order)."""
    budget = FAMS[fam][1]
    ref = float(np.mean([v[0] for v in trajs("vanilla", fam, card)]))
    return np.asarray([np.interp(GRID, np.linspace(0, 1, len(t[:budget])), np.asarray(t[:budget], float) / ref)
                       for t in trajs(arm, fam, card) if len(t) >= 2])


def curves(arms, card: str) -> dict:
    """Per arm: (array [n_functions x 101] of function-mean normalized curves, runs per function)."""
    out = {}
    for arm in arms:
        fam_means, seeds = [], []
        for fam in FAMS:
            rows = run_rows(arm, fam, card)
            if len(rows):
                fam_means.append(rows.mean(0))
                seeds.append(len(trajs(arm, fam, card)))
        out[arm] = (np.asarray(fam_means), seeds)
    return out
