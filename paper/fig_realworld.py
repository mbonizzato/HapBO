"""fig3_realworld: real-world benchmarks.

A  cortical ICMS, human-expert-map condition (every method receives the same text), all 40 rat-muscle maps,
   budget 12: exploration performance, seeds averaged within map, mean +- SEM over the maps (technical replicates).
B  SOLAR MAXEFF_RE, budget 50 shown to query 20: best feasible efficiency, mean +- SEM over seeds.
"""
from __future__ import annotations

import numpy as np

import cortical_data as CD
import solar_data as SD
import style as ST

CORT_ARMS = [  # colour key, legend, study, method
    ("HapBO", "HapBO", "human-map-all", "alpha-pi-raw-ei-matched"),
    ("ColaBO", "ColaBO", "human-map-colabo", "colabo-llm"),
    ("LLAMBO", "LLAMBO", "human-map-llambo", "llambo-ws"),
    ("LGBO", "LGBO", "human-map-lgbo", "lgbo-prior"),
    ("BORA", "BORA", "human-map-bora", "bora"),
    ("vanilla", "vanilla BO", "human-map-all", "vanilla-bo-ei"),
    ("random", "random search", "human-map-all", "random-search"),
]
SOLAR_ARMS = [  # colour key, legend, data label
    ("HapBO", "HapBO", "HapBO"), ("ColaBO", "ColaBO", "ColaBO"), ("LLAMBO", "LLAMBO", "LLAMBO"),
    ("LGBO", "LGBO", "LGBO"), ("BORA", "BORA", "BORA"), ("vanilla", "vanilla BO", "vanilla BO"),
    ("random", "random search", "random"),
]


def build(fig_dir):
    import matplotlib.pyplot as plt
    ST.set_style()
    Q, QS = CD.HORIZON, 20
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.25), gridspec_kw={"width_ratios": [1, 1.15]})
    x = np.arange(1, Q + 1)
    cort, maps = {}, {}
    for ckey, lab, study, method in CORT_ARMS:
        mu, se, _, nm = CD.explore_mean_sem(study, method, unit="map")
        ST.curve(axes[0], x, mu, se, ckey, lab)
        cort[ckey] = (round(float(mu[-1]), 4), round(float(se[-1]), 4))
        maps[ckey] = nm
    axes[0].set(xlim=(1, Q), xticks=[1, 4, 8, 12], xlabel="query", ylabel="exploration performance (↑)")
    axes[0].set_title("A  Cortical ICMS", loc="left")
    axes[0].text(0.02, 0.98, f"n = {min(maps.values())} technical replicates (muscle outputs)",
                 transform=axes[0].transAxes, ha="left", va="top", fontsize=5.9)
    ST.clean(axes[0])
    axes[0].legend(fontsize=5.9, ncol=2, loc="lower right", handlelength=1.5)
    runs = SD.load()
    sol = {}
    for ckey, lab, slabel in SOLAR_ARMS:
        rs = [r for r in runs.get(slabel, []) if len(r) >= QS]
        arr = np.asarray([SD.best_feasible(r, QS) for r in rs])
        mu, se = ST.mean_sem(arr)
        ST.curve(axes[1], np.arange(1, QS + 1), mu, se, ckey, lab)
        sol[ckey] = (round(float(mu[-1]), 1), round(float(se[-1]), 1), len(rs))
    axes[1].set(xlim=(1, QS), xticks=[1, 5, 10, 15, 20], xlabel="query",
                ylabel="best feasible efficiency (↑)")
    axes[1].set_title("B  SOLAR (unknown optimum)", loc="left")
    ST.clean(axes[1])
    axes[1].legend(fontsize=5.5, ncol=2, loc="lower right", handlelength=1.4)
    fig.tight_layout()
    ST.save(fig, fig_dir / "fig3_realworld")
    return {"fig3_realworld": {"cortical_q12": cort, "maps": maps, "solar_q20": sol}}
