"""supp/figS_cake_vs_hapbo_b16: CAKE and CALIPER vs HapBO on HPOBench, budget 16 (40 tasks, 10 seeds).

Rows: good prior per family, adversarial prior per family (HapBO prior elicited from the card + misleading hint;
CALIPER per-role priors elicited from the same cards; CAKE uses no prior), then the two grand means.
Metric: normalized simple regret 1 - best/opt (seed mean per task, mean +- SEM over tasks).
"""
from __future__ import annotations

import matplotlib
import numpy as np

import hpo_data as H
import style as ST

FAMILIES = ["lr", "rf", "xgb", "nn", "svm"]
FAMILY_NAMES = {"lr": "linear models", "rf": "random forest", "xgb": "XGBoost", "nn": "neural net", "svm": "SVM"}
ARMS = ["hapbo", "caliper", "cake", "vanilla", "random"]
SRC = {"good": {"hapbo": ("hapbo", "alpha-pi-raw-ei-matched"), "caliper": ("caliper", "caliper"),
                "cake": ("cake", "cake"), "vanilla": ("vanilla-random", "vanilla-bo-ei"),
                "random": ("vanilla-random", "random-search")},
       "adv": {"hapbo": ("hapbo-misleading", "alpha-pi-raw-ei-matched"), "caliper": ("caliper-misleading", "caliper"),
               "cake": ("cake", "cake"), "vanilla": ("vanilla-random", "vanilla-bo-ei"),
               "random": ("vanilla-random", "random-search")}}
DISP = {"hapbo": "HapBO", "caliper": "CALIPER", "cake": "CAKE", "vanilla": "vanilla", "random": "random"}
LEG = {"hapbo": "HapBO (ours)", "caliper": "CALIPER", "cake": "CAKE", "vanilla": "vanilla BO",
       "random": "random search"}


def build(fig_dir):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    ST.set_style(panel_geometry=True)
    rows = {c: {arm: {f: H.hpo_curve(*SRC[c][arm], [f]) for f in FAMILIES} for arm in ARMS} for c in SRC}
    q = np.arange(1, 17)
    fig = plt.figure(figsize=(9.1, 7.4))
    gs = GridSpec(3, 5, figure=fig, height_ratios=[1.0, 1.0, 1.35], hspace=.62, wspace=.42,
                  left=.075, right=.99, top=.95, bottom=.135)
    big = ("HapBO", "CALIPER", "CAKE")
    for r, (cond, lab) in enumerate((("good", "good prior"), ("adv", "adversarial prior"))):
        for i, fam in enumerate(FAMILIES):
            ax = fig.add_subplot(gs[r, i])
            for arm in ARMS:
                m, e = H.mean_sem(rows[cond][arm][fam])
                ST.curve(ax, q, m, e, DISP[arm], lw=2.2 if DISP[arm] in big else 1.15,
                         z=6 if DISP[arm] in big else 3)
            ax.set(xlim=(1, 16), xticks=[1, 4, 8, 12, 16], xlabel="query")
            ax.set_title(FAMILY_NAMES[fam], fontsize=8, loc="left")
            ST.clean(ax)
            if i == 0:
                ax.set_ylabel(f"normalized regret (↓)\n[{lab}]")
    summary = {}
    for cond, sl, title in (("good", slice(0, 2), "Grand mean — good prior"),
                            ("adv", slice(3, 5), "Grand mean — adversarial prior")):
        ax = fig.add_subplot(gs[2, sl])
        nt = 0
        for arm in ARMS:
            rs = [x for f in FAMILIES for x in rows[cond][arm][f]]
            m, e = H.mean_sem(rs)
            ST.curve(ax, q, m, e, DISP[arm], LEG[arm], z=6 if DISP[arm] in big else 3)
            summary.setdefault(cond, {})[LEG[arm]] = (round(float(m[-1]), 4), len(rs))
            nt = max(nt, len(rs))
        ax.set(xlim=(1, 16), xticks=[1, 4, 8, 12, 16], xlabel="query", ylabel="normalized simple regret (↓)")
        ax.set_title(f"{title} ({nt} tasks)", loc="left")
        ST.clean(ax)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, ncol=5, loc="lower center", bbox_to_anchor=(.54, .005),
               fontsize=matplotlib.rcParams["axes.titlesize"], handlelength=1.8, columnspacing=1.3)
    ST.save(fig, fig_dir / "supp" / "figS_cake_vs_hapbo_b16")
    return {"figS_cake_vs_hapbo_b16": summary}
