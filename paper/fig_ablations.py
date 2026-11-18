"""fig_ablations: prior injection site.

  A  HPOBench budget 16, standard prior (40 lr/rf/xgb/nn/svm tasks)  normalized simple regret 1 - best/opt
  B  HPOBench budget 16, adversarial prior (elicited from card + misleading hint)
  C  synthetic rho ~ 0.5 (card well50), the injection arms on the HapBO documents, vanilla BO and random search
     (grand mean over 4 functions, synthetic_data normalization)
"""
from __future__ import annotations

import numpy as np

import hpo_data as H
import style as ST
import synthetic_data as SY

FAMS = H.FAMS  # all 40 HPOBench tasks
ARMS = [  # method, label, color, linestyle, marker, lw
    ("alpha-pi-raw-ei-matched", r"$\alpha$-$\pi$ (ours)", "#0072B2", "-", None, 2.4),
    ("pi-bo-ei", r"$\pi$-BO", "#D55E00", "-", None, 1.8),
    ("prior-to-mean", r"prior$\rightarrow$mean", "#C0A000", "--", None, 1.8),
    ("prior-warm-start", "prior warm-start", "#CC79A7", "-", "o", 1.8),
    ("prior-greedy", "prior-greedy (no BO)", "#009E73", ":", None, 1.8),
    ("vanilla-bo-ei", "vanilla BO", "#4D4D4D", "--", None, 1.6),
    ("random-search", "random", "#B3B3B3", ":", None, 1.6),
]
_INJ = ("pi-bo-ei", "prior-to-mean", "prior-warm-start", "prior-greedy")
INJECTION_SRC = {  # panel -> method -> study
    "A": {"alpha-pi-raw-ei-matched": "hapbo", **{m: "injection" for m in _INJ},
          "vanilla-bo-ei": "vanilla-random", "random-search": "vanilla-random"},
    "B": {"alpha-pi-raw-ei-matched": "hapbo-misleading", **{m: "injection-misleading" for m in _INJ},
          "vanilla-bo-ei": "vanilla-random", "random-search": "vanilla-random"},
}
SYN_INJ_CARD = "well50"
SYN_INJ = {  # panel C: method -> synthetic arm (synthetic/well50.json; vanilla and random: none.json)
    "alpha-pi-raw-ei-matched": "hapbo", "pi-bo-ei": "pi-bo", "prior-to-mean": "prior-to-mean",
    "prior-warm-start": "prior-warm-start", "prior-greedy": "prior-greedy", "vanilla-bo-ei": "vanilla",
    "random-search": "random",
}


def hpo_rows(study, method):
    return H.hpo_curve(study, method, FAMS)


def build(fig_dir):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    ST.set_style_panels(1.5)
    fig = plt.figure(figsize=(13.0, 4.1))
    gs = GridSpec(1, 3, figure=fig, wspace=0.30, left=0.065, right=0.995, top=0.91, bottom=0.14)
    t1 = {"A": "A  HPOBench b16 (good prior)", "B": "B  HPOBench b16 (adversarial prior)",
          "C": "C  synthetic ρ≈0.5 (grand mean, 4 fns)"}
    zs = {"alpha-pi-raw-ei-matched": 5, "vanilla-bo-ei": 3, "random-search": 3}
    summary = {}
    q_hpo = np.arange(1, H.BUDGET + 1)
    x = SY.GRID * 100
    for c, key in enumerate("ABC"):
        ax = fig.add_subplot(gs[0, c])
        if key == "C":
            cur = SY.curves([SYN_INJ[m] for m, *_ in ARMS], SYN_INJ_CARD)
            for method, label, color, ls, mk, lw in ARMS:
                U, seeds = cur[SYN_INJ[method]]
                m, e = U.mean(0), U.std(0, ddof=1) / np.sqrt(len(U))
                ax.plot(x, m, color=color, lw=lw, ls=ls, label=label, zorder=zs.get(method, 4), marker=mk,
                        markevery=10, markersize=3.2, markeredgewidth=0)
                ax.fill_between(x, np.maximum(m - e, 1e-6), m + e, color=color, alpha=0.13, lw=0)
                summary.setdefault(key, {})[label] = (round(float(m[-1]), 4), seeds)
            ax.set_yscale("log")
            ax.set_xlim(0, 100); ax.set_xlabel("fraction of budget (%)")
        else:
            for method, label, color, ls, mk, lw in ARMS:
                rows = hpo_rows(INJECTION_SRC[key][method], method)
                m, e = H.mean_sem(rows)
                ST.curve_role(ax, q_hpo, m, e, color=color, label=label, lw=lw, ls=ls, z=zs.get(method, 4), marker=mk)
                summary.setdefault(key, {})[label] = (round(float(m[-1]), 4), len(rows))
            ax.set_xlim(1, q_hpo[-1]); ax.set_xticks([1, 4, 8, 12, 16]); ax.set_xlabel("query")
        ax.set_title(t1[key], loc="left")
        ax.set_ylabel("normalized simple regret (↓)")
        ST.value_grid(ax, "y"); ST.despine(ax)
        if c == 0:
            ax.legend(fontsize=9.5, loc="upper right", ncol=1, handlelength=1.5, framealpha=0.9)
    ST.save(fig, fig_dir / "fig_ablations")
    return {"fig_ablations": summary}
