"""Synthetic prior-peak control (fig_injection_synthetic, supp/figS_injection_synthetic_per_function).

Arms (results/paper/synthetic): HapBO (alpha-pi acquisition whose candidate batch includes the prior peaks),
vanilla BO with the same prior-peak candidates and an unweighted acquisition (vanilla-peaks), vanilla BO.
Main figure: panels rho90 / well50 / well10, normalized simple regret (synthetic_data), mean +- SEM over the four
functions, log y (as fig4_synthetic_fidelity). Supplementary grid: card x function, mean +- SEM over runs (30 HapBO and vanilla-peaks
runs = 3 documents x 10 seeds; 10 vanilla runs).
"""
from __future__ import annotations

import numpy as np

import style as ST
import synthetic_data as SY
from fig_synthetic import PANELS

ARMS = [  # arm, label, colour, linestyle, linewidth, z-order
    ("hapbo", "HapBO (α-π + prior-peak candidates)", ST.C["HapBO"], "-", 2.4, 5),
    ("vanilla-peaks", "vanilla BO + prior-peak candidates", "#D55E00", "-.", 1.8, 4),
    ("vanilla", "vanilla BO", ST.C["vanilla"], "--", 1.6, 3),
]


def _band(ax, x, mean, sem, color, label, ls, lw, z):
    ax.plot(x, mean, color=color, lw=lw, ls=ls, label=label, zorder=z)
    ax.fill_between(x, np.maximum(mean - sem, 1e-12), mean + sem, color=color, alpha=.13, lw=0, zorder=z - 1)


def _sem(a):
    return a.std(0, ddof=1) / np.sqrt(len(a))


def build(fig_dir):
    import matplotlib.pyplot as plt
    ST.set_style()
    x = SY.GRID * 100
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 3.0), sharex=True, sharey=True)
    end, auc, runs = {}, {}, {}
    for i, (ax, (card, title)) in enumerate(zip(axes, PANELS)):
        cur = SY.curves([a for a, *_ in ARMS], card)
        for arm, label, color, ls, lw, z in ARMS:
            U, n = cur[arm]
            if len(U) < 4:
                raise RuntimeError(f"fig_injection_synthetic {card} {arm}: only {len(U)} functions")
            _band(ax, x, U.mean(0), _sem(U), color, label, ls, lw, z)
            end.setdefault(card, {})[arm] = round(float(U.mean(0)[-1]), 4)
            auc.setdefault(card, {})[arm] = round(float(U.mean(0).mean()), 4)
            runs[f"{card}/{arm}"] = dict(zip(SY.FAMS, n))
        ax.set_yscale("log")
        ax.set(xlim=(0, 100), xlabel="fraction of budget (%)")
        if i == 0:
            ax.set_ylabel("normalized simple regret (↓)")
        ax.set_title(title, loc="left")
        ST.clean(ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="lower center", bbox_to_anchor=(.5, .01), fontsize=6.8, handlelength=2.6)
    fig.tight_layout(rect=[0, .09, 1, 1])
    ST.save(fig, fig_dir / "fig_injection_synthetic")

    fig, axes = plt.subplots(3, 4, figsize=(7.25, 5.8), sharex=True, sharey="col")
    for r, (card, title) in enumerate(PANELS):
        for c, (fam, (name, budget)) in enumerate(SY.FAMS.items()):
            ax = axes[r, c]
            for arm, label, color, ls, lw, z in ARMS:
                rows = SY.run_rows(arm, fam, card)
                _band(ax, x, rows.mean(0), _sem(rows), color, label, ls, lw * .8, z)
            ax.set_yscale("log")
            ax.set_xlim(0, 100)
            if r == 0:
                ax.set_title(f"{name} (b{budget})", loc="left", fontsize=8.2)
            if c == 0:
                ax.set_ylabel(f"{title.split('  ', 1)[1]}\nnorm. regret (↓)", fontsize=7.4)
            if r == 2:
                ax.set_xlabel("budget (%)")
            ST.clean(ax)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="lower center", bbox_to_anchor=(.5, .005), fontsize=6.8, handlelength=2.6)
    fig.tight_layout(rect=[0, .05, 1, 1])
    ST.save(fig, fig_dir / "supp" / "figS_injection_synthetic_per_function")
    return {"fig_injection_synthetic": {"end_of_budget": end, "mean_over_budget": auc, "runs_per_function": runs}}
