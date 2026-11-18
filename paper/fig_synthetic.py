"""fig4_synthetic_fidelity: synthetic panel (budgets: Hartmann-6 40, Styblinski-Tang-7 50, Levy-10 50, Ackley-15 150).

Panels = prior card fidelity: A rho90 (rho ~ 0.9), B well50 (~0.5), C well10 (~0.1). Normalized simple regret
(synthetic_data), mean +- SEM over the four functions, log y. TuRBO, vanilla BO and random search do not read the
card (same curves in every panel).
"""
from __future__ import annotations

import numpy as np

import style as ST
import synthetic_data as SY

ARMS = [("hapbo", "HapBO", "HapBO"), ("colabo", "ColaBO", "ColaBO"), ("lgbo", "LGBO", "LGBO"),
        ("llambo", "LLAMBO", "LLAMBO"), ("bora", "BORA", "BORA"), ("turbo", "TuRBO", "TuRBO"),
        ("vanilla", "vanilla", "vanilla BO"), ("random", "random", "random search")]
PANELS = [("rho90", "A  trustworthy prior (ρ≈0.9)"), ("well50", "B  medium prior (ρ≈0.5)"),
          ("well10", "C  poor prior (ρ≈0.1)")]


def build(fig_dir):
    import matplotlib.pyplot as plt
    ST.set_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 3.0), sharex=True, sharey=True)
    numbers, seeds = {}, {}
    for i, (ax, (card, title)) in enumerate(zip(axes, PANELS)):
        cur = SY.curves([a for a, *_ in ARMS], card)
        numbers[card] = {}
        for arm, ckey, lab in ARMS:
            U, n = cur[arm]
            if len(U) < 4:
                raise RuntimeError(f"fig4_synthetic_fidelity {card} {arm}: only {len(U)} functions")
            mu, se = U.mean(0), U.std(0, ddof=1) / np.sqrt(len(U))
            ST.curve(ax, SY.GRID * 100, mu, se, ckey, lab)
            numbers[card][arm] = round(float(mu[-1]), 4)
            seeds[f"{card}/{arm}"] = dict(zip(SY.FAMS, n))
        ax.set_yscale("log")
        ax.set(xlim=(0, 100), xlabel="fraction of budget (%)")
        if i == 0:
            ax.set_ylabel("normalized simple regret (↓)")
        ax.set_title(title, loc="left")
        ST.clean(ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=8, loc="lower center", bbox_to_anchor=(.5, .01), fontsize=6.3, handlelength=1.5,
               columnspacing=0.9)
    fig.tight_layout(rect=[0, .09, 1, 1])
    ST.save(fig, fig_dir / "fig4_synthetic_fidelity")
    return {"fig4_synthetic_fidelity": {"end_of_budget": numbers, "runs_per_function": seeds}}
