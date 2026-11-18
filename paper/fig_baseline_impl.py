"""fig_baseline_impl_sensitivity: baseline-implementation sensitivity, 32 lr/rf/xgb/nn tasks, budget 16, 10 seeds.

Columns LGBO | LLAMBO, rows standard | adversarial prior. Each panel: HapBO + the baseline's two implementations
(solid = the variant used in the main experiments, dashed = the alternative):
  LGBO    paper equation (lgbo, solid)               vs authors' public release (official-lgbo, dashed)
  LLAMBO  Mode 2 zero-shot warm start (solid)         vs Mode 1 random initialization (dashed)
Standard rows: regret / (grid maximum - grid mean); adversarial rows: regret / vanilla BO's mean query-1 regret.
Seeds averaged inside each task, mean +- SEM over tasks.
"""
from __future__ import annotations

import numpy as np

import hpo_data as H
import style as ST

FAMS4 = ["lr", "rf", "xgb", "nn"]
Q = 16


def _curve(units: dict, scale: dict):
    rows = [units[k] / scale[k] for k in units if k in scale and scale[k] > 1e-12]
    a = np.asarray(rows)
    return a.mean(0), (a.std(0, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.zeros(Q)), len(a)


def build(fig_dir):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    ST.set_style(panel_geometry=True)
    matplotlib.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    span = {}
    for f in FAMS4:
        for t in H.TASKS:
            mx, mean = H.grid_max_mean(f, t)
            span[(f, t)] = mx - mean
    van = H.task_regret("vanilla-random", "vanilla-bo-ei", FAMS4)
    q1 = {k: float(v[0]) for k, v in van.items()}
    c_hap, c_lgbo, c_ll = ST.C["HapBO"], ST.C["LGBO"], ST.C["LLAMBO"]
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.9), sharex=True)
    (aLGs, aLLs), (aLGa, aLLa) = axes
    x = np.arange(1, Q + 1)
    nums = {}

    def draw(ax, study, method, scale, color, ls, lw, key, z):
        m, e, n = _curve(H.task_regret(study, method, FAMS4), scale)
        ax.plot(x, m, color=color, ls=ls, lw=lw, zorder=z)
        ax.fill_between(x, np.maximum(m - e, 1e-12), m + e, color=color, alpha=0.14, lw=0, zorder=z - 1)
        nums[key] = (round(float(m[-1]), 4), n)
    hs, ha = ("hapbo", "alpha-pi-raw-ei-matched"), ("hapbo-misleading", "alpha-pi-raw-ei-matched")
    draw(aLGs, *hs, span, c_hap, "-", 2.4, "std HapBO", 6)
    draw(aLGs, "lgbo", "lgbo", span, c_lgbo, "-", 2.0, "std LGBO paper eq.", 5)
    draw(aLGs, "lgbo-official", "official-lgbo", span, c_lgbo, "--", 2.0, "std LGBO public release", 4)
    draw(aLLs, *hs, span, c_hap, "-", 2.4, "std HapBO", 6)
    draw(aLLs, "llambo", "llambo-ws", span, c_ll, "-", 2.0, "std LLAMBO mode 2", 5)
    draw(aLLs, "llambo-mode1", "llambo", span, c_ll, "--", 2.0, "std LLAMBO mode 1", 4)
    draw(aLGa, *ha, q1, c_hap, "-", 2.4, "adv HapBO", 6)
    draw(aLGa, "lgbo-misleading", "lgbo", q1, c_lgbo, "-", 2.0, "adv LGBO paper eq.", 5)
    draw(aLGa, "lgbo-official-misleading", "official-lgbo", q1, c_lgbo, "--", 2.0, "adv LGBO public release", 4)
    draw(aLLa, *ha, q1, c_hap, "-", 2.4, "adv HapBO", 6)
    draw(aLLa, "llambo-misleading", "llambo-ws", q1, c_ll, "-", 2.0, "adv LLAMBO mode 2", 5)
    draw(aLLa, "llambo-mode1-misleading", "llambo", q1, c_ll, "--", 2.0, "adv LLAMBO mode 1", 4)
    for ax in (aLGs, aLLs, aLGa, aLLa):
        ax.set_yscale("log"); ax.set_ylabel("norm. simple regret (↓)")
        ax.grid(axis="y", alpha=0.4); ax.set_axisbelow(True)
        ax.set_xlim(1, Q); ax.set_xticks([1, 4, 8, 12, 16])
    for ax in (aLGa, aLLa):
        ax.set_xlabel("query")
    aLGs.set_title("LGBO", fontsize=12, fontweight="bold", pad=16)
    aLLs.set_title("LLAMBO", fontsize=12, fontweight="bold", pad=16)
    aLGs.annotate("Standard prior", xy=(-0.40, 0.5), xycoords="axes fraction", rotation=90, va="center", ha="center",
                  fontsize=10, fontweight="bold")
    aLGa.annotate("Adversarial prior", xy=(-0.40, 0.5), xycoords="axes fraction", rotation=90, va="center",
                  ha="center", fontsize=10, fontweight="bold")

    def panel_legend(ax, entries):
        ax.legend([Line2D([0], [0], color=c, ls=ls, lw=2.0) for _, c, ls in entries], [t for t, _, _ in entries],
                  fontsize=7.2, loc="upper center", bbox_to_anchor=(0.5, -0.36), handlelength=2.0, framealpha=0.9,
                  borderpad=0.5, labelspacing=0.35)
    panel_legend(aLGa, [("HapBO", c_hap, "-"), ("LGBO — paper eq. (used)", c_lgbo, "-"),
                        ("LGBO — public release", c_lgbo, "--")])
    panel_legend(aLLa, [("HapBO", c_hap, "-"), ("LLAMBO — mode 2 / warmstart (used)", c_ll, "-"),
                        ("LLAMBO — mode 1 / rand.", c_ll, "--")])
    fig.tight_layout(rect=[0.07, 0.15, 1, 1])
    ST.save(fig, fig_dir / "fig_baseline_impl_sensitivity")
    return {"fig_baseline_impl_sensitivity": nums}
