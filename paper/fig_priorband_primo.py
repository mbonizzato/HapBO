"""supp/figS_priorband_primo_vs_hapbo_b16: PriorBand and PriMO vs HapBO on HPOBench, budget 16 (40 tasks,
10 seeds).

Rows: good prior per family, adversarial prior per family (card plus the misleading hint; PriorBand and PriMO get a point
prior elicited from the same cards), then the two grand means.
HapBO, vanilla BO and random search: x = query. PriorBand and PriMO (NePS 0.17.0, native multi-fidelity): x = cumulative
full-fidelity-equivalent cost (each evaluation costs z/z_max), curve = best full-fidelity evaluation so far.
A point is averaged only where every run behind it is defined (PriorBand from cost 1, PriMO from cost 3).
Metric: normalized simple regret 1 - best/opt (seed mean per task, mean +- SEM over tasks).
"""
from __future__ import annotations

import matplotlib
import numpy as np

import hpo_data as H
import style as ST

FAMILIES = ["lr", "rf", "xgb", "nn", "svm"]
FAMILY_NAMES = {"lr": "linear models", "rf": "random forest", "xgb": "XGBoost", "nn": "neural net", "svm": "SVM"}
ARMS = ["hapbo", "priorband", "primo", "vanilla", "random"]
MF = ("priorband", "primo")  # multi-fidelity arms: x = cumulative cost
SRC = {"good": {"hapbo": ("hapbo", "alpha-pi-raw-ei-matched"), "priorband": ("priorband", "priorband"),
                "primo": ("primo", "primo"), "vanilla": ("vanilla-random", "vanilla-bo-ei"),
                "random": ("vanilla-random", "random-search")},
       "adv": {"hapbo": ("hapbo-misleading", "alpha-pi-raw-ei-matched"),
               "priorband": ("priorband-misleading", "priorband"), "primo": ("primo-misleading", "primo"),
               "vanilla": ("vanilla-random", "vanilla-bo-ei"), "random": ("vanilla-random", "random-search")}}
DISP = {"hapbo": "HapBO", "priorband": "PriorBand", "primo": "PriMO", "vanilla": "vanilla", "random": "random"}
LEG = {"hapbo": "HapBO (ours)", "priorband": "PriorBand", "primo": "PriMO", "vanilla": "vanilla BO",
       "random": "random search"}
SEEDS = 10
COST_TOL = 1e-9


def query_curve(run: dict, opt: float):
    c = H.best_curve(run, H.BUDGET)
    return None if c is None else [1.0 - v / opt for v in c]


def cost_curve(run: dict, opt: float):
    """Best full-fidelity value among evaluations with cumulative cost <= x, x = 1..16 (None before the first)."""
    if run["status"] != "completed" or run["y"] is None:
        return None
    out = []
    for x in range(1, H.BUDGET + 1):
        vals = [v for cost, v in run["y"] if cost <= x + COST_TOL]
        out.append(None if not vals else 1.0 - max(vals) / opt)
    return out


def task_curves(study: str, method: str, mf: bool) -> dict:
    """{unit: [per-seed curve]} over the 40 tasks; every task must hold all 10 completed runs."""
    out = {}
    for f in FAMILIES:
        for t in H.TASKS:
            unit = H.unit(f, t)
            opt = H.grid_max(f, t)
            curves = [(cost_curve if mf else query_curve)(r, opt) for r in H.runs("hpo", study, method, unit)]
            curves = [c for c in curves if c is not None]
            if len(curves) != SEEDS:
                raise ValueError(f"{study}/{method}/{unit}: {len(curves)} completed runs, expected {SEEDS}")
            out[unit] = curves
    return out


def aggregate(per_task: dict, units: list):
    """Per x: seed mean per task, then mean +- SEM over tasks, only where every run is defined."""
    m, e, n = np.full(H.BUDGET, np.nan), np.full(H.BUDGET, np.nan), [0] * H.BUDGET
    for x in range(H.BUDGET):
        task_means = []
        for u in units:
            vals = [c[x] for c in per_task[u]]
            if any(v is None for v in vals):
                task_means = None
                break
            task_means.append(float(np.mean(vals)))
        if task_means:
            a = np.asarray(task_means)
            m[x] = a.mean()
            e[x] = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
            n[x] = len(a)
    return m, e, n


def repeat_share(study: str, method: str) -> dict:
    rs = [r for f in FAMILIES for t in H.TASKS for r in H.runs("hpo", study, method, H.unit(f, t))]
    units = [r["repeat_units"] / H.BUDGET for r in rs]
    return {"repeat_share_of_budget_mean": round(float(np.mean(units)), 4), "runs": len(rs)}


def build(fig_dir):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    ST.set_style(panel_geometry=True)
    data = {c: {arm: task_curves(*SRC[c][arm], arm in MF) for arm in ARMS} for c in SRC}
    x = np.arange(1, H.BUDGET + 1)
    fig = plt.figure(figsize=(9.1, 7.4))
    gs = GridSpec(3, 5, figure=fig, height_ratios=[1.0, 1.0, 1.35], hspace=.62, wspace=.42,
                  left=.075, right=.99, top=.95, bottom=.135)
    big = ("HapBO", "PriorBand", "PriMO")

    def plot(ax, per_task, units, arm):
        m, e, n = aggregate(per_task, units)
        ok = ~np.isnan(m)
        lw, z = (2.2, 6) if DISP[arm] in big else (1.15, 3)
        ax.plot(x[ok], m[ok], color=ST.C[DISP[arm]], ls="-", lw=lw, label=LEG[arm], zorder=z)
        hi = m[ok] + e[ok]
        ax.fill_between(x[ok], np.minimum(np.maximum(m[ok] - e[ok], 1e-12), hi), hi, color=ST.C[DISP[arm]],
                        alpha=.13, lw=0, zorder=z - 1)
        return m, e, n

    for r, (cond, lab) in enumerate((("good", "good prior"), ("adv", "adversarial prior"))):
        for i, fam in enumerate(FAMILIES):
            ax = fig.add_subplot(gs[r, i])
            units = [H.unit(fam, t) for t in H.TASKS]
            for arm in ARMS:
                plot(ax, data[cond][arm], units, arm)
            ax.set(xlim=(1, 16), xticks=[1, 4, 8, 12, 16], xlabel="query / cost")
            ax.set_title(FAMILY_NAMES[fam], fontsize=8, loc="left")
            ST.clean(ax)
            if i == 0:
                ax.set_ylabel(f"normalized regret (↓)\n[{lab}]")
    summary = {}
    units_all = [H.unit(f, t) for f in FAMILIES for t in H.TASKS]
    for cond, sl, title in (("good", slice(0, 2), "Grand mean — good prior"),
                            ("adv", slice(3, 5), "Grand mean — adversarial prior")):
        ax = fig.add_subplot(gs[2, sl])
        for arm in ARMS:
            m, e, n = plot(ax, data[cond][arm], units_all, arm)
            first = next(i + 1 for i in range(H.BUDGET) if n[i])
            summary.setdefault(cond, {})[LEG[arm]] = {"q16": round(float(m[-1]), 4), "sem16": round(float(e[-1]), 4),
                                                      "x": "cost" if arm in MF else "query", "first_x": first,
                                                      "n_tasks": n[-1]}
            if arm in MF:
                summary[cond][LEG[arm]].update(repeat_share(*SRC[cond][arm]))
        ax.set(xlim=(1, 16), xticks=[1, 4, 8, 12, 16], xlabel="query  |  full-fidelity-equivalent cost",
               ylabel="normalized simple regret (↓)")
        ax.set_title(f"{title} ({len(units_all)} tasks)", loc="left")
        ST.clean(ax)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, ncol=5, loc="lower center", bbox_to_anchor=(.54, .005),
               fontsize=matplotlib.rcParams["axes.titlesize"], handlelength=1.8, columnspacing=1.3)
    ST.save(fig, fig_dir / "supp" / "figS_priorband_primo_vs_hapbo_b16")
    return {"figS_priorband_primo_vs_hapbo_b16": summary}
