"""Representation ablation (fig_representation): how the elicited belief becomes pi(x), in the main-evaluation style.

Arms: full-PSL (the PSL document as compiled), point-only, Gaussian-only, direct-LLM (HPOBench only) and vanilla BO.
Every variant keeps the trust block and alpha of its source prior; only the representation of the belief differs.
Inverted-map condition: as in the main evaluation (fig5_robustness), full-PSL reads the inverted-map prior; the
variants are those of the prior elicited from the card plus the misleading hint.

  A  row 1  grand means: synthetic rho ~ 0.9 (4 functions, synthetic_data normalization, log y)
                         | HPOBench b16, good prior (40 tasks, grid-span normalized)
                         | HPOBench b16, inverted-map prior (40 tasks, vanilla-q1 normalized, log y)
  B  row 2  average rank of the same three conditions (synthetic: per function on the function-mean curve, mean over
            the 4 functions; HPOBench: fig_hpo.ranks, per task and query)
  C  row 3  synthetic rho ~ 0.9 per function
  D  row 4  HPOBench good prior per family (grid-span normalized)
  E  row 5  HPOBench inverted-map prior per family (vanilla-q1 normalized, log y)

Studies (results/paper/hpo/): good prior hapbo, representation-{pointonly,gaussian,directllm}; inverted-map prior
hapbo-inverted (full-PSL, as in fig5_robustness) and representation-{pointonly,gaussian,directllm}-misleading (the
representation variants of the prior elicited from the card plus the misleading hint); vanilla BO from vanilla-random
in both. Synthetic (results/paper/synthetic/): rho90 arms hapbo, pointonly, gaussian; vanilla from none.json.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

import fig_hpo as F
import hpo_data as H
import style as ST
import synthetic_data as SY

M = "alpha-pi-raw-ei-matched"
COLOR = {"full-PSL": "#0072B2", "point-only": "#D55E00", "gaussian-only": "#009E73", "direct-LLM": "#CC79A7",
         "vanilla BO": "#4D4D4D"}
HPO_ARMS = {  # label -> ({condition: study}, method); "std" = good prior, "adv" = inverted-map prior
    "full-PSL": ({"std": "hapbo", "adv": "hapbo-inverted"}, M),
    "point-only": ({"std": "representation-pointonly", "adv": "representation-pointonly-misleading"}, M),
    "gaussian-only": ({"std": "representation-gaussian", "adv": "representation-gaussian-misleading"}, M),
    "direct-LLM": ({"std": "representation-directllm", "adv": "representation-directllm-misleading"}, M),
    "vanilla BO": ({"std": "vanilla-random", "adv": "vanilla-random"}, "vanilla-bo-ei"),
}
SYN_ARMS = (("full-PSL", "hapbo"), ("point-only", "pointonly"), ("gaussian-only", "gaussian"), ("vanilla BO", "vanilla"))
SYN_CARD = "rho90"
REF = "vanilla BO"   # vanilla-q1 reference of the inverted-map panels


def curve(ax, x, m, s, name, lw=None):
    ours = name == "full-PSL"
    ax.plot(x, m, color=COLOR[name], lw=lw or (2.4 if ours else 1.55), label=name, zorder=6 if ours else 3)
    if s is not None:
        ax.fill_between(x, np.maximum(m - s, 1e-12), m + s, color=COLOR[name], alpha=.13, lw=0)


def hpo_load(condition):
    """Seed-mean simple-regret curve per (family, task) per arm, run counts, the tasks every arm covers, arm order."""
    data, counts = H.collect(HPO_ARMS, condition)
    keys = sorted(set.intersection(*[set(v) for v in data.values()]))
    return data, counts, keys, [a for a in HPO_ARMS if a in data]


def vq1_rows(data, keys, arm):
    rows = []
    for k in keys:
        ref = data[REF][k][0]
        if ref > 1e-12:
            rows.append(data[arm][k] / ref)
    return rows


def synthetic():
    """Per-function normalized runs, grand-mean curves and average rank over the 4 functions."""
    per_fn = {arm: {fam: SY.run_rows(arm, fam, SYN_CARD) for fam in SY.FAMS} for _, arm in SYN_ARMS}
    cur = SY.curves([arm for _, arm in SYN_ARMS], SYN_CARD)
    rk = np.zeros((len(SYN_ARMS), len(SY.GRID)))
    for fam in SY.FAMS:
        mat = np.round(np.array([per_fn[arm][fam].mean(0) for _, arm in SYN_ARMS]) / 1e-9)
        rk += np.apply_along_axis(rankdata, 0, mat)
    return per_fn, cur, rk / len(SY.FAMS)


def build(fig_dir):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    cond = {}
    for c in ("std", "adv"):
        data, counts, keys, arms = hpo_load(c)
        log = c == "adv"
        cond[c] = (data, keys, arms, vq1_rows if log else F.span_rows, log, counts)
    per_fn, cur, rk_s = synthetic()
    x, q = SY.GRID * 100, np.arange(1, H.BUDGET + 1)
    summary = {"counts": {c: cond[c][5] for c in cond}, "tasks": {c: len(cond[c][1]) for c in cond}}

    ST.set_style(panel_geometry=True)
    fig = plt.figure(figsize=(9.1, 12.4))
    W = dict(left=.068, right=.99)
    top = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], hspace=.5, wspace=.42, top=.965, bottom=.665, **W)
    r3 = GridSpec(1, 4, figure=fig, wspace=.42, top=.575, bottom=.455, **W)
    r4 = GridSpec(1, 5, figure=fig, wspace=.42, top=.385, bottom=.265, **W)
    r5 = GridSpec(1, 5, figure=fig, wspace=.42, top=.195, bottom=.075, **W)

    def tag(ax, letter):
        ax.text(-0.32, 1.16, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="left")

    # row 1 (A): grand means
    ax = fig.add_subplot(top[0, 0])
    tag(ax, "A")
    syn = {}
    for lab, arm in SYN_ARMS:
        U, seeds = cur[arm]
        m = U.mean(0)
        curve(ax, x, m, U.std(0, ddof=1) / np.sqrt(len(U)), lab)
        syn[lab] = (round(float(m[-1]), 4), seeds)
    ax.set_yscale("log")
    ax.set(xlim=(0, 100), xlabel="fraction of budget (%)", ylabel="normalized simple regret (↓)")
    ax.set_title("Synthetic ρ≈0.9", loc="left")
    F.clean(ax)
    summary["synthetic"] = syn
    titles = {"std": "HPOBench, good prior", "adv": "HPOBench, adversarial prior"}
    for j, c in enumerate(("std", "adv"), 1):
        data, keys, arms, rows, log, _ = cond[c]
        ax = fig.add_subplot(top[0, j])
        grand = {}
        for a in arms:
            m, s = H.mean_sem(rows(data, keys, a))
            curve(ax, q, m, s, a)
            grand[a] = round(float(m[-1]), 4)
        if log:
            ax.set_yscale("log")
        ax.set(xlim=(1, H.BUDGET), xticks=[1, 4, 8, 12, 16], xlabel="query")
        ax.set_title(f"{titles[c]} ({len(keys)} tasks)", loc="left")
        F.clean(ax)
        summary[f"{c}_grand_q16"] = grand

    # row 2 (B): average ranks
    ax = fig.add_subplot(top[1, 0])
    tag(ax, "B")
    for i, (lab, _) in enumerate(SYN_ARMS):
        curve(ax, x, rk_s[i], None, lab)
    ax.set(xlim=(0, 100), xlabel="fraction of budget (%)", ylabel="average rank (1 = best)", ylim=(4.5, .5),
           yticks=range(1, 5))
    F.clean(ax)
    summary["synthetic_rank"] = {lab: round(float(rk_s[i][-1]), 2) for i, (lab, _) in enumerate(SYN_ARMS)}
    for j, c in enumerate(("std", "adv"), 1):
        data, keys, arms, *_ = cond[c]
        ax = fig.add_subplot(top[1, j])
        rk = F.ranks(data, keys, arms)
        for a in arms:
            curve(ax, q, rk[a], None, a)
        ax.set(xlim=(1, H.BUDGET), xticks=[1, 4, 8, 12, 16], xlabel="query", ylim=(len(arms) + .5, .5),
               yticks=range(1, len(arms) + 1))
        F.clean(ax)
        summary[f"{c}_rank_q16"] = {a: round(float(rk[a][-1]), 2) for a in arms}

    # row 3 (C): synthetic per function
    fig.text(.5, .595, "Synthetic ρ≈0.9, per function", ha="center", fontweight="bold", fontsize=9)
    for i, fam in enumerate(SY.FAMS):
        ax = fig.add_subplot(r3[0, i])
        if i == 0:
            tag(ax, "C")
        for lab, arm in SYN_ARMS:
            r = per_fn[arm][fam]
            curve(ax, x, r.mean(0), r.std(0, ddof=1) / np.sqrt(len(r)), lab, lw=2.0 if lab == "full-PSL" else 1.25)
        ax.set_yscale("log")
        ax.set(xlim=(0, 100), xlabel="fraction of budget (%)")
        ax.set_title(SY.FAMS[fam][0], fontsize=8, loc="left")
        F.clean(ax)
        if i == 0:
            ax.set_ylabel("normalized regret (↓)")

    # rows 4, 5 (D, E): HPOBench per family
    for gsr, c, ttl, ty, letter in ((r4, "std", "HPOBench, good prior, per family", .405, "D"),
                                    (r5, "adv", "HPOBench, adversarial prior, per family", .215, "E")):
        data, keys, arms, rows, log, _ = cond[c]
        fig.text(.5, ty, ttl, ha="center", fontweight="bold", fontsize=9)
        for i, fam in enumerate(H.FAMS):
            ax = fig.add_subplot(gsr[0, i])
            if i == 0:
                tag(ax, letter)
            fk = [k for k in keys if k[0] == fam]
            for a in arms:
                m, s = H.mean_sem(rows(data, fk, a))
                curve(ax, q, m, s, a, lw=2.0 if a == "full-PSL" else 1.25)
            if log:
                ax.set_yscale("log")
            ax.set(xlim=(1, H.BUDGET), xticks=[1, 8, 16], xlabel="query")
            ax.set_title(F.FAM_NAME[fam], fontsize=8, loc="left")
            F.clean(ax)
            if i == 0:
                ax.set_ylabel("normalized regret (↓)")
    handles = [plt.Line2D([], [], color=COLOR[a], lw=2.2) for a in COLOR]
    fig.legend(handles, list(COLOR), ncol=5, loc="lower center", bbox_to_anchor=(.54, .0), handlelength=1.5,
               columnspacing=1.1, handletextpad=0.5, prop={"size": plt.rcParams["axes.titlesize"], "weight": "normal"})
    # fixed saved extent (9.454 x 12.384 in), the page size of the figure in the paper
    from matplotlib.transforms import Bbox
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"fig_representation.{ext}",
                    bbox_inches=Bbox.from_extents(-0.11038, 0.01653, 9.34363, 12.40074))
    plt.close(fig)
    return {"fig_representation": summary}
