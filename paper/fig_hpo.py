"""HPOBench figures (40 tasks = 5 families x 8, budget 16, 10 seeds; seeds averaged inside each task, the task is the unit).

  fig2_hpo_compact   standard prior: average task-wise rank | grand mean (grid-span normalized)
  fig2_hpo_b16       standard prior: 5 family panels + rank + grand mean (grid-span normalized)
  fig5_adv_compact   adversarial prior: average rank | grand mean (vanilla-q1 normalized, log y)
  fig5_robustness    adversarial prior: 5 family panels + rank + grand mean (vanilla-q1 normalized, log y)

Adversarial condition: HapBO uses the inverted-map prior (hpo/hapbo-inverted); the baselines read the card plus the
misleading hint. Ranks: per task and query, rank methods on the seed-mean regret (1 = best, ties within 1e-9
averaged), mean over tasks.
"""
from __future__ import annotations

import numpy as np

import hpo_data as H
import style as ST

QCAP = H.BUDGET
ORDER = ["HapBO", "ColaBO", "LLAMBO", "LGBO", "BORA", "Vanilla BO", "Random"]
COLOR = {"HapBO": "#0072B2", "ColaBO": "#009E73", "LLAMBO": "#E69F00", "BORA": "#CC79A7",
         "LGBO": "#882255", "Vanilla BO": "#4D4D4D", "Random": "#B3B3B3"}
LEGEND = {"HapBO": "HapBO", "ColaBO": "ColaBO", "LLAMBO": "LLAMBO (mode 2)", "LGBO": "LGBO",
          "BORA": "BORA", "Vanilla BO": "vanilla BO", "Random": "random search"}
FAM_NAME = {"lr": "linear models", "rf": "random forest", "xgb": "XGBoost", "nn": "neural net", "svm": "SVM"}
ARMS = {  # label -> ({condition: study}, method)
    "HapBO": ({"std": "hapbo", "adv": "hapbo-inverted"}, "alpha-pi-raw-ei-matched"),
    "LGBO": ({"std": "lgbo", "adv": "lgbo-misleading"}, "lgbo"),
    "ColaBO": ({"std": "colabo", "adv": "colabo-misleading"}, "colabo-llm"),
    "Vanilla BO": ({"std": "vanilla-random", "adv": "vanilla-random"}, "vanilla-bo-ei"),
    "BORA": ({"std": "bora", "adv": "bora-misleading"}, "bora"),
    "LLAMBO": ({"std": "llambo", "adv": "llambo-misleading"}, "llambo-ws"),
    "Random": ({"std": "vanilla-random", "adv": "vanilla-random"}, "random-search"),
}

_CACHE: dict = {}


def load(condition: str):
    """Seed-mean regret curve per (family, task) per arm, run counts, and the tasks every arm covers."""
    if condition not in _CACHE:
        data, counts = H.collect(ARMS, condition)
        keys = sorted(set.intersection(*[set(v) for v in data.values()]))
        _CACHE[condition] = (data, counts, keys)
    return _CACHE[condition]


def curve(ax, x, m, s, name, lw=None, label=True):
    c = COLOR[name]
    ax.plot(x, m, color=c, lw=lw or (2.4 if name == "HapBO" else 1.55),
            label=LEGEND[name] if label else None, zorder=6 if name == "HapBO" else 3)
    if s is not None:
        ax.fill_between(x, np.maximum(m - s, 1e-12), m + s, color=c, alpha=.13, lw=0,
                        zorder=5 if name == "HapBO" else 2)


def clean(ax):
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


def ranks(data, keys, arms):
    """Average task-wise rank per query (seed-mean regret; invariant to per-task normalization)."""
    from scipy.stats import rankdata
    out = {a: np.zeros(QCAP) for a in arms}
    for k in keys:
        mat = np.array([data[a][k] for a in arms])
        mat = np.round(mat / 1e-9)
        r = np.apply_along_axis(rankdata, 0, mat)
        for i, a in enumerate(arms):
            out[a] += r[i]
    return {a: v / len(keys) for a, v in out.items()}


def vq1_rows(data, keys, arm):
    rows = []
    for k in keys:
        ref = data["Vanilla BO"][k][0]
        if ref > 1e-12:
            rows.append(data[arm][k] / ref)
    return rows


def span_rows(data, keys, arm):
    return [np.maximum(data[arm][k] / H.grid_span(*k), 0) for k in keys]


def rank_panel(ax, data, keys, arms, title):
    q = np.arange(1, QCAP + 1)
    rk = ranks(data, keys, arms)
    for a in arms:
        curve(ax, q, rk[a], None, a)
    ax.set(xlim=(1, QCAP), xticks=[1, 4, 8, 12, 16], xlabel="query", ylabel="average rank (1 = best)",
           ylim=(len(arms) + .5, .5), yticks=np.arange(1, len(arms) + 1))
    ax.set_title(title, loc="left")
    clean(ax)
    return rk


def grand_vq1_panel(ax, data, keys, arms, title, label=True):
    q = np.arange(1, QCAP + 1)
    res = {}
    for a in arms:
        m, s = H.mean_sem(vq1_rows(data, keys, a))
        curve(ax, q, m, s, a, label=label)
        res[a] = (m, s)
    ax.set_yscale("log")
    ax.set(xlim=(1, QCAP), xticks=[1, 4, 8, 12, 16], xlabel="query", ylabel="normalized simple regret (↓)")
    ax.set_title(title, loc="left")
    clean(ax)
    return res


def legend_below(fig, ax, ncol, anchor_y=0.0, rename=None):
    h, l = ax.get_legend_handles_labels()
    l = [(rename or {}).get(t, t) for t in l]
    fig.legend(h, l, ncol=ncol, loc="lower center", bbox_to_anchor=(.5, anchor_y), fontsize=7,
               handlelength=1.6, columnspacing=1.1)


def _r(d, k=4):
    return {a: round(float((v[0] if isinstance(v, tuple) else v)[-1]), k) for a, v in d.items()}


def fig2_compact(out):
    import matplotlib.pyplot as plt
    ST.set_style()
    data, counts, keys = load("std")
    arms = [a for a in ORDER if a in data]
    fig, (rax, gax) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    rk = rank_panel(rax, data, keys, arms, f"Average rank ({len(keys)} tasks)")
    q = np.arange(1, QCAP + 1)
    grand = {}
    for a in arms:
        m, s = H.mean_sem(span_rows(data, keys, a))
        curve(gax, q, m, s, a, label=False)
        grand[a] = (m, s)
    gax.set(xlim=(1, QCAP), xticks=[1, 4, 8, 12, 16], xlabel="query", ylabel="normalized simple regret (↓)")
    gax.set_ylim(bottom=0)
    gax.set_title(f"Grand mean ({len(keys)} tasks)", loc="left")
    clean(gax)
    legend_below(fig, rax, ncol=len(arms), rename={"LLAMBO (mode 2)": "LLAMBO"})
    fig.tight_layout(rect=[0, .09, 1, 1])
    ST.save(fig, out / "fig2_hpo_compact")
    return {"n_tasks": len(keys), "runs": counts, "rank_q16": _r(rk, 2), "grand_q16": _r(grand)}


def fig5_compact(out):
    import matplotlib.pyplot as plt
    ST.set_style()
    data, counts, keys = load("adv")
    arms = [a for a in ORDER if a in data]
    fig, (rax, gax) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    rk = rank_panel(rax, data, keys, arms, f"Average rank ({len(keys)} tasks)")
    grand = grand_vq1_panel(gax, data, keys, arms, f"Grand mean ({len(keys)} tasks)", label=False)
    legend_below(fig, rax, ncol=len(arms), rename={"LLAMBO (mode 2)": "LLAMBO"})
    fig.tight_layout(rect=[0, .09, 1, 1])
    ST.save(fig, out / "fig5_adv_compact")
    return {"n_tasks": len(keys), "runs": counts, "rank_q16": _r(rk, 2), "grand_q16": _r(grand)}


def _family_grid(fig):
    from matplotlib.gridspec import GridSpec
    return GridSpec(2, 5, figure=fig, height_ratios=[1.0, 1.35], hspace=.52, wspace=.40,
                    left=.068, right=.99, top=.94, bottom=.23)


def fig2_b16(out):
    import matplotlib.pyplot as plt
    ST.set_style(panel_geometry=True)
    data, counts, keys = load("std")
    arms = [a for a in ORDER if a in data]
    q = np.arange(1, QCAP + 1)
    fig = plt.figure(figsize=(9.1, 5.3))
    gs = _family_grid(fig)
    fam_q16 = {}
    for i, fam in enumerate(H.FAMS):
        ax = fig.add_subplot(gs[0, i])
        fk = [k for k in keys if k[0] == fam]
        for a in arms:
            m, s = H.mean_sem(span_rows(data, fk, a))
            curve(ax, q, m, s, a, lw=2.0 if a == "HapBO" else 1.25, label=False)
            fam_q16.setdefault(fam, {})[a] = round(float(m[-1]), 4)
        ax.set(xlim=(1, QCAP), xticks=[1, 4, 8, 12, 16], xlabel="query")
        ax.set_title(FAM_NAME[fam], fontsize=8, loc="left")
        clean(ax)
        if i == 0:
            ax.set_ylabel("normalized regret (↓)")
    rax = fig.add_subplot(gs[1, :2])
    rk = rank_panel(rax, data, keys, arms, f"Average rank across {len(keys)} tasks")
    gax = fig.add_subplot(gs[1, 3:])
    grand = {}
    for a in arms:
        m, s = H.mean_sem(span_rows(data, keys, a))
        curve(gax, q, m, s, a, label=False)
        grand[a] = round(float(m[-1]), 4)
    gax.set(xlim=(1, QCAP), xticks=[1, 4, 8, 12, 16], xlabel="query", ylabel="normalized simple regret (↓)")
    gax.set_title(f"Grand mean across {len(keys)} tasks", loc="left")
    clean(gax)
    h, l = rax.get_legend_handles_labels()
    l = ["LLAMBO" if t == "LLAMBO (mode 2)" else t for t in l]
    fig.legend(h, l, ncol=4, loc="lower center", bbox_to_anchor=(.54, .085), handlelength=1.8, columnspacing=1.25)
    ST.save(fig, out / "fig2_hpo_b16")
    return {"n_tasks": len(keys), "runs": counts, "grand_q16": grand, "rank_q16": _r(rk, 2), "family_q16": fam_q16}


def fig5_robustness(out):
    import matplotlib.pyplot as plt
    ST.set_style(panel_geometry=True)
    data, counts, keys = load("adv")
    arms = [a for a in ORDER if a in data]
    q = np.arange(1, QCAP + 1)
    fig = plt.figure(figsize=(9.1, 5.3))
    gs = _family_grid(fig)
    fams = {}
    for i, fam in enumerate(H.FAMS):
        ax = fig.add_subplot(gs[0, i])
        fk = [k for k in keys if k[0] == fam]
        for a in arms:
            m, s = H.mean_sem(vq1_rows(data, fk, a))
            curve(ax, q, m, s, a, lw=2.0 if a == "HapBO" else 1.25, label=False)
            fams.setdefault(fam, {})[a] = round(float(m[-1]), 4)
        ax.set_yscale("log")
        ax.set(xlim=(1, QCAP), xticks=[1, 8, 16], xlabel="query")
        ax.set_title(FAM_NAME[fam], fontsize=8, loc="left")
        clean(ax)
        if i == 0:
            ax.set_ylabel("normalized regret (↓)")
    rax = fig.add_subplot(gs[1, :2])
    rk = rank_panel(rax, data, keys, arms, f"Average rank across {len(keys)} tasks")
    gax = fig.add_subplot(gs[1, 3:])
    grand = grand_vq1_panel(gax, data, keys, arms, f"Grand mean across {len(keys)} HPOBench tasks", label=False)
    h, l = rax.get_legend_handles_labels()
    l = ["LLAMBO" if t == "LLAMBO (mode 2)" else t for t in l]
    fig.legend(h, l, ncol=4, loc="lower center", bbox_to_anchor=(.54, .085), handlelength=1.8, columnspacing=1.25)
    ST.save(fig, out / "fig5_robustness")
    return {"n_tasks": len(keys), "runs": counts, "grand_q16": _r(grand), "rank_q16": _r(rk, 2), "family_q16": fams}


def build(fig_dir):
    return {"fig2_hpo_compact": fig2_compact(fig_dir), "fig5_adv_compact": fig5_compact(fig_dir),
            "fig2_hpo_b16": fig2_b16(fig_dir), "fig5_robustness": fig5_robustness(fig_dir)}
