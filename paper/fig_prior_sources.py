"""Cortical prior sources (exploration performance over 12 queries, HapBO matched EI vs vanilla BO EI).

  fig_prior_sources_main                     A LLM only (llm-geometry), B LLM + human (llm-expert-hint), C human only
  supp/figS2_prior_sources_and_trajectories  A-C compiled prior fields pi(x), D-F example truth maps, G-I curves
"""
from __future__ import annotations

import numpy as np

import cortical_data as CD
import style as ST

CONDS = [  # key, label, HapBO study, colour key
    ("geo", "LLM only", "llm-geometry", "llm"),
    ("mix", "LLM + human", "llm-expert-hint", "mixed"),
    ("human", "Human only", "human-map", "human"),
]
VANILLA = ("human-map", "vanilla-bo-ei")
HAPBO = "alpha-pi-raw-ei-matched"


def fig_main(fig_dir):
    import matplotlib.pyplot as plt
    ST.set_style()
    Q = CD.HORIZON
    x = np.arange(1, Q + 1)
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.45), sharey=True)
    nums = {}
    vmu, vse, *_ = CD.explore_mean_sem(*VANILLA)
    for i, (ax, (_, label, study, ckey)) in enumerate(zip(axes, CONDS)):
        hmu, hse, _, hmaps = CD.explore_mean_sem(study, HAPBO)
        ST.curve(ax, x, hmu, hse, ckey, label, lw=2.15, ls="-")
        ST.curve(ax, x, vmu, vse, "vanilla", "vanilla BO", lw=1.45, ls="--")
        ax.set(xlim=(1, Q), xticks=[1, 4, 8, 12], ylim=(0.35, 1.0), xlabel="query")
        ax.set_title(f"{chr(65 + i)}  {label} vs vanilla", loc="left", fontsize=8.5)
        ST.clean(ax)
        ax.legend(loc="lower right", fontsize=6.5)
        nums[label] = {"hapbo_q8": round(float(hmu[7]), 4), "hapbo_q12": round(float(hmu[-1]), 4), "maps": hmaps}
    axes[0].set_ylabel("exploration performance (↑)")
    fig.tight_layout()
    ST.save(fig, fig_dir / "fig_prior_sources_main")
    nums["vanilla"] = {"q8": round(float(vmu[7]), 4), "q12": round(float(vmu[-1]), 4)}
    return nums


def fig_s2(fig_dir):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    ST.set_style()
    Q = CD.HORIZON
    src = CD.prior_sources()
    fig = plt.figure(figsize=(7.25, 7.35))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.0, 1.0, 1.35], hspace=.53, wspace=.34,
                  left=.09, right=.93, bottom=.07, top=.955)
    prior_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    truth_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    curve_axes = [fig.add_subplot(gs[2, i]) for i in range(3)]
    for i, (ax, (c, label, *_)) in enumerate(zip(prior_axes, CONDS)):
        f = np.asarray(src["priors"][c]["field"], float)
        im_p = ax.imshow(f, origin="upper", cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"{chr(65 + i)}  {label}", loc="left", fontsize=8.5)
        ax.set_xticks(range(4), range(1, 5)); ax.set_yticks(range(8), range(1, 9))
        ax.set_xlabel("medio-lateral index", fontsize=7.2)
    prior_axes[0].set_ylabel("elicited prior\nrostro-caudal index", fontsize=7.5)
    cb = fig.colorbar(im_p, ax=prior_axes, fraction=.025, pad=.018)
    cb.set_label("prior plausibility $\\pi(x)$", fontsize=7.3); cb.ax.tick_params(labelsize=6.5)
    for i, (ax, ex) in enumerate(zip(truth_axes, src["examples"])):
        im_t = ax.imshow(np.asarray(ex["field"], float), origin="upper", cmap="magma", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"{chr(68 + i)}  Example {i + 1}", loc="left", fontsize=8.5)
        ax.set_xticks(range(4), range(1, 5)); ax.set_yticks(range(8), range(1, 9))
        ax.set_xlabel("medio-lateral index", fontsize=7.2)
    truth_axes[0].set_ylabel("ground truth\nrostro-caudal index", fontsize=7.5)
    cb = fig.colorbar(im_t, ax=truth_axes, fraction=.025, pad=.018)
    cb.set_label("within-map normalized response", fontsize=7.3); cb.ax.tick_params(labelsize=6.5)
    x = np.arange(1, Q + 1)
    vmu, vse, *_ = CD.explore_mean_sem(*VANILLA)
    nums = {}
    for i, (ax, (c, label, study, ckey)) in enumerate(zip(curve_axes, CONDS)):
        hmu, hse, *_ = CD.explore_mean_sem(study, HAPBO)
        ST.curve(ax, x, hmu, hse, ckey, label, lw=2.15, ls="-")
        ST.curve(ax, x, vmu, vse, "vanilla", "vanilla BO", lw=1.45, ls="--")
        ax.set(xlim=(1, Q), xticks=[1, 4, 8, 12], ylim=(0, 1.03), xlabel="query")
        ax.set_title(f"{chr(71 + i)}  {label} vs vanilla", loc="left", fontsize=8.5)
        ST.clean(ax); ax.legend(loc="lower right", fontsize=6.3)
        nums[label] = round(float(hmu[-1]), 4)
    curve_axes[0].set_ylabel("exploration performance (↑)")
    for ax in curve_axes[1:]:
        ax.tick_params(labelleft=False)
    ST.save(fig, fig_dir / "supp" / "figS2_prior_sources_and_trajectories")
    nums["vanilla"] = round(float(vmu[-1]), 4)
    return nums


def build(fig_dir):
    return {"fig_prior_sources_main": fig_main(fig_dir), "figS2_prior_sources_and_trajectories": fig_s2(fig_dir)}
