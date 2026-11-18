"""Shared palette, rcParams and plotting helpers of the paper figures.

Two rcParams sets are used:
  set_style()          compact two-column figures (fig2_*, fig3_realworld, fig4_synthetic_fidelity, fig5_*,
                       fig_prior_sources_main, fig_baseline_impl_sensitivity,
                       supp/figS2_prior_sources_and_trajectories, supp/figS_cake_vs_hapbo_b16)
  set_style_panels(s)  multi-panel ablation / acquisition figures (fig_ablations, fig_bounded_shaping and its
                       panels), every font size multiplied by `s`
set_style() also draws supp/figS_priorband_primo_vs_hapbo_b16, supp/figS_hypbo_llmei_vs_hapbo_b16 and the
fixed-alpha sweep (fig_alpha_sweep_hapbo).

Colour maps:
  C            method / prior-source colours (Okabe-Ito, colour-blind safe; HypBO and LLM-EI: indigo and violet,
               checked against every other method colour under simulated colour-vision deficiencies)
  ROLE         semantic roles of the acquisition figures (ours / vanilla)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np

C = {
    "HapBO": "#0072B2", "ColaBO": "#009E73", "LLAMBO": "#E69F00",
    "BORA": "#CC79A7", "LGBO": "#882255", "vanilla": "#4D4D4D",
    "random": "#B3B3B3", "TuRBO": "#56B4E9", "piBO": "#D55E00",
    "human": "#0072B2", "mixed": "#009E73", "llm": "#E69F00",
    "CAKE": "#D55E00", "CALIPER": "#009E73", "PriorBand": "#E69F00", "PriMO": "#CC79A7",
    "HypBO": "#332288", "LLM-EI": "#9955FF",
}

ROLE = {"ours": "#0072B2", "vanilla": "#4D4D4D"}
OURS_SHORT = r"$\alpha$-$\pi$"


def set_style(panel_geometry: bool = False):
    """Compact rcParams; panel_geometry=True keeps the line / tick geometry of set_style_panels underneath
    (used by the family-panel figures fig2_hpo_b16, fig5_robustness, fig_baseline_impl_sensitivity and
    supp/figS_cake_vs_hapbo_b16). supp/figS_priorband_primo_vs_hapbo_b16, supp/figS_hypbo_llmei_vs_hapbo_b16
    and fig_representation use panel_geometry=True as well."""
    mpl.rcdefaults()
    if panel_geometry:
        _set_panel_rc()
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03, "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.3, "axes.titlesize": 9.2, "axes.labelsize": 8.5,
        "axes.titleweight": "bold", "xtick.labelsize": 7.3, "ytick.labelsize": 7.3,
        "legend.fontsize": 6.7, "legend.frameon": False, "axes.linewidth": 0.75,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": "#E7E7E7", "grid.linewidth": 0.55,
    })


def _set_panel_rc():
    """rcParams of the multi-panel figures (set_style_panels; geometry of set_style(panel_geometry=True))."""
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "axes.titleweight": "bold",
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.6, "legend.frameon": False,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "grid.color": "#E6E6E6", "grid.linewidth": 0.6, "lines.linewidth": 1.8, "lines.solid_capstyle": "round",
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def set_style_panels(scale: float = 1.0):
    """rcParams of the multi-panel ablation / acquisition figures, every font size multiplied by `scale`."""
    mpl.rcdefaults()
    _set_panel_rc()
    for k in ("font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize", "ytick.labelsize", "legend.fontsize"):
        mpl.rcParams[k] = float(mpl.rcParams[k]) * scale


def curve(ax, x, mean, sem, name, label=None, *, lw=None, ls="-", z=3):
    """Learning curve in colour C[name] with a +- SEM band (floored at 1e-12 for log axes)."""
    color = C[name]
    ax.plot(x, mean, color=color, lw=lw or (2.4 if name == "HapBO" else 1.55), ls=ls, label=label or name, zorder=z)
    if sem is not None:
        ax.fill_between(x, np.maximum(np.asarray(mean) - np.asarray(sem), 1e-12), np.asarray(mean) + np.asarray(sem),
                        color=color, alpha=.13, lw=0, zorder=z - 1)


def curve_role(ax, x, mean, sem=None, *, color, label, lw=1.8, ls="-", z=3, marker=None):
    """Learning curve in an explicit colour with an optional +- SEM band (ablation figures)."""
    ax.plot(x, mean, color=color, label=label, lw=lw, ls=ls, zorder=z, marker=marker,
            markersize=3.2, markeredgewidth=0)
    if sem is not None:
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.15, lw=0, zorder=z - 1)


def clean(ax, grid="y"):
    ax.grid(axis=grid, zorder=0)
    ax.set_axisbelow(True)


def despine(ax, left=True, bottom=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)
    if not bottom:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(bottom=False)


def value_grid(ax, axis="y"):
    ax.grid(axis=axis, color="#E6E6E6", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def mean_sem(arr):
    arr = np.asarray(arr, dtype=float)
    return arr.mean(0), arr.std(0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros(arr.shape[1])


def save(fig, path_noext: Path):
    """Write <path>.pdf and <path>.png and close the figure."""
    import matplotlib.pyplot as plt
    path_noext = Path(path_noext)
    path_noext.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_noext}.{ext}")
    plt.close(fig)
