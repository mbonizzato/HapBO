"""fig_bounded_shaping: bounded shaping across BO variants (HapBO matched vs vanilla BO, 10 seeds).

  fig_acq_across   EI / PI / UCB; rows HPOBench (40 tasks, 1 - best/opt, budget 16) and
                   cortical (40 rat-muscle maps as technical replicates, prior from the geometry facts plus the
                   expert hint, best/opt, 12 queries)
  fig_batch_bo     EI, sequential (q = 1) vs Kriging-Believer batch (q = 3)
  fig_kernel_bo    EI with the Matérn-5/2 (acquisition study), RBF and Matérn-1/2 GP kernels; y axes shared per row
  fig_bounded_shaping  the panels of the three figures above in one figure at text width: A-C acquisition,
                   D-E batch, F-H kernels; y axes shared per row
"""
from __future__ import annotations

import numpy as np

import cortical_data as CD
import hpo_data as H
import style as ST

FAMS = H.FAMS  # all 40 HPOBench tasks
ACQ = [("A", "EI", "alpha-pi-raw-ei-matched", "vanilla-bo-ei"),
       ("B", "PI", "alpha-pi-raw-pi-matched", "vanilla-bo-pi"),
       ("C", "UCB", "alpha-pi-raw-ucb-matched", "vanilla-bo-ucb")]
ROWS = [("hpo", "acquisition", "batch", "normalized\nsimple regret\n(lower better)"),
        ("cortical", "acquisition", "batch", "normalized\nperformance\n(higher better)")]
KERNELS = [("F", "Matérn-5/2", "matern52", "acquisition"),  # the default kernel: the acquisition study's EI arms
           ("G", "RBF", "rbf", "kernel-rbf"),
           ("H", "Matérn-1/2", "matern12", "kernel-matern12")]


def curve(kind, study, method):
    """(mean, SEM, n, queries) of one arm: HPOBench over the budget, cortical over CD.HORIZON queries."""
    rows = H.hpo_curve(study, method, FAMS) if kind == "hpo" else CD.cort_curve(study, method, unit="map")
    m, e = H.mean_sem(rows)
    return m, e, len(rows), np.arange(1, len(m) + 1)


def _r(m, n):
    return [round(float(m[2]), 4), round(float(m[-1]), 4), n]


def build(fig_dir):
    import matplotlib.pyplot as plt
    ST.set_style_panels(2.0)
    ours_c, van_c = ST.ROLE["ours"], ST.ROLE["vanilla"]
    acq_num = {}
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.2), sharex="row")
    for r, (kind, sweep, _, ylab) in enumerate(ROWS):
        for c, (letter, acq, api, van) in enumerate(ACQ):
            ax = axes[r, c]
            for arm, color, lab in ((api, ours_c, ST.OURS_SHORT), (van, van_c, "vanilla")):
                m, e, n, q = curve(kind, sweep, arm)
                ST.curve_role(ax, q, m, e, color=color, label=f"{lab} (n={n})",
                              lw=2.2 if color == ours_c else 1.7, ls="-" if color == ours_c else "--")
                acq_num[f"{kind} {acq} {lab}"] = _r(m, n)
            ST.despine(ax); ST.value_grid(ax, "y")
            if r == 0:
                ax.set_title(f"{letter}   {acq}", loc="left", fontweight="bold", fontsize=24)
            ax.set_xlabel("query")
            if c == 0:
                ax.set_ylabel(ylab, fontsize=18)
            ax.legend(fontsize=15, framealpha=0.9, loc="best")
    fig.tight_layout()
    ST.save(fig, fig_dir / "fig_acq_across")
    bat_num = {}
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), sharex="row")
    cols = [("D", "no batch\n(sequential, $q{=}1$)", 1), ("E", "batch\n($q{=}3$, Kriging Believer)", 2)]
    for r, row in enumerate(ROWS):
        kind, ylab = row[0], row[3]
        for c, (letter, col_lab, si) in enumerate(cols):
            ax = axes[r, c]
            for arm, color, lab, lw, ls in (("alpha-pi-raw-ei-matched", ours_c, ST.OURS_SHORT, 2.3, "-"),
                                            ("vanilla-bo-ei", van_c, "vanilla", 1.7, "--")):
                m, e, n, q = curve(kind, row[si], arm)
                ST.curve_role(ax, q, m, e, color=color, label=f"{lab} (n={n})", lw=lw, ls=ls)
                bat_num[f"{kind} {letter} {lab}"] = _r(m, n)
            ST.despine(ax); ST.value_grid(ax, "y")
            if r == 0:
                ax.set_title(f"{letter}   {col_lab}", loc="left", fontweight="bold", fontsize=22)
            ax.set_xlabel("query")
            if c == 0:
                ax.set_ylabel(ylab, fontsize=18)
            ax.legend(fontsize=15, framealpha=0.9, loc="best")
    fig.tight_layout()
    ST.save(fig, fig_dir / "fig_batch_bo")
    # same values as the three separate figures (not repeated in numbers.json); drawn before fig_kernel_bo, whose
    # style settings the next builders inherit
    bounded_shaping(fig_dir)
    return {"fig_acq_across": acq_num, "fig_batch_bo": bat_num, "fig_kernel_bo": kernel_bo(fig_dir)}


def kernel_bo(fig_dir):
    """fig_kernel_bo: HapBO matched vs vanilla BO (EI) under three GP kernels, one y axis per row."""
    import matplotlib.pyplot as plt
    ST.set_style_panels(2.0)
    ours_c, van_c = ST.ROLE["ours"], ST.ROLE["vanilla"]
    num = {}
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.2), sharex="row", sharey="row")
    for r, (kind, _, _, ylab) in enumerate(ROWS):
        for c, (letter, label, kernel, study) in enumerate(KERNELS):
            ax = axes[r, c]
            for arm, color, lab in (("alpha-pi-raw-ei-matched", ours_c, ST.OURS_SHORT),
                                    ("vanilla-bo-ei", van_c, "vanilla")):
                m, e, n, q = curve(kind, study, arm)
                ST.curve_role(ax, q, m, e, color=color, label=f"{lab} (n={n})",
                              lw=2.2 if color == ours_c else 1.7, ls="-" if color == ours_c else "--")
                num[f"{kind} {kernel} {lab}"] = _r(m, n)
            ST.despine(ax); ST.value_grid(ax, "y")
            if r == 0:
                ax.set_title(f"{letter}   {label}", loc="left", fontweight="bold", fontsize=24)
            ax.set_xlabel("query")
            if c == 0:
                ax.set_ylabel(ylab, fontsize=18)
            ax.legend(fontsize=15, framealpha=0.9, loc="best")
    fig.tight_layout()
    ST.save(fig, fig_dir / "fig_kernel_bo")
    return num


def bounded_shaping(fig_dir):
    """fig_bounded_shaping: panels A-C (acquisition) on line 1 and D-H (batch + kernel) on line 2, every panel the
    same size, drawn at the paper's text width (5.5 in) so the font sizes below are the printed ones.
    Same data and arms as fig_acq_across / fig_batch_bo / fig_kernel_bo."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    ST.set_style_panels(1.0)
    ours_c, van_c = ST.ROLE["ours"], ST.ROLE["vanilla"]
    FS_TICK, FS_LAB, FS_TITLE, FS_LEG = 6.5, 7.0, 6.5, 7.0
    YLAB = {"hpo": "normalized\nregret (↓)", "cortical": "normalized\nperformance (↑)"}
    # (letter, title, {kind: study}) per panel; the EI arms of A, D and F are the same runs
    line1 = [(L, acq, {"hpo": "acquisition", "cortical": "acquisition"}, api, van) for L, acq, api, van in ACQ]
    ei = ("alpha-pi-raw-ei-matched", "vanilla-bo-ei")
    line2 = [("D", "no batch\n(sequential, $q{=}1$)", {"hpo": "acquisition", "cortical": "acquisition"}, *ei),
             ("E", "batch, $q{=}3$\n(Kriging Believer)", {"hpo": "batch", "cortical": "batch"}, *ei)]
    line2 += [(L, lab, {"hpo": st, "cortical": st}, *ei) for L, lab, _, st in KERNELS[:3]]
    W = 5.5
    left, right, gap, sep = 0.50, 0.03, 0.09, 0.10
    pw = (W - left - right - 4 * gap - sep) / 5
    ph, t_top, x_gap, x_bot, line_gap = 0.88, 0.30, 0.20, 0.30, 0.02
    line_h = t_top + ph + x_gap + ph + x_bot
    H_ = 2 * line_h + line_gap
    fig = plt.figure(figsize=(W, H_))
    lims = {"hpo": [np.inf, -np.inf], "cortical": [np.inf, -np.inf]}
    axes = {"hpo": [], "cortical": []}
    num = {}
    for li, panels in enumerate((line1, line2)):
        y0 = H_ - li * (line_h + line_gap) - t_top          # top of the first subrow (inches from bottom)
        for c, (letter, title, studies, api, van) in enumerate(panels):
            x0 = left + c * (pw + gap) + (sep if li == 1 and c >= 2 else 0.0)   # extra space between E and F
            for r, kind in enumerate(("hpo", "cortical")):
                yb = y0 - ph - r * (ph + x_gap)
                ax = fig.add_axes([x0 / W, yb / H_, pw / W, ph / H_])
                for arm, color, lw, ls in ((api, ours_c, 1.3, "-"), (van, van_c, 1.0, "--")):
                    m, e, n, q = curve(kind, studies[kind], arm)
                    ST.curve_role(ax, q, m, e, color=color, label=None, lw=lw, ls=ls)
                    lims[kind][0] = min(lims[kind][0], float((m - e).min()))
                    lims[kind][1] = max(lims[kind][1], float((m + e).max()))
                    num[f"{kind} {letter} {'HapBO' if color == ours_c else 'vanilla'}"] = _r(m, n)
                if kind == "cortical":
                    ax.text(0.97, 0.05, f"n = {n}", transform=ax.transAxes, ha="right", va="bottom",
                            fontsize=FS_TICK)
                ST.despine(ax); ST.value_grid(ax, "y")
                ax.tick_params(labelsize=FS_TICK, length=2, width=0.6, pad=1.5)
                for s in ("left", "bottom"):
                    ax.spines[s].set_linewidth(0.6)
                ax.set_xticks([5, 10, 15] if kind == "hpo" else [5, 10])
                if r == 0:
                    ax.set_title(f"{letter}   {title}", loc="left", fontweight="bold", fontsize=FS_TITLE, pad=3)
                if r == 1:
                    ax.set_xlabel("query", fontsize=FS_LAB, labelpad=1)
                if c == 0:
                    ax.set_ylabel(YLAB[kind], fontsize=FS_LAB, labelpad=2)
                else:
                    ax.tick_params(labelleft=False)
                axes[kind].append(ax)
    # thin separator between the batch panels (D, E) and the kernel panels (F-H) of line 2
    xm = (left + (pw + gap) + pw + left + 2 * (pw + gap) + sep) / 2
    y_top = H_ - (line_h + line_gap) - t_top + 0.18
    y_bot = H_ - (line_h + line_gap) - t_top - ph - x_gap - ph - 0.22
    fig.add_artist(Line2D([xm / W, xm / W], [y_bot / H_, y_top / H_], transform=fig.transFigure,
                          color="#8a8a8a", lw=0.7))
    for kind, (lo, hi) in lims.items():
        pad = 0.04 * (hi - lo)
        for ax in axes[kind]:
            ax.set_ylim(max(0.0, lo - pad) if kind == "hpo" else lo - pad, hi + pad)
    handles = [Line2D([], [], color=ours_c, lw=1.3, ls="-", label="HapBO"),
               Line2D([], [], color=van_c, lw=1.0, ls="--", label="vanilla")]
    # legend in the free space right of line 1 (line 1 has 3 of the 5 panel slots)
    lx = (left + 3 * (pw + gap) + 0.15) / W
    ly = (H_ - t_top - 0.05) / H_
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(lx, ly), fontsize=FS_LEG, frameon=False,
               handlelength=2.4, borderaxespad=0)
    fig.text(lx, (H_ - t_top - ph - x_gap - 0.10) / H_, "cortical: n = technical\nreplicates (muscle outputs)",
             ha="left", va="top", fontsize=FS_TICK)
    ST.save(fig, fig_dir / "fig_bounded_shaping")
    return num
