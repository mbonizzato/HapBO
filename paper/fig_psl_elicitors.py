"""PSL elicitors figure (fig_elicitation_ladder) and its table (table_ladder_cost).

What one-shot PSL elicitation captures, what it costs and what it buys, for five elicitors (Qwen3.5, Haiku 5.5,
Sonnet 5.5, Opus 5.5, Fable 5.1) and two routes: one API call (60-token cap, 3 draws) and agentic (Claude Code /
Qwen Code session, token cap 2 x dimension).

Row 1 (A): a 2-D slice of the rho90 card model (goodness -M, min-max on the slice) and pi(x) of each elicitor's one-call
  document (draw 1) on the same slice, through the card-model optimum c (white x). The slice (Styblinski-Tang-7,
  x1 vs x6) is the `slice` entry of the data file.
Row 2: per elicitor one bar per route: (B) precision Spearman(pi, card-model goodness), (C) agreement top-10% overlap,
  (D) tokens per elicitation, (E) list-price US$ per elicitation; bar = mean over the 4 functions (tokens and $:
  geometric mean), dots = functions. Fidelity from scripts/score_fidelity.py (100k uniform points).
Row 3: (F) variety: distinct PSL token types used across the 4 functions (one call, mean per-draw union, of 14);
  (G) what the one-call PSL tokens encode (share per type, pooled over functions and draws);
  (H) HapBO regret on the well truth: per elicitor the mean of the one-call and agentic grand-average curves
  (normalized by the vanilla-BO query-1 regret per function, budget mapped to 0-100 %), vanilla BO dashed.
Table: final normalized regret (100 % of the budget), mean +- SEM over the one-call and agentic function curves.

Data: results/paper/psl_elicitors/psl_elicitors.json and the documents in experiments/psl_elicitor_priors/.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "results" / "paper" / "psl_elicitors" / "psl_elicitors.json"
TABLES = HERE / "tables"

AGENTS = [("qwen3.5", "Qwen3.5", "#e87ba4"), ("haiku-5.5", "Haiku 5.5", "#1baf7a"),
          ("sonnet-5.5", "Sonnet 5.5", "#eb6834"), ("opus-5.5", "Opus 5.5", "#2a78d6"),
          ("fable-5.1", "Fable 5.1", "#6250d6")]
FAMS = ["hartmann6", "styblinski_tang7", "levy10", "ackley15"]
FNAME = {"hartmann6": "Hartmann-6", "styblinski_tang7": "Styblinski-Tang-7", "levy10": "Levy-10",
         "ackley15": "Ackley-15"}
GRID = np.linspace(0, 100, 101)
# PSL vocabulary order: shaping tokens first, meta tokens last (they do not shape pi); `categorical` needs a
# categorical dimension and every input here is continuous, so 14 types apply.
TYPES = ["bump", "joint_region", "periodic", "trend", "ridge", "plateau", "threshold", "avoid", "anchors",
         "conditional", "baseline", "modality", "smoothness", "noise"]
N_APPLICABLE = 14
TCOL = {"bump": "#2a78d6", "joint_region": "#7aa9e6", "periodic": "#eb6834", "trend": "#f2a07b", "ridge": "#1baf7a",
        "plateau": "#8fd6b8", "threshold": "#6250d6", "avoid": "#b4aaf0", "anchors": "#d6a21b",
        "conditional": "#e87ba4", "baseline": "#9a9a95", "modality": "#bdbdb8", "smoothness": "#d6d6d1",
        "noise": "#ebebe6"}


def _paths():
    for p in (str(ROOT / "src"), str(ROOT / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)


def slice_points(fam, c, i, j, n):
    """n x n grid over (x_i, x_j) through the point c (other coordinates fixed at c)."""
    import synthetic_truth as T

    _, lo, hi = T.BOUNDS[fam]
    g = np.linspace(lo, hi, n)
    A, B = np.meshgrid(g, g)
    X = np.tile(np.asarray(c, float), (n * n, 1))
    X[:, i], X[:, j] = A.ravel(), B.ravel()
    return X


def pi_slice(doc_path, fam, X):
    """pi(x) of a PSL document compiled with the production compiler, on native points X (columns x0..x{d-1})."""
    import synthetic_truth as T
    from prior.landscape.landscape_generator import compile_prior

    cp = compile_prior(json.loads(Path(doc_path).read_text(encoding="utf-8")))
    names = [f"x{k}" for k in range(T.BOUNDS[fam][0])]
    order = [names.index(str(p).lower()) for p in cp.parameter_names]
    return np.asarray(cp.pi(X[:, order]), float)


def build(fig_dir: Path, n: int = 161) -> dict:
    _paths()
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    import synthetic_truth as T

    D = json.loads(DATA.read_text(encoding="utf-8"))
    with matplotlib.rc_context():
        matplotlib.rc_file_defaults()
        plt.rcParams.update({"pdf.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                             "font.size": 7.5, "axes.linewidth": 0.6, "axes.spines.top": False,
                             "axes.spines.right": False})
        summary = {}
        fig = plt.figure(figsize=(7.25, 7.4))
        outer = fig.add_gridspec(3, 1, height_ratios=[0.8, 1.15, 1.25], hspace=0.6)

        # ---------------- row 1: slices
        sl = D["slice"]
        fam, i, j, draw = sl["family"], sl["i"], sl["j"], sl["draw"]
        c = T.load_well_truth()["card_model"][fam]["x_opt"]
        _, lo, hi = T.BOUNDS[fam]
        X = slice_points(fam, c, i, j, n)
        g = -T.card_model(fam)(X)
        panels = [("card model", f"{FNAME[fam]}, x{i} vs x{j}", (g - g.min()) / (g.max() - g.min()))]
        for key, lab, _ in AGENTS:
            path = ROOT / D["documents"][key][f"draw{draw}"][fam]
            panels.append((lab, "one API call", pi_slice(path, fam, X)))
        g1 = outer[0].subgridspec(1, len(panels) + 1, width_ratios=[1] * len(panels) + [0.07], wspace=0.12)
        for k, (t, sub, Z) in enumerate(panels):
            ax = fig.add_subplot(g1[0, k])
            im = ax.imshow(Z.reshape(n, n), origin="lower", extent=(lo, hi, lo, hi), cmap="viridis", vmin=0, vmax=1,
                           interpolation="bilinear")
            ax.plot(c[i], c[j], marker="x", color="white", ms=4, mew=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)
            ax.set_title(t, fontsize=8, family="monospace", pad=10)
            ax.text(0.5, 1.03, sub, transform=ax.transAxes, ha="center", fontsize=6.3, color="#555")
            if k == 0:
                ax.text(-0.18, 1.32, "A", transform=ax.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold")
                ax.set_xlabel(f"$x_{{{i}}}$", labelpad=1)
                ax.set_ylabel(f"$x_{{{j}}}$", labelpad=1)
        cax = fig.add_subplot(g1[0, -1])
        cb = fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1])
        cb.set_label(r"$\pi(x)$ / goodness", fontsize=7)
        summary["slice"] = {"family": fam, "i": i, "j": j}

        # ---------------- row 2: precision, agreement, tokens, $
        g2 = outer[1].subgridspec(1, 4, wspace=0.42)
        xs = np.arange(len(AGENTS))
        W = 0.38
        RND = [("one-call", -W / 2, None), ("agentic", W / 2, "////")]

        def metric(ag, route, f, mk):
            vals = D["metrics"][ag][route][f][mk]
            return float(np.mean(vals)) if vals else None

        def use(ag, route, f):
            u = D["usage"].get(ag, {}).get(route, {}).get(f)
            return None if u is None else (u["tokens"], u["usd"])

        def bars(ax, getter, log=False):
            out = {}
            for k, (agent, lab, col) in enumerate(AGENTS):
                for route, off, hatch in RND:
                    vals = getter(agent, route)
                    if not vals:
                        continue
                    m = float(np.exp(np.mean(np.log(np.maximum(vals, 1e-9))))) if log else float(np.mean(vals))
                    ax.bar(k + off, m, width=W, color=col if hatch is None else "white", edgecolor=col, hatch=hatch,
                           lw=0.8, alpha=0.9)
                    ax.scatter(np.full(len(vals), k + off), vals, s=5, color="#333", zorder=3, lw=0)
                    out[f"{agent}/{route}"] = round(m, 4)
            ax.set_xticks(xs)
            ax.set_xticklabels([lab for _, lab, _ in AGENTS], rotation=45, ha="right")
            if log:
                ax.set_yscale("log")
            ax.grid(axis="y", alpha=0.3, lw=0.5)
            return out

        titles = ["precision", "agreement", "cost: tokens", "cost: list price"]
        ylabs = [r"Spearman($\pi$, model)", "top-10% overlap", "tokens / elicitation", "US$ / elicitation"]
        getters = [lambda ag, r: [v for f in FAMS if (v := metric(ag, r, f, "spearman_model")) is not None],
                   lambda ag, r: [v for f in FAMS if (v := metric(ag, r, f, "top10_model")) is not None],
                   lambda ag, r: [u[0] for f in FAMS if (u := use(ag, r, f)) is not None],
                   lambda ag, r: [u[1] for f in FAMS if (u := use(ag, r, f)) is not None and u[1] > 0]]
        for k in range(4):
            ax = fig.add_subplot(g2[0, k])
            summary[titles[k]] = bars(ax, getters[k], log=k >= 2)
            ax.set_title(f"{'BCDE'[k]}  {titles[k]}", loc="left", fontsize=8, fontweight="bold")
            ax.set_ylabel(ylabs[k], fontsize=7)
            if k < 2:
                ax.set_ylim(0, 1.02)
            if k == 3:
                ax.set_xticklabels(["Qwen3.5\n(local, $0)"] + [lab for _, lab, _ in AGENTS[1:]], rotation=45,
                                   ha="right")
            if k == 0:
                row2_top = ax.get_position().y1
        fig.legend(handles=[Patch(facecolor="#999", edgecolor="#999", label="one API call"),
                            Patch(facecolor="white", edgecolor="#999", hatch="////", label="agentic")],
                   loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, row2_top + 0.035))

        # ---------------- row 3: variety, composition (one call), well-truth regret
        g3 = outer[2].subgridspec(1, 4, width_ratios=[0.8, 1.1, 0.45, 1.6], wspace=0.45)
        axv, axc, axr = (fig.add_subplot(g3[0, k]) for k in (0, 1, 3))
        comp_all = {}
        for k, (agent, lab, col) in enumerate(AGENTS):
            unions, comp = [], collections.Counter()
            for tag in sorted(D["token_types"][agent]):
                used = set()
                for f in FAMS:
                    counts = D["token_types"][agent][tag].get(f)
                    if counts:
                        used |= {t for t, m in counts.items() if m}
                        comp.update(counts)
                if used:
                    unions.append(len(used))
            axv.bar(k, np.mean(unions), width=0.65, color=col)
            axv.scatter(np.full(len(unions), k), unions, s=5, color="#333", zorder=3, lw=0)
            tot = sum(comp.values())
            comp_all[agent] = {t: comp.get(t, 0) / tot for t in TYPES}
            bottom = 0.0
            for t in TYPES:
                s = comp_all[agent][t]
                if s:
                    axc.bar(k, s, bottom=bottom, width=0.7, color=TCOL[t], edgecolor="white", lw=0.3)
                    bottom += s
            summary.setdefault("variety_one_call", {})[agent] = round(float(np.mean(unions)), 2)
        summary["composition_one_call"] = {ag: {t: round(v, 3) for t, v in dd.items() if v}
                                           for ag, dd in comp_all.items()}
        for ax in (axv, axc):
            ax.set_xticks(xs)
            ax.set_xticklabels([lab for _, lab, _ in AGENTS], rotation=45, ha="right")
        axv.set_ylim(0, N_APPLICABLE)
        axv.set_ylabel("token types used (of 14)", fontsize=7)
        axv.set_title("F  variety", loc="left", fontsize=8, fontweight="bold")
        axc.set_ylim(0, 1)
        axc.set_ylabel("share of PSL tokens", fontsize=7)
        axc.set_title("G  what the tokens encode", loc="left", fontsize=8, fontweight="bold")
        used_types = [t for t in TYPES if any(comp_all[ag].get(t, 0) for ag in comp_all)]
        axc.legend(handles=[Patch(color=TCOL[t], label=t) for t in used_types][::-1], fontsize=5.6, frameon=False,
                   loc="center left", bbox_to_anchor=(1.02, 0.5), handlelength=0.9, labelspacing=0.25,
                   borderaxespad=0)

        R = D["regret"]
        van = {f: np.asarray(R["vanilla"][f], float) for f in FAMS}
        ref = {f: float(van[f][:, 0].mean()) for f in FAMS}

        def cells(agent=None, route=None):
            out = []
            for f in FAMS:
                r = van[f] if agent is None else np.asarray(R[agent][route][f], float)
                m = r.mean(0) / ref[f]
                out.append(np.interp(GRID, np.linspace(0, 100, len(m)), m))
            return np.asarray(out)

        vv = cells().mean(0)
        table = []
        axr.plot(GRID, vv, color="#4d4d4d", ls="--", lw=1.2, label="vanilla BO")
        for agent, lab, col in AGENTS:
            api, ag = cells(agent, "one-call"), cells(agent, "agentic")
            m = 0.5 * (api.mean(0) + ag.mean(0))
            both = np.concatenate([api, ag])
            se = both.std(0, ddof=1) / np.sqrt(len(both))
            axr.plot(GRID, m, color=col, lw=1.4, label=lab)
            axr.fill_between(GRID, np.maximum(m - se, 1e-4), m + se, color=col, alpha=0.07, lw=0)
            summary.setdefault("well_regret_final", {})[agent] = round(float(m[-1]), 4)
            table.append((lab, m, se))
        summary["well_regret_final"]["vanilla"] = round(float(vv[-1]), 4)
        vcells = cells()
        table.append(("vanilla BO", vv, vcells.std(0, ddof=1) / np.sqrt(len(vcells))))
        best = {q: min(round(float(m[q]), 3) for lab, m, _ in table if lab != "vanilla BO") for q in (50, 100)}

        def cell(m, se, q):  # bold = best among elicitors at the printed precision (ties all bold)
            s = f"{m[q]:.3f} $\\pm$ {se[q]:.3f}"
            return f"\\textbf{{{m[q]:.3f}}} $\\pm$ {se[q]:.3f}" if round(float(m[q]), 3) == best[q] else s

        tex = ["\\begin{tabular}{lc}", "\\toprule", "Elicitor & final regret\\\\", "\\midrule"]
        for lab, m, se in table:
            if lab == "vanilla BO":
                tex.append("\\midrule")
            tex.append(f"{lab} & {cell(m, se, 100)}\\\\")
            summary.setdefault("table", {})[lab] = {"q50": round(float(m[50]), 4), "q50_sem": round(float(se[50]), 4),
                                                    "q100": round(float(m[100]), 4),
                                                    "q100_sem": round(float(se[100]), 4)}
        tex += ["\\bottomrule", "\\end{tabular}"]
        TABLES.mkdir(parents=True, exist_ok=True)
        (TABLES / "table_ladder_cost.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
        axr.set(yscale="log", xlim=(0, 100))
        axr.set_xlabel("fraction of budget (%)")
        axr.set_ylabel("normalized simple regret (↓)", fontsize=7)
        axr.set_title("H  HapBO regret, well truth", loc="left", fontsize=8, fontweight="bold")
        axr.grid(axis="y", alpha=0.3, lw=0.5)
        axr.legend(fontsize=6, frameon=False, loc="lower left")

        fig_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(fig_dir / f"fig_elicitation_ladder.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
    return {"fig_elicitation_ladder": summary}
