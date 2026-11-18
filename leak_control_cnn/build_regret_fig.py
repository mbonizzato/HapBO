"""fig_leakctrl_regret: query-regret curves (mean +/- SEM over seeds) on the cnn-leak-of benchmark.

Reads results/paper/cnn_leak/<arm>.json (scripts/collect_paper_results.py --only cnn), one file per arm:
  {"optimum": f*, "budget": 30, "runs": [[ground-truth accuracy per query], ...]}
Arms: hapbo, colabo, llambo, bora, lgbo, vanilla, random.
Simple regret = optimum - best-so-far accuracy, floored at 1e-4 for the log axis.

usage: python leak_control_cnn/build_regret_fig.py [--data results/paper/cnn_leak] [--out paper/figures]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
C = {"HapBO": "#0072B2", "ColaBO": "#009E73", "LLAMBO": "#E69F00",
     "BORA": "#CC79A7", "LGBO": "#882255", "vanilla": "#4D4D4D", "random": "#B3B3B3"}
ARMS = [  # file stem, colour key, legend
    ("hapbo", "HapBO", "HapBO"), ("colabo", "ColaBO", "ColaBO"), ("llambo", "LLAMBO", "LLAMBO"),
    ("bora", "BORA", "BORA"), ("lgbo", "LGBO", "LGBO"), ("vanilla", "vanilla", "vanilla BO"),
    ("random", "random", "random search"),
]
Q = 30
FLOOR = 1e-4
# saved extent per format (x0, y0, x1, y1 in inches): fixed, the page size of the figure in the paper
EXTENT = {"pdf": (0.22693480916, 0.40361033056, 6.28778684903, 4.581),
          "png": (0.23459750067, 0.40855569412, 6.28725580327, 4.581)}


def arm_curve(path: Path):
    """(mean, SEM, n) of the simple-regret curve over the first Q queries."""
    d = json.loads(path.read_text(encoding="utf-8"))
    best = np.array([np.maximum.accumulate(np.asarray(r, float))[:Q] for r in d["runs"] if len(r) >= Q])
    reg = d["optimum"] - best
    n = len(reg)
    return reg.mean(0), reg.std(0, ddof=1) / np.sqrt(n), n


def build(data_dir: Path, out_dir: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03, "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 16.6, "axes.titlesize": 18.4, "axes.labelsize": 17.0,
        "axes.titleweight": "bold", "xtick.labelsize": 14.6, "ytick.labelsize": 14.6,
        "legend.fontsize": 13.4, "legend.frameon": False, "axes.linewidth": 0.75,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    x = np.arange(1, Q + 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    nums = {}
    for stem, ckey, label in ARMS:
        path = data_dir / f"{stem}.json"
        if not path.is_file():
            raise FileNotFoundError(f"fig_leakctrl_regret: missing {path}")
        mu, se, n = arm_curve(path)
        nums["LLAMBO (mode 2)" if stem == "llambo" else label] = (round(float(mu[-1]), 4), n)  # LLAMBO run mode
        mu = np.maximum(mu, FLOOR)
        lw = 2.4 if stem == "hapbo" else 1.6
        ls = "--" if stem in ("vanilla", "random") else "-"
        ax.plot(x, mu, ls, color=C[ckey], lw=lw, label=label)
        ax.fill_between(x, np.maximum(mu - se, FLOOR), mu + se, color=C[ckey], alpha=0.13)
    ax.set_yscale("log")
    ax.set_xlabel("query")
    ax.set_ylabel("simple regret\n(optimum $-$ best-so-far)")
    ax.set_xlim(1, Q)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False, columnspacing=1.0,
              handlelength=1.6)
    ax.grid(True, which="both", alpha=0.13)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    from matplotlib.transforms import Bbox
    fig.savefig(out_dir / "fig_leakctrl_regret.pdf", bbox_inches=Bbox.from_extents(*EXTENT["pdf"]))
    fig.savefig(out_dir / "fig_leakctrl_regret.png", dpi=170, bbox_inches=Bbox.from_extents(*EXTENT["png"]))
    plt.close(fig)
    return {"fig_leakctrl_regret": {"q30": nums}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "results" / "paper" / "cnn_leak")
    ap.add_argument("--out", type=Path, default=ROOT / "paper" / "figures")
    a = ap.parse_args()
    print(json.dumps(build(a.data, a.out)))


if __name__ == "__main__":
    main()
