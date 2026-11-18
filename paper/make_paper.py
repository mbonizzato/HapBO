#!/usr/bin/env python3
"""Regenerate every paper figure from the compact data in results/paper/ (offline: no benchmark data, no LLM).

Writes paper/figures/*.{pdf,png}, paper/figures/supp/*.{pdf,png}, paper/tables/table_ladder_cost.tex and
paper/figures/numbers.json (the values quoted in the text: end-of-budget means, ranks, run counts).

usage: python paper/make_paper.py [--only hpo,realworld,synthetic,prior_sources,ablations,representation,acquisition,
                                          baseline_impl,injection_synthetic,cake_caliper,priorband_primo,hypbo_llmei,
                                          leak_control_cnn,psl_elicitors,alpha_sweep]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "leak_control_cnn"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")

FIG = HERE / "figures"


def _builders():
    import build_regret_fig
    import fig_ablations
    import fig_acquisition
    import fig_alpha_sweep
    import fig_baseline_impl
    import fig_cake_caliper
    import fig_hpo
    import fig_hypbo_llmei
    import fig_injection_synthetic
    import fig_priorband_primo
    import fig_prior_sources
    import fig_psl_elicitors
    import fig_realworld
    import fig_representation
    import fig_synthetic
    return {
        "hpo": lambda: fig_hpo.build(FIG),
        "realworld": lambda: fig_realworld.build(FIG),
        "synthetic": lambda: fig_synthetic.build(FIG),
        "prior_sources": lambda: fig_prior_sources.build(FIG),
        "ablations": lambda: fig_ablations.build(FIG),
        "representation": lambda: fig_representation.build(FIG),
        "injection_synthetic": lambda: fig_injection_synthetic.build(FIG),
        "acquisition": lambda: fig_acquisition.build(FIG),
        "baseline_impl": lambda: fig_baseline_impl.build(FIG),
        "cake_caliper": lambda: fig_cake_caliper.build(FIG),
        "priorband_primo": lambda: fig_priorband_primo.build(FIG),
        "hypbo_llmei": lambda: fig_hypbo_llmei.build(FIG),
        "leak_control_cnn": lambda: build_regret_fig.build(ROOT / "results" / "paper" / "cnn_leak", FIG),
        "psl_elicitors": lambda: fig_psl_elicitors.build(FIG),
        "alpha_sweep": lambda: fig_alpha_sweep.build(FIG),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list of builders (default: all)")
    a = ap.parse_args()
    builders = _builders()
    names = [n.strip() for n in a.only.split(",") if n.strip()] or list(builders)
    FIG.mkdir(parents=True, exist_ok=True)
    npath = FIG / "numbers.json"
    numbers = json.loads(npath.read_text(encoding="utf-8")) if a.only and npath.is_file() else {}
    for n in names:
        print(f"[{n}]", flush=True)
        numbers.update(builders[n]())
    npath.write_text(json.dumps(numbers, indent=1, default=str), encoding="utf-8")
    print("wrote", FIG.relative_to(ROOT))


if __name__ == "__main__":
    main()
