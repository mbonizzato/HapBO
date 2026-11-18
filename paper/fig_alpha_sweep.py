"""fig_alpha_sweep_hapbo: HapBO with its trust weight alpha fixed instead of LLM-compiled (HPOBench budget 16,
40 tasks).

  left    card prior: regret vs query per alpha (grid-span normalized, the metric of fig2_hpo_compact)
  middle  card + misleading hint (the adversarial prior of fig_ablations panel B): regret vs query per alpha
          (vanilla-q1 normalized, the metric of fig5_adv_compact, log scale)
  right   card + misleading hint: regret at query 16 vs alpha

Arms: alpha fixed at 0.9, 0.7, 0.5, 0.3, 0.1 (hpo/alpha-sweep, hpo/alpha-sweep-misleading), the LLM-compiled alpha
(hpo/hapbo, hpo/hapbo-misleading; drawn at the task mean of the compiled alpha of the priors) and alpha = 0, i.e.
vanilla BO (hpo/vanilla-random). Every arm is alpha-pi-raw-ei-matched with the same priors, seeds and matched
initial design. Metrics as in hpo_data, regret = grid maximum - best so far, clipped at 0; seed mean per task,
mean +- SEM over the tasks of the (task, seed) pairs completed in every arm.
"""
from __future__ import annotations

import sys

import numpy as np

import hpo_data as H
import style as ST

FIXED = (0.9, 0.7, 0.5, 0.3, 0.1)
COMPILED, VANILLA = "compiled", "vanilla"
STUDIES = {"card": ("alpha-sweep", "hapbo", "llm-card", "span"),
           "misleading": ("alpha-sweep-misleading", "hapbo-misleading", "llm-card-misleading", "vq1")}
HAPBO = "alpha-pi-raw-ei-matched"
Q = H.BUDGET


def method_id(alpha: float) -> str:
    """Catalog id of the fixed-alpha arm (experiments/catalog/methods)."""
    return f"{HAPBO}-alpha-{alpha:.1f}".replace(".", "-")


def arm_source(cond: str) -> dict:
    """arm -> (study, method); arm = fixed alpha, 'compiled' or 'vanilla'."""
    sweep, hapbo, _, _ = STUDIES[cond]
    src = {a: (sweep, method_id(a)) for a in FIXED}
    src[COMPILED] = (hapbo, HAPBO)
    src[VANILLA] = ("vanilla-random", "vanilla-bo-ei")
    return src


def compiled_alpha(prior_id: str) -> dict:
    """{unit: alpha compiled from the task's PSL prior (experiments/priors/hpobench-<unit>/<task>/<prior_id>)}."""
    src = str(H.ROOT / "src")
    if src not in sys.path:
        sys.path.append(src)
    from prior.landscape.landscape_generator import compile_prior, load_record

    out = {}
    for f in H.FAMS:
        for t in H.TASKS:
            u, slug = H.unit(f, t), t.replace("_", "-")
            path = H.ROOT / "experiments" / "priors" / f"hpobench-{u}" / slug / prior_id / "prior.json"
            out[u] = float(compile_prior(load_record(path)).alpha)
    return out


def vanilla_ref_q1() -> dict:
    """{(fam, task): vanilla BO seed-mean regret at query 1} (vanilla-q1 normalizer, as in fig5_adv_compact)."""
    out = {}
    for f in H.FAMS:
        for t in H.TASKS:
            c, _ = H.seed_regret("vanilla-random", "vanilla-bo-ei", f, t)
            if c is not None:
                out[(f, t)] = float(c[0])
    return out


def _denominator(metric: str, fam: str, task: str, vref: dict):
    if metric == "span":
        return H.grid_span(fam, task)
    ref = vref.get((fam, task))
    return ref if ref is not None and ref > 1e-12 else None


def _best_curves(study: str, method: str) -> dict:
    """{(fam, task, seed): best-so-far curve} of the completed runs."""
    out = {}
    for f in H.FAMS:
        for t in H.TASKS:
            for r in H.runs("hpo", study, method, H.unit(f, t)):
                c = H.best_curve(r, Q)
                if c is not None:
                    out[(f, t, int(r["seed"]))] = c
    return out


def condition_summary(cond: str, vref: dict) -> dict:
    """Per arm: task mean and SEM curves of the condition's metric; compiled-alpha task mean."""
    _, _, prior_id, metric = STUDIES[cond]
    curves = {arm: _best_curves(*src) for arm, src in arm_source(cond).items()}
    pairs = sorted(set.intersection(*(set(c) for c in curves.values())))
    pairs = [k for k in pairs if _denominator(metric, k[0], k[1], vref) is not None]
    tasks = sorted({(f, t) for f, t, _ in pairs})
    index = {t: [i for i, k in enumerate(pairs) if (k[0], k[1]) == t] for t in tasks}
    arms = {}
    for arm, cs in curves.items():
        m = np.array([np.maximum((H.grid_max(f, t) - cs[(f, t, s)]) / _denominator(metric, f, t, vref), 0.0)
                      for f, t, s in pairs])
        per_task = np.array([m[index[t]].mean(0) for t in tasks])
        mean, sem = per_task.mean(0), per_task.std(0, ddof=1) / np.sqrt(len(tasks))
        arms[arm] = {"mean": mean, "sem": sem}
    alphas = compiled_alpha(prior_id)
    used = {H.unit(f, t) for f, t in tasks}
    return {"metric": metric, "n_tasks": len(tasks), "n_pairs": len(pairs), "arms": arms,
            "compiled_alpha_mean": float(np.mean([a for u, a in alphas.items() if u in used]))}


def ramp(alpha: float) -> str:
    """Blues ramp by trust level: alpha = 0.1 light ... alpha = 0.9 dark."""
    import matplotlib as mpl
    return mpl.colors.to_hex(mpl.colormaps["Blues"](0.25 + 0.70 * float(alpha)))


def curves_panel(ax, S: dict) -> None:
    q = np.arange(1, Q + 1)
    xa = S["compiled_alpha_mean"]
    order = [(f"α = {a:.1f}", a, a) for a in FIXED] + [(f"LLM-compiled α (mean ≈ {xa:.2f})", COMPILED, xa)]
    for label, arm, a in sorted(order, key=lambda o: -o[2]):
        m, e = S["arms"][arm]["mean"], S["arms"][arm]["sem"]
        if arm == COMPILED:
            ax.plot(q, m, color=ramp(a), lw=2.6, zorder=6, label=label)
            ax.fill_between(q, np.maximum(m - e, 1e-12), m + e, color=ramp(a), alpha=0.15, lw=0, zorder=5)
        else:
            ax.plot(q, m, color=ramp(a), lw=1.2, zorder=4, label=label)
    ax.plot(q, S["arms"][VANILLA]["mean"], color=ST.C["vanilla"], lw=1.1, ls="--", zorder=3,
            label="vanilla EI (α = 0)")
    if S["metric"] == "vq1":
        import matplotlib.ticker as mt
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mt.FixedLocator([0.15, 0.2, 0.3, 0.5, 0.7, 1.0]))
        ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(mt.NullFormatter())
    else:
        ax.set_ylim(bottom=0)
    ax.set(xlim=(1, Q), xticks=[1, 4, 8, 12, 16])
    ax.set_xlabel("query")
    ax.set_ylabel("normalized regret (↓)", fontsize=7.4)


def q16_panel(ax, S: dict) -> None:
    pts = [(a, S["arms"][a]) for a in sorted(FIXED)]
    pts.insert(0, (0.0, S["arms"][VANILLA]))
    xs = np.array([a for a, _ in pts])
    ys = np.array([e["mean"][-1] for _, e in pts])
    es = np.array([e["sem"][-1] for _, e in pts])
    ax.plot(xs, ys, color=ST.C["HapBO"], lw=1.4, zorder=3)
    for a, y, s in zip(xs, ys, es):
        col = ST.C["vanilla"] if a == 0 else ramp(a)
        ax.errorbar([a], [y], yerr=[s], color=col, marker="o", ms=5, lw=0, capsize=2.2, elinewidth=0.9, zorder=5)
    xa, c = S["compiled_alpha_mean"], S["arms"][COMPILED]
    ax.errorbar([xa], [c["mean"][-1]], yerr=[c["sem"][-1]], color=ramp(xa), marker="o", ms=5, mec="black", mew=0.6,
                lw=0, capsize=2.2, elinewidth=0.9, zorder=7)
    ax.axvline(xa, color=ramp(xa), lw=0.6, ls=":", zorder=1)
    ax.axhline(ys[0], color=ST.C["vanilla"], lw=1.0, ls="--", zorder=2)
    ax.set(xlim=(-0.05, 0.95), xticks=[0, 0.1, 0.3, 0.5, 0.7, 0.9])
    ax.set_xticklabels(["0", ".1", ".3", ".5", ".7", ".9"])
    ax.set_xlabel("trust weight α")
    ax.set_ylim(0, 1.12 * max(list(ys + es) + [c["mean"][-1] + c["sem"][-1]]))
    ax.set_ylabel("final normalized regret (↓)", fontsize=7.4)


def make_figure():
    """(figure, summary) of fig_alpha_sweep_hapbo."""
    import matplotlib.pyplot as plt
    ST.set_style()
    vref = vanilla_ref_q1()
    G, S = condition_summary("card", vref), condition_summary("misleading", vref)
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(10.4, 2.95))
    curves_panel(a0, G)
    a0.set_title("HPOBench (good prior)", loc="left", fontsize=7.6)
    a0.legend(fontsize=6.0, loc="upper right", handlelength=1.8, framealpha=0.9,
              title="highest → lowest trust", title_fontsize=6.0)
    curves_panel(a1, S)
    a1.set_title("HPOBench (adversarial prior)", loc="left", fontsize=7.6)
    q16_panel(a2, S)
    a2.set_title(f"HPOBench b16 (adversarial prior) – q{Q}", loc="left", fontsize=7.6)
    for ax in (a0, a1, a2):
        ST.value_grid(ax, "y")
    fig.tight_layout()
    summary = {cond: {"metric": X["metric"], "n_tasks": X["n_tasks"], "n_pairs": X["n_pairs"],
                      "compiled_alpha_mean": round(X["compiled_alpha_mean"], 4),
                      "q16": {str(arm): round(float(e["mean"][-1]), 4) for arm, e in X["arms"].items()}}
               for cond, X in (("card", G), ("misleading", S))}
    return fig, summary


def build(fig_dir):
    fig, summary = make_figure()
    ST.save(fig, fig_dir / "fig_alpha_sweep_hapbo")
    return {"fig_alpha_sweep_hapbo": summary}
