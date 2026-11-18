#!/usr/bin/env python3
"""Collect the compact data read by paper/make_paper.py into results/paper/.

Inputs (written by the campaign runners; see REPRODUCING.md):
  results/studies/<group>/<study>/runs/<benchmark>/<unit>/<method>/budget-<B>/seed-<s>/{status.json,evaluations.jsonl}
    (multi-fidelity PriorBand / PriMO runs: mf_evaluations.jsonl, mf_accounting.json)
  results/exploration_scores/cortical/<study>.json         (scripts/score_exploration.py)
  results/diagnostics/solar_external/<stem>_penalized_MAXEFF_RE_s<seed>_prescreen.jsonl   (scripts/solar_externals.py)
  results/diagnostics/solar_bora/bora_MAXEFF_RE_s<seed>_prescreen.json                 (scripts/bora_solar_cached.py)
  results/synthetic/well/<card>/... and .../none/...       (scripts/run_synthetic_panel.py; layout in collect_synthetic)
  benchmark/HPOBench, benchmark/NeuroMOSAIC/dataset1/rat (task statistics and example maps)

Outputs (results/paper/, committed):
  hpo/<study>.json, cortical/<study>.json, solar/<study>.json   run trajectories per method / budget / unit / seed
  hpo/task_stats.json                                         grid maximum, minimum, mean and optimum per task
  hpo/{priorband,primo}[-misleading].json                    full-fidelity evaluations vs cumulative cost per run
  cortical/map_optimum.json                                   best true mean response per map
  cortical/exploration/<study>.json                           exploration-performance curves (budget-12 runs)
  cortical/prior_sources.json                                 compiled prior fields and example truth maps
                                                              (supp/figS2_prior_sources_and_trajectories)
  solar/external.json                                         ColaBO, LGBO, LLAMBO, BORA records per seed
  synthetic/<card>.json, synthetic/none.json                  best-so-far regret trajectories per arm and function
  cnn_leak/<arm>.json                                         CNN leak-control trajectories (fig_leakctrl_regret)

usage: python scripts/collect_paper_results.py [--only hpo,cortical,solar,synthetic,cnn]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper"

HPO_STUDIES = ["hapbo", "vanilla-random", "hapbo-inverted", "hapbo-misleading", "lgbo", "lgbo-misleading",
               "lgbo-official", "lgbo-official-misleading", "colabo", "colabo-misleading", "bora", "bora-misleading",
               "llambo", "llambo-misleading", "llambo-mode1", "llambo-mode1-misleading", "injection",
               "injection-misleading", "representation-pointonly", "representation-gaussian",
               "representation-directllm", "representation-pointonly-misleading", "representation-gaussian-misleading",
               "representation-directllm-misleading",
               "acquisition", "batch", "cake", "caliper", "caliper-misleading", "hypbo",
               "hypbo-misleading", "llm-ei", "llm-ei-misleading", "kernel-rbf", "kernel-matern12", "alpha-sweep",
               "alpha-sweep-misleading"]
MF_HPO_STUDIES = ["priorband", "priorband-misleading", "primo", "primo-misleading"]  # multi-fidelity (NePS)
CORTICAL_STUDIES = ["human-map", "human-map-all", "acquisition", "batch", "kernel-rbf", "kernel-matern12"]
EXPLORATION_STUDIES = ["human-map", "llm-geometry", "llm-expert-hint", "human-map-all", "human-map-colabo",
                       "human-map-llambo", "human-map-lgbo", "human-map-bora"]
FAMS = ["lr", "rf", "xgb", "nn", "svm"]
TASKS = ["australian", "blood-transfusion", "car", "credit-g", "kc1", "phoneme", "segment", "vehicle"]
SYNTH_FAMS = ("hartmann6", "styblinski_tang7", "levy10", "ackley15")
SYNTH_INJECTION_ARMS = ("pi-bo", "prior-to-mean", "prior-warm-start", "prior-greedy")
SYNTH_REPRESENTATION_CARD = "rho90"  # card of the representation ablation (fig_representation)
SYNTH_REPRESENTATION_ARMS = {"pointonly": "hapbo-pointonly", "gaussian": "hapbo-gaussian"}  # arm -> run directory
SOLAR_STEMS = {"ColaBO": "colabo", "LGBO": "lgbo", "LLAMBO": "llambo-ws"}
FIGS2_PRIORS = {"geo": "llm-geometry", "mix": "llm-expert-hint", "human": "human-expert-map"}
FIGS2_EXAMPLES = [("rat1", "LeftKneeFlexor"), ("rat2", "LeftTibialisAnterioris"), ("rat3", "RightTibialisAnterioris")]


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT) if ROOT in path.parents else path} ({path.stat().st_size / 1e3:.0f} kB)")


def _jsonl(p: Path):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _record(rec: dict, fields):
    if len(fields) == 1:
        return rec.get(fields[0])
    return [int(bool(rec.get(f))) if isinstance(rec.get(f), bool) else rec.get(f) for f in fields]


# ------------------------------------------------------------------------------------------------ study runs
def collect_study(run_roots, fields=("ground_truth_objective",), budgets=None) -> dict:
    """{benchmark, methods: {method: {budget-XXXX: {unit: [{seed, status, y}]}}}} from one or more runs/ trees.

    y holds `fields` of every evaluation of a completed run (one value per query, or a list when several fields)."""
    out: dict = {"fields": list(fields), "methods": {}}
    for run_root in run_roots:
        for bench in sorted(p for p in Path(run_root).iterdir() if p.is_dir()):
            out["benchmark"] = bench.name
            for unit in sorted(p for p in bench.iterdir() if p.is_dir()):
                for mdir in sorted(p for p in unit.iterdir() if p.is_dir()):
                    for bdir in sorted(mdir.glob("budget-*")):
                        if budgets and bdir.name not in budgets:
                            continue
                        rows = out["methods"].setdefault(mdir.name, {}).setdefault(bdir.name, {}).setdefault(unit.name, [])
                        for sdir in sorted(bdir.glob("seed-*")):
                            st = sdir / "status.json"
                            status = json.loads(st.read_text())["status"] if st.is_file() else "missing"
                            ev = sdir / "evaluations.jsonl"
                            y = [_record(r, fields) for r in _jsonl(ev)] if ev.is_file() else None
                            rows.append({"seed": int(sdir.name.split("-")[1]), "status": status, "y": y})
    return out


def collect_mf_study(run_roots, budgets=None) -> dict:
    """Multi-fidelity runs (PriorBand / PriMO, cost-charged): per run the [cumulative cost, true value] of
    every full-fidelity evaluation (mf_evaluations.jsonl) and the run's cost accounting (mf_accounting.json).

    The x-axis of these runs is the cumulative full-fidelity-equivalent cost (each evaluation costs z/z_max),
    not the query index."""
    out: dict = {"fields": ["cumulative_cost", "true_full_fidelity_value"],
                 "x": "cumulative full-fidelity-equivalent cost", "methods": {}}
    for run_root in run_roots:
        for bench in sorted(p for p in Path(run_root).iterdir() if p.is_dir()):
            out["benchmark"] = bench.name
            for unit in sorted(p for p in bench.iterdir() if p.is_dir()):
                for mdir in sorted(p for p in unit.iterdir() if p.is_dir()):
                    for bdir in sorted(mdir.glob("budget-*")):
                        if budgets and bdir.name not in budgets:
                            continue
                        rows = out["methods"].setdefault(mdir.name, {}).setdefault(bdir.name, {}).setdefault(unit.name, [])
                        for sdir in sorted(bdir.glob("seed-*")):
                            st = sdir / "status.json"
                            status = json.loads(st.read_text())["status"] if st.is_file() else "missing"
                            ev, acc = sdir / "mf_evaluations.jsonl", sdir / "mf_accounting.json"
                            y = ([[r["cumulative_cost"], r["true_full_fidelity_value"]] for r in _jsonl(ev)
                                  if r["full_fidelity"]] if ev.is_file() else None)
                            a = json.loads(acc.read_text()) if acc.is_file() else {}
                            rows.append({"seed": int(sdir.name.split("-")[1]), "status": status, "y": y,
                                         "cost_used": a.get("cost_used"), "n_evaluations": a.get("n_evaluations"),
                                         "repeat_units": a.get("repeat_units")})
    return out


def hpo_task_stats(benchmark_dir: Path) -> dict:
    from datasets.hpobench import load_hpobench_problem
    stats = {}
    for f in FAMS:
        for t in TASKS:
            p = load_hpobench_problem(model=f, task=t.replace("-", "_"), benchmark_directory=benchmark_dir)
            y = np.asarray(p.ground_truth, float)
            stats[f"{f}-{t}"] = {"max": float(y.max()), "min": float(y.min()), "mean": float(y.mean()),
                                 "optimum": float(p.optimum_value)}
    return stats


def _rat_problem(data_dir: Path, rat: str, muscle: str):
    from datasets.neuromosaic_dataset1_rat import load_neuromosaic_dataset1_rat_problem
    return load_neuromosaic_dataset1_rat_problem(subject_path=next(Path(data_dir).glob(f"{rat}_*.mat")),
                                                 muscle=muscle, subject=rat)


def cortical_map_optimum(data_dir: Path, units) -> dict:
    out = {}
    for unit in units:
        rat, slug = unit.split("-", 1)
        p = _rat_problem(data_dir, rat, "".join(w.capitalize() for w in slug.split("-")))
        out[unit] = float(np.max(p.ground_truth))
    return out


def prior_sources(priors_dir: Path, data_dir: Path, prior_ids=FIGS2_PRIORS) -> dict:
    """Compiled plausibility field pi(x) of each prior (8 x 4 electrode grid) and three min-max normalized truth maps."""
    import hashlib
    from prior.landscape.landscape_generator import compile_prior
    out = {"priors": {}, "examples": []}
    cands = [{"electrode_mediolateral_index": ml, "electrode_rostrocaudal_index": rc}
             for rc in range(1, 9) for ml in range(1, 5)]
    for cond, pid in prior_ids.items():
        files = sorted(Path(priors_dir).glob(f"*/{pid}/prior.json"))
        if not files or len({hashlib.sha256(f.read_bytes()).hexdigest() for f in files}) != 1:
            raise RuntimeError(f"{pid}: expected one prior shared by all muscles")
        comp = compile_prior(json.loads(files[0].read_text(encoding="utf-8")))
        field = np.full((8, 4), np.nan)
        for v, c in zip(np.asarray(comp.pi(cands), float), cands):
            field[c["electrode_rostrocaudal_index"] - 1, c["electrode_mediolateral_index"] - 1] = v
        out["priors"][cond] = {"prior_id": pid, "field": field.tolist()}
    for rat, muscle in FIGS2_EXAMPLES:
        p = _rat_problem(data_dir, rat, muscle)
        g = np.full((8, 4), np.nan)
        for c, v in zip(p.native_candidates, p.ground_truth):
            g[c["channel_x"] - 1, c["channel_y"] - 1] = v
        g = (g - np.nanmin(g)) / (np.nanmax(g) - np.nanmin(g))
        out["examples"].append({"rat": rat, "muscle": muscle, "field": g.tolist()})
    return out


# ------------------------------------------------------------------------------------------------ SOLAR
def solar_external(external: Path, bora: Path, stems=SOLAR_STEMS) -> dict:
    """Per arm: list over seeds (sorted by file name) of [[feasible, objective], ...]."""
    out = {}
    for label, stem in stems.items():
        out[label] = [[[int(bool(r.get("feasible", False))), r.get("objective")] for r in _jsonl(p)]
                      for p in sorted(Path(external).glob(f"{stem}_penalized_MAXEFF_RE_s*_prescreen.jsonl"))]
    out["BORA"] = [[[int(bool(r.get("feasible", False))), r.get("objective")]
                    for r in json.loads(p.read_text(encoding="utf-8"))["records"]]
                   for p in sorted(Path(bora).glob("bora_MAXEFF_RE_s*_prescreen.json"))]
    return out


# ------------------------------------------------------------------------------------------------ synthetic
def _traj(d, fam, key):
    return (d or {}).get(fam, {}).get(key, [])


def _j(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def _round(trs):
    return [[float(f"{v:.10g}") for v in t] for t in trs]


def collect_synthetic(card_root: Path, card: str) -> dict:
    """Best-so-far regret trajectories of every card-reading arm for one card (rho90, well50 or well10).

    Layout written by scripts/run_synthetic_panel.py under results/synthetic/well/<card>/:
      hapbo/<fam>-draw<d>/trajectories.json                         (3 draws)
      hapbo-{pointonly,gaussian}/<fam>-draw<d>/trajectories.json    (card rho90: representation arms pointonly, gaussian)
      colabo/<fam>-<card>-seed<k>.json, lgbo/<fam>-<card>-lgbo.json,
      llambo/<fam>-<card>-zero-shot.json, bora/<fam>-formula-<card>.json
      vanilla-peaks/<fam>-draw<d>/trajectories.json                 (control arm; 3 draws, the hapbo documents)
      <inj>/<fam>-draw<d>/trajectories.json, inj = pi-bo, prior-to-mean, prior-warm-start, prior-greedy
                                                                    (injection arms; listed only when present)"""
    doc_arms = {"hapbo": "hapbo", **(SYNTH_REPRESENTATION_ARMS if card == SYNTH_REPRESENTATION_CARD else {})}
    out = {a: {} for a in (*doc_arms, "colabo", "lgbo", "llambo", "bora", "vanilla-peaks")}
    for fam in SYNTH_FAMS:
        for arm, sub, key in (*((a, s, "hapbo") for a, s in doc_arms.items()),
                              ("vanilla-peaks", "vanilla-peaks", "vanilla-peaks")):
            tr = []
            for d in (1, 2, 3):
                tr += _traj(_j(card_root / sub / f"{fam}-draw{d}" / "trajectories.json"), fam, key)
            out[arm][fam] = _round(tr)
        tr = []
        files = sorted((card_root / "colabo").glob(f"{fam}-{card}-seed*.json"),
                       key=lambda q: int(q.stem.rsplit("seed", 1)[1]))
        for q in files:
            tr += (_j(q) or {}).get("trajectories", [])
        out["colabo"][fam] = _round(tr)
        out["lgbo"][fam] = _round((_j(card_root / "lgbo" / f"{fam}-{card}-lgbo.json") or {}).get("trajectories", []))
        out["llambo"][fam] = _round((_j(card_root / "llambo" / f"{fam}-{card}-zero-shot.json") or {})
                                    .get("trajectories", []))
        d = _j(card_root / "bora" / f"{fam}-formula-{card}.json") or {}
        out["bora"][fam] = _round([r["trajectory"] for r in d.get("runs", [])])
    for arm in SYNTH_INJECTION_ARMS:
        if (card_root / arm).is_dir():
            out[arm] = {fam: _round([t for d in (1, 2, 3)
                                     for t in _traj(_j(card_root / arm / f"{fam}-draw{d}" / "trajectories.json"),
                                                    fam, arm)])
                        for fam in SYNTH_FAMS}
    return out


def collect_synthetic_none(none_root: Path) -> dict:
    """Card-independent arms: none/{vanilla,random,turbo}/trajectories.json."""
    out = {"vanilla": {}, "random": {}, "turbo": {}}
    for arm in out:
        d = _j(none_root / arm / "trajectories.json")
        for fam in SYNTH_FAMS:
            out[arm][fam] = _round(_traj(d, fam, arm))
    return out


# ------------------------------------------------------------------------------------------------ CNN leak control
CNN_ARMS = {  # arm -> (study, method)
    "hapbo": ("cnn-leak-of-campaign", "alpha-pi-raw-ei-matched"),
    "vanilla": ("cnn-leak-of-campaign", "vanilla-bo-ei"),
    "random": ("cnn-leak-of-campaign", "random-search"),
    "colabo": ("cnn-leak-of-colabo", "colabo-llm"),
    "llambo": ("cnn-leak-of-llambo-ws", "llambo-ws"),
    "lgbo": ("cnn-leak-of-lgbo", "lgbo"),
    "bora": ("cnn-leak-of-bora", "bora"),
}


def cnn_optimum() -> float:
    from datasets.cnn_leak import load_cnn_leak_problem
    return float(load_cnn_leak_problem(dataset="fashion-of").optimum_value)


def collect_cnn_arm(run_root: Path, method: str, optimum: float, budget: int = 30) -> dict:
    """Run format of results/paper/cnn_leak/<arm>.json:
    {"optimum": f*, "budget": 30, "runs": [[ground-truth accuracy per query], ...]} (completed runs only)."""
    s = collect_study([run_root], budgets={f"budget-{budget:04d}"})
    runs = [r["y"] for bud in s["methods"].get(method, {}).values() for rows in bud.values() for r in rows
            if r["y"] is not None]
    return {"optimum": optimum, "budget": budget, "runs": runs}


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="hpo,cortical,solar,synthetic,cnn")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--benchmark", type=Path, default=ROOT / "benchmark")
    a = ap.parse_args()
    import sys
    sys.path[:0] = [str(ROOT / "src")]
    only = set(a.only.split(","))
    studies = a.results / "studies"
    if "hpo" in only:
        for s in HPO_STUDIES:
            _dump(OUT / "hpo" / f"{s}.json", collect_study([studies / "hpo" / s / "runs"], budgets={"budget-0016"}))
        for s in MF_HPO_STUDIES:
            _dump(OUT / "hpo" / f"{s}.json", collect_mf_study([studies / "hpo" / s / "runs"], budgets={"budget-0016"}))
        _dump(OUT / "hpo" / "task_stats.json", hpo_task_stats(a.benchmark / "HPOBench"))
    if "cortical" in only:
        data = a.benchmark / "NeuroMOSAIC" / "dataset1" / "rat"
        units = set()
        for s in CORTICAL_STUDIES:
            c = collect_study([studies / "cortical" / s / "runs"])
            for bud in c["methods"].values():
                for b in bud.values():
                    units.update(b)
            _dump(OUT / "cortical" / f"{s}.json", c)
        _dump(OUT / "cortical" / "map_optimum.json", cortical_map_optimum(data, sorted(units)))
        for s in EXPLORATION_STUDIES:
            score_file = a.results / "exploration_scores" / "cortical" / f"{s}.json"
            _dump(OUT / "cortical" / "exploration" / f"{s}.json", json.loads(score_file.read_text(encoding="utf-8")))
        _dump(OUT / "cortical" / "prior_sources.json",
              prior_sources(ROOT / "experiments" / "priors" / "neuromosaic-dataset1-rat", data))
    if "solar" in only:
        _dump(OUT / "solar" / "maxeff-re.json", collect_study([studies / "solar" / "maxeff-re" / "runs"],
                                                              fields=("feasible", "objective"),
                                                              budgets={"budget-0050"}))
        diag = a.results / "diagnostics"
        _dump(OUT / "solar" / "external.json", solar_external(diag / "solar_external", diag / "solar_bora"))
    if "synthetic" in only:
        syn = a.results / "synthetic" / "well"
        for card in ("rho90", "well50", "well10"):
            _dump(OUT / "synthetic" / f"{card}.json", collect_synthetic(syn / card, card))
        _dump(OUT / "synthetic" / "none.json", collect_synthetic_none(syn / "none"))
    if "cnn" in only:
        opt = cnn_optimum()
        for arm, (study, method) in CNN_ARMS.items():
            root = studies / "cnn-leak" / study / "runs"
            if root.is_dir():
                _dump(OUT / "cnn_leak" / f"{arm}.json", collect_cnn_arm(root, method, opt))


if __name__ == "__main__":
    main()
