#!/usr/bin/env python3
"""Write the study manifests of the paper campaigns: experiments/studies/{hpo,cortical,solar}/<name>/study.yaml.

HPOBench: 40 tasks (5 model families x 8 datasets), budget 16, seeds 1001-1010.
Cortical (NeuroMOSAIC dataset 1, rat): the 16 flexor maps (prior sources) or all 40 rat-muscle maps (baseline
comparison, BO variants), budget 12, seeds 1001-1010.
SOLAR (MAXEFF_RE): budget 50, seeds 2001-2020.

Studies with ``misleading_description: true`` give the baselines that read task text the misleading
domain hint of experiments/misleading_descriptions/; prior-consuming arms of the misleading studies read
priors elicited from the card plus that hint. Run a study with
``python scripts/run_study_parallel.py experiments/studies/<group>/<name>``.

usage: python scripts/make_studies.py [--check]   (--check: compare with the files on disk, write nothing)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "experiments" / "studies"
FAMS = ["lr", "rf", "xgb", "nn", "svm"]
TASKS = ["australian", "blood_transfusion", "car", "credit_g", "kc1", "phoneme", "segment", "vehicle"]
SEEDS = "[1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]"
SEEDS_SOLAR = "[" + ", ".join(str(s) for s in range(2001, 2021)) + "]"

INTERNAL = "{n_initial: 1, length_scale: 0.2, noise_std: 0.001, improvement_xi: 0.0}"
EXTERNAL = "n_initial: 1, length_scale: 1.0, noise_std: 0.05, improvement_xi: 0.0"
LGBO = "n_initial: 2, length_scale: 1.0, noise_std: 0.05, improvement_xi: 0.0"
LLAMBO_WS = "n_initial: 5, length_scale: 1.0, noise_std: 0.05, improvement_xi: 0.0, llambo_init_mode: zero-shot"
COLABO = EXTERNAL + ", colabo_max_internal_trials: 96"
MIS = ", misleading_description: true"
LEG = ", baseline_task_legend: true"  # coordinate legend for the x-space baselines
CORTICAL = "n_initial: 1, length_scale: 1.0, noise_std: 0.05, improvement_xi: 0.0"
# GP kernels of the kernel studies (design.optimizer.kernel; every other study uses the default Matern-5/2)
KERNELS = [("rbf", "RBF (squared-exponential)"), ("matern12", "Matern-1/2 (exponential)")]


def hpo_prior(prior_id: str, filename: str = "prior.json"):
    def fn(fam: str, task: str) -> str:
        slug = task.replace("_", "-")
        return f"experiments/priors/hpobench-{fam}-{slug}/{slug}/{prior_id}/{filename}"
    return fn


def colabo_belief(kind: str):
    return lambda fam, task: f"experiments/colabo_llm_priors/hpobench-tabular/{fam}-{task}/{kind}.json"


def hypbo_hypotheses(kind: str):
    return lambda fam, task: f"experiments/hypbo_llm_hypotheses/hpobench-tabular/{fam}-{task}/{kind}.json"


def no_prior(fam: str, task: str):
    return None


def hpo_benchmarks(path_fn, fams=FAMS) -> str:
    lines = []
    for fam in fams:
        for task in TASKS:
            lines.append("    - benchmark_id: hpobench-tabular")
            lines.append(f"      include: [{fam}-{task}]")
            p = path_fn(fam, task)
            if p:
                lines.append(f"      prior_path: {p}")
    return "\n".join(lines)


def manifest(group: str, name: str, question: str, budgets: str, seeds: str, methods: list[str], optimizer: str,
             benchmarks: str) -> tuple[Path, str]:
    text = (
        "artifact_format: llm-bo-study-v1\n"
        f"study_id: {name}\n"
        f"phase: {group}\n"
        "status: ready\n"
        "execution_mode: optimization\n"
        f'research_question: "{question}"\n'
        "design:\n"
        f"  budgets: {budgets}\n"
        f"  seeds: {seeds}\n"
        f"  methods: [{', '.join(methods)}]\n"
        f"  optimizer: {optimizer}\n"
        "  benchmarks:\n"
        f"{benchmarks}\n"
    )
    return STUDIES / group / name / "study.yaml", text


def hpo_studies():
    H = []
    four = FAMS[:4]

    def add(name, question, methods, optimizer, path_fn, fams=FAMS):
        H.append(manifest("hpo", name, question, "[16]", SEEDS, methods, optimizer, hpo_benchmarks(path_fn, fams)))

    add("hapbo", "HapBO (alpha-pi, EI) with the prior elicited from the task card.",
        ["alpha-pi-raw-ei-matched"], INTERNAL, hpo_prior("llm-card"))
    add("hapbo-misleading", "HapBO with the prior elicited from the task card plus the misleading hint.",
        ["alpha-pi-raw-ei-matched"], INTERNAL, hpo_prior("llm-card-misleading"))
    add("hapbo-inverted", "HapBO with the inverted-map prior (adversarial prior).",
        ["alpha-pi-raw-ei-matched"], INTERNAL, hpo_prior("inverted"))
    # Fixed-alpha sweep: HapBO with alpha fixed instead of compiled; the compiled-alpha arm is hapbo /
    # hapbo-misleading and alpha = 0 is vanilla-bo-ei of vanilla-random.
    sweep = [f"alpha-pi-raw-ei-matched-alpha-0-{a}" for a in (9, 7, 5, 3, 1)]
    add("alpha-sweep", "HapBO with the trust weight alpha fixed at 0.9, 0.7, 0.5, 0.3 or 0.1 instead of the "
        "compiled alpha, card prior.", sweep, INTERNAL, hpo_prior("llm-card"))
    add("alpha-sweep-misleading", "HapBO with alpha fixed at 0.9, 0.7, 0.5, 0.3 or 0.1, prior elicited from the card "
        "plus the misleading hint.", sweep, INTERNAL, hpo_prior("llm-card-misleading"))
    # prior_path gives vanilla BO the task's GP input encoding schema; its beliefs are not used.
    add("vanilla-random", "Prior-free references: vanilla BO (EI) and random search.",
        ["vanilla-bo-ei", "random-search"], INTERNAL, hpo_prior("base"))
    add("lgbo", "LGBO (paper equation) with the task text.", ["lgbo"], "{" + LGBO + LEG + "}", no_prior)
    add("lgbo-misleading", "LGBO (paper equation) with the task text plus the misleading hint.",
        ["lgbo"], "{" + LGBO + MIS + LEG + "}", no_prior)
    add("lgbo-official", "LGBO with the lift operator of the authors' official implementation, task text.",
        ["official-lgbo"], "{" + LGBO + LEG + "}", no_prior, four)
    add("lgbo-official-misleading", "LGBO with the lift operator of the authors' official implementation, task text "
        "plus the misleading hint.", ["official-lgbo"], "{" + LGBO + MIS + LEG + "}", no_prior, four)
    add("colabo", "ColaBO with the LLM (c, s) belief elicited from the task card.",
        ["colabo-llm"], "{" + COLABO + "}", colabo_belief("llm-card"))
    add("colabo-misleading", "ColaBO with the LLM (c, s) belief elicited from the task card plus the misleading hint.",
        ["colabo-llm"], "{" + COLABO + "}", colabo_belief("llm-card-misleading"))
    add("hypbo", "HypBO with the LLM box hypotheses elicited from the task card.",
        ["hypbo-llm"], "{" + EXTERNAL + "}", hypbo_hypotheses("llm-card"))
    add("hypbo-misleading", "HypBO with the LLM box hypotheses elicited from the task card plus the misleading hint.",
        ["hypbo-llm"], "{" + EXTERNAL + "}", hypbo_hypotheses("llm-card-misleading"))
    # LLM-EI: the vanilla-bo-ei arm gated by the utility fitted to the LLM's pairwise survey of the card.
    add("llm-ei", "LLM-EI with the utility fitted to the LLM's pairwise survey of the task card.",
        ["llm-ei"], INTERNAL, hpo_prior("llm-ei-card", "llm_ei_utility.json.gz"))
    add("llm-ei-misleading", "LLM-EI with the utility fitted to the LLM's pairwise survey of the card plus the "
        "misleading hint.", ["llm-ei"], INTERNAL, hpo_prior("llm-ei-card-misleading", "llm_ei_utility.json.gz"))
    add("bora", "BORA with the task text.", ["bora"], "{" + EXTERNAL + LEG + "}", no_prior)
    add("bora-misleading", "BORA with the task text plus the misleading hint.",
        ["bora"], "{" + EXTERNAL + MIS + LEG + "}", no_prior)
    add("llambo", "LLAMBO Mode 2 (zero-shot warm start from the task card).",
        ["llambo-ws"], "{" + LLAMBO_WS + LEG + "}", hpo_prior("llm-card"))
    add("llambo-misleading", "LLAMBO Mode 2 with the task card plus the misleading hint.",
        ["llambo-ws"], "{" + LLAMBO_WS + MIS + LEG + "}", hpo_prior("llm-card"))
    add("llambo-mode1", "LLAMBO Mode 1 (random initial design) with the task text.",
        ["llambo"], "{" + EXTERNAL + LEG + "}", no_prior, four)
    add("llambo-mode1-misleading", "LLAMBO Mode 1 with the task text plus the misleading hint.",
        ["llambo"], "{" + EXTERNAL + MIS + LEG + "}", no_prior, four)
    inj = ["pi-bo-ei", "prior-to-mean", "prior-warm-start", "prior-greedy"]
    add("injection", "Prior-injection sites (pi-BO, prior mean, warm start, prior-greedy), card prior.",
        inj, INTERNAL, hpo_prior("llm-card"))
    add("injection-misleading", "Prior-injection sites, prior elicited from the card plus the misleading hint.",
        inj, INTERNAL, hpo_prior("llm-card-misleading"))
    for v, label in (("pointonly", "point-only"), ("gaussian", "Gaussian-only"), ("directllm", "direct-LLM")):
        add(f"representation-{v}", f"HapBO with the {label} representation of the card prior.",
            ["alpha-pi-raw-ei-matched"], INTERNAL, hpo_prior(f"llm-card-{v}"))
    for v, label in (("pointonly", "point-only"), ("gaussian", "Gaussian-only"), ("directllm", "direct-LLM")):
        add(f"representation-{v}-misleading", f"HapBO with the {label} representation of the prior elicited from "
            "the card plus the misleading hint.", ["alpha-pi-raw-ei-matched"], INTERNAL,
            hpo_prior(f"llm-card-misleading-{v}"))
    add("acquisition", "HapBO and vanilla BO under EI, PI and UCB, card prior.",
        ["alpha-pi-raw-ei-matched", "vanilla-bo-ei", "alpha-pi-raw-pi-matched", "vanilla-bo-pi",
         "alpha-pi-raw-ucb-matched", "vanilla-bo-ucb"], INTERNAL, hpo_prior("llm-card"))
    add("batch", "HapBO and vanilla BO with batches of 3 queries (Kriging believer), card prior.",
        ["alpha-pi-raw-ei-matched", "vanilla-bo-ei"],
        "{n_initial: 1, length_scale: 0.2, noise_std: 0.001, improvement_xi: 0.0, q_batch: 3}",
        hpo_prior("llm-card"))
    for kernel, label in KERNELS:
        add(f"kernel-{kernel}", f"HapBO and vanilla BO (EI) with the {label} GP kernel, card prior.",
            ["alpha-pi-raw-ei-matched", "vanilla-bo-ei"], INTERNAL[:-1] + f", kernel: {kernel}}}",
            hpo_prior("llm-card"))
    add("cake", "CAKE (LLM-generated kernels), prior-free.", ["cake"], "{n_initial: 1}", no_prior)
    add("caliper", "CALIPER with the multi-role prior elicited from the task card.",
        ["caliper"], INTERNAL, hpo_prior("caliper-card", "caliper_prior.json.gz"))
    add("caliper-misleading", "CALIPER with the multi-role prior elicited from the card plus the misleading hint.",
        ["caliper"], INTERNAL, hpo_prior("caliper-card-misleading", "caliper_prior.json.gz"))
    # PriorBand / PriMO (NePS, multi-fidelity): budget 16 = full-fidelity cost units (each evaluation costs z/z_max).
    for algo, label, kwargs in (("priorband", "PriorBand", "{sample_prior_first: highest_fidelity}"),
                                ("primo", "PriMO", "{initial_design_size: 5}")):
        neps = "{torch_threads: 1, neps_kwargs: " + kwargs + "}"
        add(algo, f"{label} (NePS, multi-fidelity, cost in full-fidelity units) with the point prior elicited "
            "from the task card.", [algo], neps, hpo_prior("neps-point-card", "neps_prior.json"))
        add(f"{algo}-misleading", f"{label} (NePS, multi-fidelity, cost in full-fidelity units) with the point prior "
            "elicited from the card plus the misleading hint.", [algo], neps,
            hpo_prior("neps-point-card-misleading", "neps_prior.json"))
    return H


def cortical_studies():
    C = []

    def bench(prior: str | None, colabo: bool = False, benchmark: str = "neuromosaic-dataset1-rat-flexor") -> str:
        lines = [f"    - benchmark_id: {benchmark}", "      include: all"]
        if prior and colabo:
            lines.append(f"      prior_path: experiments/colabo_llm_priors/neuromosaic-dataset1-rat/{{muscle_slug}}/{prior}.json")
        elif prior:
            lines.append(f"      prior_path: experiments/priors/neuromosaic-dataset1-rat/{{muscle_slug}}/{prior}/prior.json")
        return "\n".join(lines)

    def add(name, question, methods, optimizer, prior, colabo=False, benchmark="neuromosaic-dataset1-rat-flexor"):
        C.append(manifest("cortical", name, question, "[12]", SEEDS, methods, "{" + optimizer + "}",
                          bench(prior, colabo, benchmark)))

    ALL = "neuromosaic-dataset1-rat-all"
    add("human-map", "HapBO, vanilla BO and random search with the human expert map.",
        ["alpha-pi-raw-ei-matched", "vanilla-bo-ei", "random-search"], CORTICAL, "human-expert-map")
    add("llm-geometry", "HapBO with the prior elicited from the array-geometry facts only.",
        ["alpha-pi-raw-ei-matched"], CORTICAL, "llm-geometry")
    add("llm-expert-hint", "HapBO with the prior elicited from the geometry facts plus the expert hint.",
        ["alpha-pi-raw-ei-matched"], CORTICAL, "llm-expert-hint")
    add("human-map-all", "HapBO, vanilla BO and random search with the human expert map, all 40 rat-muscle maps.",
        ["alpha-pi-raw-ei-matched", "vanilla-bo-ei", "random-search"], CORTICAL, "human-expert-map", benchmark=ALL)
    add("human-map-colabo", "ColaBO with the LLM (c, s) belief from the shared human-map text.",
        ["colabo-llm"], CORTICAL + ", colabo_max_internal_trials: 96, cortical_text: human",
        "human-expert-map-shared", colabo=True, benchmark=ALL)
    add("human-map-llambo", "LLAMBO Mode 2 with the shared human-map text as warm-start context.",
        ["llambo-ws"], "n_initial: 5, length_scale: 1.0, noise_std: 0.05, improvement_xi: 0.0, "
        "llambo_init_mode: zero-shot, cortical_text: human", "human-expert-map-shared", benchmark=ALL)
    add("human-map-lgbo", "LGBO (paper equation) with the shared human-map text in the loop.",
        ["lgbo-prior"], "n_initial: 2, length_scale: 1.0, noise_std: 0.05, improvement_xi: 0.0, cortical_text: human",
        "human-expert-map-shared", benchmark=ALL)
    add("human-map-bora", "BORA with the shared human-map text.",
        ["bora"], CORTICAL + ", cortical_text: human", None, benchmark=ALL)
    add("acquisition", "HapBO and vanilla BO under EI, PI and UCB, prior from the geometry facts plus the expert hint.",
        ["alpha-pi-raw-ei-matched", "vanilla-bo-ei", "alpha-pi-raw-pi-matched", "vanilla-bo-pi",
         "alpha-pi-raw-ucb-matched", "vanilla-bo-ucb"], CORTICAL, "llm-expert-hint", benchmark=ALL)
    add("batch", "HapBO and vanilla BO with batches of 3 queries (Kriging believer), prior from the geometry facts "
        "plus the expert hint.", ["alpha-pi-raw-ei-matched", "vanilla-bo-ei"], CORTICAL + ", q_batch: 3",
        "llm-expert-hint", benchmark=ALL)
    for kernel, label in KERNELS:
        add(f"kernel-{kernel}", f"HapBO and vanilla BO (EI) with the {label} GP kernel, prior from the geometry facts "
            "plus the expert hint.", ["alpha-pi-raw-ei-matched", "vanilla-bo-ei"], CORTICAL + f", kernel: {kernel}",
            "llm-expert-hint", benchmark=ALL)
    return C


def solar_studies():
    bench = ("    - benchmark_id: solar-single-objective\n"
             "      include: [MAXEFF_RE]\n"
             "      prior_path: experiments/priors/solar-maxeff-re/expert/prior.json")
    return [manifest("solar", "maxeff-re", "HapBO, vanilla BO and random search on SOLAR MAXEFF_RE with the SOLAR card prior.",
                     "[50]", SEEDS_SOLAR, ["alpha-pi-raw-ei-matched", "vanilla-bo-ei", "random-search"],
                     "{n_initial: 3, length_scale: 1.0, noise_std: 0.05, improvement_xi: 0.0}", bench)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare with the files on disk; write nothing")
    args = ap.parse_args()
    bad = 0
    for path, text in hpo_studies() + cortical_studies() + solar_studies():
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                print("differs:", path.relative_to(ROOT))
                bad += 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        print("wrote", path.relative_to(ROOT).as_posix())
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
