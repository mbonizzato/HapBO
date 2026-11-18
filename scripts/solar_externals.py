#!/usr/bin/env python3
"""SOLAR external baselines (ColaBO, LLAMBO, LGBO; Linux-only C++ SOLAR binary).

The external baselines run on a frozen SOLAR candidate pool outside the study harness. They have
no constraint support and see a graded death penalty, the standard fallback for constraint-blind
baselines: feasible -> objective; real-but-infeasible -> 1e5 + min(max_violation, 1e4);
crash -> 1e6.

The internal arms (solar/maxeff-re study) see the raw objective and the constraint vector
(per-constraint GPs, constrained EI).

Usage (hapbo env; LLAMBO and LGBO need the LLM endpoint):
  python scripts/solar_externals.py --baseline colabo --seed 2001 \
      --cs-json experiments/colabo_llm_priors/solar-maxeff-re/maxeff-re.json
  python scripts/solar_externals.py --baseline llambo --task-legend --seed 2001
  python scripts/solar_externals.py --baseline lgbo --task-legend --seed 2001

scripts/run_solar_baselines.py runs the paper seeds (with --prescreen). Writes
results/diagnostics/solar_external/<stem>_penalized_<instance>_s<seed>[_prescreen].{jsonl,summary.json}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # baselines/* import as `src.<module>`

from datasets.solar import (  # noqa: E402
    SOLAR_EXECUTABLE,
    evaluate_solar,
    generate_solar_candidate_pool,
    resolve_solar_instance,
)

OUT = ROOT / "results" / "diagnostics" / "solar_external"
CRASH_VALUE = 1e6
INFEASIBLE_BASE = 1e5


def encode(candidates, spec) -> np.ndarray:
    cols = []
    for v in spec.variables:
        vals = np.asarray([c[v.name] for c in candidates], dtype=float)
        cols.append((vals - v.lower) / (v.sampling_upper - v.lower))
    return np.column_stack(cols)


def penalize(ev) -> float:
    if ev.hidden_failure:
        return CRASH_VALUE
    if ev.feasible:
        return float(ev.objective)
    max_violation = float(np.max(np.maximum(ev.constraints, 0.0)))
    return INFEASIBLE_BASE + min(max_violation, 1e4)


def run(args) -> None:
    from baselines import get_baseline
    from baselines.base import config_key
    from baselines.bridge import require_full_budget
    from baselines.runner import run_ask_tell_loop

    spec = resolve_solar_instance(args.instance)
    pool = generate_solar_candidate_pool(args.instance, args.pool_size,
                                         seed=args.seed, prescreen=args.prescreen)
    encoded = encode(pool, spec)
    key_to_index = {config_key(c): i for i, c in enumerate(pool)}
    search_space = {"candidates": [dict(c) for c in pool], "encoded_candidates": encoded}

    objective_meta = solar_objective_meta(spec, legend=args.task_legend)

    prior = None
    zero_shot = args.baseline == "llambo" and args.llambo_mode == "zero-shot"
    if zero_shot:  # LLAMBO Mode 2: 5 zero-shot initial points; context = the SOLAR card from '## 1. Domain'
        # on (domain, search space with meanings/scales, optimization description), the same recipe as the HPO Mode-2 runs
        card = (ROOT / "experiments/priors/solar-maxeff-re/expert/prompt_card.md").read_text(encoding="utf-8")
        objective_meta["warmstart_context"] = card[card.index("## 1. Domain"):].strip()
    n_init = 5 if zero_shot else 3
    if args.baseline == "colabo":
        cs = json.loads(Path(args.cs_json).read_text(encoding="utf-8"))
        c = cs["c"] if isinstance(cs["c"], dict) else {
            v.name: cs["c"][i] for i, v in enumerate(spec.variables)}
        sigma = float(cs.get("sigma", 1.0 - float(cs["s"]) + 1e-6))
        c_unit = np.array([(float(c[v.name]) - v.lower) / (v.sampling_upper - v.lower)
                           for v in spec.variables])
        d2 = np.sum(((encoded - c_unit[None, :]) / sigma) ** 2, axis=1)
        values = np.exp(-0.5 * d2)
        prior = {"values": values, "confidence": sigma}

    stem = f"{'llambo-ws' if zero_shot else args.baseline}_penalized_{args.instance}_s{args.seed}" + (
        "_prescreen" if args.prescreen else "")
    # Route LGBO token+latency telemetry to a file next to this run's output.
    if args.baseline == "lgbo":
        OUT.mkdir(parents=True, exist_ok=True)
        os.environ["LGBO_TELEMETRY"] = str(OUT / f"{stem}.llm_telemetry.json")

    optimizer = get_baseline(
        args.baseline, search_space=search_space, objective_meta=objective_meta,
        prior=prior, seed=args.seed, budget=args.budget, n_initial=n_init,
        **({"llambo_init_mode": "zero-shot"} if zero_shot else {}),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    bridged = callable(getattr(optimizer, "bridge_accounting", None))
    if bridged:
        optimizer.worker_log_path = str(OUT / f"{stem}.worker_stderr.log")
    records = []
    simulated: dict[int, dict] = {}  # candidate index -> simulator record (repeated proposals reuse it)

    def objective(config):
        ev = evaluate_solar(args.instance, dict(config), seed=args.seed)
        index = key_to_index[config_key(config)]
        simulated[index] = {
            "candidate_index": index,
            "objective": ev.objective,
            "constraints": ev.constraints.tolist(),
            "feasible": ev.feasible, "hidden_failure": ev.hidden_failure,
            "max_violation": float(np.max(np.maximum(ev.constraints, 0.0)))
            if not ev.hidden_failure else None,
            "fed_value": penalize(ev),
        }
        return simulated[index]["fed_value"]

    def on_evaluation(count, config, observation):
        # Every proposal is a charged query; a repeated proposal reuses the simulator record.
        repeat = bool(optimizer.trace[count - 1]["repeat"]) if bridged else False
        records.append({"query": count, **simulated[key_to_index[config_key(config)]], "repeat": repeat})

    t0 = time.time()
    try:
        run_ask_tell_loop(optimizer, objective, budget=args.budget, n_init=n_init,
                          on_evaluation=on_evaluation)
    finally:
        close = getattr(optimizer, "close", None)
        if close:
            close()
    if not records:  # same as the study worker
        raise RuntimeError(f"{args.baseline} worker stopped before proposing any query")
    require_full_budget(len(records), args.budget, args.baseline)  # a short run fails (no padding)
    feas = [r for r in records if r["feasible"]]
    summary = {
        "baseline": args.baseline, "variant": "penalized", "instance": args.instance,
        "llambo_mode": args.llambo_mode if args.baseline == "llambo" else None, "n_init": n_init,
        "seed": args.seed, "budget": args.budget, "n_evals": len(records),
        "crash_count": sum(r["hidden_failure"] for r in records),
        "feasible_count": len(feas),
        "first_feasible_query": next((r["query"] for r in records if r["feasible"]), None),
        "best_feasible_objective": min((r["objective"] for r in feas), default=None),
        "best_raw_objective_nonhf": min(
            (r["objective"] for r in records if not r["hidden_failure"]), default=None),
        "elapsed_seconds": time.time() - t0,
        "bridge_accounting": optimizer.bridge_accounting() if bridged else None,
        "coordinate_legend": objective_meta.get("coordinate_legend"),
    }
    (OUT / f"{stem}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    (OUT / f"{stem}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def solar_objective_meta(spec, *, legend: bool = False) -> dict:
    """LLM-visible text of the SOLAR external baselines."""
    objective_meta = {
        "maximize": False, "n_constraints": 0,
        "model": "central-receiver solar thermal plant simulator",
        "task": "regression",
        "metric": "negative receiver efficiency (lower is better)",
        "description": solar_task_text(negated=True),
        "task_description": solar_task_text(negated=True),
        "prompt_setting": "partial_context",
    }
    if legend:
        # The x-space baselines (in-loop LLAMBO, LGBO) see x0..x6; give them the coordinate legend
        # from the same SOLAR card (names, meanings) and the bounds encode() uses.
        legend_text = solar_coordinate_legend(spec)
        objective_meta["description"] += "\n\n" + legend_text
        objective_meta["task_description"] += "\n\n" + legend_text
        objective_meta["coordinate_legend"] = legend_text
    return objective_meta


def solar_coordinate_legend(spec) -> str:
    """x0..x{d-1} legend for the SOLAR baselines (names/meanings from the SOLAR card, linear
    mapping between the bounds used by :func:`encode`)."""
    from experiments.worker import _box_legend, _parse_card_dimensions

    card = (ROOT / "experiments/priors/solar-maxeff-re/expert/prompt_card.md").read_text(encoding="utf-8")
    bounds = [(v.name, float(v.lower), float(v.sampling_upper)) for v in spec.variables]
    return _box_legend(bounds, _parse_card_dimensions(card))


def solar_task_text(negated: bool = False) -> str:
    """Domain + Optimization-description sections of the SOLAR card (facts only: no PSL
    instructions, no question block), shared with BORA."""
    txt = (ROOT / "experiments/priors/solar-maxeff-re/expert/prompt_card.md").read_text(encoding="utf-8")
    domain = txt.split("## 1. Domain")[1].split("## 2. Search space")[0].strip()
    opt = txt.split("## 3. Optimization description")[1].split("## 4.")[0].strip()
    if negated:  # the values this baseline receives are the negative efficiency, to be minimized
        opt = opt.replace("Direction: maximize" + chr(10), "")
        opt = opt.replace("; higher is better.",
                          "; higher efficiency is better. The values reported to you are the NEGATIVE "
                          "efficiency, so lower reported values are better. Designs that violate a constraint are "
                          "reported as large positive values (1e5 plus the violation) and simulator crashes as 1e6.")
    return domain + "\n\n" + opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="MAXEFF_RE")
    ap.add_argument("--pool-size", type=int, default=256)
    ap.add_argument("--baseline", choices=["colabo", "llambo", "lgbo"], required=True)
    ap.add_argument("--seed", type=int, default=2001)
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--cs-json", default=None)
    ap.add_argument("--task-legend", action="store_true",
                    help="append the x0..x6 coordinate legend from the SOLAR card (in-loop LLAMBO, LGBO)")
    ap.add_argument("--out-dir", default=None,
                    help="output folder relative to the repo (default results/diagnostics/solar_external)")
    ap.add_argument("--llambo-mode", choices=["zero-shot", "random"], default="zero-shot",
                    help="LLAMBO initialization: zero-shot warm start (Mode 2, 5 points; the paper's runs) "
                         "or random (Mode 1, 3 points)")
    ap.add_argument("--prescreen", action="store_true",
                    help="filter the pool on the simulator's documented a-priori input "
                         "relations (Python-evaluated; the simulator is never queried)")
    args = ap.parse_args()
    if args.out_dir:
        global OUT
        OUT = ROOT / args.out_dir
    if not Path(SOLAR_EXECUTABLE).is_file():
        raise SystemExit(f"SOLAR binary missing at {SOLAR_EXECUTABLE} (Linux-only C++ binary required)")
    if args.baseline == "colabo" and not args.cs_json:
        raise SystemExit("colabo run needs --cs-json")
    run(args)


if __name__ == "__main__":
    main()
