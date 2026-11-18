#!/usr/bin/env python3
"""Run continuous BORA on the synthetic panel.

BORA optimizes the native continuous box and the true analytic objective is
evaluated directly at every proposal.  There is no finite candidate pool and no
snapping.  The selected formula card is included in BORA's experiment
context, so every rho condition receives the same information used by the other
synthetic-panel methods.

The hidden objective is selected by SYNTH_TRUTH (scripts/synthetic_truth.py; default
``well``). Displayed values use 4 significant digits by default (BORA's default_precision),
which keeps BORA's prompts under the model's context limit on the 10-15 dimensional
functions. Upstream failures are recorded as failed seeds (not rerun).

Outputs one resumable JSON file per (card, family), with the trajectory layout of
scripts/run_synthetic_arms.py: ``{family: {"bora": [per-query-regret trajectories]}}``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bora_runtime import configure_qwen, install_token_dump, runtime_settings  # noqa: E402
from estimate_lib import FAMS  # noqa: E402

FAMILIES = tuple(FAMS)
RHOS = ("rho90", "well50", "well10")
BUDGETS = {"hartmann6": 40, "styblinski_tang7": 50, "levy10": 50, "ackley15": 150}


def card_text(family: str, rho: str) -> str:
    path = ROOT / "experiments" / "landscape_estimates" / family / f"card_formula_{rho}_domain.txt"
    return path.read_text(encoding="utf-8").strip()


def run_seed(family: str, rho: str, seed: int, budget: int, n_init: int, precision: int = 12) -> dict:
    dim, lo, hi, objective, f_opt, _ = FAMS[family]
    from bora.bora import Bora
    from bora.experiment import Experiment, Parameter, Target, Type

    parameters = []
    for j in range(dim):
        parameter = Parameter(
            name=f"x{j}", type=Type.continuous,
            description=f"continuous control parameter x{j} in [{lo:g}, {hi:g}]",
        )
        parameter.set_bounds(float(lo), float(hi))
        parameters.append(parameter)

    trajectory: list[float] = []
    evaluations: list[dict] = []
    best = float("inf")

    def evaluate(point):
        nonlocal best
        x = np.asarray(point, dtype=float).reshape(1, -1)
        value = float(objective(x)[0])
        best = min(best, value)
        regret = max(best - float(f_opt), 0.0)
        trajectory.append(regret)
        evaluations.append({
            "query": len(trajectory),
            "x": [float(v) for v in x[0]],
            "objective": value,
            "best_objective": best,
            "simple_regret": regret,
        })
        return np.asarray([-value], dtype=float)  # BORA maximizes; suite functions minimize.

    card = card_text(family, rho)
    experiment = Experiment(
        name="analytic black-box objective",
        description=(
            "Minimize an expensive black-box objective. The reported target value is the negative of the true "
            "objective's value; a larger target is better. The following approximate analytic "
            "model is prior scientific information; its parameters may be inaccurate.\n\n" + card
        ),
        parameters=parameters,
        target=Target(
            name="negative objective",
            description="negative true black-box objective (larger is better)",
        ),
        objective_function=evaluate,
        domain=card,
        default_precision=precision,
    )
    model, _, api_key = runtime_settings()
    bora = Bora(
        experiment=experiment,
        api_key=api_key,
        log_path=tempfile.mkdtemp(prefix=f"bora_{family}_{rho}_s{seed}_"),
        llm_model=model,
        verbose=False,
        random_seed=seed,
        allow_duplicate_points=False,
    )
    started = time.time()
    bora.maximize(n_init_points=n_init, n_iter=budget - n_init)
    if len(trajectory) != budget:
        raise RuntimeError(f"BORA evaluated {len(trajectory)} points, expected {budget}")
    return {
        "seed": seed,
        "trajectory": trajectory,
        "evaluations": evaluations,
        "final_regret": trajectory[-1],
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--rho", required=True, choices=RHOS)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--only-seed", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--precision", type=int, default=4,
                        help="BORA default_precision for displayed values")
    parser.add_argument("--out-root", default="results/estimate_panel/bora")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    budget = int(args.budget or BUDGETS[args.family])
    if not 1 <= args.n_init < budget:
        raise SystemExit("--n-init must be positive and less than the budget")
    card = card_text(args.family, args.rho)
    bora_root = Path(os.environ.get("BORA_ROOT", ROOT / "external" / "bora")).resolve()
    if not (bora_root / "bora" / "bora.py").is_file():
        raise SystemExit(f"pinned BORA checkout missing at {bora_root}")
    if args.validate_only:
        print(json.dumps({"family": args.family, "rho": args.rho, "budget": budget,
                          "n_init": args.n_init, "card_chars": len(card),
                          "bora_root": str(bora_root)}, indent=2))
        return

    sys.path.insert(0, str(bora_root))
    os.chdir(bora_root)  # upstream prompt templates are resolved relative to its checkout.
    model, base_url, api_key = runtime_settings()
    configure_qwen(llm_model=model, base_url=base_url, api_key=api_key)

    out_dir = (ROOT / args.out_root).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.family}-formula-{args.rho}.json"
    token_path = out_dir / f"{args.family}-formula-{args.rho}.tokens.json"
    install_token_dump(token_path)
    payload = {
        "artifact_format": "hapbo-bora-continuous-v1",
        "family": args.family,
        "rho": args.rho,
        "budget": budget,
        "n_init": args.n_init,
        "method": "bora",
        "runs": [],
        "failures": [],
    }
    if out_path.is_file():
        previous = json.loads(out_path.read_text(encoding="utf-8"))
        if (previous.get("family"), previous.get("rho"), previous.get("budget"),
                previous.get("n_init")) != (args.family, args.rho, budget, args.n_init):
            raise SystemExit(f"incompatible checkpoint at {out_path}")
        payload = previous

    completed = {int(run["seed"]) for run in payload.get("runs", [])}
    if args.only_seed is not None:
        seeds = [args.only_seed]
    else:
        seeds = list(range(args.seed_start, args.seeds))
    for seed in seeds:
        if seed in completed:
            print(f"skip completed seed {seed}", flush=True)
            continue
        print(f"== BORA continuous {args.family} {args.rho} b{budget} seed {seed} ==", flush=True)
        try:
            result = run_seed(args.family, args.rho, seed, budget, args.n_init, args.precision)
            payload.setdefault("runs", []).append(result)
            payload["runs"].sort(key=lambda item: int(item["seed"]))
            payload["failures"] = [f for f in payload.get("failures", [])
                                   if int(f["seed"]) != seed]
            print(f"seed {seed}: final regret {result['final_regret']:.6g}", flush=True)
        except Exception as exc:
            payload.setdefault("failures", []).append({
                "seed": seed, "error": repr(exc), "traceback": traceback.format_exc()
            })
            print(f"seed {seed} FAILED: {exc!r}", flush=True)
        trajectories = [run["trajectory"] for run in payload.get("runs", [])]
        payload["trajectories"] = {args.family: {"bora": trajectories}}
        payload["seeds_done"] = len(payload.get("runs", []))
        payload["seeds_target"] = len(seeds)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"BORA_CONTINUOUS_DONE {out_path} {payload.get('seeds_done', 0)}/{len(seeds)}", flush=True)


if __name__ == "__main__":
    main()

