#!/usr/bin/env python3
"""Run BORA on simulator-frozen, prescreened SOLAR candidate pools.

The optimizer runs in the isolated bora env.  Each continuous proposal is snapped in normalized
space to the corresponding seed's 256-point pool, exactly like the finite BORA
adapter (src/baselines/workers/bora_worker.py).  BORA's native constrained acquisition receives all six
constraint values only after an objective evaluation; assistant-side candidate
validation is deliberately permissive for unseen points, preventing an uncharged
simulator-constraint oracle.
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
sys.path.insert(0, str(ROOT / "src"))

from bora_runtime import configure_qwen, install_token_dump, runtime_settings  # noqa: E402
from datasets.solar import resolve_solar_instance  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default="MAXEFF_RE")
    parser.add_argument("--seed-start", type=int, default=2001)
    parser.add_argument("--seed-end", type=int, default=2020)
    parser.add_argument("--only-seed", type=int)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--cache-root", default="results/diagnostics/solar_pool_cache")
    parser.add_argument("--out-root", default="results/diagnostics/solar_bora")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    bora_root = Path(os.environ.get("BORA_ROOT", ROOT / "external" / "bora")).resolve()
    if not (bora_root / "bora" / "bora.py").is_file():
        raise SystemExit(f"pinned BORA checkout missing at {bora_root}")
    seeds = [args.only_seed] if args.only_seed is not None else list(
        range(args.seed_start, args.seed_end + 1))
    cache_dir = (ROOT / args.cache_root).resolve()
    missing = [seed for seed in seeds
               if not (cache_dir / f"{args.instance}_s{seed}_prescreen.json").is_file()]
    if missing:
        raise SystemExit(f"missing frozen SOLAR pools for seeds: {missing}")
    if args.validate_only:
        print(json.dumps({"instance": args.instance, "seeds": seeds,
                          "budget": args.budget, "n_init": args.n_init,
                          "cache_root": str(cache_dir), "bora_root": str(bora_root)}, indent=2))
        return

    sys.path.insert(0, str(bora_root))
    os.chdir(bora_root)
    model, base_url, api_key = runtime_settings()
    configure_qwen(llm_model=model, base_url=base_url, api_key=api_key)
    out_dir = (ROOT / args.out_root).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    install_token_dump(out_dir / f"bora_{args.instance}.tokens.json")
    # SOLAR card facts (Domain + Optimization description), the same text the other SOLAR baselines get
    # (read directly: this script runs in BORA's own env, without the harness imports)
    _txt = (ROOT / "experiments/priors/solar-maxeff-re/expert/prompt_card.md").read_text(encoding="utf-8")
    card = (_txt.split("## 1. Domain")[1].split("## 2. Search space")[0].strip() + "\n\n"
            + _txt.split("## 3. Optimization description")[1].split("## 4.")[0].strip())
    spec = resolve_solar_instance(args.instance)

    from bora.bora import Bora
    from bora.constraint import ConstraintModel
    from bora.experiment import Constraint, Experiment, Parameter, Target, Type

    failed_seeds = []
    for seed in seeds:
        out_path = out_dir / f"bora_{args.instance}_s{seed}_prescreen.json"
        failure = out_dir / f"bora_{args.instance}_s{seed}_prescreen.failed.json"
        if out_path.is_file():
            print(f"skip completed seed {seed}", flush=True)
            continue
        try:
            frozen = json.loads(
                (cache_dir / f"{args.instance}_s{seed}_prescreen.json").read_text(encoding="utf-8"))
            rows = frozen["rows"]
            encoded = np.asarray([row["encoded"] for row in rows], dtype=float)
            parameters = []
            for variable in spec.variables:
                parameter = Parameter(
                    name=variable.name,
                    type=Type.continuous,
                    description=f"{variable.name}, design variable",
                )
                parameter.set_bounds(float(variable.lower), float(variable.sampling_upper))
                parameters.append(parameter)

            records: list[dict] = []
            observed_constraints: dict[tuple[float, ...], np.ndarray] = {}

            def proposal_key(values) -> tuple[float, ...]:
                return tuple(round(float(v), 12) for v in values)

            def snap(values) -> int:
                unit = np.asarray([
                    (float(values[j]) - variable.lower)
                    / (variable.sampling_upper - variable.lower)
                    for j, variable in enumerate(spec.variables)
                ])
                return int(np.argmin(np.sum((encoded - unit[None, :]) ** 2, axis=1)))

            def objective(values):
                values = np.asarray(values, dtype=float).reshape(-1)
                index = snap(values)
                row = rows[index]
                constraints = np.asarray(row["constraints"], dtype=float)
                if row["hidden_failure"]:
                    constraints = np.full(spec.n_constraints, 1e6, dtype=float)
                    target = -1e6
                else:
                    target = -float(row["objective"])
                observed_constraints[proposal_key(values)] = constraints
                # Every proposal is charged; a repeat of an already-evaluated design is
                # flagged (bookkeeping only).
                repeat = any(r["candidate_index"] == index for r in records)
                records.append({
                    "query": len(records) + 1,
                    "proposed": [float(v) for v in values],
                    "candidate_index": index,
                    "candidate": row["candidate"],
                    "objective": float(row["objective"]),
                    "constraints": [float(v) for v in row["constraints"]],
                    "feasible": bool(row["feasible"]),
                    "hidden_failure": bool(row["hidden_failure"]),
                    "fed_target": target,
                    "repeat": repeat,
                })
                return np.asarray([target], dtype=float)

            def constraints(**kwargs):
                values = [kwargs[variable.name] for variable in spec.variables]
                # TargetSpace calls the objective immediately before this callback.
                # Unseen calls originate from the LLM assistant's pre-validation and
                # must not reveal simulator-only constraints for free.
                return observed_constraints.get(
                    proposal_key(values), np.full(spec.n_constraints, -1.0, dtype=float))

            constraint_model = ConstraintModel(
                fun=constraints,
                lb=np.full(spec.n_constraints, -1e12, dtype=float),
                ub=np.zeros(spec.n_constraints, dtype=float),
                random_state=seed,
            )
            experiment = Experiment(
                name="central-receiver design problem",
                description=card,
                parameters=parameters,
                target=Target(
                    name="receiver efficiency",
                    description="receiver efficiency; larger is better, subject to six constraints",
                ),
                objective_function=objective,
                domain=card,
                constraint=Constraint(
                    description="the six simulation constraints c1..c6 must each be <= 0",
                    func=constraint_model,
                ),
                default_precision=9,
            )
            bora = Bora(
                experiment=experiment,
                api_key=api_key,
                log_path=tempfile.mkdtemp(prefix=f"bora_solar_{seed}_"),
                llm_model=model,
                verbose=False,
                random_seed=seed,
                allow_duplicate_points=False,
            )
            started = time.time()
            bora.maximize(n_init_points=args.n_init, n_iter=args.budget - args.n_init)
            if len(records) != args.budget:
                raise RuntimeError(f"BORA evaluated {len(records)} points, expected {args.budget}")
            feasible = [record for record in records if record["feasible"]]
            payload = {
                "artifact_format": "hapbo-bora-solar-v1",
                "baseline": "bora",
                "instance": args.instance,
                "seed": seed,
                "budget": args.budget,
                "n_init": args.n_init,
                "candidate_pool_prescreen": True,
                "pool_source": str(cache_dir / f"{args.instance}_s{seed}_prescreen.json"),
                "constraint_handling": "BORA native constrained acquisition; no assistant oracle",
                "n_evals": len(records),
                "feasible_count": len(feasible),
                "hidden_failure_count": sum(record["hidden_failure"] for record in records),
                "first_feasible_query": next(
                    (record["query"] for record in records if record["feasible"]), None),
                "best_feasible_objective": min(
                    (record["objective"] for record in feasible), default=None),
                "elapsed_seconds": time.time() - started,
                "repeat_count": sum(record["repeat"] for record in records),
                "records": records,
            }
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            failure.unlink(missing_ok=True)
            print(f"seed {seed}: feasible={len(feasible)} "
                  f"best={payload['best_feasible_objective']}", flush=True)
        except Exception as exc:
            failure.write_text(json.dumps({"seed": seed, "error": repr(exc),
                                           "traceback": traceback.format_exc()}, indent=2),
                               encoding="utf-8")
            failed_seeds.append(seed)
            print(f"seed {seed} FAILED: {exc!r}", flush=True)
    if failed_seeds:
        print(f"BORA_SOLAR_FAILED seeds={failed_seeds}", flush=True)
        raise SystemExit(1)
    print("BORA_SOLAR_DONE", flush=True)


if __name__ == "__main__":
    main()
