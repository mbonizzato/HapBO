# Research studies

Each study directory holds one `study.yaml` that specifies benchmarks, methods, budgets, seeds
and optimizer settings. One planned run is one method, benchmark problem, budget, seed, process
and result directory (`results/studies/<group>/<name>/runs/...`).

| Group | Studies |
| --- | --- |
| `hpo/` | HPOBench campaign (40 tasks, budget 16, seeds 1001-1010): HapBO with the card, misleading and inverted priors; HapBO with alpha fixed at 0.9, 0.7, 0.5, 0.3, 0.1 (card and misleading priors); vanilla BO and random search; LGBO, LGBO with the lift operator of the official implementation, ColaBO, BORA, LLAMBO Mode 2 and Mode 1 (each with and without the misleading hint); prior-injection sites; representation variants (card and misleading priors); acquisition functions; GP kernels; batch BO; CAKE; CALIPER; PriorBand and PriMO (multi-fidelity, with and without the misleading hint); HypBO and LLM-EI (each with and without the misleading hint) |
| `cortical/` | NeuroMOSAIC rat maps (budget 12, seeds 1001-1010). The 16 flexor maps: HapBO, vanilla BO and random search with the human expert map; HapBO with the LLM-only and LLM + expert-hint priors. All 40 rat-muscle maps: HapBO, vanilla BO and random search with the human expert map; ColaBO / LLAMBO / LGBO / BORA with the shared human-map text; acquisition functions, GP kernels and batch BO with the LLM + expert-hint prior |
| `solar/` | SOLAR MAXEFF_RE (budget 50, seeds 2001-2020): HapBO, vanilla BO and random search |
| `cnn-leak/` | the CNN leak-control studies (`cnn-leak-of`, budget 30, seeds 1-10; `leak_control_cnn/`): HapBO, vanilla BO and random search; ColaBO, LLAMBO Mode 2, LGBO, BORA |

`design.optimizer` holds the GP and acquisition settings shared by every arm of a study; `kernel`
selects the covariance of the GP surrogate (`matern52`, the default when the key is absent, `rbf` or
`matern12`; the `kernel-rbf` and `kernel-matern12` studies repeat the EI arms of `acquisition` with those
kernels). A study that sets `kernel` may use only methods that fit that GP (strategies `alpha-pi`,
`vanilla-bo`, `pi-bo`, `prior-to-mean`, `prior-warm-start`, `llm-ei`); the planner rejects it with external
baselines, random search or prior-greedy.

`scripts/make_studies.py` writes the `hpo/`, `cortical/` and `solar/` manifests;
`scripts/validate_studies.py` checks every manifest against the catalog. Run a study with
`python scripts/run_study_parallel.py experiments/studies/<group>/<name> --workers N`.
