# Baseline optimizer interfaces

All methods implement the dictionary-based `BaselineOptimizer` ask/tell
contract. The shared runner owns objective evaluation and writes baseline runs
under `results/<dataset-instance>/<method>/<prior>/seed-<seed>/`. Dataset
instance components precede the method so independent tasks cannot overwrite
one another.

`random` and `vanilla_bo` run in the main environment. LLAMBO, ColaBO, and
BORA use the line-delimited JSON subprocess bridge in `bridge.py`; this keeps
their mutually incompatible research environments out of the main process.
The parent remains the only component that evaluates objectives, so caching,
budget accounting, and result persistence stay shared across methods.

Budget rule of the bridge: every worker proposal is one charged query. A
proposal that snaps to an already-evaluated candidate returns the cached value
(the objective is not re-run) and is recorded with `"repeat": true`. The
budget is never filled with random candidates or padded: a run fails if the
worker stops before the budget is spent or the upstream method raises. A study
run (`src/experiments/worker.py`) writes `worker_stderr.log` and
`bridge_accounting.json` (budget, queries used, unique, repeats, first query
without a proposal); the accounting file is written for completed runs and for
runs whose worker stopped early after at least one query, not when the worker
proposes nothing or the upstream method raises. The SOLAR external-baseline
runner (`scripts/solar_externals.py`) writes `<stem>.worker_stderr.log` and keeps
the accounting in `<stem>.summary.json`, which only a completed run writes.

Coordinate legend: the x-space baselines (LGBO, BORA, in-loop LLAMBO) see
anonymous coordinates `x0..x{d-1}` in `[0, 1]`. Studies that set
`optimizer.baseline_task_legend: true` append a legend built from the task card
HapBO elicits from (parameter names and meanings, native range, log/linear
mapping of the encoding) on HPOBench and the CNN leak control; on SOLAR the
same legend is enabled with `scripts/solar_externals.py --task-legend`.
Each run writes the LLM-visible text to `baseline_task_text.json`.

## Implemented external baselines

### LLAMBO (`llambo`)

The worker runs the pinned upstream LLAMBO discriminative optimizer. LLAMBO's
continuous `[0,1]^d` proposals are deterministically snapped to the nearest
finite encoded candidate. Its asynchronous OpenAI 0.x call is adapted to
the configured local OpenAI-compatible Qwen endpoint without changing the
upstream optimizer, prompt templates, or discriminative acquisition loop.

Use `LLAMBO_PYTHON` to select the isolated interpreter, or set `LLAMBO_ENV`
for a Conda environment name (default `llambo`). The endpoint is inherited
from `LLM_BASE_URL`, with `OPENAI_BASE_URL` as a fallback. The adapter exposes
the paper-code defaults through `llambo_candidates`, `llambo_templates`,
`llambo_generations`, and `llambo_alpha`.

### ColaBO (`colabo`)

The worker reconstructs upstream's `colabo_logei` Ax `GenerationStrategy`
directly from the pinned ColaBO Ax/BoTorch checkout: `DefaultPrior`, PRIOR
initialization, `SingleTaskGP` with the paper's log-normal GP setup, and
`qPriorLogExpectedImprovement`. It deliberately does not import upstream
`benchmarking.mappings`, whose public checkout unconditionally imports the
absent `mf-prior-bench` package. This benchmark-independent path runs the same
ColaBO algorithm against this repository's objective and aligned discrete
prior.

Use `COLABO_PYTHON` to select the Python 3.9 interpreter, or `COLABO_ENV`
(default `colabo`). The paper-code acquisition defaults are retained:
76,800 MC samples, resampling fraction 0.0067, 1,024 raw samples, and 20
restarts. Continuous Ax proposals are snapped to the nearest finite candidate;
cached duplicate trials are charged as repeat queries until the budget is
used or the configured internal cap is exhausted.

### BORA (`bora`)

BORA also runs through an isolated worker and a locally served OpenAI-compatible endpoint. See
`external/PINNED_COMMITS.txt` for its pinned revision and compatibility notes.

### PriorBand and PriMO (`priorband`, `primo`)

Study strategy `external-mf` (`neps_mf.py`, `workers/neps_worker.py`): the official NePS package
(`neural-pipeline-search==0.17.0`) runs in its own environment and is driven through
`neps.AskAndTell`; the parent evaluates every proposal. Each hyperparameter is a NePS `Integer`
over the index of its grid values, so every proposal is one table row. The runs are natively
multi-fidelity on the HPOBench tables (`src/datasets/hpobench_mf.py`): an evaluation at fidelity z
costs z / z_max full-fidelity units, a promotion is a fresh evaluation, a repeated (configuration,
fidelity) proposal is charged and served from the cache, and a run stops at cumulative cost =
budget (the first proposal that would exceed it is not evaluated). Each run writes
`mf_evaluations.jsonl` (every evaluation, not one query per line), `cost_curves.json` and
`mf_accounting.json`. Select the interpreter with `NEPS_PYTHON`, or set `NEPS_ENV` (default
`neps`). The point priors come from `src/experiments/neps_point_elicitation.py`.

### HypBO (`hypbo`, method `hypbo-llm`)

The worker runs the pinned upstream HypBO in the configuration of its
synthetic-function experiments (DBO `BayesianOptimization` over the continuous
encoded box) and snaps every point it evaluates to the nearest finite
candidate. Its hypotheses are boxes built from a one-shot LLM elicitation
(`scripts/elicit_hypbo_llm.py`, BORA supplement C.3): each hypothesis is a
guessed optimum `c` with a confidence `s`, and its box is centred at `c` with
side `(ub - lb) * (1 - s + 1e-6)` in the encoded space, clipped to the bounds
(`workers/box_hypotheses.py`); a later hypothesis is dropped only when its box
and a kept box with the same centre are both narrower than `1e-4`. Its
per-hypothesis and global initial samples are charged like every other
proposal. `scripts/check_hypbo_hypotheses.py` gates the elicited
artifacts before any run. Use `HYPBO_PYTHON` to select the
Python 3.8 interpreter, or `HYPBO_ENV` (default `hypbo`).

### Other methods

`lgbo` runs the paper-equation posterior lift (`workers/lgbo_worker.py`);
`official_lgbo` runs the public-release lift operator with everything else shared
(`workers/official_lgbo_worker.py`); both use the helpers in `workers/lgbo_common.py`.
`cake` (`cake.py`) is a finite-candidate port of the CAKE reference code: the LLM evolves GP
kernel expressions before every query. `caliper` (`caliper.py`) is a single-objective
reimplementation of CALIPER from the paper; its per-role priors are elicited once, offline
(`src/experiments/caliper_elicitation.py`), and it makes no LLM calls during optimization.
The default model id of the in-loop LLM baselines is defined in `workers/llm_defaults.py`
(`LLM_MODEL` overrides it).
The prior-guided method of this repository is selected by `main.py --method ours`.

| Key | Integration | Environment |
| --- | --- | --- |
| `random` | native | main environment |
| `vanilla_bo` | native wrapper | main environment |
| `llambo` | pinned upstream worker | isolated Python 3.8 |
| `colabo` | pinned Ax/BoTorch worker | isolated Python 3.9 |
| `bora` | pinned upstream worker | isolated Python 3.11 |
| `hypbo` | pinned upstream worker | isolated Python 3.8 |
| `lgbo` | isolated worker, paper-equation posterior lift | LLAMBO environment (numpy, scipy, aiohttp) |
| `official_lgbo` | isolated worker, public-release lift operator | LLAMBO environment |
| `cake` | in-process port of the reference code, LLM in the loop | main environment with torch, gpytorch, botorch |
| `caliper` | in-process reimplementation from the paper, offline-elicited priors | main environment |
| `priorband`, `primo` | NePS 0.17.0 ask-and-tell worker, multi-fidelity, cost-charged | isolated Python 3.12 |

The external checkouts are intentionally absent from the main dependency set
and are imported only by their worker processes. Exact commits, setup details,
known compatibility adaptations, and the validation environment are in
`external/PINNED_COMMITS.txt` and `requirements-baselines.txt`.
