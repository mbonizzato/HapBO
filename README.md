# HapBO — Harnessed α-π Bayesian Optimization via Expert Natural Language Priors

Replication package for the paper **"HapBO: Harnessed α-π Bayesian Optimization via Expert Natural
Language Priors"** (see [Citation](#citation)).

HapBO elicits an expert / LLM belief about where good solutions lie **once**, compiles it
deterministically into a plausibility field `π(x)` over the search space, and admits it into
Bayesian optimization only as a **bounded multiplier** on the acquisition function:

```text
M(x) = (1 - α) + α · π(x)          shaped acquisition:  max( A_raw(x) · M(x), 0 )
```

with `π` min-max-normalized to `[0, 1]` and `α` capped at `0.9`. Because `M(x) ≥ 1 - α > 0`, a
*wrong* belief costs at most a bounded constant factor and can **never zero out a candidate** — the
loop always retains the ability to recover. This repository contains the optimizer, the prior
compiler, the LLM-elicitation tooling, the research campaign (study manifests, elicited priors,
external baselines), the compact run data behind every figure, and the figure generators.

> This is the **replication package for a research paper**, not a general-purpose BO library.

---

## What's in this repository

| Path | Contents |
|---|---|
| `main.py` | Single-run BO / prior CLI (synthetic + one benchmark task at a time) |
| `src/` | Optimizer (`bayesian_optimization/`), prior compiler + vocabulary (`prior/`), dataset adapters (`datasets/`), campaign engine (`experiments/`), external-baseline bridge (`baselines/`), LLM elicitor (`llm/`) |
| `experiments/` | `catalog/` (benchmarks, methods), `studies/` (campaign manifests), `priors/` (elicited priors), `hpo_task_descriptors.json` (HPOBench card inputs), `misleading_descriptions/` (HPOBench misleading hints), `elicitation/` (cortical card inputs), `colabo_llm_priors/` (ColaBO beliefs), `hypbo_llm_hypotheses/` (HypBO hypotheses), `landscape_estimates/`, `synthetic_cards/`, `synthetic_priors/` (synthetic panel), `psl_elicitor_priors/` (PSL elicitor comparison) |
| `scripts/` | Study runner, card builders, elicitation drivers, continuous-panel runners, campaign drivers, downloaders |
| `paper/` | Figure generators (`make_paper.py`) — see `paper/README.md` |
| `results/paper/` | Compact run data read by the figure generators |
| `leak_control_cnn/` | CNN leak-control benchmark (sweep, figure) |
| `external/` | Pinned-commit manifests and the BORA source patch for the external baselines (not vendored code) |
| `tests/` | `pytest` suite |
| `benchmark/`, `results/` (except `results/paper/`) | **gitignored** — datasets and raw run outputs are fetched / regenerated |

---

## Installation

```bash
conda env create --file environment.yaml   # creates env "hapbo"
conda activate hapbo
```

Python 3.11. Core deps (conda-forge): `numpy>=2.0, scipy>=1.14, matplotlib>=3.9, pandas>=2.2,
pyarrow>=16, pyyaml>=6, python-dotenv>=1.0, scikit-learn>=1.5, jsonschema>=4.18, pytest>=8`; via
pip: `anthropic`, `openai` (for LLM elicitation and the `check_*` scripts). Pandas + PyArrow read
the official HPOBench tabular files; scikit-learn fetches the CNN leak-control image subsample
(`leak_control_cnn/prepare_data.py`); jsonschema validates the prior schema.

**External baselines are installed separately.** LLAMBO, ColaBO, BORA, etc. pin incompatible
Python/BoTorch versions and are kept out of `environment.yaml`; each runs in its own
isolated env. See `requirements-baselines.txt` and `external/PINNED_COMMITS.txt`.

---

## Reproducing the paper

Every figure regenerates offline from the committed compact data:

```bash
python paper/make_paper.py          # writes paper/figures/ (about a minute, no benchmark data, no LLM)
```

[`REPRODUCING.md`](REPRODUCING.md) gives, per figure, the full chain that produced that data:

```text
cards -> elicited priors / beliefs -> experiments/studies/**/study.yaml -> scripts/run_study_parallel.py
      -> results/studies/** -> scripts/collect_paper_results.py -> results/paper/ -> paper/make_paper.py
```

---

## The method engine (single-run CLI)

`main.py` runs one BO problem at a time. With no arguments it runs the synthetic example (a discrete
4×8 grid, noisy objective, true optimum at `(7, 4)`, 16 evaluations):

```bash
python main.py                       # == python main.py synthetic-bo
python main.py --prior --alpha 0.7   # add the default approximate prior + α-π shaping
```

Subcommands: `synthetic-bo` (default), `neuromosaic-bo`, `hpobench-bo`,
`solar-bo`, `prior-init`, `prior-finalize`, `landscape-test`.

Key flags (`python main.py --help` for all): `--queries` (budget, default 16), `--acquisition
{expected-improvement,ucb}` (EI default), `--alpha` (prior strength, default 0.7, only active with
`--prior`, capped at 0.9), `--kappa` (UCB), `--improvement-xi` (EI improvement), `--length-scale` (Matérn-5/2
scale, default 2.0), `--initial` (random init, default 1), `--noise-std`, `--seed`, `--plot`.

The α-π rule multiplies the normalized prior into the **raw** acquisition, then zero-clamps — it does
not min-max-normalize the acquisition first:

```text
max( A_raw(x) · ((1 - α) + α · π_norm(x)), 0 )
```

In-engine comparators are `--baselines vanilla-bo` (same GP/EI, no prior) and `--baselines
random-search`. The named external baselines (below) are **not** run through `main.py`.

---

## Benchmarks and data

Data payloads live under the gitignored `benchmark/` tree; fetch them per benchmark:

- **NeuroMOSAIC cortical (Dataset 1)** — the neurostimulation benchmark: rat M1
  (`neuromosaic-dataset1-rat`, subjects `rat1`–`rat6`, 8×4 electrode grid). The `.mat` payloads are
  public and downloaded from OSF (https://osf.io/54vhx/overview; place them under
  `benchmark/NeuroMOSAIC/dataset1/rat`; grid geometry is read from each subject file).
- **HPOBench** — five tabular families `lr, nn, rf, svm, xgb` over eight classification tasks.
  Fetch one combination: `python scripts/download_hpobench.py --model rf --task credit_g`. The
  paper campaign uses a 16-query budget.
- **SOLAR** — the official C++ concentrated-solar-power simulator, nine single-objective instances
  (5–31 D). Prepare with `python scripts/setup_solar.py`; each query runs the simulator live and the
  acquisition is multiplied by the modeled probability of satisfying all constraints.
- **Synthetic** — the continuous panel of the prior-fidelity (`ρ`) study (Hartmann-6 /
  Styblinski-Tang-7 / Levy-10 / Ackley-15, hidden objective in `scripts/synthetic_truth.py`, run by
  `scripts/run_synthetic_panel.py`).

Example single tasks:

```bash
python main.py neuromosaic-bo --dataset neuromosaic-dataset1-rat --subject rat1 \
    --muscle LeftKneeFlexor --queries 12 --seed 42 --plot
python main.py hpobench-bo --hpobench-model rf --hpobench-task credit_g --queries 16 --plot
python main.py solar-bo --solar-instance MAXEFF_RE --queries 16 --plot
```

The dataset commands use the shipped prior `human-expert-map` (NeuroMOSAIC), `llm-card` (HPOBench) or
`expert` (SOLAR MAXEFF_RE) unless `--prior-id` or `--dataset-prior-file` selects another one.

---

## Expert priors

Priors are versioned artifacts under
`experiments/priors/<dataset-id>/<target-slug>/<prior-id>/` holding `prompt_card.md`,
`metadata.json`, and (once finalized) `prior.json`. The vocabulary card lives once at
`src/prior/expert_knowledge/vocabulary_card.md` and is referenced by path and hash, never copied
(priors elicited with another card record that card; see `experiments/priors/README.md`).

**Scripted elicitation (primary, reproducible).** `scripts/elicit_prior.py` runs the LLM once,
validates + compiles the belief document, and writes a finalized, hash-provenanced `prior.json`.
The provider is configurable; the paper's local runs use an OpenAI-compatible endpoint
(`--provider openai --model Qwen3.5-122B-A10B-GPTQ-Int4 --base-url http://localhost:8000/v1`), and
`--provider anthropic` is also supported. `check_local_llm.py` / `check_anthropic_api.py` smoke-
test the endpoints.

**Manual elicitation (fallback).** Create the directory with `python main.py prior-init …`, hand the
LLM the prompt card + vocabulary card, save the returned JSON as `prior.json`, then
`python main.py prior-finalize --prior-directory …` to lock it and generate landscape diagnostics.
Always use a new `prior-id` rather than editing a finalized prior.

**Human-source provenance.** A belief may be tagged `user_stated` or `user_confirmed` only when its
prompt card carries a human (expert) statement, declared as `card.human_statement: true` in
`metadata.json` (`main.py prior-init --card-human-statement`, `scripts/elicit_prior.py
--card-human-statement`). Loading, finalization and the elicitor's validate-and-retry loop reject the
tag otherwise.

The vocabulary card describes the meta tokens (`smoothness`, `noise`, `modality`) as optimizer hints; this
implementation validates them but does not use them.

Shipped priors (full list in `experiments/priors/README.md`): HPOBench `llm-card` (task card),
`llm-card-misleading` (card plus misleading hint), `inverted` (inverted map), `base`, the
representation variants `llm-card(-misleading)-{pointonly,gaussian,directllm}`, the CALIPER multi-role priors, the
PriorBand / PriMO point priors `neps-point-card(-misleading)` and the LLM-EI survey answers and utilities
`llm-ei-card(-misleading)`;
cortical `human-expert-map`, `human-expert-map-shared`, `llm-geometry`,
`llm-expert-hint`; SOLAR `expert`; the 60 synthetic-panel documents in `experiments/synthetic_priors/` (36 elicited
documents and the point-only and Gaussian-only variants of the 12 rho90 documents);
and the CNN leak-control prior.

---

## Research campaign: the study system

The paper's runs are produced by the **study engine**, not `main.py`:

```bash
python scripts/validate_studies.py                          # validate catalogs + manifests
python scripts/run_study.py <study-dir> --write-plan        # dry-run: executable vs blocked runs
python scripts/run_study.py <study-dir> --execute --run-id <id>   # execute one run
python scripts/summarize_study.py <results-dir>             # run/failure/aggregate tables
```

`run_study.py` resolves each method × problem × budget × seed tuple into a planned run and launches
**one subprocess per run** into `scripts/run_experiment.py` (`src/experiments/worker.py`), which
either drives the internal α-π/vanilla engine or bridges to an external baseline — so no two methods
ever share an optimizer trajectory. Each completed leaf under `results/studies/` holds
`study_run.json`, `status.json`, `run_config.json` (with the prior hash + git state),
`evaluations.jsonl`, `metrics.json`, and `best_candidate.json` (the multi-fidelity PriorBand / PriMO
runs write `mf_evaluations.jsonl`, `cost_curves.json` and `mf_accounting.json` instead of
`evaluations.jsonl`).

Study tree: `experiments/studies/{hpo, cortical, solar, cnn-leak}` (`scripts/make_studies.py` writes the
first three; `cnn-leak/` holds the CNN leak-control studies). Run a whole study in parallel with
`python scripts/run_study_parallel.py experiments/studies/<group>/<name> --workers N`. The catalog
under `experiments/catalog/{benchmarks, methods}` defines the named baselines
(`bora, cake, caliper, colabo-llm, hypbo-llm, lgbo, lgbo-prior, llambo, llambo-ws, llm-ei, official-lgbo, priorband,
primo`).

Method ids in the catalog and in the result files: `alpha-pi-raw-ei-matched` is HapBO (α-π shaping of the raw
acquisition, EI, the same seeded initial design as the vanilla arms); `alpha-pi-raw-pi-matched` and
`alpha-pi-raw-ucb-matched` use PI and UCB instead of EI; `alpha-pi-raw-ei-matched-alpha-0-<k>` (k = 1, 3, 5, 7, 9)
fixes α = 0.k instead of the compiled α; `vanilla-bo-{ei,pi,ucb}` are the matching arms without a prior.

---

## External baselines

The upstream LLM-/prior-guided baselines are integrated for the head-to-head comparison. They are
**pinned public checkouts, not vendored**: cloned into the gitignored `external/<name>/` at a fixed
commit, installed in **isolated conda environments**, and invoked through a subprocess bridge
(`src/baselines/*` + `src/baselines/workers/*`). Provenance, licenses, and every upstream deviation
are logged in `external/PINNED_COMMITS.txt`.

| Baseline | Upstream | Integration |
|---|---|---|
| **LLAMBO** | `tennisonliu/LLAMBO` | in-loop LLM on the finite grid; `llambo-ws` = warm-start (Mode-2) |
| **colabo-llm** | `hvarfner/colabo` (no declared license; not vendored) | ColaBO-LogEI via Ax; one-shot (c, s) → N(c, 1−s+ε) point prior |
| **BORA** | `Ablatif6c/bora-the-explorer` | LLM assistant pointed at the locally served endpoint; native continuous, SOLAR via cached candidate pools |
| **HypBO** | `Ablatif6c/HypBO` | box hypotheses from a one-shot LLM elicitation (BORA supplement C.3); every evaluated point snapped to the grid |
| **TuRBO** | `uber-research/TuRBO` | trust-region BO, continuous-only (synthetic panel arm) |
| **LGBO** | from-paper implementation | posterior lift from in-loop LLM guidance (paper equation); also continuous on the synthetic panel |
| **official-lgbo** | `Xinzhe309/LGBO` | the public-release lift operator, everything else shared with LGBO |

CAKE and CALIPER are implemented in-repo (`src/baselines/cake.py`, `src/baselines/caliper.py`). PriorBand and PriMO
run from the official NePS package (`neural-pipeline-search==0.17.0`) in their own environment,
natively multi-fidelity on the HPOBench tables with every evaluation charged z/z_max
(`src/baselines/neps_mf.py`). LLM-EI is implemented in-repo from its published text (the public
repository does not contain its percentile gate): the LLM answers a pairwise survey offline
(`src/experiments/llm_ei_survey.py`), a preference GP fitted to the answers gives the utility
(`src/experiments/llm_ei_utility.py`), and the study worker runs vanilla BO restricted to the top of that
utility (`src/experiments/llm_ei.py`). piBO (`pi-bo-ei`) follows Hvarfner et al. (2022), Sec. 3.2, Algorithm 1 and
Sec. 4, with the compiled prior as the user belief: the initial design is the prior mode followed by points sampled
from the prior, and EI is weighted by the prior raised to beta / n with beta = N / 10 for N BO iterations
(`src/bayesian_optimization/bo.py`, `acquisition.pi_bo_shape`).

Per-baseline continuous runners live in `scripts/` (`llambo_continuous.py`, `colabo_llm_continuous.py`,
`bora_continuous.py`, `lgbo_continuous.py`, and `run_synthetic_arms.py` for HapBO, vanilla BO, random
search, TuRBO and the vanilla-peaks control, i.e. vanilla BO whose candidates include HapBO's prior peaks);
`scripts/run_synthetic_panel.py` drives them over the whole panel and
`scripts/run_solar_baselines.py` runs the external baselines on SOLAR.

---

## Testing

```bash
pytest                 # from the repo root
```

`tests/` covers the optimizer, prior compiler, dataset adapters, baselines, study system,
injection arms, the synthetic hidden objectives and the LGBO operator. Tests skip
when an optional dependency (`scipy`, `scikit-learn`, `botorch`, the NePS interpreter `NEPS_PYTHON`, the HypBO
interpreter `HYPBO_PYTHON` with `HYPBO_ROOT` or `external/hypbo`) or a gitignored
data payload (the NeuroMOSAIC `.mat` files) is absent (`importorskip` / `skipUnless`), so a fresh clone runs the
subset whose inputs are present.

---

## Using the engine from Python

```python
from main import run_prior_guided_optimization

result = run_prior_guided_optimization(
    native_candidates=native_candidates,   # list of dicts (schema-keyed) or a matrix in schema order
    objective=evaluate_configuration,      # receives one native candidate, returns a scalar
    max_evaluations=40,
    random_state=42,                       # keyword-only
)
print(result.best_native_candidate)
print(result.optimization.best_y)
```

The runner loads a versioned `prior.json`, compiles it (`compile_prior`), builds aligned numeric GP
features (log-aware scaling, ordinal ranks, one-hot categoricals — override with
`encoded_candidates=`), evaluates one `π` value per native configuration, applies the compiler's `α`
through the α-π rule, and follows the schema's maximize/minimize direction. See
[`src/bayesian_optimization/README.md`](src/bayesian_optimization/README.md) for the algorithm
details.

---

## Repository layout

```text
./
├── main.py                     # single-run BO / prior CLI
├── environment.yaml            # conda env "hapbo" (Python 3.11)
├── requirements-baselines.txt  # recipe: install each external baseline in its own env
├── check_anthropic_api.py      # verify the Anthropic API key / model
├── check_local_llm.py          # smoke-test the local OpenAI-compatible LLM endpoint
├── LICENSE
├── src/
│   ├── bayesian_optimization/  # GP (Matérn-5/2, RBF, Matérn-1/2), EI/PI/UCB, α-π shaping, ask/tell loop, batch selection
│   ├── prior/                  # prior compiler + landscape generator + expert_knowledge/vocabulary cards
│   ├── datasets/               # rat, HPOBench, SOLAR, CNN adapters
│   ├── experiments/            # study planner / execution / worker (campaign engine)
│   ├── baselines/              # external-baseline adapters + isolated-env workers
│   └── llm/                    # PSL elicitor
├── experiments/
│   ├── catalog/                # benchmarks, methods (incl. named baselines)
│   ├── studies/                # hpo, cortical, solar, cnn-leak
│   ├── priors/                 # versioned elicited priors
│   ├── hpo_task_descriptors.json  # HPOBench task descriptors (card inputs)
│   ├── misleading_descriptions/   # HPOBench misleading hints
│   ├── elicitation/            # cortical card inputs
│   ├── colabo_llm_priors/      # ColaBO (c, s) beliefs
│   ├── hypbo_llm_hypotheses/   # HypBO box hypotheses
│   ├── landscape_estimates/    # synthetic formula cards, specs, well-truth parameters
│   ├── synthetic_cards/        # LLM inputs of the synthetic elicitation
│   ├── synthetic_priors/       # compiled synthetic-panel documents
│   └── psl_elicitor_priors/    # PSL documents of the elicitor comparison
├── scripts/                    # study runner, builders, elicitation and campaign drivers, runners
├── paper/                      # figure generators (see paper/README.md)
├── results/paper/              # compact data read by the figure generators
├── leak_control_cnn/           # CNN leak-control benchmark
├── external/                   # pinned-commit manifests + BORA source patch
├── tests/                      # pytest suite
└── benchmark/                  # gitignored data payloads (OSF https://osf.io/54vhx/overview, download scripts)
```

---

## Notes

- **Benchmark data is not shipped.** `benchmark/` and raw run outputs are gitignored. Fetch
  NeuroMOSAIC Dataset 1 from OSF (https://osf.io/54vhx/overview), HPOBench tables via `scripts/download_hpobench.py`, and build SOLAR
  via `scripts/setup_solar.py` before running those benchmarks. The compact data behind the figures
  is committed under `results/paper/`.
- **LLM endpoint.** `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` (OpenAI-compatible server, e.g. a local
  vLLM serving Qwen3.5-122B-A10B-GPTQ-Int4); `ANTHROPIC_API_KEY` for the Anthropic provider. Keys are
  read from the environment (or a gitignored `.env`); none are in the tree.

---

## Citation

If you use this code, please cite:

```bibtex
@misc{giroux2026hapbo,
  title  = {{HapBO}: Harnessed $\alpha$-$\pi$ {Bayesian} Optimization via Expert Natural Language Priors},
  author = {Giroux, Jean-S{\'e}bastien and Guay Hottin, Rose and Bonizzato, Marco},
  year   = {2026}
}
```

The benchmarks are third-party data; please also cite them:

- **HPOBench**: K. Eggensperger, P. Müller, N. Mallik, M. Feurer, R. Sass, A. Klein, N. Awad, M. Lindauer and
  F. Hutter. HPOBench: A Collection of Reproducible Multi-Fidelity Benchmark Problems for HPO. *Proceedings of the
  Neural Information Processing Systems Track on Datasets and Benchmarks*, 2021.
- **NeuroMOSAIC Dataset 1** (cortical benchmark): L. Kardassevitch, D. Burchielli, N. Dancause and M. Bonizzato.
  NeuroMOSAICS: A collection of neurostimulation datasets - Multi-scale Open-Source Across Interfaces Conditions
  & Species. *NeuroLibre Reproducible Preprints* 33, 2025. https://doi.org/10.55458/neurolibre.00033. The rat
  files are downloaded from OSF (https://osf.io/54vhx/, CC BY 4.0); `results/paper/` contains values derived
  from them.
- **SOLAR**: N. Andrés-Thió, C. Audet, M. Diago, A. E. Gheribi, S. Le Digabel, X. Lebeuf, M. Lemyre Garneau and
  C. Tribes. Solar: a solar thermal power plant simulator for blackbox optimization benchmarking. *Optimization
  and Engineering* 26(3):1815-1861, 2025. https://doi.org/10.1007/s11081-024-09952-x.

The external baselines are listed with their upstream repositories and licenses in `external/PINNED_COMMITS.txt`.
