# Reproducing the paper figures

Two levels of reproduction are supported.

1. **Figures from the committed data** (minutes, laptop):

   ```bash
   python paper/make_paper.py
   ```

   reads `results/paper/` and writes every data figure to `paper/figures/` (see `paper/README.md`
   for the module, file name and data of each figure).

2. **Campaigns from scratch** (GPU LLM server, external baselines, benchmark data): the chains
   below regenerate the cards, priors, beliefs and runs, then `python scripts/collect_paper_results.py`
   rebuilds `results/paper/` from the raw outputs and step 1 draws the figures.

## Common setup

| Item | Setup |
| --- | --- |
| Main environment | `conda env create --file environment.yaml` (env `hapbo`) |
| LLM endpoint | OpenAI-compatible server; `LLM_BASE_URL` (default `http://localhost:8000/v1`), `LLM_MODEL` (default `Qwen3.5-122B-A10B-GPTQ-Int4`), `LLM_API_KEY`. All elicitations and in-loop LLM calls of the paper used this model with thinking disabled unless noted. |
| External baselines | `external/README.md` (clone, pin, apply the BORA patch); interpreters `LLAMBO_PYTHON`, `COLABO_PYTHON`, `LGBO_PYTHON`, `OFFICIAL_LGBO_PYTHON`, `HYPBO_PYTHON`, BORA env `BORA_ENV`; PriorBand / PriMO: `NEPS_PYTHON` (NePS 0.17.0, `requirements-baselines.txt`) |
| HPOBench | `python scripts/download_hpobench.py --model <lr|rf|xgb|nn|svm> --task <task>` -> `benchmark/HPOBench` |
| NeuroMOSAIC Dataset 1 (rat) | download from OSF (https://osf.io/54vhx/overview) -> `benchmark/NeuroMOSAIC/dataset1/rat` |
| SOLAR | `python scripts/setup_solar.py` (Linux C++ simulator) |
| Studies | `python scripts/run_study_parallel.py experiments/studies/<group>/<name> --workers N` writes `results/studies/<group>/<name>/runs/...`; `python scripts/validate_studies.py` checks every manifest |

`python scripts/make_studies.py` writes the hpo, cortical and solar study manifests (`--check` compares
them with the committed files); the `experiments/studies/cnn-leak/` manifests are maintained by hand.
Budgets: HPOBench 16, cortical 12, SOLAR 50, CNN leak control 30. Seeds: HPOBench and cortical 1001-1010,
SOLAR 2001-2020, CNN leak control 1-10.

## Inputs shared by several figures

| Input | Produced by |
| --- | --- |
| HPOBench task cards (`base`, `llm-card`, `llm-card-misleading` `prompt_card.md`) and misleading hints (`experiments/misleading_descriptions/`) | `python scripts/build_hpo_cards.py` from `experiments/hpo_task_descriptors.json` |
| HPOBench HapBO priors `base`, `llm-card`, `llm-card-misleading` | `python scripts/elicit_hpo_priors.py --card <id>` (seed 7, temperature 0) |
| HPOBench `inverted` priors | `python scripts/make_inverted_priors.py` (no LLM) |
| HPOBench representation variants `llm-card-{pointonly,gaussian}`, `llm-card-misleading-{pointonly,gaussian}` | `python scripts/make_representation_variants.py <prior_dir> ...` (no LLM); `llm-card-directllm`, `llm-card-misleading-directllm`: `python scripts/make_directllm_priors.py <prior_dir> ...` |
| ColaBO (c, s) beliefs | HPOBench and cortical: `python scripts/elicit_colabo_llm_priors.py` (`--adversarial` for the HPOBench misleading-hint beliefs; thinking off); SOLAR: step `colabo-solar` of `python scripts/elicit_cortical_solar_priors.py` (`scripts/elicit_colabo_llm.py --enable-thinking`, thinking on); synthetic: `python scripts/elicit_colabo_llm.py --dataset-id synthetic-continuous-<slug> --slug <slug>-<card> --prompt-card-file experiments/synthetic_cards/<card>/<fam>/TASK.md --schema-json experiments/landscape_estimates/<fam>/schema.json --instruction formula --enable-thinking` per function and card (thinking on; `<slug>` is `<fam>` with `-` for `_`). Each belief file records its inference settings (`temperature`, `enable_thinking`) |
| HypBO hypotheses (`hpo/hypbo`, `hpo/hypbo-misleading`) | `python scripts/elicit_hypbo_llm.py [--adversarial] --workers 4` (HPOBench), then the gate `python scripts/check_hypbo_hypotheses.py`, which re-reads the raw replies the elicitor keeps under `results/elicitation/hypbo/` (not shipped) |
| CALIPER priors `caliper-card(-misleading)` | `python -m src.experiments.caliper_elicitation --all --card llm-card|llm-card-misleading` |
| PriorBand / PriMO point priors `neps-point-card(-misleading)` | `python -m src.experiments.neps_point_elicitation --all --card llm-card|llm-card-misleading` (seed 7, temperature 0) |
| LLM-EI survey answers and utilities `llm-ei-card(-misleading)` | `python -m src.experiments.llm_ei_survey --all --card llm-card|llm-card-misleading --rounds 10 --workers 12` (seed 7, temperature 0; writes `survey_answers.json`), then `python -m src.experiments.llm_ei_utility --all --card llm-card|llm-card-misleading` (BoTorch; writes `llm_ei_utility.json.gz`) |
| Cortical human-expert-map prior | `python scripts/build_cortical_expert_prior.py` (no LLM) |
| Cortical and SOLAR cards and shared texts | `python scripts/build_cortical_solar_cards.py`; priors and beliefs: `python scripts/elicit_cortical_solar_priors.py`, then `python scripts/replicate_cortical_priors.py` |
| Synthetic cards and LLM inputs | `python scripts/build_synthetic_cards.py <step>` for the steps `specs`, `cards`, `well`, `tasks`, in that order (`experiments/landscape_estimates/`, `experiments/synthetic_cards/`; `well` takes minutes per card) |
| Synthetic HapBO documents | `python scripts/elicit_synthetic_priors.py` (3 draws per card and function; temperature 0.6, top-p 0.95, thinking on); point-only and Gaussian-only variants of the rho90 documents: `python scripts/make_representation_variants.py --psl-docs` (no LLM) |
| Cortical exploration scores | `python scripts/score_exploration.py --studies human-map llm-geometry llm-expert-hint human-map-all human-map-colabo human-map-llambo human-map-lgbo human-map-bora` (reads `results/studies/cortical/` and `benchmark/NeuroMOSAIC/dataset1/rat/`) -> `results/exploration_scores/cortical/` |

## Per-figure chains

Figures are named by their output file in `paper/figures/` (`supp/` for the supplementary figures).

### `fig2_hpo_compact`, `fig2_hpo_b16` — HPOBench, standard prior
Studies `hpo/hapbo`, `hpo/vanilla-random`, `hpo/colabo`, `hpo/llambo` (Mode 2), `hpo/lgbo`, `hpo/bora`.
40 tasks, budget 16, 10 seeds. Regret normalized by the grid span; rank panel over tasks.

### `fig3_realworld` — cortical stimulation and SOLAR
- Panel A, cortical: `cortical/human-map-all` (HapBO EI, vanilla BO, random search), `cortical/human-map-colabo`,
  `cortical/human-map-llambo`, `cortical/human-map-lgbo`, `cortical/human-map-bora`, all 40 rat-muscle maps
  (`neuromosaic-dataset1-rat-all`); exploration performance from `scripts/score_exploration.py`, 12 queries
  (every cortical study has `budgets: [12]`), mean ± SEM over the maps (technical replicates). The LLAMBO values
  (`human-map-llambo`): the 2 runs in which upstream raised (no candidate points generated; a candidate key
  outside the search space, KeyError) count as failed; 398 of the 400 runs are scored.
- Panel B, SOLAR MAXEFF_RE: `solar/maxeff-re` (HapBO, vanilla BO, random search) and the external arms
  `python scripts/run_solar_baselines.py` (ColaBO, LGBO, LLAMBO Mode 2 through the SOLAR external-baseline
  runner `scripts/solar_externals.py`; BORA through `scripts/cache_solar_pools.py` and
  `scripts/bora_solar_cached.py`), 20 seeds, budget 50 shown to query 20. ColaBO peaks at about 75 GB RSS.

### `fig4_synthetic_fidelity` — synthetic panel
Hidden objective `SYNTH_TRUTH=well` (scripts/synthetic_truth.py). Functions Hartmann-6 (budget 40),
Styblinski-Tang-7 (50), Levy-10 (50), Ackley-15 (150); cards rho90, well50, well10.
`python scripts/run_synthetic_panel.py` runs every arm (`--dry-run` prints each command): HapBO
(`run_synthetic_arms.py --label <fam>-draw<d> --families <fam> --arms hapbo --prior-file
experiments/synthetic_priors/<card>/<fam>/draw<d>/prior.json --alpha-override 0.325`,
3 draws x 10 seeds), vanilla BO, random search and TuRBO (card-independent), ColaBO
(`colabo_llm_continuous.py`), LLAMBO Mode 2 (`llambo_continuous.py --mode zero-shot`), BORA
(`bora_continuous.py --precision 4`) and LGBO (`lgbo_continuous.py`). Outputs under
`results/synthetic/`; regret normalized by the vanilla-BO query-1 regret per function.

### `fig5_adv_compact`, `fig5_robustness` — HPOBench, adversarial prior
HapBO with the inverted map (`hpo/hapbo-inverted`); baselines with the card plus the misleading hint
(`hpo/colabo-misleading`, `hpo/llambo-misleading`, `hpo/lgbo-misleading`, `hpo/bora-misleading`);
`hpo/vanilla-random`. Regret normalized by the vanilla-BO query-1 regret.

### `fig_prior_sources_main`, `supp/figS2_prior_sources_and_trajectories` — cortical prior sources
`cortical/llm-geometry`, `cortical/llm-expert-hint`, `cortical/human-map` (HapBO and vanilla BO);
exploration performance to query 12.
`supp/figS2_prior_sources_and_trajectories` panels A-C are the compiled priors `llm-geometry`, `llm-expert-hint`,
`human-expert-map`; D-F three example truth maps.

### `fig_ablations` — prior-injection site
- Panels A, B: `hpo/hapbo`, `hpo/injection` (standard) and `hpo/hapbo-misleading`, `hpo/injection-misleading`
  (prior elicited from the card plus the misleading hint), with `hpo/vanilla-random`; all 40 tasks.
  piBO (`pi-bo-ei`) as in Hvarfner et al. (2022): prior max(pi, 0) / max(pi) + 1e-12; initial design: its mode,
  then points sampled from it (n_initial = 1 here: the mode); EI weighted by it to the power beta / n, n = 1 at the
  first BO query, beta = N / 10 with N = budget - n_initial BO iterations (1.5 for HPOBench).
- Panel C: synthetic panel, well50 card (rho ~ 0.5): the synthetic prior-injection arms (next section) and HapBO on
  the HapBO documents, vanilla BO and random search (the `fig4_synthetic_fidelity` runs); grand mean over the 4
  functions, log scale.

### `fig_ablations` panel C — synthetic prior-injection arms
The four prior-injection arms of `fig_ablations` on the continuous synthetic path
(`run_continuous_bo(prior_strategy=...)`; continuous_bo module docstring), with the HapBO documents, seeds and hidden
objective of the synthetic panel:
`python scripts/run_synthetic_panel.py --cards well50 --arms pi-bo,prior-to-mean,prior-warm-start,prior-greedy`
(`run_synthetic_arms.py --label <fam>-draw<d> --families <fam> --arms <arm> --prior-file
experiments/synthetic_priors/<card>/<fam>/draw<d>/prior.json`,
3 draws x 10 seeds). The candidate set of these arms is the distinct points of HapBO's first acquisition batch (its
Sobol points and the prior peaks); every later batch is HapBO's. piBO starts at the prior mode of the candidate set,
samples the other initial points from the prior over it and weights EI by the prior to the power beta / n,
beta = N / 10 (N = budget - 3 BO iterations); prior-to-mean uses the GP prior mean compiled alpha x standardized
prior; prior-warm-start starts at the highest-prior candidates; prior-greedy queries them in descending prior order
without a GP.

### `fig_representation` — representation ablation
Full-PSL, point-only, Gaussian-only and direct-LLM representations of the same belief (alpha and trust of the source
prior), with vanilla BO; grand mean, average rank and per-function / per-family panels. The point-only and
Gaussian-only synthetic documents exist for the rho90 card only.
- Synthetic panel, rho90 card: full documents, `draw<d>-pointonly`, `draw<d>-gaussian` (arms `hapbo-pointonly`,
  `hapbo-gaussian` of `scripts/run_synthetic_panel.py`), vanilla BO (the `fig4_synthetic_fidelity` runs).
- HPOBench, good prior: `hpo/hapbo`, `hpo/representation-{pointonly,gaussian,directllm}`, `hpo/vanilla-random`;
  40 tasks, grid-span normalized.
- HPOBench, inverted-map prior: `hpo/hapbo-inverted` (full-PSL),
  `hpo/representation-{pointonly,gaussian,directllm}-misleading` (representation variants of the prior elicited from
  the card plus the misleading hint), `hpo/vanilla-random`; 40 tasks, vanilla-q1 normalized.

### `fig_alpha_sweep_hapbo` — fixed-alpha sweep
HapBO (`alpha-pi-raw-ei-matched`) with the trust weight alpha fixed instead of compiled from the PSL document:
`hpo/alpha-sweep` (card prior) and `hpo/alpha-sweep-misleading` (card plus the misleading hint), methods
`alpha-pi-raw-ei-matched-alpha-0-{9,7,5,3,1}` (catalog field `alpha`). The compiled-alpha arm is `hpo/hapbo` /
`hpo/hapbo-misleading`, alpha = 0 is vanilla BO (`hpo/vanilla-random`). 40 tasks, budget 16, seeds 1001-1010.

### `fig_injection_synthetic`, `supp/figS_injection_synthetic_per_function` — prior-peak control
Synthetic panel of `fig4_synthetic_fidelity` (same cards, documents, seeds and hidden objective) with a third arm,
vanilla BO whose acquisition candidates include HapBO's prior peaks and whose acquisition is unweighted:
`python scripts/run_synthetic_panel.py --arms vanilla-peaks` (`run_synthetic_arms.py --label <fam>-draw<d> --families
<fam> --arms vanilla-peaks --prior-file experiments/synthetic_priors/<card>/<fam>/draw<d>/prior.json
--alpha-override 0.325`).
HapBO and vanilla BO are the `fig4_synthetic_fidelity` runs.

### `fig_bounded_shaping` (panels also as `fig_acq_across`, `fig_batch_bo`, `fig_kernel_bo`) — acquisition, batch, kernel
`hpo/acquisition`, `cortical/acquisition` (EI, PI, UCB for HapBO and vanilla BO) and `hpo/batch`,
`cortical/batch` (`q_batch: 3`, Kriging believer); HPOBench rows use all 40 tasks (budget 16), cortical rows
all 40 rat-muscle maps with the `llm-expert-hint` prior, 12 queries, mean ± SEM over the maps.
`fig_kernel_bo`: EI for HapBO and vanilla BO with the Matérn-5/2 kernel (the EI arms of the
acquisition studies), `hpo/kernel-rbf`, `cortical/kernel-rbf` (`kernel: rbf`) and `hpo/kernel-matern12`,
`cortical/kernel-matern12` (`kernel: matern12`). `fig_bounded_shaping` draws the panels of the three figures in
one figure (A-C acquisition, D-E batch, F-H kernels).

### `fig_elicitation_ladder`, `tables/table_ladder_cost.tex` — PSL elicitors
Five elicitors (Qwen3.5-122B-A10B-GPTQ-Int4 on the local endpoint, `qwen`; Claude Haiku 5.5, Sonnet 5.5, Opus 5.5,
Fable 5.1: `claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-opus-5-5`, `claude-fable-5-1`) write a PSL document
for each rho90 card of the synthetic panel, by two routes:

- **One API call** (no tools; 60-token cap; 3 draws): `python scripts/elicit_synthetic_priors.py --elicitor <id>
  --cards rho90 --draws 1-3 --usage`. Inputs: `experiments/synthetic_cards/ELICITATION_PROMPT.md` with the card's
  `TASK.md`, `PSL_VOCABULARY.md` and `psl_schema.json`; up to 4 attempts with a correction turn. Claude: Messages
  API (`ANTHROPIC_API_KEY`), streaming, `max_tokens` 64000, model-default sampling; Qwen: thinking on,
  temperature 0.6, top-p 0.95, seed 1000 x draw + attempt.
- **Agentic** (one coding-agent session per elicitor and function; token cap 2 x dimension, i.e. 12 / 14 / 20 / 30
  PSL tokens for Hartmann-6 / Styblinski-Tang-7 / Levy-10 / Ackley-15; wall-clock cap 1 h):
  `python scripts/agentic_elicit.py --elicitor <id> --work-dir <dir> --out-root <out>`. Each session starts in a
  fresh workspace (`TASK.md`, `PSL_VOCABULARY.md`, `psl_schema.json`, a copy of the PSL compiler) with the prompt
  `experiments/synthetic_cards/AGENT_PROMPT_2N_<function>.md`. Claude elicitors run the Claude Code CLI in print
  mode (`claude -p <prompt> --model <id> --permission-mode bypassPermissions --output-format stream-json`);
  Qwen runs the Qwen Code CLI against `LLM_BASE_URL` / `LLM_MODEL` (`--bwrap` reproduces the bubblewrap sandbox on
  Linux). A run counts when the session exits cleanly with a `prior.json`; tokens and list-price cost come from the
  session stream (`usage.json`).

> **Warning:** the agentic route gives the coding agent unrestricted shell access (Claude Code with `bypassPermissions`, Qwen Code with `--yolo` unless `--bwrap` is set); run it in a container or VM.

The documents used in the figure are in `experiments/psl_elicitor_priors/<elicitor>/{draw1,draw2,draw3,agentic}/`
(`<elicitor>`: `qwen3.5`, `haiku-5.5`, `sonnet-5.5`, `opus-5.5`, `fable-5.1`).
Fidelity to the card model on 100k fixed uniform points: `python scripts/score_fidelity.py <prior.json> <function>`
(`spearman_model`, `top10_model`). Regret on the well truth: `python scripts/run_synthetic_arms.py --label <label>
--families <function> --arms hapbo --prior-file <prior.json> --alpha-override 0.325 --seeds 10` per document
(`SYNTH_TRUTH=well`),
vanilla BO from `scripts/run_synthetic_panel.py`. `paper/fig_psl_elicitors.py` reads
`results/paper/psl_elicitors/psl_elicitors.json` (fidelity, usage, token-type counts, regret trajectories and, in its
`slice` entry, the slice of row 1: Styblinski-Tang-7, x1 vs x6).

### `fig_leakctrl_regret` — CNN leak control
`leak_control_cnn/README.md`: `python leak_control_cnn/prepare_data.py`, `python leak_control_cnn/sweep_cnn.py`
(training grid) and `python leak_control_cnn/build_table.py`, then the CNN leak-control studies
`cnn-leak/cnn-leak-of-campaign` (HapBO, vanilla BO, random search), `cnn-leak/cnn-leak-of-colabo`,
`cnn-leak/cnn-leak-of-llambo-ws` (LLAMBO Mode 2), `cnn-leak/cnn-leak-of-lgbo`, `cnn-leak/cnn-leak-of-bora`.
`python scripts/collect_paper_results.py --only cnn` writes `results/paper/cnn_leak/<arm>.json`.

The CNN prior `experiments/priors/cnn-leak/card-overfit/` is a one-shot elicitation through
`scripts/elicit_prior.py` (Qwen3.5-122B-A10B-GPTQ-Int4, seed 7, temperature 0, thinking off; generation notes in
its `metadata.json`) with the ML-example vocabulary card, selected with
`HAPBO_VOCABULARY_CARD=src/prior/expert_knowledge/vocabulary_card_ml_example.md`. Its `prompt_card.md` holds the
domain text `leak_control_cnn/elicitation/cnn_leak_domain_overfit.txt`, the optimization text
`leak_control_cnn/elicitation/cnn_leak_opt.txt` and the four dimensions of
`leak_control_cnn/elicitation/cnn_leak_schema.json`; `scripts/elicit_prior.py --prompt-card-file` sends a stored
card verbatim. The ColaBO belief `experiments/colabo_llm_priors/cnn-leak/fashion-of/card-overfit.json` is a (c, s)
point prior elicited once from the same `prompt_card.md` (same model, temperature 0, thinking off; settings recorded in
the file).

### `fig_baseline_impl_sensitivity` — baseline-implementation sensitivity
`hpo/lgbo` vs `hpo/lgbo-official` (lift operator of the authors' official implementation) and
`hpo/llambo` (Mode 2) vs `hpo/llambo-mode1`, each with its misleading counterpart; HapBO `hpo/hapbo` /
`hpo/hapbo-misleading`; 32 tasks (lr, rf, xgb, nn).

### `supp/figS_cake_vs_hapbo_b16` — CAKE and CALIPER
`hpo/hapbo`, `hpo/hapbo-misleading`, `hpo/caliper`, `hpo/caliper-misleading`, `hpo/cake`,
`hpo/vanilla-random`.

### `supp/figS_priorband_primo_vs_hapbo_b16` — PriorBand and PriMO
`hpo/priorband`, `hpo/priorband-misleading`, `hpo/primo`, `hpo/primo-misleading` (NePS 0.17.0 worker,
`NEPS_PYTHON`), plus `hpo/hapbo`, `hpo/hapbo-misleading`, `hpo/vanilla-random`. The lower-fidelity
evaluations are read from the rows of the same HPOBench tables downloaded above (no separate download;
`src/datasets/hpobench_mf.py`). Budget 16 is in full-fidelity cost units: each evaluation costs z/z_max, and the
runs stop at cumulative cost 16. The figure plots the best full-fidelity evaluation against cumulative
cost for PriorBand and PriMO and against the query index for the other arms.

### `supp/figS_hypbo_llmei_vs_hapbo_b16` — HypBO and LLM-EI
`hpo/hypbo`, `hpo/hypbo-misleading` (HypBO worker, `HYPBO_PYTHON`; hypotheses from `scripts/elicit_hypbo_llm.py`,
gated by `scripts/check_hypbo_hypotheses.py`), `hpo/llm-ei`, `hpo/llm-ei-misleading` (main environment; utilities
from `src/experiments/llm_ei_survey.py` and `src/experiments/llm_ei_utility.py`), plus `hpo/hapbo`,
`hpo/hapbo-misleading`, `hpo/vanilla-random`.

## Checks

```bash
pytest
python scripts/validate_studies.py
python scripts/make_studies.py --check
```
