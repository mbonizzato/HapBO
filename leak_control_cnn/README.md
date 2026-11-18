# Leak-control benchmark: a bespoke CNN trained from scratch

**Purpose.** Controls for benchmark contamination: HPOBench and the analytic functions are public, so an LLM
may have memorized where their optima are. Here the optimization target is a small custom convolutional encoder
trained from scratch, whose hyperparameter-response surface is computed over a finite 4-D grid on a fixed image
subsample. No public leaderboard, paper or repository contains the accuracy of every configuration of this grid,
so the location of the optimum cannot be recalled. An LLM can still form a reasonable prior from general ML
knowledge (learning rate around 1e-3, moderate capacity, light regularization); a gain of HapBO over vanilla BO
on this benchmark is therefore transferable prior shaping, not recall.

## The problem

- **Network** (`model.py`, `ConvEncoderClassifier`): `Conv3×3(C)→BN→ReLU→MaxPool`
  → `Conv3×3(2C)→BN→ReLU→MaxPool` → `AdaptiveAvgPool(4×4)` → `Flatten` →
  `Dropout` → `Linear(10)`; ~3.9k (C=8) to ~57.7k (C=48) parameters.
- **Data**: a fixed train/val/test subsample of **Fashion-MNIST** (28×28), split seed fixed so every
  configuration sees an identical objective.
- **Training regime** `fashion-of` (`src/datasets/cnn_leak.py:REGIMES`): 2k/2k/2k train/val/test images,
  35 epochs, batch 64, an overfitting regime in which moderate dropout and weight decay are optimal.
- **Search space** (4-D, **750 configs**): `learning_rate` (6) × `base_channels` (5) × `dropout` (5) ×
  `weight_decay` (5).
- **Objective**: validation accuracy averaged over 3 training seeds (seed SD stored as the response SD); higher
  is better.
- **BO budget**: 30 queries over 750 options, 10 seeds (the CNN leak-control studies
  `experiments/studies/cnn-leak/cnn-leak-of-*`).

## Building the benchmark table (CPU)

Training one configuration (3 seeds) takes minutes on one CPU core. Convolutions are run single-threaded
and configurations in parallel.

```bash
pip install torch scikit-learn                     # CPU torch + scikit-learn (dataset download)

# 1. fetch and cache the fixed Fashion-MNIST subsample (needs internet once)
python leak_control_cnn/prepare_data.py --dataset fashion-of

# 2. (optional) wiring / environment check
python leak_control_cnn/smoke_test.py --with-training

# 3. train every configuration (resumable: existing result files are skipped)
python leak_control_cnn/sweep_cnn.py --dataset fashion-of --workers 8

# 4. aggregate into the benchmark table read by the harness
python leak_control_cnn/build_table.py --dataset fashion-of
#    -> benchmark/cnn_leak/table_fashion-of.json  (mean, sd, seeds per config)
```

`sweep_cnn.py --start/--stop` restricts the sweep to a range of configuration indices (e.g. to split it over
machines). `build_table.py` refuses to write a partial table unless `--allow-partial`.

## Running the BO comparison

The studies `experiments/studies/cnn-leak/cnn-leak-of-{campaign,colabo,llambo-ws,lgbo,bora}` run HapBO (matched
initialization), vanilla GP-EI, random search, ColaBO, LLAMBO Mode 2, LGBO and BORA (30 queries, 10 seeds) with the
prior `experiments/priors/cnn-leak/card-overfit/` and the ColaBO belief `experiments/colabo_llm_priors/cnn-leak/`:

```bash
python scripts/run_study_parallel.py experiments/studies/cnn-leak/cnn-leak-of-campaign --workers 4
python scripts/collect_paper_results.py --only cnn       # -> results/paper/cnn_leak/<arm>.json
python leak_control_cnn/build_regret_fig.py             # -> paper/figures/fig_leakctrl_regret.{pdf,png}
```

The prior is a one-shot elicitation through `scripts/elicit_prior.py` (Qwen3.5-122B-A10B-GPTQ-Int4, seed 7,
temperature 0, thinking off; generation notes in its `metadata.json`) with the ML-example vocabulary card
`src/prior/expert_knowledge/vocabulary_card_ml_example.md`, selected with the environment variable
`HAPBO_VOCABULARY_CARD`. Its `prompt_card.md` holds the domain text `elicitation/cnn_leak_domain_overfit.txt`, the
optimization text `elicitation/cnn_leak_opt.txt` and the four dimensions of `elicitation/cnn_leak_schema.json`.

## How it plugs into the harness

- Loader `src/datasets/cnn_leak.py` → `load_cnn_leak_problem()` returns an `OfflineOptimizationProblem`
  (same contract as the cortical/HPOBench adapters).
- Catalog `experiments/catalog/benchmarks/cnn-leak-of.yaml` (`adapter: cnn-leak`, `family: offline-finite`).
- Worker dispatch branch in `src/experiments/worker.py` (`adapter == "cnn-leak"`).

## Files

| file | role |
|---|---|
| `model.py` | the `ConvEncoderClassifier` |
| `prepare_data.py` | fetch + cache the fixed image subsample |
| `train_cnn.py` | train one configuration (all seeds) → `results/<dataset>/result_<id>.json` |
| `sweep_cnn.py` | train every configuration in parallel processes |
| `build_table.py` | aggregate results → `benchmark/cnn_leak/table_<dataset>.json` |
| `smoke_test.py` | wiring / environment check (`--with-training` for end to end) |
| `plot_cnn_arch_pnn.py` | architecture drawing, panel A of the CNN leak-control figure (needs PlotNeuralNet) |
| `build_regret_fig.py` | `fig_leakctrl_regret`, panel B: query-regret curves from `results/paper/cnn_leak/` |
| `elicitation/` | domain text (`cnn_leak_domain_overfit.txt`), optimization text (`cnn_leak_opt.txt`) and search-space schema (`cnn_leak_schema.json`) of the prior's card |
