# Benchmark data

The benchmark payloads are not part of the repository: everything under `benchmark/` except this file is
gitignored. This page explains how to download or build each dataset, where the code expects it, and how
the loaders turn the raw files into the optimization problems used in the paper.

The figures can be regenerated without any of this data (`python paper/make_paper.py` reads the compact
run data in `results/paper/`). The data are needed only to run the campaigns (see `REPRODUCING.md`).

| Benchmark | Used in the paper for | Location | How to obtain |
| --- | --- | --- | --- |
| HPOBench tabular | 40 HPO tasks (`fig2_hpo_compact`, `fig2_hpo_b16`, `fig5_adv_compact`, `fig5_robustness`, `fig_ablations` panels A-B, `fig_representation`, `fig_alpha_sweep_hapbo`, `fig_bounded_shaping`, `fig_baseline_impl_sensitivity`, `supp/figS_cake_vs_hapbo_b16`, `supp/figS_priorband_primo_vs_hapbo_b16`, `supp/figS_hypbo_llmei_vs_hapbo_b16`) | `benchmark/HPOBench/` | `python scripts/download_hpobench.py` |
| NeuroMOSAIC Dataset 1, rat | cortical panel, 40 rat-muscle maps from 6 rats (`fig3_realworld` panel A, `fig_bounded_shaping`) and the 16 flexor maps (`fig_prior_sources_main`, `supp/figS2_prior_sources_and_trajectories`) | `benchmark/NeuroMOSAIC/dataset1/rat/` | manual download from OSF |
| SOLAR | MAXEFF_RE (`fig3_realworld` panel B) | `benchmark/SOLAR/` | `python scripts/setup_solar.py` (Linux) |
| CNN leak control | un-memorizable CNN grid (`fig_leakctrl_regret`) | `benchmark/cnn_leak/` | built locally with `leak_control_cnn/` |
| Synthetic functions | synthetic panel (`fig4_synthetic_fidelity`, `fig_ablations` panel C, `fig_representation`, `fig_injection_synthetic`, `supp/figS_injection_synthetic_per_function`, `fig_elicitation_ladder`, `tables/table_ladder_cost.tex`) | none | analytic, no download |

## HPOBench tabular

**Download.** `python scripts/download_hpobench.py` fetches the five official HPOBench tabular archives
(`lr`, `nn`, `rf`, `svm`, `xgb`; figshare URLs in `src/datasets/hpobench.py`, `DOWNLOAD_URLS`) and extracts
the eight OpenML classification tasks also used by LLAMBO. With no arguments it downloads all five models
and all eight tasks; `--model` and `--task` select a subset, `--verify-only` checks what is present, and
`--keep-archives` keeps the ZIP files.

| Task | OpenML task id |
| --- | --- |
| credit_g | 31 |
| vehicle | 53 |
| kc1 | 3917 |
| phoneme | 9952 |
| blood_transfusion | 10101 |
| australian | 146818 |
| car | 146821 |
| segment | 146822 |

**Layout.** `benchmark/HPOBench/<model>/<task_id>/<model>_<task_id>_data.parquet.gzip` and
`<model>_<task_id>_metadata.json` (one pair per model and task, 40 pairs).

**Processing** (`src/datasets/hpobench.py`, `load_hpobench_problem`):
- Only the rows at the maximum fidelity are kept, read from the task's fidelity ConfigSpace.
- The objective of a configuration is its validation accuracy (`result.info.val_scores.acc`), averaged over
  the benchmark seeds available for that configuration. A duplicate full-fidelity row for the same
  configuration and seed is an error.
- The candidate set is every configuration in the table at that fidelity. The direction is maximize; the
  optimum is the best mean validation accuracy in the table, and simple regret is measured against it.
- PriorBand and PriMO (`supp/figS_priorband_primo_vs_hapbo_b16`) also read the lower-fidelity rows of the same files
  (`src/datasets/hpobench_mf.py`); no separate download is needed.

## NeuroMOSAIC Dataset 1 (rat cortical stimulation)

**Download.** Get the published Dataset 1 `.mat` files from OSF (https://osf.io/54vhx/overview) and place the
six rat files in `benchmark/NeuroMOSAIC/dataset1/rat/`:

```
rat1_M1_190716.mat  rat2_M1_190617.mat  rat3_M1_190728.mat
rat4_M1_191109.mat  rat5_M1_191112.mat  rat6_M1_200218.mat
```

**Processing** (`src/datasets/neuromosaic_dataset1.py`):
- Each file holds one subject's M1 electrode array. `ch2xy` gives the electrode grid coordinates and fixes
  the candidate order. `sorted_resp` and `sorted_isvalid` hold the single-trial responses and their validity
  per electrode and muscle. `sorted_respMean` is the mean over valid trials.
- One optimization problem is one subject and one muscle. The baseline comparison and the BO variants use all
  40 rat-muscle maps, listed in `experiments/catalog/benchmarks/neuromosaic-dataset1-rat-all.yaml`. The
  prior-source comparison uses the 16 flexor maps listed in
  `experiments/catalog/benchmarks/neuromosaic-dataset1-rat-flexor.yaml`: left knee flexor and left tibialis
  anterior for rats 1–6, and right tibialis anterior for rats 1, 2, 3 and 6.
- The benchmark is noisy, as in the OptimizeNeurostim emulator. Each query returns one valid single trial,
  drawn uniformly at random from that electrode's trials. The draw uses common random numbers per electrode,
  seeded from the run seed, so arms that query the same electrode get the same trial.
- The ground truth of an electrode is its valid-trial mean (`sorted_respMean`). Regret and normalized
  performance are measured against the best mean on that map.

## SOLAR (concentrated solar power simulator)

**Setup (Linux).** `python scripts/setup_solar.py` clones the official simulator
(https://github.com/bbopt/solar.git) at the pinned commit `dda8fd7f98d4392aa07b3e3056888624a00712a0` into
`benchmark/SOLAR/source/`. It applies two small source patches (listed in `PATCHES` in the script), builds the
binary with `make` (`benchmark/SOLAR/source/bin/solar`) and verifies it. A C++ toolchain and `make` are
required. The binary does not run on Windows; the SOLAR tests skip there.

**Processing** (`src/datasets/solar.py`):
- The paper uses the single-objective instance `MAXEFF_RE` at fidelity 1.0 with one replication. The
  objective is receiver efficiency, maximized; the simulator returns its negative.
- Each run optimizes over a finite, seeded Latin-hypercube candidate pool of 256 configurations (row 0 is the
  official starting point). The pool is pre-screened with the simulator's own documented a-priori input
  relations (`APRIORI_CHECKS`), evaluated in Python without calling the simulator; unscreened pools are
  dominated by inputs the binary refuses.
- A refused input returns the sentinel `1e20` (a hidden failure, "crash"). A configuration is feasible when
  every constraint value is ≤ 0.
- The reported metric is the best feasible efficiency so far, 0 until the first feasible design.
- `python scripts/cache_solar_pools.py` evaluates complete pre-screened pools once and stores the objective and
  constraint values. Optimizers can then run from these tables on a machine without the binary; the
  SOLAR BORA runs use this route.

## CNN leak-control benchmark

The response surface of a small convolutional encoder over a 750-configuration grid (6 learning rates ×
5 base widths × 5 dropout rates × 5 weight decays), computed locally:

```
python leak_control_cnn/prepare_data.py --dataset fashion-of   # fixed image subsample (needs internet once)
python leak_control_cnn/sweep_cnn.py --dataset fashion-of      # train the grid
python leak_control_cnn/build_table.py --dataset fashion-of    # -> benchmark/cnn_leak/table_fashion-of.json
```

The table stores, per configuration, the validation accuracy averaged over training seeds
(`{mean, sd, seeds, n_seeds}`). `src/datasets/cnn_leak.py` loads it as a finite, known-optimum benchmark. The
benchmark uses the `fashion-of` regime (2k training images, 35 epochs, batch 64; `REGIMES` in
`src/datasets/cnn_leak.py`); see `experiments/catalog/benchmarks/cnn-leak-of.yaml`.

## Synthetic functions

Hartmann-6, Styblinski–Tang-7, Levy-10 and Ackley-15 need no data. The hidden objective of the synthetic
panel is defined in `scripts/synthetic_truth.py` (world `well`, the default; parameters in
`experiments/landscape_estimates/well_truth.json`), and the prior cards in `experiments/synthetic_cards/` and
`experiments/landscape_estimates/`.
