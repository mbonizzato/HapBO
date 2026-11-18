# Paper figures

`make_paper.py` regenerates every data figure of the paper from the compact run data committed under
`../results/paper/`. It needs only the Python dependencies of `../environment.yaml` (numpy, scipy, matplotlib);
no benchmark data, LLM or optimizer run is involved.

```bash
python paper/make_paper.py                     # all figures
python paper/make_paper.py --only hpo,synthetic
```

Outputs: `figures/*.{pdf,png}`, `figures/supp/*.{pdf,png}`, `tables/table_ladder_cost.tex` and `figures/numbers.json` (the plotted end-point values,
ranks and run counts).

## Generators

| module (`--only` key) | figures | data (`results/paper/`) |
|---|---|---|
| `fig_hpo.py` (`hpo`) | `fig2_hpo_compact`, `fig2_hpo_b16`, `fig5_adv_compact`, `fig5_robustness` | `hpo/{hapbo,hapbo-inverted,vanilla-random,colabo*,llambo*,lgbo*,bora*}.json`, `hpo/task_stats.json` |
| `fig_realworld.py` (`realworld`) | `fig3_realworld` | `cortical/exploration/human-map*.json`, `solar/*.json` |
| `fig_synthetic.py` (`synthetic`) | `fig4_synthetic_fidelity` | `synthetic/{rho90,well50,well10,none}.json` |
| `fig_prior_sources.py` (`prior_sources`) | `fig_prior_sources_main`, `supp/figS2_prior_sources_and_trajectories` | `cortical/exploration/{llm-geometry,llm-expert-hint,human-map}.json`, `cortical/prior_sources.json` |
| `fig_ablations.py` (`ablations`) | `fig_ablations` | `hpo/{hapbo,hapbo-misleading,injection*,vanilla-random}.json`, `synthetic/{well50,none}.json` |
| `fig_representation.py` (`representation`) | `fig_representation` | `hpo/{hapbo,hapbo-inverted,representation-*,vanilla-random}.json`, `hpo/task_stats.json`, `synthetic/{rho90,none}.json` |
| `fig_injection_synthetic.py` (`injection_synthetic`) | `fig_injection_synthetic`, `supp/figS_injection_synthetic_per_function` | `synthetic/{rho90,well50,well10,none}.json` |
| `fig_acquisition.py` (`acquisition`) | `fig_bounded_shaping`, `fig_acq_across`, `fig_batch_bo`, `fig_kernel_bo` | `hpo/{acquisition,batch,kernel-rbf,kernel-matern12}.json`, `cortical/{acquisition,batch,kernel-rbf,kernel-matern12}.json` |
| `fig_baseline_impl.py` (`baseline_impl`) | `fig_baseline_impl_sensitivity` | `hpo/{hapbo,hapbo-misleading,lgbo*,lgbo-official*,llambo*,llambo-mode1*,vanilla-random}.json` |
| `fig_cake_caliper.py` (`cake_caliper`) | `supp/figS_cake_vs_hapbo_b16` | `hpo/{hapbo,hapbo-misleading,caliper*,cake,vanilla-random}.json` |
| `fig_priorband_primo.py` (`priorband_primo`) | `supp/figS_priorband_primo_vs_hapbo_b16` | `hpo/{hapbo,hapbo-misleading,priorband*,primo*,vanilla-random}.json` |
| `fig_hypbo_llmei.py` (`hypbo_llmei`) | `supp/figS_hypbo_llmei_vs_hapbo_b16` | `hpo/{hapbo,hapbo-misleading,hypbo*,llm-ei*,vanilla-random}.json` |
| `fig_alpha_sweep.py` (`alpha_sweep`) | `fig_alpha_sweep_hapbo` | `hpo/{alpha-sweep,alpha-sweep-misleading,hapbo,hapbo-misleading,vanilla-random}.json`, `hpo/task_stats.json`, compiled alpha of the priors in `../experiments/priors/` |
| `../leak_control_cnn/build_regret_fig.py` (`leak_control_cnn`) | `fig_leakctrl_regret` | `cnn_leak/<arm>.json` |
| `fig_psl_elicitors.py` (`psl_elicitors`) | `fig_elicitation_ladder`, `tables/table_ladder_cost.tex` | `psl_elicitors/psl_elicitors.json`, documents in `../experiments/psl_elicitor_priors/` |

Shared modules: `style.py` (palette, rcParams, plotting helpers), `hpo_data.py`, `cortical_data.py`,
`solar_data.py`, `synthetic_data.py` (readers and metric definitions; each module docstring states the metric).

Other figures: the method diagram and the CNN architecture panel (`../leak_control_cnn/plot_cnn_arch_pnn.py`)
are drawings.

## Compact data (`results/paper/`)

| path | content |
|---|---|
| `hpo/<study>.json`, `cortical/<study>.json` | per study: `methods -> budget-<B> -> unit -> [{seed, status, y}]`, `y` = ground-truth objective of every query |
| `hpo/task_stats.json` | per HPOBench task: grid maximum, minimum, mean and optimum |
| `hpo/{priorband,primo}[-misleading].json` | multi-fidelity runs: `methods -> budget-0016 -> unit -> [{seed, status, y, cost_used, n_evaluations, repeat_units}]`, `y` = `[cumulative cost, ground-truth objective]` of every full-fidelity evaluation (x = cumulative full-fidelity-equivalent cost) |
| `cortical/map_optimum.json` | best true mean response per map |
| `cortical/exploration/<study>.json` | exploration performance per run (`scripts/score_exploration.py`) |
| `cortical/prior_sources.json` | compiled prior fields of the three cortical prior sources and three normalized truth maps |
| `solar/maxeff-re.json`, `solar/external.json` | SOLAR evaluations `[feasible, objective]` per seed (internal arms; ColaBO, LGBO, LLAMBO, BORA) |
| `synthetic/<card>.json`, `synthetic/none.json` | best-so-far simple-regret trajectories per arm and function |
| `cnn_leak/<arm>.json` | CNN leak-control trajectories; format in `cnn_leak/README.md` |
| `psl_elicitors/psl_elicitors.json` | PSL elicitors (5 elicitors x one-call / agentic x 4 functions): fidelity (`spearman_model`, `top10_model` per document), tokens and list-price US$ per elicitation, PSL token-type counts of the one-call documents, well-truth regret trajectories per seed (one-call: 3 draws x 10 seeds; agentic: 10 seeds; vanilla BO), the figure slice and the document paths |

Every cortical run holds 12 queries (budget 12 = `cortical_data.HORIZON`).

`<study>` is the name of the manifest in `experiments/studies/<group>/<study>/study.yaml`. After running the
campaigns (see `../REPRODUCING.md`), rebuild this folder with

```bash
python scripts/collect_paper_results.py      # reads results/studies, results/exploration_scores, results/diagnostics, results/synthetic
```

`fig_leakctrl_regret` arms: `results/paper/cnn_leak/<arm>.json` (format in `cnn_leak/README.md`), collected from
`results/studies/cnn-leak/cnn-leak-of-*` with `python scripts/collect_paper_results.py --only cnn`.
