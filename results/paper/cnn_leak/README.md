# CNN leak-control trajectories (`fig_leakctrl_regret`)

One JSON file per arm, read by `leak_control_cnn/build_regret_fig.py`:

| file | arm | study manifest (`experiments/studies/cnn-leak/`) |
|---|---|---|
| `hapbo.json`, `vanilla.json`, `random.json` | HapBO, vanilla BO, random search | `cnn-leak-of-campaign` |
| `colabo.json` | ColaBO | `cnn-leak-of-colabo` |
| `llambo.json` | LLAMBO (Mode 2, zero-shot warm start) | `cnn-leak-of-llambo-ws` |
| `bora.json` | BORA | `cnn-leak-of-bora` |
| `lgbo.json` | LGBO (paper equation) | `cnn-leak-of-lgbo` |

Format (written by `python scripts/collect_paper_results.py --only cnn` from
`results/studies/cnn-leak/<study>/runs/`):

```json
{"optimum": 0.8608333468437195, "budget": 30, "runs": [[<ground-truth validation accuracy at query 1..30>], ...]}
```
