# Dataset adapter organization

Dataset modules expose finite native configuration dictionaries, aligned
numeric candidate arrays where appropriate, and objective callables that reveal
one observation at a time to the outer optimization loop.

NeuroMOSAIC adapters mirror the benchmark release hierarchy explicitly:

```text
neuromosaic_common.py          shared OfflineOptimizationProblem contract
neuromosaic_dataset1.py        shared MATLAB parser for cortical Dataset 1 files
neuromosaic_dataset1_rat.py    Dataset 1 rat registry, defaults, and loader
```

The rat module delegates parsing and validation to
`load_neuromosaic_dataset1_problem` and depends on the neutral problem type in
`neuromosaic_common.py`.

Use the explicit Dataset 1 module and public names such as
`load_neuromosaic_dataset1_rat_problem`.

The other adapters are `hpobench.py` (HPOBench tabular, full fidelity),
`hpobench_mf.py` (the lower-fidelity rows read by PriorBand and PriMO), `solar.py`
(SOLAR simulator) and `cnn_leak.py` (CNN leak-control table). `benchmark/README.md`
describes the data each one reads.
