# External research implementations

Third-party research repositories are cloned below this directory and are excluded by
`.gitignore`; their source and history are not vendored. Reproducibility comes from the exact
revisions in `PINNED_COMMITS.txt`, the local source patch in `patches/`, and isolated runtime
environments. ColaBO does not declare a repository license, so do not copy or redistribute its code.

Clone and pin the baselines:

```bash
git clone https://github.com/tennisonliu/LLAMBO.git external/LLAMBO
git -C external/LLAMBO checkout 196fe237f60a3d3a2fa53cbf8f474ec20a01dd57

git clone https://github.com/hvarfner/colabo.git external/colabo
git -C external/colabo checkout 435121e0d3b0980ac04e08b8cfbf4473593a2342

git clone https://github.com/Ablatif6c/bora-the-explorer.git external/bora
git -C external/bora checkout 83b2ebf86f4f3dca0d65cd44c6c76f08ca9b0032
git -C external/bora apply ../patches/bora-local.patch

git clone https://github.com/Ablatif6c/HypBO.git external/hypbo
git -C external/hypbo checkout c3293fe8041c6d312e2476a6a93d0465f5f79981

git clone https://github.com/uber-research/TuRBO.git external/TuRBO
git -C external/TuRBO checkout de0db39f481d9505bb3610b7b7aa0ebf7702e4a5
```

Reference sources of the in-repository ports (not executed; the operators are reimplemented in
`src/baselines/`):

```bash
git clone https://github.com/Xinzhe309/LGBO.git external/LGBO          # official-lgbo lift operator
git -C external/LGBO checkout 01e56e1165a030ac74961f14d407bcc188f66ce6

git clone https://github.com/richardcsuwandi/cake.git external/cake     # CAKE
git -C external/cake checkout ece5cbc04b3e1d14d22a4fe80f38e7d16ff7c809
```

PriorBand and PriMO use the PyPI package `neural-pipeline-search==0.17.0` (no checkout) in their own
environment, selected with `NEPS_PYTHON` or `NEPS_ENV` (see `requirements-baselines.txt`).

Do not install any checkout into the main HapBO environment. Workers are selected with
`LLAMBO_PYTHON`, `COLABO_PYTHON`, `LGBO_PYTHON` (defaults to the LLAMBO interpreter),
`OFFICIAL_LGBO_PYTHON`, `BORA_PYTHON` and `HYPBO_PYTHON`, or with the corresponding Conda environment names
(`LLAMBO_ENV`, `COLABO_ENV`, `LGBO_ENV`, `BORA_ENV`, `HYPBO_ENV`). See `requirements-baselines.txt` and
`PINNED_COMMITS.txt` for the environment recipes and the documented deviations; the package pins of the
ColaBO environment are in `colabo_requirements.txt`. CAKE and TuRBO
run in-process and need `torch`, `gpytorch` (and `botorch` for CAKE) in the interpreter that runs
them (see `requirements-baselines.txt`).
