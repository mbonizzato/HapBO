"""Build the fixed image subsample for the leak-control CNN sweep, ONCE.

It needs internet access to fetch the dataset once. It writes a compact ``.npz``
that every training process then loads with numpy only (no internet, no torchvision
needed inside the sweep).

The subsample (2k train / 2k val / 2k test, ``REGIMES`` in src/datasets/cnn_leak.py)
and its split are fixed by DATA_SPLIT_SEED so the whole 750-config grid sees an
identical objective.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from datasets.cnn_leak import (  # noqa: E402
    DATA_SPLIT_SEED,
    DEFAULT_DATASET,
    REGIMES,
    regime_for,
)

# OpenML dataset id of each image source in REGIMES.
_OPENML_IDS = {"fashion": 40996}


def _fetch(source: str):
    from sklearn.datasets import fetch_openml

    if source not in _OPENML_IDS:
        raise ValueError(f"unsupported image source {source!r}; choose from {sorted(_OPENML_IDS)}")
    bunch = fetch_openml(data_id=_OPENML_IDS[source], as_frame=False)
    X = np.asarray(bunch.data, dtype=np.float32)          # (70000, 784), 0..255
    y = np.asarray(bunch.target).astype(np.int64)          # string labels -> int
    return X, y


def data_path(dataset: str = DEFAULT_DATASET) -> Path:
    root = _ROOT / "benchmark" / "cnn_leak"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"data_{dataset}.npz"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=sorted(REGIMES))
    args = parser.parse_args()

    from sklearn.model_selection import train_test_split

    reg = regime_for(args.dataset)
    n_train, n_val, n_test = reg["n_train"], reg["n_val"], reg["n_test"]

    X, y = _fetch(reg["source"])
    X = (X / 255.0) * 2.0 - 1.0                            # -> [-1, 1]
    X = X.reshape(-1, 1, 28, 28).astype(np.float32)

    need = n_train + n_val + n_test
    # Fixed stratified subsample of `need` points, then split into tr/val/te.
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=need, random_state=DATA_SPLIT_SEED, stratify=y
    )
    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X_sub, y_sub, train_size=n_train, random_state=DATA_SPLIT_SEED, stratify=y_sub
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_rest, y_rest, train_size=n_val, random_state=DATA_SPLIT_SEED, stratify=y_rest
    )

    out = data_path(args.dataset)
    np.savez_compressed(
        out,
        X_tr=X_tr, y_tr=y_tr, X_val=X_val, y_val=y_val, X_te=X_te, y_te=y_te,
    )
    print(
        f"wrote {out}\n  train {X_tr.shape} val {X_val.shape} test {X_te.shape} "
        f"| classes {len(np.unique(y_tr))} | dtype {X_tr.dtype}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
