"""Train ONE grid config of the leak-control CNN, all seeds, from scratch.

One call == one config (``--task-id`` in [0, 750)). It trains the config
for every seed in ``SEEDS``, records per-seed validation accuracy, and writes
``results/result_<task-id>.json`` with the seed-averaged mean and SD. Depends on
numpy + torch only; the image data comes from the cached ``.npz`` written by
prepare_data.py, so no internet/torchvision is needed during training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))

from datasets.cnn_leak import (  # noqa: E402
    DEFAULT_DATASET,
    SEEDS,
    config_key,
    enumerate_grid,
    regime_for,
)
from model import ConvEncoderClassifier  # noqa: E402  (sibling module)


def _load_data(dataset: str):
    npz = _ROOT / "benchmark" / "cnn_leak" / f"data_{dataset}.npz"
    if not npz.is_file():
        raise FileNotFoundError(
            f"{npz} missing. Run `python leak_control_cnn/prepare_data.py "
            f"--dataset {dataset}` first."
        )
    d = np.load(npz)
    return (
        torch.from_numpy(d["X_tr"]).float(), torch.from_numpy(d["y_tr"]).long(),
        torch.from_numpy(d["X_val"]).float(), torch.from_numpy(d["y_val"]).long(),
    )


def _train_one_seed(config, seed, data, *, epochs, batch) -> float:
    Xtr, ytr, Xval, yval = data
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = ConvEncoderClassifier(config["base_channels"], config["dropout"])
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    lossf = nn.CrossEntropyLoss()
    n = Xtr.shape[0]
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr[idx])
            if not torch.isfinite(loss):
                return 0.0  # divergence -> documented failure score
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        preds = model(Xval).argmax(1)
        return float((preds == yval).float().mean().item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, required=True, help="config index in [0, 750)")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--epochs", type=int, default=None, help="default: per-regime")
    parser.add_argument("--batch", type=int, default=None, help="default: per-regime")
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 -> OMP_NUM_THREADS, default 1)")
    parser.add_argument("--outdir", default=None, help="default: results/<dataset>/ (per-regime)")
    args = parser.parse_args()

    threads = args.threads or int(os.environ.get("OMP_NUM_THREADS", "1"))
    torch.set_num_threads(max(1, threads))

    reg = regime_for(args.dataset)
    epochs = args.epochs if args.epochs is not None else reg["epochs"]
    batch = args.batch if args.batch is not None else reg["batch"]

    grid = enumerate_grid()
    if not (0 <= args.task_id < len(grid)):
        raise SystemExit(f"--task-id {args.task_id} out of range [0, {len(grid)})")
    config = grid[args.task_id]
    key = config_key(config)

    outdir = Path(args.outdir) if args.outdir else (_HERE / "results" / args.dataset)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"result_{args.task_id:04d}.json"
    if out.is_file():
        print(f"skip task {args.task_id}: {out} exists")
        return 0

    data = _load_data(args.dataset)
    t0 = time.perf_counter()
    accs = [
        _train_one_seed(config, s, data, epochs=epochs, batch=batch)
        for s in SEEDS
    ]
    elapsed = time.perf_counter() - t0

    record = {
        "task_id": args.task_id,
        "dataset": args.dataset,
        "config": config,
        "config_key": key,
        "seed_accuracies": accs,
        "seeds": list(SEEDS),
        "mean": float(np.mean(accs)),
        "sd": float(np.std(accs)),
        "epochs": epochs,
        "batch": batch,
        "threads": threads,
        "train_time_sec": round(elapsed, 2),
    }
    tmp = out.with_suffix(".json.part")
    tmp.write_text(json.dumps(record), encoding="utf-8")
    tmp.replace(out)
    print(
        f"task {args.task_id:4d} {key} -> mean={record['mean']:.4f} "
        f"sd={record['sd']:.4f} ({elapsed:.0f}s, {threads} thr)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
