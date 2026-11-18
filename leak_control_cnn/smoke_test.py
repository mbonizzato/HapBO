"""Fast validation of the leak-control CNN wiring (no heavy training).

Checks: grid size/encoding/keys, model forward + param counts, and the full
table->loader->OfflineOptimizationProblem round-trip against a synthetic table.
With --with-training it also trains 2 configs (1 epoch) on a tiny synthetic npz
to exercise train_cnn.py + build_table.py end to end. Prints PASS/FAIL.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

from datasets.cnn_leak import (  # noqa: E402
    PARAMETER_NAMES,
    config_key,
    enumerate_grid,
    encode_candidates,
    load_cnn_leak_problem,
    table_path,
)


def _check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return bool(cond)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-training", action="store_true")
    args = parser.parse_args()
    ok = True

    print("grid / encoding / keys")
    grid = enumerate_grid()
    keys = [config_key(c) for c in grid]
    enc = encode_candidates(grid)
    ok &= _check("grid has 750 configs", len(grid) == 750)
    ok &= _check("4 named dimensions", tuple(grid[0].keys()) == PARAMETER_NAMES)
    ok &= _check("keys unique", len(set(keys)) == 750)
    ok &= _check("encoding shape (750,4)", enc.shape == (750, 4))
    ok &= _check("encoding in [0,1]", enc.min() >= -1e-9 and enc.max() <= 1 + 1e-9)

    print("model forward / param counts")
    import torch
    from model import ConvEncoderClassifier, parameter_count
    m = ConvEncoderClassifier(8, 0.25)
    y = m(torch.randn(4, 1, 28, 28))
    ok &= _check("forward -> (4,10)", tuple(y.shape) == (4, 10))
    ok &= _check("params grow with width", parameter_count(48) > parameter_count(8))

    print("synthetic table -> loader round-trip")
    ds = "smoketest"
    rng = np.random.default_rng(0)
    table = {k: {"mean": float(rng.uniform(0.1, 0.9)), "sd": 0.01,
                 "seeds": [], "n_seeds": 3} for k in keys}
    tpath = table_path(ds)
    tpath.write_text(json.dumps(table), encoding="utf-8")
    try:
        prob = load_cnn_leak_problem(dataset=ds)
        ok &= _check("n_candidates == 750", prob.n_candidates == 750)
        ok &= _check("ground_truth len 750", prob.ground_truth.shape == (750,))
        ok &= _check("response_sd len 750", prob.response_standard_deviation.shape == (750,))
        ok &= _check("parameter_names match", tuple(prob.parameter_names) == PARAMETER_NAMES)
        ok &= _check("maximize True", prob.maximize is True)
        best = prob.optimum_candidate
        ok &= _check("objective(optimum) == optimum_value",
                     abs(prob.objective(best) - prob.optimum_value) < 1e-9)
        ok &= _check("optimum is grid argmax",
                     abs(prob.optimum_value - float(prob.ground_truth.max())) < 1e-9)
    finally:
        tpath.unlink(missing_ok=True)

    if args.with_training:
        print("train_cnn.py + build_table.py end to end (tiny, 1 epoch)")
        ds2 = "smoketrain"
        npz = _ROOT / "benchmark" / "cnn_leak" / f"data_{ds2}.npz"
        npz.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(1)
        np.savez_compressed(
            npz,
            X_tr=rng.standard_normal((256, 1, 28, 28)).astype(np.float32),
            y_tr=rng.integers(0, 10, 256).astype(np.int64),
            X_val=rng.standard_normal((128, 1, 28, 28)).astype(np.float32),
            y_val=rng.integers(0, 10, 128).astype(np.int64),
        )
        resdir = _HERE / "results_smoke"
        for tid in (0, 749):
            subprocess.run(
                [sys.executable, str(_HERE / "train_cnn.py"), "--task-id", str(tid),
                 "--dataset", ds2, "--epochs", "1", "--threads", "1",
                 "--outdir", str(resdir)],
                check=True,
            )
        wrote = list(resdir.glob("result_*.json"))
        ok &= _check("2 result files written", len(wrote) == 2)
        rec = json.loads((resdir / "result_0000.json").read_text())
        ok &= _check("result has mean in [0,1]", 0.0 <= rec["mean"] <= 1.0)
        ok &= _check("result has 3 seed accs", len(rec["seed_accuracies"]) == 3)
        # cleanup
        for f in wrote:
            f.unlink()
        resdir.rmdir()
        npz.unlink(missing_ok=True)

    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
