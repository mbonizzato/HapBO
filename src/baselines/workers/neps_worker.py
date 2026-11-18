"""NePS multi-fidelity worker (PriorBand / PriMO) for the cost-charged runner ``baselines.neps_mf``.

Runs inside the isolated NePS environment (official ``neural-pipeline-search==0.17.0``) and
drives NePS through its ask-and-tell interface; the parent process owns EVALUATE, the cost
ledger and the budget stop.

Protocol (line-delimited JSON):
  parent -> worker : one spec line, then per eval request either {"value": acc} or {"stop": true}
  worker -> parent : {"type": "info", ...} once, then {"type": "eval", ...} per asked trial,
                     {"type": "log", ...} for diagnostics, {"type": "done"} at the end

Search space: every tabular hyperparameter is an ``HPOInteger(0, n - 1)`` over the index of its
sorted grid values, so every configuration NePS proposes (sampled, mutated, promoted or acquired
by BO and decoded) is exactly one table row. The fidelity is the family's table fidelity with
its true bounds; NePS derives the rungs from (bounds, eta). The prior (one grid index +
confidence level per hyperparameter) is declared on the parameters (``prior`` /
``prior_confidence``) and, for PriMO, also passed as its single prior centre with the NePS
confidence mapping (low .25 / medium .5 / high .75).

NePS minimizes: the observed validation accuracy ``acc`` is reported as ``1 - acc`` (HPOBench's
official function value); PriMO takes the objective as a one-element vector ``[1 - acc]`` (its
scalarization asserts a sequence; with one objective the scalarization weight is 1).

RNG: Python ``random``, NumPy's global RNG and torch's global RNG are seeded from the run seed
before the optimizer is built (NePS draws from all three); torch uses ``torch_threads`` threads.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback

import harness  # noqa: F401  (captures the protocol stdout at import)
from harness import finish, log, read_spec


def _emit(obj: dict) -> None:
    harness._emit(obj)  # noqa: SLF001 -- the protocol channel captured by harness


def _plain(value):
    """JSON-safe scalar (NePS may hand back numpy / torch scalars)."""

    if hasattr(value, "item"):
        value = value.item()
    return value


def _build(spec: dict):
    from neps.optimizers import algorithms
    from neps.sampling.priors import PRIOR_CONFIDENCE_MAPPING
    from neps.space import HPOFloat, HPOInteger, SearchSpace

    algorithm = spec["algorithm"]
    prior = spec.get("prior")
    elements = {}
    for dim in spec["dimensions"]:
        name, n = dim["name"], int(dim["n"])
        if prior is not None:
            elements[name] = HPOInteger(0, n - 1, prior=int(prior["indices"][name]),
                                        prior_confidence=prior["confidence"][name])
        else:
            elements[name] = HPOInteger(0, n - 1)
    fid = spec["fidelity"]
    if fid["type"] == "int":
        elements[fid["name"]] = HPOInteger(int(fid["lower"]), int(fid["upper"]), is_fidelity=True)
    else:
        elements[fid["name"]] = HPOFloat(float(fid["lower"]), float(fid["upper"]), is_fidelity=True)
    space = SearchSpace(elements)
    kwargs = dict(spec.get("optimizer_kwargs") or {})
    if algorithm == "priorband":
        optimizer = algorithms.priorband(space, **kwargs)
        rung_to_fid = dict(optimizer.rung_to_fid)
    elif algorithm == "primo":
        if prior is None:
            raise ValueError("primo needs a prior")
        names = [dim["name"] for dim in spec["dimensions"]]
        centers = {"objective": {name: int(prior["indices"][name]) for name in names}}
        confidences = {"objective": {name: float(PRIOR_CONFIDENCE_MAPPING[prior["confidence"][name]])
                                     for name in names}}
        optimizer = algorithms.primo(space, prior_centers=centers, prior_confidences=confidences, **kwargs)
        rung_to_fid = dict(optimizer.bracket_optimizer.rung_to_fid)
    else:
        raise ValueError(f"unknown algorithm {algorithm!r}")
    return space, optimizer, {int(k): _plain(v) for k, v in rung_to_fid.items()}


def main() -> int:
    spec = read_spec()
    seed = int(spec["seed"])
    import numpy as np
    import torch

    torch.set_num_threads(int(spec.get("torch_threads", 1)))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)

    import neps

    algorithm = spec["algorithm"]
    _, optimizer, rung_to_fid = _build(spec)
    _emit({
        "type": "info",
        "neps_version": getattr(neps, "__version__", None) or _dist_version(),
        "algorithm": algorithm,
        "rung_to_fidelity": rung_to_fid,
        "optimizer_kwargs": spec.get("optimizer_kwargs") or {},
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    })
    ask_and_tell = neps.AskAndTell(optimizer)
    names = [dim["name"] for dim in spec["dimensions"]]
    fid_name = spec["fidelity"]["name"]
    while True:
        started = time.perf_counter()
        trial = ask_and_tell.ask()
        ask_seconds = time.perf_counter() - started
        config = trial.config
        raw = {name: _plain(config[name]) for name in names}
        if any(int(value) != value for value in raw.values()):
            raise ValueError(f"NePS proposed a non-integer grid index: {raw}")
        _emit({
            "type": "eval",
            "trial_id": trial.id,
            "previous_trial_id": trial.metadata.previous_trial_id,
            "indices": {name: int(value) for name, value in raw.items()},
            "fidelity": _plain(config[fid_name]),
            "ask_seconds": ask_seconds,
        })
        line = sys.stdin.readline()
        if not line:
            raise SystemExit("neps worker: parent closed the pipe")
        reply = json.loads(line)
        if reply.get("stop"):
            break
        loss = 1.0 - float(reply["value"])
        ask_and_tell.tell(trial, [loss] if algorithm == "primo" else loss)
    finish()
    return 0


def _dist_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("neural-pipeline-search")
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 -- surface the traceback to the parent's worker log
        log(traceback.format_exc())
        sys.stderr.write(traceback.format_exc())
        sys.stderr.flush()
        os._exit(1)
