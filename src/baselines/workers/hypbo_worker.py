"""HypBO worker (runs in the isolated `hypbo` env).

Runs the pinned upstream HypBO (Ablatif6c/HypBO) in the configuration of its synthetic-function
experiments (main_continuous.py): DBO ``BayesianOptimization`` over the continuous encoded box,
``discretization=False``, model_kwargs ``{"pbounds", "random_seed"}``, the default failure limits
(global 5, local 2) and gamma 0.  Every point HypBO evaluates (its per-hypothesis initial samples,
its global initial samples and every search proposal) is snapped to the nearest finite candidate
in the encoded space and served by the parent through the bridge callback.  The hypotheses are
the boxes computed by the parent (BORA supplement C.3), passed to HypBO in upstream's box form.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # harness, finite, box_hypotheses

import numpy as np  # noqa: E402

from box_hypotheses import box_inequalities  # noqa: E402
from finite import nearest_index  # noqa: E402
from harness import finish, log, make_evaluator, read_spec  # noqa: E402


def main() -> None:
    spec = read_spec()
    encoded = np.asarray(spec["encoded"], dtype=float)
    budget = int(spec["budget"])
    n_init = max(1, int(spec["n_init"]))
    seed = int(spec.get("seed") or 0)
    maximize = bool(spec.get("maximize", True))
    config = dict(spec.get("config") or {})
    evaluate = make_evaluator()
    n_candidates, dim = encoded.shape
    if dim > 10:
        # DBO sorts the parameter names and maps x<k> to array column k; from x10 on the sorted
        # order and that mapping disagree.
        raise ValueError("the HypBO worker supports at most 10 encoded dimensions")

    root = os.environ.get("HYPBO_ROOT")
    if not root or not os.path.isfile(os.path.join(root, "hypbo.py")):
        raise RuntimeError("HYPBO_ROOT must point to the pinned upstream checkout")
    sys.path.insert(0, root)
    os.chdir(root)

    from DBO.bayes_opt.bayesian_optimization import BayesianOptimization  # noqa: E402
    from hypbo import HypBO  # noqa: E402
    from hypothesis import Hypothesis  # noqa: E402

    lower = encoded.min(axis=0)
    upper = encoded.max(axis=0)
    feature_names = [f"x{j}" for j in range(dim)]
    pbounds = {name: (float(lower[j]), float(upper[j])) for j, name in enumerate(feature_names)}

    hypotheses = []
    for k, box in enumerate(config.get("hypotheses") or []):
        lo = np.asarray(box["lb"], dtype=float)
        hi = np.asarray(box["ub"], dtype=float)
        if lo.shape != (dim,) or hi.shape != (dim,):
            raise ValueError(f"hypothesis {k} does not match the encoded dimension {dim}")
        coeff_inequalities, const_inequalities = box_inequalities(lo, hi)
        hypotheses.append(
            Hypothesis(
                name=str(box.get("name", f"H{k + 1}")),
                lb=lo,
                ub=hi,
                feature_names=feature_names,
                coeff_inequalities=coeff_inequalities,
                const_inequalities=const_inequalities,
                coeff_equalities=None,
                const_equalities=None,
            )
        )

    def func(sample) -> float:
        # HypBO passes a 1-D array in the model's sorted key order (x0..x{d-1}).
        index = nearest_index(encoded, np.asarray(sample, dtype=float).reshape(-1))
        value = float(evaluate(index))
        return value if maximize else -value  # HypBO maximizes

    settings = {
        "n_processes": int(config.get("n_processes", 1)),
        "discretization": False,
        "global_failure_limit": int(config.get("global_failure_limit", 5)),
        "local_failure_limit": int(config.get("local_failure_limit", 2)),
        "gamma": float(config.get("gamma", 0.0)),
    }
    batch = int(config.get("batch", 1))
    log(
        f"HypBO dim={dim} n_candidates={n_candidates} budget={budget} n_init={n_init} "
        f"hypotheses={len(hypotheses)} settings={settings} batch={batch}"
    )
    for hypothesis in hypotheses:
        log(f"{hypothesis.name}: sol_bounds lb={np.round(hypothesis.sol_bounds[0], 6).tolist()} "
            f"ub={np.round(hypothesis.sol_bounds[1], 6).tolist()}")

    hbo = HypBO(
        func=func,
        feature_names=feature_names,
        model=BayesianOptimization,
        model_kwargs={"pbounds": pbounds, "random_seed": seed},
        hypotheses=hypotheses,
        seed=seed,
        **settings,
    )
    # HypBO evaluates its initial design and then `budget` search iterations; the parent stops it
    # once the charged budget is used, so the run never ends short of the budget.
    hbo.search(budget=budget, n_init=n_init, batch=batch)
    finish()


if __name__ == "__main__":
    main()
