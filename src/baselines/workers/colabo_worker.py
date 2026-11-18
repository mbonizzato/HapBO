"""ColaBO-LogEI worker using the pinned upstream Ax and BoTorch forks.

The stock runner imports its optional PD1 adapter (and therefore ``mfpbench``)
unconditionally. This worker reconstructs the published ``colabo_logei`` Ax
GenerationStrategy directly from the available algorithm modules, then snaps
continuous Ax proposals to the nearest finite HapBO candidate.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from finite import nearest_index, normalize_rows  # noqa: E402
from harness import finish, log, make_evaluator, read_spec  # noqa: E402


def main() -> None:
    spec = read_spec()
    encoded = normalize_rows(spec["encoded"])
    budget = int(spec["budget"])
    n_init = max(1, min(int(spec["n_init"]), budget))
    seed = int(spec.get("seed") or 0)
    maximize = bool(spec.get("maximize", True))
    config = dict(spec.get("config") or {})
    evaluate = make_evaluator()
    n_total, dim = encoded.shape

    root = os.environ.get("COLABO_ROOT")
    if not root or not os.path.isdir(root):
        raise RuntimeError("COLABO_ROOT must point to the pinned upstream checkout")
    ax_root = os.path.join(root, "Ax")
    botorch_root = os.path.join(root, "botorch")
    if not os.path.isdir(ax_root) or not os.path.isdir(botorch_root):
        raise RuntimeError("COLABO_ROOT is missing its patched Ax or BoTorch tree")
    sys.path[0:0] = [ax_root, botorch_root]
    os.chdir(botorch_root)

    import numpy as np  # noqa: E402
    import torch  # noqa: E402
    from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy  # noqa: E402
    from ax.modelbridge.registry import Models  # noqa: E402
    from ax.models.torch.botorch_modular.surrogate import Surrogate  # noqa: E402
    from ax.service.ax_client import AxClient  # noqa: E402
    from ax.service.utils.instantiation import ObjectiveProperties  # noqa: E402
    from benchmarking.mapping.gp_priors import get_covar_module  # noqa: E402
    from botorch.acquisition.prior_monte_carlo import qPriorLogExpectedImprovement  # noqa: E402
    from botorch.models import SingleTaskGP  # noqa: E402
    from botorch.utils.prior import DefaultPrior  # noqa: E402

    torch.manual_seed(seed)
    np.random.seed(seed)
    bounds = torch.tensor([[0.0] * dim, [1.0] * dim], dtype=torch.float)
    prior_values = np.asarray(config["prior_values"], dtype=float).reshape(-1)
    if prior_values.shape != (n_total,) or not np.all(np.isfinite(prior_values)):
        raise ValueError("ColaBO worker received invalid candidate-aligned prior values")
    prior_index = int(np.argmax(prior_values))
    prior_location = torch.tensor(encoded[prior_index], dtype=torch.float)
    confidence = float(config.get("confidence", 0.2))
    user_prior = DefaultPrior(bounds, prior_location, confidence=confidence)

    gp_params = {
        "const": {"loc": 0.0, "scale": 1.0},
        "ls": {"loc": 0.0, "scale": 1.0},
        "ops": {"loc": 0.0, "scale": 1.0},
        "noise": {"loc": 0.0, "scale": 1.0},
    }
    model_options = get_covar_module(
        "singletask_lognormal",
        dim,
        prior_location=prior_location,
        gp_params=gp_params,
        gp_constraints={},
    )

    steps = [
        GenerationStep(
            model=Models.PRIOR,
            num_trials=1,
            model_kwargs={"seed": seed, "prior": user_prior},
        )
    ]
    if n_init > 1:
        steps.append(
            GenerationStep(
                model=Models.PRIOR,
                num_trials=n_init - 1,
                model_kwargs={
                    "seed": seed,
                    "prior": user_prior,
                    "generate_default": False,
                },
            )
        )
    # Continuous proposals can collide after finite snapping. Give upstream a
    # bounded number of additional internal trials; duplicate objective requests
    # are served from the bridge cache and charged as repeat queries.
    # The ceiling bounds PROPOSALS, not unique evaluations: with a sharp prior on
    # a small grid (e.g. the 32-electrode cortical grid), collisions legitimately
    # exceed n_total, so the runaway guard is a multiple of n_total (16 * n_total).
    max_internal_trials = min(
        max(budget, int(config.get("max_internal_trials", 3 * budget))),
        16 * n_total,
    )
    num_bo = max_internal_trials - n_init
    if num_bo > 0:
        acquisition_options = {
            "user_prior": user_prior,
            "decay_beta": float(budget),
            "num_samples": int(config.get("num_samples", 76_800)),
            "resampling_fraction": float(config.get("resampling_fraction", 0.0067)),
        }
        optimizer_options = {
            "raw_samples": int(config.get("raw_samples", 1_024)),
            "num_restarts": int(config.get("num_restarts", 20)),
            "retry_on_optimization_warning": False,
            "options": {
                "nonnegative": False,
                "sample_around_best": True,
                "sample_around_best_sigma": 0.1,
                "maxiter": 100,
                "batch_limit": 48,
            },
        }
        steps.append(
            GenerationStep(
                model=Models.BOTORCH_MODULAR,
                num_trials=num_bo,
                model_kwargs={
                    "surrogate": Surrogate(
                        botorch_model_class=SingleTaskGP,
                        model_options=model_options,
                    ),
                    "botorch_acqf_class": qPriorLogExpectedImprovement,
                    "acquisition_options": acquisition_options,
                    "refit_on_update": True,
                    "refit_params": {},
                },
                model_gen_kwargs={
                    "model_gen_options": {"optimizer_kwargs": optimizer_options}
                },
            )
        )

    client = AxClient(generation_strategy=GenerationStrategy(steps=steps))
    client.create_experiment(
        name=f"hapbo_colabo_{seed}",
        parameters=[
            {"name": f"x_{column + 1}", "type": "range", "bounds": [0.0, 1.0]}
            for column in range(dim)
        ],
        objectives={"objective": ObjectiveProperties(minimize=False)},
    )
    log(
        f"ColaBO-LogEI prior_index={prior_index} confidence={confidence} "
        f"n_init={n_init} max_internal_trials={max_internal_trials}"
    )
    snapped_indices: set[int] = set()
    for _ in range(max_internal_trials):
        parameters, trial_index = client.get_next_trial()
        point = [float(parameters[f"x_{column + 1}"]) for column in range(dim)]
        index = nearest_index(encoded, point)
        snapped_indices.add(index)
        value = float(evaluate(index))
        internal_value = value if maximize else -value
        client.complete_trial(trial_index=trial_index, raw_data={"objective": internal_value})
        if len(snapped_indices) >= budget:
            break
    finish()


if __name__ == "__main__":
    main()
