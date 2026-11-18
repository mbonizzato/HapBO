from __future__ import annotations

import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import main as main_module  # noqa: E402
from bayesian_optimization import BayesianOptimization, probability_of_feasibility  # noqa: E402
from datasets.solar import (  # noqa: E402
    SOLAR_INSTANCES,
    SolarEvaluation,
    apriori_valid,
    evaluate_solar,
    generate_solar_candidate_pool,
)


def _write_linear_prior(path: Path, spec) -> Path:
    """A flat test prior over the instance's box, in native simulator order."""

    dimensions = [
        {
            "name": variable.name,
            "type": "continuous",
            "range": [variable.lower, variable.sampling_upper],
            "scale": "linear",
            "meaning": variable.description,
        }
        for variable in spec.variables
    ]
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema": {"problem": "SOLAR test", "objective": "minimize", "dimensions": dimensions},
                "beliefs": [{"token": "baseline", "level": 0.5}],
                "trust": {"overall_confidence": "medium"},
            }
        ),
        encoding="utf-8",
    )
    return path


class SolarTests(unittest.TestCase):
    def test_selected_registry_has_expected_dimensions(self) -> None:
        expected = {
            "MAXNRG_H1": (9, 5),
            "MINSURF_H1": (14, 12),
            "MINCOST_C1": (20, 13),
            "MINCOST_C2": (29, 16),
            "MAXCOMP_HTF1": (20, 12),
            "MINCOST_TS": (5, 6),
            "MAXEFF_RE": (7, 6),
            "MINCOST_UNCONSTRAINED": (5, 0),
            "MINCOST_CH": (31, 16),
        }
        self.assertEqual(
            {name: (spec.n_dimensions, spec.n_constraints) for name, spec in SOLAR_INSTANCES.items()},
            expected,
        )
        for spec in SOLAR_INSTANCES.values():
            self.assertEqual(len({variable.name for variable in spec.variables}), spec.n_dimensions)

    def test_candidate_pool_starts_with_official_x0(self) -> None:
        candidates = generate_solar_candidate_pool("MAXEFF_RE", 8, seed=7)
        spec = SOLAR_INSTANCES["MAXEFF_RE"]
        self.assertEqual(
            candidates[0],
            {variable.name: variable.initial for variable in spec.variables},
        )
        self.assertEqual(candidates, generate_solar_candidate_pool("MAXEFF_RE", 8, seed=7))

    @unittest.skipIf(sys.platform == "win32", "the executable fixture is a POSIX shell script")
    def test_live_adapter_parses_objective_and_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "solar"
            executable.write_text(
                "#!/usr/bin/env python3\nprint('-12.5 -1 0 2 -3 -4')\n",
                encoding="utf-8",
            )
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
            candidate = generate_solar_candidate_pool("MAXNRG_H1", 2)[0]
            result = evaluate_solar("MAXNRG_H1", candidate, executable=executable)
        self.assertEqual(result.objective, -12.5)
        np.testing.assert_array_equal(result.constraints, [-1, 0, 2, -3, -4])
        self.assertFalse(result.feasible)

    def test_constrained_bo_prefers_feasible_best_observation(self) -> None:
        optimizer = BayesianOptimization(
            np.arange(3, dtype=float).reshape(-1, 1),
            maximize=False,
            n_constraints=1,
            n_initial=1,
            random_state=0,
        )
        optimizer.tell(0, 1.0, np.array([2.0]))
        optimizer.tell(1, 10.0, np.array([-1.0]))
        optimizer.tell(2, 5.0, np.array([0.0]))
        self.assertEqual(optimizer.best_observation()[0], 2)
        self.assertEqual(optimizer.result().constraint_values.shape, (3, 1))
        probability = probability_of_feasibility(
            np.array([-1.0, 1.0]), np.array([0.0, 0.0])
        )
        np.testing.assert_array_equal(probability, [1.0, 0.0])

    @unittest.skipIf(sys.platform == "win32", "the executable fixture is a POSIX shell script")
    def test_solar_cli_runs_against_a_live_executable_fixture(self) -> None:
        spec = SOLAR_INSTANCES["MAXEFF_RE"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "solar"
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "from pathlib import Path\n"
                "values = [float(x) for x in Path(__import__('sys').argv[2]).read_text().split()]\n"
                f"print(sum(values), *([-1] * {spec.n_constraints}))\n",
                encoding="utf-8",
            )
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
            prior_path = _write_linear_prior(root / "expert" / "prior.json", spec)
            args = main_module.build_parser().parse_args(
                [
                    "solar-bo",
                    "--solar-instance",
                    spec.name,
                    "--solar-executable",
                    str(executable),
                    "--solar-candidate-pool-size",
                    "4",
                    "--dataset-prior-file",
                    str(prior_path),
                    "--queries",
                    "2",
                    "--quiet",
                ]
            )
            with mock.patch.object(main_module, "PROJECT_ROOT", root):
                result = main_module.run_solar(args)
        self.assertEqual(result.y.size, 2)

    def test_solar_cli_follows_the_study_worker_path(self) -> None:
        # Pre-screened pool, crashes modeled as hidden failures, and query 1 from ask()
        # (the seeded initial design), as in experiments.worker._run_solar.
        spec = SOLAR_INSTANCES["MAXEFF_RE"]
        pool_size, seed, n_initial = 16, 3, 2
        expected_first = BayesianOptimization(
            np.zeros((pool_size, 1)), n_initial=n_initial, random_state=seed
        ).ask()[0]
        self.assertNotEqual(expected_first, 0)  # row 0 is the official starting point
        evaluated: list[dict] = []

        def fake_evaluate(instance, candidate, **kwargs):
            evaluated.append(dict(candidate))
            crashed = len(evaluated) == 2
            return SolarEvaluation(
                objective=1e20 if crashed else -float(len(evaluated)),
                constraints=np.full(spec.n_constraints, 1e20 if crashed else -1.0),
                feasible=not crashed,
                hidden_failure=crashed,
                raw_output="",
            )

        optimizers: list[BayesianOptimization] = []

        class RecordingOptimization(BayesianOptimization):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                optimizers.append(self)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "solar"
            executable.write_text("not executed\n", encoding="utf-8")
            prior_path = _write_linear_prior(root / "expert" / "prior.json", spec)
            args = main_module.build_parser().parse_args(
                [
                    "solar-bo",
                    "--solar-executable", str(executable),
                    "--solar-candidate-pool-size", str(pool_size),
                    "--dataset-prior-file", str(prior_path),
                    "--queries", "4",
                    "--initial", str(n_initial),
                    "--seed", str(seed),
                    "--quiet",
                ]
            )
            with (
                mock.patch.object(main_module, "PROJECT_ROOT", root),
                mock.patch.object(main_module, "evaluate_solar", fake_evaluate),
                mock.patch.object(main_module, "BayesianOptimization", RecordingOptimization),
            ):
                result = main_module.run_solar(args)

        self.assertEqual(args.solar_instance, "MAXEFF_RE")
        self.assertEqual(int(result.evaluated_indices[0]), expected_first)
        self.assertTrue(all(apriori_valid("MAXEFF_RE", candidate) for candidate in evaluated))
        (optimizer,) = optimizers
        self.assertTrue(optimizer.model_hidden_failures)
        self.assertEqual(optimizer.observed_crashes, [False, True, False, False])
        self.assertEqual(result.y[1], 0.0)  # the crash placeholder, never fitted
        self.assertNotEqual(result.best_index, int(result.evaluated_indices[1]))


if __name__ == "__main__":
    unittest.main()
