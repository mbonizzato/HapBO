from __future__ import annotations

import json
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
from src.baselines import BASELINE_REGISTRY, get_baseline  # noqa: E402
from src.baselines.runner import run_baseline_experiment  # noqa: E402
from src.llm import LLMClient  # noqa: E402


class BaselineRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.search_space = {
            "candidates": tuple({"x": float(x)} for x in range(7)),
            "encoded_candidates": np.arange(7, dtype=float).reshape(-1, 1),
        }
        self.objective_meta = {"maximize": True, "n_constraints": 0}

    def test_registry_contains_all_core_keys(self) -> None:
        # The core arms are always registered; "cake" is registered only where the GP
        # stack (torch/botorch/gpytorch) is installed, so the test checks the core set.
        core = {"random", "vanilla_bo", "llambo", "colabo", "bora", "hypbo", "caliper", "lgbo", "official_lgbo"}
        self.assertTrue(core.issubset(set(BASELINE_REGISTRY)))

    def test_unknown_baseline_lists_valid_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "random.*vanilla_bo.*llambo"):
            get_baseline(
                "nope",
                search_space=self.search_space,
                objective_meta=self.objective_meta,
            )

    def test_import_does_not_require_external_submodules(self) -> None:
        # The registry must import and expose every baseline WITHOUT importing the
        # external upstreams (they are optional, gitignored local checkouts used
        # only at ask/tell time via a subprocess bridge).
        for name in ("llambo", "colabo", "lgbo"):
            self.assertIn(name, BASELINE_REGISTRY)
        self.assertNotIn("botorch", sys.modules)  # heavy upstream deps not imported

    def test_lgbo_is_configured_as_an_isolated_worker(self) -> None:
        # LGBO is a from-paper reimplementation driven through the subprocess bridge
        # (region-lifted GP mean shift). It resolves to its isolated worker, exposes its
        # region-lift config and takes no prior (requires_prior: false; it generates its
        # own guidance in the loop).
        optimizer = get_baseline(
            "lgbo",
            search_space=self.search_space,
            objective_meta=self.objective_meta,
            seed=3,
            budget=5,
        )
        self.assertIn("lgbo_worker.py", optimizer.resolved_command()[-1])
        self.assertEqual(optimizer.worker_config()["n_grid"], 64)

    def test_llambo_is_configured_as_an_isolated_worker(self) -> None:
        optimizer = get_baseline(
            "llambo",
            search_space=self.search_space,
            objective_meta=self.objective_meta,
            seed=3,
            budget=5,
        )
        self.assertIn("llambo_worker.py", optimizer.resolved_command()[-1])
        self.assertEqual(optimizer.worker_config()["n_candidates"], 10)

    def test_colabo_requires_and_serializes_an_aligned_prior(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate-aligned prior"):
            get_baseline(
                "colabo",
                search_space=self.search_space,
                objective_meta=self.objective_meta,
                seed=3,
                budget=5,
            ).worker_config()
        optimizer = get_baseline(
            "colabo",
            search_space=self.search_space,
            objective_meta=self.objective_meta,
            prior={"values": np.arange(7, dtype=float), "confidence": 0.3},
            seed=3,
            budget=5,
        )
        self.assertIn("colabo_worker.py", optimizer.resolved_command()[-1])
        self.assertEqual(optimizer.worker_config()["confidence"], 0.3)


class BaselineRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = tuple({"x": float(x)} for x in range(7))
        self.search_space = {
            "candidates": self.candidates,
            "encoded_candidates": np.arange(7, dtype=float).reshape(-1, 1),
        }

    @staticmethod
    def objective(config: dict[str, float]) -> float:
        return -((config["x"] - 4.0) ** 2)

    def test_random_and_vanilla_run_and_write_expected_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for method in ("random", "vanilla_bo"):
                with self.subTest(method=method):
                    result = run_baseline_experiment(
                        method,
                        search_space=self.search_space,
                        objective=self.objective,
                        objective_meta={"maximize": True},
                        dataset_path="tiny-dataset",
                        prior_id="shared-prior",
                        seed=9,
                        budget=5,
                        n_init=1,
                        results_root=directory,
                    )
                    expected = (
                        Path(directory)
                        / "tiny-dataset"
                        / method
                        / "shared-prior"
                        / "seed-0009"
                    ).resolve()
                    self.assertEqual(result.output_directory, expected)
                    self.assertEqual(len(result.history), 5)
                    self.assertEqual(len(set(result.optimization.evaluated_indices)), 5)
                    self.assertTrue((expected / "run_config.json").is_file())
                    self.assertTrue((expected / "metrics.json").is_file())
                    self.assertEqual(
                        len((expected / "evaluations.jsonl").read_text().splitlines()),
                        5,
                    )

    def test_main_method_routes_random_and_vanilla_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            real_runner = run_baseline_experiment

            def temporary_runner(*args, **kwargs):
                kwargs["results_root"] = temporary_root / "results"
                return real_runner(*args, **kwargs)

            with mock.patch.object(
                main_module,
                "run_baseline_experiment",
                side_effect=temporary_runner,
            ):
                for method in ("random", "vanilla_bo"):
                    with self.subTest(method=method):
                        exit_code = main_module.main(
                            [
                                "synthetic-bo",
                                "--method",
                                method,
                                "--queries",
                                "5",
                                "--initial",
                                "1",
                                "--quiet",
                            ]
                        )
                        self.assertEqual(exit_code, 0)
                        output = (
                            temporary_root
                            / "results"
                            / "synthetic-bo"
                            / method
                            / "no-prior"
                            / "seed-0042"
                        )
                        self.assertTrue((output / "run_config.json").is_file())
                        self.assertTrue((output / "metrics.json").is_file())


class _FakeProvider:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, **kwargs):
        from src.llm.providers import Completion

        self.calls += 1
        return Completion(text=f"resp:{prompt}", tokens=7, cost=0.5)


class LLMClientTests(unittest.TestCase):
    def test_cache_budget_and_result_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = LLMClient("test-model", 0.1, 100, root / "cache", budget=1)
            client.set_run_directory(root / "results")
            fake = _FakeProvider()
            client._provider = fake  # inject provider so no network call happens

            # First call executes the provider, then stores + logs it.
            self.assertEqual(client.complete("prompt A", seed=1), "resp:prompt A")
            self.assertEqual(client.call_count, 1)
            self.assertEqual(fake.calls, 1)
            # Identical arguments -> cache hit, no provider call, no budget use.
            self.assertEqual(client.complete("prompt A", seed=1), "resp:prompt A")
            self.assertEqual(fake.calls, 1)
            # A different prompt needs a new call, but the budget is spent.
            with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                client.complete("prompt B", seed=1)

            records = [
                json.loads(line)
                for line in (root / "results" / "llm_calls.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(records[0]["status"], "ok")
            self.assertEqual(records[0]["tokens"], 7)
            self.assertEqual(records[-1]["status"], "cache_hit")

    def test_identity_fields_change_the_cache_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = LLMClient("m", 0.0, 10, Path(directory) / "c", budget=0)

            def key(**over):
                base = dict(
                    system=None, response_schema=None, seed=1, elicitation_replicate=0, extra={}
                )
                base.update(over)
                return client.prompt_hash("p", **client._identity(**base))

            reference = key()
            self.assertNotEqual(reference, key(seed=2))
            self.assertNotEqual(reference, key(elicitation_replicate=1))
            self.assertNotEqual(reference, key(response_schema={"type": "object"}))


if __name__ == "__main__":
    unittest.main()
