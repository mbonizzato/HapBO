from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from main import build_parser, main  # noqa: E402
from datasets.hpobench import (  # noqa: E402
    MODELS,
    TASKS,
    download_hpobench_data,
    hpobench_search_space_summary,
    load_hpobench_problem,
)


class HPOBenchTests(unittest.TestCase):
    def _metadata(self) -> dict:
        x_discrete = {
            "hyperparameters": [
                {"name": "max_depth", "type": "ordinal", "sequence": [1, 2]},
                {
                    "name": "max_features",
                    "type": "ordinal",
                    "sequence": [0.25, 1.0],
                },
            ]
        }
        x_original = {
            "hyperparameters": [
                {"name": "max_depth", "type": "uniform_int", "log": True},
                {"name": "max_features", "type": "uniform_float", "log": False},
            ]
        }
        z_discrete = {
            "hyperparameters": [
                {
                    "name": "n_estimators",
                    "type": "ordinal",
                    "sequence": [16, 512],
                },
                {
                    "name": "subsample",
                    "type": "ordinal",
                    "sequence": [0.5, 1.0],
                },
            ]
        }
        return {
            "config_spaces": {
                "x": json.dumps(x_original),
                "x_discrete": json.dumps(x_discrete),
                "z_discrete": json.dumps(z_discrete),
            }
        }

    def _write_task_files(self, root: Path) -> tuple[Path, Path]:
        directory = root / "rf" / "31"
        directory.mkdir(parents=True)
        parquet_path = directory / "rf_31_data.parquet.gzip"
        metadata_path = directory / "rf_31_metadata.json"
        rows = []
        for max_depth, max_features, accuracies in (
            (1, 0.25, (0.70, 0.74)),
            (2, 1.0, (0.80, 0.84)),
        ):
            for seed, accuracy in zip((11, 22), accuracies):
                rows.append(
                    {
                        "max_depth": max_depth,
                        "max_features": max_features,
                        "n_estimators": 512,
                        "subsample": 1.0,
                        "seed": seed,
                        "result": {"info": {"val_scores": {"acc": accuracy}}},
                    }
                )
            rows.append(
                {
                    "max_depth": max_depth,
                    "max_features": max_features,
                    "n_estimators": 16,
                    "subsample": 0.5,
                    "seed": 11,
                    "result": {"info": {"val_scores": {"acc": 0.99}}},
                }
            )
        pd.DataFrame(rows).to_parquet(parquet_path)
        metadata_path.write_text(json.dumps(self._metadata()), encoding="utf-8")
        return parquet_path, metadata_path

    def _write_prior(self, path: Path) -> None:
        prior = {
            "schema": {
                "problem": "Synthetic HPOBench fixture",
                "objective": "maximize",
                "dimensions": [
                    {
                        "name": "max_depth",
                        "type": "ordinal",
                        "values": [1, 2],
                        "meaning": "tree depth",
                    },
                    {
                        "name": "max_features",
                        "type": "ordinal",
                        "values": [0.25, 1.0],
                        "meaning": "feature fraction",
                    },
                ],
            },
            "beliefs": [
                {"token": "baseline", "level": 0.5},
                {"token": "smoothness", "level": "medium"},
                {"token": "noise", "level": "medium"},
                {"token": "modality", "expectation": "few"},
            ],
            "trust": {"overall_confidence": "low", "notes": "test fixture"},
        }
        path.write_text(json.dumps(prior), encoding="utf-8")

    def test_registry_matches_paper_models_and_llambo_tasks(self) -> None:
        self.assertEqual(MODELS, ("lr", "nn", "rf", "svm", "xgb"))
        self.assertEqual(TASKS["credit_g"], 31)
        self.assertEqual(TASKS["segment"], 146822)
        self.assertEqual(len(TASKS), 8)

    def test_loader_uses_full_fidelity_and_averages_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_task_files(root)
            problem = load_hpobench_problem(
                model="rf",
                task="credit_g",
                benchmark_directory=root,
            )

        self.assertEqual(problem.n_candidates, 2)
        self.assertEqual(problem.metadata["full_fidelity"], {
            "n_estimators": 512,
            "subsample": 1.0,
        })
        np.testing.assert_allclose(np.sort(problem.ground_truth), [0.72, 0.82])
        self.assertAlmostEqual(problem.optimum_value, 0.82)
        self.assertAlmostEqual(
            problem.objective({"max_depth": 2, "max_features": 1.0}),
            0.82,
        )

    def test_downloader_extracts_only_selected_task_and_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            parquet_path, metadata_path = self._write_task_files(source_root)
            archive_path = root / "source-rf.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.write(parquet_path, "31/rf_31_data.parquet.gzip")
                archive.write(metadata_path, "31/rf_31_metadata.json")

            def opener(_request):
                return archive_path.open("rb")

            output_root = root / "output"
            manifest = download_hpobench_data(
                models=("rf",),
                tasks=("credit_g",),
                benchmark_directory=output_root,
                opener=opener,
            )
            summary = hpobench_search_space_summary(
                "rf",
                "credit_g",
                benchmark_directory=output_root,
            )

            self.assertEqual(len(manifest["extracted"]), 1)
            self.assertFalse((output_root / "rf" / "rf.zip").exists())
            self.assertTrue((output_root / "download_manifest.json").is_file())
            self.assertEqual(summary["dimensions"][0]["name"], "max_depth")
            self.assertEqual(summary["dimensions"][0]["scale"], "log")

    def test_cli_accepts_and_runs_hpobench_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_task_files(root)
            prior_path = root / "prior.json"
            self._write_prior(prior_path)
            plot_path = root / "results" / "bo_performance.png"
            args = build_parser().parse_args(
                [
                    "hpobench-bo",
                    "--hpobench-model",
                    "rf",
                    "--hpobench-task",
                    "credit_g",
                ]
            )
            self.assertEqual(args.hpobench_model, "rf")
            self.assertEqual(args.hpobench_task, "credit_g")
            exit_code = main(
                [
                    "hpobench-bo",
                    "--hpobench-model",
                    "rf",
                    "--hpobench-task",
                    "credit_g",
                    "--hpobench-benchmark-directory",
                    str(root),
                    "--dataset-prior-file",
                    str(prior_path),
                    "--queries",
                    "2",
                    "--baselines",
                    "vanilla-bo",
                    "random-search",
                    "--quiet",
                    "--plot",
                    str(plot_path),
                ]
            )
            self.assertTrue(plot_path.is_file())
            run_config = json.loads(
                (plot_path.parent / "run_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(run_config["model"], "rf")
            self.assertEqual(run_config["task"]["openml_task_id"], 31)
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
