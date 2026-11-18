from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import _path_component, save_neuromosaic_run_records  # noqa: E402
from src.prior.landscape.landscape_generator import (  # noqa: E402
    DEFAULT_PRIOR_DATASET_ID,
    DEFAULT_PRIOR_ID,
    DEFAULT_PRIOR_TARGET_NAME,
    DEFAULT_PRIOR_TARGET_SLUG,
    PRIOR_RESULTS_DIRECTORY,
    RECORD_PATH,
    VOCABULARY_CARD_PATH,
    _sha256_text_artifact,
    finalize_prior_experiment,
    initialize_prior_experiment,
    load_record,
    resolve_prior_path,
    results_directory_for_prior,
    target_path_component,
    validate_prior_compatibility,
    validate_prior_metadata,
    validate_record,
)


class PriorLayoutTests(unittest.TestCase):
    def test_default_prior_and_results_use_dataset_and_prior_ids(self) -> None:
        expected_prior = (
            PROJECT_ROOT
            / "experiments"
            / "priors"
            / DEFAULT_PRIOR_DATASET_ID
            / DEFAULT_PRIOR_TARGET_SLUG
            / DEFAULT_PRIOR_ID
            / "prior.json"
        )
        expected_results = (
            PROJECT_ROOT
            / "results"
            / "prior"
            / DEFAULT_PRIOR_DATASET_ID
            / DEFAULT_PRIOR_TARGET_SLUG
            / DEFAULT_PRIOR_ID
        )

        self.assertEqual(RECORD_PATH, expected_prior)
        self.assertEqual(PRIOR_RESULTS_DIRECTORY, expected_results)
        self.assertEqual(results_directory_for_prior(RECORD_PATH), expected_results)

    def test_prior_path_is_inferred_from_dataset_target_and_only_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = (
                root
                / "fixture-dataset"
                / "left-tibialis-anterioris"
                / "expert-a"
                / "prior.json"
            )
            expected.parent.mkdir(parents=True)
            expected.touch()

            resolved = resolve_prior_path(
                "fixture-dataset",
                target_name="LeftTibialisAnterioris",
                experiments_directory=root,
            )

        self.assertEqual(resolved, expected)

    def test_checked_in_rat_priors_record_exact_muscle_names(self) -> None:
        exact_names = {
            "left-gluteus": "LeftGluteus",
            "left-knee-extensor": "LeftKneeExtensor",
            "left-knee-flexor": "LeftKneeFlexor",
            "left-medialis-gastrocnemius": "LeftMedialisGastrocnemius",
            "left-tibialis-anterioris": "LeftTibialisAnterioris",
            "right-gluteus": "RightGluteus",
            "right-medialis-gastrocnemius": "RightMedialisGastrocnemius",
            "right-tibialis-anterioris": "RightTibialisAnterioris",
        }
        prior_root = (
            PROJECT_ROOT / "experiments" / "priors" / DEFAULT_PRIOR_DATASET_ID
        )

        shipped = [slug for slug in exact_names if (prior_root / slug).is_dir()]
        self.assertTrue(shipped)
        for target_slug in shipped:
            target_name = exact_names[target_slug]
            with self.subTest(target_name=target_name):
                metadata_path = (
                    prior_root
                    / target_slug
                    / DEFAULT_PRIOR_ID
                    / "metadata.json"
                )
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                self.assertEqual(metadata["scope"]["target_name"], target_name)
                self.assertEqual(metadata["scope"]["target_slug"], target_slug)

    def test_checked_in_rat_priors_share_the_canonical_coordinate_schema(self) -> None:
        prior_root = (
            PROJECT_ROOT / "experiments" / "priors" / DEFAULT_PRIOR_DATASET_ID
        )
        expected_names = (
            "electrode_mediolateral_index",
            "electrode_rostrocaudal_index",
        )
        expected_values = (list(range(1, 5)), list(range(1, 9)))

        for prior_path in sorted(prior_root.glob("*/*/prior.json")):
            with self.subTest(prior_path=prior_path):
                record = json.loads(prior_path.read_text(encoding="utf-8"))
                validate_record(record, source=str(prior_path))
                dimensions = record["schema"]["dimensions"]
                self.assertEqual(
                    tuple(dimension["name"] for dimension in dimensions),
                    expected_names,
                )
                self.assertEqual(
                    tuple(dimension["values"] for dimension in dimensions),
                    expected_values,
                )

    def test_checked_in_rat_prompt_cards_describe_the_canonical_schema(self) -> None:
        prior_root = (
            PROJECT_ROOT / "experiments" / "priors" / DEFAULT_PRIOR_DATASET_ID
        )
        # the cards render each ordinal axis as "values: [...]"
        mediolateral_pattern = (
            r"name:\s*electrode_mediolateral_index[\s\S]{0,200}?"
            r"values:\s*\[1, 2, 3, 4\]"
        )
        rostrocaudal_pattern = (
            r"name:\s*electrode_rostrocaudal_index[\s\S]{0,240}?"
            r"values:\s*\[1, 2, 3, 4, 5, 6, 7, 8\]"
        )

        for prompt_path in sorted(prior_root.glob("*/*/prompt_card.md")):
            with self.subTest(prompt_path=prompt_path):
                prompt = prompt_path.read_text(encoding="utf-8")
                self.assertRegex(prompt, mediolateral_pattern)
                self.assertRegex(prompt, rostrocaudal_pattern)

    def test_prior_id_disambiguates_multiple_inferred_versions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prior_root = root / "solar-mincost-c1"
            for prior_id in ("expert-a", "expert-b"):
                path = prior_root / prior_id / "prior.json"
                path.parent.mkdir(parents=True)
                path.touch()

            with self.assertRaisesRegex(ValueError, "multiple HapBO priors"):
                resolve_prior_path(
                    "solar-mincost-c1",
                    experiments_directory=root,
                )
            resolved = resolve_prior_path(
                "solar-mincost-c1",
                prior_id="expert-b",
                experiments_directory=root,
            )

        self.assertEqual(resolved, prior_root / "expert-b" / "prior.json")

    def test_missing_inferred_prior_reports_expected_location(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ValueError,
                r"no HapBO prior found.*<prior-id>[\\/]prior\.json",
            ):
                resolve_prior_path(
                    "fixture-dataset",
                    target_name="Missing Target",
                    experiments_directory=directory,
                )

    def test_default_prior_metadata_hashes_every_input_snapshot(self) -> None:
        metadata = validate_prior_metadata(RECORD_PATH)

        self.assertIsNotNone(metadata)
        assert metadata is not None
        for filename, file_metadata in metadata["files"].items():
            artifact_path = RECORD_PATH.parent / filename
            actual_hash = _sha256_text_artifact(artifact_path)
            self.assertEqual(actual_hash, file_metadata["sha256"])
        self.assertEqual(
            _sha256_text_artifact(VOCABULARY_CARD_PATH),
            metadata["vocabulary"]["sha256"],
        )
        self.assertEqual(load_record(RECORD_PATH)["schema"]["objective"], "maximize")

    def test_changed_snapshot_is_rejected_by_metadata_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prior_directory = Path(directory) / "dataset" / "target" / "prior-a"
            prior_directory.mkdir(parents=True)
            for filename in ("prompt_card.md", "prior.json"):
                source = RECORD_PATH.parent / filename
                (prior_directory / filename).write_bytes(source.read_bytes())
            metadata = json.loads(
                (RECORD_PATH.parent / "metadata.json").read_text(encoding="utf-8")
            )
            metadata["dataset_id"] = "dataset"
            metadata["prior_id"] = "prior-a"
            metadata["scope"]["target_name"] = "target"
            metadata["scope"]["target_slug"] = "target"
            (prior_directory / "metadata.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            (prior_directory / "prompt_card.md").write_text(
                "changed",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "recorded sha256"):
                validate_prior_metadata(prior_directory / "prior.json")

    def test_manual_prior_can_be_initialized_and_finalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prior_directory = initialize_prior_experiment(
                "test-dataset",
                "manual-prior",
                provider="manual-provider",
                model="manual-model",
                target_name="Exact Target",
                experiments_directory=root / "priors",
                card_human_statement=True,  # RECORD_PATH is the human expert map (user_stated)
            )
            self.assertEqual(
                prior_directory,
                root / "priors" / "test-dataset" / "exact-target" / "manual-prior",
            )
            self.assertTrue((prior_directory / "prompt_card.md").is_file())
            self.assertFalse((prior_directory / "prior.json").exists())

            (prior_directory / "prior.json").write_bytes(RECORD_PATH.read_bytes())
            compiler = finalize_prior_experiment(
                prior_directory,
                generate_landscape=False,
            )
            metadata = json.loads(
                (prior_directory / "metadata.json").read_text(encoding="utf-8")
            )

            self.assertEqual(metadata["status"], "finalized")
            self.assertEqual(metadata["scope"]["target_name"], "Exact Target")
            self.assertEqual(
                metadata["vocabulary"]["path"],
                "src/prior/expert_knowledge/vocabulary_card.md",
            )
            self.assertEqual(compiler.record["schema"]["objective"], "maximize")
            self.assertEqual(
                load_record(prior_directory / "prior.json")["schema"]["objective"],
                "maximize",
            )
            with self.assertRaisesRegex(ValueError, "already finalized"):
                finalize_prior_experiment(
                    prior_directory,
                    generate_landscape=False,
                )

    def test_neuromosaic_initialization_requires_exact_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "require an exact target_name"):
                initialize_prior_experiment(
                    "neuromosaic-dataset1-rat",
                    "missing-target",
                    experiments_directory=Path(directory),
                )

    def test_result_path_components_are_stable(self) -> None:
        self.assertEqual(
            _path_component("LeftMedialisGastrocnemius"),
            "left-medialis-gastrocnemius",
        )
        self.assertEqual(
            target_path_component("LeftMedialisGastrocnemius"),
            "left-medialis-gastrocnemius",
        )

    def test_prior_compatibility_requires_the_exact_target(self) -> None:
        metadata = validate_prior_compatibility(
            RECORD_PATH,
            dataset_id=DEFAULT_PRIOR_DATASET_ID,
            target_name=DEFAULT_PRIOR_TARGET_NAME,
        )
        self.assertIsNotNone(metadata)
        with self.assertRaisesRegex(ValueError, "does not match selected target"):
            validate_prior_compatibility(
                RECORD_PATH,
                dataset_id=DEFAULT_PRIOR_DATASET_ID,
                target_name="RightGluteus",
            )

    def test_run_records_capture_prior_identity_and_metrics(self) -> None:
        problem = SimpleNamespace(
            dataset_id="neuromosaic-dataset1-rat",
            subject_id="rat1_M1_190716",
            target_name="LeftMedialisGastrocnemius",
        )
        result = SimpleNamespace(
            best_y=1.2,
            best_index=3,
            evaluated_indices=np.array([0, 3]),
        )
        run_metrics = SimpleNamespace(
            simple_regret=np.array([0.5, 0.1]),
            cumulative_regret=np.array([0.5, 0.6]),
            normalized_best_performance=np.array([0.7, 0.95]),
            first_optimum_query=None,
            first_95_percent_query=2,
        )
        runs = [
            SimpleNamespace(
                key="hapbo",
                label="HapBO",
                result=result,
                metrics=run_metrics,
            )
        ]
        args = SimpleNamespace(
            dataset_prior_file=RECORD_PATH,
            acquisition="expected-improvement",
            improvement_xi=0.0,
            kappa=2.0,
            queries=2,
            n_initial=1,
            length_scale=2.0,
            noise_std=0.05,
            seed=42,
            baselines=(),
        )

        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory)
            plot_path = output_directory / "bo_performance.png"
            plot_path.touch()
            config_path, metrics_path = save_neuromosaic_run_records(
                output_directory,
                problem,
                runs,
                args,
                plot_path,
            )
            config = json.loads(config_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertEqual(config["prior"]["id"], DEFAULT_PRIOR_ID)
        self.assertEqual(config["optimization"]["seed"], 42)
        self.assertEqual(metrics["methods"]["hapbo"]["best_index"], 3)


if __name__ == "__main__":
    unittest.main()
