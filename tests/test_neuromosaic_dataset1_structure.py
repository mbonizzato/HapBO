from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from main import (  # noqa: E402
    adapt_neuromosaic_candidates_to_prior,
    build_parser,
    run_neuromosaic,
)
from datasets import (  # noqa: E402
    NEUROMOSAIC_DATASET1_RAT_DATASET_ID,
    NEUROMOSAIC_DATASET1_RAT_DEFAULT_SUBJECT_PATH,
)


class NeuroMosaicDataset1StructureTests(unittest.TestCase):
    def test_canonical_dataset1_names_match_dataset_ids_and_paths(self) -> None:
        self.assertEqual(
            NEUROMOSAIC_DATASET1_RAT_DATASET_ID,
            "neuromosaic-dataset1-rat",
        )
        self.assertIn(
            "benchmark/NeuroMOSAIC/dataset1/rat",
            NEUROMOSAIC_DATASET1_RAT_DEFAULT_SUBJECT_PATH.as_posix(),
        )

    def test_non_default_rat_muscle_resolves_its_own_prior(self) -> None:
        args = build_parser().parse_args(
            [
                "neuromosaic-bo",
                "--dataset",
                "neuromosaic-dataset1-rat",
                "--subject",
                "rat1",
                "--muscle",
                "LeftTibialisAnterioris",
            ]
        )
        problem = SimpleNamespace(
            dataset_id="neuromosaic-dataset1-rat",
            target_name="LeftTibialisAnterioris",
            n_candidates=32,
        )
        resolved_prior = PROJECT_ROOT / "resolved-tibialis-prior.json"

        with (
            patch(
                "main.load_neuromosaic_dataset1_rat_problem",
                return_value=problem,
            ),
            patch(
                "main.resolve_prior_path",
                return_value=resolved_prior,
            ) as resolve_mock,
            patch("main.load_record", side_effect=RuntimeError("prior resolved")),
        ):
            with self.assertRaisesRegex(RuntimeError, "prior resolved"):
                run_neuromosaic(args)
        resolve_mock.assert_called_once_with(
            "neuromosaic-dataset1-rat",
            target_name="LeftTibialisAnterioris",
            prior_id="human-expert-map",
            explicit_path=None,
        )

    def test_anatomical_prior_axes_map_to_the_native_rat_grid(self) -> None:
        problem = SimpleNamespace(
            native_candidates=(
                {"channel_x": 1, "channel_y": 1},
                {"channel_x": 8, "channel_y": 4},
            )
        )
        names = (
            "electrode_mediolateral_index",
            "electrode_rostrocaudal_index",
        )

        candidates, to_dataset_candidate = adapt_neuromosaic_candidates_to_prior(
            problem,
            names,
        )

        self.assertEqual(
            candidates,
            (
                {
                    "electrode_mediolateral_index": 1,
                    "electrode_rostrocaudal_index": 1,
                },
                {
                    "electrode_mediolateral_index": 4,
                    "electrode_rostrocaudal_index": 8,
                },
            ),
        )
        self.assertEqual(
            to_dataset_candidate(
                {
                    "electrode_mediolateral_index": 4,
                    "electrode_rostrocaudal_index": 8,
                }
            ),
            {"channel_y": 4, "channel_x": 8},
        )


if __name__ == "__main__":
    unittest.main()
