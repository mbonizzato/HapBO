from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.prior.landscape.landscape_generator import (  # noqa: E402
    CompiledPrior,
    DimensionSpec,
    RECORD_PATH,
    _two_dimensional_prior_grid,
    _two_dimensional_prior_display_grid,
    compile_prior,
    load_record,
    plot_combined_prior,
    plot_two_dimensional_prior,
)


class PriorVisualizationTests(unittest.TestCase):
    def test_neuromosaic_prior_uses_the_canonical_four_by_eight_axes(self) -> None:
        compiler = compile_prior(load_record(RECORD_PATH))

        x_values, y_values, pi_values = _two_dimensional_prior_grid(compiler)

        np.testing.assert_array_equal(x_values, np.arange(1, 5))
        np.testing.assert_array_equal(y_values, np.arange(1, 9))
        self.assertEqual(pi_values.shape, (8, 4))
        self.assertAlmostEqual(
            pi_values[1, 0],
            compiler.pi(
                {
                    "electrode_mediolateral_index": 1,
                    "electrode_rostrocaudal_index": 2,
                }
            ),
        )

    def test_neuromosaic_display_matches_bo_response_map_orientation(self) -> None:
        compiler = compile_prior(load_record(RECORD_PATH))

        (
            x_dimension,
            y_dimension,
            x_values,
            y_values,
            pi_values,
        ) = _two_dimensional_prior_display_grid(compiler)

        self.assertEqual(x_dimension.name, "electrode_rostrocaudal_index")
        self.assertEqual(y_dimension.name, "electrode_mediolateral_index")
        np.testing.assert_array_equal(x_values, np.arange(1, 9))
        np.testing.assert_array_equal(y_values, np.arange(1, 5))
        self.assertEqual(pi_values.shape, (4, 8))
        self.assertAlmostEqual(
            pi_values[0, 1],
            compiler.pi(
                {
                    "electrode_mediolateral_index": 1,
                    "electrode_rostrocaudal_index": 2,
                }
            ),
        )

    def test_ordinal_continuous_grid_preserves_native_coordinates(self) -> None:
        compiler = CompiledPrior(
            record={
                "beliefs": [
                    {"token": "baseline", "level": 0.1},
                    {
                        "token": "bump",
                        "dim": "level",
                        "mode": "medium",
                        "weight": 0.7,
                    },
                    {
                        "token": "trend",
                        "dim": "dose",
                        "direction": "increase",
                        "weight": 0.5,
                    },
                ]
            },
            dimensions=[
                DimensionSpec(
                    name="level",
                    type="ordinal",
                    values=("low", "medium", "high"),
                ),
                DimensionSpec(
                    name="dose",
                    type="continuous",
                    range=(1.0, 100.0),
                    scale="log",
                ),
            ],
            alpha=0.4,
        )

        x_values, y_values, pi_values = _two_dimensional_prior_grid(
            compiler,
            n_points=5,
        )

        np.testing.assert_array_equal(x_values, ["low", "medium", "high"])
        np.testing.assert_allclose(y_values, np.geomspace(1.0, 100.0, 5))
        self.assertEqual(pi_values.shape, (5, 3))
        self.assertAlmostEqual(
            pi_values[2, 1],
            compiler.pi({"level": "medium", "dose": y_values[2]}),
        )

    def test_two_continuous_dimensions_produce_a_contour_plot(self) -> None:
        compiler = CompiledPrior(
            record={
                "beliefs": [
                    {"token": "baseline", "level": 0.1},
                    {
                        "token": "joint_region",
                        "dims": ["x", "y"],
                        "mode": {"x": 0.25, "y": 0.75},
                        "weight": 1.0,
                    },
                ]
            },
            dimensions=[
                DimensionSpec("x", "continuous", range=(0.0, 1.0)),
                DimensionSpec("y", "continuous", range=(0.0, 1.0)),
            ],
            alpha=0.5,
        )

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "continuous-prior.png"
            plot_two_dimensional_prior(
                compiler,
                n_points=25,
                out_path=output_path,
                show=False,
            )
            self.assertGreater(output_path.stat().st_size, 0)

    def test_combined_plot_dispatches_to_two_dimensional_view(self) -> None:
        compiler = compile_prior(load_record(RECORD_PATH))

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "combined-prior.png"
            plot_combined_prior(compiler, out_path=output_path, show=False)
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
