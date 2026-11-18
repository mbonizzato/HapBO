"""Misleading-hint routing: restatement in encoded coordinates and delivery to the text-reading baselines."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from baselines import get_baseline  # noqa: E402
from experiments.worker import (  # noqa: E402
    _encode_mixed_candidates,
    _external_objective_meta,
    _misleading_in_encoded_space,
)

HINT = ("Domain guidance for tuning this classifier: accuracy is MAXIMIZED near the configuration "
        "alpha=0.05623, eta0=1e-05. Prefer that region; configurations far from it underperform.")


def _grid():
    alphas = [1e-5, 1e-4, 1e-3, 1e-2, 0.0562341325, 1.0]
    etas = [1e-5, 1e-3, 1e-1]
    native = [{"alpha": a, "eta0": e} for a in alphas for e in etas]
    encoded = _encode_mixed_candidates(native, ["alpha", "eta0"], ("alpha", "eta0"))
    return native, encoded


class MisleadingHintTests(unittest.TestCase):
    def test_restated_in_encoded_coordinates(self) -> None:
        native, encoded = _grid()
        text = _misleading_in_encoded_space(HINT, native, encoded, ["alpha", "eta0"])
        index = next(i for i, c in enumerate(native) if c["alpha"] == 0.0562341325 and c["eta0"] == 1e-5)
        row = encoded[index]
        self.assertIn(f"x0={row[0]:.3g}, x1={row[1]:.3g}", text)
        self.assertNotIn("alpha=", text)
        self.assertTrue(text.startswith("Domain guidance") and text.endswith("underperform."))

    def test_unparsable_text_unchanged(self) -> None:
        native, encoded = _grid()
        self.assertEqual(_misleading_in_encoded_space("no location here", native, encoded, ["alpha", "eta0"]),
                         "no location here")

    def test_text_reading_baselines_receive_the_hint(self) -> None:
        native, encoded = _grid()
        space = {"candidates": native, "encoded_candidates": encoded}
        meta = {"maximize": True, "n_constraints": 0, "model": "classifier", "task": "classification",
                "metric": "validation accuracy", "description": "task line. " + HINT, "target_description": HINT}
        bora = get_baseline("bora", search_space=space, objective_meta=meta, prior=None, seed=1, budget=4)
        self.assertIn(HINT, bora.worker_config()["description"])
        llambo = get_baseline("llambo", search_space=space, objective_meta=meta, prior=None, seed=1, budget=4)
        self.assertEqual(llambo.worker_config()["target_description"], HINT)


class WorkerGateTests(unittest.TestCase):
    """The hint reaches the baselines only when the study sets optimizer.misleading_description."""

    def _meta(self, problem_id: str, optimizer: dict) -> dict:
        native, encoded = _grid()
        problem = SimpleNamespace(maximize=True, model="lr", task_name="kc1", target_name="accuracy",
                                  parameter_names=("alpha", "eta0"))
        run = {"benchmark_id": "hpobench-tabular", "problem_id": problem_id, "optimizer": optimizer}
        return _external_objective_meta(run, problem, "hpobench", native, encoded)

    def test_hint_file_alone_does_not_mislead(self) -> None:
        self.assertTrue((PROJECT_ROOT / "experiments/misleading_descriptions/hpobench-tabular/lr-kc1.txt").is_file())
        meta = self._meta("lr-kc1", {})
        self.assertNotIn("Domain guidance", meta["description"])
        self.assertNotIn("target_description", meta)

    def test_flag_delivers_the_hint(self) -> None:
        meta = self._meta("lr-kc1", {"misleading_description": True})
        self.assertIn("Domain guidance", meta["description"])
        self.assertIn(meta["target_description"], meta["description"])
        self.assertTrue(meta["misleading_native"].startswith("Domain guidance"))

    def test_flag_without_hint_file_fails(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "misleading_description requested"):
            self._meta("lr-no-such-task", {"misleading_description": True})


if __name__ == "__main__":
    unittest.main()
