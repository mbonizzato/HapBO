"""PriorBand / PriMO multi-fidelity arm: loader, cost ledger, repeat cache, prior and rung mapping,
result files, and an end-to-end NePS run on a toy table.

The check that the maximum-fidelity column of every task equals the full-fidelity loader needs the
HPOBench tables and is skipped when they are absent. The end-to-end test needs the NePS 0.17.0
interpreter (``NEPS_PYTHON``) and is skipped otherwise.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "paper"))

from baselines.neps_mf import (  # noqa: E402
    MFLedger,
    cost_curves,
    fidelity_spec,
    grid_dimensions,
    index_to_row,
    map_rungs,
    method_settings,
    prior_for_neps,
    run_neps_mf,
    write_mf_results,
)
from datasets.hpobench import MODELS, TASKS, hpobench_task_path, load_hpobench_problem  # noqa: E402
from datasets.hpobench_mf import MF_FIDELITY, HPOBenchMFProblem, load_hpobench_mf_problem  # noqa: E402

FAMILY_FIDELITIES = {
    "lr": ("iter", [12, 37, 111, 333, 1000]),
    "nn": ("iter", [3, 9, 27, 81, 243]),
    "rf": ("n_estimators", [18, 56, 170, 512]),
    "xgb": ("n_estimators", [74, 222, 666, 2000]),
    "svm": ("subsample", [1 / 9, 1 / 3, 1.0]),
}


def _tables_present() -> list[tuple[str, str]]:
    return [(m, t) for m in MODELS for t in TASKS if all(p.is_file() for p in hpobench_task_path(m, t))]


def _write_mf_fixture(root: Path, *, drop_cell: bool = False, declared=(16, 48, 144)) -> None:
    """rf/credit_g fixture: 2x2 grid, 3 n_estimators fidelities, 2 seeds, plus subsample=0.5 rows
    that must be ignored."""

    directory = root / "rf" / "31"
    directory.mkdir(parents=True)
    rows = []
    rng = np.random.default_rng(0)
    for depth in (1, 2):
        for feat in (0.25, 1.0):
            for fid in (16.0, 48.0, 144.0):
                if drop_cell and (depth, feat, fid) == (2, 1.0, 48.0):
                    continue
                for seed in (11, 22):
                    acc = float(np.round(0.3 + 0.1 * depth + 0.1 * feat + fid / 1000 + 0.01 * rng.random(), 6))
                    rows.append({"max_depth": depth, "max_features": feat, "n_estimators": fid,
                                 "subsample": 1.0, "seed": seed,
                                 "result": {"info": {"val_scores": {"acc": acc}}}})
            rows.append({"max_depth": depth, "max_features": feat, "n_estimators": 144.0, "subsample": 0.5,
                         "seed": 11, "result": {"info": {"val_scores": {"acc": 0.99}}}})
    pd.DataFrame(rows).to_parquet(directory / "rf_31_data.parquet.gzip")
    x_discrete = {"hyperparameters": [
        {"name": "max_depth", "type": "ordinal", "sequence": [1, 2]},
        {"name": "max_features", "type": "ordinal", "sequence": [0.25, 1.0]}]}
    z_discrete = {"hyperparameters": [
        {"name": "n_estimators", "type": "ordinal", "sequence": list(declared)},
        {"name": "subsample", "type": "ordinal", "sequence": [0.5, 1.0]}]}
    (directory / "rf_31_metadata.json").write_text(json.dumps({"config_spaces": {
        "x": json.dumps(x_discrete), "x_discrete": json.dumps(x_discrete), "z_discrete": json.dumps(z_discrete)}}))


class LoaderTests(unittest.TestCase):
    def test_mf_values_average_seeds_and_match_full_fidelity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_mf_fixture(root)
            mf = load_hpobench_mf_problem(model="rf", task="credit_g", benchmark_directory=root)
            full = load_hpobench_problem(model="rf", task="credit_g", benchmark_directory=root)
        self.assertEqual(mf.fidelity_name, "n_estimators")
        self.assertEqual(mf.fidelity_values, (16.0, 48.0, 144.0))
        self.assertEqual(mf.fixed_fidelities, {"subsample": 1.0})
        np.testing.assert_allclose(mf.costs, [16 / 144, 48 / 144, 1.0])
        self.assertEqual(mf.native_candidates, full.native_candidates)
        self.assertTrue(np.array_equal(mf.ground_truth, full.ground_truth))
        self.assertTrue(np.all(mf.seed_counts == 2))
        self.assertTrue(np.all(np.diff(mf.values, axis=1) > 0))  # fixture accuracy grows with fidelity
        self.assertLess(mf.values.max(), 0.99)  # the subsample=0.5 rows are excluded

    def test_incomplete_grid_or_wrong_declared_fidelities_raise(self) -> None:
        for kwargs in ({"drop_cell": True}, {"declared": (16, 50, 144)}):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _write_mf_fixture(root, **kwargs)
                with self.assertRaises(ValueError):
                    load_hpobench_mf_problem(model="rf", task="credit_g", benchmark_directory=root)

    def test_max_fidelity_column_equals_the_full_fidelity_loader(self) -> None:
        present = _tables_present()
        if not present:
            self.skipTest("HPOBench tabular data not available locally")
        for model, task in present:
            name, table = FAMILY_FIDELITIES[model]
            with self.subTest(task=f"{model}-{task}"):
                full = load_hpobench_problem(model=model, task=task)
                mf = load_hpobench_mf_problem(model=model, task=task)
                self.assertEqual((mf.fidelity_name, MF_FIDELITY[model]), (name, name))
                np.testing.assert_allclose(mf.fidelity_values, table, rtol=1e-6)
                np.testing.assert_allclose(mf.costs, np.asarray(table, float) / table[-1], rtol=1e-15)
                self.assertEqual(mf.native_candidates, full.native_candidates)
                self.assertTrue(np.array_equal(mf.ground_truth, full.ground_truth))  # bit for bit
                self.assertTrue(np.all(np.isfinite(mf.values)))


def _toy_problem():
    candidates = tuple({"a": a, "b": b} for a in (0.1, 1.0, 10.0) for b in (1, 2))
    values = np.array([[0.50, 0.60, 0.70],
                       [0.52, 0.61, 0.69],
                       [0.55, 0.65, 0.80],
                       [0.40, 0.60, 0.75],
                       [0.58, 0.66, 0.72],
                       [0.45, 0.50, 0.55]])
    return candidates, values, [1 / 9, 1 / 3, 1.0]


class LedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates, self.values, self.costs = _toy_problem()
        self.ledger = MFLedger(self.values, self.costs, self.values[:, -1], budget=2.0,
                               fidelity_values=[1, 3, 9], candidates=self.candidates)

    def test_costs_accumulate_and_the_over_budget_proposal_is_refused(self) -> None:
        charged = [self.ledger.charge(0, 0, trial_id="1_rung_0"), self.ledger.charge(1, 1, trial_id="2_rung_1"),
                   self.ledger.charge(2, 2, trial_id="2_rung_2")]
        self.assertTrue(all(r is not None for r in charged))
        self.assertAlmostEqual(self.ledger.cumulative_cost, 1 / 9 + 1 / 3 + 1.0)
        self.assertIsNone(self.ledger.charge(3, 2, trial_id="4_rung_2"))  # 1.444 + 1 > 2: not evaluated
        self.assertEqual((self.ledger.refused["candidate_index"], self.ledger.refused["trial_id"]), (3, "4_rung_2"))
        self.assertEqual(len(self.ledger.records), 3)
        self.assertLessEqual(self.ledger.cumulative_cost, 2.0)

    def test_an_exact_fit_is_evaluated_despite_float_accumulation(self) -> None:
        values = np.full((40, 3), 0.5)
        ledger = MFLedger(values, [1 / 9, 1 / 3, 1.0], values[:, -1], budget=16.0)
        plan = [(i, 0) for i in range(18)] + [(i, 1) for i in range(12)] + [(i, 2) for i in range(10)]
        self.assertTrue(all(ledger.charge(row, rung) is not None for row, rung in plan))  # 2 + 4 + 10 = 16
        self.assertAlmostEqual(ledger.cumulative_cost, 16.0, places=12)
        self.assertIsNone(ledger.charge(39, 0))

    def test_a_repeat_is_charged_and_served_from_the_cache(self) -> None:
        first = self.ledger.charge(4, 1)
        self.ledger.values[4, 1] = -1.0  # a re-evaluation would read this; the cache must not
        again = self.ledger.charge(4, 1)
        self.assertEqual((first["repeat"], again["repeat"]), (False, True))
        self.assertEqual(again["objective"], first["objective"])
        self.assertAlmostEqual(self.ledger.cumulative_cost, 2 / 3)
        self.assertFalse(self.ledger.charge(4, 2)["repeat"])  # same configuration, new fidelity
        acc = self.ledger.accounting()
        self.assertEqual((acc["n_evaluations"], acc["n_repeats"]), (3, 1))
        self.assertEqual((acc["per_rung"]["1"]["n"], acc["per_rung"]["1"]["n_repeats"]), (2, 1))
        self.assertAlmostEqual(acc["repeat_units"], 1 / 3)

    def test_the_curve_is_the_best_full_fidelity_evaluation_against_cost(self) -> None:
        ledger = MFLedger(self.values, self.costs, self.values[:, -1], budget=3.0)
        r1 = ledger.charge(4, 0)            # c = .111, observed .58 (true .72): no full-fidelity evaluation yet
        ledger.charge(1, 2)                 # c = 1.111, full fidelity .69
        ledger.charge(2, 2)                 # c = 2.111, full fidelity .80
        self.assertIsNone(r1["best_full_fidelity_true_value"])
        curves = cost_curves(ledger.records, 3, optimum=0.80)
        self.assertEqual(curves["best_full_fidelity"]["true_value"], [None, 0.69, 0.80])
        self.assertEqual(curves["n_evaluations"], [1, 2, 3])

    def test_a_study_cannot_change_the_arm_settings(self) -> None:
        method = {"method_id": "priorband", "neps_algorithm": "priorband",
                  "neps_kwargs": {"sample_prior_first": "highest_fidelity"}}
        settings = method_settings(method, {"neps_kwargs": {"sample_prior_first": "highest_fidelity"}})
        self.assertEqual((settings["neps_kwargs"], settings["max_proposals"]),
                         ({"sample_prior_first": "highest_fidelity"}, 2000))
        with self.assertRaises(ValueError):
            method_settings(method, {"neps_kwargs": {}})


class MappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates, _, _ = _toy_problem()
        self.dims = grid_dimensions(["a", "b"], self.candidates)

    def test_grid_index_bijection(self) -> None:
        rows = index_to_row(self.dims, self.candidates)
        self.assertEqual(len(rows), 6)
        for (i, j), row in rows.items():
            self.assertEqual(self.candidates[row], {"a": self.dims[0]["values"][i], "b": self.dims[1]["values"][j]})
        with self.assertRaises(ValueError):
            index_to_row(self.dims, self.candidates[:-1])  # not a full grid

    def _prior(self, **override):
        doc = {"artifact_format": "neps-point-prior-v1", "parameters": {
            "a": {"value": 10.0, "index": 2, "n_values": 3, "confidence": "high"},
            "b": {"value": 1, "index": 0, "n_values": 2, "confidence": "low"}}}
        doc["parameters"]["a"].update(override)
        return doc

    def test_prior_mapping(self) -> None:
        self.assertEqual(prior_for_neps(self._prior(), self.dims),
                         {"indices": {"a": 2, "b": 0}, "confidence": {"a": "high", "b": "low"}})
        for bad in ({"index": 1}, {"value": 3.0}, {"confidence": "very high"}, {"n_values": 4}):
            with self.assertRaises(ValueError):
                prior_for_neps(self._prior(**bad), self.dims)

    def test_eta3_rungs_land_on_the_table_fidelities(self) -> None:
        self.assertEqual(fidelity_spec([12.0, 37.0, 111.0, 333.0, 1000.0], "iter"),
                         {"name": "iter", "type": "int", "lower": 12, "upper": 1000})
        self.assertEqual(fidelity_spec([1 / 9, 1 / 3, 1.0], "subsample")["type"], "float")
        for name, (_, table) in FAMILY_FIDELITIES.items():  # NePS rungs: z_max / 3**k
            cast = float if name == "svm" else int
            rungs = {r: cast(table[-1] / 3 ** (len(table) - 1 - r)) for r in range(len(table))}
            self.assertEqual(map_rungs(rungs, table), {r: r for r in range(len(table))}, name)
        with self.assertRaises(ValueError):
            map_rungs({0: 12, 1: 1000}, [12, 37, 111, 333, 1000])  # rungs do not cover the table


class ResultFileTests(unittest.TestCase):
    def test_mf_log_and_the_figure_curve(self) -> None:
        import fig_priorband_primo as FIG

        candidates, values, costs = _toy_problem()
        ledger = MFLedger(values, costs, values[:, -1], budget=2.0, fidelity_values=[1, 3, 9], candidates=candidates)
        for row, rung in ((4, 0), (2, 0), (2, 1), (1, 2), (0, 0)):
            ledger.charge(row, rung, trial_id=f"{row}_rung_{rung}")
        self.assertIsNone(ledger.charge(3, 2))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            metrics = write_mf_results(out, method_id="priorband", label="PriorBand", records=ledger.records,
                                       accounting=ledger.accounting(), ground_truth=values[:, -1],
                                       candidates=candidates, budget=2)
            self.assertFalse((out / "evaluations.jsonl").exists())  # one line per evaluation, not per query
            records = [json.loads(line) for line in (out / "mf_evaluations.jsonl").read_text().splitlines()]
            curves = json.loads((out / "cost_curves.json").read_text())
        self.assertEqual(len(records), 5)
        self.assertEqual((metrics["best_index"], metrics["n_evaluations"]), (1, 5))
        self.assertAlmostEqual(metrics["final_normalized_regret"], 1 - 0.69 / 0.80)
        # the paper figure recomputes curve (a) from [cumulative cost, true value] of the full-fidelity evaluations
        run = {"status": "completed",
               "y": [[r["cumulative_cost"], r["true_full_fidelity_value"]] for r in records if r["full_fidelity"]]}
        regret = curves["best_full_fidelity"]["normalized_regret"]
        self.assertEqual(FIG.cost_curve(run, 0.80)[:2], regret)
        per_task = {"t1": [[None, 0.4] + [0.2] * 14, [None, None] + [0.1] * 14],
                    "t2": [[0.5, 0.3] + [0.3] * 14, [0.5, 0.3] + [0.1] * 14]}
        m, _, n = FIG.aggregate(per_task, ["t1", "t2"])
        self.assertTrue(np.isnan(m[0]) and np.isnan(m[1]))  # a point is averaged only where every run is defined
        self.assertEqual(n[:3], [0, 0, 2])


@unittest.skipUnless(os.environ.get("NEPS_PYTHON"), "set NEPS_PYTHON to the NePS 0.17.0 interpreter")
class NePSWorkerTests(unittest.TestCase):
    """End to end through the official NePS package on a toy table (the paper's two arms)."""

    PRIOR = {"indices": {"a": 1, "b": 2}, "confidence": {"a": "medium", "b": "high"}}
    ARMS = {"priorband": {"algorithm": "priorband", "optimizer_kwargs": {"sample_prior_first": "highest_fidelity"}},
            "primo": {"algorithm": "primo", "optimizer_kwargs": {"initial_design_size": 5}}}

    def _problem(self) -> HPOBenchMFProblem:
        candidates = tuple({"a": a, "b": b} for a in (0.001, 0.01, 0.1, 1.0, 10.0) for b in (1, 2, 4, 8))
        fids = (3.0, 9.0, 27.0, 81.0)
        values = np.array([[0.9 - 0.05 * abs(np.log10(c["a"]) + 1) - 0.02 * abs(np.log2(c["b"]) - 2) - 0.1 / f
                            for f in fids] for c in candidates])
        return HPOBenchMFProblem(
            dataset_id="hpobench", model="nn", task_name="toy", task_id=0, parameter_names=("a", "b"),
            native_candidates=candidates, fidelity_name="iter", fidelity_values=fids, fixed_fidelities={},
            values=values, seed_counts=np.ones_like(values, dtype=int),
            metadata={"declared_fidelity_values": [3, 9, 27, 81]})

    def test_arms_run_deterministically_within_budget(self) -> None:
        problem = self._problem()
        for name, kwargs in self.ARMS.items():
            with self.subTest(arm=name):
                out = [run_neps_mf(problem, prior=self.PRIOR, seed=1001, budget=8.0, **kwargs) for _ in range(2)]
                acc = out[0]["accounting"]
                self.assertEqual((acc["neps_version"], acc["stop_reason"]), ("0.17.0", "budget"))
                self.assertLessEqual(acc["cost_used"], 8.0 + 1e-9)
                self.assertGreater(acc["cost_used"] + acc["refused"]["cost"], 8.0)
                for r in out[0]["records"]:
                    self.assertEqual(r["candidate"], problem.native_candidates[r["candidate_index"]])
                    self.assertEqual(r["fidelity"], problem.fidelity_values[r["rung"]])
                strip = [[{k: v for k, v in r.items() if k != "ask_seconds"} for r in o["records"]] for o in out]
                self.assertEqual(strip[0], strip[1])  # same seed, same run
                first = out[0]["records"][0]
                if name == "priorband":  # the prior mode first, at the maximum fidelity
                    self.assertEqual((first["phase"], first["rung"], first["indices"]),
                                     ("prior_mode", 3, self.PRIOR["indices"]))
                else:
                    self.assertEqual(first["phase"], "initial_design")
                    self.assertIn("bo", {r["phase"] for r in out[0]["records"]})


if __name__ == "__main__":
    unittest.main()
