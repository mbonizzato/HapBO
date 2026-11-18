"""Vanilla BO with the prior-peak candidates: run_continuous_bo(shape_acquisition=False).

The control arm keeps the acquisition unshaped (no multiplier, no zero clamp, Sobol initial design) but
scores, at every step, the same prior-peak candidates as the alpha-pi arm run with the same prior, alpha
and seed.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "scripts")]

from bayesian_optimization import continuous_bo  # noqa: E402
from bayesian_optimization.continuous_bo import run_continuous_bo  # noqa: E402

D = 3
BOUNDS = np.array([[-2.0, 2.0]] * D)
X_OPT = np.array([0.7, -1.1, 0.3])                    # optimum of the objective (native)
PRIOR_PEAK = np.array([1.2, -0.4, -1.0])              # the prior's mode (native), away from X_OPT
N_SOBOL = 128


def objective(x) -> float:
    x = np.asarray(x, float)
    return float(np.sum((x - X_OPT) ** 2) + 0.2 * np.sum(np.sin(3.0 * x)))


def make_prior(with_peak_candidates: bool = True):
    def prior_fn(x) -> float:
        return float(np.exp(-np.sum((np.asarray(x, float) - PRIOR_PEAK) ** 2) / (2 * 0.35 ** 2)))

    if with_peak_candidates:
        prior_fn.peak_candidates = PRIOR_PEAK[None, :]
    return prior_fn


def run(**kw):
    args = dict(budget=8, n_initial=2, n_sobol=N_SOBOL, n_restarts=2, seed=5)
    args.update(kw)
    return run_continuous_bo(objective, BOUNDS, 0.0, **args)


def spy_batches(**kw):
    """Run and record every per-step acquisition candidate batch (the GP predictions on > N_SOBOL rows)."""
    batches = []
    predict = continuous_bo.GaussianProcessSurrogate.predict

    def spy(self, X, *a, **k):
        X = np.atleast_2d(np.asarray(X, float))
        if len(X) > N_SOBOL - 1:
            batches.append(X.copy())
        return predict(self, X, *a, **k)

    with mock.patch.object(continuous_bo.GaussianProcessSurrogate, "predict", spy):
        res = run(**kw)
    return res, batches


class DefaultPathTests(unittest.TestCase):
    def test_explicit_shape_flag_equals_default(self) -> None:
        cases = {"vanilla": dict(), "alpha-pi": dict(alpha=0.5, prior_fn=make_prior())}
        for name, kw in cases.items():
            with self.subTest(case=name):
                a = run(**kw)
                b = run(**kw, shape_acquisition=True)
                np.testing.assert_array_equal(a.X_native, b.X_native)
                np.testing.assert_array_equal(a.simple_regret, b.simple_regret)
                if "prior_fn" in kw:
                    self.assertIsNotNone(a.prior_peaks)
                np.testing.assert_array_equal(a.prior_peaks, b.prior_peaks)


class VanillaPeaksTests(unittest.TestCase):
    def test_candidate_batches_hold_the_alpha_pi_prior_peaks(self) -> None:
        hap, hap_batches = spy_batches(alpha=0.325, prior_fn=make_prior())
        vpk, vpk_batches = spy_batches(alpha=0.325, prior_fn=make_prior(), shape_acquisition=False)
        self.assertIsNotNone(hap.prior_peaks)
        np.testing.assert_array_equal(vpk.prior_peaks, hap.prior_peaks)
        # 8 screen climbs + 1 climb from the document peak
        self.assertEqual(len(vpk.prior_peaks), 9)
        unit_peak = (PRIOR_PEAK - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])
        self.assertLess(np.min(np.linalg.norm(vpk.prior_peaks - unit_peak, axis=1)), 1e-4)
        k = len(vpk.prior_peaks)
        for batches in (hap_batches, vpk_batches):
            self.assertEqual(len(batches), 8 - 2)                         # one batch per BO step
            for step, batch in enumerate(batches):
                self.assertEqual(batch.shape, (N_SOBOL + k, D))
                np.testing.assert_array_equal(batch[-k:], hap.prior_peaks)
                np.testing.assert_array_equal(batch[:N_SOBOL], continuous_bo._sobol(N_SOBOL, D, 5 + 2 + step))

    def test_initial_design_is_the_vanilla_sobol_design(self) -> None:
        van = run()
        vpk = run(alpha=0.325, prior_fn=make_prior(), shape_acquisition=False)
        np.testing.assert_array_equal(vpk.X_native[:2], van.X_native[:2])

    def test_alpha_only_gates_the_peak_search(self) -> None:
        # unshaped acquisition: any alpha > 0 gives the same run; alpha = 0 drops the peaks
        a = run(alpha=0.325, prior_fn=make_prior(), shape_acquisition=False)
        b = run(alpha=0.9, prior_fn=make_prior(), shape_acquisition=False)
        np.testing.assert_array_equal(a.X_native, b.X_native)
        np.testing.assert_array_equal(a.prior_peaks, b.prior_peaks)
        c = run(alpha=0.0, prior_fn=make_prior(), shape_acquisition=False)
        self.assertIsNone(c.prior_peaks)
        np.testing.assert_array_equal(c.X_native, run().X_native)

    def test_shaping_changes_the_run(self) -> None:
        hap = run(alpha=0.9, prior_fn=make_prior())
        vpk = run(alpha=0.9, prior_fn=make_prior(), shape_acquisition=False)
        self.assertFalse(np.array_equal(hap.X_native, vpk.X_native))

    def test_constant_prior_appends_no_peaks_and_equals_vanilla(self) -> None:
        def flat(x) -> float:
            return 1.0

        flat.peak_candidates = PRIOR_PEAK[None, :]
        van = run()
        for shape in (False, True):
            with self.subTest(shape_acquisition=shape):
                res, batches = spy_batches(alpha=0.325, prior_fn=flat, shape_acquisition=shape)
                self.assertIsNone(res.prior_peaks)
                self.assertTrue(all(len(b) == N_SOBOL for b in batches))
                np.testing.assert_array_equal(res.X_native, van.X_native)
                np.testing.assert_array_equal(res.simple_regret, van.simple_regret)


class RunnerTests(unittest.TestCase):
    def test_driver_runs_vanilla_peaks_with_the_hapbo_prior_and_flags(self) -> None:
        import run_synthetic_panel as rsp

        hap = list(rsp.commands("hapbo", ["rho90"], ["hartmann6"], 10))
        vpk = list(rsp.commands("vanilla-peaks", ["rho90"], ["hartmann6"], 10))
        self.assertEqual(len(vpk), len(hap))
        for h, v in zip(hap, vpk):
            swap = {"results/synthetic/well/rho90/hapbo": "results/synthetic/well/rho90/vanilla-peaks",
                    "hapbo": "vanilla-peaks"}
            self.assertEqual(v, [swap.get(x, x) for x in h])
        self.assertNotIn("vanilla-peaks", rsp.ARMS)

    def test_driver_runs_the_representation_arms_on_the_rho90_documents_only(self) -> None:
        import run_synthetic_panel as rsp

        for arm in rsp.REPRESENTATION_ARMS:
            cmds = list(rsp.commands(arm, rsp.CARDS, rsp.FAMILIES, 10))
            self.assertEqual(len(cmds), len(rsp.FAMILIES) * len(rsp.DRAWS))
            for c in cmds:
                prior = c[c.index("--prior-file") + 1]
                self.assertTrue(prior.startswith("experiments/synthetic_priors/rho90/"), prior)
                self.assertTrue(prior.endswith(f"{arm[len('hapbo'):]}/prior.json"), prior)
                self.assertTrue((PROJECT_ROOT / prior).is_file(), prior)

    PRIOR = PROJECT_ROOT / "experiments/synthetic_priors/rho90/hartmann6/draw1/prior.json"
    PEAK = 0.3                                                   # mode of the stand-in prior (every coordinate)

    def run_panel(self, *extra):
        """run_synthetic_arms.main on hartmann6 (budget 5, one seed, arms hapbo, vanilla-peaks, vanilla) with a
        cheap stand-in prior; returns the recorded run_continuous_bo calls, prior loads, trajectories, provenance."""
        import json
        import tempfile

        import run_synthetic_arms as rsa

        calls, loads = [], []
        rcb = continuous_bo.run_continuous_bo

        def recording(*a, **k):
            res = rcb(*a, **k)
            calls.append((k, res))
            return res

        def loader(family, dim, prior_file, document_peaks=True):
            loads.append((family, dim, Path(prior_file), document_peaks))
            peak = np.full(dim, self.PEAK)

            def prior_fn(x) -> float:
                return float(np.exp(-np.sum((np.asarray(x, float) - peak) ** 2) / (2 * 0.1 ** 2)))

            prior_fn.peak_candidates = peak[None, :]
            return prior_fn, 0.7

        with tempfile.TemporaryDirectory() as tmp:
            argv = ["run_synthetic_arms.py", "--label", "e2e", "--out-root", tmp, "--families", "hartmann6",
                    "--arms", "hapbo,vanilla-peaks,vanilla", "--prior-file", str(self.PRIOR),
                    "--alpha-override", "0.325", "--seeds", "1", "--restarts", "2", *extra]
            with mock.patch.object(sys, "argv", argv), mock.patch.dict(rsa.BUDGETS, {"hartmann6": 5}), \
                    mock.patch.object(rsa, "load_prior_fn", loader), \
                    mock.patch.object(continuous_bo, "run_continuous_bo", recording):
                rsa.main()
            traj = json.loads((Path(tmp) / "e2e" / "trajectories.json").read_text(encoding="utf-8"))["hartmann6"]
            prov = json.loads((Path(tmp) / "e2e" / "run_provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(loads, [("hartmann6", 6, self.PRIOR, True)])
        self.assertEqual(set(traj), {"hapbo", "vanilla-peaks", "vanilla"})
        self.assertEqual(prov["vanilla_peaks"], {"shape_acquisition": False, "prior_peak_candidates": True,
                                                 "document_peaks": True})
        for arm in traj:
            self.assertEqual(len(traj[arm][0]), 5)
        return calls, traj, prov

    def test_runner_vanilla_peaks_arm(self) -> None:
        calls, traj, prov = self.run_panel()
        (hk, hres), (vk, vres), (nk, nres) = calls
        self.assertEqual((hk["alpha"], vk["alpha"]), (0.325, 0.325))
        self.assertIs(hk["prior_fn"], vk["prior_fn"])
        self.assertEqual((hk.get("shape_acquisition", True), vk["shape_acquisition"]), (True, False))
        self.assertNotIn("prior_fn", nk)
        self.assertIsNotNone(hres.prior_peaks)
        np.testing.assert_array_equal(vres.prior_peaks, hres.prior_peaks)
        for res in (hres, vres):                                          # vanilla's Sobol initial design
            np.testing.assert_array_equal(res.X_native[:3], nres.X_native[:3])
        for arm in traj:
            self.assertEqual(traj[arm][0][:3], traj["vanilla"][0][:3])

    def test_runner_requires_a_prior_file(self) -> None:
        import run_synthetic_arms as rsa

        argv = ["run_synthetic_arms.py", "--label", "x", "--arms", "vanilla-peaks", "--out-root", "unused"]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit) as cm:
                rsa.main()
        self.assertIn("--prior-file", str(cm.exception))

    def test_collector_reads_vanilla_peaks_draw_major(self) -> None:
        import json
        import tempfile

        import collect_paper_results as cpr

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for d in (1, 2, 3):
                cell = root / "vanilla-peaks" / f"hartmann6-draw{d}"
                cell.mkdir(parents=True)
                runs = [[float(10 * d + s), 0.123456789012345] for s in range(2)]
                (cell / "trajectories.json").write_text(json.dumps({"hartmann6": {"vanilla-peaks": runs}}))
            out = cpr.collect_synthetic(root, "rho90")
            other = {card: list(cpr.collect_synthetic(root, card)) for card in ("well50", "well10")}
        self.assertEqual(list(out), ["hapbo", "pointonly", "gaussian", "colabo", "lgbo", "llambo", "bora",
                                     "vanilla-peaks"])
        # the representation arms (pointonly, gaussian) belong to the rho90 card only
        for card, arms in other.items():
            self.assertEqual(arms, ["hapbo", "colabo", "lgbo", "llambo", "bora", "vanilla-peaks"], card)
        self.assertEqual([t[0] for t in out["vanilla-peaks"]["hartmann6"]], [10.0, 11.0, 20.0, 21.0, 30.0, 31.0])
        self.assertEqual(out["vanilla-peaks"]["hartmann6"][0][1], 0.123456789)       # 10 significant digits
        self.assertEqual(out["vanilla-peaks"]["ackley15"], [])
        self.assertEqual(out["hapbo"]["hartmann6"], [])


if __name__ == "__main__":
    unittest.main()
