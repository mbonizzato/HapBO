"""Prior-injection arms of the continuous path: run_continuous_bo(prior_strategy=...).

pi-bo, prior-to-mean, prior-warm-start and prior-greedy are the finite-grid arms of fig_ablations (experiments.worker) on
the continuous synthetic path. They locate and score the same prior peaks as the alpha-pi arm; their candidate set
C0 is the distinct points of the alpha-pi arm's first acquisition batch (Sobol of seed + n_initial plus the peaks).
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "scripts")]

from bayesian_optimization import acquisition, continuous_bo  # noqa: E402
from bayesian_optimization.acquisition import PI_BO_PRIOR_EPSILON, pi_bo_prior, pi_bo_shape  # noqa: E402
from bayesian_optimization.bo import pi_bo_initial_design  # noqa: E402
from bayesian_optimization.continuous_bo import DISTINCT_TOL, distinct_by_prior, run_continuous_bo  # noqa: E402

D = 3
BOUNDS = np.array([[-2.0, 2.0]] * D)
X_OPT = np.array([0.7, -1.1, 0.3])
PEAK_A = np.array([1.2, -0.4, -1.0])      # main prior mode (native)
PEAK_B = np.array([-1.3, 1.1, 0.9])       # second, lower mode
N_SOBOL = 128
INJECTION = ("pi-bo", "prior-to-mean", "prior-warm-start", "prior-greedy")


def objective(x) -> float:
    x = np.asarray(x, float)
    return float(np.sum((x - X_OPT) ** 2) + 0.2 * np.sum(np.sin(3.0 * x)))


def make_prior(bimodal: bool = True):
    def prior_fn(x) -> float:
        x = np.asarray(x, float)
        v = np.exp(-np.sum((x - PEAK_A) ** 2) / (2 * 0.35 ** 2))
        if bimodal:
            v += 0.9 * np.exp(-np.sum((x - PEAK_B) ** 2) / (2 * 0.3 ** 2))
        return float(0.05 + v)

    prior_fn.peak_candidates = np.vstack([PEAK_A, PEAK_B]) if bimodal else PEAK_A[None, :]
    return prior_fn


def to_unit(x):
    return (np.asarray(x, float) - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])


def to_native(u):
    return BOUNDS[:, 0] + (BOUNDS[:, 1] - BOUNDS[:, 0]) * np.atleast_2d(u)


def run(**kw):
    args = dict(budget=8, n_initial=3, n_sobol=N_SOBOL, n_restarts=2, seed=5)
    args.update(kw)
    return run_continuous_bo(objective, BOUNDS, 0.0, **args)


def prior_values(prior_fn, U):
    return np.array([prior_fn(x) for x in to_native(U)])


def candidate_set(prior_fn, seed=5):
    """C0 rows before deduplication = the alpha-pi arm's first acquisition batch (Sobol of seed + n_initial, then
    its prior peaks), and their prior values."""
    hap, batches = spy_batches(alpha=0.325, prior_fn=prior_fn, seed=seed, budget=4)
    cand = np.vstack([continuous_bo._sobol(N_SOBOL, D, seed + 3), hap.prior_peaks])
    np.testing.assert_array_equal(batches[0], cand)
    return cand, prior_values(prior_fn, cand), hap.prior_peaks


def spy_batches(**kw):
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


class DistinctTests(unittest.TestCase):
    def test_descending_stable_and_merges_near_duplicates(self) -> None:
        U = np.array([[0.1, 0.1], [0.5, 0.5], [0.5 + 1e-5, 0.5 - 1e-5], [0.9, 0.2], [0.3, 0.3]])
        v = np.array([0.2, 0.9, 0.95, 0.9, 0.2])
        np.testing.assert_array_equal(distinct_by_prior(U, v), [2, 3, 0, 4])        # row 1 merged into row 2
        np.testing.assert_array_equal(distinct_by_prior(U, v, limit=2), [2, 3])
        np.testing.assert_array_equal(distinct_by_prior(U, v, tol=0.0), [2, 1, 3, 0, 4])
        far = U.copy()
        far[2] = [0.5 + 2 * DISTINCT_TOL, 0.5]
        self.assertEqual(len(distinct_by_prior(far, v)), 5)


class InjectionArmTests(unittest.TestCase):
    def test_the_initial_pool_is_the_alpha_pi_first_batch(self) -> None:
        prior = make_prior()
        hap, hap_batches = spy_batches(alpha=0.325, prior_fn=prior)
        for arm in INJECTION:
            with self.subTest(arm=arm):
                pools = []
                real = continuous_bo.distinct_by_prior

                def spy(U, values, *a, **k):
                    pools.append(np.array(U, copy=True))
                    return real(U, values, *a, **k)

                with mock.patch.object(continuous_bo, "distinct_by_prior", side_effect=spy):
                    run(prior_fn=prior, prior_strategy=arm, prior_mean_beta=0.4)
                self.assertEqual(len(pools), 1)
                np.testing.assert_array_equal(pools[0], hap_batches[0])

    def test_every_injection_arm_shares_the_alpha_pi_prior_peaks_and_batches(self) -> None:
        prior = make_prior()
        hap, _ = spy_batches(alpha=0.325, prior_fn=prior)
        k = len(hap.prior_peaks)
        self.assertEqual(k, 8 + 2)
        for arm in ("pi-bo", "prior-to-mean", "prior-warm-start"):
            with self.subTest(arm=arm):
                res, batches = spy_batches(prior_fn=prior, prior_strategy=arm, prior_mean_beta=0.4)
                np.testing.assert_array_equal(res.prior_peaks, hap.prior_peaks)
                self.assertEqual(len(batches), 8 - 3)
                for step, batch in enumerate(batches):
                    np.testing.assert_array_equal(batch[-k:], hap.prior_peaks)
                    np.testing.assert_array_equal(batch[:N_SOBOL], continuous_bo._sobol(N_SOBOL, D, 5 + 3 + step))

    def test_alpha_does_not_enter_the_injection_arms(self) -> None:
        prior = make_prior()
        for arm in INJECTION:
            with self.subTest(arm=arm):
                a = run(prior_fn=prior, prior_strategy=arm, prior_mean_beta=0.4)
                b = run(prior_fn=prior, prior_strategy=arm, prior_mean_beta=0.4, alpha=0.9)
                np.testing.assert_array_equal(a.X_native, b.X_native)

    def test_warm_start_design_is_the_top_distinct_prior_points_of_c0(self) -> None:
        prior = make_prior()
        cand, pc, peaks = candidate_set(prior)
        res = run(prior_fn=prior, prior_strategy="prior-warm-start")
        want = cand[distinct_by_prior(cand, pc, limit=3)]
        np.testing.assert_array_equal(res.X_native[:3], to_native(want))
        # the prior argmax first, then the second mode (the 9 other climbs of mode A are merged)
        self.assertLess(np.max(np.abs(to_unit(res.X_native[0]) - to_unit(PEAK_A))), 1e-3)
        self.assertLess(np.max(np.abs(to_unit(res.X_native[1]) - to_unit(PEAK_B))), 1e-2)
        self.assertGreater(np.max(np.abs(res.X_native[2] - res.X_native[0])), 1e-3)
        pv = prior_values(prior, to_unit(res.X_native[:3]))
        self.assertTrue(np.all(np.diff(pv) <= 0))

    def test_greedy_follows_the_prior_without_a_gp(self) -> None:
        prior = make_prior()
        cand, pc, _ = candidate_set(prior)
        with mock.patch.object(continuous_bo.GaussianProcessSurrogate, "fit", side_effect=AssertionError("GP fit")):
            res = run(prior_fn=prior, prior_strategy="prior-greedy", budget=20)
        np.testing.assert_array_equal(res.X_native, to_native(cand[distinct_by_prior(cand, pc, limit=20)]))
        self.assertTrue(np.all(np.diff(prior_values(prior, to_unit(res.X_native))) <= 0))
        self.assertEqual(len(res.simple_regret), 20)
        with self.assertRaises(ValueError):
            run(prior_fn=prior, prior_strategy="prior-greedy", budget=N_SOBOL + 50)

    def test_pi_bo_initial_design_is_bo_pi_bo_initial_design_on_c0(self) -> None:
        prior = make_prior()
        cand, pc, _ = candidate_set(prior)
        keep = np.sort(distinct_by_prior(cand, pc))
        for seed_shift in (0, 1):
            seed = 5 + seed_shift
            if seed_shift:
                cand, pc, _ = candidate_set(prior, seed=seed)
                keep = np.sort(distinct_by_prior(cand, pc))
            res = run(prior_fn=prior, prior_strategy="pi-bo", seed=seed)
            want = cand[keep][list(pi_bo_initial_design(pc[keep], 3, seed))]
            np.testing.assert_array_equal(res.X_native[:3], to_native(want))
        a = run(prior_fn=prior, prior_strategy="pi-bo")
        np.testing.assert_array_equal(a.X_native, run(prior_fn=prior, prior_strategy="pi-bo").X_native)

    def test_pi_bo_starts_at_the_prior_mode_of_c0(self) -> None:
        prior = make_prior()
        for seed in (5, 6, 7):
            cand, pc, _ = candidate_set(prior, seed=seed)
            keep = np.sort(distinct_by_prior(cand, pc))
            res = run(prior_fn=prior, prior_strategy="pi-bo", seed=seed)
            mode = cand[keep][int(np.argmax(pc[keep]))]
            np.testing.assert_array_equal(res.X_native[0], to_native(mode)[0])           # the mode first
            self.assertLess(np.max(np.abs(to_unit(res.X_native[0]) - to_unit(PEAK_A))), 1e-3)
            for x in res.X_native[1:3]:                                                  # then sampled, not the mode
                self.assertGreater(np.max(np.abs(x - res.X_native[0])), 1e-9)

    def test_pi_bo_beta_auto_is_n_over_10(self) -> None:
        prior = make_prior()
        for budget, n_init, kw in ((8, 3, {}), (13, 3, {"pi_bo_beta": "auto"}), (9, 2, {}), (8, 3, {"pi_bo_beta": 0.5})):
            seen = []

            def spy(raw, prior, beta, t, valid_mask=None, reference_max=None):
                seen.append(float(beta))
                return pi_bo_shape(raw, prior, beta, t, valid_mask, reference_max)

            with mock.patch.object(acquisition, "pi_bo_shape", side_effect=spy):
                res = run(prior_fn=prior, prior_strategy="pi-bo", budget=budget, n_initial=n_init, **kw)
            want = kw.get("pi_bo_beta", "auto")
            want = (budget - n_init) / 10 if want == "auto" else want
            self.assertTrue(seen and all(b == want for b in seen), (budget, n_init, kw, set(seen)))
            self.assertEqual(res.pi_bo_beta, want)
        self.assertIsNone(run(prior_fn=prior, prior_strategy="prior-warm-start").pi_bo_beta)
        with self.assertRaises(ValueError):
            run(prior_fn=prior, prior_strategy="pi-bo", pi_bo_beta="N/10")

    def test_pi_bo_weighting_and_decay_counter(self) -> None:
        prior = make_prior()
        cand, pc, _ = candidate_set(prior)
        c0_max = float(pc[distinct_by_prior(cand, pc)].max())
        calls = []

        def spy(raw, prior_values, beta, t, valid_mask=None, reference_max=None):
            calls.append((len(np.atleast_1d(raw)), float(beta), float(t), reference_max))
            return pi_bo_shape(raw, prior_values, beta, t, valid_mask, reference_max)

        for n_initial, ts in ((2, [1.0, 2.0, 3.0, 4.0]), (0, [1.0, 2.0, 3.0, 4.0, 5.0])):
            calls.clear()
            with mock.patch.object(acquisition, "pi_bo_shape", side_effect=spy):
                run(prior_fn=prior, prior_strategy="pi-bo", n_initial=n_initial, budget=6, pi_bo_beta=2.0)
            batch_calls = [c for c in calls if c[0] > N_SOBOL - 1]
            self.assertEqual([c[2] for c in batch_calls], ts)                  # t = 1 at the first BO query
            self.assertTrue(all(c[1] == 2.0 for c in calls))
        self.assertTrue(all(c[3] == c0_max for c in calls))                     # piBO's normalizer: the C0 maximum

    def test_reference_max_keeps_pi_bo_prior(self) -> None:
        rng = np.random.default_rng(0)
        raw, pv = rng.random(50), rng.random(50)
        for t in (1, 2, 5):    # the reference max of a batch that holds it: the finite-grid prior
            np.testing.assert_array_equal(pi_bo_shape(raw, pv, 1.5, t, reference_max=float(pv.max())),
                                          pi_bo_shape(raw, pv, 1.5, t))
        p = pi_bo_prior(np.array([-1.0, 0.25, 0.5, 2.0]), reference_max=1.0)   # values above the reference capped
        np.testing.assert_array_equal(p, np.array([0.0, 0.25, 0.5, 1.0]) + PI_BO_PRIOR_EPSILON)
        np.testing.assert_array_equal(pi_bo_prior(np.array([0.3, 0.7]), reference_max=0.0), np.ones(2))

    def test_pi_bo_flat_acquisition_goes_to_the_prior_mode(self) -> None:
        prior = make_prior()

        def flat(self, mean, std, best_value, prior=None):
            return np.ones(len(np.atleast_1d(mean)))

        with mock.patch.object(continuous_bo.ExpectedImprovement, "score", flat):
            res = run(prior_fn=prior, prior_strategy="pi-bo", budget=5)
        for x in res.X_native[3:]:
            self.assertLess(np.max(np.abs(to_unit(x) - to_unit(PEAK_A))), 1e-3)

    def test_prior_to_mean_is_beta_times_the_standardized_prior_over_c0(self) -> None:
        from experiments.worker import _standardize_pi

        prior = make_prior()
        cand, pc, _ = candidate_set(prior)
        keep = np.sort(distinct_by_prior(cand, pc))
        made = []
        real = continuous_bo.GaussianProcessSurrogate

        def recording(*a, **k):
            made.append(k)
            return real(*a, **k)

        with mock.patch.object(continuous_bo, "GaussianProcessSurrogate", side_effect=recording):
            res = run(prior_fn=prior, prior_strategy="prior-to-mean", prior_mean_beta=0.4)
        m = made[0]["prior_mean"]
        np.testing.assert_array_equal(m(cand[keep]), 0.4 * _standardize_pi(pc[keep]))
        self.assertTrue(made[0]["normalize_y"])
        # vanilla Sobol initial design, then a different run
        van = run()
        np.testing.assert_array_equal(res.X_native[:3], van.X_native[:3])
        self.assertFalse(np.array_equal(res.X_native, van.X_native))

    def test_constant_prior_has_no_peaks(self) -> None:
        def flat(x) -> float:
            return 1.0

        flat.peak_candidates = PEAK_A[None, :]
        van, van_batches = spy_batches()
        for arm in ("prior-to-mean", "prior-warm-start"):
            with self.subTest(arm=arm):
                res, batches = spy_batches(prior_fn=flat, prior_strategy=arm, prior_mean_beta=0.4)
                self.assertIsNone(res.prior_peaks)
                self.assertTrue(all(len(b) == N_SOBOL for b in batches))
                for b, v in zip(batches, van_batches):
                    np.testing.assert_array_equal(b, v)                  # vanilla's candidate batches
                if arm == "prior-to-mean":                               # zero prior mean: the vanilla run
                    np.testing.assert_array_equal(res.X_native, van.X_native)
                else:                                                    # ties: the first rows of the first batch
                    first = continuous_bo._sobol(N_SOBOL, D, 5 + 3)[:3]
                    np.testing.assert_array_equal(res.X_native[:3], to_native(first))

    def test_validation(self) -> None:
        prior = make_prior()
        bad = [dict(prior_strategy="pibo", prior_fn=prior), dict(prior_strategy="pi-bo"),
               dict(prior_strategy="pi-bo", prior_fn=prior, shape_acquisition=False),
               dict(prior_strategy="prior-to-mean", prior_fn=prior),
               dict(prior_strategy="pi-bo", prior_fn=prior, pi_bo_beta=-1.0)]
        for kw in bad:
            with self.subTest(kw=sorted(kw)):
                with self.assertRaises(ValueError):
                    run(**kw)

    def test_default_strategy_is_alpha_pi(self) -> None:
        prior = make_prior()
        for kw in (dict(), dict(alpha=0.5, prior_fn=prior), dict(alpha=0.5, prior_fn=prior, shape_acquisition=False)):
            a, b = run(**kw), run(**kw, prior_strategy="alpha-pi")
            np.testing.assert_array_equal(a.X_native, b.X_native)
            np.testing.assert_array_equal(a.simple_regret, b.simple_regret)


class RunnerTests(unittest.TestCase):
    PRIOR = PROJECT_ROOT / "experiments/synthetic_priors/rho90/hartmann6/draw1/prior.json"
    PEAK = 0.3

    def run_panel(self, arms, *extra):
        import run_synthetic_arms as rsa

        calls = []
        rcb = continuous_bo.run_continuous_bo

        def recording(*a, **k):
            res = rcb(*a, **k)
            calls.append(k)
            return res

        def loader(family, dim, prior_file, document_peaks=True):
            peak = np.full(dim, self.PEAK)

            def prior_fn(x) -> float:
                return float(np.exp(-np.sum((np.asarray(x, float) - peak) ** 2) / (2 * 0.1 ** 2)))

            prior_fn.peak_candidates = peak[None, :]
            return prior_fn, 0.7

        with tempfile.TemporaryDirectory() as tmp:
            argv = ["run_synthetic_arms.py", "--label", "e2e", "--out-root", tmp, "--families", "hartmann6",
                    "--arms", ",".join(arms), "--prior-file", str(self.PRIOR), "--alpha-override", "0.325",
                    "--restarts", "2", *extra]
            with mock.patch.object(sys, "argv", argv), mock.patch.dict(rsa.BUDGETS, {"hartmann6": 5}), \
                    mock.patch.object(rsa, "load_prior_fn", loader), \
                    mock.patch.object(continuous_bo, "run_continuous_bo", recording):
                rsa.main()
            traj = json.loads((Path(tmp) / "e2e" / "trajectories.json").read_text(encoding="utf-8"))["hartmann6"]
            prov = json.loads((Path(tmp) / "e2e" / "run_provenance.json").read_text(encoding="utf-8"))
        return calls, traj, prov

    def test_injection_arms_in_the_runner(self) -> None:
        calls, traj, prov = self.run_panel(list(INJECTION) + ["vanilla"], "--seeds", "1")
        self.assertEqual(set(traj), set(INJECTION) | {"vanilla"})
        for k, arm in zip(calls, INJECTION):
            self.assertEqual(k["prior_strategy"], arm)
            self.assertEqual(k["prior_mean_beta"], 0.7)            # compiled alpha, not --alpha-override
            self.assertEqual(k["pi_bo_beta"], "auto")
            self.assertNotIn("alpha", k)
            self.assertEqual(len(traj[arm][0]), 5)
        self.assertEqual(traj["prior-to-mean"][0][:3], traj["vanilla"][0][:3])
        self.assertEqual(prov["injection"]["arms"], list(INJECTION))
        self.assertEqual(prov["injection"]["pi_bo_beta"], "auto")
        self.assertEqual(prov["injection"]["pi_bo_beta_resolved"], {"hartmann6": 0.2})        # (5 - 3) / 10
        self.assertEqual(prov["injection"]["distinct_tol"], DISTINCT_TOL)

    def test_one_seed_per_process_equals_the_seed_in_a_full_run(self) -> None:
        _, full, _ = self.run_panel(["pi-bo", "prior-warm-start"], "--seeds", "2")
        _, one, _ = self.run_panel(["pi-bo", "prior-warm-start"], "--seed-start", "1", "--seeds", "2")
        for arm in ("pi-bo", "prior-warm-start"):
            self.assertEqual(one[arm], [full[arm][1]])

    def test_driver_and_collector(self) -> None:
        import collect_paper_results as cpr
        import run_synthetic_panel as rsp

        for arm in INJECTION:
            cmds = list(rsp.commands(arm, ["rho90"], ["hartmann6"], 10))
            self.assertEqual(len(cmds), 3)
            for d, c in enumerate(cmds, 1):
                self.assertEqual(c[c.index("--arms") + 1], arm)
                self.assertEqual(c[c.index("--prior-file") + 1],
                                 f"experiments/synthetic_priors/rho90/hartmann6/draw{d}/prior.json")
                self.assertEqual(c[c.index("--out-root") + 1], f"results/synthetic/well/rho90/{arm}")
                self.assertNotIn("--alpha-override", c)
            self.assertIn(arm, rsp.CONTROL_ARMS)
            self.assertNotIn(arm, rsp.ARMS)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for d in (1, 2, 3):
                cell = root / "pi-bo" / f"hartmann6-draw{d}"
                cell.mkdir(parents=True)
                runs = [[float(10 * d + s), 0.123456789012345] for s in range(2)]
                (cell / "trajectories.json").write_text(json.dumps({"hartmann6": {"pi-bo": runs}}))
            out = cpr.collect_synthetic(root, "rho90")
        self.assertIn("pi-bo", out)
        self.assertNotIn("prior-greedy", out)                          # listed only when present
        self.assertEqual([t[0] for t in out["pi-bo"]["hartmann6"]], [10.0, 11.0, 20.0, 21.0, 30.0, 31.0])
        self.assertEqual(out["pi-bo"]["hartmann6"][0][1], 0.123456789)
        self.assertEqual(out["pi-bo"]["ackley15"], [])


if __name__ == "__main__":
    unittest.main()
