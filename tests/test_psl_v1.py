"""PSL v1.0 compiler semantics: nested troughs, nested validation, periodic in native units,
ridge points in the reference set, and categorical anchor strata."""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.prior.landscape.landscape_generator import (
    PSL_VERSION,
    compile_prior,
    validate_record,
)


def _doc(dimensions, beliefs, confidence="medium"):
    return {
        "schema": {"objective": "maximize", "dimensions": dimensions},
        "beliefs": beliefs,
        "trust": {"overall_confidence": confidence},
    }


XY = [
    {"name": "x", "type": "continuous", "range": [0.0, 1.0]},
    {"name": "y", "type": "continuous", "range": [0.0, 1.0]},
]
ANCHOR_WEIGHT_ONE = {"token": "trend", "dim": "y", "direction": "increase", "weight": 1.0,
                     "provenance": "user_stated"}


class VersionTest(unittest.TestCase):
    def test_version(self):
        self.assertEqual(PSL_VERSION, "1.0")


class NestedShapeTest(unittest.TestCase):
    def _conditional(self, mode, sign):
        then = {"token": "bump", "dim": "x", "mode": 0.5, "ci": [0.4, 0.6], "sign": sign}
        return _doc(XY, [
            {"token": "baseline", "level": 0.5},
            ANCHOR_WEIGHT_ONE,
            {"token": "conditional", "when": {"dim": "y", "region": [0.0, 1.0]}, "then": then,
             "mode": mode, "weight": 0.5, "provenance": "user_stated"},
        ])

    def test_gate_mode_trough_lowers_the_score_at_its_mode(self):
        peak = compile_prior(self._conditional("gate", "peak"))
        trough = compile_prior(self._conditional("gate", "trough"))
        u = np.array([[0.5, 0.5], [0.05, 0.5]])
        s_peak, s_trough = peak._raw_score(u), trough._raw_score(u)
        self.assertGreater(s_peak[0], s_peak[1])
        self.assertLess(s_trough[0], s_trough[1])
        # Symmetric around the shape-free score: +w*phi for the peak, -w*phi for the trough.
        base = 0.5 + 0.5
        np.testing.assert_allclose(s_peak + s_trough, 2 * base, atol=1e-12)

    def test_synergy_mode_trough_is_a_dip_gate(self):
        prior = compile_prior(self._conditional("synergy", "trough"))
        u = np.array([[0.5, 0.5], [0.05, 0.5]])
        s = prior._raw_score(u)
        # Gate 1 - w*(1 - (1 - bump)) = 1 - w*bump: lowest at the trough's mode.
        self.assertLess(s[0], s[1])
        np.testing.assert_allclose(s[0], 1.0 * (1 - 0.5), atol=1e-12)

    @staticmethod
    def _with_then(then, mode="gate"):
        return _doc(XY, [ANCHOR_WEIGHT_ONE, {
            "token": "conditional", "when": {"dim": "y", "region": [0.0, 0.5]}, "then": then,
            "mode": mode, "weight": 0.5, "provenance": "user_stated"}])

    def test_nested_fields_are_validated(self):
        bad = [
            {"token": "bump", "dim": "x", "mode": 0.5, "weight": 1.5},
            {"token": "bump", "dim": "x", "mode": 0.5, "provenance": "hearsay"},
            {"token": "bump", "dim": "x", "mode": 0.5, "skew": 2.0},
            {"token": "bump", "dim": "x", "mode": 0.5, "sign": "valley"},
            {"token": "bump", "dim": "x", "mode": 0.5, "note": 3},
            {"token": "threshold", "dim": "x", "at": 0.5, "active": "above", "sharpness": "crisp"},
            {"token": "bump", "dim": "x", "mode": 0.5, "unknown": 1},
            {"token": "plateau", "region": {"x": [0.2, 0.4]}, "dim": "x"},
        ]
        for then in bad:
            with self.subTest(then=then), self.assertRaises(ValueError):
                validate_record(self._with_then(then))
        good = {"token": "bump", "dim": "x", "mode": 0.5, "ci": [0.4, 0.6], "skew": -0.5,
                "note": "ok"}
        validate_record(self._with_then(good))

    def test_nested_weight_and_provenance_are_accepted_and_ignored(self):
        then = {"token": "bump", "dim": "x", "mode": 0.5, "ci": [0.4, 0.6]}
        restated = {**then, "weight": 0.8, "provenance": "llm_domain"}
        plain, extra = compile_prior(self._with_then(then)), compile_prior(self._with_then(restated))
        u = np.random.default_rng(0).random((500, 2))
        np.testing.assert_array_equal(plain._pi_unit(u), extra._pi_unit(u))
        self.assertEqual(plain.alpha, extra.alpha)

    def test_synergy_trough_acts_only_inside_the_when_region(self):
        then = {"token": "bump", "dim": "x", "mode": 0.5, "ci": [0.4, 0.6], "sign": "trough"}
        prior = compile_prior(self._with_then(then, mode="synergy"))
        inside, outside = np.array([[0.5, 0.25]]), np.array([[0.5, 0.75]])
        # Score before gating: default baseline 0.1 plus the trend in y. Outside y in [0, 0.5]
        # the gate is 1; inside it dips to 1 - w at the trough's mode.
        self.assertAlmostEqual(prior._raw_score(inside)[0], (0.1 + 0.25) * 0.5, places=12)
        self.assertAlmostEqual(prior._raw_score(outside)[0], 0.1 + 0.75, places=12)


class PeriodicTest(unittest.TestCase):
    def _shape(self, dim, period, phase, native_points):
        doc = _doc([dim], [{"token": "periodic", "dim": dim["name"], "period": period,
                            "phase": phase, "weight": 1.0, "provenance": "user_stated"}])
        prior = compile_prior(doc)
        belief = doc["beliefs"][0]
        unit, _ = prior._encode([[p] for p in native_points])
        return prior._periodic(belief, unit)

    def test_linear_period_extending_past_the_range_is_not_truncated(self):
        dim = {"name": "x", "type": "continuous", "range": [0.0, 10.0]}
        values = self._shape(dim, period=4.0, phase=8.0, native_points=[8.0, 10.0, 0.0, 4.0])
        np.testing.assert_allclose(values, [1.0, 0.0, 1.0, 1.0], atol=1e-12)

    def test_linear_in_range_is_periodic_in_native_units(self):
        dim = {"name": "x", "type": "continuous", "range": [0.0, 10.0]}
        values = self._shape(dim, period=2.5, phase=1.0, native_points=[1.0, 3.5, 6.0, 2.25])
        np.testing.assert_allclose(values, [1.0, 1.0, 1.0, 0.0], atol=1e-12)

    def test_log_range_is_periodic_in_native_units(self):
        dim = {"name": "x", "type": "continuous", "range": [1.0, 100.0], "scale": "log"}
        values = self._shape(dim, period=20.0, phase=10.0, native_points=[10.0, 30.0, 50.0, 20.0])
        np.testing.assert_allclose(values, [1.0, 1.0, 1.0, 0.0], atol=1e-9)

    def test_ordinal_period_uses_native_values(self):
        dim = {"name": "x", "type": "ordinal", "values": [1, 2, 4, 8, 16]}
        values = self._shape(dim, period=4.0, phase=4.0, native_points=[4, 8, 16, 2])
        np.testing.assert_allclose(values, [1.0, 1.0, 1.0, 0.0], atol=1e-12)

    def test_non_numeric_ordinal_is_rejected(self):
        dim = {"name": "x", "type": "ordinal", "values": ["low", "mid", "high"]}
        doc = _doc([dim], [{"token": "periodic", "dim": "x", "period": 1.0, "weight": 1.0,
                            "provenance": "user_stated"}])
        with self.assertRaises(ValueError):
            validate_record(doc)

    def test_phase_defaults_to_zero_in_native_units(self):
        dim = {"name": "x", "type": "continuous", "range": [5.0, 50.0]}
        doc = _doc([dim], [{"token": "periodic", "dim": "x", "period": 10.0, "weight": 1.0,
                            "provenance": "user_stated"}])
        prior = compile_prior(doc)
        unit, _ = prior._encode([[10.0], [20.0], [15.0]])
        np.testing.assert_allclose(prior._periodic(doc["beliefs"][0], unit), [1.0, 1.0, 0.0],
                                   atol=1e-12)

    def test_phase_outside_the_range_is_accepted(self):
        dim = {"name": "x", "type": "continuous", "range": [5.0, 50.0]}
        values = self._shape(dim, period=10.0, phase=-5.0, native_points=[5.0, 15.0, 10.0])
        np.testing.assert_allclose(values, [1.0, 1.0, 0.0], atol=1e-12)

    def test_phase_at_the_upper_bound(self):
        dim = {"name": "x", "type": "continuous", "range": [0.0, 10.0]}
        values = self._shape(dim, period=4.0, phase=10.0, native_points=[10.0, 6.0, 8.0])
        np.testing.assert_allclose(values, [1.0, 1.0, 0.0], atol=1e-12)

    def test_single_level_ordinal(self):
        dim = {"name": "x", "type": "ordinal", "values": [3]}
        values = self._shape(dim, period=2.0, phase=3.0, native_points=[3])
        np.testing.assert_allclose(values, [1.0], atol=1e-12)

    def test_vanishing_period_is_rejected(self):
        dim = {"name": "x", "type": "continuous", "range": [1.0, 1e12], "scale": "log"}
        doc = _doc([dim], [{"token": "periodic", "dim": "x", "period": 1e-300, "weight": 1.0,
                            "provenance": "user_stated"}])
        with self.assertRaises(ValueError):
            validate_record(doc)

    def test_in_range_linear_path_matches_the_closed_form_encoded_formula(self):
        dim = {"name": "x", "type": "continuous", "range": [-3.0, 7.0]}
        belief = {"token": "periodic", "dim": "x", "period": 2.3, "phase": 0.7, "weight": 1.0,
                  "provenance": "user_stated"}
        prior = compile_prior(_doc([dim], [belief]))
        unit = np.random.default_rng(1).random((1000, 1))
        phase = (0.7 - (-3.0)) / 10.0
        period = max(abs((0.7 + 2.3 - (-3.0)) / 10.0 - phase), np.finfo(float).eps)
        expected = 0.5 * (1.0 + np.cos(2 * np.pi * (unit[:, 0] - phase) / period))
        np.testing.assert_array_equal(prior._periodic(belief, unit), expected)


class RidgeReferenceTest(unittest.TestCase):
    def test_ridge_through_point_sets_the_top_of_the_scale(self):
        # The line through (0.9, 0.2) misses the centre and the corners, so only the declared
        # point attains the ridge's exact maximum.
        doc = _doc(XY, [
            {"token": "baseline", "level": 0.1},
            {"token": "ridge", "dims": ["x", "y"], "relation": "positive", "width": "narrow",
             "through": {"x": 0.9, "y": 0.2}, "weight": 1.0, "provenance": "user_stated"},
        ])
        prior = compile_prior(doc)
        through = np.array([[0.9, 0.2]])
        self.assertEqual(prior._normalization_max, prior._raw_score(through)[0])
        self.assertEqual(prior._pi_unit(through)[0], 1.0)


class CategoricalAnchorTest(unittest.TestCase):
    def _prior(self):
        dims = [
            {"name": "x", "type": "continuous", "range": [0.0, 1.0]},
            {"name": "c", "type": "categorical", "values": ["a", "b", "c", "d"]},
        ]
        return compile_prior(_doc(dims, [
            {"token": "baseline", "level": 0.1},
            {"token": "anchors", "bandwidth": 0.3, "provenance": "user_stated",
             "points": [{"coords": {"x": 0.5, "c": "b"}, "value": 0.9}]},
        ]))

    def test_anchor_is_exact_and_confined_to_its_stratum(self):
        prior = self._prior()
        self.assertEqual(prior.pi({"x": 0.5, "c": "b"}), 0.9)
        baseline = prior.pi({"x": 0.5, "c": "d"})
        self.assertAlmostEqual(baseline, 0.1, places=12)
        for level in ("a", "c"):
            self.assertEqual(prior.pi({"x": 0.5, "c": level}), baseline)
        self.assertGreater(prior.pi({"x": 0.6, "c": "b"}), baseline)

    def test_several_categorical_dimensions_and_all_categorical_schema(self):
        dims = [
            {"name": "x", "type": "continuous", "range": [0.0, 1.0]},
            {"name": "c", "type": "categorical", "values": ["a", "b"]},
            {"name": "k", "type": "categorical", "values": ["p", "q", "r"]},
        ]
        prior = compile_prior(_doc(dims, [
            {"token": "baseline", "level": 0.1},
            {"token": "anchors", "bandwidth": 0.3, "provenance": "user_stated",
             "points": [{"coords": {"x": 0.5, "c": "b", "k": "q"}, "value": 0.9}]},
        ]))
        self.assertEqual(prior.pi({"x": 0.5, "c": "b", "k": "q"}), 0.9)
        self.assertGreater(prior.pi({"x": 0.6, "c": "b", "k": "q"}), 0.1 + 1e-9)
        for c, k in (("a", "q"), ("b", "p"), ("b", "r"), ("a", "r")):
            self.assertAlmostEqual(prior.pi({"x": 0.6, "c": c, "k": k}), 0.1, places=12)
        only = compile_prior(_doc(dims[1:], [
            {"token": "baseline", "level": 0.1},
            {"token": "anchors", "provenance": "user_stated",
             "points": [{"coords": {"c": "a", "k": "r"}, "value": 0.7}]},
        ]))
        self.assertEqual(only.pi({"c": "a", "k": "r"}), 0.7)
        self.assertAlmostEqual(only.pi({"c": "b", "k": "r"}), 0.1, places=12)

    def test_distance_uses_geometric_dimensions_only(self):
        prior = self._prior()
        # One geometric dimension: support radius h*sqrt(1) = 0.3 around x = 0.5.
        self.assertEqual(prior.pi({"x": 0.85, "c": "b"}), prior.pi({"x": 0.85, "c": "a"}))
        self.assertGreater(prior.pi({"x": 0.75, "c": "b"}), prior.pi({"x": 0.75, "c": "a"}))
        self.assertTrue(math.isfinite(prior.pi({"x": 1.0, "c": "b"})))


if __name__ == "__main__":
    unittest.main()
