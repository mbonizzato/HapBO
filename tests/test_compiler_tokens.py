"""LLM-free coverage of every canonical PSL compiler token."""

from __future__ import annotations

import unittest

from src.prior.landscape.landscape_generator import VALID_TOKENS, compile_prior


def _record(token: str) -> dict:
    dimensions = [
        {"name": "x", "type": "continuous", "range": [0.0, 1.0]},
        {"name": "y", "type": "continuous", "range": [0.0, 1.0]},
        {"name": "cat", "type": "categorical", "values": ["a", "b"]},
    ]
    shaping = {
        "bump": {"token": "bump", "dim": "x", "mode": 0.5, "ci": [0.3, 0.7]},
        "trend": {"token": "trend", "dim": "x", "direction": "increase"},
        "threshold": {"token": "threshold", "dim": "x", "at": 0.5, "active": "above"},
        "periodic": {"token": "periodic", "dim": "x", "period": 0.5, "phase": 0.0},
        "categorical": {"token": "categorical", "dim": "cat", "preferences": {"a": 0.1, "b": 1.0}},
        "joint_region": {"token": "joint_region", "dims": ["x", "y"], "mode": {"x": 0.5, "y": 0.5}, "ci": {"x": [0.3, 0.7], "y": [0.3, 0.7]}},
        "ridge": {"token": "ridge", "dims": ["x", "y"], "relation": "positive", "width": "narrow"},
        "conditional": {"token": "conditional", "when": {"dim": "y", "region": [0.4, 0.6]}, "then": {"token": "trend", "dim": "x", "direction": "increase"}},
        "plateau": {"token": "plateau", "region": {"x": [0.4, 0.6]}},
        "avoid": {"token": "avoid", "region": {"x": [0.4, 0.6]}},
    }
    if token in shaping:
        belief = {**shaping[token], "weight": 1.0, "provenance": "user_stated"}
        beliefs = [{"token": "baseline", "level": 0.1}, belief]
    elif token == "baseline":
        beliefs = [{"token": "baseline", "level": 0.4, "tilt": {"x": 0.5}}]
    elif token == "anchors":
        beliefs = [{"token": "baseline", "level": 0.1}, {"token": "anchors", "points": [{"coords": {"x": 0.25, "y": 0.5, "cat": "a"}, "value": 0.8}], "provenance": "user_stated"}]
    elif token == "smoothness":
        beliefs = [{"token": "smoothness", "level": "rough"}]
    elif token == "noise":
        beliefs = [{"token": "noise", "level": "high", "repeats_recommended": True}]
    elif token == "modality":
        beliefs = [{"token": "modality", "expectation": "many"}]
    else:  # pragma: no cover - fails loudly if the canonical vocabulary grows.
        raise AssertionError(f"no closed-form check for token {token!r}")
    return {
        "schema": {"problem": "compiler unit test", "objective": "maximize", "dimensions": dimensions},
        "beliefs": beliefs,
        "trust": {"overall_confidence": "high"},
    }


class CompilerTokenTests(unittest.TestCase):
    def test_all_fifteen_canonical_tokens(self) -> None:
        self.assertEqual(len(VALID_TOKENS), 15)
        self.assertEqual(set(VALID_TOKENS), set(_record(token)["beliefs"][-1]["token"] for token in VALID_TOKENS))
        for token in sorted(VALID_TOKENS):
            with self.subTest(token=token):
                prior = compile_prior(_record(token))
                low = {"x": 0.0, "y": 0.0, "cat": "a"}
                middle = {"x": 0.5, "y": 0.5, "cat": "a"}
                high = {"x": 1.0, "y": 1.0, "cat": "b"}
                if token == "bump":
                    self.assertGreater(prior.pi(middle), prior.pi(low))
                elif token in {"trend", "threshold"}:
                    self.assertGreater(prior.pi(high), prior.pi(low))
                elif token == "periodic":
                    trough = {"x": 0.25, "y": 0.5, "cat": "a"}
                    self.assertGreater(prior.pi(low), prior.pi(trough))
                elif token == "categorical":
                    preferred = {"x": 0.5, "y": 0.5, "cat": "b"}
                    self.assertGreater(prior.pi(preferred), prior.pi(middle))
                elif token == "joint_region":
                    self.assertGreater(prior.pi(middle), prior.pi(low))
                elif token == "ridge":
                    off_ridge = {"x": 0.0, "y": 1.0, "cat": "a"}
                    self.assertGreater(prior.pi(middle), prior.pi(off_ridge))
                elif token == "conditional":
                    active = {"x": 1.0, "y": 0.5, "cat": "a"}
                    inactive = {"x": 1.0, "y": 0.0, "cat": "a"}
                    self.assertGreater(prior.pi(active), prior.pi(inactive))
                elif token == "plateau":
                    self.assertGreater(prior.pi(middle), prior.pi(low))
                elif token == "avoid":
                    self.assertLess(prior.pi(middle), prior.pi(low))
                elif token == "anchors":
                    anchor = {"x": 0.25, "y": 0.5, "cat": "a"}
                    self.assertAlmostEqual(prior.pi(anchor), 0.8, places=12)
                elif token == "baseline":
                    self.assertGreater(prior.pi(high), prior.pi(low))
                elif token in {"smoothness", "noise", "modality"}:
                    # Meta tokens are valid PSL but do not shape the prior: pi stays flat.
                    self.assertEqual(prior.pi(middle), prior.pi(low))
                    self.assertEqual(prior.pi(high), prior.pi(low))


if __name__ == "__main__":
    unittest.main()
