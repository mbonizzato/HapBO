"""A constant shaping score compiles to the baseline level.

When the raw PSL score is constant (a document with only ``baseline``, ``anchors`` and meta
tokens), min-max normalization is undefined; the pre-anchor field is ``baseline.level``
(default 0.1), so candidates not covered by any anchor take the baseline value and the
anchors are blended in as for any other document.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prior.landscape.landscape_generator import DEFAULT_BASELINE, compile_prior, load_record  # noqa: E402


def _two_d(beliefs: list[dict]) -> dict:
    return {
        "schema": {
            "objective": "maximize",
            "dimensions": [
                {"name": "x", "type": "continuous", "range": [0.0, 1.0]},
                {"name": "y", "type": "continuous", "range": [0.0, 1.0]},
            ],
        },
        "beliefs": beliefs,
        "trust": {"overall_confidence": "high"},
    }


ANCHOR = {"x": 0.3, "y": 0.7}


def _anchors(value: float = 0.9, coords: dict = ANCHOR) -> dict:
    return {
        "token": "anchors",
        "points": [{"coords": dict(coords), "value": value}],
        "provenance": "user_stated",
    }


def _grid(n: int = 101) -> list[dict]:
    axis = np.linspace(0.0, 1.0, n)
    return [{"x": float(a), "y": float(b)} for a in axis for b in axis]


def test_baseline_plus_one_anchor():
    """baseline 0.1 + one anchor 0.9 -> 0.9 at the anchor, 0.1 elsewhere."""

    prior = compile_prior(_two_d([{"token": "baseline", "level": 0.1}, _anchors(0.9)]))
    assert prior.pi(ANCHOR) == pytest.approx(0.9, abs=1e-12)
    points = _grid()
    values = np.asarray(prior.pi(points))
    assert points[int(np.argmax(values))] == pytest.approx(ANCHOR)
    assert values.max() == pytest.approx(0.9, abs=1e-12)
    # Outside the compact Wendland support of the anchor the field is the baseline level.
    far = np.asarray([np.hypot(p["x"] - 0.3, p["y"] - 0.7) / np.sqrt(2) >= 0.15 for p in points])
    assert far.sum() > 0.8 * len(points)
    np.testing.assert_array_equal(values[far], np.full(int(far.sum()), 0.1))
    assert np.all((values[~far] >= 0.1 - 1e-12) & (values[~far] <= 0.9 + 1e-12))


def test_anchors_only_without_baseline_uses_default_level():
    prior = compile_prior(_two_d([_anchors(0.9), {"token": "smoothness", "level": "smooth"}]))
    assert DEFAULT_BASELINE == 0.1
    assert prior.pi(ANCHOR) == pytest.approx(0.9, abs=1e-12)
    assert prior.pi({"x": 0.9, "y": 0.1}) == 0.1
    values = np.asarray(prior.pi(_grid()))
    assert _grid()[int(np.argmax(values))] == pytest.approx(ANCHOR)


def test_low_anchor_below_baseline_is_a_trough():
    prior = compile_prior(_two_d([{"token": "baseline", "level": 0.4}, _anchors(0.05)]))
    assert prior.pi(ANCHOR) == pytest.approx(0.05, abs=1e-12)
    assert prior.pi({"x": 0.9, "y": 0.1}) == 0.4


def test_constant_document_without_anchors_is_flat_at_baseline():
    prior = compile_prior(_two_d([{"token": "baseline", "level": 0.25}]))
    values = np.asarray(prior.pi(_grid(11)))
    np.testing.assert_array_equal(values, np.full(len(values), 0.25))
    # The alpha-pi multiplier is a positive constant, so the acquisition argmax is vanilla's.
    m = np.asarray(prior.multiplier(_grid(11)))
    assert np.ptp(m) == 0.0 and m[0] > 0


def test_finite_domain_table():
    record = {
        "schema": {
            "objective": "maximize",
            "dimensions": [{"name": "c", "type": "categorical", "values": ["a", "b", "c", "d"]}],
        },
        "beliefs": [
            {"token": "baseline", "level": 0.2},
            {
                "token": "anchors",
                "points": [{"coords": {"c": "b"}, "value": 0.7}, {"coords": {"c": "d"}, "value": 0.95}],
                "provenance": "user_stated",
                "bandwidth": 0.01,
            },
        ],
        "trust": {"overall_confidence": "medium"},
    }
    prior = compile_prior(record)
    got = [prior.pi({"c": v}) for v in ["a", "b", "c", "d"]]
    assert got == pytest.approx([0.2, 0.7, 0.2, 0.95], abs=1e-12)


def test_mixed_document_is_min_max_normalized():
    record = _two_d([
        {"token": "baseline", "level": 0.1, "tilt": {"x": 0.2}},
        {"token": "bump", "dim": "y", "mode": 0.4, "ci": [0.2, 0.6], "weight": 1.0, "provenance": "literature"},
    ])
    prior = compile_prior(record)
    values = np.asarray(prior.pi(_grid(41)))
    assert values.max() == pytest.approx(1.0, abs=1e-9)
    assert values.min() >= 0.0


def _shipped_documents() -> list[Path]:
    root = PROJECT_ROOT / "experiments"
    return sorted(root.glob("priors/**/prior.json")) + sorted(root.glob("synthetic_priors/**/prior.json")) + sorted(
        root.glob("psl_elicitor_priors/**/prior.json"))


def test_shipped_constant_score_documents_stay_between_baseline_and_anchor_values():
    docs = _shipped_documents()
    assert docs
    n_constant = 0
    for path in docs:
        prior = compile_prior(load_record(path))
        if not np.isclose(prior._normalization_min, prior._normalization_max, rtol=1e-12, atol=1e-12):
            continue
        n_constant += 1
        level = prior._baseline_level()
        values = [level] + [float(pt["value"]) for b in prior._beliefs if b["token"] == "anchors" for pt in b["points"]]
        unit = stats.qmc.Halton(d=len(prior.dimensions), scramble=False).random(256)
        pi = prior._pi_unit(unit)
        assert pi.min() >= min(values) - 1e-9 and pi.max() <= max(values) + 1e-9, path
    assert n_constant > 0
