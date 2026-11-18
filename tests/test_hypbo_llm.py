"""hypbo-llm: BORA supplement C.3 box hypotheses, elicitation parsing, and the HypBO bridge."""

from __future__ import annotations

import json
import math
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import check_hypbo_hypotheses as gate  # noqa: E402
import elicit_hypbo_llm as elicit  # noqa: E402
from src.baselines import get_baseline  # noqa: E402
from src.baselines.base import config_key  # noqa: E402
from src.baselines.runner import run_ask_tell_loop  # noqa: E402
from src.baselines.workers.box_hypotheses import (  # noqa: E402
    EPSILON,
    box_inequalities,
    hypothesis_box,
    select_hypotheses,
)
from src.experiments.worker import _llm_hypothesis_boxes  # noqa: E402

LOG_VALUES = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
LIN_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DIMS = [
    {"name": "alpha", "type": "ordinal", "values": LOG_VALUES, "scale": "log"},
    {"name": "depth", "type": "ordinal", "values": LIN_VALUES, "scale": "linear"},
]


# --------------------------------------------------------------------------- box construction

def test_box_side_is_bounds_span_times_one_minus_score_plus_epsilon_centred_at_c():
    assert EPSILON == 1e-6  # BORA supplement C.2 floor
    lo, hi = hypothesis_box([0.0, 5.0], 0.75, [-4.0, 0.0], [4.0, 20.0])
    np.testing.assert_allclose(hi - lo, [8.0 * (0.25 + EPSILON), 20.0 * (0.25 + EPSILON)], rtol=1e-12)
    np.testing.assert_allclose((lo + hi) / 2, [0.0, 5.0], atol=1e-12)


def test_box_is_clipped_to_the_bounds():
    half = 0.5 * (0.4 + EPSILON)
    lo, hi = hypothesis_box([0.1, 0.95], 0.6, [0.0, 0.0], [1.0, 1.0])
    np.testing.assert_allclose(lo, [0.0, 0.95 - half], rtol=1e-12)
    np.testing.assert_allclose(hi, [0.1 + half, 1.0], rtol=1e-12)


def test_box_extremes_of_the_score():
    lo, hi = hypothesis_box([0.3, 0.7], 0.0, [0.0, 0.0], [1.0, 1.0])
    np.testing.assert_allclose(lo, [0.0, 0.2 - EPSILON / 2], rtol=1e-12)  # s = 0: full span, clipped
    np.testing.assert_allclose(hi, [0.8 + EPSILON / 2, 1.0], rtol=1e-12)
    lo, hi = hypothesis_box([0.3, 0.7], 1.0, [0.0, 0.0], [1.0, 1.0])
    np.testing.assert_allclose(hi - lo, [EPSILON, EPSILON], rtol=1e-9)  # s = 1: side 1e-6, not a point
    lo, hi = hypothesis_box([1.0, 0.0], 1.0, [0.0, 0.0], [1.0, 1.0])  # on a corner: half of it
    np.testing.assert_allclose(hi - lo, [EPSILON / 2, EPSILON / 2], rtol=1e-9)
    assert np.all(hi > lo)


@pytest.mark.parametrize("score", [-0.1, 1.5])
def test_box_rejects_scores_outside_unit_interval(score):
    with pytest.raises(ValueError):
        hypothesis_box([0.5], score, [0.0], [1.0])


def test_box_rejects_centre_outside_bounds():
    with pytest.raises(ValueError):
        hypothesis_box([1.2], 0.5, [0.0], [1.0])


def test_box_inequalities_follow_upstream_box_form():
    coeff, const = box_inequalities([0.1, 0.2], [0.4, 0.9])
    np.testing.assert_array_equal(coeff, [[1, 0], [-1, 0], [0, 1], [0, -1]])
    np.testing.assert_allclose(const, [0.4, -0.1, 0.9, -0.2])
    inside, outside = np.array([0.3, 0.5]), np.array([0.3, 0.95])
    assert np.all(coeff.dot(inside) <= const)
    assert not np.all(coeff.dot(outside) <= const)


def test_derive_boxes_maps_log_and_linear_dims_and_round_trips_native_bounds():
    hypotheses = [
        {"c_raw": None, "c": {"alpha": 1e-2, "depth": 5}, "s": 0.5},
        {"c_raw": None, "c": {"alpha": 1e-4, "depth": 9}, "s": 0.8},
    ]
    boxes = elicit.derive_boxes(hypotheses, DIMS)
    assert len(boxes) == 2
    first, second = boxes
    np.testing.assert_allclose(first["center_encoded"], [0.5, 0.5])
    np.testing.assert_allclose(first["box_encoded"]["lb"], [0.25, 0.25], atol=1e-6)
    np.testing.assert_allclose(first["box_encoded"]["ub"], [0.75, 0.75], atol=1e-6)
    # log dimension: the encoded box [0.25, 0.75] spans 1e-3 .. 1e-1 in native units
    np.testing.assert_allclose(first["box_native"]["alpha"], [1e-3, 1e-1], rtol=1e-5)
    np.testing.assert_allclose(first["box_native"]["depth"], [3.0, 7.0], rtol=1e-5)
    # second hypothesis sits on a corner: clipped to the bounds
    np.testing.assert_allclose(second["box_encoded"]["lb"], [0.0, 0.9], atol=1e-6)
    np.testing.assert_allclose(second["box_encoded"]["ub"], [0.1, 1.0], atol=1e-6)
    assert second["box_native"]["alpha"][0] == pytest.approx(1e-4)
    assert math.isclose(second["side_encoded"], 0.2 + EPSILON)
    assert first["kept"] and second["kept"]


def _box(center, score):
    return hypothesis_box(center, score, [0.0, 0.0], [1.0, 1.0])


def test_a_same_centre_box_is_dropped_only_when_both_boxes_are_narrow():
    specs = [([0.5, 0.5], 1.0), ([0.2, 0.3], 0.6), ([0.5, 0.5], 1.0), ([0.5, 0.5], 0.99999),
             ([0.5, 0.5], 0.3), ([0.5, 0.5], 0.6), ([0.2, 0.3], 0.6), ([0.2, 0.3], 1.0), ([0.2, 0.3], 1.0)]
    boxes = [_box(c, s) for c, s in specs]
    kept, dropped = select_hypotheses([c for c, _ in specs], [lo for lo, _ in boxes], [hi for _, hi in boxes])
    # narrow pairs at a shared centre: H3 and H4 duplicate H1, H9 duplicates H8; wide boxes stay
    assert kept == [0, 1, 4, 5, 6, 7]
    assert dropped == {2: 0, 3: 0, 8: 7}


def test_a_narrow_box_after_a_wide_one_at_the_same_centre_is_kept():
    specs = [([0.5, 0.5], 0.3), ([0.5, 0.5], 1.0), ([0.5, 0.5], 1.0)]
    boxes = [_box(c, s) for c, s in specs]
    kept, dropped = select_hypotheses([c for c, _ in specs], [lo for lo, _ in boxes], [hi for _, hi in boxes])
    assert kept == [0, 1] and dropped == {2: 1}
    # the 1e-4 threshold is on the largest side of the clipped box
    lo, hi = _box([1.0, 1.0], 1.0 - 1.5e-4)  # side 1.5e-4 halved by the corner
    kept, dropped = select_hypotheses([[1.0, 1.0]] * 2, [lo, lo], [hi, hi])
    assert dropped == {1: 0}


def test_derive_boxes_marks_same_centre_hypotheses_as_dropped():
    hypotheses = [
        {"c_raw": None, "c": {"alpha": 1e-2, "depth": 5}, "s": 1.0},
        {"c_raw": None, "c": {"alpha": 1e-2, "depth": 5}, "s": 1.0},  # narrow exact duplicate
        {"c_raw": None, "c": {"alpha": 1e-2, "depth": 5}, "s": 0.3},  # same centre, wide box
        {"c_raw": None, "c": {"alpha": 1e-1, "depth": 5}, "s": 1.0},
    ]
    boxes = elicit.derive_boxes(hypotheses, DIMS)
    assert [b["kept"] for b in boxes] == [True, False, True, True]
    assert [b.get("same_center_as") for b in boxes] == [None, "H1", None, None]


# --------------------------------------------------------------------------- prompt and parsing

_CARD = textwrap.dedent(
    """\
    # Prompt B

    ## 0. Prompt
    Write a PSL prior.

    ## 1. Domain
    A classifier.

    ## 2. Search space
    Parameter 1
    name: alpha
    type: ordinal
    values: [0.0001, 0.001, 0.01, 0.1, 1.0]  # exact allowed values
    meaning: regularization; values are log-spaced

    Parameter 2
    name: depth
    type: ordinal
    values: [1, 2, 3, 4, 5, 6, 7, 8, 9]  # exact allowed values
    meaning: tree depth

    ## 3. Optimization description
    Objective: accuracy.

    ## 4. Questions to consider before writing the prior
    Think, then write the prior.
    """
)


def test_prompt_keeps_the_card_from_domain_on_and_asks_for_hypotheses():
    prompt = elicit.build_prompt(_CARD, Path("card.md"))
    assert prompt.startswith("# Prompt B\n\n## 0. Prompt\n")
    assert "Write a PSL prior." not in prompt
    assert "## 1. Domain\nA classifier." in prompt
    assert '{"hypotheses": [{"c": {"alpha": <allowed value>, "depth": <allowed value>}, ' in prompt
    assert "The number of hypotheses is up to you." in prompt
    assert prompt.rstrip().endswith("then give your answer.")
    misleading = elicit.build_prompt(_CARD, Path("card.md"), hint="Use alpha=1.")
    assert misleading.rstrip().endswith("## Domain guidance\nUse alpha=1.")


def test_parser_takes_the_last_hypotheses_object_and_snaps_values():
    reply = (
        'Reasoning: {not json}. Low alpha {"c": 1} is good.\n'
        '{"hypotheses": [{"c": {"alpha": 0.0012, "depth": 4.4}, "s": 0.6}, '
        '{"c": {"ALPHA": 0.1, "Depth": 9}, "s": 0.3}]}'
    )
    obj = elicit.extract_answer(reply)
    parsed, error = elicit.validate_hypotheses(obj, DIMS)
    assert error == ""
    assert [h["c"] for h in parsed] == [{"alpha": 1e-3, "depth": 4}, {"alpha": 0.1, "depth": 9}]
    assert [h["s"] for h in parsed] == [0.6, 0.3]


@pytest.mark.parametrize(
    "answer, message",
    [
        ({"hypotheses": []}, "non-empty list"),
        ({"hypotheses": "x"}, "non-empty list"),
        ({"hypotheses": [3]}, "hypothesis 1 must be a JSON object"),
        ({"hypotheses": [{"c": {"alpha": 0.01, "depth": 2}, "s": 1.3}]}, "hypothesis 1:"),
        ({"hypotheses": [{"c": {"alpha": 0.01, "depth": 2}, "s": 0.5}, {"c": {"alpha": 0.01}, "s": 0.5}]},
         "hypothesis 2: missing parameter 'depth'"),
    ],
)
def test_invalid_hypotheses_are_rejected_with_a_reason(answer, message):
    parsed, error = elicit.validate_hypotheses(answer, DIMS)
    assert parsed is None
    assert message in error


def test_reply_without_hypotheses_key_is_not_parsed():
    assert elicit.extract_answer('{"c": {"alpha": 0.01}, "s": 0.5}') is None


def test_elicit_retries_with_the_validation_error(monkeypatch):
    replies = iter([
        ('{"hypotheses": [{"c": {"alpha": 0.01}, "s": 0.5}]}', {"prompt_tokens": 10, "completion_tokens": 5}, "stop"),
        ('{"hypotheses": [{"c": {"alpha": 0.01, "depth": 3}, "s": 0.5}]}',
         {"prompt_tokens": 12, "completion_tokens": 6}, "stop"),
    ])
    prompts = []

    def fake_call(prompt):
        prompts.append(prompt)
        return next(replies)

    monkeypatch.setattr(elicit, "call_llm", fake_call)
    result = elicit.elicit("PROMPT", DIMS, max_attempts=4)
    assert result["error"] == ""
    assert len(result["attempts"]) == 2
    assert result["attempts"][0]["error"].startswith("hypothesis 1: missing parameter")
    assert prompts[1].startswith("PROMPT\n\nYour previous answer was invalid: hypothesis 1: missing parameter")
    assert result["hypotheses"][0]["c"] == {"alpha": 0.01, "depth": 3}


# --------------------------------------------------------------------------- harness loader

class _GridProblem:
    """A 5 x 9 ordinal grid with HPOBench-style case differences in the parameter names."""

    parameter_names = ("Alpha", "depth")

    def __init__(self):
        self.native_candidates = tuple({"Alpha": a, "depth": d} for a in LOG_VALUES for d in LIN_VALUES)
        log_alpha = np.log([c["Alpha"] for c in self.native_candidates])
        depth = np.array([c["depth"] for c in self.native_candidates], dtype=float)
        self.encoded = np.column_stack([
            (log_alpha - log_alpha.min()) / np.ptp(log_alpha), (depth - 1.0) / 8.0])


def _artifact(tmp_path, hypotheses, **extra):
    path = tmp_path / "llm-card.json"
    path.write_text(json.dumps({"artifact_format": "hypbo-llm-hypotheses-v1",
                                "hypotheses": hypotheses, **extra}), encoding="utf-8")
    return path


def test_loader_builds_one_encoded_box_per_hypothesis(tmp_path):
    problem = _GridProblem()
    hypotheses = elicit.derive_boxes(
        [{"c_raw": None, "c": {"alpha": 1e-2, "depth": 5}, "s": 0.5},
         {"c_raw": None, "c": {"alpha": 1.0, "depth": 1}, "s": 0.9}], DIMS)
    prior = _llm_hypothesis_boxes(problem, _artifact(tmp_path, hypotheses),
                                  problem.native_candidates, problem.encoded)
    boxes = prior["hypotheses"]
    assert [b["name"] for b in boxes] == ["H1", "H2"]
    np.testing.assert_allclose(boxes[0]["lb"], [0.25, 0.25], atol=1e-6)
    np.testing.assert_allclose(boxes[0]["ub"], [0.75, 0.75], atol=1e-6)
    np.testing.assert_allclose(boxes[1]["lb"], [0.95, 0.0], atol=1e-6)
    np.testing.assert_allclose(boxes[1]["ub"], [1.0, 0.05], atol=1e-6)
    assert [b["score"] for b in boxes] == [0.5, 0.9]
    assert prior["dropped"] == []


def test_loader_drops_only_narrow_same_centre_boxes_and_lists_them(tmp_path):
    problem = _GridProblem()
    hypotheses = elicit.derive_boxes(
        [{"c_raw": None, "c": {"alpha": 1e-2, "depth": 5}, "s": 1.0},
         {"c_raw": None, "c": {"alpha": 1e-2, "depth": 5}, "s": 1.0},
         {"c_raw": None, "c": {"alpha": 1e-1, "depth": 2}, "s": 0.5},
         {"c_raw": None, "c": {"alpha": 1e-2, "depth": 5}, "s": 0.3}], DIMS)
    prior = _llm_hypothesis_boxes(problem, _artifact(tmp_path, hypotheses),
                                  problem.native_candidates, problem.encoded)
    assert [b["name"] for b in prior["hypotheses"]] == ["H1", "H3", "H4"]
    assert prior["dropped"] == [{"name": "H2", "score": 1.0, "same_center_as": "H1"}]
    s1 = prior["hypotheses"][0]
    np.testing.assert_allclose(np.subtract(s1["ub"], s1["lb"]), [EPSILON, EPSILON], rtol=1e-6)


def test_loader_rejects_a_recorded_box_that_disagrees_with_the_encoding(tmp_path):
    problem = _GridProblem()
    hypotheses = elicit.derive_boxes([{"c_raw": None, "c": {"alpha": 1e-2, "depth": 5}, "s": 0.5}], DIMS)
    hypotheses[0]["box_encoded"]["lb"][0] = 0.3
    with pytest.raises(ValueError, match="disagrees"):
        _llm_hypothesis_boxes(problem, _artifact(tmp_path, hypotheses), problem.native_candidates,
                              problem.encoded)


def test_loader_rejects_off_grid_values_and_other_formats(tmp_path):
    problem = _GridProblem()
    off_grid = [{"c": {"alpha": 0.02, "depth": 5}, "s": 0.5}]
    with pytest.raises(ValueError, match="not a grid value"):
        _llm_hypothesis_boxes(problem, _artifact(tmp_path, off_grid), problem.native_candidates,
                              problem.encoded)
    wrong = tmp_path / "other.json"
    wrong.write_text(json.dumps({"artifact_format": "colabo-llm-point-prior-v1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="hypbo-llm-hypotheses-v1"):
        _llm_hypothesis_boxes(problem, wrong, problem.native_candidates, problem.encoded)


# --------------------------------------------------------------------------- adapter and bridge

_STUB_HYPOTHESIS = '''
import numpy as np


class Hypothesis:
    def __init__(self, name, lb, ub, feature_names=[], coeff_inequalities=None,
                 const_inequalities=None, coeff_equalities=None, const_equalities=None):
        self.name, self.lb, self.ub, self.feature_names = name, np.asarray(lb), np.asarray(ub), feature_names
        self.coeff_inequalities, self.const_inequalities = coeff_inequalities, const_inequalities
        self.coeff_equalities, self.const_equalities = coeff_equalities, const_equalities
        self.sol_bounds = (self.lb, self.ub)
'''

_STUB_HYPBO = '''
import json
import os

import numpy as np


class HypBO:
    """Stand-in with upstream's evaluation order: one initial sample per hypothesis, then
    max(1, n_init - H) global initial samples, then `budget` iterations."""

    def __init__(self, func, feature_names, model, model_kwargs, hypotheses=[], seed=0,
                 n_processes=4, discretization=True, global_failure_limit=5,
                 local_failure_limit=2, gamma=0.0):
        self.func, self.hypotheses, self.model_kwargs = func, hypotheses, model_kwargs
        self.record = {
            "feature_names": feature_names, "model": model.__name__,
            "model_kwargs": {k: (v if k != "pbounds" else {n: list(b) for n, b in v.items()})
                             for k, v in model_kwargs.items()},
            "seed": seed, "n_processes": n_processes, "discretization": discretization,
            "global_failure_limit": global_failure_limit, "local_failure_limit": local_failure_limit,
            "gamma": gamma,
            "hypotheses": [{"name": h.name, "lb": h.lb.tolist(), "ub": h.ub.tolist(),
                            "coeff": np.asarray(h.coeff_inequalities).tolist(),
                            "const": np.asarray(h.const_inequalities).tolist()} for h in hypotheses],
        }

    def search(self, budget, n_init=5, batch=1):
        self.record.update(budget=budget, n_init=n_init, batch=batch)
        with open(os.environ["HYPBO_STUB_LOG"], "w") as handle:
            json.dump(self.record, handle)
        stop = int(os.environ.get("HYPBO_STUB_STOP", "0"))
        points = [(h.lb + h.ub) / 2 for h in self.hypotheses]
        points += [np.zeros(len(self.record["feature_names"]))] * max(1, n_init - len(self.hypotheses))
        points += [np.full(len(self.record["feature_names"]), 0.5 if i % 2 else 1.0) for i in range(budget)]
        for count, point in enumerate(points, start=1):
            self.func(point)
            if stop and count >= stop:
                return
'''


@pytest.fixture()
def stub_hypbo(tmp_path, monkeypatch):
    root = tmp_path / "hypbo"
    (root / "DBO" / "bayes_opt").mkdir(parents=True)
    (root / "hypbo.py").write_text(_STUB_HYPBO, encoding="utf-8")
    (root / "hypothesis.py").write_text(_STUB_HYPOTHESIS, encoding="utf-8")
    (root / "DBO" / "__init__.py").write_text("", encoding="utf-8")
    (root / "DBO" / "bayes_opt" / "__init__.py").write_text("", encoding="utf-8")
    (root / "DBO" / "bayes_opt" / "bayesian_optimization.py").write_text(
        "class BayesianOptimization:\n    pass\n\n\nclass DiscreteBayesianOptimization:\n    pass\n",
        encoding="utf-8")
    log = tmp_path / "stub_log.json"
    monkeypatch.setenv("HYPBO_PYTHON", sys.executable)
    monkeypatch.setenv("HYPBO_ROOT", str(root))
    monkeypatch.setenv("HYPBO_STUB_LOG", str(log))
    return log


def _grid_space():
    grid = [(a, b) for a in np.linspace(0.0, 1.0, 5) for b in np.linspace(0.0, 1.0, 5)]
    return {"candidates": [{"a": float(a), "b": float(b)} for a, b in grid],
            "encoded_candidates": np.asarray(grid, dtype=float)}


_PRIOR = {"hypotheses": [
    {"name": "H1", "lb": [0.0, 0.0], "ub": [0.5, 0.5]},
    {"name": "H2", "lb": [0.5, 0.6], "ub": [1.0, 0.9]},
]}


def _hypbo(budget, prior=_PRIOR):
    return get_baseline("hypbo", search_space=_grid_space(), objective_meta={"maximize": True},
                        prior=prior, seed=1003, budget=budget, n_initial=1)


def test_adapter_requires_boxes_inside_the_encoded_bounds():
    with pytest.raises(ValueError, match="hypotheses"):
        _hypbo(4, prior=None).worker_config()
    with pytest.raises(ValueError, match="inside the encoded bounds"):
        _hypbo(4, prior={"hypotheses": [{"lb": [0.0, 0.0], "ub": [1.2, 1.0]}]}).worker_config()
    config = _hypbo(4).worker_config()
    assert config["hypotheses"][1] == {"name": "H2", "lb": [0.5, 0.6], "ub": [1.0, 0.9]}
    assert {k: config[k] for k in ("global_failure_limit", "local_failure_limit", "gamma", "batch",
                                   "n_processes")} == {"global_failure_limit": 5, "local_failure_limit": 2,
                                                       "gamma": 0.0, "batch": 1, "n_processes": 1}


def test_every_hypbo_evaluation_is_charged_and_repeats_are_served_from_cache(stub_hypbo):
    calls = []

    def objective(config):
        calls.append(config_key(config))
        return config["a"] + 2.0 * config["b"]

    optimizer = _hypbo(budget=8)
    try:
        history = run_ask_tell_loop(optimizer, objective, budget=8, n_init=1)
    finally:
        optimizer.close()
    accounting = optimizer.bridge_accounting()
    # 2 hypothesis initial samples + 1 global initial sample + 5 iterations = the budget
    assert len(history) == 8 and accounting["queries_used"] == 8
    keys = [config_key(config) for config, _ in history]
    assert accounting["n_repeats"] == len(keys) - len(set(keys)) > 0
    assert len(calls) == len(set(keys)) == accounting["n_unique"]  # one objective call per unique point
    assert accounting["proposals_exhausted_at"] is None
    # the first two queries are the box centres snapped to the grid, the third the global sample
    assert [config for config, _ in history[:3]] == [{"a": 0.25, "b": 0.25}, {"a": 0.75, "b": 0.75},
                                                     {"a": 0.0, "b": 0.0}]
    record = json.loads(stub_hypbo.read_text())
    assert record["model"] == "BayesianOptimization"
    assert record["discretization"] is False and record["n_processes"] == 1
    assert record["model_kwargs"] == {"pbounds": {"x0": [0.0, 1.0], "x1": [0.0, 1.0]}, "random_seed": 1003}
    assert (record["budget"], record["n_init"], record["batch"]) == (8, 1, 1)
    assert record["feature_names"] == ["x0", "x1"]
    assert record["hypotheses"][1] == {"name": "H2", "lb": [0.5, 0.6], "ub": [1.0, 0.9],
                                       "coeff": [[1, 0], [-1, 0], [0, 1], [0, -1]],
                                       "const": [1.0, -0.5, 0.9, -0.6]}


def test_a_short_hypbo_run_is_not_filled(stub_hypbo, monkeypatch):
    monkeypatch.setenv("HYPBO_STUB_STOP", "3")
    calls = []
    optimizer = _hypbo(budget=8)
    try:
        history = run_ask_tell_loop(optimizer, lambda c: calls.append(c) or c["a"], budget=8, n_init=1)
    finally:
        optimizer.close()
    accounting = optimizer.bridge_accounting()
    assert len(history) == 3 and len(calls) == 3
    assert accounting["queries_used"] == 3 and accounting["budget"] == 8
    assert accounting["proposals_exhausted_at"] == 4


# --------------------------------------------------------------------------- prompt parity with colabo-llm

def test_every_prompt_matches_the_committed_colabo_llm_prompt_from_the_domain_section_on():
    import re

    jobs = elicit.hpo_jobs(False) + elicit.hpo_jobs(True)
    assert len(jobs) == 80
    for job in jobs:
        card = job["card_path"].read_text(encoding="utf-8")
        hint = (job["hint_path"].read_text(encoding="utf-8").strip()
                if job["modality"] == elicit.ADVERSARIAL_MODALITY else None)
        ours = elicit.build_prompt(card, job["card_path"], hint)
        reference = (PROJECT_ROOT / "experiments" / "colabo_llm_priors" / job["benchmark_id"] / job["task"]
                     / f'{job["modality"]}.prompt.md').read_text(encoding="utf-8")
        tail = re.compile(r"(?m)^(?:## )?1\. Domain\s*$")
        assert ours[tail.search(ours).start():] == reference[tail.search(reference).start():], job["task"]


# --------------------------------------------------------------------------- post-elicitation gate


def _gate(entries_json: str, budget: int = 16, tamper=None):
    raw = "Some reasoning.\n" + '{"hypotheses": [' + entries_json + "]}"
    parsed, error = elicit.validate_hypotheses(elicit.extract_answer(raw), DIMS)
    assert error == "", error
    artifact = {"task": "t", "modality": "llm-card", "dimensions": DIMS, "attempts": 1,
                "hypotheses": elicit.derive_boxes(parsed, DIMS)}
    if tamper:
        tamper(artifact)
    return gate.check_artifact(artifact, budget, raw)


def test_gate_passes_a_clean_artifact():
    row = _gate('{"c": {"alpha": 0.01, "depth": 5}, "s": 0.7}, {"c": {"alpha": 0.1, "depth": 2}, "s": 0.4}')
    assert row["stop"] == [] and not any(row["flags"].values())
    assert (row["h_raw"], row["h_kept"]) == (2, 2)


@pytest.mark.parametrize(
    "entries, flag",
    [
        ('{"c": {"alpha": "0.01", "depth": 5}, "s": 0.7}', "c-non-numeric"),
        ('{"c": {"alpha": NaN, "depth": 5}, "s": 0.7}', "c-non-finite"),
        ('{"c": {"alpha": 0.01, "depth": 1000}, "s": 0.7}', "c-out-of-range"),
        ('{"c": {"alpha": 0.01, "depth": 5}, "s": true}', "s-type"),
        ('{"c": {"alpha": 0.01, "depth": 5}, "s": "0.7"}', "s-type"),
    ],
)
def test_gate_stops_on_values_the_parser_converts(entries, flag):
    row = _gate(entries)
    assert flag in row["stop"]


@pytest.mark.parametrize(
    "entries, flag, count",
    [
        ('{"c": {"alpha": 0.01, "depth": 5}, "s": 1}', "s-one", 1),
        ('{"c": {"alpha": 0.0012, "depth": 5}, "s": 0.5}', "c-off-grid", 1),
        ('{"c": {"alpha": 0.01, "depth": 5, "beta": 3}, "s": 0.5}', "key-unknown", 1),
        ('{"c": {"alpha": 0.01, "ALPHA": 0.1, "depth": 5}, "s": 0.5}', "key-case-dup", 1),
        ('{"c": {"alpha": 0.01, "alpha": 0.1, "depth": 5}, "s": 0.5}', "key-repeated", 1),
        ('{"c": {"alpha": 0.01, "depth": 5}, "s": 0.5, "why": "x"}', "entry-extra", 1),
        ('{"c": {"alpha": 0.01, "depth": 5}, "s": 0.5}, {"c": {"alpha": 0.01, "depth": 5}, "s": 0.5}, '
         '{"c": {"alpha": 0.01, "depth": 5}, "s": 0.9}', "duplicate", 1),
        ('{"c": {"alpha": 0.01, "depth": 5}, "s": 0.5}, {"c": {"alpha": 0.01, "depth": 5}, "s": 0.5}, '
         '{"c": {"alpha": 0.01, "depth": 5}, "s": 0.9}', "same-centre", 2),
        ('{"c": {"alpha": 0.01, "depth": 5}, "s": 1}, {"c": {"alpha": 0.01, "depth": 5}, "s": 1}, '
         '{"c": {"alpha": 0.01, "depth": 5}, "s": 0.3}', "dropped", 1),
    ],
)
def test_gate_counts_without_stopping(entries, flag, count):
    row = _gate(entries)
    assert row["flags"].get(flag) == count
    assert row["stop"] == []


def test_gate_stops_when_the_initial_design_fills_the_budget():
    distinct = [f'{{"c": {{"alpha": {a}, "depth": {d}}}, "s": 0.5}}' for a in LOG_VALUES for d in (1, 2, 3)]
    assert "budget" not in _gate(", ".join(distinct[:14]))["stop"]  # 14 + 1 = 15 < 16
    row = _gate(", ".join(distinct[:15]))
    assert "budget" in row["stop"] and row["h_kept"] == 15


def test_gate_stops_when_the_stored_hypotheses_differ_from_the_raw_reply():
    def tamper(artifact):
        artifact["hypotheses"][0]["s"] = 0.9

    assert "mismatch" in _gate('{"c": {"alpha": 0.01, "depth": 5}, "s": 0.7}', tamper=tamper)["stop"]


# --------------------------------------------------------------------------- real upstream (opt-in)

_REAL_ROOT = Path(os.environ.get("HYPBO_ROOT", str(PROJECT_ROOT / "external" / "hypbo")))


@pytest.mark.skipif(not os.environ.get("HYPBO_PYTHON") or not (_REAL_ROOT / "hypbo.py").is_file(),
                    reason="set HYPBO_PYTHON (and HYPBO_ROOT or external/hypbo) to run against the real upstream")
def test_real_upstream_runs_the_budget_with_s1_boxes_and_duplicates(tmp_path):
    import threading

    values = list(range(10))
    dims = [{"name": name, "type": "ordinal", "values": values, "scale": "linear"} for name in ("a", "b")]

    class Grid:
        parameter_names = ("a", "b")
        native_candidates = tuple({"a": a, "b": b} for a in values for b in values)

    encoded = np.array([[c["a"] / 9.0, c["b"] / 9.0] for c in Grid.native_candidates])
    hypotheses = elicit.derive_boxes(
        [{"c_raw": None, "c": {"a": 2, "b": 3}, "s": 0.6},
         {"c_raw": None, "c": {"a": 5, "b": 5}, "s": 1.0},
         {"c_raw": None, "c": {"a": 5, "b": 5}, "s": 1.0},
         {"c_raw": None, "c": {"a": 5, "b": 5}, "s": 0.99999},
         {"c_raw": None, "c": {"a": 5, "b": 5}, "s": 0.3}], dims)
    prior = _llm_hypothesis_boxes(Grid, _artifact(tmp_path, hypotheses), Grid.native_candidates, encoded)
    # (c, 1.0) followed by (c, 0.3) at the same centre: both kept; the narrow duplicates are dropped
    assert [b["name"] for b in prior["hypotheses"]] == ["H1", "H2", "H5"]
    assert [d["name"] for d in prior["dropped"]] == ["H3", "H4"]
    optimizer = get_baseline(
        "hypbo", search_space={"candidates": [dict(c) for c in Grid.native_candidates],
                               "encoded_candidates": encoded},
        objective_meta={"maximize": True}, prior=prior, seed=1001, budget=8, n_initial=1,
        worker_log_path=str(tmp_path / "worker.log"))
    # a stalled worker is killed, which surfaces as a bridge error instead of a hung test
    watchdog = threading.Timer(900, lambda: optimizer._proc is not None and optimizer._proc.kill())
    watchdog.start()
    try:
        history = run_ask_tell_loop(optimizer, lambda c: -((c["a"] - 3) ** 2 + (c["b"] - 3) ** 2),
                                    budget=8, n_init=1)
    finally:
        watchdog.cancel()
        optimizer.close()
    accounting = optimizer.bridge_accounting()
    assert len(history) == 8
    assert accounting["queries_used"] == 8 and accounting["proposals_exhausted_at"] is None


def test_gate_requires_one_artifact_per_study_prior_path(tmp_path):
    expected = gate.expected_artifacts()
    assert len(expected) == 80
    assert sorted({path.split("/")[-1] for path in expected}) == ["llm-card-misleading.json", "llm-card.json"]
    raw = '{"hypotheses": [{"c": {"alpha": 0.01, "depth": 5}, "s": 0.7}]}'
    parsed, _ = elicit.validate_hypotheses(elicit.extract_answer(raw), DIMS)
    artifact = json.dumps({"task": "t", "modality": "llm-card", "dimensions": DIMS,
                           "attempts": 1, "hypotheses": elicit.derive_boxes(parsed, DIMS)})
    replies = tmp_path / "replies"
    rows, problems = gate.run_gate(tmp_path, 16, replies_root=replies)  # empty root
    assert rows == [] and len(problems) == 80
    for path in expected:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(artifact, encoding="utf-8")
        reply = (replies / path).with_suffix(".replies.json")
        reply.parent.mkdir(parents=True, exist_ok=True)
        reply.write_text(json.dumps({"attempt_log": [{"attempt": 1, "raw_reply": raw}]}), encoding="utf-8")
    rows, problems = gate.run_gate(tmp_path, 16, replies_root=replies)
    assert len(rows) == 80 and problems == [] and not any(row["stop"] for row in rows)
    (replies / expected[0]).with_suffix(".replies.json").unlink()  # raw reply not kept
    rows, _ = gate.run_gate(tmp_path, 16, replies_root=replies)
    assert [row["stop"] for row in rows if row["stop"]] == [["mismatch"]]
    (tmp_path / expected[0]).unlink()  # partial elicitation
    assert gate.run_gate(tmp_path, 16, replies_root=replies)[1] == [f"missing {expected[0]}"]
    (tmp_path / expected[0]).write_text(artifact, encoding="utf-8")
    extra = tmp_path / "hpobench-tabular" / "lr-car" / "llm-card-extra.json"
    extra.write_text(artifact, encoding="utf-8")
    assert gate.run_gate(tmp_path, 16, replies_root=replies)[1] == [
        "unexpected hpobench-tabular/lr-car/llm-card-extra.json"]
