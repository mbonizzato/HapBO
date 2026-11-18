"""LLM-EI arm: survey design, prompt rendering, answer parsing, compact answers, utility
artifact, the percentile gate in the finite-candidate BO loop, and the worker dispatch.

The worker tests run the real ``_run_offline`` on a synthetic HPO grid (same code path as
the HPOBench tables). The preference-GP fit needs BoTorch (skipped without it).
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from bayesian_optimization.bo import BayesianOptimization  # noqa: E402
from experiments import llm_ei_survey as S  # noqa: E402
from experiments import worker  # noqa: E402
from experiments.catalog import load_catalog  # noqa: E402
from experiments.llm_ei import (  # noqa: E402
    ARTIFACT_FORMAT, PAPER_SCHEDULE, grid_fingerprint, load_llm_ei_context, parse_schedule, percentile_for,
)
from experiments.llm_ei_utility import comparisons, load_answers, write_artifact  # noqa: E402

CATALOG = load_catalog()
PRIORS = PROJECT_ROOT / "experiments" / "priors"
SVM_CARD = PRIORS / "hpobench-svm-vehicle" / "vehicle"
HPO_PRIOR = PRIORS / "hpobench-lr-australian" / "australian" / "base" / "prior.json"


# --------------------------------------------------------------------------- survey design
def test_survey_has_no_self_or_repeated_pairs_and_is_nested() -> None:
    big = S.build_survey(441, 6, seed=3)
    keys = [(min(q["a"], q["b"]), max(q["a"], q["b"])) for q in big["questions"]]
    assert all(a != b for a, b in keys)
    assert len(set(keys)) == len(keys) == big["n_questions"]
    small = S.build_survey(441, 2, seed=3)
    assert small["questions"] == big["questions"][:small["n_questions"]]
    assert S.build_survey(441, 2, seed=4)["questions"] != small["questions"]


def test_survey_rounds_are_balanced_and_subsample_large_grids() -> None:
    survey = S.build_survey(30000, 4, m_max=800, seed=1)
    assert survey["n_items"] == 800 and len(set(survey["items"])) == 800
    items = set(survey["items"])
    first_round = [q for q in survey["questions"] if q["round"] == 1]
    assert all(q["a"] in items and q["b"] in items for q in survey["questions"])
    # one round: each surveyed item at most once as A and once as B
    assert len({q["a"] for q in first_round}) == len(first_round)
    assert len({q["b"] for q in first_round}) == len(first_round)
    # every candidate appears once as A and once as B per round, before self/repeat removal
    kept = survey["n_questions"] + survey["removed"]["self"] + survey["removed"]["repeat"]
    assert kept == 4 * 800


def test_survey_reproduces_the_papers_question_counts() -> None:
    # Paper: 792 experiments x L=10 -> 7,788-7,842 questions; 1,728 x L=5 -> 8,610.
    for n, rounds, low, high in ((792, 10, 7700, 7900), (1728, 5, 8560, 8640)):
        assert low <= S.build_survey(n, rounds, m_max=10**6, seed=0)["n_questions"] <= high


# --------------------------------------------------------------------------- prompts
def test_card_context_keeps_task_information_only() -> None:
    std = (SVM_CARD / "llm-card" / "prompt_card.md").read_text(encoding="utf-8")
    adv = (SVM_CARD / "llm-card-misleading" / "prompt_card.md").read_text(encoding="utf-8")
    ctx = S.card_context(std)
    assert ctx.startswith("## 1. Domain\n") and "## 3. Optimization description" in ctx
    assert "PSL" not in ctx and "## 4." not in ctx and "## 0." not in ctx and "write the prior" not in ctx
    for heading in ("1. Domain", "2. Search space", "3. Optimization description"):
        assert S.card_sections(std)[heading] in ctx  # verbatim
    adv_ctx = S.card_context(adv)
    assert adv_ctx.startswith(ctx) and adv_ctx.endswith(S.card_sections(adv)["Domain guidance"])
    with pytest.raises(ValueError):
        S.card_sections(std + "\n## 5. Extra\nx\n")


def test_rendered_question_uses_the_cards_value_tokens() -> None:
    text = (SVM_CARD / "llm-card" / "prompt_card.md").read_text(encoding="utf-8")
    prompter = S.TaskPrompter(text, ["C", "gamma"])
    system, user = prompter.render({"C": 1.0, "gamma": 0.0009765625}, {"C": 1024.0, "gamma": 4.0})
    assert system == S.SYSTEM_PROMPT
    assert "Setup A:\nc: 1\ngamma: 0.0009765625\n\nSetup B:\nc: 1024\ngamma: 4\n\n" in user
    assert user.endswith(S.OUTPUT_TEXT)
    with pytest.raises(ValueError):
        prompter.render({"C": 3.0, "gamma": 1.0}, {"C": 1.0, "gamma": 1.0})  # off-card value


# --------------------------------------------------------------------------- answers
@pytest.mark.parametrize("text, expected", [
    ('{"Setup": "A", "reasoning": "x"}', ("A", "json")),
    ('```json\n{"setup": "Setup B", "reasoning": "y"}\n```', ("B", "json")),
    ('{\n  "Setup": "B",\n  "reasoning": "Setup A uses', ("B", "regex")),  # truncated (reasoning-stop)
    ('{\n  "Setup": "A",\n  "reasoning": "Setup A uses a moderate learning rate and a small', ("A", "regex")),
    ('```' + 'json\n{\n  "Setup": "B",\n  "reasoning": "Setup B', ("B", "regex")),
])
def test_parse_answer_accepts(text, expected) -> None:
    assert S.parse_answer(text) == expected


@pytest.mark.parametrize("text", ['{"Setup": "C"}', '{"answer": "A"}', "I think A", '"Setup": "A" ... "Setup": "B"'])
def test_parse_answer_rejects(text) -> None:
    with pytest.raises(S.SurveyAnswerError):
        S.parse_answer(text)


class _FakeClient:
    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def complete(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.answers.pop(0)


def test_ask_question_feeds_back_the_validator_error() -> None:
    client = _FakeClient(['{"Setup": "maybe"}', '{"Setup": "B", "reasoning": "z"}'])
    result = S.ask_question(client, "sys", "user", seed=7)
    assert result["answer"] == "B" and result["attempts"] == 2
    assert "was REJECTED" in client.prompts[1] and client.prompts[1].startswith("user")
    bad = S.ask_question(_FakeClient(["no"] * 4), "sys", "user", seed=7)
    assert bad["answer"] is None and bad["attempts"] == S.MAX_ATTEMPTS


def test_comparisons_map_answers_to_winner_loser() -> None:
    questions = [{"q": 0, "a": 5, "b": 9}, {"q": 1, "a": 2, "b": 3}, {"q": 2, "a": 7, "b": 1}]
    pairs, counts = comparisons(questions, "B-A")
    assert pairs.tolist() == [[9, 5], [7, 1]]
    assert counts == {"asked": 3, "valid": 2, "invalid": 1}
    with pytest.raises(ValueError):
        comparisons(questions, "AB")


def _rows(survey, *, attempts=1, raw='{\n  "Setup": "A",\n  "reasoning": "'):
    return {q["q"]: {"q": q["q"], "answer": "A" if q["q"] % 3 else "B", "attempts": attempts, "parse": "regex",
                     "raw": raw} for q in survey["questions"]}


def test_compact_answers_round_trip(tmp_path) -> None:
    survey = S.build_survey(60, 3, seed=11)
    doc = S.compact_answers(model="lr", task="kc1", card="llm-card", n_grid=60, survey=survey, rows=_rows(survey),
                            llm={"model": "m"}, card_sha256="c", context_sha256="x")
    assert doc["mode"] == "reasoning-stop"
    assert len(doc["answers"]) == survey["n_questions"] and set(doc["answers"]) <= {"A", "B"}
    path = tmp_path / "survey_answers.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    loaded, rebuilt = load_answers(path, 60)
    assert rebuilt["questions"] == survey["questions"] and loaded["answers"] == doc["answers"]
    doc["survey"]["questions_sha256"] = "0" * 64
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_answers(path, 60)
    with pytest.raises(ValueError):
        load_answers(path, 61)


def test_check_survey_flags_retries_and_answer_order() -> None:
    survey = S.build_survey(10, 1, seed=1)
    rows = _rows(survey)
    n = survey["n_questions"]
    assert S.check_survey(rows, n)["ok"]
    assert not S.check_survey(rows, n + 1)["ok"]  # unanswered question
    rows[1] = {**rows[1], "attempts": 2}
    report = S.check_survey(rows, n)
    assert report["retried"] == 1 and not report["ok"]
    rows[1] = {**rows[1], "attempts": 1, "raw": '{\n  "reasoning": "x", "Setup": "B"'}
    report = S.check_survey(rows, n)
    assert report["setup_first"] == n - 1 and not report["ok"]


def test_shipped_answers_rebuild_their_survey() -> None:
    files = sorted(PRIORS.glob("hpobench-*/*/llm-ei-card*/survey_answers.json"))
    assert len(files) == 80
    for path in files[:: 13]:  # a spread of families and both cards
        doc = json.loads(path.read_text(encoding="utf-8"))
        loaded, survey = load_answers(path, int(doc["task"]["n_candidates"]))
        assert loaded["check"]["invalid"] == 0 and loaded["check"]["retried"] == 0
        assert (path.parent / "llm_ei_utility.json.gz").is_file()


# --------------------------------------------------------------------------- gate
def test_paper_schedule() -> None:
    assert parse_schedule(None) == PAPER_SCHEDULE
    assert [percentile_for(n, PAPER_SCHEDULE) for n in (0, 15, 30, 31, 40, 41)] == [85, 85, 85, 15, 15, 0]
    with pytest.raises(ValueError):
        parse_schedule([[40, 85], [30, 15]])


def _grid(n=12):
    xs = np.linspace(0.0, 1.0, n)
    X = np.array([[a, b] for a in xs for b in xs])
    truth = -((X[:, 0] - 0.7) ** 2 + (X[:, 1] - 0.3) ** 2)
    return X, truth


def _run(X, truth, gate, seed=1001, budget=10):
    opt = BayesianOptimization(X, length_scale=0.2, noise_variance=1e-6, normalize_y=True,
                               random_state=seed, n_initial=1, candidate_gate=gate)
    while opt.n_observations < budget:
        index, _ = opt.ask()
        opt.tell(index, float(truth[index]))
    return opt


def test_gate_restricts_first_query_and_acquisition() -> None:
    X, truth = _grid()
    g = X[:, 0] + 0.1 * X[:, 1]
    allowed = g >= np.percentile(g, 85)
    opt = _run(X, truth, lambda n: allowed)
    assert all(allowed[i] for i in opt.evaluated_indices) and not opt.gate_fallbacks_


def test_gate_none_or_all_true_reproduces_vanilla() -> None:
    X, truth = _grid()
    vanilla = _run(X, truth, None).evaluated_indices
    assert _run(X, truth, lambda n: None).evaluated_indices == vanilla
    assert _run(X, truth, lambda n: np.ones(len(X), bool)).evaluated_indices == vanilla


def test_gate_falls_back_when_exhausted() -> None:
    X, truth = _grid(4)
    allowed = np.zeros(len(X), bool)
    allowed[:3] = True
    opt = _run(X, truth, lambda n: allowed, budget=6)
    assert set(opt.evaluated_indices[:3]) == {0, 1, 2} and opt.gate_fallbacks_ == [3, 4, 5]


# --------------------------------------------------------------------------- worker
def _synthetic_task(tmp_path: Path):
    alpha = [float(v) for v in np.logspace(-5, 0, 8)]
    eta0 = [float(v) for v in np.logspace(-4, 0, 7)]
    native = tuple({"alpha": a, "eta0": e} for e in eta0 for a in alpha)
    la, le = np.log10([c["alpha"] for c in native]), np.log10([c["eta0"] for c in native])
    truth = 0.9 - 0.02 * ((la + 1.5) ** 2 + (le + 2.0) ** 2)
    lookup = {(c["alpha"], c["eta0"]): float(t) for c, t in zip(native, truth)}
    problem = SimpleNamespace(
        native_candidates=native, parameter_names=("alpha", "eta0"), model="lr", task_name="synthetic",
        n_candidates=len(native), maximize=True, ground_truth=truth, optimum_value=float(truth.max()),
        objective=lambda c: lookup[(c["alpha"], c["eta0"])], dataset_id="hpobench", target_name="accuracy",
    )
    card = json.loads(HPO_PRIOR.read_text(encoding="utf-8"))
    card["schema"] = {"objective": "maximize", "dimensions": [
        {"name": "alpha", "type": "ordinal", "values": alpha},
        {"name": "eta0", "type": "ordinal", "values": eta0},
    ]}
    card["beliefs"] = [{"token": "baseline", "level": 0.05}]
    prior = tmp_path / "card" / "prior.json"
    prior.parent.mkdir(parents=True, exist_ok=True)
    prior.write_text(json.dumps(card), encoding="utf-8")
    return problem, prior


def _utility_artifact(tmp_path, problem, prior, g, **override) -> Path:
    doc = {"artifact_format": ARTIFACT_FORMAT, "task": {"model": problem.model, "task": problem.task_name},
           "card": "llm-card", "encoding_prior_path": str(prior),
           "grid_fingerprint": grid_fingerprint(problem.native_candidates, problem.parameter_names),
           "survey": {"mode": "reasoning-stop", "n_questions": 10}, "utility": [float(v) for v in g]}
    doc.update(override)
    path = tmp_path / "utility" / "llm_ei_utility.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_artifact(path, doc)  # the shipped form
    return path


def _worker_run(monkeypatch, tmp_path, problem, method_id, prior_path, name):
    monkeypatch.setattr(worker, "_load_offline_problem", lambda *a, **k: problem)
    run = {"run_id": name, "method_id": method_id, "seed": 1003, "budget": 8, "parameters": {},
           "optimizer": {"n_initial": 1, "length_scale": 0.2, "noise_std": 0.001, "improvement_xi": 0.0},
           "prior_path": str(prior_path)}
    directory = tmp_path / name
    directory.mkdir()
    return worker._run_offline(directory, run, {"adapter": "hpobench"}, CATALOG.methods[method_id])


def test_worker_llm_ei_is_vanilla_with_a_flat_utility(monkeypatch, tmp_path) -> None:
    problem, prior = _synthetic_task(tmp_path)
    vanilla = _worker_run(monkeypatch, tmp_path, problem, "vanilla-bo-ei", prior, "v")
    flat = _utility_artifact(tmp_path, problem, prior, np.zeros(problem.n_candidates))
    llm_ei = _worker_run(monkeypatch, tmp_path, problem, "llm-ei", flat, "l")
    assert llm_ei["metrics"]["evaluated_indices"] == vanilla["metrics"]["evaluated_indices"]
    assert llm_ei["gp_encoding"] == vanilla["gp_encoding"]
    assert [step["percentile"] for step in llm_ei["llm_ei"]["gate_log"]] == [85.0] * 8


def test_worker_llm_ei_queries_only_the_top_of_g(monkeypatch, tmp_path) -> None:
    problem, prior = _synthetic_task(tmp_path)
    g = np.asarray([c["alpha"] for c in problem.native_candidates])  # prefers large alpha
    path = _utility_artifact(tmp_path, problem, prior, g)
    out = _worker_run(monkeypatch, tmp_path, problem, "llm-ei", path, "l")
    allowed = g >= np.percentile(g, 85)
    assert all(allowed[i] for i in out["metrics"]["evaluated_indices"])
    assert out["llm_ei"]["gate_fallbacks"] == []


def test_utility_artifact_is_checked(monkeypatch, tmp_path) -> None:
    problem, prior = _synthetic_task(tmp_path)
    method = CATALOG.methods["llm-ei"]
    good = _utility_artifact(tmp_path, problem, prior, np.zeros(problem.n_candidates))
    context = load_llm_ei_context(problem, "hpobench", method, good)
    assert context.utility.size == problem.n_candidates
    plain = good.with_suffix("")  # llm_ei_utility.json: the uncompressed form reads the same
    plain.write_bytes(gzip.decompress(good.read_bytes()))
    assert load_llm_ei_context(problem, "hpobench", method, plain).artifact_sha256 == context.artifact_sha256
    assert good.read_bytes() == _utility_artifact(tmp_path, problem, prior, np.zeros(problem.n_candidates)).read_bytes()
    for override in ({"grid_fingerprint": "0" * 64}, {"task": {"model": "rf", "task": "synthetic"}},
                     {"utility": [0.0] * 3}, {"artifact_format": "other"}):
        bad = _utility_artifact(tmp_path, problem, prior, np.zeros(problem.n_candidates), **override)
        with pytest.raises(ValueError):
            load_llm_ei_context(problem, "hpobench", method, bad)
    with pytest.raises(FileNotFoundError):
        load_llm_ei_context(problem, "hpobench", method, tmp_path / "missing.json")
    with pytest.raises(ValueError):
        load_llm_ei_context(problem, "solar", method, good)


# --------------------------------------------------------------------------- preference GP
def test_preference_gp_recovers_an_oracle_ranking() -> None:
    pytest.importorskip("botorch")
    from experiments.llm_ei_utility import fit_utility

    X, truth = _grid(10)
    survey = S.build_survey(len(X), 4, seed=0)
    pairs = np.array([(q["a"], q["b"]) if truth[q["a"]] > truth[q["b"]] else (q["b"], q["a"])
                      for q in survey["questions"]])
    g, diag = fit_utility(X, pairs)
    from scipy.stats import spearmanr

    assert spearmanr(g, truth).statistic > 0.9 and diag["n_comparisons"] == len(pairs)


def test_fit_fallback_accepts_only_line_search_stops() -> None:
    pytest.importorskip("botorch")
    import warnings

    from botorch.exceptions.warnings import OptimizationWarning
    from experiments.llm_ei_utility import _accept_line_search_stop

    def message(category, text):
        return warnings.WarningMessage(text, category, "f.py", 1)

    abnormal = "`scipy_minimize` terminated with status OptimizationStatus.FAILURE, ... ABNORMAL: "
    assert _accept_line_search_stop(message(OptimizationWarning, abnormal))
    assert not _accept_line_search_stop(message(OptimizationWarning, "some other optimizer failure"))


def test_utility_refuses_a_reasoning_stop_survey_with_retries(monkeypatch, tmp_path) -> None:
    import datasets.hpobench as H
    from experiments import llm_ei_utility as U

    problem = SimpleNamespace(n_candidates=10, native_candidates=[{"a": float(i)} for i in range(10)],
                              parameter_names=("a",))
    monkeypatch.setattr(H, "load_hpobench_problem", lambda **k: problem)
    survey = S.build_survey(10, 1, seed=20260)
    doc = S.compact_answers(model="lr", task="kc1", card="llm-card", n_grid=10, survey=survey,
                            rows=_rows(survey), llm=None, card_sha256="c", context_sha256="x")
    doc["check"]["retried"] = 1
    path = S.answers_path("lr", "kc1", "llm-card", priors_root=tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="reasoning-stop"):
        U.build_utility("lr", "kc1", "llm-card", priors_root=tmp_path, out_root=tmp_path / "out")
    assert not (tmp_path / "out").exists()
