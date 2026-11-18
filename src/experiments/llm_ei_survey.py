"""Pairwise LLM survey of the LLM-EI baseline (Patel et al., Digital Discovery 2026).

LLM-EI distils an LLM's knowledge into a utility g(x): the LLM answers many A/B questions
("which of these two setups gives the higher objective?") and a preference GP fitted to the
answers gives g (``src/experiments/llm_ei_utility.py``). This module builds and asks the
survey on one HPOBench task.

Survey design (paper, Sec. 2.2.2): two arrays, each holding L copies of every surveyed
candidate, are paired at random; self-pairs and repeated questions are removed. The pairing
is built in rounds: a round pairs a random permutation of the surveyed candidates with an
independent random permutation (each candidate appears once as A and once as B), so L rounds
are the paper's design with L copies per array. Questions that repeat an earlier unordered
pair or pair a candidate with itself are dropped (this reproduces the paper's question
counts, e.g. 8,610 for 1,728 candidates and L = 5). Grids larger than ``m_max`` = 800
candidates (the paper's dataset size) are surveyed on a seeded random subset; g is then
predicted on the full grid. The design is a deterministic function of the grid size, L,
``m_max`` and the survey seed (:func:`build_survey`).

LLM-visible text (rendered for audit with ``--render``):

* system: the first sentence of the card's section 0, verbatim (:data:`SYSTEM_PROMPT`);
* user: the card's ``## 1. Domain``, ``## 2. Search space`` and ``## 3. Optimization
  description`` sections and, on the misleading card, its ``## Domain guidance`` section,
  verbatim; then the question (:data:`TASK_TEXT`, the two setups, :data:`OUTPUT_TEXT`),
  modelled on the prompt of Fig. S2 of the LLM-EI paper. Section 0's prior-writing instruction and
  section 4 (questions to answer "before writing the prior") are instructions for the PSL
  elicitation, not task information, and are not shown. Setup values are printed with the
  card's own value tokens.

Answers: the prompt asks for the paper's answer (A/B and the reasoning in one JSON object) and
generation stops after 32 tokens, inside the reasoning field after a complete ``Setup`` field
(the model writes the answer first); :func:`check_survey` then requires every answer to be
valid at the first attempt with the ``Setup`` field first, or the task fails. The answers
record this as ``mode: reasoning-stop``.

Sampling as the HapBO elicitation (temperature 0, seed 7, thinking off). Validation as the
other elicitations: at most 4 attempts per question; on a malformed answer the exact
validator error is fed back as a corrective turn. A question still invalid after 4 attempts
is recorded as invalid and left out of the preference data (never imputed).

Output::

    results/llm_ei_surveys/hpobench-<fam>-<task>/<card>/      working files (not committed):
        survey.json, answers-reasoning-stop.jsonl (raw answers), prompt_samples-reasoning-stop.md, llm_cache/,
        llm_calls/
    experiments/priors/hpobench-<fam>-<task>/<task>/llm-ei-card/survey_answers.json             (llm-card)
    experiments/priors/hpobench-<fam>-<task>/<task>/llm-ei-card-misleading/survey_answers.json  (llm-card-misleading)

``survey_answers.json`` holds the survey settings, a hash of the question list and one
character per question (``A``, ``B``, or ``-`` for an invalid answer); the utility is fitted
from it.

Usage (LLM endpoint up, see REPRODUCING.md)::

    python -m src.experiments.llm_ei_survey --card llm-card --all --rounds 10 --workers 12
    python -m src.experiments.llm_ei_survey --card llm-card --tasks svm-vehicle --rounds 10 --render   # prompts only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# source card directory -> output directory of the compact answers and the utility
CARD_IDS = {"llm-card": "llm-ei-card", "llm-card-misleading": "llm-ei-card-misleading"}
MODE = "reasoning-stop"  # recorded in survey_answers.json
MAX_TOKENS = 32  # generation stops inside the reasoning field, after the answer
MAX_ATTEMPTS = 4
DEFAULT_ROUNDS = 10
DEFAULT_M_MAX = 800
DEFAULT_SURVEY_SEED = 20260
ARTIFACT_FORMAT = "llm-ei-survey-v1"
ANSWERS_FORMAT = "llm-ei-survey-answers-v1"
ANSWERS_FILE = "survey_answers.json"

SYSTEM_PROMPT = "You are an expert in the domain described below."

TASK_TEXT = (
    "## Task\n"
    "For the hyperparameter optimization problem above, predict which setup leads to a "
    "higher objective value and output the response (A or B) along with reasoning in "
    "JSON format."
)
OUTPUT_TEXT = (
    "Output: Setup (A or B) with higher objective value and the reasoning in a JSON "
    "object with 'Setup' and 'reasoning' keys."
)


class SurveyAnswerError(ValueError):
    """An answer that does not name exactly one of the two setups."""


# ---------------------------------------------------------------------------
# card
# ---------------------------------------------------------------------------


def card_path(model: str, task: str, card: str, *, priors_root: Path | None = None) -> Path:
    slug = task.replace("_", "-")
    root = priors_root or (_ROOT / "experiments" / "priors")
    return root / f"hpobench-{model}-{slug}" / slug / card / "prompt_card.md"


_SECTION = re.compile(r"^## (.+)$", re.M)
_CONTEXT_SECTIONS = ("1. Domain", "2. Search space", "3. Optimization description")
_KNOWN_SECTIONS = ("0. Prompt", *_CONTEXT_SECTIONS,
                   "4. Questions to consider before writing the prior", "Domain guidance")


def card_sections(text: str) -> dict[str, str]:
    """``{heading: section text incl. its heading line}``; fails on unknown headings."""

    text = text.replace("\r\n", "\n")
    marks = list(_SECTION.finditer(text))
    sections: dict[str, str] = {}
    for i, mark in enumerate(marks):
        heading = mark.group(1).strip()
        if heading not in _KNOWN_SECTIONS:
            raise ValueError(f"unexpected card section {heading!r}")
        if heading in sections:
            raise ValueError(f"duplicate card section {heading!r}")
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        sections[heading] = text[mark.start():end].strip()
    missing = [h for h in _CONTEXT_SECTIONS if h not in sections]
    if missing:
        raise ValueError(f"card lacks sections {missing}")
    return sections


def card_context(text: str) -> str:
    """Sections 1-3 and (misleading card) the Domain guidance section, verbatim."""

    sections = card_sections(text)
    parts = [sections[h] for h in _CONTEXT_SECTIONS]
    if "Domain guidance" in sections:
        parts.append(sections["Domain guidance"])
    return "\n\n".join(parts)


def card_value_tokens(text: str) -> dict[str, list[tuple[float, str]]]:
    """Per card parameter name: its listed values as (number, printed token)."""

    space = card_sections(text)["2. Search space"]
    out: dict[str, list[tuple[float, str]]] = {}
    for block in re.split(r"\n\s*\n", space):
        name = re.search(r"^name: (\S+)$", block, re.M)
        values = re.search(r"^values: \[(.*?)\]", block, re.M)
        if not name or not values:
            continue
        tokens = [token.strip() for token in values.group(1).split(",")]
        out[name.group(1)] = [(float(token), token) for token in tokens]
    if not out:
        raise ValueError("card lists no parameter values")
    return out


def _token_for(value: float, listed: Sequence[tuple[float, str]]) -> str:
    for number, token in listed:
        if math.isclose(float(value), number, rel_tol=1e-6, abs_tol=1e-12):
            return token
    raise ValueError(f"grid value {value!r} is not among the card's listed values")


class TaskPrompter:
    """Renders survey questions for one task/card."""

    def __init__(self, card_text: str, parameter_names: Sequence[str]) -> None:
        self.card_text = card_text.replace("\r\n", "\n")
        self.context = card_context(self.card_text)
        tokens = card_value_tokens(self.card_text)
        by_lower = {name.lower(): name for name in tokens}
        self.names: list[tuple[str, str]] = []  # (grid name, card name), card order
        grid_by_lower = {str(n).lower(): str(n) for n in parameter_names}
        if set(by_lower) != set(grid_by_lower):
            raise ValueError(f"card parameters {sorted(by_lower)} != grid parameters {sorted(grid_by_lower)}")
        for lower, card_name in by_lower.items():
            self.names.append((grid_by_lower[lower], card_name))
        self.tokens = tokens

    def setup_block(self, label: str, config: Mapping[str, Any]) -> str:
        lines = [f"Setup {label}:"]
        for grid_name, card_name in self.names:
            lines.append(f"{card_name}: {_token_for(float(config[grid_name]), self.tokens[card_name])}")
        return "\n".join(lines)

    def render(self, config_a: Mapping[str, Any], config_b: Mapping[str, Any]) -> tuple[str, str]:
        """``(system, user)`` of one question."""

        user = "\n\n".join((
            self.context,
            TASK_TEXT,
            self.setup_block("A", config_a),
            self.setup_block("B", config_b),
            OUTPUT_TEXT,
        ))
        return SYSTEM_PROMPT, user


# ---------------------------------------------------------------------------
# survey design
# ---------------------------------------------------------------------------


def build_survey(
    n_grid: int,
    rounds: int = DEFAULT_ROUNDS,
    *,
    m_max: int = DEFAULT_M_MAX,
    seed: int = DEFAULT_SURVEY_SEED,
) -> dict[str, Any]:
    """Surveyed candidates and the questions of ``rounds`` rounds (see module docstring).

    The first r rounds of an L-round survey are the r-round survey (same seed)."""

    if n_grid < 2:
        raise ValueError("need at least two candidates")
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    rng = np.random.default_rng(seed)
    if n_grid > m_max:
        items = np.sort(rng.choice(n_grid, size=m_max, replace=False))
    else:
        items = np.arange(n_grid)
    seen: set[tuple[int, int]] = set()
    questions: list[dict[str, int]] = []
    removed = {"self": 0, "repeat": 0}
    for current in range(1, int(rounds) + 1):
        left, right = rng.permutation(items), rng.permutation(items)
        for a, b in zip(left.tolist(), right.tolist()):
            if a == b:
                removed["self"] += 1
                continue
            key = (min(a, b), max(a, b))
            if key in seen:
                removed["repeat"] += 1
                continue
            seen.add(key)
            questions.append({"q": len(questions), "a": int(a), "b": int(b), "round": current})
    return {
        "artifact_format": ARTIFACT_FORMAT,
        "n_grid": int(n_grid),
        "m_max": int(m_max),
        "seed": int(seed),
        "rounds": int(rounds),
        "items": items.astype(int).tolist(),
        "n_items": int(items.size),
        "n_questions": len(questions),
        "removed": removed,
        "questions": questions,
    }


def questions_sha256(questions: Sequence[Mapping[str, Any]]) -> str:
    """sha256 of the ordered question list ``[[a, b], ...]`` (grid indices)."""

    payload = json.dumps([[int(q["a"]), int(q["b"])] for q in questions], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# answers
# ---------------------------------------------------------------------------

_SETUP_FIELD = re.compile(r"""["']setup["']\s*:\s*["']?\s*(?:setup\s+)?([AB])\b""", re.I)


def parse_answer(text: str) -> tuple[str, str]:
    """``("A"|"B", how)`` from one answer, or raise :class:`SurveyAnswerError`.

    The JSON object is parsed first; a truncated or otherwise malformed object is accepted
    only when its ``Setup`` field still names exactly one setup (``how`` = ``"regex"``)."""

    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            document = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            document = None
        if isinstance(document, dict):
            keys = {str(k).strip().lower(): k for k in document}
            if "setup" not in keys:
                raise SurveyAnswerError("the JSON object has no 'Setup' key")
            value = str(document[keys["setup"]]).strip().upper()
            value = re.sub(r"^SETUP\s+", "", value)
            if value not in ("A", "B"):
                raise SurveyAnswerError(f"'Setup' must be \"A\" or \"B\", got {document[keys['setup']]!r}")
            return value, "json"
    found = {m.group(1).upper() for m in _SETUP_FIELD.finditer(text)}
    if len(found) == 1:
        return found.pop(), "regex"
    if not found:
        raise SurveyAnswerError("no JSON object with a 'Setup' key in the answer")
    raise SurveyAnswerError("the answer names both setups")


def ask_question(client: Any, system: str, user: str, *, seed: int, max_attempts: int = MAX_ATTEMPTS) -> dict[str, Any]:
    """Validate-and-retry loop for one question (never raises on an invalid answer)."""

    prompt = user
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        raw = client.complete(prompt, system=system, response_schema=None, seed=seed)
        try:
            answer, how = parse_answer(raw)
        except SurveyAnswerError as exc:
            attempts.append({"attempt": attempt, "error": str(exc), "raw": raw})
            prompt = (
                user
                + f"\n\n## Your previous answer (#{attempt}) was REJECTED\n"
                + f"Validator error: {exc}\nReturn ONLY the corrected JSON object."
            )
            continue
        return {"answer": answer, "parse": how, "attempts": attempt, "raw": raw,
                "rejected": [{"attempt": a["attempt"], "error": a["error"]} for a in attempts]}
    return {"answer": None, "parse": None, "attempts": max_attempts, "raw": attempts[-1]["raw"],
            "rejected": [{"attempt": a["attempt"], "error": a["error"]} for a in attempts]}


def setup_first(raw: str) -> bool:
    """True when the 'Setup' key precedes any 'reasoning' key in a raw answer."""

    low = str(raw or "").lower()
    i_setup, i_reason = low.find('"setup"'), low.find('"reasoning"')
    return i_setup >= 0 and (i_reason < 0 or i_setup < i_reason)


def check_survey(rows: Mapping[int, Mapping[str, Any]], n_questions: int) -> dict[str, Any]:
    """Counts over the answers to questions 0..n_questions-1: unanswered, invalid, retried, parse
    paths, answer field first. Every answer must be valid at the first attempt with the 'Setup'
    field before the reasoning (the stop point is after it)."""

    plain = [rows[q] for q in range(n_questions) if q in rows]
    report = {"questions": n_questions, "answered": len(plain),
              "invalid": sum(r["answer"] is None for r in plain),
              "retried": sum(int(r["attempts"]) > 1 for r in plain),
              "parse_regex": sum(r.get("parse") == "regex" for r in plain),
              "parse_json": sum(r.get("parse") == "json" for r in plain),
              "setup_first": sum(setup_first(r.get("raw", "")) for r in plain)}
    problems = []
    if report["answered"] < n_questions:
        problems.append(f"{n_questions - report['answered']} questions unanswered")
    if report["invalid"] or report["retried"]:
        problems.append(f"reasoning-stop needs a valid first answer everywhere: {report['invalid']} invalid, "
                        f"{report['retried']} retried")
    if report["setup_first"] < report["answered"]:
        problems.append(f"reasoning-stop needs the answer field first: {report['answered'] - report['setup_first']} "
                        "answers start with the reasoning")
    report["ok"] = not problems
    report["problems"] = problems
    return report


def compact_answers(
    *,
    model: str,
    task: str,
    card: str,
    n_grid: int,
    survey: Mapping[str, Any],
    rows: Mapping[int, Mapping[str, Any]],
    llm: Mapping[str, Any] | None,
    card_sha256: str,
    context_sha256: str,
) -> dict[str, Any]:
    """The committed form of an answered survey: settings, question hash, one character per question."""

    questions = survey["questions"]
    check = check_survey(rows, len(questions))
    if check["answered"] < len(questions):
        raise ValueError(f"{len(questions) - check['answered']} questions unanswered")
    answers = "".join("-" if rows[q["q"]]["answer"] is None else str(rows[q["q"]]["answer"]) for q in questions)
    return {
        "artifact_format": ANSWERS_FORMAT,
        "task": {"model": model, "task": task, "n_candidates": int(n_grid)},
        "card": card,
        "card_sha256": card_sha256,
        "context_sha256": context_sha256,
        "survey": {"rounds": int(survey["rounds"]), "m_max": int(survey["m_max"]), "seed": int(survey["seed"]),
                   "n_items": int(survey["n_items"]), "n_questions": len(questions),
                   "questions_sha256": questions_sha256(questions)},
        "mode": MODE,
        "llm": dict(llm) if llm else None,
        "check": {k: check[k] for k in ("answered", "invalid", "retried", "parse_regex", "parse_json", "setup_first")},
        "answers": answers,
    }


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def _llm_settings() -> dict[str, Any]:
    return {
        "model": os.getenv("LLM_MODEL", "Qwen3.5-122B-A10B-GPTQ-Int4"),
        "provider": os.getenv("LLM_PROVIDER", "openai"),
        "base_url": os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
        "temperature": float(os.getenv("LLMEI_TEMPERATURE", "0.0")),
        "seed": int(os.getenv("LLMEI_SEED", "7")),
        "max_tokens": int(os.getenv("LLMEI_MAX_TOKENS", str(MAX_TOKENS))),
        "enable_thinking": str(os.getenv("LLM_ENABLE_THINKING", "false")).lower() in {"1", "true", "yes"},
    }


class _ClientPool:
    """One LLMClient per worker thread (shared cache directory, per-thread call log)."""

    def __init__(self, out_dir: Path, settings: Mapping[str, Any]) -> None:
        self.out_dir = out_dir
        self.settings = dict(settings)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._count = 0

    def get(self):
        client = getattr(self._local, "client", None)
        if client is None:
            from llm.client import LLMClient

            with self._lock:
                self._count += 1
                k = self._count
            client = LLMClient(
                model=self.settings["model"], temperature=self.settings["temperature"],
                max_tokens=self.settings["max_tokens"], cache_dir=self.out_dir / "llm_cache",
                budget=10**9, provider=self.settings["provider"], base_url=self.settings["base_url"],
                api_key=os.getenv("LLM_API_KEY", "none"), enable_thinking=self.settings["enable_thinking"],
                request_timeout=600.0,
            )
            client.set_run_directory(self.out_dir / "llm_calls" / f"worker-{k:02d}")
            self._local.client = client
        return client


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def work_dir(model: str, task: str, card: str, root: Path | None = None) -> Path:
    slug = task.replace("_", "-")
    return (root or (_ROOT / "results" / "llm_ei_surveys")) / f"hpobench-{model}-{slug}" / card


def answers_path(model: str, task: str, card: str, *, priors_root: Path | None = None) -> Path:
    slug = task.replace("_", "-")
    root = priors_root or (_ROOT / "experiments" / "priors")
    return root / f"hpobench-{model}-{slug}" / slug / CARD_IDS[card] / ANSWERS_FILE


def _read_answers(path: Path) -> dict[int, dict[str, Any]]:
    done: dict[int, dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[int(row["q"])] = row
    return done


def run_survey(
    model: str,
    task: str,
    card: str,
    *,
    rounds: int = DEFAULT_ROUNDS,
    workers: int = 8,
    m_max: int = DEFAULT_M_MAX,
    survey_seed: int = DEFAULT_SURVEY_SEED,
    render_only: bool = False,
    work_root: Path | None = None,
    priors_root: Path | None = None,
    benchmark_directory: Path | None = None,
) -> dict[str, Any]:
    from datasets.hpobench import load_hpobench_problem

    kwargs = {} if benchmark_directory is None else {"benchmark_directory": benchmark_directory}
    problem = load_hpobench_problem(model=model, task=task, **kwargs)
    card_text = card_path(model, task, card, priors_root=priors_root).read_text(encoding="utf-8")
    prompter = TaskPrompter(card_text, problem.parameter_names)
    out = work_dir(model, task, card, work_root)
    out.mkdir(parents=True, exist_ok=True)

    survey = build_survey(problem.n_candidates, rounds, m_max=m_max, seed=survey_seed)
    survey.update({"model": model, "task": task, "card": card, "card_sha256": _sha256_text(card_text),
                   "system": SYSTEM_PROMPT, "context_sha256": _sha256_text(prompter.context)})
    survey_path = out / "survey.json"
    if survey_path.is_file():
        old = json.loads(survey_path.read_text(encoding="utf-8"))
        if (questions_sha256(old["questions"]), old.get("context_sha256")) != (
                questions_sha256(survey["questions"]), survey["context_sha256"]):
            raise SystemExit(f"{survey_path} holds another survey or card context; use another --work-root")
    survey_path.write_text(json.dumps(survey, indent=1) + "\n", encoding="utf-8")
    native = problem.native_candidates
    questions = survey["questions"]

    # Prompt samples for the audit (first three questions, as sent).
    samples = []
    for q in questions[:3]:
        system, user = prompter.render(native[q["a"]], native[q["b"]])
        samples.append(f"### question {q['q']} (A = grid {q['a']}, B = grid {q['b']})\n\n**system**\n\n```\n{system}\n```\n\n"
                       f"**user**\n\n```\n{user}\n```\n")
    (out / f"prompt_samples-{MODE}.md").write_text(
        f"# LLM-EI survey prompts: {model}-{task}, card {card}\n\n" + "\n".join(samples),
        encoding="utf-8")
    if render_only:
        return {"task": f"{model}-{task}", "card": card, "status": "rendered", "n_questions": len(questions)}

    raw_path = out / f"answers-{MODE}.jsonl"
    done = _read_answers(raw_path)
    todo = [q for q in questions if q["q"] not in done]
    settings = _llm_settings()
    pool = _ClientPool(out, settings)

    def _one(q: Mapping[str, int]) -> dict[str, Any]:
        a, b = int(q["a"]), int(q["b"])
        system, user = prompter.render(native[a], native[b])
        t0 = time.time()
        result = ask_question(pool.get(), system, user, seed=settings["seed"])
        winner = None if result["answer"] is None else (a if result["answer"] == "A" else b)
        return {"q": int(q["q"]), "a": a, "b": b, "answer": result["answer"], "winner": winner,
                "parse": result["parse"], "attempts": result["attempts"], "rejected": result["rejected"],
                "latency_s": round(time.time() - t0, 3), "raw": result["raw"]}

    started = time.time()
    n_new = n_invalid = 0
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        with raw_path.open("a", encoding="utf-8") as handle:
            futures = [executor.submit(_one, q) for q in todo]
            for future in as_completed(futures):
                row = future.result()  # a provider error ends the survey here (fail fast)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                n_new += 1
                n_invalid += row["answer"] is None
                if n_new % 200 == 0:
                    rate = n_new / max(time.time() - started, 1e-9)
                    print(f"[{model}-{task} {card}] {n_new}/{len(todo)} answered, {rate:.2f} q/s, "
                          f"{n_invalid} invalid", flush=True)
    except BaseException:
        executor.shutdown(wait=True, cancel_futures=True)  # drop the queued questions; resumable later
        raise
    executor.shutdown(wait=True)
    elapsed = time.time() - started

    rows = _read_answers(raw_path)
    check = check_survey(rows, len(questions))
    summary = {"task": f"{model}-{task}", "card": card, "n_questions": len(questions),
               "new": n_new, "invalid_new": n_invalid, "elapsed_s": round(elapsed, 1), "check": check,
               "finished_at_utc": datetime.now(timezone.utc).isoformat()}
    if not check["ok"]:
        summary["status"] = "check-failed"
        return summary
    document = compact_answers(model=model, task=task, card=card, n_grid=problem.n_candidates, survey=survey,
                               rows=rows, llm=settings, card_sha256=survey["card_sha256"],
                               context_sha256=survey["context_sha256"])
    target = answers_path(model, task, card, priors_root=priors_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    summary.update({"status": "ok", "answers": str(target)})
    return summary


def _parse_tasks(tokens: Sequence[str]) -> list[tuple[str, str]]:
    from datasets.hpobench import MODELS, TASKS

    pairs = []
    for token in tokens:
        model, _, task = token.partition("-")
        task = task.replace("-", "_")
        if model not in MODELS or task not in TASKS:
            raise SystemExit(f"unknown task {token!r}")
        pairs.append((model, task))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--card", choices=sorted(CARD_IDS), required=True)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help="survey size L (paper: 10)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--m-max", type=int, default=DEFAULT_M_MAX)
    parser.add_argument("--survey-seed", type=int, default=DEFAULT_SURVEY_SEED)
    parser.add_argument("--render", action="store_true", help="write survey.json + prompt samples only (no LLM)")
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--benchmark-directory", type=Path, default=None)
    args = parser.parse_args()
    if args.all:
        from datasets.hpobench import MODELS, TASKS

        pairs = [(m, t) for m in MODELS for t in TASKS]
    else:
        pairs = _parse_tasks(args.tasks)
    if not pairs:
        raise SystemExit("specify --all or --tasks <model-task> [...]")
    failed = 0
    for model, task in pairs:
        try:
            row = run_survey(model, task, args.card, rounds=args.rounds, workers=args.workers,
                             m_max=args.m_max, survey_seed=args.survey_seed, render_only=args.render,
                             work_root=args.work_root, benchmark_directory=args.benchmark_directory)
            failed += row["status"] not in ("ok", "rendered")
        except Exception as exc:  # noqa: BLE001 -- one task's provider failure must not hide the others
            failed += 1
            row = {"task": f"{model}-{task}", "card": args.card, "status": "failed",
                   "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(row), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
