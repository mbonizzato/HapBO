"""LLM-EI utility g(x): preference GP fitted to an answered LLM survey (offline step).

Following Patel et al. (Digital Discovery 2026, Sec. 2.2.2), the answered survey
{x_winner > x_loser} is modelled with Chu & Ghahramani's preference GP as implemented by
BoTorch's ``PairwiseGP`` (probit likelihood, Laplace approximation, hyperparameters by MAP
via ``fit_gpytorch_mll`` with BoTorch's default kernel and priors), and g is the posterior
mean. The GP inputs are the task's shared GP encoding (the array the vanilla and prior BO
arms use, from the schema of the task's ``base`` prior); g is predicted on the full
candidate grid, which is what the LLM-EI arm reads (``src/experiments/llm_ei.py``).

Input: ``survey_answers.json`` (written by ``src/experiments/llm_ei_survey.py``). The question
list is rebuilt from the recorded survey settings and checked against the recorded hash;
invalid answers are left out and counted.

Output, next to the answers::

    experiments/priors/hpobench-<fam>-<task>/<task>/llm-ei-card[-misleading]/llm_ei_utility.json.gz

Usage (BoTorch in the interpreter, see requirements-baselines.txt)::

    python -m src.experiments.llm_ei_utility --card llm-card --all
    python -m src.experiments.llm_ei_utility --card llm-card-misleading --tasks svm-vehicle --out-root /tmp/check
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.llm_ei import ARTIFACT_FORMAT, grid_fingerprint  # noqa: E402
from experiments.llm_ei_survey import (  # noqa: E402
    ANSWERS_FORMAT, CARD_IDS, MODE, answers_path, build_survey, questions_sha256,
)

DEFAULT_FIT_SEED = 0
UTILITY_FILE = "llm_ei_utility.json.gz"


# ---------------------------------------------------------------------------
# survey -> comparisons
# ---------------------------------------------------------------------------


def load_answers(path: Path, n_grid: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(answers document, survey design)``; the design is rebuilt and checked against the file."""

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("artifact_format") != ANSWERS_FORMAT:
        raise ValueError(f"{path}: not an {ANSWERS_FORMAT} file")
    settings = document["survey"]
    if int(document["task"]["n_candidates"]) != int(n_grid):
        raise ValueError(f"{path} was answered on another grid")
    survey = build_survey(int(n_grid), int(settings["rounds"]), m_max=int(settings["m_max"]), seed=int(settings["seed"]))
    if survey["n_questions"] != int(settings["n_questions"]) or len(document["answers"]) != survey["n_questions"]:
        raise ValueError(f"{path}: the question count does not match the survey settings")
    if questions_sha256(survey["questions"]) != settings["questions_sha256"]:
        raise ValueError(f"{path}: the rebuilt question list does not match the recorded hash")
    return document, survey


def comparisons(questions: list[Mapping[str, Any]], answers: str) -> tuple[np.ndarray, dict[str, int]]:
    """``(k x 2 array [winner, loser] of grid indices, counts)``; ``answers[i]`` is ``A``, ``B`` or ``-``."""

    if len(answers) != len(questions):
        raise ValueError("one answer per question is required")
    pairs = []
    invalid = 0
    for q, answer in zip(questions, answers):
        if answer == "-":
            invalid += 1
            continue
        if answer not in ("A", "B"):
            raise ValueError(f"unexpected answer {answer!r}")
        a, b = int(q["a"]), int(q["b"])
        pairs.append((a, b) if answer == "A" else (b, a))
    counts = {"asked": len(questions), "valid": len(pairs), "invalid": invalid}
    return np.asarray(pairs, dtype=int).reshape(-1, 2), counts


# ---------------------------------------------------------------------------
# encoding / preference GP
# ---------------------------------------------------------------------------


def encoding_prior_relpath(model: str, task: str) -> str:
    """The task's shared GP encoding: schema of its ``base`` prior (as vanilla-bo-ei in hpo/vanilla-random)."""

    slug = task.replace("_", "-")
    return f"experiments/priors/hpobench-{model}-{slug}/{slug}/base/prior.json"


def task_encoding(problem: Any, encoding_prior: Path) -> tuple[np.ndarray, str]:
    """The task's shared GP encoding, computed exactly as the vanilla BO arm does."""

    from experiments.worker import _task_gp_encoding

    return _task_gp_encoding(problem, "hpobench", Path(encoding_prior))


def _accept_line_search_stop(w: Any) -> bool:
    """BoTorch's default warning handler, plus: an L-BFGS-B line-search stop ("ABNORMAL")
    is accepted as converged instead of triggering a refit from prior-sampled values."""

    from botorch.exceptions.warnings import OptimizationWarning
    from botorch.fit import DEFAULT_WARNING_HANDLER

    if issubclass(w.category, OptimizationWarning) and "ABNORMAL" in str(w.message):
        return True
    return DEFAULT_WARNING_HANDLER(w)


def fit_preference_gp(X_items: np.ndarray, pairs_local: np.ndarray, *, seed: int = DEFAULT_FIT_SEED) -> tuple[Any, dict[str, Any]]:
    """Fit BoTorch's PairwiseGP (defaults) by MAP; return (model, diagnostics).

    Fit policy: ``fit_gpytorch_mll`` with BoTorch defaults (up to 5 attempts). If every
    attempt ends in an L-BFGS-B line-search stop (``ModelFittingError``; seen on large,
    nearly separable surveys), the model is rebuilt and fitted once more accepting that
    stop as converged; ``diagnostics["fit_policy"]`` records which policy produced g."""

    import torch
    import botorch
    from botorch.exceptions.errors import ModelFittingError
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import PairwiseGP
    from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood

    X = torch.as_tensor(np.asarray(X_items, dtype=float), dtype=torch.float64)
    C = torch.as_tensor(np.asarray(pairs_local, dtype=int), dtype=torch.long)
    t0 = time.perf_counter()
    policy = "botorch-default"
    torch.manual_seed(int(seed))
    model = PairwiseGP(X, C)
    mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
    try:
        fit_gpytorch_mll(mll)
    except ModelFittingError:
        policy = "accept-line-search-stop"
        torch.manual_seed(int(seed))
        model = PairwiseGP(X, C)
        mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll, warning_handler=_accept_line_search_stop)
    model.eval()
    fit_s = time.perf_counter() - t0
    covar = model.covar_module
    diagnostics: dict[str, Any] = {"fit_s": round(fit_s, 3), "fit_policy": policy, "n_datapoints": int(X.shape[0]),
                                   "n_comparisons": int(C.shape[0]), "botorch": botorch.__version__,
                                   "torch": torch.__version__, "fit_seed": int(seed)}
    try:
        diagnostics["outputscale"] = float(covar.outputscale.detach())
        diagnostics["lengthscale"] = covar.base_kernel.lengthscale.detach().reshape(-1).tolist()
    except AttributeError:
        pass
    return model, diagnostics


def predict_mean(model: Any, X_grid: np.ndarray, *, chunk: int = 2048) -> np.ndarray:
    """Posterior mean of the preference GP on every grid row (chunked: no full covariance)."""

    import torch

    out = []
    with torch.no_grad():
        for start in range(0, X_grid.shape[0], chunk):
            Xc = torch.as_tensor(np.asarray(X_grid[start:start + chunk], dtype=float), dtype=torch.float64)
            out.append(model.posterior(Xc).mean.reshape(-1).numpy())
    return np.concatenate(out).astype(float)


def fit_utility(encoded: np.ndarray, pairs: np.ndarray, *, seed: int = DEFAULT_FIT_SEED) -> tuple[np.ndarray, dict[str, Any]]:
    """g on the full grid from ``pairs`` ([winner, loser] grid indices)."""

    if pairs.size == 0:
        raise ValueError("no valid comparisons")
    items = np.unique(pairs.reshape(-1))
    local = {int(g): i for i, g in enumerate(items.tolist())}
    pairs_local = np.vectorize(local.__getitem__)(pairs)
    model, diagnostics = fit_preference_gp(encoded[items], pairs_local, seed=seed)
    t0 = time.perf_counter()
    g = predict_mean(model, encoded)
    diagnostics["predict_s"] = round(time.perf_counter() - t0, 3)
    if not np.all(np.isfinite(g)):
        raise FloatingPointError("non-finite utility")
    return g, diagnostics


# ---------------------------------------------------------------------------
# artifact
# ---------------------------------------------------------------------------


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Compact JSON, gzip-compressed without a file name or timestamp (reproducible bytes)."""

    data = (json.dumps(artifact, separators=(",", ":")) + "\n").encode("utf-8")
    with Path(path).open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
        handle.write(data)


def survey_provenance(document: Mapping[str, Any], counts: Mapping[str, int]) -> dict[str, Any]:
    """The ``survey`` block of a utility artifact."""

    settings = document["survey"]
    check = document.get("check") or {}
    return {
        "mode": document["mode"], "rounds": int(settings["rounds"]), "n_questions": int(settings["n_questions"]),
        **counts, "retried": int(check.get("retried", 0)), "parse_regex": int(check.get("parse_regex", 0)),
        "n_items_surveyed": int(settings["n_items"]), "m_max": int(settings["m_max"]),
        "survey_seed": int(settings["seed"]), "questions_sha256": settings["questions_sha256"],
        "answers_sha256": hashlib.sha256(document["answers"].encode("ascii")).hexdigest(),
        "card_sha256": document.get("card_sha256"),
    }


def build_utility(
    model: str,
    task: str,
    card: str,
    *,
    priors_root: Path | None = None,
    out_root: Path | None = None,
    seed: int = DEFAULT_FIT_SEED,
    benchmark_directory: Path | None = None,
) -> dict[str, Any]:
    from datasets.hpobench import load_hpobench_problem

    kwargs = {} if benchmark_directory is None else {"benchmark_directory": benchmark_directory}
    problem = load_hpobench_problem(model=model, task=task, **kwargs)
    source = answers_path(model, task, card, priors_root=priors_root)
    document, survey = load_answers(source, problem.n_candidates)
    if document["mode"] != MODE:
        raise ValueError(f"{source}: survey mode {document['mode']!r}, expected {MODE!r}")
    check = document.get("check") or {}
    if check.get("invalid") or check.get("retried") or check.get("setup_first") != check.get("answered"):
        raise ValueError(f"{source}: a reasoning-stop survey needs a valid first answer with the answer field first")
    pairs, counts = comparisons(survey["questions"], document["answers"])
    encoding_rel = encoding_prior_relpath(model, task)
    encoded, provenance = task_encoding(problem, _ROOT / encoding_rel)
    g, diagnostics = fit_utility(encoded, pairs, seed=seed)
    artifact = {
        "artifact_format": ARTIFACT_FORMAT,
        "task": {"model": model, "task": task, "n_candidates": int(problem.n_candidates)},
        "card": card,
        "grid_fingerprint": grid_fingerprint(problem.native_candidates, problem.parameter_names),
        "encoding_prior_path": encoding_rel,
        "encoding_provenance": provenance.replace(str(_ROOT.as_posix()) + "/", ""),
        "survey": survey_provenance(document, counts),
        "llm": document.get("llm"),
        "preference_model": {"model": "botorch.models.PairwiseGP (defaults, probit, Laplace/MAP)", **diagnostics},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "utility": [round(float(v), 8) for v in g],
    }
    out = source.parent if out_root is None else Path(out_root) / source.parent.relative_to(source.parents[3])
    out.mkdir(parents=True, exist_ok=True)
    path = out / UTILITY_FILE
    write_artifact(path, artifact)
    return {"task": f"{model}-{task}", "card": card, **counts, "fit_s": diagnostics["fit_s"],
            "fit_policy": diagnostics["fit_policy"], "path": str(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--card", choices=sorted(CARD_IDS), required=True)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out-root", type=Path, default=None,
                        help="write <out-root>/hpobench-<fam>-<task>/<task>/<dir>/llm_ei_utility.json.gz instead")
    parser.add_argument("--seed", type=int, default=DEFAULT_FIT_SEED)
    parser.add_argument("--threads", type=int, default=int(os.getenv("LLMEI_TORCH_THREADS", "8")))
    args = parser.parse_args()
    import torch

    torch.set_num_threads(args.threads)
    from datasets.hpobench import MODELS, TASKS
    from experiments.llm_ei_survey import _parse_tasks

    pairs = [(m, t) for m in MODELS for t in TASKS] if args.all else _parse_tasks(args.tasks)
    failed = 0
    for model, task in pairs:
        try:
            row = build_utility(model, task, args.card, out_root=args.out_root, seed=args.seed)
        except (ValueError, FileNotFoundError, FloatingPointError) as exc:
            failed += 1
            row = {"task": f"{model}-{task}", "card": args.card, "status": "failed", "error": str(exc)}
        print(json.dumps(row), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
