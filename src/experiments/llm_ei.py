"""LLM-EI baseline (Patel et al., Digital Discovery 2026): run-time side.

The arm is vanilla BO with expected improvement on the finite candidate set (same GP,
same fixed hyperparameters, same GP input encoding, same seeded initial design size),
except that the acquisition is restricted to the candidates whose LLM utility g(x) is at
or above the p(n)-th percentile of g over the candidate set (the paper's indicator weight
pi(g(x), p(n))), and the first query is drawn at random from that set. n is the number of
observations made so far. The paper's schedule is p(n) = 85 for n <= 30, 15 for
30 < n <= 40 and 0 (no restriction) afterwards; at a budget of 16 the acquisition is EI
over the top 15 % of g throughout.

g is read from a utility artifact (``llm_ei_utility.json.gz``) written offline by
``src/experiments/llm_ei_utility.py`` (LLM survey -> preference GP -> posterior mean on the full
grid); no LLM is called and no preference model is fitted at BO time. The artifact carries
a fingerprint of the candidate grid it was predicted on and the prior file whose schema
gives the task's shared GP input encoding; both are checked here. A plain ``.json``
artifact is read as well.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ARTIFACT_FORMAT = "llm-ei-utility-v1"
PAPER_SCHEDULE: tuple[tuple[int, float], ...] = ((30, 85.0), (40, 15.0))
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def grid_fingerprint(native_candidates: Sequence[Mapping[str, Any]], parameter_names: Sequence[str]) -> str:
    """sha256 of the candidate grid in its native order (names + values, 12 significant digits)."""

    rows = [[f"{float(candidate[name]):.12g}" for name in parameter_names] for candidate in native_candidates]
    payload = json.dumps({"names": list(parameter_names), "rows": rows}, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_schedule(raw: Any) -> tuple[tuple[int, float], ...]:
    """``[[n_max, percentile], ...]`` (ascending n_max); beyond the last step p = 0."""

    if raw is None:
        return PAPER_SCHEDULE
    steps = tuple((int(n_max), float(p)) for n_max, p in raw)
    if not steps or any(b[0] <= a[0] for a, b in zip(steps, steps[1:])):
        raise ValueError("llm_ei_schedule must be a non-empty list of [n_max, percentile] with ascending n_max")
    if any(not 0.0 <= p < 100.0 for _, p in steps):
        raise ValueError("llm_ei_schedule percentiles must lie in [0, 100)")
    return steps


def percentile_for(n: int, schedule: Sequence[tuple[int, float]]) -> float:
    for n_max, p in schedule:
        if n <= n_max:
            return float(p)
    return 0.0


@dataclass
class LLMEIContext:
    utility: np.ndarray
    schedule: tuple[tuple[int, float], ...]
    encoding_prior_path: Path
    artifact_path: Path
    artifact_sha256: str
    provenance: dict[str, Any]
    log: list[dict[str, Any]] = field(default_factory=list)

    def gate(self, n: int) -> np.ndarray | None:
        """Allowed-candidate mask for the query chosen after ``n`` observations (None = all)."""

        p = percentile_for(int(n), self.schedule)
        if p <= 0.0:
            mask = None
        else:
            threshold = float(np.percentile(self.utility, p))
            mask = self.utility >= threshold
        self.log.append({"n": int(n), "percentile": p,
                         "allowed": int(self.utility.size if mask is None else mask.sum())})
        return mask

    def summary(self) -> dict[str, Any]:
        return {
            "utility_artifact": self.artifact_path.as_posix(),
            "utility_artifact_sha256": self.artifact_sha256,
            "schedule": [list(step) for step in self.schedule],
            "encoding_prior_path": self.encoding_prior_path.as_posix(),
            "survey": self.provenance.get("survey"),
            "fit_policy": (self.provenance.get("preference_model") or {}).get("fit_policy"),
            "gate_log": self.log,
        }


def load_llm_ei_context(
    problem: Any,
    adapter: str,
    method: Mapping[str, Any],
    utility_path: Path | None,
) -> LLMEIContext:
    """Read and verify the utility artifact for this task (raises on any mismatch)."""

    if adapter != "hpobench":
        raise ValueError(f"llm-ei is wired for the HPOBench tables only, not {adapter!r}")
    if utility_path is None or not Path(utility_path).is_file():
        raise FileNotFoundError(f"llm-ei utility artifact missing: {utility_path}")
    raw_bytes = Path(utility_path).read_bytes()
    if Path(utility_path).suffix == ".gz":
        raw_bytes = gzip.decompress(raw_bytes)
    artifact = json.loads(raw_bytes.decode("utf-8"))
    if artifact.get("artifact_format") != ARTIFACT_FORMAT:
        raise ValueError(f"{utility_path}: not an {ARTIFACT_FORMAT} artifact")
    task = artifact.get("task", {})
    if (task.get("model"), task.get("task")) != (getattr(problem, "model", None), getattr(problem, "task_name", None)):
        raise ValueError(f"{utility_path}: utility is for {task}, run is {problem.model}-{problem.task_name}")
    fingerprint = grid_fingerprint(problem.native_candidates, problem.parameter_names)
    if artifact.get("grid_fingerprint") != fingerprint:
        raise ValueError(f"{utility_path}: candidate grid fingerprint mismatch")
    utility = np.asarray(artifact["utility"], dtype=float).reshape(-1)
    if utility.size != problem.n_candidates or not np.all(np.isfinite(utility)):
        raise ValueError(f"{utility_path}: utility must hold one finite value per candidate")
    encoding_prior_path = PROJECT_ROOT / str(artifact["encoding_prior_path"])
    return LLMEIContext(
        utility=utility,
        schedule=parse_schedule(method.get("llm_ei_schedule")),
        encoding_prior_path=encoding_prior_path,
        artifact_path=Path(utility_path),
        artifact_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        provenance={k: artifact.get(k) for k in ("survey", "card", "llm", "preference_model")},
    )
