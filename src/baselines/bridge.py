"""Subprocess callback bridge for running a faithful external baseline.

Published BO libraries drive their own optimization loops (they call the
objective). To reproduce a method faithfully while keeping the objective, budget,
seed, and finite candidate set under our control, we run the upstream method in
its own isolated conda env as a subprocess "worker". The worker optimises over
OUR finite candidate set; whenever it wants to evaluate a candidate it emits

    {"type": "eval", "index": i}

on stdout and blocks reading

    {"value": y}

on stdin. ``run_charged`` serves those callbacks with the harness's objective, so the
upstream algorithm (model, acquisition, LLM calls) is unchanged.

Budget rule: EVERY proposal consumes one budget unit.  An upstream method that relaxes
to a continuous space may re-propose an already-evaluated candidate after snapping; that
proposal returns the cached observation (the objective is not re-run, so noisy objectives
are not re-sampled) and is recorded as a query flagged ``repeat``.  The bridge never fills
the budget with random candidates: if the worker stops (``done``) before the budget is
consumed the trace ends there, the accounting records ``proposals_exhausted_at`` and the
harness fails the run (:func:`require_full_budget`).
``bridge_policy`` (optimizer kwarg) accepts only ``"charge"``.

Worker stderr and ``{"type": "log"}`` messages are written to ``worker_log_path``
when the harness sets it.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from collections import deque
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

from .base import BaselineOptimizer, Config, Observation, config_key

BRIDGE_POLICIES = ("charge",)


class BridgeWorkerError(RuntimeError):
    """The external baseline worker failed or violated the protocol."""


class SubprocessBridgeOptimizer(BaselineOptimizer):
    """Base adapter that drives an external optimizer over a JSON callback pipe.

    Subclasses set :attr:`worker_command` (the isolated-env interpreter + worker
    module) and may override :meth:`worker_config` / :meth:`worker_env`.
    """

    #: Command that launches the worker inside the baseline's isolated env.
    worker_command: list[str] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._proc: subprocess.Popen | None = None
        self._done = False
        self._observation_cache: dict[str, Observation] = {}
        self._trace: list[dict[str, Any]] = []
        self._stderr_lines: deque[str] = deque(maxlen=2_000)
        self._stderr_thread: threading.Thread | None = None
        policy = str(self.kwargs.get("bridge_policy", "charge")).strip().lower()
        if policy not in BRIDGE_POLICIES:
            raise ValueError(f"bridge_policy must be one of {BRIDGE_POLICIES}, got {policy!r}")
        self.bridge_policy = policy
        #: Optional path; worker stderr + protocol log messages are appended there.
        self.worker_log_path: str | os.PathLike | None = self.kwargs.get("worker_log_path")
        self._log_handle: Any = None
        self._log_lock = threading.Lock()

    # ---- subclass hooks -------------------------------------------------
    def worker_config(self) -> dict[str, Any]:
        """Baseline-specific settings forwarded to the worker (JSON-serializable)."""

        return {}

    def worker_env(self) -> dict[str, str] | None:
        """Extra environment variables for the worker process (e.g. LLM endpoint)."""

        return None

    def resolved_command(self) -> list[str]:
        if not self.worker_command:
            raise BridgeWorkerError(f"{type(self).__name__} did not set worker_command")
        return list(self.worker_command)

    # ---- protocol -------------------------------------------------------
    def _write_log(self, text: str) -> None:
        handle = self._log_handle
        if handle is None:
            return
        with self._log_lock:
            try:
                handle.write(text if text.endswith("\n") else text + "\n")
                handle.flush()
            except (OSError, ValueError):
                pass

    def _launch(self) -> None:
        if self.worker_log_path is not None and self._log_handle is None:
            os.makedirs(os.path.dirname(os.path.abspath(self.worker_log_path)) or ".", exist_ok=True)
            self._log_handle = open(self.worker_log_path, "a", encoding="utf-8")
        env = dict(os.environ)
        extra = self.worker_env()
        if extra:
            env.update({str(k): str(v) for k, v in extra.items()})
        self._proc = subprocess.Popen(
            self.resolved_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name=f"{self.name}-stderr",
            daemon=True,
        )
        self._stderr_thread.start()
        spec = {
            "candidates": [dict(candidate) for candidate in self.candidates],
            "encoded": self.encoded_candidates.tolist(),
            "budget": int(self.budget),
            "n_init": int(self.kwargs.get("n_initial", self.kwargs.get("n_init", 1))),
            "seed": None if self.seed is None else int(self.seed),
            "maximize": bool(self.objective_meta.get("maximize", True)),
            "config": self.worker_config(),
        }
        self._send(spec)

    def _send(self, obj: Any) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        try:
            self._proc.stdin.write(json.dumps(obj) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise BridgeWorkerError(f"worker pipe closed: {self._stderr()}") from exc

    def _read_message(self) -> dict[str, Any]:
        assert self._proc is not None and self._proc.stdout is not None
        # Tolerate a bounded number of non-protocol lines (e.g. a `conda run`
        # banner or stray library output) before a JSON protocol message.
        for _ in range(1000):
            line = self._proc.stdout.readline()
            if not line:
                raise BridgeWorkerError(f"worker exited unexpectedly: {self._stderr()}")
            stripped = line.strip()
            if not stripped or not stripped.startswith("{"):
                continue
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                continue
        raise BridgeWorkerError(f"no protocol message from worker: {self._stderr()}")

    def _stderr(self) -> str:
        return "".join(self._stderr_lines)[-4_000:]

    def _drain_stderr(self) -> None:
        """Continuously drain chatty upstream output so its pipe cannot deadlock."""

        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                self._stderr_lines.append(line)
                self._write_log(line)
        except (OSError, ValueError):
            return

    def _next_eval_index(self) -> int | None:
        """Next ``eval`` request of the worker (``None``: the worker sent ``done``)."""

        while True:
            message = self._read_message()
            kind = message.get("type")
            if kind == "done":
                self._done = True
                self._write_log(f"[bridge] worker done after {len(self._trace)} charged queries "
                                f"(budget {self.budget})")
                return None
            if kind == "log":
                self._write_log(f"[log] {message.get('message', '')}")
                continue
            if kind != "eval":
                raise BridgeWorkerError(f"unexpected worker message: {message}")
            index = int(message["index"])
            if not 0 <= index < len(self.candidates):
                raise BridgeWorkerError(f"worker proposed out-of-range index {index}")
            return index

    # ---- charged loop ---------------------------------------------------
    def run_charged(
        self,
        objective: Callable[[Config], Observation],
        budget: int | None = None,
        on_evaluation: Callable[[int, Config, Observation], None] | None = None,
    ) -> tuple[tuple[Config, Observation], ...]:
        """Serve the worker's proposals; EVERY proposal consumes one budget unit.

        A proposal of an already-evaluated candidate returns the cached observation
        (the objective is not called again) and is a query flagged ``repeat``.  If the
        worker stops before the budget is consumed the trace ends there;
        callers fail such a run (:func:`require_full_budget`).
        Returns the charged trace as ``(config, observation)`` pairs."""

        budget = int(self.budget if budget is None else budget)
        self._run_budget = budget
        if self._proc is None:
            self._launch()
        while len(self._trace) < budget:
            index = self._next_eval_index()
            if index is None:
                break
            config = dict(self.candidates[index])
            key = config_key(config)
            repeat = key in self._evaluated
            if repeat:
                observation = self._observation_cache[key]
            else:
                observation = objective(dict(config))
                self._record(config, observation)
                self._observation_cache[key] = deepcopy(observation)
            value = float(observation["objective"] if isinstance(observation, Mapping) else observation)
            self._trace.append({"config": config, "observation": observation, "repeat": repeat})
            if on_evaluation is not None:
                on_evaluation(len(self._trace), dict(config), observation)
            self._send({"value": value})
        return tuple((dict(row["config"]), deepcopy(row["observation"])) for row in self._trace)

    @property
    def trace(self) -> tuple[dict[str, Any], ...]:
        """Charged queries in order: ``{"config", "observation", "repeat"}``."""

        return tuple(dict(row) for row in self._trace)

    def bridge_accounting(self) -> dict[str, Any]:
        """Per-run bookkeeping persisted by the harness (metrics / run_config)."""

        used = len(self._trace)
        budget = int(getattr(self, "_run_budget", self.budget))
        n_repeats = sum(1 for row in self._trace if row["repeat"])
        exhausted = used < budget
        return {
            "bridge_policy": self.bridge_policy,
            "budget": budget,
            "queries_used": used,
            "n_unique": used - n_repeats,
            "n_repeats": n_repeats,
            # first query index with no worker proposal (None: the worker used the whole budget)
            "proposals_exhausted_at": (used + 1) if exhausted else None,
            "worker_log": None if self.worker_log_path is None else str(self.worker_log_path),
        }

    # ---- BaselineOptimizer contract ------------------------------------
    def warmstart(self, n_init: int) -> list[Config]:
        # The upstream worker owns its own initial design; the harness must not
        # inject warmstart points (they would double-count against the budget).
        return []

    def ask(self) -> Config:
        raise BridgeWorkerError(
            f"{type(self).__name__} is driven by run_charged; "
            "use baselines.runner.run_ask_tell_loop, which dispatches to it")

    def tell(self, config: Mapping[str, Any], observation: Observation) -> None:
        raise BridgeWorkerError(f"{type(self).__name__}: use run_charged")

    def close(self) -> None:
        if self._proc is None:
            self._close_log()
            return
        for stream in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            self._proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self._proc.kill()
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1)
        self._proc = None
        self._stderr_thread = None
        self._close_log()

    def _close_log(self) -> None:
        handle, self._log_handle = self._log_handle, None
        if handle is not None:
            with self._log_lock:
                try:
                    handle.close()
                except OSError:
                    pass

    def __del__(self) -> None:  # best-effort cleanup
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


def require_full_budget(used: int, budget: int, name: str) -> None:
    """A run whose worker stops before the budget is spent is a failed run (it is not padded)."""

    if int(used) < int(budget):
        raise BridgeWorkerError(f"{name} worker stopped after {int(used)} of {int(budget)} queries")


__all__ = [
    "SubprocessBridgeOptimizer",
    "BridgeWorkerError",
    "BRIDGE_POLICIES",
    "require_full_budget",
]
