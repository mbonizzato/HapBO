"""Worker-side protocol helpers for the subprocess baseline bridge.

Protocol (line-delimited JSON):
  parent -> worker : one problem-spec line, then one {"value": y} line per eval
  worker -> parent : {"type":"eval","index":i} | {"type":"log",...} | {"type":"done"}

The worker's real stdout (protocol channel) is captured at import so upstream
libraries that print to stdout cannot corrupt the protocol; use :func:`log` for
diagnostics (they are forwarded to the parent and ignored).
"""

from __future__ import annotations

import json
import sys

# Capture the true protocol stream, then redirect Python-level stdout to stderr so
# chatty upstream libraries can't corrupt the JSON channel.
_PROTOCOL_OUT = sys.stdout
sys.stdout = sys.stderr


def read_spec() -> dict:
    line = sys.stdin.readline()
    if not line:
        raise SystemExit("baseline worker: no problem spec on stdin")
    return json.loads(line)


def _emit(obj: dict) -> None:
    _PROTOCOL_OUT.write(json.dumps(obj) + "\n")
    _PROTOCOL_OUT.flush()


def make_evaluator():
    """Return ``evaluate(index) -> float`` that round-trips one objective value."""

    def evaluate(index: int) -> float:
        _emit({"type": "eval", "index": int(index)})
        line = sys.stdin.readline()
        if not line:
            raise SystemExit("baseline worker: parent closed the pipe")
        return float(json.loads(line)["value"])

    return evaluate


def log(message: object) -> None:
    _emit({"type": "log", "message": str(message)})


def finish() -> None:
    _emit({"type": "done"})


__all__ = ["read_spec", "make_evaluator", "log", "finish"]
