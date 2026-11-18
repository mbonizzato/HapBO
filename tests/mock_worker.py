"""A trivial worker used by tests/test_bridge.py to test the subprocess bridge end-to-end.

It runs random search over the finite candidate set and deliberately re-requests
the best-so-far candidate sometimes, exercising the bridge's duplicate-serving
(cache) path. Runnable as a standalone script (no package install needed).
"""

from __future__ import annotations

import os
import sys

# Import the protocol helpers directly (avoid triggering baselines/__init__.py,
# which pulls in the heavy repo-side deps a worker's env need not have).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src", "baselines", "workers"))

import numpy as np  # noqa: E402

from harness import finish, log, make_evaluator, read_spec  # noqa: E402


def main() -> None:
    spec = read_spec()
    n = len(spec["candidates"])
    budget = int(spec["budget"])
    maximize = bool(spec.get("maximize", True))
    rng = np.random.default_rng(spec.get("seed"))
    evaluate = make_evaluator()

    log(f"mock worker: {n} candidates, budget {budget}, maximize={maximize}")
    seen: dict[int, float] = {}
    order = [int(i) for i in rng.permutation(n)]
    position = 0
    while len(seen) < budget and position < len(order):
        index = order[position]
        position += 1
        seen[index] = evaluate(index)
        # Re-request the incumbent sometimes -> must be served from cache.
        if len(seen) >= 2 and rng.random() < 0.3:
            best = (max if maximize else min)(seen, key=lambda k: seen[k])
            evaluate(best)
    finish()


if __name__ == "__main__":
    main()
