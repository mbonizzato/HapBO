"""Box hypotheses for HypBO (BORA supplement C.3), shared by the parent and the HypBO worker.

An LLM hypothesis is a predicted optimum location ``c`` with a confidence score ``s`` in [0, 1].
BORA draws a hyper-rectangle around ``c`` with side length ``L = (ub - lb) * (1 - s)``, where
``lb`` and ``ub`` bound the input space; the box is clipped to those bounds.  The side carries the
``1e-6`` floor of BORA's ColaBO construction (supplement C.2, ``sigma = 1 - s + 1e-6``), so a
hypothesis with ``s = 1`` is a box of side ``1e-6 * (ub - lb)``, not a point.  HypBO receives the
box in the form its synthetic-function experiments use (``utils.get_good_hypothesis``): the
hypothesis bounds are the box and the inequality rows ``x_j <= hi_j``, ``-x_j <= -lo_j``.
"""

from __future__ import annotations

import numpy as np

#: Floor on the relative box side (BORA supplement C.2 epsilon).
EPSILON = 1e-6
#: Boxes whose largest side (encoded units) is below this count as narrow in select_hypotheses.
NARROW_SIDE = 1e-4


def hypothesis_box(center, score, lower, upper):
    """Return ``(lo, hi)``: the box of side ``(upper - lower) * (1 - score + EPSILON)`` centred at
    ``center``, clipped to ``[lower, upper]``."""

    center = np.asarray(center, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    score = float(score)
    if not (center.shape == lower.shape == upper.shape):
        raise ValueError("center and bounds must have the same dimension")
    if not np.all(np.isfinite(center)) or not np.all(upper >= lower):
        raise ValueError("bounds must be finite with upper >= lower")
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"confidence score must be in [0, 1], got {score}")
    if np.any(center < lower) or np.any(center > upper):
        raise ValueError("hypothesis centre lies outside the input space")
    half = 0.5 * (upper - lower) * (1.0 - score + EPSILON)
    return np.clip(center - half, lower, upper), np.clip(center + half, lower, upper)


def select_hypotheses(centers, lows, highs):
    """Return ``(kept, dropped)``: the indices of the boxes passed to HypBO, and a mapping from
    every dropped box to the kept box it duplicates.

    A later box is dropped only when it and an already kept box with the same centre are both
    narrow (largest side below ``NARROW_SIDE``).  HypBO redraws a hypothesis's initial sample until
    it differs (``np.allclose``) from the samples already drawn, and all its local models share the
    run seed, so two same-centre boxes narrower than that tolerance never yield a sample and the
    run stops progressing.  Every other box is passed as given."""

    kept, dropped, narrow_kept = [], {}, {}
    for index, (center, low, high) in enumerate(zip(centers, lows, highs)):
        key = tuple(float(v) for v in np.asarray(center, dtype=float).reshape(-1))
        side = float(np.max(np.asarray(high, dtype=float) - np.asarray(low, dtype=float)))
        narrow = side < NARROW_SIDE
        if narrow and key in narrow_kept:
            dropped[index] = narrow_kept[key]
            continue
        kept.append(index)
        if narrow:
            narrow_kept[key] = index
    return kept, dropped


def box_inequalities(lo, hi):
    """Inequality rows of a box in upstream's form: ``A x <= b`` with rows ``e_j`` and
    ``-e_j`` interleaved per dimension and ``b = [hi_0, -lo_0, hi_1, -lo_1, ...]``."""

    lo = np.asarray(lo, dtype=float).reshape(-1)
    hi = np.asarray(hi, dtype=float).reshape(-1)
    if lo.shape != hi.shape or np.any(hi < lo):
        raise ValueError("box bounds must have equal shapes and hi >= lo")
    dim = lo.size
    rows = []
    const = []
    for j in range(dim):
        upper_row = [0] * dim
        upper_row[j] = 1
        lower_row = [0] * dim
        lower_row[j] = -1
        rows.extend([upper_row, lower_row])
        const.extend([hi[j], -lo[j]])
    return np.array(rows), np.array(const)


__all__ = ["EPSILON", "NARROW_SIDE", "box_inequalities", "hypothesis_box", "select_hypotheses"]
