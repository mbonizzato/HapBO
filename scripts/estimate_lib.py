#!/usr/bin/env python3
"""Analytic test functions and fidelity targets of the synthetic panel.

``FAMS`` maps each family to ``(dim, lo, hi, f, f_opt, x_opt)``: the vectorized textbook function (native
coordinates, minimized) passed through ``synthetic_truth.apply``, so every importer gets the hidden objective
selected by ``SYNTH_TRUTH`` (default ``well``). ``RHOS`` maps the formula card (rho90) to its target Pearson
correlation with the test function. The perturbed card models themselves are built by ``formula_estimate_lib.py``.
"""
from __future__ import annotations

import numpy as np

# family -> (dim, lo, hi, f, f_opt, x_opt)  -- same defs as colabo_llm_continuous.py
_HA = np.array([1.0, 1.2, 3.0, 3.2])
_HM = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
_HP = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991],
                       [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]])


def _hartmann6(X):
    X = np.atleast_2d(np.asarray(X, float))
    inner = np.sum(_HM[None, :, :] * (X[:, None, :] - _HP[None, :, :]) ** 2, axis=2)
    return -np.sum(_HA[None, :] * np.exp(-inner), axis=1)


def _styblinski(X):
    X = np.atleast_2d(np.asarray(X, float))
    return 0.5 * np.sum(X ** 4 - 16.0 * X ** 2 + 5.0 * X, axis=1)


def _levy(X):
    X = np.atleast_2d(np.asarray(X, float))
    w = 1.0 + (X - 1.0) / 4.0
    t1 = np.sin(np.pi * w[:, 0]) ** 2
    t3 = (w[:, -1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[:, -1]) ** 2)
    mid = np.sum((w[:, :-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:, :-1] + 1.0) ** 2),
                 axis=1)
    return t1 + mid + t3


def _ackley(X):
    X = np.atleast_2d(np.asarray(X, float))
    d = X.shape[1]
    return (-20.0 * np.exp(-0.2 * np.sqrt(np.sum(X ** 2, axis=1) / d))
            - np.exp(np.sum(np.cos(2.0 * np.pi * X), axis=1) / d) + 20.0 + np.e)


FAMS = {
    "hartmann6": (6, 0.0, 1.0, _hartmann6, -3.322368,
                  np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])),
    "styblinski_tang7": (7, -5.0, 5.0, _styblinski, -39.16616570 * 7, np.full(7, -2.903534)),
    "levy10": (10, -10.0, 10.0, _levy, 0.0, np.ones(10)),
    "ackley15": (15, -32.768, 32.768, _ackley, 0.0, np.zeros(15)),
}
# Hidden objective of the synthetic panel (scripts/synthetic_truth.py; env SYNTH_TRUTH, default well)
FAMS = __import__('synthetic_truth').apply(FAMS)
RHOS = {"rho90": 0.9}
