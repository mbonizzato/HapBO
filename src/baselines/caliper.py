"""Single-objective adaptation of CALIPER (evidence-gated LLM prior layers).

Reimplemented from the paper (Chen & Ban, "CALIPER: Evidence-Gated LLM Prior Layers
for Multi-Objective Bayesian Optimization", arXiv:2606.01730, 2026); no public
reference implementation exists.  The multi-objective machinery collapses to a
single objective here, but the *defining* structure is preserved verbatim:

* a per-expert-role **reputation market** that learns online trust ``alpha_e``
  from observed feedback (softmax of a discounted capital account);
* a **3-armed counterfactual prior gate** ``{no_conf, conf, drop}`` that decides,
  from the marginal likelihood of each arm's residuals, how the cached prior
  enters the surrogate -- a DISCRETE evidence gate, in deliberate contrast to
  HapBO's continuous alpha-pi floor;
* a **residual GP**: the gated prior mean is subtracted before fitting and added
  back at prediction, so it shifts the posterior mean but not its variance.

The LLM is queried ONCE, offline, to produce the cached per-role prior means and
confidences (see ``src/experiments/caliper_elicitation.py`` / the study ``prior_path``);
this online worker makes no LLM calls during optimization.

Reductions vs the paper (summarized in experiments/catalog/methods/caliper.yaml):
* one objective => no hypervolume; ``qLogNEHVI`` becomes analytic EI over the
  finite candidate table; per-objective weights and the residual-UCB diagnostic
  acquisition are dropped;
* the cross-objective "market-trust" gate folds into the single update gate;
* the reputation market keeps its per-EXPERT-ROLE structure (>=2 roles), which is
  what makes the gate non-degenerate.

Default hyperparameters follow Table 1 of the CALIPER paper.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

try:  # main process imports the ``src.`` package; the isolated worker adds ``src/`` to sys.path
    from src.bayesian_optimization.acquisition import ExpectedImprovement
    from src.bayesian_optimization.surrogate import GaussianProcessSurrogate
except ImportError:  # pragma: no cover - worker-context fallback
    from bayesian_optimization.acquisition import ExpectedImprovement
    from bayesian_optimization.surrogate import GaussianProcessSurrogate

from .base import BaselineOptimizer, Config, Observation, config_key


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.asarray(values, dtype=float) - float(np.max(values))
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    return weights / total if total > 0 else np.full_like(weights, 1.0 / weights.size)


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance without a SciPy dependency."""

    a2 = np.sum(a * a, axis=1).reshape(-1, 1)
    b2 = np.sum(b * b, axis=1).reshape(1, -1)
    d2 = np.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0)
    return np.sqrt(d2)


def _matern52(distance: np.ndarray, lengthscale: float) -> np.ndarray:
    scaled = np.sqrt(5.0) * distance / max(lengthscale, 1e-9)
    return (1.0 + scaled + (scaled * scaled) / 3.0) * np.exp(-scaled)


def _fixed_kernel_log_marginal_likelihood(
    X: np.ndarray, residuals: np.ndarray, *, lengthscale: float, noise: float
) -> float:
    """Log marginal likelihood of ``residuals`` under a fixed Matern-5/2 GP.

    This is the App. F.3 evidence-scoring GP: unit signal variance, fixed noise,
    median-pairwise-distance lengthscale.  Ranking across the gate arms is what it
    is used for, so the shared scale is immaterial.
    """

    n = X.shape[0]
    K = _matern52(_euclidean_distance(X, X), lengthscale)
    K = K + (noise + 1e-8) * np.eye(n)
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        K = K + 1e-6 * np.eye(n)
        L = np.linalg.cholesky(K)
    r = residuals.reshape(-1, 1)
    solved = np.linalg.solve(L, r)
    alpha = np.linalg.solve(L.T, solved)
    log_det = 2.0 * float(np.sum(np.log(np.diag(L))))
    quad = float((r * alpha).sum())  # r,alpha are (n,1); avoid float() on a (1,1) array (NumPy 2)
    return -0.5 * quad - 0.5 * log_det - 0.5 * n * math.log(2.0 * math.pi)


class CALIPER(BaselineOptimizer):
    """Evidence-gated LLM-prior BO over a finite candidate table (single objective)."""

    name = "caliper"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.maximize = bool(self.objective_meta.get("maximize", True))
        if int(self.objective_meta.get("n_constraints", 0)):
            raise ValueError("CALIPER single-objective adaptation is unconstrained")
        if not self.maximize:
            raise ValueError("CALIPER single-objective adaptation assumes a maximization objective")

        self._mu, self._c, self._roles = self._load_prior(self.prior)
        n_roles = self._mu.shape[0]
        if n_roles < 1:
            raise ValueError("CALIPER requires at least one expert role in the cached prior")

        # -- reputation market (Table 1 of the CALIPER paper) ---------------
        self.reward_step = float(self.kwargs.get("reward_step", 0.45))
        self.discount = float(self.kwargs.get("discount", 0.015))
        self.softmax_temp = float(self.kwargs.get("softmax_temp", 0.55))
        self.reward_clip = (
            float(self.kwargs.get("reward_clip_low", -2.0)),
            float(self.kwargs.get("reward_clip_high", 0.5)),
        )
        # -- gates ----------------------------------------------------------
        self.prior_gate_eta = float(self.kwargs.get("prior_gate_eta", 1.0))
        self.min_updates = int(self.kwargs.get("min_updates", 4))
        self.count_scale = float(self.kwargs.get("count_scale", 4.0))
        self.delta_drop = float(self.kwargs.get("delta_drop", 0.05))
        self.gate_gp_noise = float(self.kwargs.get("gate_gp_noise", 0.05))
        self.update_gate_eta = float(self.kwargs.get("update_gate_eta", 1.0))
        # -- residual GP surrogate (shared study conventions) ---------------
        self.length_scale = self.kwargs.get("length_scale", 1.0)
        self.noise_variance = float(self.kwargs.get("noise_std", 0.05)) ** 2
        self.improvement_xi = float(self.kwargs.get("improvement_xi", 0.0))

        # -- state ----------------------------------------------------------
        self._K = np.zeros(n_roles, dtype=float)          # capital, init 0
        self._pi_u = np.array([0.5, 0.5], dtype=float)    # update-gate Hedge over {no_conf, conf}
        self._X = np.asarray(self.encoded_candidates, dtype=float)
        self._obs_idx: list[int] = []
        self._obs_y: list[float] = []
        self._asked_indices: dict[str, int] = {}
        self.records: list[dict[str, Any]] = []

    # -- artifact -----------------------------------------------------------

    def _load_prior(self, prior: Any) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Align the cached multi-role artifact to this run's candidate order."""

        if not isinstance(prior, Mapping):
            raise ValueError("CALIPER requires a cached multi-role prior artifact (dict)")
        roles = list(prior["expert_roles"])
        keys = list(prior["candidate_keys"])
        mu = np.asarray(prior["mu"], dtype=float)
        conf = np.asarray(prior["c"], dtype=float)
        if mu.shape != (len(roles), len(keys)) or conf.shape != mu.shape:
            raise ValueError("CALIPER artifact mu/c must be [n_roles, n_candidates]")
        # Map artifact columns to THIS run's candidate order by config_key (robust
        # to row reordering; fail-closed if a candidate is missing).
        col_of = {str(key): index for index, key in enumerate(keys)}
        order = np.empty(len(self.candidates), dtype=int)
        for row, candidate in enumerate(self.candidates):
            key = config_key(candidate)
            if key not in col_of:
                raise ValueError("CALIPER artifact is missing a candidate; re-elicit the prior")
            order[row] = col_of[key]
        mu = np.clip(mu[:, order], 0.0, 1.0)
        conf = np.clip(conf[:, order], 1e-4, 1.0 - 1e-4)
        return mu, conf, roles

    # -- reputation market & gate ------------------------------------------

    def _alpha(self) -> np.ndarray:
        """Per-role trust weights: softmax of discounted capital."""

        return _softmax(self._K / self.softmax_temp)

    def _prior_arm(self, arm: str, alpha: np.ndarray) -> np.ndarray:
        """Return the arm's prior mean over ALL candidates, in [0, 1]."""

        if arm == "drop":
            return np.zeros(self._mu.shape[1], dtype=float)
        if arm == "no_conf":
            weight = alpha[:, None]                       # confidence == 1
        else:  # "conf"
            weight = alpha[:, None] * self._c             # confidence-weighted
        numerator = np.sum(weight * self._mu, axis=0)
        denominator = np.sum(weight, axis=0)
        return numerator / np.maximum(denominator, 1e-12)

    def _running_scale(self) -> float:
        if len(self._obs_y) < 2:
            return 1.0
        scale = float(np.std(np.asarray(self._obs_y, dtype=float), ddof=1))
        return scale if scale > 1e-9 else 1.0

    def _evidence_score(self, arm_prior: np.ndarray) -> float:
        idx = np.asarray(self._obs_idx, dtype=int)
        X = self._X[idx]
        residuals = np.asarray(self._obs_y, dtype=float) - arm_prior[idx]
        distances = _euclidean_distance(X, X)
        upper = distances[np.triu_indices_from(distances, k=1)]
        lengthscale = float(np.median(upper)) if upper.size else 1.0
        if not math.isfinite(lengthscale) or lengthscale <= 1e-9:
            lengthscale = 1.0
        log_ml = _fixed_kernel_log_marginal_likelihood(
            X, residuals, lengthscale=lengthscale, noise=self.gate_gp_noise
        )
        return log_ml / max(len(idx), 1)

    def _gate_probabilities(self, alpha: np.ndarray) -> np.ndarray:
        """Return pi_p over {no_conf, conf, drop} with finite-data shrinkage."""

        n = len(self._obs_idx)
        e_no_conf = np.array([1.0, 0.0, 0.0])
        if n < self.min_updates:
            return e_no_conf
        arms = {
            "no_conf": self._prior_arm("no_conf", alpha),
            "conf": self._prior_arm("conf", alpha),
            "drop": self._prior_arm("drop", alpha),
        }
        scores = {name: self._evidence_score(prior) for name, prior in arms.items()}
        deltas = {"no_conf": 0.0, "conf": 0.0, "drop": self.delta_drop}
        logits = np.array(
            [
                self.prior_gate_eta * (scores[name] - scores["no_conf"] - deltas[name])
                for name in ("no_conf", "conf", "drop")
            ]
        )
        kappa = math.sqrt(n / (n + self.count_scale))
        return (1.0 - kappa) * e_no_conf + kappa * _softmax(logits)

    def _gated_prior(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (mixed prior over all candidates, gate probs, trust weights)."""

        alpha = self._alpha()
        pi_p = self._gate_probabilities(alpha)
        mixed = pi_p[0] * self._prior_arm("no_conf", alpha) + pi_p[1] * self._prior_arm(
            "conf", alpha
        )  # drop contributes zero
        return mixed, pi_p, alpha

    # -- ask / tell ---------------------------------------------------------

    def _reserved(self) -> set[str]:
        return set(self._evaluated) | set(self._pending)

    def ask(self) -> Config:
        if len(self._history) + len(self._pending) >= self.budget:
            raise RuntimeError("the baseline evaluation budget is exhausted")
        if not self._obs_idx:
            config = self._random_unqueried_config()
            self._asked_indices[config_key(config)] = self._index_for(config)
            return config

        prior, pi_p, alpha = self._gated_prior()
        observed = np.asarray(self._obs_idx, dtype=int)
        residual_targets = np.asarray(self._obs_y, dtype=float) - prior[observed]

        surrogate = GaussianProcessSurrogate(
            length_scale=self.length_scale,
            noise_variance=self.noise_variance,
            normalize_y=True,
        )
        surrogate.fit(self._X[observed], residual_targets)
        residual_mean, residual_std = surrogate.predict(self._X)
        # Reconstruct the objective posterior: mean shifted by the prior, variance
        # unchanged (residual std is the objective std under a mean shift).
        reconstructed_mean = residual_mean + prior

        reserved = self._reserved()
        excluded = np.array(
            [config_key(candidate) in reserved for candidate in self.candidates], dtype=bool
        )
        best_value = float(np.max(self._obs_y))
        scores = ExpectedImprovement(xi=self.improvement_xi).score(
            reconstructed_mean, residual_std, best_value=best_value, excluded_mask=excluded
        )
        if not np.any(np.isfinite(scores)):
            config = self._random_unqueried_config()
            self._asked_indices[config_key(config)] = self._index_for(config)
            self._record_query(self._index_for(config), pi_p, alpha)
            return config

        best = np.flatnonzero(scores == np.max(scores))
        index = int(self._rng.choice(best))
        config = dict(self.candidates[index])
        key = config_key(config)
        self._pending.add(key)
        self._asked_indices[key] = index
        self._record_query(index, pi_p, alpha)
        return config

    def tell(self, config: Mapping[str, Any], observation: Observation) -> None:
        key = config_key(config)
        index = self._asked_indices.pop(key, self._index_for(config))
        value = (
            float(observation["objective"])
            if isinstance(observation, Mapping)
            else float(observation)
        )
        self._record(config, observation)

        # Read the update gate BEFORE folding in the new observation.
        rho = float(self._pi_u[1])
        scale = self._running_scale()
        alpha = self._alpha()

        # Update gate: 2-armed Hedge over {no_conf, conf}.
        loss_no_conf = abs(value - self._prior_arm("no_conf", alpha)[index]) / scale
        loss_conf = abs(value - self._prior_arm("conf", alpha)[index]) / scale
        centered = np.array([loss_no_conf, loss_conf]) - 0.5 * (loss_no_conf + loss_conf)
        self._pi_u = self._pi_u * np.exp(-self.update_gate_eta * centered)
        total = float(np.sum(self._pi_u))
        self._pi_u = self._pi_u / total if total > 0 else np.array([0.5, 0.5])

        # Reputation market: per expert role.
        low, high = self.reward_clip
        for role in range(self._mu.shape[0]):
            epsilon = abs(self._mu[role, index] - value) / scale
            reward = float(np.clip(0.5 - 0.5 * epsilon * epsilon, low, high))
            effective_conf = 1.0 + rho * (self._c[role, index] - 1.0)
            self._K[role] = (1.0 - self.discount) * self._K[role] + (
                self.reward_step * effective_conf * reward
            )

        self._obs_idx.append(index)
        self._obs_y.append(value)

    def _record_query(self, index: int, pi_p: np.ndarray, alpha: np.ndarray) -> None:
        self.records.append(
            {
                "query": len(self._history) + 1,
                "candidate_index": int(index),
                "trust_weights": [float(a) for a in alpha],
                "gate_probs": {
                    "no_conf": float(pi_p[0]),
                    "conf": float(pi_p[1]),
                    "drop": float(pi_p[2]),
                },
                "update_gate_rho": float(self._pi_u[1]),
                "capital": [float(k) for k in self._K],
                "roles": list(self._roles),
            }
        )


__all__ = ["CALIPER"]
