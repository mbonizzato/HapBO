"""The complete ask/tell loop for discrete Bayesian optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .acquisition import (
    ExpectedImprovement,
    PriorShapedUCB,
    ProbabilityOfImprovement,
    normalize_prior,
    pi_bo_prior,
    pi_bo_shape,
    probability_of_feasibility,
)
from .candidate_sampler import CandidateSampler
from .surrogate import DEFAULT_KERNEL, GaussianProcessSurrogate, validate_kernel


PRIOR_SHAPING_MODES = ("alpha-pi", "pi-bo", "none")
# Separates piBO's initial-design draw from every other use of the random seed.
PI_BO_INITIAL_DESIGN_STREAM = 0x50494230


def pi_bo_initial_design(
    prior: np.ndarray, n_initial: int, random_state: Optional[int] = None
) -> tuple[int, ...]:
    """Initial design of piBO (Hvarfner et al. 2022, Sec. 3.2 and Algorithm 1) on a finite candidate set.

    The first point is the mode of the piBO prior :func:`~.acquisition.pi_bo_prior`
    (the highest-prior candidate; the lowest index among ties).  The other
    ``n_initial - 1`` points are drawn from the remaining candidates with probability
    proportional to the prior (successive draws without replacement), from a
    generator seeded by ``random_state`` alone.  A constant prior has no mode: all
    ``n_initial`` points are then drawn.
    """
    weights = pi_bo_prior(prior)
    count = int(n_initial)
    if not 1 <= count <= weights.size:
        raise ValueError("n_initial must be between 1 and the number of candidates")
    candidates = np.arange(weights.size)
    design: list[int] = []
    if float(np.ptp(weights)) > 0.0:
        mode = int(np.argmax(weights))
        design.append(mode)
        candidates = np.delete(candidates, mode)
    if count > len(design):
        seed = None if random_state is None else [int(random_state), PI_BO_INITIAL_DESIGN_STREAM]
        rng = np.random.default_rng(seed)
        w = weights[candidates]
        design += [int(i) for i in rng.choice(candidates, size=count - len(design), replace=False, p=w / w.sum())]
    return tuple(design)


@dataclass(frozen=True)
class OptimizationResult:
    """Immutable summary returned after an optimization run."""

    best_index: int
    best_x: np.ndarray
    best_y: float
    evaluated_indices: np.ndarray
    X: np.ndarray
    y: np.ndarray
    constraint_values: np.ndarray


class BayesianOptimization:
    """Bayesian optimization over a fixed, finite candidate set.

    Parameters
    ----------
    candidates:
        Array of shape ``(n_candidates, n_features)``.
    prior:
        Optional numeric preference with one value per candidate.  This class
        does not create the prior and does not depend on the prior compiler (``src/prior/``).
    alpha:
        Prior influence of the alpha-pi rule, in [0, 0.9]; zero recovers the
        unshaped acquisition.
    n_initial:
        Number of initialization evaluations, drawn at random from the unqueried
        candidates with the ``random_state`` generator, so a prior arm and a no-prior
        arm with the same seed share their initial design (HapBO as in the paper).
    initial_indices:
        Optional explicit initial design (candidate indices), queried first in order
        (the prior-warm-start and prior-greedy arms of the study worker).
    prior_shaping, pi_bo_beta:
        ``"alpha-pi"`` (the default) shapes the acquisition with the bounded
        alpha-pi rule.  ``"pi-bo"`` is piBO (Hvarfner et al. 2022, Sec. 3.2 and
        Algorithm 1): the ``n_initial`` initial points are the mode of the piBO
        prior followed by points sampled from it (:func:`pi_bo_initial_design`,
        seeded by ``random_state``), and every later acquisition is weighted by
        the prior raised to ``pi_bo_beta / t`` (:func:`~.acquisition.pi_bo_shape`),
        with ``t = 1`` at the first query after the initial design;
        ``initial_indices`` is not accepted; ``pi_bo_beta`` is required (Hvarfner et
        al. use ``N / 10`` for ``N`` BO iterations, as the study worker does).
        ``"none"`` leaves the acquisition unshaped.
    maximize:
        If false, observed objectives are negated internally so the acquisition
        function can still be maximized.
    acquisition:
        ``"expected-improvement"`` (the default) uses EI. ``"ucb"`` uses the
        upper confidence bound. Either acquisition can be alpha-pi shaped by
        an optional prior.
    candidate_gate:
        Optional ``n_observations -> allowed mask`` (or ``None`` = all allowed).  When
        set, random initialization queries are drawn from the allowed, unqueried
        candidates and the acquisition argmax is taken over them only (LLM-EI's
        percentile indicator; disallowed candidates are pruned, not merely scored
        zero).  If no allowed candidate is left unqueried the gate is ignored for
        that query and the fallback is recorded in ``gate_fallbacks_``.  ``None``
        (the default) allows every candidate.
    n_constraints:
        Number of black-box constraints returned through :meth:`tell`.
        Feasibility is defined as every constraint value being at most zero;
        the acquisition is multiplied by their joint modeled probability of
        feasibility.
    kernel:
        GP covariance of every surrogate the optimizer builds (objective unless
        ``surrogate`` is given, constraints, crash model): ``"matern52"`` (the
        default), ``"rbf"`` or ``"matern12"``.  The Kriging-Believer batch path
        refits the objective surrogate, so it uses the same kernel.
    """

    def __init__(
        self,
        candidates: np.ndarray,
        *,
        length_scale: float = 1.0,
        kappa: float = 2.0,
        alpha: float = 0.0,
        prior: Optional[np.ndarray] = None,
        n_initial: int = 1,
        noise_variance: float = 1e-6,
        signal_variance: float = 1.0,
        normalize_y: bool = False,
        maximize: bool = True,
        random_state: Optional[int] = None,
        surrogate: Optional[GaussianProcessSurrogate] = None,
        acquisition: str = "expected-improvement",
        improvement_xi: float = 0.0,
        n_constraints: int = 0,
        prior_mean: Optional[Callable[[np.ndarray], np.ndarray | float]] = None,
        initial_indices: Optional[Sequence[int]] = None,
        prior_shaping: str = "alpha-pi",
        pi_bo_beta: Optional[float] = None,
        model_hidden_failures: bool = False,
        ucb_center_incumbent: bool = False,
        candidate_gate: Optional[Callable[[int], Optional[np.ndarray]]] = None,
        kernel: str = DEFAULT_KERNEL,
    ) -> None:
        self.kernel = validate_kernel(kernel)
        self.sampler = CandidateSampler(candidates, random_state=random_state)
        self.candidates = self.sampler.candidates
        self.n_candidates = self.candidates.shape[0]
        if n_constraints < 0:
            raise ValueError("n_constraints must be non-negative")
        if prior_shaping not in PRIOR_SHAPING_MODES:
            raise ValueError(f"prior_shaping must be one of {PRIOR_SHAPING_MODES}")

        n_initial = int(n_initial)
        if not 1 <= n_initial <= self.n_candidates:
            raise ValueError("n_initial must be between 1 and the number of candidates")

        if prior is not None:
            prior = np.asarray(prior, dtype=float).reshape(-1)
            if prior.size != self.n_candidates:
                raise ValueError("prior must have one value per candidate")
            # Validate now; retain the original scale for user inspection.
            normalize_prior(prior)
            self.prior = prior.copy()
        else:
            self.prior = None

        if initial_indices is None:
            self.initial_indices: Optional[tuple[int, ...]] = None
        else:
            resolved_indices = [int(i) for i in initial_indices]
            if any(not 0 <= i < self.n_candidates for i in resolved_indices):
                raise ValueError("initial_indices must be valid candidate indices")
            if len(set(resolved_indices)) != len(resolved_indices):
                raise ValueError("initial_indices must be distinct")
            if len(resolved_indices) > self.n_candidates:
                raise ValueError("initial_indices cannot exceed the number of candidates")
            self.initial_indices = tuple(resolved_indices)
        if prior_shaping == "pi-bo":
            if pi_bo_beta is None or not float(pi_bo_beta) >= 0.0:
                raise ValueError("pi-bo needs pi_bo_beta >= 0 (Hvarfner et al. 2022: N / 10 for N BO iterations)")
            if self.initial_indices is not None:
                raise ValueError(
                    "pi-bo builds its initial design from the prior; initial_indices is not accepted"
                )
            if self.prior is not None:
                self.initial_indices = pi_bo_initial_design(self.prior, n_initial, random_state)

        self.n_initial = n_initial
        self.maximize = bool(maximize)
        self.prior_shaping = prior_shaping
        self.pi_bo_beta = None if pi_bo_beta is None else float(pi_bo_beta)
        self.prior_mean = prior_mean
        # The objective, constraint and crash surrogates share the kernel, so a
        # study's arms differ only in how (or whether) the prior enters.
        # A proper GP prior mean must fit residuals in standardized space.
        main_normalize_y = bool(normalize_y) or (prior_mean is not None)
        self.surrogate = surrogate or GaussianProcessSurrogate(
            length_scale=length_scale,
            signal_variance=signal_variance,
            noise_variance=noise_variance,
            normalize_y=main_normalize_y,
            prior_mean=prior_mean,
            kernel=self.kernel,
        )
        self.n_constraints = int(n_constraints)
        self.constraint_surrogates = [
            GaussianProcessSurrogate(
                length_scale=length_scale,
                signal_variance=signal_variance,
                noise_variance=noise_variance,
                normalize_y=True,
                kernel=self.kernel,
            )
            for _ in range(self.n_constraints)
        ]
        if acquisition == "ucb":
            self.acquisition = PriorShapedUCB(
                kappa=float(kappa),
                alpha=alpha,
                ucb_center_incumbent=bool(ucb_center_incumbent),
            )
        elif acquisition == "expected-improvement":
            self.acquisition = ExpectedImprovement(
                xi=improvement_xi,
                alpha=alpha,
            )
        elif acquisition == "probability-of-improvement":
            self.acquisition = ProbabilityOfImprovement(
                xi=improvement_xi,
                alpha=alpha,
            )
        else:
            raise ValueError(
                "acquisition must be 'ucb', 'expected-improvement', or "
                "'probability-of-improvement'"
            )
        self.acquisition_name = acquisition

        self.queried_mask = np.zeros(self.n_candidates, dtype=bool)
        self.evaluated_indices: list[int] = []
        self.observed_values: list[float] = []
        self.observed_constraints: list[np.ndarray] = []
        # Hidden-failure ("crash") modeling for live simulators that refuse some
        # inputs (SOLAR 1e20 sentinels). When enabled, crashed observations are
        # EXCLUDED from the objective/constraint GPs (a 1e20 sentinel flattens a
        # standardized GP into a binary detector) and a dedicated crash surrogate
        # supplies P(simulates) as an extra feasibility factor — the standard
        # hidden-constraint treatment. Default off.
        self.model_hidden_failures = bool(model_hidden_failures)
        self.observed_crashes: list[bool] = []
        self.crash_surrogate = (
            GaussianProcessSurrogate(
                length_scale=length_scale,
                signal_variance=signal_variance,
                noise_variance=noise_variance,
                normalize_y=True,
                kernel=self.kernel,
            )
            if self.model_hidden_failures
            else None
        )
        self.candidate_gate = candidate_gate
        self.gate_fallbacks_: list[int] = []
        self.last_scores_: Optional[np.ndarray] = None
        self.last_feasibility_probability_: Optional[np.ndarray] = None
        # Posterior mean over all candidates from the most recent surrogate-driven
        # ``ask`` (``None`` when no surrogate was fit -- initial design / all-crashed).
        # Consumed by the Kriging-Believer batch proposal to fantasize
        # a picked candidate's outcome before choosing the next member of a batch.
        self.last_mean_: Optional[np.ndarray] = None

    @property
    def n_observations(self) -> int:
        return len(self.observed_values)

    def _training_values(self) -> np.ndarray:
        values = np.asarray(self.observed_values, dtype=float)
        return values if self.maximize else -values

    def _feasible_observations(self) -> np.ndarray:
        if self.n_constraints == 0:
            feasible = np.ones(self.n_observations, dtype=bool)
        else:
            constraints = np.asarray(self.observed_constraints, dtype=float)
            feasible = np.all(constraints <= 0.0, axis=1)
        if self.model_hidden_failures and self.observed_crashes:
            feasible &= ~np.asarray(self.observed_crashes, dtype=bool)
        return feasible

    def _simulated_observations(self) -> np.ndarray:
        """Mask of observations that actually simulated (non-crashed)."""

        if not self.model_hidden_failures or not self.observed_crashes:
            return np.ones(self.n_observations, dtype=bool)
        return ~np.asarray(self.observed_crashes, dtype=bool)

    def _gate_allowed(self) -> Optional[np.ndarray]:
        """Allowed mask from ``candidate_gate`` for the next query (None = no gating)."""
        if self.candidate_gate is None:
            return None
        allowed = self.candidate_gate(self.n_observations)
        if allowed is None:
            return None
        allowed = np.asarray(allowed, dtype=bool).reshape(-1)
        if allowed.size != self.n_candidates:
            raise ValueError("candidate_gate must return one entry per candidate")
        if not np.any(allowed & ~self.queried_mask):
            self.gate_fallbacks_.append(self.n_observations)
            return None
        return allowed

    def ask(self) -> tuple[int, np.ndarray]:
        """Choose the next candidate and return ``(index, candidate)``."""
        if self.n_observations >= self.n_candidates:
            raise RuntimeError("All candidates have already been evaluated")
        # Reset the fantasy hook; only a surrogate-driven ask repopulates it.
        self.last_mean_ = None

        # An explicit initial design (piBO's initial design, or the highest-prior
        # design of the prior-warm-start and prior-greedy arms) takes precedence
        # over random init.
        if (
            self.initial_indices is not None
            and self.n_observations < len(self.initial_indices)
        ):
            index = int(self.initial_indices[self.n_observations])
            if self.queried_mask[index]:
                index = self.sampler.random(self.queried_mask)
            self.last_scores_ = None
            self.last_feasibility_probability_ = None
            return index, self.candidates[index].copy()

        # Random initial design from the seeded generator: a prior arm and a no-prior
        # arm with the same seed share it.
        if self.n_observations < self.n_initial:
            allowed = self._gate_allowed()
            excluded = self.queried_mask if allowed is None else (self.queried_mask | ~allowed)
            index = self.sampler.random(excluded)
            self.last_scores_ = None
            self.last_feasibility_probability_ = None
            return index, self.candidates[index].copy()

        observed_X = self.candidates[np.asarray(self.evaluated_indices, dtype=int)]
        # Hidden-failure gating: crashed observations carry sentinel values, not
        # measurements — fit the objective (and constraint) GPs on simulated
        # observations only. Without hidden-failure modeling every observation
        # counts as simulated.
        simulated = self._simulated_observations()
        # The acquisition only performs alpha-pi shaping; piBO/none inject the
        # prior elsewhere (piBO's decay below, the GP prior mean, or the initial design).
        acquisition_prior = self.prior if self.prior_shaping == "alpha-pi" else None
        if simulated.any():
            self.surrogate.fit(observed_X[simulated], self._training_values()[simulated])
            mean, std = self.surrogate.predict(self.candidates)
            self.last_mean_ = mean
            training_values = self._training_values()
            feasible = self._feasible_observations() & simulated
            incumbent_values = training_values[feasible]
            if incumbent_values.size == 0:
                incumbent_values = training_values[simulated]
            best_value = float(np.max(incumbent_values))
            if isinstance(self.acquisition, (ExpectedImprovement, ProbabilityOfImprovement)):
                scores = self.acquisition.score(
                    mean,
                    std,
                    best_value=best_value,
                    prior=acquisition_prior,
                    excluded_mask=self.queried_mask,
                )
            else:  # PriorShapedUCB -- best_value used only when ucb_center_incumbent
                scores = self.acquisition.score(
                    mean,
                    std,
                    prior=acquisition_prior,
                    excluded_mask=self.queried_mask,
                    best_value=best_value,
                )
        else:
            # Every observation so far crashed: no objective information exists.
            # Fall back to the alpha-pi multiplier over a flat base score (the
            # crash-probability factor below still steers toward simulable inputs).
            scores = np.ones(self.n_candidates, dtype=float)
            if acquisition_prior is not None and self.acquisition.alpha > 0:
                alpha = float(self.acquisition.alpha)
                scores = (1.0 - alpha) + alpha * normalize_prior(acquisition_prior)
        if self.prior is not None and self.prior_shaping == "pi-bo":
            # piBO's decay counter (Hvarfner et al. 2022, Algorithm 1): t = 1 at the
            # first query after the n_initial-point initial design.
            scores = pi_bo_shape(
                scores,
                self.prior,
                self.pi_bo_beta,
                self.n_observations - self.n_initial + 1,
                valid_mask=~self.queried_mask,
            )
        if self.n_constraints and simulated.any():
            observed_constraints = np.asarray(self.observed_constraints, dtype=float)
            feasibility_probability = np.ones(self.n_candidates, dtype=float)
            for column, constraint_surrogate in enumerate(self.constraint_surrogates):
                constraint_surrogate.fit(
                    observed_X[simulated], observed_constraints[simulated, column]
                )
                constraint_mean, constraint_std = constraint_surrogate.predict(
                    self.candidates
                )
                feasibility_probability *= probability_of_feasibility(
                    constraint_mean,
                    constraint_std,
                )
            valid = ~self.queried_mask
            scores[valid] *= feasibility_probability[valid]
            scores[~valid] = -np.inf
            self.last_feasibility_probability_ = feasibility_probability
        else:
            self.last_feasibility_probability_ = None
        if self.crash_surrogate is not None and self.observed_crashes:
            crash_labels = np.where(
                np.asarray(self.observed_crashes, dtype=bool), 1.0, -1.0
            )
            # Fit only with both classes present (a constant-label GP carries no
            # gradient); P(label <= 0) = probability the input actually simulates.
            if crash_labels.min() < 0.0 < crash_labels.max():
                self.crash_surrogate.fit(observed_X, crash_labels)
                crash_mean, crash_std = self.crash_surrogate.predict(self.candidates)
                simulate_probability = probability_of_feasibility(crash_mean, crash_std)
                valid = ~self.queried_mask
                scores[valid] *= simulate_probability[valid]
                scores[~valid] = -np.inf
        allowed = self._gate_allowed()
        excluded = self.queried_mask
        if allowed is not None:
            scores = np.where(allowed, scores, -np.inf)
            excluded = self.queried_mask | ~allowed  # keeps best()'s random fallback inside the gate
        index = self.sampler.best(scores, excluded)
        self.last_scores_ = scores
        return index, self.candidates[index].copy()

    def tell(
        self,
        index: int,
        value: float,
        constraints: Optional[np.ndarray] = None,
        *,
        crashed: bool = False,
    ) -> None:
        """Record an objective value for a candidate returned by :meth:`ask`.

        ``crashed=True`` marks a hidden failure (the simulator refused the input);
        with ``model_hidden_failures`` enabled the observation is excluded from the
        objective/constraint GPs and feeds the crash surrogate instead, so ``value``
        and ``constraints`` may be sentinels/placeholders for such observations.
        """
        index = int(index)
        value = float(value)
        if not 0 <= index < self.n_candidates:
            raise IndexError("candidate index is out of range")
        if self.queried_mask[index]:
            raise ValueError(f"candidate {index} has already been evaluated")
        if not np.isfinite(value):
            raise ValueError("objective value must be finite")
        if constraints is None:
            constraint_values = np.empty(0, dtype=float)
        else:
            constraint_values = np.asarray(constraints, dtype=float).reshape(-1)
        if constraint_values.size != self.n_constraints:
            raise ValueError(
                f"constraints must contain exactly {self.n_constraints} values"
            )
        if not np.all(np.isfinite(constraint_values)):
            raise ValueError("constraint values must be finite")
        self.evaluated_indices.append(index)
        self.observed_values.append(value)
        self.observed_constraints.append(constraint_values.copy())
        self.observed_crashes.append(bool(crashed))
        self.queried_mask[index] = True

    def _pop_observation(self) -> None:
        """Undo the most recent :meth:`tell` (used to roll back KB fantasies)."""
        if not self.evaluated_indices:
            raise RuntimeError("no observation to pop")
        index = self.evaluated_indices.pop()
        self.observed_values.pop()
        self.observed_constraints.pop()
        self.observed_crashes.pop()
        self.queried_mask[index] = False

    def ask_batch(self, q: int) -> list[int]:
        """Propose up to ``q`` candidates for parallel evaluation (Kriging Believer).

        Standard batch construction: pick the acquisition argmax, fantasize its
        outcome as the GP posterior mean, refit, and repeat, so successive picks in
        the same batch are diversified without any real feedback.  The fantasies are
        rolled back before returning, so this records **no** real observation --- the
        caller evaluates the batch and calls :meth:`tell` for each pick.

        Only surrogate-driven picks are fantasized.  During the initial design
        there is no posterior to fantasize from, so the batch stops after that
        single pick; the caller evaluates it, which advances the loop toward the
        surrogate-driven phase.
        With ``q == 1`` the returned singleton makes the loop identical to the
        sequential ask/tell path.
        """
        q = int(q)
        if q < 1:
            raise ValueError("q must be positive")
        batch: list[int] = []
        fantasized = 0
        try:
            for _ in range(q):
                if self.n_observations >= self.n_candidates:
                    break
                index, _ = self.ask()
                batch.append(index)
                if self.last_mean_ is None:
                    # Initial design / all-crashed: nothing to believe.
                    break
                fantasy = float(self.last_mean_[index])
                # ``tell`` stores raw objective values; _training_values negates them
                # for a minimize objective, so undo the maximize-space posterior mean.
                self.tell(index, fantasy if self.maximize else -fantasy)
                fantasized += 1
        finally:
            for _ in range(fantasized):
                self._pop_observation()
        return batch

    def best_observation(self) -> tuple[int, np.ndarray, float]:
        """Return ``(index, candidate, value)`` for the best observation so far."""
        if self.n_observations == 0:
            raise RuntimeError("No observations are available")
        values = np.asarray(self.observed_values)
        feasible = self._feasible_observations()
        if np.any(feasible):
            eligible = np.flatnonzero(feasible)
            local_position = int(
                np.argmax(values[eligible])
                if self.maximize
                else np.argmin(values[eligible])
            )
            position = int(eligible[local_position])
        else:
            constraints = np.asarray(self.observed_constraints, dtype=float)
            violations = np.sum(np.maximum(constraints, 0.0), axis=1)
            position = int(np.argmin(violations))
        index = self.evaluated_indices[position]
        return index, self.candidates[index].copy(), float(values[position])

    def result(self) -> OptimizationResult:
        """Build an optimization result from all observations collected so far."""
        best_index, best_x, best_y = self.best_observation()
        indices = np.asarray(self.evaluated_indices, dtype=int)
        return OptimizationResult(
            best_index=best_index,
            best_x=best_x,
            best_y=best_y,
            evaluated_indices=indices.copy(),
            X=self.candidates[indices].copy(),
            y=np.asarray(self.observed_values, dtype=float),
            constraint_values=np.asarray(
                self.observed_constraints,
                dtype=float,
            ).reshape(self.n_observations, self.n_constraints),
        )

    def run(
        self,
        objective: Callable[[np.ndarray], float],
        max_evaluations: int,
        callback: Optional[Callable[[int, np.ndarray, float, "BayesianOptimization"], None]] = None,
    ) -> OptimizationResult:
        """Run ask/evaluate/tell until ``max_evaluations`` total observations."""
        if max_evaluations < 1:
            raise ValueError("max_evaluations must be positive")
        if max_evaluations > self.n_candidates:
            raise ValueError("max_evaluations cannot exceed the number of candidates")
        if max_evaluations < self.n_observations:
            raise ValueError("max_evaluations is below the number already observed")

        while self.n_observations < max_evaluations:
            index, candidate = self.ask()
            value = float(objective(candidate.copy()))
            self.tell(index, value)
            if callback is not None:
                callback(index, candidate.copy(), value, self)
        return self.result()


def bo(
    objective: Callable[[np.ndarray], float],
    candidates: np.ndarray,
    max_evaluations: int,
    **optimizer_kwargs,
) -> OptimizationResult:
    """Convenience function that constructs and runs :class:`BayesianOptimization`."""
    optimizer = BayesianOptimization(candidates, **optimizer_kwargs)
    return optimizer.run(objective, max_evaluations)
