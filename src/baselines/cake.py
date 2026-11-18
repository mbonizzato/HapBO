# Portions of this file are ported from CAKE (https://github.com/richardcsuwandi/cake,
# commit ece5cbc04b3e1d14d22a4fe80f38e7d16ff7c809), distributed under the MIT License:
#
# Copyright (c) 2025 Richard Cornelius Suwandi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Finite-candidate CAKE adaptation: LLM-evolved GP kernels ranked by BAKER.

Context-Aware Kernel Evolution (Suwandi et al., NeurIPS 2025) replaces the
fixed GP kernel with a population of kernel *expressions* that one evolutionary
generation refines before every query.  Each generation scores the population by
BIC, asks the LLM for one crossover and, with probability ``mutation_prob``, one
mutation, truncates back to the fittest ``num_population`` expressions, and then
selects the query by BAKER: the kernel whose expected improvement, weighted by
its softmax BIC fitness, is largest.

Three adaptations are forced by this project's finite-candidate contract; they are
also listed in external/PINNED_COMMITS.txt (CAKE entry).

* The reference maximizes expected improvement with ``optimize_acqf`` over a
  continuous box.  Here every kernel scores the benchmark's candidate table
  exhaustively and the argmax over the still-available rows is exact.
* The reference wraps each GP in ``Normalize(d)``, whose bounds are learned from
  the training subset.  This project's candidate encoding already min-max
  normalizes every numeric column over the *complete* table, so the input
  transform would rescale an already-unit cube by a handful of observations.
  The encoding is the reference transform applied to the whole domain, so no
  further input transform is installed.
* The reference draws with the global NumPy random state and calls OpenAI; both
  are replaced by this project's seeded generator and the shared LLM client (the
  OpenAI-compatible endpoint of LLM_BASE_URL / LLM_MODEL).

One deviation is deliberate.  The released ``fit_gp_model`` computes its "BIC"
from ``mll(model(train_x), train_y)`` *after* switching the model to eval mode,
so the quantity is the posterior predictive density at the training inputs,
evaluated against untransformed targets while the model speaks in standardized
ones, and divided by the observation count before the full ``k log n`` penalty
is added.  The paper defines the ordinary BIC over the log marginal likelihood,
which is what this implementation computes: ``-2 log p(y | X, theta_hat) +
k log n`` at the fitted hyperparameters, in the model's standardized outcome
space.  Ranking is what BIC is used for here, and both the fitness softmax and
the BAKER weights are invariant to that shared outcome scale.

Unlike every other GP-based arm in this study, CAKE fits its kernel
hyperparameters by marginal likelihood at every generation.  This is inherent to
the method rather than a configuration choice: BIC is only defined at fitted
hyperparameters, and a frozen lengthscale would leave the evolved structure
unable to express anything.  The study's shared ``length_scale`` and
``noise_std`` therefore do not apply to this arm.
"""

from __future__ import annotations

import contextlib
import math
import os
import re
import tempfile
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.exceptions.warnings import NumericsWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from gpytorch.kernels import (
    LinearKernel,
    MaternKernel,
    PeriodicKernel,
    RBFKernel,
    RQKernel,
    ScaleKernel,
)
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood

from .base import BaselineOptimizer, Config, Observation, config_key
from .workers.llm_defaults import DEFAULT_LLM_MODEL

try:  # main process imports the ``src.`` package; the isolated worker adds ``src/`` to sys.path
    from src.llm.client import LLMClient
except ImportError:  # pragma: no cover - worker-context fallback
    from llm.client import LLMClient


@contextlib.contextmanager
def exact_settings():
    """Double-precision exact GP solves.

    Self-contained so the arm needs no other GP module.
    """

    with gpytorch.settings.fast_pred_var(False), gpytorch.settings.max_cholesky_size(
        100_000
    ), gpytorch.settings.cholesky_jitter(
        double_value=1e-12
    ), gpytorch.settings.min_variance(double_value=0.0):
        yield


# Every symbol the reference's ``parse_kernel`` understands. ``M1`` is
# parseable but absent from the evolvable set, exactly as upstream.
BASE_KERNEL_FACTORIES = {
    "SE": lambda d: RBFKernel(ard_num_dims=d),
    "PER": lambda d: PeriodicKernel(ard_num_dims=d),
    "LIN": lambda d: LinearKernel(ard_num_dims=d),
    "RQ": lambda d: RQKernel(ard_num_dims=d),
    "M1": lambda d: MaternKernel(nu=0.5, ard_num_dims=d),
    "M3": lambda d: MaternKernel(nu=1.5, ard_num_dims=d),
    "M5": lambda d: MaternKernel(nu=2.5, ard_num_dims=d),
}
BASE_KERNELS = ("SE", "PER", "LIN", "RQ", "M3", "M5")
OPERATORS = ("+", "*")

SYSTEM_PROMPT_TEMPLATE = """
You are an expert in machine learning, specializing in Gaussian processes. Here are the observations we have collected so far:
{observations}

Please analyze these observations to identify patterns in the data that can be captured by a kernel function.
You can use any of the following base kernels: {base_kernels}, and combine these kernels using the following operators: {operators}.
Your goal is to construct a kernel expression that best explains the observed data.
The kernel will be evaluated using a fitness score normalized between [0, 1], where higher values indicate better fit to the data.
"""

# The reference's parser reads a "Kernel:" line and an "Analysis:" line that its
# own prompts never request, so a compliant model almost never produces one and
# every proposal falls back. The format is stated here instead; nothing else in
# the prompt text is changed.
RESPONSE_FORMAT = """
Respond in exactly this format, with no other text:
Kernel: <kernel expression>
Analysis: <brief reasoning>
"""

CROSSOVER_PROMPT_TEMPLATE = """
You are given two parent kernels and their fitness scores:
{parent_kernel1} ({fitness1}),  {parent_kernel2} ({fitness2})

Please propose a new kernel that has a potentially higher fitness score.
You may combine the parent kernels using any of the operators from: {operators}.
Briefly explain your reasoning behind the proposed kernel.
""" + RESPONSE_FORMAT

MUTATION_PROMPT_TEMPLATE = """
You are given a kernel and its fitness score:
{kernel} ({fitness})

Please propose a new kernel that has a potentially higher fitness score.
You may replace a base kernel in the current expression with another base kernel from the set: {base_kernels}.
Briefly explain your reasoning behind the proposed kernel.
""" + RESPONSE_FORMAT


def parse_kernel_expression(expression: str, n_features: int) -> ScaleKernel:
    """Build the scaled covariance module named by one kernel expression.

    Follows the reference's ``parse_kernel``: innermost parenthesized groups are
    scaled and folded into named sub-kernels, then the remaining flat expression
    is folded left to right with no operator precedence, so ``SE + PER * LIN``
    means ``(SE + PER) * LIN``.  Malformed expressions raise, which is how a
    useless LLM proposal is discarded.
    """

    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("a kernel expression must be a non-empty string")
    if not isinstance(n_features, int) or isinstance(n_features, bool) or n_features < 1:
        raise ValueError("n_features must be a positive integer")

    kernels: dict[str, Any] = {
        name: factory(n_features) for name, factory in BASE_KERNEL_FACTORIES.items()
    }

    def fold(subexpression: str) -> ScaleKernel:
        names = re.findall(r"[\w]+", subexpression)
        operators = re.findall(r"[+*]", subexpression)
        if not names or len(operators) != len(names) - 1:
            raise ValueError(f"malformed kernel expression {subexpression!r}")
        if re.sub(r"[\w()+*\s]", "", subexpression):
            raise ValueError(f"malformed kernel expression {subexpression!r}")
        result = kernels[names[0]]
        for position, operator in enumerate(operators):
            right = kernels[names[position + 1]]
            result = result + right if operator == "+" else result * right
        return ScaleKernel(result)

    expression = expression.strip()
    if len(re.findall(r"[\w]+", expression)) > 10:
        # A runaway proposal would otherwise fit a model with hundreds of
        # hyperparameters against a few observations.
        raise ValueError("kernel expression has too many terms")
    while "(" in expression:
        groups = re.findall(r"\(([^()]+)\)", expression)
        if not groups:
            raise ValueError(f"unbalanced parentheses in {expression!r}")
        for group in groups:
            name = f"SubKernel{len(kernels)}"
            kernels[name] = fold(group)
            expression = expression.replace(f"({group})", name, 1)
    return fold(expression)


def fit_kernel_gp(
    X: np.ndarray,
    y: np.ndarray,
    expression: str,
    *,
    noise_init: float = 1e-4,
) -> tuple[SingleTaskGP, float]:
    """Fit one kernel expression by marginal likelihood and return its BIC."""

    train_X = torch.as_tensor(np.asarray(X, dtype=float), dtype=torch.double)
    train_Y = torch.as_tensor(np.asarray(y, dtype=float), dtype=torch.double).reshape(-1, 1)
    covariance = parse_kernel_expression(expression, int(train_X.shape[-1])).double()
    with warnings.catch_warnings():
        # Standardize warns for a single observation and then falls back to a
        # unit scale, which is the behaviour this arm relies on at n_initial=1.
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        model = SingleTaskGP(
            train_X,
            train_Y,
            covar_module=covariance,
            outcome_transform=Standardize(m=1),
        ).double()
        model.likelihood.noise = float(noise_init)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # BIC at the fitted hyperparameters, from the log marginal likelihood
        # in train mode, as the paper defines it.
        model.train()
        model.likelihood.train()
        with torch.no_grad(), exact_settings():
            prior = model.likelihood(model(*model.train_inputs))
            log_marginal_likelihood = float(prior.log_prob(model.train_targets))
        model.eval()
        model.likelihood.eval()

    n_parameters = sum(parameter.numel() for parameter in model.parameters())
    n_observations = int(train_X.shape[0])
    bic = -2.0 * log_marginal_likelihood + n_parameters * math.log(n_observations)
    if not math.isfinite(bic):
        raise ValueError(f"kernel {expression!r} produced a non-finite BIC")
    return model, bic


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    return weights / float(np.sum(weights))


class CAKE(BaselineOptimizer):
    """Context-Aware Kernel Evolution over a finite candidate table."""

    name = "cake"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.num_crossover = int(self.kwargs.get("num_crossover", 1))
        self.mutation_prob = float(self.kwargs.get("mutation_prob", 0.7))
        self.num_population = int(self.kwargs.get("num_population", 6))
        self.max_crossover_length = int(self.kwargs.get("max_crossover_length", 10))
        self.noise_init = float(self.kwargs.get("noise_init", 1e-4))
        self.max_observations_in_prompt = int(
            self.kwargs.get("max_observations_in_prompt", 50)
        )
        self.maximize = bool(self.objective_meta.get("maximize", True))
        self.n_constraints = int(self.objective_meta.get("n_constraints", 0))

        if self.num_crossover < 0:
            raise ValueError("CAKE num_crossover must be non-negative")
        if not 0.0 <= self.mutation_prob <= 1.0:
            raise ValueError("CAKE mutation_prob must lie in [0, 1]")
        if self.num_population < 1:
            raise ValueError("CAKE num_population must be positive")
        if self.max_crossover_length < 1:
            raise ValueError("CAKE max_crossover_length must be positive")
        if self.noise_init <= 0:
            raise ValueError("CAKE noise_init must be positive")
        if self.n_constraints:
            raise ValueError("CAKE currently supports unconstrained objectives only")
        if self.llm_client is None:
            self.llm_client = self._build_default_llm_client()

        self._X = np.asarray(self.encoded_candidates, dtype=float)
        self._train_indices: list[int] = []
        self._train_values: list[float] = []
        self._asked_indices: dict[str, int] = {}

        # Expression -> {"fitness": bic, "model": model}. Seeded with the
        # reference's diverse set of base kernels.
        self.population: dict[str, dict[str, Any]] = {
            kernel: {} for kernel in BASE_KERNELS
        }
        self.population_prob = np.full(len(BASE_KERNELS), 1.0 / len(BASE_KERNELS))
        self.kernel_records: list[dict[str, Any]] = []
        self.crossover_count = 0
        self.mutation_count = 0
        self.fallback_count = 0
        self.parse_failure_count = 0
        self.fit_failure_count = 0
        self.last_selected_kernel_: str | None = None

    def _build_default_llm_client(self) -> LLMClient:
        """Point CAKE at a locally served OpenAI-compatible endpoint from the env.

        Mirrors how the other in-loop arms source their endpoint (LLM_BASE_URL /
        LLM_MODEL / LLM_PROVIDER, Qwen thinking disabled). The per-run cache and
        usage log land in ``CAKE_RUN_DIR`` (set by the worker) so token/latency
        telemetry is recorded exactly like every other LLM arm.
        """

        run_dir = os.environ.get("CAKE_RUN_DIR")
        cache_dir = (
            Path(run_dir) / "llm_cache"
            if run_dir
            else Path(tempfile.mkdtemp(prefix="cake_llm_"))
        )
        client = LLMClient(
            model=str(
                self.kwargs.get(
                    "llm_model",
                    os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL),
                )
            ),
            temperature=float(self.kwargs.get("llm_temperature", 1.0)),
            max_tokens=int(self.kwargs.get("llm_max_tokens", 2048)),
            cache_dir=cache_dir,
            budget=int(self.kwargs.get("llm_call_budget", 512)),
            provider=str(os.environ.get("LLM_PROVIDER", "openai")),
            base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            enable_thinking=str(os.environ.get("LLM_ENABLE_THINKING", "false")).lower()
            == "true",
        )
        if run_dir:
            client.set_run_directory(Path(run_dir))
        return client

    # -- diagnostics -------------------------------------------------------

    @property
    def best_kernel(self) -> str | None:
        """Return the fittest expression currently in the population."""

        scored = {
            kernel: entry["fitness"]
            for kernel, entry in self.population.items()
            if "fitness" in entry
        }
        if not scored:
            return None
        return min(scored, key=scored.get)

    def population_summary(self) -> list[dict[str, Any]]:
        """Return the population with its BIC fitness and BAKER weight."""

        kernels = list(self.population)
        return [
            {
                "kernel": kernel,
                "bic": float(self.population[kernel]["fitness"])
                if "fitness" in self.population[kernel]
                else None,
                "weight": float(self.population_prob[position])
                if position < self.population_prob.size
                else None,
            }
            for position, kernel in enumerate(kernels)
        ]

    # -- LLM ---------------------------------------------------------------

    def _observation_block(self) -> str:
        # Oriented values, so the prompt describes the same maximization
        # problem the GPs are fitted to, as the reference's always-maximizing
        # driver does.
        rows = list(zip(self._train_indices, self._oriented_values().tolist()))
        # A long run would otherwise resend every observation; the study
        # budget stays below this bound.
        rows = rows[-self.max_observations_in_prompt :]
        return "\n".join(
            f"x = {[float(entry) for entry in self._X[index]]}, y = {float(value)}"
            for index, value in rows
        )

    def _system_prompt(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(
            observations=self._observation_block(),
            base_kernels=list(BASE_KERNELS),
            operators=list(OPERATORS),
        )

    @staticmethod
    def parse_response(response: str) -> tuple[str, str]:
        """Split one completion into its kernel expression and its analysis."""

        if not isinstance(response, str) or "Kernel:" not in response:
            raise ValueError("the response declares no kernel")
        start = response.index("Kernel:") + len("Kernel:")
        end = response.find("\n", start)
        kernel = (response[start:] if end < 0 else response[start:end]).strip()
        if not kernel:
            raise ValueError("the response declares an empty kernel")
        analysis = ""
        if "Analysis:" in response:
            analysis = response[
                response.index("Analysis:") + len("Analysis:") :
            ].strip()
        return kernel, analysis

    def _propose(self, prompt: str) -> tuple[str, str]:
        response = self.llm_client.complete(
            prompt,
            system=self._system_prompt(),
        )
        return self.parse_response(response)

    # -- evolution ---------------------------------------------------------

    def _register(self, expression: str) -> bool:
        """Fit one expression and admit it to the population when it holds."""

        try:
            model, bic = fit_kernel_gp(
                self._X[np.asarray(self._train_indices, dtype=int)],
                self._oriented_values(),
                expression,
                noise_init=self.noise_init,
            )
        except Exception:
            self.fit_failure_count += 1
            return False
        self.population[expression] = {"fitness": float(bic), "model": model}
        return True

    def _oriented_values(self) -> np.ndarray:
        values = np.asarray(self._train_values, dtype=float)
        return values if self.maximize else -values

    def compute_fitness(self) -> None:
        """Refit every surviving expression against the observations so far."""

        survivors: dict[str, dict[str, Any]] = {}
        for expression in list(self.population):
            if self._register(expression):
                survivors[expression] = self.population[expression]
            else:
                self.population.pop(expression, None)
        self.population = survivors
        if not self.population:
            self.population_prob = np.empty(0, dtype=float)
            return
        fitness = np.asarray(
            [entry["fitness"] for entry in self.population.values()], dtype=float
        )
        # The reference standardizes before the parent-selection softmax for
        # numerical stability; ddof=1 matches torch.std. A single survivor, or
        # a tie, leaves the population uniform rather than undefined.
        spread = float(fitness.std(ddof=1)) if fitness.size > 1 else 0.0
        standardized = (
            (fitness - float(fitness.mean())) / spread if spread > 1e-12
            else np.zeros_like(fitness)
        )
        self.population_prob = _softmax(-standardized)

    def generate_kernels(self) -> None:
        """Apply one LLM crossover and, with ``mutation_prob``, one mutation."""

        if not self.population:
            return
        mating_pool = list(self.population)
        for _ in range(self.num_crossover):
            if len(mating_pool) < 2:
                break
            first, second = self._rng.choice(
                np.asarray(mating_pool, dtype=object),
                size=2,
                p=self.population_prob,
                replace=False,
            )
            try:
                expression, _ = self._propose(
                    CROSSOVER_PROMPT_TEMPLATE.format(
                        parent_kernel1=first,
                        parent_kernel2=second,
                        fitness1=self.population[first]["fitness"],
                        fitness2=self.population[second]["fitness"],
                        operators=list(OPERATORS),
                    ),
                )
            except Exception:
                # The reference's own fallback: join the parents with a random
                # operator rather than lose the generation.
                self.fallback_count += 1
                operator = str(self._rng.choice(np.asarray(OPERATORS, dtype=object)))
                expression = f"{first} {operator} {second}"
            self.crossover_count += 1
            # The reference discards long crossovers, which keeps the composed
            # expressions from growing without bound.
            if len(expression) < self.max_crossover_length:
                if not self._register(expression):
                    self.parse_failure_count += 1
                    expression = None
            else:
                expression = None
            self._last_crossover = expression

        self._last_mutation = None
        if self._rng.random() < self.mutation_prob and self.population:
            target = min(self.population, key=lambda key: self.population[key]["fitness"])
            self.mutation_count += 1
            try:
                expression, _ = self._propose(
                    MUTATION_PROMPT_TEMPLATE.format(
                        kernel=target,
                        fitness=self.population[target]["fitness"],
                        base_kernels=list(BASE_KERNELS),
                    ),
                )
            except Exception:
                # The reference skips the mutation entirely when the call or the
                # parse fails; there is no fallback expression here.
                self.fallback_count += 1
                return
            if self._register(expression):
                self._last_mutation = expression
            else:
                self.parse_failure_count += 1

    def update_population(self) -> None:
        """Keep the fittest expressions and recompute the BAKER weights."""

        ordered = sorted(self.population.items(), key=lambda item: item[1]["fitness"])
        self.population = dict(ordered[: self.num_population])
        fitness = np.asarray(
            [entry["fitness"] for entry in self.population.values()], dtype=float
        )
        # BAKER weights the acquisition by the raw-BIC softmax, as the reference
        # does: lower BIC, higher weight.
        self.population_prob = (
            _softmax(-fitness) if fitness.size else np.empty(0, dtype=float)
        )

    def run_generation(self) -> None:
        """Run one CAKE generation against the observations collected so far."""

        self._last_crossover = None
        self._last_mutation = None
        self.compute_fitness()
        self.generate_kernels()
        self.update_population()

    # -- BAKER selection ---------------------------------------------------

    def _available_indices(self) -> np.ndarray:
        reserved = set(self._evaluated) | set(self._pending)
        return np.asarray(
            [
                index
                for index, candidate in enumerate(self.candidates)
                if config_key(candidate) not in reserved
            ],
            dtype=int,
        )

    def _expected_improvement(
        self, model: SingleTaskGP, indices: np.ndarray, best_f: float
    ) -> np.ndarray:
        X = torch.as_tensor(self._X[indices], dtype=torch.double).unsqueeze(-2)
        with warnings.catch_warnings():
            # The reference scores with the analytic EI, not its log form.
            warnings.simplefilter("ignore", NumericsWarning)
            acquisition = ExpectedImprovement(
                model, best_f=torch.tensor(best_f, dtype=torch.double)
            )
        with torch.no_grad(), exact_settings():
            values = acquisition(X).reshape(-1).cpu().numpy()
        return np.asarray(values, dtype=float)

    def select_query(self) -> int:
        """Return the candidate index BAKER ranks highest, or -1 when stuck."""

        available = self._available_indices()
        if available.size == 0:
            raise RuntimeError("all CAKE candidates are unavailable")
        if not self.population:
            return -1
        best_f = float(np.max(self._oriented_values()))
        kernels = list(self.population)
        scores: list[float] = []
        choices: list[int] = []
        for position, kernel in enumerate(kernels):
            try:
                values = self._expected_improvement(
                    self.population[kernel]["model"], available, best_f
                )
                local = int(np.argmax(values))
                acquisition_value = float(values[local])
                if not math.isfinite(acquisition_value):
                    raise ValueError("non-finite acquisition value")
            except Exception:
                scores.append(-np.inf)
                choices.append(-1)
                continue
            scores.append(float(self.population_prob[position]) * acquisition_value)
            choices.append(int(available[local]))
        scores_array = np.asarray(scores, dtype=float)
        if not np.any(np.isfinite(scores_array)):
            return -1
        # Ties are broken uniformly, as the reference does.
        best = np.flatnonzero(scores_array == np.max(scores_array))
        chosen = int(self._rng.choice(best))
        if choices[chosen] < 0:
            return -1
        self.last_selected_kernel_ = kernels[chosen]
        return choices[chosen]

    # -- ask/tell ----------------------------------------------------------

    def ask(self) -> Config:
        if len(self._history) + len(self._pending) >= self.budget:
            raise RuntimeError("the baseline evaluation budget is exhausted")
        if not self._train_indices:
            config = self._random_unqueried_config()
            self._asked_indices[config_key(config)] = self._index_for(config)
            return config

        self.run_generation()
        index = self.select_query()
        if index < 0:
            # No kernel produced a usable model or acquisition surface, so the
            # generation yields nothing to rank and the query stays seeded.
            self.last_selected_kernel_ = None
            config = self._random_unqueried_config()
            self._asked_indices[config_key(config)] = self._index_for(config)
            self._record_generation(self._index_for(config))
            return config

        config = dict(self.candidates[index])
        key = config_key(config)
        self._pending.add(key)
        self._asked_indices[key] = index
        self._record_generation(index)
        return config

    def _record_generation(self, index: int) -> None:
        self.kernel_records.append(
            {
                "query": len(self._history) + 1,
                "candidate_index": int(index),
                "selected_kernel": self.last_selected_kernel_,
                "best_kernel": self.best_kernel,
                "crossover_kernel": getattr(self, "_last_crossover", None),
                "mutation_kernel": getattr(self, "_last_mutation", None),
                "population": self.population_summary(),
            }
        )

    def tell(self, config: Mapping[str, Any], observation: Observation) -> None:
        key = config_key(config)
        index = self._asked_indices.pop(key, self._index_for(config))
        value = (
            float(observation["objective"])
            if isinstance(observation, Mapping)
            else float(observation)
        )
        self._record(config, observation)
        self._train_indices.append(index)
        self._train_values.append(value)


__all__ = [
    "BASE_KERNELS",
    "BASE_KERNEL_FACTORIES",
    "CAKE",
    "OPERATORS",
    "fit_kernel_gp",
    "parse_kernel_expression",
]
