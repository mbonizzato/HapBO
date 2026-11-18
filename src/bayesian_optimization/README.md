# Bayesian optimization loop

This package is the Bayesian-optimization engine of HapBO; it does not depend on
the prior compiler (`src/prior/`). It optimizes over a finite candidate set
(`bo.py`) or over a continuous box (`continuous_bo.py`, used by the synthetic
panel).

The optimizer maximizes an expensive black-box objective. For example, each
candidate could be an electrode coordinate and evaluating the objective could
mean running an experiment and measuring its response.

## Modules

- `surrogate.py`: Gaussian-process surrogate with a Matérn 5/2 (default), RBF or
  Matérn 1/2 kernel.
- `acquisition.py`: Expected Improvement, Probability of Improvement, UCB, the
  alpha-pi shaping and the piBO prior weighting.
- `candidate_sampler.py`: chooses unqueried candidates.
- `bo.py`: the ask/evaluate/tell loop over a finite candidate set, sequential
  or in Kriging-Believer batches (`ask_batch`).
- `continuous_bo.py`: `run_continuous_bo`, the Expected-Improvement loop over a
  continuous box. Each step scores a Sobol batch plus the prior peaks, refines the
  best points with L-BFGS-B and evaluates the maximizer; the alpha-pi prior is
  min-max-normalized over a fixed reference sample of the box. It also holds the
  continuous prior-injection arms (module docstring). `scripts/run_synthetic_arms.py`
  runs it.

`bo.py` needs only NumPy; `continuous_bo.py` also needs SciPy. The repository's complete Conda
environment can be created and activated from the repository root with:

```bash
conda env create --file environment.yaml
conda activate hapbo
```

If that environment already exists, use
`conda env update --name hapbo --file environment.yaml` instead.

## Run the included synthetic example

The executable is located at the repository root in `main.py`, two levels above
this package. From the repository root, run the HapBO-style 4x8
synthetic grid with:

```bash
python main.py
```

This uses Expected Improvement, 16 evaluations, and a noisy objective whose
true peak is at `(7, 4)`. To activate the default approximate upper-right prior
and alpha-pi shaping, run:

```bash
python main.py --prior --alpha 0.7
```

The default prior center is `(8, 4)`, adjacent to but different from the true
optimum. Define a different prior center explicitly with:

```bash
python main.py --prior --prior-x 5 --prior-y 3 --prior-spread 1.5 --alpha 0.7
```

`main.py` constructs this example prior from the user-facing coordinates. The
BO engine itself accepts an arbitrary numeric prior array and does not hard-code
any prior location.

Use `--help` to see all settings:

```bash
python main.py --help
```

These include `--queries`, `--acquisition`, `--improvement-xi`, `--kappa`, `--length-scale`,
`--initial`, `--noise-std`, and `--seed`. Use `--acquisition ucb` to select UCB
instead of the default `expected-improvement`.

To save a four-panel performance figure, add `--plot`:

```bash
python main.py --prior --alpha 0.7 --plot
```

Without an explicit file argument, the figure is saved to
`results/synthetic-bo/bo_performance.png`.

Add the prior-free GP and uniform-random baselines to the
run and plot with:

```bash
python main.py --prior --baselines vanilla-bo random-search --plot
```

The comparison figure contains expected responses, simple regret, cumulative
regret, and normalized best-so-far performance for every selected method.
Regret is calculated from the known noise-free synthetic objective:

```text
instantaneous_regret(t) = f(optimum) - f(x_t)
simple_regret(t)        = f(optimum) - max(f(x_1), ..., f(x_t))
cumulative_regret(t)    = sum(instantaneous_regret(1:t))
```

Using ground truth prevents observation noise from producing artificial
negative regret. Because the default prior is centered at `(8, 4)` instead of
the optimum `(7, 4)`, simple regret starts above zero and records whether the BO
loop subsequently discovers the optimum.

## The algorithm, step by step

### 1. Define the candidates

The steps below describe the finite-candidate loop (`BayesianOptimization` in
`bo.py`). The search space is a matrix `X` with one possible experiment per row:

```text
X.shape = (number of candidates, number of dimensions)
```

Already evaluated candidates are tracked and are never selected again.

### 2. Collect initialization observations

Before fitting a model, the optimizer needs observed objective values.

- The first `n_initial` candidates are sampled randomly (seeded by `random_state`), with or
  without a prior, so a HapBO arm and a vanilla arm with the same seed share their initial
  design, as in the paper.
- `initial_indices` replaces the random draw with an explicit initial design, and
  `prior_shaping="pi-bo"` uses piBO's initial design (step 5).

### 3. Fit the Gaussian-process surrogate

The Gaussian process estimates the unknown objective using the observations.
The Matérn 5/2 covariance between candidates is

```text
k(x, x') = signal_variance
           * (1 + sqrt(5)r + 5r^2/3)
           * exp(-sqrt(5)r)
```

where `r` is the Euclidean distance scaled by `length_scale`. The `kernel`
setting selects the covariance:

| `kernel` | `k(x, x')` |
| --- | --- |
| `"matern52"` (default) | `signal_variance * (1 + sqrt(5)r + 5r^2/3) * exp(-sqrt(5)r)` |
| `"rbf"` | `signal_variance * exp(-r^2/2)` |
| `"matern12"` | `signal_variance * exp(-r)` |

All of the optimizer's GPs (objective, constraints, crash model) use the same
kernel; an unknown name raises `ValueError`.

For every candidate, the fitted GP returns:

- `mean(x)`: its predicted objective value;
- `std(x)`: the model's uncertainty at that candidate.

The implementation uses a Cholesky factorization for stable GP inference.

### 4. Calculate the acquisition function

The primary optimizer uses Expected Improvement (EI) by default:

```text
improvement(x) = mean(x) - best_observed_value - xi
EI(x) = improvement(x) * Phi(z) + std(x) * phi(z)
z = improvement(x) / std(x)
```

Here, `Phi` and `phi` are the standard normal CDF and PDF. A larger non-negative
`xi` requires a larger improvement over the best observation.

UCB is available by setting `acquisition="ucb"` in Python or passing
`--acquisition ucb` to `main.py`:

```text
UCB(x) = mean(x) + kappa * std(x)
```

A larger `kappa` favors uncertain candidates and therefore more exploration. A
smaller `kappa` favors candidates with a high predicted value.
`acquisition="probability-of-improvement"` selects the Probability of
Improvement, `PI(x) = Phi(z)`. Every acquisition starts with initialization
observations before the GP is fitted.

### 5. Optionally apply alpha-pi shaping

The BO code accepts an optional numeric `prior` with one preference score per
candidate. It does not know whether this array was constructed manually, by an
LLM, or by another system.

The alpha-pi rule normalizes only the prior. It first multiplies the raw
acquisition and then zero-clamps:

```text
weighted(x) = acquisition_raw(x)
              * (alpha * prior_normalized(x) + (1 - alpha))
shaped(x) = max(weighted(x), 0)
```

- `alpha = 0` gives the standard, unshaped acquisition.
- A larger `alpha` biases selection more strongly toward high-prior regions.
- Alpha-pi limits `alpha` to `0.9`, so even a low-prior candidate
  retains a multiplier of at least `0.1`.
- The raw acquisition is never shifted by min-max normalization before prior
  weighting; negative weighted values are clamped to zero afterward.
- A uniform prior makes the multiplier equal to one. For EI this has no effect;
  for a base acquisition that can be negative, such as UCB, the prescribed zero
  clamp can still collapse negative values to zero.

The prior changes the acquisition scores; it does not change GP training.

`prior_shaping="pi-bo"` applies piBO (Hvarfner et al. 2022) instead: the initial
design is the prior mode followed by draws from the prior
(`pi_bo_initial_design`), and the acquisition is weighted by the prior raised to
`pi_bo_beta / t`. A `prior_mean` callable sets the GP prior mean (the
prior-to-mean arm of the study worker).

### 6. Optionally model black-box constraints

Pass `n_constraints=m` when an evaluation returns constraints
`g_1(x), ..., g_m(x)` and feasibility means every `g_j(x) <= 0`. The optimizer
fits one additional GP per constraint and multiplies the shaped objective
acquisition by:

```text
P(feasible at x) = product_j P(g_j(x) <= 0)
```

Supply the aligned values with `optimizer.tell(index, objective, constraints)`.
The reported best observation is the best feasible objective. If no feasible
query has been found, it reports the observed point with the smallest summed
positive violation. With `n_constraints=0` (the default), no constraint GP is
fitted. `model_hidden_failures=True` excludes crashed evaluations (the SOLAR
`1e20` sentinel) from the GPs and multiplies the acquisition by the modeled
probability that the simulator returns a value.

### 7. Select, evaluate, and update

The optimizer selects the unqueried candidate with the largest acquisition
score, evaluates the objective, stores the result, and repeats from step 3. At
the end, it returns the best observed candidate rather than the largest GP
prediction.

## Basic usage without a prior

When this repository's `src` directory is on `PYTHONPATH`:

```python
import numpy as np

from bayesian_optimization import BayesianOptimization


# One-dimensional candidate set: [[0.0], [0.1], ..., [10.0]]
candidates = np.linspace(0.0, 10.0, 101).reshape(-1, 1)


def objective(x: np.ndarray) -> float:
    return float(np.sin(x[0]) - 0.02 * (x[0] - 7.0) ** 2)


optimizer = BayesianOptimization(
    candidates,
    length_scale=1.0,
    improvement_xi=0.0,
    n_initial=3,
    random_state=7,
)

result = optimizer.run(objective, max_evaluations=15)
print(result.best_x, result.best_y)
```

The `bo(...)` convenience function performs the same construction and run in a
single call.

## Usage with a numeric alpha-pi prior

This example constructs a numeric prior directly, without the prior compiler.

```python
prior = np.exp(-0.5 * ((candidates[:, 0] - 7.5) / 1.5) ** 2)

optimizer = BayesianOptimization(
    candidates,
    prior=prior,
    alpha=0.6,
    length_scale=1.0,
    acquisition="expected-improvement",
    n_initial=3,
    random_state=7,
)

result = optimizer.run(objective, max_evaluations=15)
```

The repository-level `run_prior_guided_optimization` function compiles
`prior.json`, evaluates the joint prior on aligned native candidates, and
passes this array to `BayesianOptimization`. The BO package itself is
independent of how the prior was produced.

## Manual ask/tell operation

Use ask/tell when the experiment is performed outside Python:

```python
index, candidate = optimizer.ask()

# Perform the real experiment here.
measured_value = run_experiment(candidate)

optimizer.tell(index, measured_value)
```

Continue until the evaluation budget is exhausted, then call
`optimizer.result()`.

## Main parameters

| Parameter | Meaning |
| --- | --- |
| `length_scale` | How quickly objective correlation decays with distance |
| `signal_variance` | GP prior variance |
| `noise_variance` | Expected observation-noise variance |
| `kernel` | GP covariance: `"matern52"` (default), `"rbf"` or `"matern12"` |
| `acquisition` | `"expected-improvement"` (default), `"probability-of-improvement"` or `"ucb"` |
| `improvement_xi` | EI improvement threshold |
| `kappa` | UCB exploration strength |
| `prior` | Optional preference value for every candidate |
| `alpha` | Prior influence, `0`–`0.9` |
| `prior_shaping` | `"alpha-pi"` (default), `"pi-bo"` or `"none"` |
| `pi_bo_beta` | piBO decay `beta` (required with `"pi-bo"`; `N / 10` for `N` BO iterations in the studies) |
| `n_initial` | Number of initialization observations |
| `initial_indices` | Optional explicit initial design (candidate indices) |
| `maximize` | Maximize when true; minimize when false |
| `random_state` | Seed used for reproducible initialization |
