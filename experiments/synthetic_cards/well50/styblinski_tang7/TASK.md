# Prompt A -- LLM as an expert.

## 0. Prompt
You are an expert in the domain described below. Using your knowledge of this
field, generate a prior belief over the location of the optimum for the
Bayesian optimization problem defined by the search space and optimization
description below. Write the belief only in the closed PSL vocabulary you were
given; never invent tokens.

## 1. Domain
A 7-parameter black-box objective, to be MINIMIZED; each input ranges [-5, 5]. No expert intuition about the location of the optimum is available. An approximate analytic model of the objective is available. Its functional form is believed to describe the objective's general shape, but its parameter values are only approximately calibrated (fidelity unknown), so the locations and relative depths of its minima may differ from those of the true objective. The model is:

F(x) = 0.5 * sum_{i=0..6} [ y_i^4 - a_i*y_i^2 + b_i*y_i ],   where y_i = lam_i * (x_i - s_i)
per-parameter constants:
a_i  b_i  lam_i  s_i
17.89  1.842  1.03  -0.2445
18.07  2.603  1.062  0.7094
18.11  2.742  0.9378  -0.6588
17.15  1.711  1.035  -1.615
16.12  3.164  1.045  0.6248
14.48  1.507  0.8852  -1.854
15.75  1.667  1.046  -0.4228

## 2. Search space
Parameter 1
name: x0
type: continuous
range: [-5.0, 5.0]  # [low, high]
scale: linear

Parameter 2
name: x1
type: continuous
range: [-5.0, 5.0]  # [low, high]
scale: linear

Parameter 3
name: x2
type: continuous
range: [-5.0, 5.0]  # [low, high]
scale: linear

Parameter 4
name: x3
type: continuous
range: [-5.0, 5.0]  # [low, high]
scale: linear

Parameter 5
name: x4
type: continuous
range: [-5.0, 5.0]  # [low, high]
scale: linear

Parameter 6
name: x5
type: continuous
range: [-5.0, 5.0]  # [low, high]
scale: linear

Parameter 7
name: x6
type: continuous
range: [-5.0, 5.0]  # [low, high]
scale: linear

## 3. Optimization description
Objective: a scalar black-box value, lower is better.
Direction: minimize
Evaluation budget: about 50 evaluations.
Known constraints or prior observations: none; the only information available is the approximate analytic model in the domain description above.

Questions to consider before writing the prior:
- Where in the search box are the model's low (good) regions, how many are there, and how do their depths compare?
- Which parameters does the model value depend on most strongly, and which interact?
- How much should an approximately calibrated model's minimum locations be trusted?
