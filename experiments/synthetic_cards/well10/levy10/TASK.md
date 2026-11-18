# Prompt A -- LLM as an expert.

## 0. Prompt
You are an expert in the domain described below. Using your knowledge of this
field, generate a prior belief over the location of the optimum for the
Bayesian optimization problem defined by the search space and optimization
description below. Write the belief only in the closed PSL vocabulary you were
given; never invent tokens.

## 1. Domain
A 10-parameter black-box objective, to be MINIMIZED; each input ranges [-10, 10]. No expert intuition about the location of the optimum is available. An approximate analytic model of the objective is available. Its functional form is believed to describe the objective's general shape, but its parameter values are only approximately calibrated (fidelity unknown), so the locations and relative depths of its minima may differ from those of the true objective. The model is:

F(x) = sin^2(pi*w_0) + sum_{i=0..8} (w_i - 1)^2 * [1 + amp_i * sin^2(kap_i*pi*w_i + 1)] + (w_9 - 1)^2 * [1 + sin^2(2*pi*w_9)],   where w_i = 1 + (x_i - o_i)/4
per-parameter constants (amp/kap unused for i=9):
o_i  amp_i  kap_i
0.182  9.773  0.9442
0.8299  10.19  1.016
-9.533  10.64  1.047
-0.042  10.83  0.9675
0.03906  9.213  0.9682
0.1158  10.3  1.057
-0.07893  10.32  0.9979
1.644  10.53  1.023
0.6852  10.9  0.968
6.693  -  -

## 2. Search space
Parameter 1
name: x0
type: continuous
range: [-10.0, 10.0]  # [low, high]
scale: linear

Parameter 2
name: x1
type: continuous
range: [-10.0, 10.0]  # [low, high]
scale: linear

Parameter 3
name: x2
type: continuous
range: [-10.0, 10.0]  # [low, high]
scale: linear

Parameter 4
name: x3
type: continuous
range: [-10.0, 10.0]  # [low, high]
scale: linear

Parameter 5
name: x4
type: continuous
range: [-10.0, 10.0]  # [low, high]
scale: linear

Parameter 6
name: x5
type: continuous
range: [-10.0, 10.0]  # [low, high]
scale: linear

Parameter 7
name: x6
type: continuous
range: [-10.0, 10.0]  # [low, high]
scale: linear

Parameter 8
name: x7
type: continuous
range: [-10.0, 10.0]  # [low, high]
scale: linear

Parameter 9
name: x8
type: continuous
range: [-10.0, 10.0]  # [low, high]
scale: linear

Parameter 10
name: x9
type: continuous
range: [-10.0, 10.0]  # [low, high]
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
