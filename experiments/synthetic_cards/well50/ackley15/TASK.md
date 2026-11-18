# Prompt A -- LLM as an expert.

## 0. Prompt
You are an expert in the domain described below. Using your knowledge of this
field, generate a prior belief over the location of the optimum for the
Bayesian optimization problem defined by the search space and optimization
description below. Write the belief only in the closed PSL vocabulary you were
given; never invent tokens.

## 1. Domain
A 15-parameter black-box objective, to be MINIMIZED; each input ranges [-32.768, 32.768]. No expert intuition about the location of the optimum is available. An approximate analytic model of the objective is available. Its functional form is believed to describe the objective's general shape, but its parameter values are only approximately calibrated (fidelity unknown), so the locations and relative depths of its minima may differ from those of the true objective. The model is:

F(x) = -20 * exp( -0.2121 * sqrt( (1/15) * sum_i (x_i - c_i)^2 ) ) - exp( (1/15) * sum_i cos( kap_i * 2*pi * (x_i - c_i) ) ) + 20 + e
per-parameter constants:
c_i  kap_i
18.1  1.057
-29.04  0.9319
-16.48  1.029
18.57  1.008
-12.92  1.023
-14.87  1.045
18.48  0.9375
14.94  0.9907
-17.51  0.9321
-17.88  0.9703
7.261  1.038
-6.172  0.9566
11.69  0.9802
15.58  1.068
14.39  1.065

## 2. Search space
Parameter 1
name: x0
type: continuous
range: [-32.768, 32.768]  # [low, high]
scale: linear

Parameter 2
name: x1
type: continuous
range: [-32.768, 32.768]  # [low, high]
scale: linear

Parameter 3
name: x2
type: continuous
range: [-32.768, 32.768]  # [low, high]
scale: linear

Parameter 4
name: x3
type: continuous
range: [-32.768, 32.768]  # [low, high]
scale: linear

Parameter 5
name: x4
type: continuous
range: [-32.768, 32.768]  # [low, high]
scale: linear

Parameter 6
name: x5
type: continuous
range: [-32.768, 32.768]  # [low, high]
scale: linear

Parameter 7
name: x6
type: continuous
range: [-32.768, 32.768]  # [low, high]
scale: linear

Parameter 8
name: x7
type: continuous
range: [-32.768, 32.768]  # [low, high]
scale: linear

Parameter 9
name: x8
type: continuous
range: [-32.768, 32.768]  # [low, high]
scale: linear

Parameter 10
name: x9
type: continuous
range: [-32.768, 32.768]  # [low, high]
scale: linear

Parameter 11
name: x10
type: continuous
range: [-32.768, 32.768]  # [low, high]
scale: linear

Parameter 12
name: x11
type: continuous
range: [-32.768, 32.768]  # [low, high]
scale: linear

Parameter 13
name: x12
type: continuous
range: [-32.768, 32.768]  # [low, high]
scale: linear

Parameter 14
name: x13
type: continuous
range: [-32.768, 32.768]  # [low, high]
scale: linear

Parameter 15
name: x14
type: continuous
range: [-32.768, 32.768]  # [low, high]
scale: linear

## 3. Optimization description
Objective: a scalar black-box value, lower is better.
Direction: minimize
Evaluation budget: about 150 evaluations.
Known constraints or prior observations: none; the only information available is the approximate analytic model in the domain description above.

Questions to consider before writing the prior:
- Where in the search box are the model's low (good) regions, how many are there, and how do their depths compare?
- Which parameters does the model value depend on most strongly, and which interact?
- How much should an approximately calibrated model's minimum locations be trusted?
