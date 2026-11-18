# Prompt A -- LLM as an expert.

## 0. Prompt
You are an expert in the domain described below. Using your knowledge of this
field, generate a prior belief over the location of the optimum for the
Bayesian optimization problem defined by the search space and optimization
description below. Write the belief only in the closed PSL vocabulary you were
given; never invent tokens.

## 1. Domain
A 15-parameter black-box objective, to be MINIMIZED; each input ranges [-32.768, 32.768]. No expert intuition about the location of the optimum is available. An approximate analytic model of the objective is available. Its functional form is believed to describe the objective's general shape, but its parameter values are only approximately calibrated (fidelity unknown), so the locations and relative depths of its minima may differ from those of the true objective. The model is:

F(x) = -20 * exp( -0.2638 * sqrt( (1/15) * sum_i (x_i - c_i)^2 ) ) - exp( (1/15) * sum_i cos( kap_i * 2*pi * (x_i - c_i) ) ) + 20 + e
per-parameter constants:
c_i  kap_i
18.41  1.261
-32.27  0.687
-13.99  1.133
30.64  1.036
-30.13  1.105
-24.35  1.207
-5.012  0.7128
15.68  0.9572
-11.84  0.6879
4.693  0.8637
10.45  1.175
-18.44  0.8006
9.261  0.9088
-4.679  1.314
-19.54  1.3

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
