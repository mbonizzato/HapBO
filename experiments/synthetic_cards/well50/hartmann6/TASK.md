# Prompt A -- LLM as an expert.

## 0. Prompt
You are an expert in the domain described below. Using your knowledge of this
field, generate a prior belief over the location of the optimum for the
Bayesian optimization problem defined by the search space and optimization
description below. Write the belief only in the closed PSL vocabulary you were
given; never invent tokens.

## 1. Domain
A 6-parameter black-box objective, to be MINIMIZED; each input ranges [0, 1]. No expert intuition about the location of the optimum is available. An approximate analytic model of the objective is available. Its functional form is believed to describe the objective's general shape, but its parameter values are only approximately calibrated (fidelity unknown), so the locations and relative depths of its minima may differ from those of the true objective. The model is:

F(x) = - sum_{k=1..4} alpha_k * exp( - sum_{j=0..5} A[k][j] * (x_j - P[k][j])^2 )
alpha = [1.332, 2.115, 2.047, 1.291]
A (4 rows x 6 cols):
j=0  j=1  j=2  j=3  j=4  j=5
9.489  3.496  4.25  1.474  2.453  12.84
0.07788  14.29  9.547  0.1509  9.549  20.58
4.276  5.36  2.46  11.89  11.07  14.12
5.729  7.313  0.07137  8.335  0.1789  16.75
P (4 rows x 6 cols):
j=0  j=1  j=2  j=3  j=4  j=5
0.4704  0.2377  0.5354  0.2771  0.5081  0.4092
0.4984  0.4457  0.866  0.106  0.04525  0.98
0.5164  0.4227  0.2563  0.6663  0.5327  0.8223
0.3355  0.98  0.4912  0.2166  0.04525  0.03393

## 2. Search space
Parameter 1
name: x0
type: continuous
range: [0.0, 1.0]  # [low, high]
scale: linear

Parameter 2
name: x1
type: continuous
range: [0.0, 1.0]  # [low, high]
scale: linear

Parameter 3
name: x2
type: continuous
range: [0.0, 1.0]  # [low, high]
scale: linear

Parameter 4
name: x3
type: continuous
range: [0.0, 1.0]  # [low, high]
scale: linear

Parameter 5
name: x4
type: continuous
range: [0.0, 1.0]  # [low, high]
scale: linear

Parameter 6
name: x5
type: continuous
range: [0.0, 1.0]  # [low, high]
scale: linear

## 3. Optimization description
Objective: a scalar black-box value, lower is better.
Direction: minimize
Evaluation budget: about 40 evaluations.
Known constraints or prior observations: none; the only information available is the approximate analytic model in the domain description above.

Questions to consider before writing the prior:
- Where in the search box are the model's low (good) regions, how many are there, and how do their depths compare?
- Which parameters does the model value depend on most strongly, and which interact?
- How much should an approximately calibrated model's minimum locations be trusted?
