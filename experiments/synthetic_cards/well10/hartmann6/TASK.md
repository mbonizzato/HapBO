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
alpha = [1.073, 1.402, 2.822, 2.779]
A (4 rows x 6 cols):
j=0  j=1  j=2  j=3  j=4  j=5
180.8  56.85  256.2  55.82  34.12  165.8
1.027  200.1  280.8  2.034  152.5  282.5
59.99  71.49  34.15  190.4  286.9  170.9
265.4  143.5  1  176.1  2.146  267
P (4 rows x 6 cols):
j=0  j=1  j=2  j=3  j=4  j=5
0.3108  0.2053  0.6592  0.02  0.7774  0.5599
0.3962  0.4413  0.9455  0.2177  0.06195  0.98
0.1905  0.1908  0.8044  0.02272  0.02791  0.3005
0.4942  0.98  0.896  0.3985  0.04525  0.03393

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
