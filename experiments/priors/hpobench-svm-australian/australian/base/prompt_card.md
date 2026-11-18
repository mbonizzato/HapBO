# Prompt A -- LLM as an expert.

## 0. Prompt
You are an expert in the domain described below. Using your knowledge of this
field, generate a prior belief over the location of the optimum for the
Bayesian optimization problem defined by the search space and optimization
description below. Write the belief only in the closed PSL vocabulary you were
given; never invent tokens.

## 1. Domain
System under study: hyperparameter optimization of a scikit-learn SVM classifier with an RBF kernel on a credit-application approval task (under 1,000 records, about 15 mixed numeric and categorical features, 2 roughly balanced classes). The two knobs are c, the SVM penalty parameter usually written C (larger c means weaker regularization), and gamma, the RBF kernel coefficient (larger gamma means a narrower kernel), both on a log scale. General SVM hyperparameter-tuning knowledge applies.

## 2. Search space
Parameter 1
name: c
type: ordinal
values: [0.0009765625, 0.001953125, 0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]  # exact allowed values
meaning: SVM penalty parameter (larger means weaker regularization); values are log-spaced

Parameter 2
name: gamma
type: ordinal
values: [0.0009765625, 0.001953125, 0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]  # exact allowed values
meaning: RBF kernel coefficient (larger means a narrower kernel); values are log-spaced

## 3. Optimization description
Objective: mean validation accuracy (fraction correct) on a held-out split, averaged over several random seeds; higher is better.
Direction: maximize
Evaluation budget: about 8 to 32 evaluations (low-budget regime).
Known constraints or prior observations: none.
