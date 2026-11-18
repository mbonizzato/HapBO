# Prompt A -- LLM as an expert.

## 0. Prompt
You are an expert in the domain described below. Using your knowledge of this
field, generate a prior belief over the location of the optimum for the
Bayesian optimization problem defined by the search space and optimization
description below. Write the belief only in the closed PSL vocabulary you were
given; never invent tokens.

## 1. Domain
System under study: hyperparameter optimization of a scikit-learn random-forest classifier on an image-region classification task (about 2,000 records, about 20 numeric features, several balanced classes). The four knobs are max_depth (log), max_features (an exponent e: each split tries round(n_features ** e) features, so 0 means one feature and 1 means all features), min_samples_leaf, and min_samples_split (log). General random-forest hyperparameter-tuning knowledge applies.

## 2. Search space
Parameter 1
name: max_depth
type: ordinal
values: [1, 2, 3, 5, 8, 13, 20, 32, 50]  # exact allowed values
meaning: maximum tree depth; values are log-spaced

Parameter 2
name: max_features
type: ordinal
values: [0, 0.1111111119389534, 0.2222222238779068, 0.3333333432674408, 0.4444444477558136, 0.5555555820465088, 0.6666666865348816, 0.7777777910232544, 0.8888888955116272, 1]  # exact allowed values
meaning: exponent e: each split tries round(n_features ** e) features (0 = one feature, 1 = all)

Parameter 3
name: min_samples_leaf
type: ordinal
values: [1, 3, 5, 7, 9, 11, 13, 15, 17, 20]  # exact allowed values
meaning: minimum number of samples per leaf

Parameter 4
name: min_samples_split
type: ordinal
values: [2, 3, 5, 8, 12, 20, 32, 50, 80, 128]  # exact allowed values
meaning: minimum number of samples required to split a node; values are log-spaced

## 3. Optimization description
Objective: mean validation accuracy (fraction correct) on a held-out split, averaged over several random seeds; higher is better.
Direction: maximize
Evaluation budget: about 8 to 32 evaluations (low-budget regime).
Known constraints or prior observations: none.
