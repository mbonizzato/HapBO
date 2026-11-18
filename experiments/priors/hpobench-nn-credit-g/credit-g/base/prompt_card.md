# Prompt A -- LLM as an expert.

## 0. Prompt
You are an expert in the domain described below. Using your knowledge of this
field, generate a prior belief over the location of the optimum for the
Bayesian optimization problem defined by the search space and optimization
description below. Write the belief only in the closed PSL vocabulary you were
given; never invent tokens.

## 1. Domain
System under study: hyperparameter optimization of a scikit-learn multilayer-perceptron classifier (ReLU activations, Adam optimizer, a fixed number of training iterations) on a credit-risk classification task (about 1,000 records, about 20 mixed numeric and categorical features, 2 moderately imbalanced classes). The five knobs are network depth and width, the Adam initial learning rate, the L2 penalty alpha, and the mini-batch size. General multilayer-perceptron hyperparameter-tuning knowledge applies.

## 2. Search space
Parameter 1
name: alpha
type: ordinal
values: [9.99999993922529e-09, 7.74263710923151e-08, 5.994842240397702e-07, 4.641588930098806e-06, 3.5938137443736196e-05, 0.00027825593133457005, 0.002154434798285365, 0.01668100617825985, 0.1291549652814865, 1]  # exact allowed values
meaning: L2 penalty on the weights; values are log-spaced

Parameter 2
name: batch_size
type: ordinal
values: [4, 6, 10, 16, 25, 40, 64, 101, 161, 256]  # exact allowed values
meaning: samples per Adam update; values are log-spaced

Parameter 3
name: depth
type: ordinal
values: [1, 2, 3]  # exact allowed values
meaning: number of hidden layers, all of equal width

Parameter 4
name: learning_rate_init
type: ordinal
values: [9.999999747378752e-06, 3.5938137443736196e-05, 0.00012915497063659132, 0.00046415888937190175, 0.0016681005945429206, 0.005994842387735844, 0.02154434658586979, 0.07742636650800705, 0.2782559394836426, 1]  # exact allowed values
meaning: initial Adam step size; values are log-spaced

Parameter 5
name: width
type: ordinal
values: [16, 25, 40, 64, 101, 161, 256, 406, 645, 1024]  # exact allowed values
meaning: neurons per hidden layer; values are log-spaced

## 3. Optimization description
Objective: mean validation accuracy (fraction correct) on a held-out split, averaged over several random seeds; higher is better.
Direction: maximize
Evaluation budget: about 8 to 32 evaluations (low-budget regime).
Known constraints or prior observations: none.
