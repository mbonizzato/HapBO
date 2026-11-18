# Prompt A -- LLM as an expert.

## 0. Prompt
You are an expert in the domain described below. Using your knowledge of this
field, generate a prior belief over the location of the optimum for the
Bayesian optimization problem defined by the search space and optimization
description below. Write the belief only in the closed PSL vocabulary you were
given; never invent tokens.

## 1. Domain
A small convolutional image classifier trained from scratch. The network has two
convolutional blocks (each: Conv 3x3 -> batch-norm -> ReLU -> 2x2 max-pool),
followed by adaptive average pooling, a dropout layer, and a single linear
classification head. It is trained with the Adam optimizer for many epochs (until
the training loss has largely converged) on a SMALL dataset of only a few thousand
low-resolution grayscale images spanning roughly ten balanced classes. The
objective is validation-set classification accuracy.

## 2. Search space
Parameter 1
name: learning_rate
type: ordinal
values: [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]  # exact allowed values
scale: log
units: unitless
role: Adam optimizer learning rate

Parameter 2
name: base_channels
type: ordinal
values: [8, 16, 24, 32, 48]  # exact allowed values
scale: linear
units: unitless
role: number of channels in the first conv block (the second block uses twice this)

Parameter 3
name: dropout
type: ordinal
values: [0.0, 0.1, 0.25, 0.4, 0.5]  # exact allowed values
scale: linear
units: unitless
role: dropout probability before the linear head

Parameter 4
name: weight_decay
type: ordinal
values: [0.0, 1e-05, 0.0001, 0.001, 0.01]  # exact allowed values
scale: log
units: unitless
role: L2 weight decay coefficient

## 3. Optimization description
Objective: validation-set classification accuracy (higher is better).
Direction: maximize.
Evaluation budget: approximately 30 configurations.
