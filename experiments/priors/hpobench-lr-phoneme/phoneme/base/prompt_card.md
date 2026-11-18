# Prompt A -- LLM as an expert.

## 0. Prompt
You are an expert in the domain described below. Using your knowledge of this
field, generate a prior belief over the location of the optimum for the
Bayesian optimization problem defined by the search space and optimization
description below. Write the belief only in the closed PSL vocabulary you were
given; never invent tokens.

## 1. Domain
System under study: hyperparameter optimization of a scikit-learn SGD-trained logistic-regression classifier (log-loss, L2 penalty) on a speech-sound classification task (about 5,000 records, a handful of numeric features, 2 moderately imbalanced classes). The two knobs are the L2 regularization strength alpha and the initial SGD learning rate eta0, both on a log scale. General logistic-regression hyperparameter-tuning knowledge applies.

## 2. Search space
Parameter 1
name: alpha
type: ordinal
values: [9.999999747378752e-06, 1.6155980119947344e-05, 2.6101572075276636e-05, 4.2169649532297626e-05, 6.812920764787123e-05, 0.00011006941349478438, 0.00017782794020604342, 0.0002872984914574772, 0.00046415888937190175, 0.0007498941849917173, 0.0012115276185795665, 0.0019573417957872152, 0.003162277629598975, 0.005108969751745462, 0.008254041895270348, 0.013335213996469975, 0.02154434658586979, 0.034807004034519196, 0.05623413249850273, 0.09085175395011902, 0.14677992463111877, 0.23713737726211548, 0.3831186890602112, 0.6189658045768738, 1]  # exact allowed values
meaning: L2 regularization strength; values are log-spaced

Parameter 2
name: eta0
type: ordinal
values: [9.999999747378752e-06, 1.6155980119947344e-05, 2.6101572075276636e-05, 4.2169649532297626e-05, 6.812920764787123e-05, 0.00011006941349478438, 0.00017782794020604342, 0.0002872984914574772, 0.00046415888937190175, 0.0007498941849917173, 0.0012115276185795665, 0.0019573417957872152, 0.003162277629598975, 0.005108969751745462, 0.008254041895270348, 0.013335213996469975, 0.02154434658586979, 0.034807004034519196, 0.05623413249850273, 0.09085175395011902, 0.14677992463111877, 0.23713737726211548, 0.3831186890602112, 0.6189658045768738, 1]  # exact allowed values
meaning: initial SGD learning rate; values are log-spaced

## 3. Optimization description
Objective: mean validation accuracy (fraction correct) on a held-out split, averaged over several random seeds; higher is better.
Direction: maximize
Evaluation budget: about 8 to 32 evaluations (low-budget regime).
Known constraints or prior observations: none.
