# Prompt A -- LLM as an expert.

## 0. Prompt
You are an expert in the domain described below. Using your knowledge of this
field, generate a prior belief over the location of the optimum for the
Bayesian optimization problem defined by the search space and optimization
description below. Write the belief only in the closed PSL vocabulary you were
given; never invent tokens.

## 1. Domain
System under study: hyperparameter optimization of an XGBoost gradient-boosted-tree classifier on a donor-return prediction task (under 1,000 records, a handful of numeric features, 2 moderately imbalanced classes). The four knobs are the learning rate eta (log), max_depth (log), colsample_bytree (per-tree feature-subsampling fraction), and the L2 leaf regularization reg_lambda (log). General gradient-boosted-tree hyperparameter-tuning knowledge applies.

## 2. Search space
Parameter 1
name: colsample_bytree
type: ordinal
values: [0.10000000149011612, 0.20000000298023224, 0.30000001192092896, 0.4000000059604645, 0.5, 0.6000000238418579, 0.699999988079071, 0.800000011920929, 0.8999999761581421, 1]  # exact allowed values
meaning: per-tree feature-subsampling fraction

Parameter 2
name: eta
type: ordinal
values: [0.0009765625, 0.0021094917319715023, 0.004556754138320684, 0.009843133389949799, 0.02126234397292137, 0.045929204672575, 0.09921256452798843, 0.21431098878383636, 0.4629373550415039, 1]  # exact allowed values
meaning: boosting learning rate; values are log-spaced

Parameter 3
name: max_depth
type: ordinal
values: [1, 2, 3, 5, 8, 13, 20, 32, 50]  # exact allowed values
meaning: maximum tree depth; values are log-spaced

Parameter 4
name: reg_lambda
type: ordinal
values: [0.0009765625, 0.004556754138320684, 0.02126234397292137, 0.09921256452798843, 0.4629373550415039, 2.1601195335388184, 10.079368591308594, 47.0315055847168, 219.45445251464844, 1024]  # exact allowed values
meaning: L2 leaf regularization; values are log-spaced

## 3. Optimization description
Objective: mean validation accuracy (fraction correct) on a held-out split, averaged over several random seeds; higher is better.
Direction: maximize
Evaluation budget: about 8 to 32 evaluations (low-budget regime).
Known constraints or prior observations: none.
