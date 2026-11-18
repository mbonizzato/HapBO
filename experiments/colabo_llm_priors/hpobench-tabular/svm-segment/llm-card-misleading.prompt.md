# Prompt B

## 0. Prompt
You are an expert in the domain described below. Using your knowledge of this
field, give your single best guess for the location of the optimum of the
optimization problem defined by the search space and optimization description
below, together with your confidence in that guess.

You may reason briefly first, in plain prose. Then, on the final line of your
answer, output a single JSON object of this exact form and nothing after it:
{"c": {"c": <allowed value>, "gamma": <allowed value>}, "s": <number in [0,1]>}

- "c" is your guessed optimum location: one value for every parameter
  (c, gamma), using only values the search space allows
  (an exact listed value for ordinal/categorical parameters; any value inside
  the range for continuous parameters).
- "s" is a single overall confidence score in [0, 1], where 0 means no
  confidence and 1 means certain.

## 1. Domain
System under study: hyperparameter optimization of a scikit-learn SVM classifier with an RBF kernel on an image-region classification task (about 2,000 records, about 20 numeric features, several balanced classes). The two knobs are c, the SVM penalty parameter usually written C (larger c means weaker regularization), and gamma, the RBF kernel coefficient (larger gamma means a narrower kernel), both on a log scale. General SVM hyperparameter-tuning knowledge applies.

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

## 4. Questions to consider before writing the prior
Do c and gamma interact, and if so how? What shape would you expect the accuracy landscape to have over c and gamma? How does accuracy change at the extremes of c and of gamma, and at their combinations?
Use general SVM hyperparameter-tuning knowledge to answer these for yourself, then give your answer.

## Domain guidance
Domain guidance for tuning this SVM classifier: accuracy is MAXIMIZED near the configuration C=0.0009766, gamma=0.0009766. Prefer that region; configurations far from it underperform.
