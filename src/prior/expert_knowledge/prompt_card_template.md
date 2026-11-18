# Prompt A -- LLM as an expert. 

## 0. Prompt 
You are an expert in {{domain}}. Using your knowledge of this field, generate
a prior belief over the location of the optimum for the Bayesian optimization
problem with the following parameters: {{search_space}} and the optimization description {{optimization_description}}.

## 1. Domain
{{domain}}

System under study: {2-4 sentences describing the experimental system}

{{/domain}}


## 2. Search space 
{{search_space}}

Parameter 1
name: {parameter_name}
type: {continuous | discrete | ordinal | categorical}
range: {[low, high] | [option_1, option_2, ...]}
scale: {linear | log}
units: {units, or "unitless"}
role: {one sentence: what this parameter controls}
(repeat for all D parameters)

{{/search_space}}

## 3. Optimization description
{{optimization_description}}

Objective: {measured quantity, with units}
Direction: {maximize | minimize}
Evaluation budget: {N} evaluations
Known constraints or prior observations: {if any, else "none"}

{{/optimization_description}}