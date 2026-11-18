# Prompt A -- LLM as an expert.

## 0. Prompt
You are an expert in the domain described below. Using your knowledge of this
field, generate a prior belief over the location of the optimum for the
Bayesian optimization problem defined by the search space and optimization
description below. Write the belief only in the closed PSL vocabulary you were
given; never invent tokens.

## 1. Domain
Concentrated solar power (CSP) engineering: central-receiver (power tower) plants with molten salt as heat transfer fluid; specifically cavity-receiver thermal design.

System under study: A molten-salt CSP tower plant. A radially staggered field of sun-tracking heliostats reflects direct normal irradiance (assumed 1 kW/m2) onto a cavity receiver with a rectangular aperture at the top of a tower; molten salt is pumped through an array of tubes lining the inner cavity surface and absorbs the concentrated flux, while heat is lost back to the environment by reflection, re-radiation, and conduction through the insulated receiver wall. The simulation covers only the heliostat field and the receiver unit over a 24-hour period, so the design task is to shape the cavity aperture, the tube bundle, and the salt outlet temperature so that as large a fraction as possible of the intercepted solar flux ends up in the salt. Radiative transfer is computed by ray tracing, which makes the objective and two of the constraints stochastic.

## 2. Search space
Parameter 1
name: receiver_aperture_height_m
type: continuous
range: [1, 30]  # [low, high]
scale: linear
meaning: vertical size of the rectangular cavity opening; a taller aperture intercepts more of the reflected beam (less spillage) but increases re-radiation and other thermal losses out of the cavity (m)

Parameter 2
name: receiver_aperture_width_m
type: continuous
range: [1, 30]  # [low, high]
scale: linear
meaning: horizontal size of the cavity opening; trades spillage capture against radiative loss, and caps how much tube bundle fits inside the receiver (constraint c5) (m)

Parameter 3
name: receiver_outlet_temperature_k
type: continuous
range: [793, 995]  # [low, high]
scale: linear
meaning: target temperature of the salt leaving the receiver; affects re-radiation and conduction losses and the salt mass flow (K)

Parameter 4
name: receiver_tube_count
type: continuous
range: [1, 8567]  # [low, high]
scale: log
meaning: number of parallel salt tubes over the cavity surface; more tubes add heat-transfer area and lower the per-tube flow rate, but crowd the aperture and change pressure drop and pumping power (count, integer-valued)

Parameter 5
name: receiver_insulation_m
type: continuous
range: [0.01, 5]  # [low, high]
scale: log
meaning: thickness of the receiver-wall insulation; affects conduction losses through the receiver wall and adds material cost against the plant-cost constraint (m)

Parameter 6
name: receiver_tube_inner_diameter_m
type: continuous
range: [0.005, 0.1]  # [low, high]
scale: log
meaning: internal bore of the receiver tubes; sets salt velocity, convective heat-transfer coefficient, pressure drop, and therefore the pumping share of the parasitic load (m)

Parameter 7
name: receiver_tube_outer_diameter_m
type: continuous
range: [0.0055, 0.1]  # [low, high]
scale: log
meaning: external diameter of the receiver tubes; with the inner diameter it sets the wall thickness that must withstand the internal salt pressure, and with the tube count it sets how much of the aperture width the bundle occupies (m)

## 3. Optimization description
Objective: receiver efficiency (energy transferred to the molten salt, aggregated over the 24-hour simulation); higher is better.
Direction: maximize
Evaluation budget: about 30 to 50 evaluations.
Known constraints or prior observations: six constraints, all of which must hold:
c1: total plant cost does not exceed $45M (simulated, deterministic).
c2: receiver-tube pressure does not exceed the steel yield pressure (simulated, stochastic).
c3: tube inner diameter does not exceed outer diameter (known before simulation).
c4: receiver outlet temperature is at least the turbine inlet temperature (simulated, deterministic).
c5: the tube bundle fits inside the receiver aperture, tube count x outer diameter <= (pi/2) x aperture width (known before simulation).
c6: parasitic losses remain below 3% of generated output (simulated, stochastic).
Designs violating c3 or c5 are rejected before simulation. Some designs also crash the simulator and return no valid output.

## 4. Questions to consider before writing the prior
How noisy do you expect the efficiency estimate to be, and does its noise level vary across the design space? What fraction of the design space do you expect to satisfy all constraints, and where do you expect infeasible or crashing designs to concentrate? Which constraints are likely to be binding for high-efficiency designs? For each parameter, how would you expect efficiency to change across its range? Which parameters are likely to interact most strongly?
Use general CSP receiver-design knowledge to answer these for yourself, then write the prior.
