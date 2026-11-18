# Task
Write a prior belief over the location of the optimum for the Bayesian optimization problem below, using only the closed PSL vocabulary you were given; never invent tokens.

## 1. Domain
Intracortical microstimulation (ICMS) of the rat primary motor cortex (M1). Stimulation is delivered through one electrode of a chronically implanted planar microelectrode array; single-electrode stimulation trains evoke motor responses recorded as motor-evoked potentials (MEPs) by EMG.

Array geometry: a 4-column by 8-row rectangular grid of 32 electrodes. The 4 columns are spaced 0.375 mm apart along the medio-lateral axis; the 8 rows are spaced 0.250 mm apart along the rostro-caudal axis. Index conventions: medio-lateral index 1 = most medial, 4 = most lateral; rostro-caudal index 1 = most rostral (anterior), 8 = most caudal (posterior).

Stereotaxic placement: the most antero-medial electrode (medio-lateral index 1, rostro-caudal index 1) sits at 1.1 mm posterior and 1.3 mm lateral to bregma. The array extends caudally (to 1.1 + 7*0.250 = ~2.85 mm posterior) and laterally (to 1.3 + 3*0.375 = ~2.4 mm lateral) from that corner.

Evoked responses reflect the underlying primary motor cortex excitability.

## 2. Search space
Parameter 1
name: electrode_mediolateral_index
type: ordinal
values: [1, 2, 3, 4]  # exact allowed values

Parameter 2
name: electrode_rostrocaudal_index
type: ordinal
values: [1, 2, 3, 4, 5, 6, 7, 8]  # exact allowed values

## 3. Optimization description
Optimization goal: identify the single electrode (one medio-lateral index 1-4 and one rostro-caudal index 1-8) that maximizes the evoked motor response amplitude, within a small stimulation budget (8 to 32 stimulations).

Use only integer index values (medio-lateral in {1,2,3,4}; rostro-caudal in {1..8}); never fractional values; do not use the 'categorical' token (these axes are ordered).
