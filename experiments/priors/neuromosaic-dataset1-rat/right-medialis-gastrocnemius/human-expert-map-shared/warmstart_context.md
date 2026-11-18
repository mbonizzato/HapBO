## 1. Domain
Intracortical microstimulation (ICMS) of the rat primary motor cortex (M1). Stimulation is delivered through one electrode of a chronically implanted planar microelectrode array; single-electrode stimulation trains evoke motor responses recorded as motor-evoked potentials (MEPs) by EMG.

Array geometry: a 4-column by 8-row rectangular grid of 32 electrodes. The 4 columns are spaced 0.375 mm apart along the medio-lateral axis; the 8 rows are spaced 0.250 mm apart along the rostro-caudal axis. Index conventions: medio-lateral index 1 = most medial, 4 = most lateral; rostro-caudal index 1 = most rostral (anterior), 8 = most caudal (posterior).

Stereotaxic placement: the most antero-medial electrode (medio-lateral index 1, rostro-caudal index 1) sits at 1.1 mm posterior and 1.3 mm lateral to bregma. The array extends caudally (to 1.1 + 7*0.250 = ~2.85 mm posterior) and laterally (to 1.3 + 3*0.375 = ~2.4 mm lateral) from that corner.

Evoked responses reflect the underlying primary motor cortex excitability.

## 3. Optimization description
Optimization goal: identify the single electrode (one medio-lateral index 1-4 and one rostro-caudal index 1-8) that maximizes the evoked motor response amplitude, within a small stimulation budget (8 to 32 stimulations).

## Human expert map
A human expert provided the average per-electrode response across several previous animals (normalized; medio-lateral index 1 = most medial; rostro-caudal index 1 = most rostral).

| medio-lateral index | rostro-caudal index | average response |
|---|---|---|
| 1 | 1 | 1.000 |
| 2 | 1 | 0.728 |
| 3 | 1 | 0.857 |
| 4 | 1 | 1.000 |
| 1 | 2 | 0.992 |
| 2 | 2 | 0.451 |
| 3 | 2 | 0.660 |
| 4 | 2 | 0.742 |
| 1 | 3 | 0.679 |
| 2 | 3 | 0.370 |
| 3 | 3 | 0.538 |
| 4 | 3 | 0.561 |
| 1 | 4 | 0.557 |
| 2 | 4 | 0.299 |
| 3 | 4 | 0.468 |
| 4 | 4 | 0.521 |
| 1 | 5 | 0.343 |
| 2 | 5 | 0.082 |
| 3 | 5 | 0.210 |
| 4 | 5 | 0.326 |
| 1 | 6 | 0.326 |
| 2 | 6 | 0.137 |
| 3 | 6 | 0.032 |
| 4 | 6 | 0.233 |
| 1 | 7 | 0.167 |
| 2 | 7 | 0.051 |
| 3 | 7 | 0.015 |
| 4 | 7 | 0.384 |
| 1 | 8 | 0.127 |
| 2 | 8 | 0.000 |
| 3 | 8 | 0.000 |
| 4 | 8 | 0.384 |
