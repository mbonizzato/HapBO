# Modality-(c) full human-expert cortical prior

A human expert provided the average per-electrode response across several previous
animals, following the electrode map (top row = channels 8,15,31,24 = rostrocaudal
index 1). Encoded as 32 exact anchors over the 8x4 M1 array. No LLM was used.

## Search space
name: electrode_mediolateral_index
type: ordinal
values: [1, 2, 3, 4]  # exact allowed values; 1 = most medial, 4 = most lateral

name: electrode_rostrocaudal_index
type: ordinal
values: [1, 2, 3, 4, 5, 6, 7, 8]  # exact allowed values; 1 = most rostral (top row), 8 = most caudal
