#!/usr/bin/env python3
"""Build the cortical human-expert-map prior from the expert's 8x4 map.

The expert map (average across animals, following the electrode map: top row =
channels 8,15,31,24 = rostrocaudal index 1) is encoded as 32 exact anchors over
the 8x4 M1 array and committed per muscle target (the map is topography-general,
so the same map is committed under each target to satisfy the per-target guard).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from prior.landscape.landscape_generator import (  # noqa: E402
    EXPERIMENT_PRIORS_DIRECTORY,
    finalize_prior_experiment,
    initialize_prior_experiment,
    target_path_component,
)

# rows = rostrocaudal 1..8 (top = rostral = ch 8,15,31,24); cols = mediolateral 1..4
EXPERT_MAP = [
    [1.000, 0.728, 0.857, 1.000],
    [0.992, 0.451, 0.660, 0.742],
    [0.679, 0.370, 0.538, 0.561],
    [0.557, 0.299, 0.468, 0.521],
    [0.343, 0.082, 0.210, 0.326],
    [0.326, 0.137, 0.032, 0.233],
    [0.167, 0.051, 0.015, 0.384],
    [0.127, 0.000, 0.000, 0.384],
]
DATASET_ID = "neuromosaic-dataset1-rat"
PRIOR_ID = "human-expert-map"
MUSCLES = ["LeftKneeFlexor", "LeftTibialisAnterioris", "RightTibialisAnterioris", "LeftMedialisGastrocnemius",
           "RightMedialisGastrocnemius", "LeftKneeExtensor", "LeftGluteus", "RightGluteus"]

# The record and card texts below are written verbatim into the committed prior files, whose hashes cover them;
# "modality c" in those texts labels this human-expert-map prior.


def build_record() -> dict:
    anchors = [
        {
            "coords": {
                "electrode_mediolateral_index": c + 1,
                "electrode_rostrocaudal_index": r + 1,
            },
            "value": float(EXPERT_MAP[r][c]),
        }
        for r in range(8)
        for c in range(4)
    ]
    return {
        "schema": {
            "objective": "maximize",
            "dimensions": [
                {"name": "electrode_mediolateral_index", "type": "ordinal", "values": [1, 2, 3, 4],
                 "meaning": "medio-lateral array index; 1=most medial, 4=most lateral"},
                {"name": "electrode_rostrocaudal_index", "type": "ordinal", "values": [1, 2, 3, 4, 5, 6, 7, 8],
                 "meaning": "rostro-caudal array index; 1=most rostral (top row: ch 8,15,31,24)"},
            ],
        },
        "beliefs": [
            {"token": "baseline", "level": 0.05},
            {"token": "anchors", "provenance": "user_stated", "bandwidth": 0.05, "points": anchors,
             "note": "Human expert per-electrode average response across previous animals (8x4 map)."},
            {"token": "noise", "level": "high", "repeats_recommended": True},
            {"token": "smoothness", "level": "smooth"},
            {"token": "modality", "expectation": "few"},
        ],
        "trust": {
            "overall_confidence": "high",
            "notes": "Full human-expert prior: a per-electrode map averaged across several previous "
            "animals, provided directly by the expert (modality c of the elicitation comparison).",
        },
    }


_CARD = """# Modality-(c) full human-expert cortical prior

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
"""


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    record = build_record()
    for muscle in MUSCLES:
        if (EXPERIMENT_PRIORS_DIRECTORY / DATASET_ID / target_path_component(muscle) / PRIOR_ID).exists():
            continue
        prior_dir = initialize_prior_experiment(
            DATASET_ID, PRIOR_ID,
            subject_scope="dataset-level human expert (average across animals)",
            target=f"cortical MEP: {muscle}",
            target_name=muscle,
            card_human_statement=True,  # the card and anchors are the expert's own map
        )
        (prior_dir / "prompt_card.md").write_text(_CARD, encoding="utf-8", newline="\n")
        (prior_dir / "prior.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
        )
        finalize_prior_experiment(prior_dir, generate_landscape=False)
        print(f"committed {prior_dir.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
