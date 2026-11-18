#!/usr/bin/env python3
"""Copy the left-knee-flexor cortical elicitations to the other muscle targets.

The cortical conditions use one prior per condition across muscles: the HapBO prior.json of
llm-expert-hint and the ColaBO (c, s) belief of human-expert-map-shared are elicited once on
left-knee-flexor and copied to the other seven muscle targets of the dataset; the HapBO prior.json
of llm-geometry, whose study runs on the flexor benchmark only, is copied to the two other muscles
of that benchmark (GEOMETRY_TARGETS). HapBO copies are re-finalized per target (the per-target metadata guard requires target-scoped
metadata). Run with HAPBO_VOCABULARY_CARD set
as during the elicitation (src/prior/expert_knowledge/vocabulary_card_neutral_example.md).

usage: python scripts/replicate_cortical_priors.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from prior.landscape.landscape_generator import finalize_prior_experiment  # noqa: E402

PR = ROOT / "experiments" / "priors" / "neuromosaic-dataset1-rat"
CB = ROOT / "experiments" / "colabo_llm_priors" / "neuromosaic-dataset1-rat"
SRC = "left-knee-flexor"
TARGETS = {"left-tibialis-anterioris": "LeftTibialisAnterioris", "right-tibialis-anterioris": "RightTibialisAnterioris",
           "left-medialis-gastrocnemius": "LeftMedialisGastrocnemius",
           "right-medialis-gastrocnemius": "RightMedialisGastrocnemius", "left-knee-extensor": "LeftKneeExtensor",
           "left-gluteus": "LeftGluteus", "right-gluteus": "RightGluteus"}
HAPBO_IDS = ("llm-geometry", "llm-expert-hint")
GEOMETRY_TARGETS = ("left-tibialis-anterioris", "right-tibialis-anterioris")
COLABO_IDS = ("human-expert-map-shared",)


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    for pid in HAPBO_IDS:
        src = PR / SRC / pid
        meta = json.loads((src / "metadata.json").read_text(encoding="utf-8"))
        assert meta["status"] == "finalized", src
        for slug, name in TARGETS.items():
            if pid == "llm-geometry" and slug not in GEOMETRY_TARGETS:
                continue
            dst = PR / slug / pid
            if (dst / "metadata.json").is_file():
                print("exists", dst)
                continue
            assert (dst / "prompt_card.md").read_bytes() == (src / "prompt_card.md").read_bytes(), dst
            shutil.copy(src / "prior.json", dst / "prior.json")
            m = json.loads(json.dumps(meta))
            m.update(status="draft")
            m.pop("finalized_at_utc", None)
            m["scope"].update(target=f"cortical MEP: {name}", target_name=name, target_slug=slug)
            m["generation"]["notes"] = ((m["generation"].get("notes") or "")
                                        + f"; replicated from {SRC} (one prior per condition)")
            (dst / "metadata.json").write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8", newline="\n")
            finalize_prior_experiment(dst, generate_landscape=False)
            print("replicated", dst)
    for mod in COLABO_IDS:
        src_json = CB / SRC / f"{mod}.json"
        if not src_json.is_file():
            print("no ColaBO belief", src_json)
            continue
        art = json.loads(src_json.read_text(encoding="utf-8"))
        for slug in TARGETS:
            (CB / slug).mkdir(parents=True, exist_ok=True)
            a = dict(art, task=slug, replicated_from=SRC)
            (CB / slug / f"{mod}.json").write_text(json.dumps(a, indent=2) + "\n", encoding="utf-8", newline="\n")
            shutil.copy(CB / SRC / f"{mod}.prompt.md", CB / slug / f"{mod}.prompt.md")
        print("replicated ColaBO belief", mod, art["c"], art["s"])


if __name__ == "__main__":
    main()
