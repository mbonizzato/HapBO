#!/usr/bin/env python3
"""Write the LLM-visible texts of the cortical (rat M1) and SOLAR problems.

Cortical (experiments/priors/neuromosaic-dataset1-rat/<muscle>/, eight muscle targets). Every
method of a condition receives the same shared task text (Domain + goal + the condition's
facts); each method adds only its own output-format instruction:

  llm-geometry/             shared text = Domain + goal (array geometry and placement only); written for
                            the three muscles of the flexor benchmark only (GEOMETRY_MUSCLES)
  llm-expert-hint/          shared text = Domain + goal + a verbatim expert statement
  human-expert-map-shared/  shared text = Domain + goal + the human expert map (32 anchors)
  human-expert-map/warmstart_context.md
                            Domain + goal + stereotaxic facts + the human expert map

For each of the first three, prompt_card.md is HapBO's card (one-line PSL instruction + the shared
text + search space + PSL format line) and warmstart_context.md is the shared text read by LGBO,
LLAMBO, BORA and ColaBO. human-expert-map-shared is also finalized as a prior directory holding the
human-made anchors of human-expert-map (no LLM); its card is documentation only.

SOLAR: experiments/priors/solar-maxeff-re/expert/prompt_card.md (MAXEFF_RE receiver design card with
open reasoning questions), read by HapBO and every SOLAR baseline.

Inputs: experiments/elicitation/cortical-flexor/{domain.txt,schema.json} and the human-expert-map
prior.json of each muscle (scripts/build_cortical_expert_prior.py).

usage: python scripts/build_cortical_solar_cards.py [--root <repo>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from prior.expert_knowledge.prompt_card import render_prompt_card, search_space_from_schema  # noqa: E402

MUSCLES = ["left-knee-flexor", "left-tibialis-anterioris", "right-tibialis-anterioris", "left-medialis-gastrocnemius",
           "right-medialis-gastrocnemius", "left-knee-extensor", "left-gluteus", "right-gluteus"]
# the cortical/llm-geometry study runs on neuromosaic-dataset1-rat-flexor (these three muscles)
GEOMETRY_MUSCLES = ("left-knee-flexor", "left-tibialis-anterioris", "right-tibialis-anterioris")

GOAL = ("Optimization goal: identify the single electrode (one medio-lateral index 1-4 and one rostro-caudal "
        "index 1-8) that maximizes the evoked motor response amplitude, within a small stimulation budget "
        "(8 to 32 stimulations).")
STEREOTAXIC = ("Stereotaxic facts: the most excitable core of rat primary motor cortex lies near or anterior to "
               "bregma and within a few mm lateral to the midline. This array's rostro-caudal index 1 to 8 spans "
               "AP -1.1 to -2.85 mm (posterior to bregma, 0.250 mm steps); its medio-lateral index 1 to 4 spans "
               "ML 1.3 to 2.4 mm (0.375 mm steps).")
EXPERT_QUOTE = ("A domain expert who has run similar experiments on this preparation states: \"Usually the top row "
                "is the most effective, and the two outer columns of the array are the most effective.\" (In the "
                "expert's array drawing, the top row is rostro-caudal index 1.)")
INSTRUCTION = ("Write a prior belief over the location of the optimum for the Bayesian optimization problem below, "
               "using only the closed PSL vocabulary you were given; never invent tokens.")
FORMAT = ("Use only integer index values (medio-lateral in {1,2,3,4}; rostro-caudal in {1..8}); never fractional "
          "values; do not use the 'categorical' token (these axes are ordered).")

SOLAR_DOMAIN = (
    "Concentrated solar power (CSP) engineering: central-receiver (power tower) plants with molten salt "
    "as heat transfer fluid; specifically cavity-receiver thermal design.\n\n"
    "System under study: A molten-salt CSP tower plant. A radially staggered field of sun-tracking "
    "heliostats reflects direct normal irradiance (assumed 1 kW/m2) onto a cavity receiver with a "
    "rectangular aperture at the top of a tower; molten salt is pumped through an array of tubes lining "
    "the inner cavity surface and absorbs the concentrated flux, while heat is lost back to the "
    "environment by reflection, re-radiation, and conduction through the insulated receiver wall. The "
    "simulation covers only the heliostat field and the receiver unit over a 24-hour period, so the design "
    "task is to shape the cavity aperture, the tube bundle, and the salt outlet temperature so that as large "
    "a fraction as possible of the intercepted solar flux ends up in the salt. Radiative transfer is "
    "computed by ray tracing, which makes the objective and two of the constraints stochastic."
)
SOLAR_PARAMS = [
    ("receiver_aperture_height_m", [1, 30], "linear", "m",
     "vertical size of the rectangular cavity opening; a taller aperture intercepts more of the reflected "
     "beam (less spillage) but increases re-radiation and other thermal losses out of the cavity"),
    ("receiver_aperture_width_m", [1, 30], "linear", "m",
     "horizontal size of the cavity opening; trades spillage capture against radiative loss, and caps how "
     "much tube bundle fits inside the receiver (constraint c5)"),
    ("receiver_outlet_temperature_k", [793, 995], "linear", "K",
     "target temperature of the salt leaving the receiver; affects re-radiation and conduction losses and "
     "the salt mass flow"),
    ("receiver_tube_count", [1, 8567], "log", "count, integer-valued",
     "number of parallel salt tubes over the cavity surface; more tubes add heat-transfer area and lower the "
     "per-tube flow rate, but crowd the aperture and change pressure drop and pumping power"),
    ("receiver_insulation_m", [0.01, 5], "log", "m",
     "thickness of the receiver-wall insulation; affects conduction losses through the receiver wall and adds "
     "material cost against the plant-cost constraint"),
    ("receiver_tube_inner_diameter_m", [0.005, 0.1], "log", "m",
     "internal bore of the receiver tubes; sets salt velocity, convective heat-transfer coefficient, pressure "
     "drop, and therefore the pumping share of the parasitic load"),
    ("receiver_tube_outer_diameter_m", [0.0055, 0.1], "log", "m",
     "external diameter of the receiver tubes; with the inner diameter it sets the wall thickness that must "
     "withstand the internal salt pressure, and with the tube count it sets how much of the aperture width "
     "the bundle occupies"),
]
SOLAR_OPT = (
    "Objective: receiver efficiency (energy transferred to the molten salt, aggregated over the 24-hour "
    "simulation); higher is better.\n"
    "Direction: maximize\n"
    "Evaluation budget: about 30 to 50 evaluations.\n"
    "Known constraints or prior observations: six constraints, all of which must hold:\n"
    "c1: total plant cost does not exceed $45M (simulated, deterministic).\n"
    "c2: receiver-tube pressure does not exceed the steel yield pressure (simulated, stochastic).\n"
    "c3: tube inner diameter does not exceed outer diameter (known before simulation).\n"
    "c4: receiver outlet temperature is at least the turbine inlet temperature (simulated, deterministic).\n"
    "c5: the tube bundle fits inside the receiver aperture, tube count x outer diameter <= (pi/2) x aperture "
    "width (known before simulation).\n"
    "c6: parasitic losses remain below 3% of generated output (simulated, stochastic).\n"
    "Designs violating c3 or c5 are rejected before simulation. Some designs also crash the simulator and "
    "return no valid output."
)
SOLAR_QUESTIONS = (
    "## 4. Questions to consider before writing the prior\n"
    "How noisy do you expect the efficiency estimate to be, and does its noise level vary across the design "
    "space? What fraction of the design space do you expect to satisfy all constraints, and where do you "
    "expect infeasible or crashing designs to concentrate? Which constraints are likely to be binding for "
    "high-efficiency designs? For each parameter, how would you expect efficiency to change across its "
    "range? Which parameters are likely to interact most strongly?\n"
    "Use general CSP receiver-design knowledge to answer these for yourself, then write the prior.\n"
)


def solar_space() -> str:
    blocks = []
    for i, (name, rng, scale, units, meaning) in enumerate(SOLAR_PARAMS, start=1):
        blocks.append("\n".join([f"Parameter {i}", f"name: {name}", "type: continuous",
                                 f"range: {rng}  # [low, high]", f"scale: {scale}",
                                 f"meaning: {meaning} ({units})"]))
    return "\n\n".join(blocks)


def expert_map_table(prior_path: Path) -> str:
    rec = json.loads(prior_path.read_text(encoding="utf-8"))
    pts = next(b["points"] for b in rec["beliefs"] if b["token"] == "anchors")
    rows = sorted(pts, key=lambda p: (p["coords"]["electrode_rostrocaudal_index"],
                                      p["coords"]["electrode_mediolateral_index"]))
    table = [f"| {p['coords']['electrode_mediolateral_index']} | {p['coords']['electrode_rostrocaudal_index']} "
             f"| {float(p['value']):.3f} |" for p in rows]
    return ("## Human expert map\n"
            "A human expert provided the average per-electrode response across several previous animals "
            "(normalized; medio-lateral index 1 = most medial; rostro-caudal index 1 = most rostral).\n\n"
            "| medio-lateral index | rostro-caudal index | average response |\n|---|---|---|\n"
            + "\n".join(table) + "\n")


def hapbo_card(shared: str, space: str) -> str:
    domain = shared.split("## 1. Domain")[1].split("## 3. Optimization description")[0].strip()
    rest = shared.split("## 3. Optimization description")[1].strip()
    return (f"# Task\n{INSTRUCTION}\n\n## 1. Domain\n{domain}\n\n## 2. Search space\n{space}\n\n"
            f"## 3. Optimization description\n{rest}\n\n{FORMAT}\n")


def finalize_shared_prior(pr: Path, muscle: str) -> None:
    """human-expert-map-shared as a finalized prior directory (same human-made anchors, no LLM)."""
    from prior.landscape.landscape_generator import finalize_prior_experiment

    src, dst = pr / muscle / "human-expert-map", pr / muscle / "human-expert-map-shared"
    if (dst / "metadata.json").is_file():
        return
    (dst / "prior.json").write_bytes((src / "prior.json").read_bytes())
    meta = json.loads((src / "metadata.json").read_text(encoding="utf-8"))
    meta.update(status="draft", prior_id=dst.name)
    meta.pop("finalized_at_utc", None)
    meta["generation"]["notes"] = "human-expert-map anchors (human-made, no LLM); shared task text"
    (dst / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    finalize_prior_experiment(dst, generate_landscape=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT, help="repository root to write into (default: this repo)")
    ap.add_argument("--no-finalize", action="store_true",
                    help="write texts only; do not create the human-expert-map-shared prior directories")
    args = ap.parse_args()
    root = args.root.resolve()
    el = root / "experiments" / "elicitation" / "cortical-flexor"
    pr = root / "experiments" / "priors" / "neuromosaic-dataset1-rat"

    domain = (el / "domain.txt").read_text(encoding="utf-8").strip()
    space = search_space_from_schema(json.loads((el / "schema.json").read_text(encoding="utf-8"))["dimensions"])
    head = "## 1. Domain\n" + domain + "\n\n## 3. Optimization description\n"
    for m in MUSCLES:
        table = expert_map_table(pr / m / "human-expert-map" / "prior.json")
        (pr / m / "human-expert-map" / "warmstart_context.md").write_text(
            head + GOAL + "\n\n" + STEREOTAXIC + "\n\n" + table, encoding="utf-8", newline="\n")
        shared = {
            "llm-geometry": head + GOAL,
            "llm-expert-hint": head + GOAL + "\n\n" + EXPERT_QUOTE,
            "human-expert-map-shared": (head + GOAL + "\n\n" + table).strip(),
        }
        for pid, text in shared.items():
            if pid == "llm-geometry" and m not in GEOMETRY_MUSCLES:
                continue
            d = pr / m / pid
            d.mkdir(parents=True, exist_ok=True)
            (d / "prompt_card.md").write_text(hapbo_card(text, space), encoding="utf-8", newline="\n")
            (d / "warmstart_context.md").write_text(text + "\n", encoding="utf-8", newline="\n")
        if not args.no_finalize:
            finalize_shared_prior(pr, m)

    d = root / "experiments" / "priors" / "solar-maxeff-re" / "expert"
    d.mkdir(parents=True, exist_ok=True)
    card = render_prompt_card(domain=SOLAR_DOMAIN, search_space=solar_space(), optimization_description=SOLAR_OPT)
    (d / "prompt_card.md").write_text(card + "\n" + SOLAR_QUESTIONS, encoding="utf-8", newline="\n")
    print(f"wrote cortical texts ({len(MUSCLES)} muscles; llm-geometry for {len(GEOMETRY_MUSCLES)}) + SOLAR card")


if __name__ == "__main__":
    main()
