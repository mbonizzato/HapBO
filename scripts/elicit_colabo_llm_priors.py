"""Elicit BORA-style (c, s) point priors from the local LLM for the colabo-llm baseline.

This is the offline, one-shot elicitation for the ``colabo-llm`` external baseline
(the LLM-assist ColaBO baseline, built as in the BORA paper's supplement SC.2): the
LLM is asked ONCE per (task, modality) for

  - ``c``: a guessed optimum location (one native value per search-space parameter),
  - ``s``: a confidence in [0, 1].

These become ColaBO's expert prior ``pi(x) = N(c, sigma)`` with ``sigma = 1 - s + 1e-6``
(the DefaultPrior float ``confidence`` IS the per-dimension Gaussian sigma in the
normalized [0,1] space, so the study harness passes ``confidence = sigma``).

Information shown to the LLM: IDENTICAL to what HapBO's elicitor received - the same
prompt cards, per scenario. Only the HapBO-specific PSL output instructions are replaced
by the minimal (c, s) ask; no domain information is added or removed. Inference settings:
temperature 0.0 and thinking off for every elicitation.

Modalities: HPO ``llm-card`` (the card of the experiments/studies/hpo/hapbo HapBO arm) and, with
--adversarial, ``llm-card-misleading`` (the same card + the task's misleading hint as a "Domain guidance"
section); cortical ``human-expert-map-shared`` (the shared facts-only text, i.e. the warm-start context
every method receives, plus the card's search space). Beliefs are written to
experiments/colabo_llm_priors/<benchmark>/<task>/<modality>.json with the prompt next to them; the raw
model reply goes to results/elicitation/colabo/.

Endpoint: LLM_BASE_URL, LLM_MODEL, LLM_API_KEY (OpenAI-compatible).
    python scripts/elicit_colabo_llm_priors.py [--dry-run] [--only SUBSTR] [--overwrite] [--adversarial]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
PRIORS_ROOT = _ROOT / "experiments" / "priors"
OUT_ROOT = _ROOT / "experiments" / "colabo_llm_priors"
RAW_ROOT = _ROOT / "results" / "elicitation" / "colabo"

MODEL = os.environ.get("LLM_MODEL", "Qwen3.5-122B-A10B-GPTQ-Int4")
BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
MAX_TOKENS = 4096
TIMEOUT = 300.0
EPSILON = 1e-6

CORTICAL_BENCHMARK = "neuromosaic-dataset1-rat"
CORTICAL_SLUGS = ("left-knee-flexor", "left-tibialis-anterioris", "right-tibialis-anterioris")
CORTICAL_MODALITIES = ("human-expert-map-shared",)
CORTICAL_DIMS = (
    {"name": "electrode_mediolateral_index", "type": "ordinal", "values": [1, 2, 3, 4], "scale": "linear"},
    {"name": "electrode_rostrocaudal_index", "type": "ordinal", "values": [1, 2, 3, 4, 5, 6, 7, 8], "scale": "linear"},
)
HPO_MODALITY = "llm-card"
ADVERSARIAL_MODALITY = "llm-card-misleading"
# The exact card per HPO task = the one referenced by the study manifest of the HapBO
# arm, guaranteeing identical information.
HPO_SOURCE_STUDIES = (
    _ROOT / "experiments" / "studies" / "hpo" / "hapbo" / "study.yaml",
)

def _ask_block(parameter_names: list[str], example: str) -> str:
    return (
        "## 0. Prompt\n"
        "You are an expert in the domain described below. Using your knowledge of this\n"
        "field, give your single best guess for the location of the optimum of the\n"
        "optimization problem defined by the search space and optimization description\n"
        "below, together with your confidence in that guess.\n\n"
        "You may reason briefly first, in plain prose. Then, on the final line of your\n"
        "answer, output a single JSON object of this exact form and nothing after it:\n"
        f"{example}\n\n"
        '- "c" is your guessed optimum location: one value for every parameter\n'
        f"  ({', '.join(parameter_names)}), using only values the search space allows\n"
        "  (an exact listed value for ordinal/categorical parameters; any value inside\n"
        "  the range for continuous parameters).\n"
        '- "s" is a single overall confidence score in [0, 1], where 0 means no\n'
        "  confidence and 1 means certain.\n"
    )


def _cortical_ask() -> str:
    return _ask_block(
        ["electrode_mediolateral_index", "electrode_rostrocaudal_index"],
        '{"c": {"electrode_mediolateral_index": <integer 1-4>, '
        '"electrode_rostrocaudal_index": <integer 1-8>}, "s": <number in [0,1]>}',
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_number_list(raw: str) -> list:
    values: list = []
    for v in (part.strip() for part in raw.split(",") if part.strip()):
        try:
            values.append(int(v) if re.fullmatch(r"-?\d+", v) else float(v))
        except ValueError:
            values.append(v.strip("'\""))
    return values


def _parse_card_dims(card_text: str) -> list[dict]:
    """Parse search-space dimensions from a rendered prompt card.

    Supports the multi-line block format (name:/type:/values:/scale: on separate lines)
    and the inline format (``Parameter N name: x type: continuous range: [lo, hi] scale: log ...``).
    """
    dims = []
    block = re.compile(
        r"name:\s*(?P<name>\S+)\s*\n"
        r"type:\s*(?P<type>\S+)\s*\n"
        r"(?:values:\s*\[(?P<values>[^\]]*)\]|range:\s*\[(?P<range>[^\]]*)\])"
        r"(?:[^\n]*\n(?:scale:\s*(?P<scale>\S+)|meaning:\s*(?P<meaning>[^\n]*))?)?",
    )
    for m in block.finditer(card_text):
        dim = {
            "name": m.group("name"),
            "type": m.group("type"),
            "scale": (m.group("scale")
                      or ("log" if "log-spaced" in (m.group("meaning") or "") else "linear")),
        }
        if m.group("values") is not None:
            dim["values"] = _parse_number_list(m.group("values"))
        else:
            bounds = _parse_number_list(m.group("range"))
            if len(bounds) != 2:
                continue
            dim["range"] = [float(bounds[0]), float(bounds[1])]
            dim["values"] = None
        dims.append(dim)
    if dims:
        return dims
    inline = re.compile(
        r"name:\s*(?P<name>\S+)\s+type:\s*(?P<type>\S+)\s+"
        r"(?:range:\s*\[(?P<range>[^\]]*)\]|values:\s*\[(?P<values>[^\]]*)\])"
        r"\s+scale:\s*(?P<scale>\S+)",
    )
    for m in inline.finditer(card_text):
        dim = {"name": m.group("name"), "type": m.group("type"), "scale": m.group("scale")}
        if m.group("range") is not None:
            bounds = _parse_number_list(m.group("range"))
            if len(bounds) != 2:
                continue
            dim["range"] = [float(bounds[0]), float(bounds[1])]
            dim["values"] = None
        else:
            dim["values"] = _parse_number_list(m.group("values"))
        dims.append(dim)
    return dims


# ---------------------------------------------------------------------------
# Prompt construction per modality (information-preserving card transforms)
# ---------------------------------------------------------------------------

def build_prompt_hpo(card_text: str, card_path: Path) -> str:
    """HPO card: replace the section-0 instruction; the rest of the card verbatim.

    Handles both layouts: ``## 1. Domain`` (block cards) and ``1. Domain`` on its own line."""
    m = re.search(r"(?m)^(?:## )?1\. Domain\s*$", card_text)
    if m is None:
        raise SystemExit(f"no '1. Domain' section found in {card_path} - refusing to guess")
    idx = m.start()
    dims = _parse_card_dims(card_text)
    if not dims:
        raise SystemExit(f"no search-space dims parsed from {card_path}")
    names = [d["name"] for d in dims]
    example = (
        '{"c": {' + ", ".join(f'"{n}": <allowed value>' for n in names) + '}, "s": <number in [0,1]>}'
    )
    header = "# Prompt B\n\n" + _ask_block(names, example) + "\n"
    return header + card_text[idx:]


def _cortical_ask_facts_only() -> str:
    """The (c, s) output instruction without the expert framing (cortical facts-only texts)."""
    ask = _cortical_ask()
    old = ("You are an expert in the domain described below. Using your knowledge of this" + chr(10)
           + "field, give your single best guess")
    assert old in ask
    return ask.replace(old, "Give your single best guess")


def build_prompt_context(card_path: Path) -> str:
    """Cortical: the shared text of the condition (warm-start context next to the card: Domain +
    goal [+ expert statement or map], the same text every method receives) plus the card's
    search space, with the (c, s) ask."""
    ctx = (card_path.parent / "warmstart_context.md").read_text(encoding="utf-8")
    domain = ctx.split("## 1. Domain")[1].split("## 3. Optimization description")[0].strip()
    goal = ctx.split("## 3. Optimization description")[1].strip()
    card = card_path.read_text(encoding="utf-8")
    space = card.split("## 2. Search space")[1].split("## 3. Optimization description")[0].strip()
    return ("# Prompt B\n\n" + _cortical_ask_facts_only() + "\n## 1. Domain\n" + domain
            + "\n\n## 2. Search space\n" + space + "\n\n## 3. Optimization description\n" + goal + "\n")


# ---------------------------------------------------------------------------
# LLM call + parsing + validation
# ---------------------------------------------------------------------------

def _extract_last_json(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    starts = [m.start() for m in re.finditer(r"\{", text)]
    for start in reversed(starts):
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "c" in obj and "s" in obj:
            return obj
    return None


def _snap(value, dim: dict):
    """Snap a raw value into the dimension's native domain: nearest allowed value
    for ordinal dims (log-space distance for log-scale dims), exact match for
    categorical dims, clamp into the range for continuous dims."""
    values = dim.get("values")
    if values is None:
        low, high = dim["range"]
        try:
            x = float(value)
        except (TypeError, ValueError):
            return None
        return min(max(x, low), high)
    if dim["type"] == "categorical" or any(isinstance(v, str) for v in values):
        return value if value in values else None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if dim.get("scale") == "log" and x > 0 and all(float(v) > 0 for v in values):
        return min(values, key=lambda v: abs(math.log(float(v)) - math.log(x)))
    return min(values, key=lambda v: abs(float(v) - x))


def validate_answer(obj: dict, dims: list[dict]) -> tuple[dict | None, dict | None, float | None, str]:
    c_raw = obj.get("c")
    s_raw = obj.get("s")
    if not isinstance(c_raw, dict):
        return None, None, None, '"c" must be a JSON object mapping parameter names to values'
    try:
        s = float(s_raw)
    except (TypeError, ValueError):
        return None, None, None, '"s" must be a number in [0, 1]'
    if not (0.0 <= s <= 1.0):
        return None, None, None, f'"s" must be in [0, 1], got {s_raw!r}'
    snapped: dict = {}
    if isinstance(c_raw, dict):  # svm: card says `c`, a native-space hint may say `C`
        folded = {str(k).lower(): v for k, v in c_raw.items()}
        c_raw = {d["name"]: folded[d["name"].lower()] for d in dims if d["name"].lower() in folded}
    for dim in dims:
        if dim["name"] not in c_raw:
            return None, None, None, f'missing parameter {dim["name"]!r} in "c"'
        snap = _snap(c_raw[dim["name"]], dim)
        if snap is None:
            return None, None, None, (
                f'value {c_raw[dim["name"]]!r} for {dim["name"]!r} is not an allowed value'
            )
        snapped[dim["name"]] = snap
    return c_raw, snapped, s, ""


def call_llm(prompt: str):
    from openai import OpenAI

    client = OpenAI(base_url=BASE_URL, api_key=os.environ.get("LLM_API_KEY", "none"), timeout=TIMEOUT)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Job inventory
# ---------------------------------------------------------------------------

def build_jobs() -> list[dict]:
    jobs = []
    for slug in CORTICAL_SLUGS:
        for modality in CORTICAL_MODALITIES:
            card = PRIORS_ROOT / CORTICAL_BENCHMARK / slug / modality / "prompt_card.md"
            prior = card.parent / "prior.json"
            if not card.is_file():
                continue
            jobs.append(
                {
                    "benchmark_id": CORTICAL_BENCHMARK,
                    "task": slug,
                    "modality": modality,
                    "card_path": card,
                    "prior_path": prior,
                    "dims": [dict(d) for d in CORTICAL_DIMS],
                }
            )
    import yaml

    seen = set()
    for study_path in HPO_SOURCE_STUDIES:
        study = yaml.safe_load(study_path.read_text(encoding="utf-8"))
        for bench in study["design"]["benchmarks"]:
            include = bench.get("include")
            if not (isinstance(include, list) and len(include) == 1):
                raise SystemExit(f"expected single-task include entries in {study_path}")
            task = include[0]
            prior_path = _ROOT / bench["prior_path"]
            card = prior_path.parent / "prompt_card.md"
            if not card.is_file():
                raise SystemExit(f"missing prompt card: {card}")
            if task in seen:
                raise SystemExit(f"duplicate HPO task {task!r} across source studies")
            seen.add(task)
            jobs.append(
                {
                    "benchmark_id": "hpobench-tabular",
                    "task": task,
                    "modality": HPO_MODALITY,
                    "card_path": card,
                    "prior_path": prior_path,
                    "dims": None,  # parsed from the card
                }
            )
    return jobs


def build_prompt(job: dict) -> str:
    card_text = job["card_path"].read_text(encoding="utf-8")
    if job["modality"] in (HPO_MODALITY, ADVERSARIAL_MODALITY):
        return build_prompt_hpo(card_text, job["card_path"])
    if job["modality"] in CORTICAL_MODALITIES:
        return build_prompt_context(job["card_path"])
    raise SystemExit(f"no prompt builder for modality {job['modality']!r}")


def build_adversarial_jobs() -> list[dict]:
    """Misleading-hint arm: every method receives the same misleading hint and processes it
    its own way. ColaBO's way = the standard card + the task's misleading-guidance text (the
    hint LLAMBO, BORA and LGBO receive in their task description) appended as a
    'Domain guidance' section -> one-shot (c, s)."""
    jobs = []
    for job in build_jobs():
        if job["benchmark_id"] != "hpobench-tabular":
            continue
        hint = _ROOT / "experiments" / "misleading_descriptions" / "hpobench-tabular" / f'{job["task"]}.txt'
        if not hint.is_file():
            raise SystemExit(f"missing misleading-description hint: {hint}")
        adversarial = dict(job)
        adversarial["modality"] = ADVERSARIAL_MODALITY
        adversarial["hint_path"] = hint
        jobs.append(adversarial)
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="write prompts only; no LLM calls")
    parser.add_argument("--only", default=None, help="substring filter on benchmark/task/modality")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--adversarial", action="store_true",
                        help="elicit the llm-card-misleading (c,s) set (40 HPO tasks) "
                             "instead of the standard modalities")
    args = parser.parse_args()

    jobs = build_adversarial_jobs() if args.adversarial else build_jobs()
    if args.only:
        jobs = [j for j in jobs if args.only in f'{j["benchmark_id"]}/{j["task"]}/{j["modality"]}']
    print(f"{len(jobs)} elicitation jobs")

    failures = []
    for job in jobs:
        out_dir = OUT_ROOT / job["benchmark_id"] / job["task"]
        out_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = out_dir / f'{job["modality"]}.json'
        prompt_path = out_dir / f'{job["modality"]}.prompt.md'
        label = f'{job["benchmark_id"]}/{job["task"]}/{job["modality"]}'
        if artifact_path.exists() and not args.overwrite and not args.dry_run:
            print(f"SKIP (exists) {label}")
            continue

        card_text = job["card_path"].read_text(encoding="utf-8")
        prompt = build_prompt(job)
        dims = job["dims"] or _parse_card_dims(card_text)
        if job["modality"] == ADVERSARIAL_MODALITY:
            hint = job["hint_path"].read_text(encoding="utf-8").strip()
            prompt = prompt.rstrip() + "\n\n## Domain guidance\n" + hint + "\n"
        if not dims:
            raise SystemExit(f"no dims for {label}")
        # the HPO cards end their question block with "... then write the prior."; ColaBO is
        # asked for a point guess instead, so only that closing phrase is adapted.
        prompt = prompt.replace("then write the prior.", "then give your answer.")
        prompt_path.write_text(prompt, encoding="utf-8", newline="\n")
        if args.dry_run:
            print(f"DRY {label}: prompt {len(prompt)} chars -> {prompt_path.name}")
            continue

        attempt_prompt = prompt
        raw = ""
        c_raw = c = s = None
        error = ""
        attempts = 0
        for attempt in range(1, args.max_attempts + 1):
            attempts = attempt
            raw = call_llm(attempt_prompt)
            obj = _extract_last_json(raw)
            if obj is None:
                error = "no JSON object with keys c and s found in the reply"
            else:
                c_raw, c, s, error = validate_answer(obj, dims)
            if not error:
                break
            attempt_prompt = (
                prompt
                + f"\n\nYour previous answer was invalid: {error}. "
                "Answer again; the final line must be exactly one JSON object of the requested form."
            )
        if error:
            failures.append((label, error))
            print(f"FAIL {label}: {error}")
            continue

        sigma = 1.0 - s + EPSILON
        artifact = {
            "artifact_format": "colabo-llm-point-prior-v1",
            "benchmark_id": job["benchmark_id"],
            "task": job["task"],
            "modality": job["modality"],
            "construction": "BORA supplement SC.2: pi(x) = N(c, sigma), sigma = 1 - s + 1e-6",
            "c_raw": c_raw,
            "c": c,
            "s": s,
            "sigma": sigma,
            "dimensions": dims,
            "model": MODEL,
            "base_url": BASE_URL,
            "temperature": 0.0,
            "enable_thinking": False,
            "attempts": attempts,
            "source_prompt_card": str(job["card_path"].relative_to(_ROOT)).replace("\\", "/"),
            "source_prompt_card_sha256": _sha256_text(card_text),
            **({"adversarial_hint": str(job["hint_path"].relative_to(_ROOT)).replace("\\", "/"),
                "adversarial_hint_sha256": _sha256_text(job["hint_path"].read_text(encoding="utf-8"))}
               if job.get("hint_path") else {}),
            "prompt_sha256": _sha256_text(prompt),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        raw_path = RAW_ROOT / job["benchmark_id"] / job["task"] / f'{job["modality"]}.txt'
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(raw, encoding="utf-8", newline="\n")
        artifact_path.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"OK   {label}: c={c} s={s:.3f} sigma={sigma:.6f} (attempts={attempts})")

    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for label, error in failures:
            print(f"  {label}: {error}")
        return 1
    print("\nall elicitations complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
