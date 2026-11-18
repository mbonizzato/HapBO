#!/usr/bin/env python3
"""Gate the hypbo-llm hypothesis artifacts before any run (scripts/elicit_hypbo_llm.py output).

The elicitation parser is the colabo-llm one, which converts values with float() and snaps them
to the grid. This gate re-reads every artifact's final raw reply (the elicitor's
results/elicitation/hypbo/<benchmark>/<task>/<modality>.replies.json; raw replies are not shipped)
and flags what that conversion hides. STOP flags block the runs; the others are counted and reported.

STOP:
  coverage        the root does not hold exactly one artifact per prior_path of the hpo/hypbo and
                  hpo/hypbo-misleading studies (missing or unexpected artifacts)
  c-non-numeric   a c value that is not a JSON number (string, bool, null, list, ...)
  c-non-finite    a c value that is NaN or infinite
  c-out-of-range  a c value outside the listed values / range (snapping moved it to an end)
  s-type          an s that is a bool or a string (or not a number)
  s-range         an s that is not finite or outside [0, 1]
  budget          H + 1 >= budget for the hypotheses passed to HypBO (no BO step is reached)
  mismatch        the artifact's parsed hypotheses differ from a re-parse of its raw reply (or the raw
                  reply is missing)
Counted:
  s-one           s >= 1 (runs with the 1e-6 side floor)
  c-off-grid      a c value inside the range but not a listed value (snapped)
  key-unknown     a c key that is not a parameter (case-insensitive)
  key-case-dup    two c keys that differ only in case
  key-repeated    a key repeated inside one JSON object
  entry-extra     a hypothesis entry with keys other than "c" and "s"
  duplicate       a hypothesis with the same centre and s as an earlier one
  same-centre     a hypothesis with the same centre as an earlier one (any s)
  dropped         a box not passed to HypBO (it and a kept same-centre box are narrower than 1e-4)

    python scripts/check_hypbo_hypotheses.py [--root DIR] [--replies-root DIR] [--budget 16] [--markdown OUT.md]
        [--json OUT.json]
Exit code 1 when any STOP flag is raised.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src" / "baselines" / "workers"))

from elicit_colabo_llm_priors import validate_answer  # noqa: E402
from box_hypotheses import select_hypotheses  # noqa: E402

STOP_FLAGS = ("coverage", "c-non-numeric", "c-non-finite", "c-out-of-range", "s-type", "s-range", "budget",
              "mismatch")
COUNT_FLAGS = ("s-one", "c-off-grid", "key-unknown", "key-case-dup", "key-repeated", "entry-extra", "duplicate",
               "same-centre", "dropped")
STUDIES = ("experiments/studies/hpo/hypbo/study.yaml", "experiments/studies/hpo/hypbo-misleading/study.yaml")
DEFAULT_ROOT = "experiments/hypbo_llm_hypotheses"
DEFAULT_REPLIES = "results/elicitation/hypbo"


def _pairs(pairs):
    """object_pairs_hook that keeps repeated keys visible."""

    obj = {}
    repeated = []
    for key, value in pairs:
        if key in obj:
            repeated.append(key)
        obj[key] = value
    if repeated:
        obj["__repeated_keys__"] = repeated
    return obj


def raw_answer(text: str) -> dict | None:
    """The last JSON object in the reply with a "hypotheses" key, repeated keys recorded."""

    decoder = json.JSONDecoder(object_pairs_hook=_pairs)
    for start in reversed([m.start() for m in re.finditer(r"\{", text)]):
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "hypotheses" in obj:
            return obj
    return None


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _bounds(dim: dict) -> tuple[float, float]:
    if dim.get("values") is not None:
        values = [float(v) for v in dim["values"]]
        return min(values), max(values)
    return float(dim["range"][0]), float(dim["range"][1])


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def final_reply(replies_path: Path) -> str:
    """The last attempt's raw reply in an elicitor replies file ("" when the file is missing)."""

    if not replies_path.is_file():
        return ""
    return json.loads(replies_path.read_text(encoding="utf-8"))["attempt_log"][-1]["raw_reply"]


def check_artifact(artifact: dict, budget: int, raw_reply: str) -> dict:
    dims = artifact["dimensions"]
    names = {d["name"].lower(): d for d in dims}
    flags: Counter = Counter()
    notes: list[str] = []
    obj = raw_answer(raw_reply or "")
    entries = obj.get("hypotheses") if isinstance(obj, dict) else None
    if not isinstance(entries, list):
        flags["mismatch"] += 1
        notes.append("raw reply has no hypotheses list" if raw_reply else "raw reply not found")
        entries = []
    if obj and obj.get("__repeated_keys__"):
        flags["key-repeated"] += 1
    for k, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            continue
        if entry.get("__repeated_keys__"):
            flags["key-repeated"] += 1
        if set(entry) - {"c", "s", "__repeated_keys__"}:
            flags["entry-extra"] += 1
        s = entry.get("s")
        if not _is_number(s):
            flags["s-type"] += 1
            notes.append(f"H{k} s={s!r}")
        elif not math.isfinite(s) or not 0.0 <= s <= 1.0:
            flags["s-range"] += 1
            notes.append(f"H{k} s={s!r}")
        elif s >= 1.0:
            flags["s-one"] += 1
        c = entry.get("c")
        if not isinstance(c, dict):
            continue
        if c.get("__repeated_keys__"):
            flags["key-repeated"] += 1
        keys = [key for key in c if key != "__repeated_keys__"]
        folded = Counter(str(key).lower() for key in keys)
        if any(n > 1 for n in folded.values()):
            flags["key-case-dup"] += 1
            notes.append(f"H{k} keys {keys}")
        for key in keys:
            dim = names.get(str(key).lower())
            if dim is None:
                flags["key-unknown"] += 1
                notes.append(f"H{k} unknown key {key!r}")
                continue
            value = c[key]
            if not _is_number(value):
                flags["c-non-numeric"] += 1
                notes.append(f"H{k} {key}={value!r}")
                continue
            if not math.isfinite(value):
                flags["c-non-finite"] += 1
                notes.append(f"H{k} {key}={value!r}")
                continue
            low, high = _bounds(dim)
            if (value < low and not _close(value, low)) or (value > high and not _close(value, high)):
                flags["c-out-of-range"] += 1
                notes.append(f"H{k} {key}={value!r} outside [{low:g}, {high:g}]")
            elif dim.get("values") is not None and not any(_close(value, float(v)) for v in dim["values"]):
                flags["c-off-grid"] += 1

    # the stored hypotheses must be the colabo-llm parse of the same raw entries
    stored = artifact.get("hypotheses") or []
    reparsed = []
    for entry in entries:
        clean = {key: value for key, value in entry.items() if key != "__repeated_keys__"} \
            if isinstance(entry, dict) else entry
        if isinstance(clean, dict) and isinstance(clean.get("c"), dict):
            clean["c"] = {key: value for key, value in clean["c"].items() if key != "__repeated_keys__"}
        _, snapped, s, error = validate_answer(clean, dims) if isinstance(clean, dict) else (None, None, None, "x")
        reparsed.append(None if error else (snapped, s))
    if len(reparsed) != len(stored) or any(
        r is None or r[0] != h["c"] or r[1] != h["s"] for r, h in zip(reparsed, stored)
    ):
        flags["mismatch"] += 1
        notes.append("stored hypotheses differ from the raw reply")

    centers = [tuple(h["center_encoded"]) for h in stored]
    kept, dropped = select_hypotheses(centers, [h["box_encoded"]["lb"] for h in stored],
                                      [h["box_encoded"]["ub"] for h in stored])
    flags["dropped"] += len(dropped)
    if [h.get("kept", True) for h in stored] != [k not in dropped for k in range(len(stored))]:
        flags["mismatch"] += 1
        notes.append("stored kept flags differ from the selection rule")
    for k in range(len(stored)):
        if centers[k] in centers[:k]:
            flags["same-centre"] += 1
            if any(centers[j] == centers[k] and stored[j]["s"] == stored[k]["s"] for j in range(k)):
                flags["duplicate"] += 1
    if len(kept) + 1 >= budget:
        flags["budget"] += 1
        notes.append(f"H + 1 = {len(kept) + 1} >= budget {budget}")
    return {
        "task": artifact.get("task"),
        "modality": artifact.get("modality"),
        "h_raw": len(stored),
        "h_kept": len(kept),
        "s": [h["s"] for h in stored],
        "attempts": artifact.get("attempts"),
        "usage": artifact.get("usage"),
        "flags": dict(flags),
        "stop": sorted(f for f in flags if f in STOP_FLAGS and flags[f]),
        "notes": notes,
    }


def markdown(rows: list[dict]) -> str:
    lines = ["| task | condition | H kept (parsed) | s | attempts | flags |", "|---|---|---|---|---|---|"]
    for row in rows:
        flags = ", ".join(f"{k} x{v}" for k, v in sorted(row["flags"].items()) if v) or "-"
        s = ", ".join(f"{v:g}" for v in row["s"])
        condition = "misleading" if str(row["modality"]).endswith("misleading") else "good"
        lines.append(f"| {row['task']} | {condition} | {row['h_kept']} ({row['h_raw']}) | {s} | "
                     f"{row['attempts']} | {flags} |")
    totals = Counter()
    for row in rows:
        totals.update({k: v for k, v in row["flags"].items() if v})
    lines.append("")
    lines.append("Flag totals: " + (", ".join(f"{k} {totals[k]}" for k in STOP_FLAGS + COUNT_FLAGS if totals[k])
                                    or "none"))
    stops = [row for row in rows if row["stop"]]
    lines.append("STOP flags: " + ("; ".join(f"{r['task']}/{r['modality']}: {', '.join(r['stop'])} "
                                            f"({'; '.join(r['notes'])})" for r in stops) or "none"))
    return "\n".join(lines) + "\n"


def expected_artifacts(studies=STUDIES) -> list[str]:
    """Artifact paths relative to the hypotheses root, one per prior_path of the studies."""

    import yaml

    expected = []
    for study in studies:
        manifest = yaml.safe_load((_ROOT / study).read_text(encoding="utf-8"))
        for bench in manifest["design"]["benchmarks"]:
            path = Path(bench["prior_path"]).as_posix()
            if not path.startswith(DEFAULT_ROOT + "/"):
                raise SystemExit(f"{study}: prior_path {path} is not under {DEFAULT_ROOT}")
            expected.append(path[len(DEFAULT_ROOT) + 1:])
    return expected


def run_gate(root: Path, budget: int, studies=STUDIES, replies_root: Path | None = None
             ) -> tuple[list[dict], list[str]]:
    """Check every artifact under ``root`` against its raw reply under ``replies_root``; the second value
    lists the coverage problems."""

    replies_root = Path(replies_root) if replies_root is not None else _ROOT / DEFAULT_REPLIES
    expected = expected_artifacts(studies)
    found = sorted(p.relative_to(root).as_posix() for p in root.glob("*/*/*.json"))
    problems = [f"missing {path}" for path in sorted(set(expected) - set(found))]
    problems += [f"unexpected {path}" for path in sorted(set(found) - set(expected))]
    if not expected or len(expected) != len(set(expected)):
        problems.append("the studies list no prior_path or one twice")
    rows = [check_artifact(json.loads((root / path).read_text(encoding="utf-8")), budget,
                           final_reply((replies_root / path).with_suffix(".replies.json"))) for path in found]
    return rows, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(_ROOT / DEFAULT_ROOT))
    parser.add_argument("--replies-root", default=str(_ROOT / DEFAULT_REPLIES),
                        help="raw replies written by scripts/elicit_hypbo_llm.py")
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--markdown", default=None, help="write the summary table here")
    parser.add_argument("--json", default=None, help="write the per-artifact records here")
    args = parser.parse_args()

    rows, problems = run_gate(Path(args.root), args.budget, replies_root=Path(args.replies_root))
    coverage = f"Coverage: {len(rows)} artifacts, " + (
        "one per study prior_path" if not problems else
        f"STOP ({len(problems)} problems: " + "; ".join(problems[:10]) + ("; ..." if len(problems) > 10 else "") + ")")
    report = markdown(rows) + coverage + "\n"
    print(report)
    if args.markdown:
        Path(args.markdown).write_text(report, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps({"rows": rows, "coverage_problems": problems}, indent=2) + "\n",
                                   encoding="utf-8")
    return 1 if problems or any(row["stop"] for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
