#!/usr/bin/env python3
"""Direct-LLM representation prior: the LLM scores a candidate grid directly (no PSL compiler).

One call per prior directory (HPOBench `llm-card`, `llm-card-misleading`). A scale-aware coarse candidate grid
is built (the exact product grid for an all-ordinal space of at most 128 points, otherwise a 128-point Sobol
sample; every HPOBench grid is larger, so it gets the Sobol sample), the LLM returns a scalar quality score in
[0, 1] per candidate, the scores are min-max normalized and stored as an `anchors` prior (pi = interpolation of
the raw scores, no PSL landscape structure). The trust block and the dominant provenance of the source prior are
kept, so alpha equals the source arm's value and the arm differs only in how pi is built. The task text is the
source's prompt_card.md; a source directory with a warmstart_context.md (the cortical priors) is rejected.

Writes <src_prior_dir>-directllm/ next to each source (prior.json, prompt_card.md, finalized metadata.json).
Endpoint: LLM_BASE_URL, LLM_MODEL, LLM_API_KEY (OpenAI-compatible; temperature 0, thinking off, seed 1).

usage: python scripts/make_directllm_priors.py <src_prior_dir> [<src_prior_dir> ...]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from llm.client import LLMClient  # noqa: E402
from prior.landscape.landscape_generator import compile_prior, finalize_prior_experiment, load_record  # noqa: E402

META = {"smoothness", "noise", "modality"}
CAP = 128   # power of 2 for Sobol balance
SCHEMA = {"type": "object", "properties": {"scores": {"type": "array", "items": {"type": "number"}}},
          "required": ["scores"]}


def _sobol_candidates(record, n):
    """Scale-aware quasi-random candidates."""
    from scipy.stats import qmc
    dims = record["schema"]["dimensions"]
    u = qmc.Sobol(d=len(dims), scramble=True, seed=0).random(n)
    cands = []
    for row in u:
        c = {}
        for j, d in enumerate(dims):
            if "values" in d:
                vals = d["values"]
                c[d["name"]] = vals[min(len(vals) - 1, int(row[j] * len(vals)))]
            else:
                lo, hi = float(d["range"][0]), float(d["range"][1])
                if d.get("scale") == "log" and lo > 0:
                    c[d["name"]] = float(np.exp(np.log(lo) + row[j] * (np.log(hi) - np.log(lo))))
                else:
                    c[d["name"]] = float(lo + row[j] * (hi - lo))
        cands.append(c)
    return cands


def _dominant_provenance(record):
    shaping = [b for b in record["beliefs"] if b["token"] not in META and b["token"] != "baseline"]
    if not shaping:
        return "llm_domain"
    dom = max(shaping, key=lambda b: float(b.get("weight", 1.0)))
    return dom.get("provenance", "llm_domain")


def coarse_grid(record, cap=CAP):
    dims = record["schema"]["dimensions"]
    names = [d["name"] for d in dims]
    cont = [d for d in dims if "values" not in d]
    if not cont:                                  # all-ordinal: exact product grid
        axes = [[float(v) for v in d["values"]] for d in dims]
        grid = [dict(zip(names, combo)) for combo in itertools.product(*axes)]
        if len(grid) > cap:
            grid = _sobol_candidates(record, cap)
        return names, grid
    return names, _sobol_candidates(record, cap)  # continuous dims: space-filling Sobol


def build_prompt(card, names, grid):
    lines = [card.strip(),
             "\n---\nTASK: Using ONLY your domain expertise (no experiments), score how likely "
             "EACH candidate configuration below is to MAXIMIZE the objective. Use a scalar in "
             "[0,1] (1 = very likely in the optimal region, 0 = very likely poor).",
             f"\nDimensions, in order: {names}",
             "Candidates (index: [values in dimension order]):"]
    for i, c in enumerate(grid):
        lines.append(f"{i}: {[c[n] for n in names]}")
    lines.append(f"\nReturn a JSON object {{\"scores\": [...]}} with EXACTLY {len(grid)} numbers "
                 f"in [0,1], one per candidate index, in order.")
    return "\n".join(lines)


def parse_scores(text, n):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    raw = m.group(0) if m else text
    try:
        obj = json.loads(raw)
        scores = obj["scores"] if isinstance(obj, dict) else obj
    except Exception:  # noqa: BLE001 -- fall back to every number in the answer
        nums = re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", text)
        scores = [float(x) for x in nums]
    scores = [float(x) for x in scores]
    if len(scores) < n:
        scores += [0.5] * (n - len(scores))
    return scores[:n]


def elicit(src_dir, timeout):
    src = Path(src_dir).resolve()
    if (src / "warmstart_context.md").is_file():
        raise SystemExit(f"{src}: has a warmstart_context.md (its task text is not in prompt_card.md); "
                         "only HPOBench llm-card priors are supported")
    rec = load_record(src / "prior.json")
    card = (src / "prompt_card.md").read_text(encoding="utf-8")
    names, grid = coarse_grid(rec)
    client = LLMClient(model=os.environ.get("LLM_MODEL", "Qwen3.5-122B-A10B-GPTQ-Int4"), temperature=0.0,
                       max_tokens=8192, cache_dir=REPO / "results" / "elicitation" / "directllm_cache", budget=8,
                       provider="openai", base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
                       api_key=os.environ.get("LLM_API_KEY", "none"), enable_thinking=False,
                       request_timeout=timeout, max_retries=1)
    text = client.complete(build_prompt(card, names, grid),
                           system="You are a careful domain expert providing a prior over "
                                  "configurations for Bayesian optimization.",
                           response_schema=SCHEMA, seed=1)
    s = np.asarray(parse_scores(text, len(grid)), dtype=float)
    s = np.clip(s, 0.0, None)
    s = (s - s.min()) / (s.max() - s.min() + 1e-9)
    pts = [{"coords": {n: (int(c[n]) if float(c[n]).is_integer() else float(c[n])) for n in names},
            "value": round(float(v), 4)} for c, v in zip(grid, s)]
    doc = {"schema": rec["schema"],
           "beliefs": [{"token": "baseline", "level": 0.05},
                       {"token": "anchors", "provenance": _dominant_provenance(rec),
                        "bandwidth": 0.12, "points": pts,
                        "note": "direct-LLM: raw one-shot LLM grid scores (no PSL compiler)"}]
                      + [b for b in rec["beliefs"] if b["token"] in META],
           "trust": rec["trust"]}
    dst = src.parent / f"{src.name}-directllm"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir()
    (dst / "prior.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                                    encoding="utf-8", newline="\n")
    shutil.copy2(src / "prompt_card.md", dst / "prompt_card.md")
    md = json.loads((src / "metadata.json").read_text(encoding="utf-8"))
    md["status"] = "draft"
    md["prior_id"] = dst.name
    md["files"] = {"prompt_card.md": {"sha256": None}, "prior.json": {"sha256": None}}
    vocab = (md.get("vocabulary") or {}).get("path", "src/prior/expert_knowledge/vocabulary_card.md")
    md["vocabulary"] = {"path": vocab, "sha256": None}
    md.pop("finalized_at_utc", None)
    (dst / "metadata.json").write_text(json.dumps(md, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8", newline="\n")
    finalize_prior_experiment(dst, generate_landscape=False)
    c = compile_prior(load_record(dst / "prior.json"))
    print(f"  OK {dst.name:44s} alpha={c.alpha:.3f} n_anchors={len(pts)} score_span=[{s.min():.2f},{s.max():.2f}]")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("prior_dirs", nargs="+", type=Path)
    ap.add_argument("--request-timeout", type=float, default=300.0)
    a = ap.parse_args()
    ok, fail = 0, []
    for s in a.prior_dirs:
        print(f"[{s.parent.name}/{s.name}]", flush=True)
        try:
            elicit(s, a.request_timeout)
            ok += 1
        except Exception as exc:  # noqa: BLE001 -- keep going; the response cache lets a rerun resume
            print(f"  FAIL {s.name}: {type(exc).__name__}: {exc}", flush=True)
            fail.append(s)
    print(f"\ndirect-LLM: {ok} ok, {len(fail)} failed")
    for s in fail:
        print("  failed:", s)
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
