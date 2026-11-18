#!/usr/bin/env python3
"""Point-only and Gaussian-only representation variants of an elicited prior (no LLM).

The injection site (alpha-pi) and the strength alpha (trust) are unchanged; only how the elicited
belief becomes pi(x) differs:

  <name>-pointonly : the belief collapsed to a single suggested optimum: one `bump` per dimension at the
                     argmax of the full compiled pi (scale-aware dense grid, geomspace for log dims; a Sobol
                     sample in high dimension), ci = neighbouring allowed values (ordinal) or +-15 % of the
                     range (continuous).
  <name>-gaussian  : only the Gaussian family (bump / joint_region) + baseline + meta tokens; trend, avoid,
                     threshold, periodic, ridge, conditional, categorical and anchors are dropped. A source
                     without Gaussian tokens gets one broad Gaussian joint_region at the best point of its
                     anchors belief. In --psl-docs mode a document with neither Gaussian tokens nor anchors gets
                     one broad joint_region at the full-pi argmax (ci 25 % of the range per dimension).

Prior directories (HPOBench `llm-card`, `llm-card-misleading`): the variants are written as sibling directories
next to each source and finalized with finalize_prior_experiment (sha256 of prior.json, card and vocabulary card).

  python scripts/make_representation_variants.py <prior_dir> [<prior_dir> ...]

Synthetic documents (--psl-docs): every experiments/synthetic_priors/rho90/<fam>/draw<d>/prior.json (card
PSL_DOCS_CARD, the card of the representation ablation) gets draw<d>-pointonly/ and draw<d>-gaussian/ siblings
(prior.json only; the synthetic runner reads prior.json directly). In this mode the argmax search also includes the
document's own peak coordinates (anchor points, joint_region / ridge modes, per-dimension bump modes combined), so
that a Sobol sample in 10-15 dimensions does not miss a narrow peak (--peak-points; on by default in --psl-docs
mode, off otherwise).

  python scripts/make_representation_variants.py --psl-docs [experiments/synthetic_priors]
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from prior.landscape.landscape_generator import (  # noqa: E402
    compile_prior, finalize_prior_experiment, load_record, validate_record,
)

META = {"smoothness", "noise", "modality"}
GAUSSIAN_SHAPING = {"bump", "joint_region"}
PEAK_POINTS = False  # set by --peak-points / --psl-docs
PSL_DOCS_CARD = "rho90"  # synthetic card whose documents get representation variants (--psl-docs)


def _baseline(record):
    return next((b for b in record["beliefs"] if b["token"] == "baseline"),
                {"token": "baseline", "level": 0.1})


def _meta(record):
    return [b for b in record["beliefs"] if b["token"] in META]


def _dim_specs(record):
    specs = {}
    for d in record["schema"]["dimensions"]:
        if "values" in d:
            specs[d["name"]] = ("ordinal", sorted(float(v) for v in d["values"]))
        else:
            specs[d["name"]] = ("continuous",
                                (float(d["range"][0]), float(d["range"][1]), d.get("scale", "linear")))
    return specs


def _ci_for(spec, center, *, ord_steps, cont_frac):
    """Valid ci interval: ordinal endpoints snap to allowed values; continuous endpoints
    are scale-aware (log -> multiplicative factor, linear -> additive fraction)."""
    kind, payload = spec
    if kind == "ordinal":
        vals = payload
        i = min(range(len(vals)), key=lambda k: abs(vals[k] - center))
        lo = vals[max(0, i - ord_steps)]
        hi = vals[min(len(vals) - 1, i + ord_steps)]
        if lo == hi:
            hi = vals[i + 1] if i + 1 < len(vals) else hi
            lo = vals[i - 1] if (lo == hi and i - 1 >= 0) else lo
        return [lo, hi]
    lo, hi, scale = payload
    if scale == "log" and center > 0 and lo > 0:
        f = 3.0
        return [max(lo, center / f), min(hi, center * f)]
    half = cont_frac * (hi - lo)
    a, b = max(lo, center - half), min(hi, center + half)
    if b - a < 1e-9:
        b = min(hi, a + 1e-6 * (hi - lo) + 1e-9)
    return [a, b]


def _dense_grid(record, n=9):
    dims = record["schema"]["dimensions"]
    axes = []
    for d in dims:
        if "values" in d:
            ax = [float(v) for v in d["values"]]
        else:
            lo, hi = float(d["range"][0]), float(d["range"][1])
            if d.get("scale") == "log" and lo > 0:
                ax = list(np.geomspace(lo, hi, n))
            else:
                ax = list(np.linspace(lo, hi, n))
        axes.append((d["name"], ax))
    names = [a[0] for a in axes]
    prod = 1
    for _, ax in axes:
        prod *= len(ax)
    if prod > 200_000:                # high dimension: a product grid explodes -> Sobol sample
        grid = _sobol_grid(record, 8192)
        return grid + _belief_peak_points(record) if PEAK_POINTS else grid
    return [dict(zip(names, combo)) for combo in itertools.product(*[a[1] for a in axes])]


def _belief_peak_points(record):
    dims = record["schema"]["dimensions"]
    names = [d["name"] for d in dims]
    mid = {d["name"]: (float(np.mean(d["values"])) if "values" in d
                       else 0.5 * (float(d["range"][0]) + float(d["range"][1]))) for d in dims}
    pts, combined = [], dict(mid)
    for b in record["beliefs"]:
        if b["token"] == "anchors":
            for p in b["points"]:
                pts.append({n: float(p["coords"].get(n, mid[n])) for n in names})
        elif b["token"] in ("joint_region", "ridge") and isinstance(b.get("mode") or b.get("through"), dict):
            m = b.get("mode") or b.get("through")
            pts.append({n: float(m.get(n, mid[n])) for n in names})
            combined.update({k: float(v) for k, v in m.items() if k in mid})
        elif b["token"] == "bump" and b.get("sign", "peak") == "peak" and "dim" in b:
            combined[b["dim"]] = float(b["mode"])
    pts.append(combined)
    return pts


def _sobol_grid(record, n):
    """Scale-aware quasi-random points for the argmax search in high dimension (log -> geometric)."""
    from scipy.stats import qmc
    dims = record["schema"]["dimensions"]
    m = 1 << int(np.ceil(np.log2(max(2, n))))          # Sobol prefers a power of 2
    u = qmc.Sobol(d=len(dims), scramble=True, seed=0).random(m)
    out = []
    for row in u:
        c = {}
        for j, d in enumerate(dims):
            if "values" in d:
                vals = d["values"]
                c[d["name"]] = float(vals[min(len(vals) - 1, int(row[j] * len(vals)))])
            else:
                lo, hi = float(d["range"][0]), float(d["range"][1])
                if d.get("scale") == "log" and lo > 0:
                    c[d["name"]] = float(np.exp(np.log(lo) + row[j] * (np.log(hi) - np.log(lo))))
                else:
                    c[d["name"]] = float(lo + row[j] * (hi - lo))
        out.append(c)
    return out


def _dominant_provenance(record):
    shaping = [b for b in record["beliefs"] if b["token"] not in META and b["token"] != "baseline"]
    if not shaping:
        return "llm_domain"
    dom = max(shaping, key=lambda b: float(b.get("weight", 1.0)))
    return dom.get("provenance", "llm_domain")


def _argmax_point(record):
    grid = _dense_grid(record)
    pi = np.asarray(compile_prior(record).pi(grid), dtype=float)
    return grid[int(pi.argmax())]


def make_point_only(record):
    """One `bump` per dimension at the full-prior argmax. Per-dimension bumps are added (not
    multiplied), giving a localized peak that keeps axis influence."""
    best = _argmax_point(record)
    specs = _dim_specs(record)
    prov = _dominant_provenance(record)
    bumps = []
    for d in record["schema"]["dimensions"]:
        nm = d["name"]
        coord = float(best[nm])
        ci = _ci_for(specs[nm], coord, ord_steps=1, cont_frac=0.15)
        if specs[nm][0] == "ordinal":
            coord = min(specs[nm][1], key=lambda v: abs(v - coord))
            coord = int(coord) if float(coord).is_integer() else coord
        bumps.append({"token": "bump", "dim": nm, "mode": coord, "ci": ci,
                      "weight": 0.999, "provenance": prov,
                      "note": "point-only: per-dimension bump at the full-prior argmax"})
    if bumps:
        bumps[0]["weight"] = 1.0
    return {"schema": record["schema"], "beliefs": [_baseline(record)] + bumps + _meta(record),
            "trust": record["trust"]}


def make_gaussian_only(record):
    """Only Gaussian-family shaping (bump / joint_region) + baseline + meta; an anchors-only
    source gets a single broad Gaussian joint_region at its best anchor."""
    beliefs = [_baseline(record)]
    kept = [json.loads(json.dumps(b)) for b in record["beliefs"] if b["token"] in GAUSSIAN_SHAPING]
    if kept:
        dom = max(kept, key=lambda b: float(b.get("weight", 0.0)))
        for b in kept:
            b["weight"] = 1.0 if b is dom else min(0.999, float(b.get("weight", 1.0)))
        beliefs += kept
    else:
        anc = next(b for b in record["beliefs"] if b["token"] == "anchors")
        best = max(anc["points"], key=lambda p: p["value"])
        specs = _dim_specs(record)
        dims = list(best["coords"].keys())
        ci = {dim: _ci_for(specs[dim], float(best["coords"][dim]), ord_steps=2, cont_frac=0.25) for dim in dims}
        beliefs.append({"token": "joint_region", "dims": dims,
                        "mode": dict(best["coords"]), "ci": ci, "weight": 1.0,
                        "provenance": anc.get("provenance", "user_stated"),
                        "note": "Gaussian-only: single broad Gaussian at the prior peak"})
    beliefs += _meta(record)
    return {"schema": record["schema"], "beliefs": beliefs, "trust": record["trust"]}


def gaussian_only_doc(record):
    """Gaussian-only for a free-form document: argmax fallback when it has neither Gaussian tokens nor anchors."""
    toks = {b["token"] for b in record["beliefs"]}
    if toks & GAUSSIAN_SHAPING or "anchors" in toks:
        return make_gaussian_only(record)
    best = _argmax_point(record)
    specs = _dim_specs(record)
    dims = [d["name"] for d in record["schema"]["dimensions"]]
    mode = {nm: float(best[nm]) for nm in dims}
    ci = {nm: _ci_for(specs[nm], mode[nm], ord_steps=2, cont_frac=0.25) for nm in dims}
    jr = {"token": "joint_region", "dims": dims, "mode": mode, "ci": ci, "weight": 1.0,
          "provenance": _dominant_provenance(record),
          "note": "gaussian-only fallback: no Gaussian/anchors tokens -> one broad Gaussian at the full-pi argmax"}
    return {"schema": record["schema"], "beliefs": [_baseline(record), jr] + _meta(record), "trust": record["trust"]}


def emit(src_dir: Path, suffix: str, transform, finalize: bool = True):
    """Write <src_dir>-<suffix>/ (prior.json, prompt_card.md, metadata.json) next to a prior directory."""
    src_dir = src_dir.resolve()
    record = load_record(src_dir / "prior.json") if finalize else \
        json.loads((src_dir / "prior.json").read_text(encoding="utf-8"))
    variant = transform(record)
    dst = src_dir.parent / f"{src_dir.name}-{suffix}"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir()
    (dst / "prior.json").write_text(json.dumps(variant, indent=2, ensure_ascii=False) + "\n",
                                    encoding="utf-8", newline="\n")
    if (src_dir / "prompt_card.md").is_file():
        shutil.copy2(src_dir / "prompt_card.md", dst / "prompt_card.md")
    md = json.loads((src_dir / "metadata.json").read_text(encoding="utf-8"))
    md["prior_id"] = dst.name
    if finalize:
        md["status"] = "draft"
        md["files"] = {"prompt_card.md": {"sha256": None}, "prior.json": {"sha256": None}}
        vocab = (md.get("vocabulary") or {}).get("path", "src/prior/expert_knowledge/vocabulary_card.md")
        md["vocabulary"] = {"path": vocab, "sha256": None}
        md.pop("finalized_at_utc", None)
    (dst / "metadata.json").write_text(json.dumps(md, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8", newline="\n")
    if finalize:
        finalize_prior_experiment(dst, generate_landscape=False)
    c = compile_prior(variant)
    toks = [b["token"] for b in variant["beliefs"] if b["token"] not in META]
    print(f"  OK {dst.name:44s} alpha={c.alpha:.3f} tokens={toks}")
    return dst


def psl_docs(root: Path) -> int:
    n = 0
    for src in sorted(root.glob(f"{PSL_DOCS_CARD}/*/draw*/prior.json")):
        if src.parent.name.endswith(("-pointonly", "-gaussian")):
            continue
        rec = json.loads(src.read_text(encoding="utf-8"))
        alpha = compile_prior(rec).alpha
        for suffix, fn in (("pointonly", make_point_only), ("gaussian", gaussian_only_doc)):
            var = fn(rec)
            validate_record(var)
            assert abs(compile_prior(var).alpha - alpha) < 1e-9, (src, suffix)
            dst = src.parent.parent / f"{src.parent.name}-{suffix}"
            dst.mkdir(parents=True, exist_ok=True)
            (dst / "prior.json").write_text(json.dumps(var, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                                            newline="\n")
            n += 1
    return n


def main() -> None:
    global PEAK_POINTS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("prior_dirs", nargs="*", type=Path)
    ap.add_argument("--psl-docs", nargs="?", const=REPO / "experiments" / "synthetic_priors", type=Path, default=None,
                    help="derive the variants of every rho90 synthetic document under this root")
    ap.add_argument("--peak-points", action=argparse.BooleanOptionalAction, default=None,
                    help="add the document's own peak coordinates to the high-dimensional argmax search")
    ap.add_argument("--no-finalize", action="store_true", help="prior directories: copy metadata without re-hashing")
    a = ap.parse_args()
    PEAK_POINTS = a.peak_points if a.peak_points is not None else a.psl_docs is not None
    if a.psl_docs is not None:
        print(f"{psl_docs(a.psl_docs)} variants written under {a.psl_docs}")
    for src in a.prior_dirs:
        print(f"[{src.parent.name}/{src.name}]")
        emit(src, "pointonly", make_point_only, finalize=not a.no_finalize)
        emit(src, "gaussian", make_gaussian_only, finalize=not a.no_finalize)
    if a.psl_docs is None and not a.prior_dirs:
        ap.error("give prior directories or --psl-docs")


if __name__ == "__main__":
    main()
