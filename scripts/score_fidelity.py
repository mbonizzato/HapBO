#!/usr/bin/env python3
"""Fidelity of a PSL document for the synthetic panel to the card model it was elicited from.

Metrics, on a fixed sample of 100k uniform points per function (identical for every document; seed derived from
sha256("agentic-fid:<family>")):
  spearman_model             Spearman(pi, -M) with M the rho90 card model (scripts/synthetic_truth.card_model);
                             average ranks for ties, constant pi -> 0
  top5_model, top10_model    |top_k(pi) & top_k(-M)| / k for k = 5% / 10% of the points; each top set from a stable
                             sort on (-value, index), i.e. ties broken by point index
  spearman_well, top5_well, top10_well   the same against the well truth (scripts/synthetic_truth.well_truth)
pi is compile_prior(doc).pi(X) with the columns in the document's dimension order (mapped by name from x0..x{d-1}).
Validity checks: validate_record, schema dimensions equal x0..x{d-1} with the box range and linear scale, objective
minimize, provenance llm_domain on every belief that carries one (anchors included), trust overall_confidence medium,
tokens = beliefs + anchor points (an anchors belief counts 1 + 1 per point) against --cap.

usage: python scripts/score_fidelity.py <prior.json> <family> [--cap 60] [--label NAME --out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import synthetic_truth as ST  # noqa: E402
from prior.landscape.landscape_generator import compile_prior, validate_record  # noqa: E402

N_POINTS = 100_000


def sample(fam: str, n: int = N_POINTS) -> np.ndarray:
    d, lo, hi = ST.BOUNDS[fam]
    seed = int(hashlib.sha256(("agentic-fid:" + fam).encode()).hexdigest()[:8], 16)
    return lo + (hi - lo) * np.random.default_rng(seed).random((n, d))


def task_names(fam: str) -> list[str]:
    return [f"x{i}" for i in range(ST.BOUNDS[fam][0])]


def topk_overlap(a: np.ndarray, b: np.ndarray, frac: float) -> tuple[float, int]:
    n = len(a)
    k = max(1, int(round(frac * n)))
    idx = np.arange(n)
    ta = np.lexsort((idx, -a))[:k]
    tb = np.lexsort((idx, -b))[:k]
    return float(len(np.intersect1d(ta, tb, assume_unique=True)) / k), int(np.sum(a == a[ta[-1]]))


def fidelity(pi: np.ndarray, g: np.ndarray) -> tuple[float, float, float, int, int]:
    """pi against goodness g (higher is better)."""
    rho = 0.0 if np.ptp(pi) == 0 else float(stats.spearmanr(pi, g).statistic)
    t5, tie5 = topk_overlap(pi, g, 0.05)
    t10, tie10 = topk_overlap(pi, g, 0.10)
    return rho, t5, t10, tie5, tie10


def count_tokens(doc: dict) -> int:
    return sum(1 + len(b.get("points", [])) if b.get("token") == "anchors" else 1 for b in doc.get("beliefs", []))


def checks(doc: dict, fam: str, cap: int) -> dict:
    out = {}
    try:
        validate_record(doc)
        out["valid"] = True
    except Exception as e:  # noqa: BLE001
        out["valid"], out["valid_error"] = False, str(e)[:300]
    d, lo, hi = ST.BOUNDS[fam]
    dd = doc.get("schema", {}).get("dimensions", [])
    out["dims_names_order_ok"] = [x.get("name") for x in dd] == task_names(fam)
    out["dims_range_ok"] = len(dd) == d and all(
        x.get("type") == "continuous" and [float(v) for v in x.get("range", [])] == [lo, hi]
        and x.get("scale", "linear") == "linear" for x in dd)
    out["objective_minimize"] = doc.get("schema", {}).get("objective") == "minimize"
    anchors = [b for b in doc.get("beliefs", []) if b.get("token") == "anchors"]
    out["provenance_all_llm_domain"] = all(b.get("provenance") == "llm_domain"
                                           for b in doc.get("beliefs", []) if "provenance" in b)
    out["anchors_have_provenance"] = all(b.get("provenance") == "llm_domain" for b in anchors)
    out["trust_medium"] = doc.get("trust", {}).get("overall_confidence") == "medium"
    out["tokens"] = count_tokens(doc)
    out["tokens_ok"] = out["tokens"] <= cap
    out["token_types"] = sorted({b.get("token") for b in doc.get("beliefs", [])})
    return out


def score(path: Path, fam: str, cap: int = 60, label: str | None = None) -> dict:
    raw = Path(path).read_bytes()
    doc = json.loads(raw.decode("utf-8"))
    res = {"label": label or Path(path).name, "family": fam, "md5": hashlib.md5(raw).hexdigest(), "n_points": N_POINTS}
    res.update(checks(doc, fam, cap))
    cp = compile_prior(doc)
    res["alpha_compiled"] = round(float(cp.alpha), 4)
    X = sample(fam)
    names = task_names(fam)
    order = [names.index(str(n).lower()) for n in cp.parameter_names]
    pi = np.asarray(cp.pi(X[:, order]), float)
    res["pi_min"], res["pi_max"], res["pi_mean"] = float(pi.min()), float(pi.max()), float(pi.mean())
    for tag, g in (("model", -ST.card_model(fam)(X)), ("well", -ST.well_truth(fam)(X))):
        r, t5, t10, tie5, tie10 = fidelity(pi, g)
        res[f"spearman_{tag}"], res[f"top5_{tag}"], res[f"top10_{tag}"] = r, t5, t10
        res["pi_ties_top5"], res["pi_ties_top10"] = tie5, tie10
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("prior", type=Path)
    ap.add_argument("family", choices=list(ST.FAMILIES))
    ap.add_argument("--cap", type=int, default=60, help="token cap (60 one call; 2 x dimension agentic)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", type=Path, default=None, help="write <out>/<label>__<family>.json")
    a = ap.parse_args()
    res = score(a.prior, a.family, a.cap, a.label)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / f"{res['label']}__{a.family}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"{res['label']} {a.family}: spearman_model={res['spearman_model']:+.4f} top10_model={res['top10_model']:.4f} "
          f"spearman_well={res['spearman_well']:+.4f} valid={res['valid']} tokens={res['tokens']}")


if __name__ == "__main__":
    main()
