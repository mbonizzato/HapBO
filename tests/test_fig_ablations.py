"""fig_ablations: panel C is the synthetic rho ~ 0.5 injection-site panel; its stored runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "paper"))
BUDGETS = {"hartmann6": 40, "styblinski_tang7": 50, "levy10": 50, "ackley15": 150}


def test_stored_well50_injection_arms_are_complete() -> None:
    data = json.loads((PROJECT_ROOT / "results" / "paper" / "synthetic" / "well50.json").read_text(encoding="utf-8"))
    for arm in ("pi-bo", "prior-to-mean", "prior-warm-start", "prior-greedy"):
        for fam, budget in BUDGETS.items():
            runs = data[arm][fam]
            assert len(runs) == 30 and all(len(t) == budget for t in runs), (arm, fam)
            assert all(a >= b for t in runs for a, b in zip(t, t[1:])), (arm, fam)      # best-so-far regret


def test_panel_c_is_the_synthetic_injection_panel(tmp_path) -> None:
    import fig_ablations as F
    import synthetic_data as SY

    out = F.build(tmp_path)["fig_ablations"]
    labels = [label for _, label, *_ in F.ARMS]
    assert list(out["C"]) == labels
    for method, label, *_ in F.ARMS:
        U, seeds = SY.curves([F.SYN_INJ[method]], "well50")[F.SYN_INJ[method]]
        assert out["C"][label] == (round(float(U.mean(0)[-1]), 4), seeds)
