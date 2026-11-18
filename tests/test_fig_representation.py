"""fig_representation: arms, studies and run counts of the representation ablation; ranks recomputed per task."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "paper"))


def test_every_arm_covers_the_40_tasks_with_10_seeds(tmp_path) -> None:
    import fig_representation as R

    out = R.build(tmp_path)["fig_representation"]
    assert (tmp_path / "fig_representation.pdf").is_file() and (tmp_path / "fig_representation.png").is_file()
    labels = list(R.HPO_ARMS)
    for c in ("std", "adv"):
        assert out["tasks"][c] == 40
        assert out["counts"][c] == {a: 400 for a in labels}
        assert list(out[f"{c}_rank_q16"]) == labels
        assert abs(sum(out[f"{c}_rank_q16"].values()) - 15.0) < 0.02        # ranks 1..5 per task
    assert list(out["synthetic"]) == [lab for lab, _ in R.SYN_ARMS]


def test_inverted_map_condition_reads_the_misleading_representation_studies() -> None:
    import fig_representation as R

    adv = {a: studies["adv"] for a, (studies, _) in R.HPO_ARMS.items()}
    assert adv == {"full-PSL": "hapbo-inverted", "point-only": "representation-pointonly-misleading",
                   "gaussian-only": "representation-gaussian-misleading",
                   "direct-LLM": "representation-directllm-misleading", "vanilla BO": "vanilla-random"}


def test_final_rank_matches_an_independent_count() -> None:
    import fig_hpo as F
    import fig_representation as R
    import hpo_data as H

    data, _, keys, arms = R.hpo_load("adv")
    for a in arms:                                                            # independent seed-mean regret
        studies, method = R.HPO_ARMS[a]
        for fam, task in keys[:3]:
            runs = H.runs("hpo", studies["adv"], method, H.unit(fam, task))
            best = [np.maximum.accumulate(r["y"][:H.BUDGET]) for r in runs if r["status"] == "completed"]
            assert np.allclose(H.grid_max(fam, task) - np.mean(best, axis=0), data[a][(fam, task)])
    final = np.mean([rankdata(np.round(np.array([data[a][k][-1] for a in arms]) / 1e-9)) for k in keys], axis=0)
    rk = F.ranks(data, keys, arms)
    assert np.allclose(final, [rk[a][-1] for a in arms])
