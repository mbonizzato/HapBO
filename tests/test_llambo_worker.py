"""LLAMBO study worker: the candidate frame returned by upstream's candidate sampler reaches upstream unchanged."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_WORKERS = PROJECT_ROOT / "src" / "baselines" / "workers"

# Run in a subprocess: importing the worker's harness redirects the process' stdout to stderr.
_FRAME_CHECK = """
import sys
sys.path.insert(0, {workers!r})
import pandas as pd
import llambo_worker as W

cand = pd.DataFrame({{"x0": [0.1, 0.5], "x1": [0.2, None], "x2": [None, 0.3]}})  # 'x2' is outside the search space
seen = []


class Acq:
    def get_candidate_points(self, observed_configs, observed_fvals, *args, **kwargs):
        seen.append(observed_fvals)
        return cand, 0.0, 0.0


class Llambo:
    acq_func = Acq()


llambo = Llambo()
W._bound_acquisition_scores(llambo)
configs = pd.DataFrame({{"x0": [0.0, 1.0], "x1": [0.5, 0.5]}})
for fvals in ([0.2, 0.7], [3.0, 9.0]):  # scores inside (0, 1) are passed as they are, others rescaled for the prompt
    out = llambo.acq_func.get_candidate_points(configs, pd.DataFrame({{"score": fvals}}))
    assert out[0] is cand and list(out[0].columns) == ["x0", "x1", "x2"] and len(out[0]) == 2, out[0]
assert seen[0]["score"].tolist() == [0.2, 0.7]
assert [round(v, 12) for v in seen[1]["score"].tolist()] == [0.1, 0.9]
"""


def test_candidate_frames_reach_upstream_unchanged(tmp_path: Path) -> None:
    script = tmp_path / "frame_check.py"
    script.write_text(_FRAME_CHECK.format(workers=str(_WORKERS)), encoding="utf-8")
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
