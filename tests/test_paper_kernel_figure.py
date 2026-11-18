"""fig_kernel_bo (paper/fig_acquisition.py): HapBO matched vs vanilla BO (EI) under three GP kernels.

Panel F (Matérn-5/2) reads the acquisition study, the default kernel, and reproduces panel A of
fig_acq_across; G and H read the kernel-rbf and kernel-matern12 studies. The plotted end points are
recomputed here straight from the compact data, and the y axis is shared within each row.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")

DATA = ROOT / "results" / "paper"
STUDIES = {"matern52": "acquisition", "rbf": "kernel-rbf", "matern12": "kernel-matern12"}
ARMS = {r"$\alpha$-$\pi$": "alpha-pi-raw-ei-matched", "vanilla": "vanilla-bo-ei"}
# (q3, final)
EXPECTED = {
    ("hpo", "matern52"): ((0.0128, 0.0030), (0.0630, 0.0103)),
    ("hpo", "rbf"): ((0.0128, 0.0031), (0.0651, 0.0132)),
    ("hpo", "matern12"): ((0.0135, 0.0030), (0.0647, 0.0141)),
    ("cortical", "matern52"): ((0.8155, 0.9511), (0.5838, 0.9443)),
    ("cortical", "rbf"): ((0.7586, 0.9502), (0.5902, 0.9498)),
    ("cortical", "matern12"): ((0.8494, 0.9639), (0.5838, 0.9536)),
}


def _load(group: str, study: str) -> dict:
    return json.loads((DATA / group / f"{study}.json").read_text(encoding="utf-8"))


def _independent(group: str, study: str, method: str):
    """(q3, final, n): HPOBench mean over tasks of 1 - best/grid max (all 40 tasks, 16 queries); cortical mean over
    maps of best / map optimum (all 40 rat-muscle maps, 12 queries)."""
    runs = _load(group, study)["methods"][method]
    if group == "hpo":
        stats = _load("hpo", "task_stats")
        per_task = []
        for unit, rows in runs["budget-0016"].items():
            curves = [1.0 - np.maximum.accumulate(r["y"][:16]) / stats[unit]["max"] for r in rows
                      if r["status"] == "completed"]
            per_task.append(np.mean(curves, axis=0))
        mean = np.mean(per_task, axis=0)
        return round(float(mean[2]), 4), round(float(mean[-1]), 4), len(per_task)
    opt = _load("cortical", "map_optimum")
    key = min((k for k in runs if int(k.split("-")[1]) >= 12), key=lambda k: int(k.split("-")[1]))
    per_map = []
    for unit, rows in runs[key].items():
        curves = [np.maximum.accumulate(r["y"][:12]) / opt[unit] for r in rows if r["status"] == "completed"]
        per_map.append(np.mean(curves, axis=0))
    mean = np.mean(per_map, axis=0)
    return round(float(mean[2]), 4), round(float(mean[-1]), 4), len(per_map)


@pytest.mark.parametrize("group,units,budget", [("hpo", 40, "budget-0016"), ("cortical", 40, "budget-0012")])
@pytest.mark.parametrize("kernel", ["rbf", "matern12"])
def test_kernel_study_data_pairs_the_acquisition_ei_runs(group: str, units: int, budget: str, kernel: str) -> None:
    data = _load(group, f"kernel-{kernel}")
    acquisition = _load(group, "acquisition")["methods"]
    assert sorted(data["methods"]) == sorted(ARMS.values())
    assert data["benchmark"] == _load(group, "acquisition")["benchmark"]
    for method, by_budget in data["methods"].items():
        assert list(by_budget) == [budget] and len(by_budget[budget]) == units
        ref_budget = next(iter(acquisition[method]))
        assert sorted(by_budget[budget]) == sorted(acquisition[method][ref_budget])
        for unit, rows in by_budget[budget].items():
            assert [r["seed"] for r in rows] == list(range(1001, 1011))
            assert all(r["status"] == "completed" and len(r["y"]) == int(budget.split("-")[1]) for r in rows)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    import fig_acquisition
    import style

    figures = {}
    save = style.save

    def capture(fig, path_noext):
        figures[Path(path_noext).name] = fig
        save(fig, path_noext)

    style.save = capture
    try:
        numbers = fig_acquisition.build(tmp_path_factory.mktemp("figures"))
    finally:
        style.save = save
    return numbers, figures


def test_kernel_figure_numbers(built) -> None:
    numbers, _ = built
    kernel = numbers["fig_kernel_bo"]
    assert len(kernel) == 12
    for (group, name), (ours, vanilla) in EXPECTED.items():
        for (label, method), expected in zip(ARMS.items(), (ours, vanilla)):
            q3, final, n = kernel[f"{group} {name} {label}"]
            assert (q3, final) == expected
            assert (q3, final, n) == _independent(group, STUDIES[name], method)
            assert n == 40


def test_panel_f_reproduces_panel_a(built) -> None:
    numbers, _ = built
    for group in ("hpo", "cortical"):
        for label in ARMS:
            assert numbers["fig_kernel_bo"][f"{group} matern52 {label}"] == numbers["fig_acq_across"][f"{group} EI {label}"]


def test_kernel_figure_layout(built) -> None:
    _, figures = built
    fig = figures["fig_kernel_bo"]
    axes = np.asarray(fig.axes).reshape(2, 3)
    assert [ax.get_title(loc="left") for ax in axes[0]] == ["F   Matérn-5/2", "G   RBF", "H   Matérn-1/2"]
    for row in axes:
        assert all(row[0].get_shared_y_axes().joined(row[0], ax) for ax in row[1:])
        assert all(row[0].get_shared_x_axes().joined(row[0], ax) for ax in row[1:])
    assert not axes[0, 0].get_shared_y_axes().joined(axes[0, 0], axes[1, 0])
