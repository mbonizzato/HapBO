"""NeuroMOSAIC Dataset 1 rat cortical-stimulation adapter."""

from __future__ import annotations

from pathlib import Path

from .neuromosaic_common import OfflineOptimizationProblem
from .neuromosaic_dataset1 import load_neuromosaic_dataset1_problem


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEUROMOSAIC_DATASET1_RAT_DIRECTORY = (
    PROJECT_ROOT / "benchmark" / "NeuroMOSAIC" / "dataset1" / "rat"
)
DATASET_ID = "neuromosaic-dataset1-rat"
SUBJECT_FILES = {
    "rat1": "rat1_M1_190716.mat",
    "rat2": "rat2_M1_190617.mat",
    "rat3": "rat3_M1_190728.mat",
    "rat4": "rat4_M1_191109.mat",
    "rat5": "rat5_M1_191112.mat",
    "rat6": "rat6_M1_200218.mat",
}
DEFAULT_SUBJECT = "rat1"
DEFAULT_SUBJECT_PATH = (
    NEUROMOSAIC_DATASET1_RAT_DIRECTORY / SUBJECT_FILES[DEFAULT_SUBJECT]
)
DEFAULT_MUSCLE = "LeftMedialisGastrocnemius"


def resolve_neuromosaic_dataset1_rat_subject(
    subject: str = DEFAULT_SUBJECT,
    *,
    subject_path: str | Path | None = None,
) -> Path:
    """Resolve a Dataset 1 rat ID or an explicit MATLAB file."""

    if subject_path is not None:
        return Path(subject_path).expanduser().resolve()
    try:
        filename = SUBJECT_FILES[subject]
    except KeyError as exc:
        raise ValueError(
            f"unknown NeuroMOSAIC Dataset 1 rat subject {subject!r}; "
            f"choose one of {list(SUBJECT_FILES)}"
        ) from exc
    return NEUROMOSAIC_DATASET1_RAT_DIRECTORY / filename


def load_neuromosaic_dataset1_rat_problem(
    subject_path: str | Path | None = None,
    muscle: str = DEFAULT_MUSCLE,
    *,
    subject: str = DEFAULT_SUBJECT,
    noise_seed: int | None = None,
) -> OfflineOptimizationProblem:
    """Load one Dataset 1 rat and one muscle as an offline BO task.

    ``noise_seed`` (optional) switches the objective to noisy single-trial
    polling (see ``load_neuromosaic_dataset1_problem``); default is the
    deterministic mean.
    """

    path = resolve_neuromosaic_dataset1_rat_subject(
        subject,
        subject_path=subject_path,
    )
    return load_neuromosaic_dataset1_problem(
        path,
        muscle,
        dataset_id=DATASET_ID,
        source="NeuroMOSAIC Dataset 1 (rat M1 cortical stimulation)",
        noise_seed=noise_seed,
    )


__all__ = [
    "DATASET_ID",
    "DEFAULT_MUSCLE",
    "DEFAULT_SUBJECT",
    "DEFAULT_SUBJECT_PATH",
    "NEUROMOSAIC_DATASET1_RAT_DIRECTORY",
    "OfflineOptimizationProblem",
    "SUBJECT_FILES",
    "load_neuromosaic_dataset1_rat_problem",
    "resolve_neuromosaic_dataset1_rat_subject",
]
