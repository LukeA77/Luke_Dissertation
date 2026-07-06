"""Cohort discovery for the shared LivingLab_PD drive (post-pilot cohort audit).

Layout differs from the pilot's flat `paths.raw_dir`: each patient has a
top-level `sub-<ID>-<cond>/eeg/<task-subfolder>/` directory holding one
`sub-<ID>-<cond>_raw.edf` + `sub-<ID>-<cond>_raw.csv` pair. The task subfolder
was renamed partway through the study (`livinglab` -> `living-lab-tasks`);
both names hold identical content, so both are checked. `eeg/` also contains
several unrelated task subfolders (Blocks, catsndogs, Motor, ...) which are
never matched here.

Restricted to an explicit patient allowlist (not a blind rglob) because the
shared drive also hosts a different sub-study on DSI-only patients that must
never be swept in by accident -- see memory/livinglab-data-issues.md and the
2026-07-06 cohort audit.
"""
from __future__ import annotations

from pathlib import Path

from .discovery import Recording

TASK_SUBFOLDERS = ("livinglab", "living-lab-tasks")


def _find_recording(root: Path, patient: str, cond: str) -> Recording | None:
    session_dir = root / f"{patient}-{cond}" / "eeg"
    for sub in TASK_SUBFOLDERS:
        edf = session_dir / sub / f"{patient}-{cond}_raw.edf"
        csv = session_dir / sub / f"{patient}-{cond}_raw.csv"
        if edf.exists() and csv.exists():
            return Recording(patient, cond.upper(), edf, csv)
    return None


def discover_cohort_recordings(
    root: Path, patient_ids: list[str], conditions: tuple[str, ...] = ("a", "c")
) -> tuple[list[Recording], list[str]]:
    """Find EDF+CSV pairs for an explicit patient allowlist under `root`.

    Returns (recordings, missing): `missing` lists "<patient>-<cond>" keys for
    which no pair was found under either task-subfolder naming -- reported,
    never silently skipped, since absence can mean DSI-only (different
    sub-study) or a genuine gap (see sub-70/71/130 in the cohort audit).
    """
    recordings: list[Recording] = []
    missing: list[str] = []
    for patient in patient_ids:
        for cond in conditions:
            rec = _find_recording(root, patient, cond)
            if rec is None:
                missing.append(f"{patient}-{cond}")
            else:
                recordings.append(rec)
    return recordings, missing
