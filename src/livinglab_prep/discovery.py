"""Recording discovery (README.md §1): the patient set is DISCOVERED from
paths.raw_dir, never enumerated in code. Everything downstream (windowing,
reports, and the deferred fold generator) must work unchanged from 2 -> 30
patients, so nothing here may assume a count or a specific patient id.

Filename convention (verified): '<patient>-<cond>_raw.edf', e.g.
'sub-131-a_raw.edf'; the paired CSV is '<patient>-<cond>_<date>.csv'.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import Config

# sub-131-a_raw.edf  ->  patient='sub-131', cond_letter='a'
_EDF_RE = re.compile(r"^(?P<patient>.+)-(?P<cond>[a-zA-Z])_raw\.edf$")


@dataclass(frozen=True)
class Recording:
    patient_id: str          # e.g. 'sub-131'
    condition: str           # canonical upper-case letter, e.g. 'A' / 'C'
    edf_path: Path
    csv_path: Path

    @property
    def key(self) -> str:
        return f"{self.patient_id}-{self.condition.lower()}"


def _find_csv(csv_dir: Path, patient: str, cond_letter: str) -> Path:
    """Locate the task CSV for a recording (date suffix varies)."""
    matches = sorted(csv_dir.glob(f"{patient}-{cond_letter}_*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No task CSV for {patient}-{cond_letter} in {csv_dir} "
            f"(expected '{patient}-{cond_letter}_*.csv')")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous CSVs for {patient}-{cond_letter}: {matches}")
    return matches[0]


def discover_recordings(cfg: Config) -> list[Recording]:
    """Enumerate every EDF recording and pair it with its CSV and condition label."""
    raw_dir, csv_dir = cfg.raw_dir, cfg.csv_dir
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir does not exist: {raw_dir}")

    recordings: list[Recording] = []
    for edf in sorted(raw_dir.glob("*_raw.edf")):
        m = _EDF_RE.match(edf.name)
        if not m:
            raise ValueError(f"EDF name does not match '<patient>-<cond>_raw.edf': {edf.name}")
        patient = m.group("patient")
        cond_letter = m.group("cond")
        cond = cond_letter.upper()
        if cond not in cfg.labels.condition_map:
            raise ValueError(
                f"{edf.name}: condition '{cond}' not in labels.condition_map "
                f"{list(cfg.labels.condition_map)}")
        csv = _find_csv(csv_dir, patient, cond_letter)
        recordings.append(Recording(patient, cond, edf, csv))

    if not recordings:
        raise FileNotFoundError(f"No '*_raw.edf' recordings found in {raw_dir}")
    return recordings


def patient_ids(recordings: list[Recording]) -> list[str]:
    """Unique patient ids, order-stable."""
    seen: list[str] = []
    for r in recordings:
        if r.patient_id not in seen:
            seen.append(r.patient_id)
    return seen
