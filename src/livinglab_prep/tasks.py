"""Task-timestamp CSV parsing (README.md Stage 0/5/6).

CSV columns: Tasks, Start, End  (Start/End are task-PC clock in MICROSECONDS,
i.e. Unix epoch microseconds for this dataset). We parse into TaskSegment objects
with a validity flag so the known pilot defects (zero-duration rows, a corrupt
sub-134-c UPDRS timestamp, tasks outside the recording) are surfaced, not crashed
on. See memory: livinglab-data-issues.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskSegment:
    task_id: str
    start_us: int            # raw task-PC clock (epoch microseconds)
    end_us: int

    @property
    def duration_s(self) -> float:
        return (self.end_us - self.start_us) / 1e6

    @property
    def start_s(self) -> float:
        return self.start_us / 1e6

    @property
    def end_s(self) -> float:
        return self.end_us / 1e6


def parse_task_csv(csv_path: Path) -> list[TaskSegment]:
    """Read all rows verbatim (no filtering here; validity judged later)."""
    segs: list[TaskSegment] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"Tasks", "Start", "End"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{csv_path.name}: expected columns {required}, got {reader.fieldnames}")
        for i, row in enumerate(reader):
            try:
                start = int(row["Start"])
                end = int(row["End"])
            except (TypeError, ValueError) as e:
                raise ValueError(f"{csv_path.name} row {i}: non-integer Start/End: {row}") from e
            segs.append(TaskSegment(row["Tasks"].strip(), start, end))
    return segs


def usable_duration(seg: TaskSegment, recording_len_s: float, min_len_s: float) -> tuple[bool, str]:
    """Judge whether a segment can yield windows given the recording length.

    Returns (usable, reason). A segment is unusable if its duration is
    non-positive (zero-duration logging artifact), longer than the whole
    recording (corrupt epoch), or shorter than one window.
    """
    d = seg.duration_s
    if d <= 0:
        return False, "zero_or_negative_duration"
    if d > recording_len_s:
        return False, "duration_exceeds_recording (corrupt timestamp)"
    if d < min_len_s:
        return False, f"shorter_than_window ({d:.1f}s < {min_len_s:.0f}s)"
    return True, "ok"
