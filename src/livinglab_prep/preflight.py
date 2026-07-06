"""STAGE 0 - pre-flight task-duration check (README.md §7).

Parses every task CSV, computes the distribution of task-segment durations
against each recording's length, and writes processed/_reports/task_durations.csv
plus a printed summary. Blocks (raises) if a meaningful fraction of GENUINE task
segments are shorter than window.length_s (i.e. the window is unsafe) -- zero-
duration and corrupt-timestamp artifacts are reported but excluded from that
safety test, since they are data defects, not evidence the window is too long.
"""
from __future__ import annotations

import csv as _csv
from dataclasses import dataclass
from pathlib import Path

import mne

from .config import Config
from .discovery import discover_recordings, Recording
from .tasks import parse_task_csv, usable_duration


@dataclass
class SegmentReport:
    recording_key: str
    patient_id: str
    condition: str
    task_id: str
    duration_s: float
    recording_len_s: float
    usable: bool
    reason: str


def _recording_len_s(edf_path: Path) -> float:
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    return raw.n_times / raw.info["sfreq"]


def run_preflight(cfg: Config) -> list[SegmentReport]:
    recordings = discover_recordings(cfg)
    min_len = float(cfg.window.length_s)
    rows: list[SegmentReport] = []

    for rec in recordings:
        rec_len = _recording_len_s(rec.edf_path)
        for seg in parse_task_csv(rec.csv_path):
            usable, reason = usable_duration(seg, rec_len, min_len)
            rows.append(SegmentReport(
                rec.key, rec.patient_id, rec.condition, seg.task_id,
                seg.duration_s, rec_len, usable, reason))

    _write_report(cfg, rows)
    _summarise_and_gate(cfg, rows)
    return rows


def _write_report(cfg: Config, rows: list[SegmentReport]) -> None:
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.reports_dir / "task_durations.csv"
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["recording", "patient_id", "condition", "task_id",
                    "duration_s", "recording_len_s", "usable", "reason"])
        for r in rows:
            w.writerow([r.recording_key, r.patient_id, r.condition, r.task_id,
                        f"{r.duration_s:.3f}", f"{r.recording_len_s:.1f}",
                        r.usable, r.reason])
    print(f"[stage0] wrote {out}")


def _summarise_and_gate(cfg: Config, rows: list[SegmentReport]) -> None:
    total = len(rows)
    usable = [r for r in rows if r.usable]
    zero = [r for r in rows if r.reason.startswith("zero")]
    corrupt = [r for r in rows if "corrupt" in r.reason]
    too_short = [r for r in rows if r.reason.startswith("shorter_than_window")]

    print(f"[stage0] task segments: {total} total | {len(usable)} usable | "
          f"{len(zero)} zero-duration | {len(corrupt)} corrupt | "
          f"{len(too_short)} shorter-than-{cfg.window.length_s}s")
    for r in zero + corrupt + too_short:
        print(f"[stage0]   DROP {r.recording_key} '{r.task_id}': {r.reason}")

    # Safety gate: of the GENUINE tasks (exclude zero/corrupt artifacts), the
    # fraction that are long enough for at least one window must be healthy.
    genuine = [r for r in rows if not (r.reason.startswith("zero") or "corrupt" in r.reason)]
    if genuine:
        fit_frac = sum(r.usable for r in genuine) / len(genuine)
        print(f"[stage0] genuine tasks fitting >= 1 window: {fit_frac:.0%} "
              f"(need >= {cfg.align.min_tasks_fit_frac:.0%})")
        if fit_frac < cfg.align.min_tasks_fit_frac:
            raise SystemExit(
                f"[stage0] ABORT: only {fit_frac:.0%} of genuine task segments are "
                f">= window.length_s ({cfg.window.length_s}s). Too much data would be "
                f"discarded -- reduce window.length_s (e.g. 5) in config and re-run.")
    print("[stage0] pre-flight OK: window length is safe.")


if __name__ == "__main__":
    from .config import load_config
    run_preflight(load_config())
