"""Cohort-scale amplitude-rejection sweep (follow-up to the 2-patient pilot).

The pilot found the CBraMod 100 uV whole-window peak-abs threshold discards
97-100% of windows (see memory/livinglab-amplitude-rejection.md). This script
runs the IDENTICAL sweep (src/livinglab_prep/reject.py's rule, over
config.sweep.thresholds_uV) on the 18-patient candidate cohort discovered on
the shared LivingLab_PD drive, to check whether that near-total rejection is
pilot-specific or cohort-wide.

Per recording (matches src/livinglab_prep exactly, STRICT order, no ICA -- the
pilot showed ICA recovers only a few points in the 150-300 uV band and is too
slow to re-run at 36-recording scale on this machine):

    load -> linked-ears reref -> channel-select(18) -> resample 300->200
    -> bandpass 0.3-75 -> notch 50 -> task-locked windowing (align.py/window.py)
    -> whole-window peak-abs sweep across config.sweep.thresholds_uV

A recording that fails alignment/parsing is logged and skipped so one bad
CSV cannot abort the other 35.

Run:
    python cohort_reject_sweep.py --root "C:\\Data\\LivingLab_PD"
    python cohort_reject_sweep.py --root <path> --patients sub-131,sub-134   # smoke test subset
"""
from __future__ import annotations

import argparse
import csv
import traceback
from pathlib import Path

import numpy as np

from src.livinglab_prep.align import align_recording
from src.livinglab_prep.cohort_discovery import discover_cohort_recordings
from src.livinglab_prep.config import Config, load_config
from src.livinglab_prep.discovery import Recording
from src.livinglab_prep.preprocess import (
    PreprocessedRecording,
    extract_uV,
    filter_resample,
    load_raw,
    select_channels,
)
from src.livinglab_prep.reference import apply_reference
from src.livinglab_prep.window import make_windows

import mne

mne.set_log_level("ERROR")

# 18-patient candidate cohort with COMPLETE ON/OFF EDF+CSV pairs (2026-07-06
# audit). Excludes 3 incomplete-pair patients (sub-70, sub-71, sub-130) and
# all DSI-only patients (different sub-study on the same drive) -- see
# memory/livinglab-data-issues.md. Not yet confirmed as ground truth by
# Dr Haar; treat results as provisional until that confirmation lands.
COHORT_PATIENTS = [
    "sub-34", "sub-42", "sub-48", "sub-51", "sub-53",
    "sub-111", "sub-114", "sub-115", "sub-116", "sub-117", "sub-119", "sub-121",
    "sub-123", "sub-125", "sub-129", "sub-131", "sub-134", "sub-135",
]


def _whole_window_peaks(cfg: Config, rec: Recording) -> np.ndarray:
    """-> (n_windows,) whole-window peak-abs values in uV, linked-ears reref."""
    raw = load_raw(cfg, rec)
    meas_date = raw.info.get("meas_date")
    raw = apply_reference(cfg, raw, rec.key)
    raw = select_channels(cfg, raw, rec.key)
    raw = filter_resample(cfg, raw, rec.key)
    data = extract_uV(cfg, raw, rec.key)
    pre = PreprocessedRecording(
        rec=rec, data_uV=data, ch_names=list(raw.ch_names),
        sfreq=cfg.signal.target_sfreq, n_samples=data.shape[1], meas_date=meas_date)

    align = align_recording(cfg, pre)
    windows = make_windows(cfg, pre, align)
    if not windows:
        return np.empty(0)
    stack = np.stack([w.X for w in windows], axis=0)
    return np.max(np.abs(stack), axis=(1, 2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cohort_reject_sweep")
    parser.add_argument("--root", required=True, help="root of the LivingLab_PD shared drive")
    parser.add_argument("--config", default=None, help="path to pipeline.yaml")
    parser.add_argument("--patients", default=None,
                        help="comma-separated patient-id override (default: the 18-patient cohort)")
    parser.add_argument("--out", default="cohort_sweep_survival.csv")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    root = Path(args.root)
    patients = args.patients.split(",") if args.patients else COHORT_PATIENTS

    recordings, missing = discover_cohort_recordings(root, patients)
    print(f"[cohort_reject_sweep] config_hash={cfg.config_hash} git_sha={cfg.git_sha}")
    print(f"[cohort_reject_sweep] found {len(recordings)}/{2 * len(patients)} recordings under {root}")
    if missing:
        print(f"[cohort_reject_sweep] MISSING (no EDF+CSV under either task-folder naming): {missing}")

    thresholds = list(cfg.sweep.thresholds_uV)
    peaks: dict[str, np.ndarray] = {}
    failed: list[tuple[str, str]] = []

    for rec in recordings:
        print(f"\n[cohort_reject_sweep] === {rec.key} ===")
        try:
            peaks[rec.key] = _whole_window_peaks(cfg, rec)
        except Exception as e:
            print(f"[cohort_reject_sweep] FAILED {rec.key}: {e}")
            traceback.print_exc()
            failed.append((rec.key, str(e)))

    _write_csv(args.out, thresholds, peaks)
    _print_summary(thresholds, peaks, failed, missing)


def _write_csv(out_path: str, thresholds: list[float], peaks: dict[str, np.ndarray]) -> None:
    keys = sorted(peaks)
    all_peaks = np.concatenate([peaks[k] for k in keys]) if keys else np.empty(0)
    rows = []
    for scope, pk in [(k, peaks[k]) for k in keys] + [("OVERALL", all_peaks)]:
        n = pk.shape[0]
        for thr in thresholds:
            kept = int(np.sum(pk < thr)) if n else 0
            rows.append({
                "recording": scope, "threshold_uV": thr, "n_windows": n,
                "n_kept": kept, "survival_frac": round(kept / n, 6) if n else 0.0,
            })
    if not rows:
        print("[cohort_reject_sweep] nothing to write (no recordings succeeded)")
        return
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[cohort_reject_sweep] wrote {out_path}")


def _print_summary(thresholds: list[float], peaks: dict[str, np.ndarray],
                   failed: list[tuple[str, str]], missing: list[str]) -> None:
    keys = sorted(peaks)
    if keys:
        all_peaks = np.concatenate([peaks[k] for k in keys])
        n_total = all_peaks.shape[0]
        print(f"\n=== COHORT WINDOW SURVIVAL vs THRESHOLD "
              f"(linked-ears, n={n_total} windows, {len(keys)} recordings) ===")
        hdr = f"{'threshold uV':>12}{'survive':>18}{'pct':>8}"
        print(hdr); print("-" * len(hdr))
        for thr in thresholds:
            kept = int(np.sum(all_peaks < thr))
            print(f"{thr:>12.0f}{f'{kept}/{n_total}':>18}{100 * kept / n_total:>7.1f}%")

        print("\n=== PER-RECORDING SURVIVAL @ 100 uV (pilot's baseline threshold) ===")
        for k in keys:
            pk = peaks[k]
            n = pk.shape[0]
            if n:
                kept = int(np.sum(pk < 100))
                print(f"  {k:<14} {kept}/{n} ({100 * kept / n:.1f}%)")
            else:
                print(f"  {k:<14} 0 windows")
    else:
        print("\n[cohort_reject_sweep] no recordings succeeded -- nothing to summarise")

    if failed:
        print(f"\n=== FAILED RECORDINGS ({len(failed)}) ===")
        for k, e in failed:
            print(f"  {k}: {e}")
    if missing:
        print(f"\n=== MISSING FROM DRIVE ({len(missing)}) === {missing}")


if __name__ == "__main__":
    main()
