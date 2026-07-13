"""Read-only comparison: whole-window vs patch-level amplitude rejection.

Motivating question (pilot finding, see memory/livinglab-amplitude-rejection.md):
reject.py's whole-window peak-abs rule discards ~97-100% of the pilot's 10s
windows because ONE noisy second (e.g. a footstep) anywhere in a window fails
the WHOLE window, even when the other 9 seconds are clean EEG.

This script does NOT change reject.py, window.py, or any pipeline stage -- it
reuses the exact same preprocess -> align -> window chain as the real pipeline
(Stages 1-6, unmodified) and then computes two independent stats side by side,
per config.sweep.thresholds_uV:

  1. WHOLE-WINDOW survival (today's reject.py rule): a 10s window survives iff
     its peak-abs over all 18 channels x all 2000 samples is < threshold.
  2. PATCH-LEVEL survival: each window is split into its 10 existing 1s
     patches (cfg.window.samples_per_patch -- already defined for CBraMod's
     patch-based input, just unused by reject.py today); a patch survives iff
     ITS OWN peak-abs is < threshold, independent of its neighbours.

This quantifies how much usable signal (in seconds) whole-window rejection is
currently throwing away alongside each genuinely bad second.

Run:
    python patch_reject_check.py
"""
from __future__ import annotations

import csv

import numpy as np

from src.livinglab_prep.align import align_recording
from src.livinglab_prep.config import load_config
from src.livinglab_prep.discovery import discover_recordings
from src.livinglab_prep.preprocess import preprocess_recording
from src.livinglab_prep.window import make_windows

import mne

mne.set_log_level("ERROR")


def _patch_peaks(X: np.ndarray, n_patches: int, samples_per_patch: int) -> np.ndarray:
    """(18, n_patches*samples_per_patch) -> (n_patches,) per-patch peak-abs uV."""
    n_ch, n_samp = X.shape
    assert n_samp == n_patches * samples_per_patch, (
        f"window has {n_samp} samples, expected {n_patches}x{samples_per_patch}")
    patches = X.reshape(n_ch, n_patches, samples_per_patch)
    return np.max(np.abs(patches), axis=(0, 2))


def main() -> None:
    cfg = load_config()
    recordings = discover_recordings(cfg)
    n_patches = cfg.window.n_patches()
    spp = cfg.window.samples_per_patch
    thresholds = list(cfg.sweep.thresholds_uV)

    window_peaks: dict[str, np.ndarray] = {}
    patch_peaks_all: dict[str, np.ndarray] = {}

    for rec in recordings:
        print(f"[patch_reject_check] === {rec.key} ===")
        pre = preprocess_recording(cfg, rec)
        align = align_recording(cfg, pre)
        windows = make_windows(cfg, pre, align)
        if not windows:
            print(f"[patch_reject_check] {rec.key}: no windows, skipping")
            continue
        stack = np.stack([w.X for w in windows], axis=0)              # (n_win, 18, 2000)
        window_peaks[rec.key] = np.max(np.abs(stack), axis=(1, 2))    # (n_win,)
        patch_peaks_all[rec.key] = np.stack(
            [_patch_peaks(w.X, n_patches, spp) for w in windows], axis=0)  # (n_win, 10)

    keys = sorted(window_peaks)
    if not keys:
        print("[patch_reject_check] no recordings produced windows -- nothing to compare")
        return

    all_win_peaks = np.concatenate([window_peaks[k] for k in keys])
    all_patch_peaks = np.concatenate([patch_peaks_all[k] for k in keys], axis=0)  # (N_win, 10)
    n_windows = all_win_peaks.shape[0]
    n_total_patches = all_patch_peaks.size

    print(f"\n=== WHOLE-WINDOW vs PATCH-LEVEL SURVIVAL "
          f"({n_windows} windows x {n_patches} patches = {n_total_patches} patch-seconds, "
          f"{len(keys)} recordings) ===")
    hdr = f"{'thr uV':>8}{'window surv':>16}{'patch-sec surv':>20}{'window pct':>12}{'patch-sec pct':>16}"
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for thr in thresholds:
        win_kept = int(np.sum(all_win_peaks < thr))
        patch_kept = int(np.sum(all_patch_peaks < thr))
        win_pct = 100 * win_kept / n_windows
        patch_pct = 100 * patch_kept / n_total_patches
        print(f"{thr:>8.0f}{f'{win_kept}/{n_windows}':>16}"
              f"{f'{patch_kept}/{n_total_patches}':>20}{win_pct:>11.1f}%{patch_pct:>15.1f}%")
        rows.append({
            "threshold_uV": thr,
            "n_windows": n_windows, "windows_kept": win_kept, "window_survival_pct": round(win_pct, 2),
            "n_patch_seconds": n_total_patches, "patch_seconds_kept": patch_kept,
            "patch_survival_pct": round(patch_pct, 2),
        })

    out_path = "patch_reject_check_results.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[patch_reject_check] wrote {out_path}")

    # Collateral-damage breakdown at the pilot's baseline 100 uV threshold:
    # how many windows have 0 bad patches (whole-window rule would keep them
    # anyway) vs how many are "mostly clean" (<=1 bad patch/10) yet still
    # discarded whole under today's rule.
    thr0 = 100.0
    bad_patch_count = np.sum(all_patch_peaks >= thr0, axis=1)  # (n_windows,) 0..10
    print(f"\n=== COLLATERAL DAMAGE AT {thr0:.0f} uV (bad patches per 10s window) ===")
    for k in range(n_patches + 1):
        n = int(np.sum(bad_patch_count == k))
        if n:
            print(f"  {k:>2} bad patch(es)/10: {n:>5} windows ({100 * n / n_windows:.1f}%)")
    mostly_clean = int(np.sum(bad_patch_count <= 1))
    print(f"\n  Windows with <=1 bad patch (>=9/10 clean seconds) that whole-window "
          f"rejection discards entirely: {mostly_clean}/{n_windows} "
          f"({100 * mostly_clean / n_windows:.1f}%)")


if __name__ == "__main__":
    main()
