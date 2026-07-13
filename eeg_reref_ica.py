"""Threshold sweep BEFORE ICA -> ICA artifact removal -> threshold sweep AFTER ICA.

Brackets the ICA step (src/livinglab_prep/ica.py :: clean_with_ica) with an
identical window-survival sweep before and after, so ICA's contribution is
separated from simply raising the amplitude threshold. Everything is on the
re-referenced (linked-ears) + filtered Living Lab data and follows the STRICT
order:

    reference -> bandpass 0.3-75 -> notch 50 -> [ICA] -> channel-select(18)
    -> resample 300->200 -> window -> amplitude-reject(sweep).

The two sweeps differ ONLY in the presence of ICA. No final threshold is chosen
here -- the grid (config.sweep.thresholds_uV) reports survival across a range and
the choice is made later with the supervisor.

Outputs (config-derived dir processed/reref-<scheme>_ica/, kept separate from the
reref-only and original Pz outputs):
  * sweep_survival.csv        - survival at every threshold, before vs after ICA
  * per_channel_amplitude.csv - median peak amplitude per channel for 3 conditions
  * ica_report_<key>.json     - removed components, labels, probabilities, seed
  * _reports/ica_plots/       - component topomaps + properties (human review)
  * provenance.json           - config, git sha, timestamp, input files
  * determinism_hash.txt      - hash of the numeric outputs (re-run must match)

Run:
    python eeg_reref_ica.py               # uses config/pipeline.yaml
    python eeg_reref_ica.py --config <path>
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.livinglab_prep.align import align_recording
from src.livinglab_prep.config import Config, load_config
from src.livinglab_prep.discovery import Recording, discover_recordings
from src.livinglab_prep.ica import ICAReport, clean_with_ica
from src.livinglab_prep.preprocess import (
    PreprocessedRecording,
    extract_uV,
    load_raw,
    select_channels,
)
from src.livinglab_prep.reference import apply_reference
from src.livinglab_prep.window import make_windows

import mne

mne.set_log_level("ERROR")

# Condition keys used consistently across tables.
COND_PZ = "pz"                    # original single-Pz reference (no reref, no ICA)
COND_EARS = "linkedears"          # linked-ears reref only (the BEFORE sweep)
COND_ICA = "linkedears_ica"       # linked-ears reref + ICA (the AFTER sweep)


def _filtered_raw(cfg: Config, rec: Recording, *, do_reref: bool) -> mne.io.BaseRaw:
    """Load -> (optional reref) -> bandpass -> notch, on ALL channels at orig_sfreq."""
    raw = load_raw(cfg, rec)
    if do_reref:
        raw = apply_reference(cfg, raw, rec.key)
    bp = cfg.signal.bandpass
    raw.filter(l_freq=bp.l_freq, h_freq=bp.h_freq, picks="eeg",
               method="fir", fir_design=bp.fir_design, phase=bp.phase, verbose="ERROR")
    raw.notch_filter(freqs=[cfg.signal.notch_freq], picks="eeg", verbose="ERROR")
    return raw


def _to_preprocessed(cfg: Config, raw18: mne.io.BaseRaw, rec: Recording) -> PreprocessedRecording:
    """Resample an 18-channel (canonical) Raw to target_sfreq and package for windowing."""
    meas_date = raw18.info.get("meas_date")
    raw18.resample(cfg.signal.target_sfreq, verbose="ERROR")
    data = extract_uV(cfg, raw18, rec.key)
    return PreprocessedRecording(
        rec=rec, data_uV=data, ch_names=list(raw18.ch_names),
        sfreq=cfg.signal.target_sfreq, n_samples=data.shape[1], meas_date=meas_date)


def _window_stack(cfg: Config, pre: PreprocessedRecording) -> np.ndarray:
    """Task-locked windows (identical logic to the pipeline) -> (n_win, 18, spw)."""
    align = align_recording(cfg, pre)
    windows = make_windows(cfg, pre, align)
    if not windows:
        spw = cfg.window.samples_per_window(cfg.signal.target_sfreq)
        return np.empty((0, len(cfg.channels.keep), spw), dtype=np.float32)
    return np.stack([w.X for w in windows], axis=0)


def _whole_window_peaks(stack: np.ndarray) -> np.ndarray:
    """Per-window peak abs value over all channels+samples -> (n_win,)."""
    if stack.shape[0] == 0:
        return np.empty(0)
    return np.max(np.abs(stack), axis=(1, 2))


def _per_channel_median_peak(stack: np.ndarray) -> np.ndarray:
    """Median over windows of per-window per-channel max|x| -> (n_channels,)."""
    if stack.shape[0] == 0:
        return np.full(stack.shape[1], np.nan)
    return np.median(np.max(np.abs(stack), axis=2), axis=0)


def _survival(peaks: np.ndarray, threshold: float) -> tuple[int, int]:
    """(_n_kept, n_total): a window is kept iff its whole-window peak < threshold."""
    n = peaks.shape[0]
    kept = int(np.sum(peaks < threshold)) if n else 0
    return kept, n


def _git_sha(root: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "no-commit"
    except Exception:
        return "no-git"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="eeg_reref_ica")
    parser.add_argument("--config", default=None, help="path to pipeline.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if not cfg.ica.enabled:
        raise SystemExit("[eeg_reref_ica] ica.enabled is false; nothing to do.")
    if not cfg.reref.enabled:
        raise SystemExit("[eeg_reref_ica] reref.ref_channels is empty; this experiment "
                         "assumes the linked-ears reference is active.")

    out_dir = cfg.ica_run_dir
    plot_dir = out_dir / "_reports" / "ica_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = list(cfg.sweep.thresholds_uV)
    channels = list(cfg.channels.keep)
    print(f"[eeg_reref_ica] config_hash={cfg.config_hash} git_sha={cfg.git_sha}")
    print(f"[eeg_reref_ica] out={out_dir}")
    print(f"[eeg_reref_ica] sweep grid (uV): {thresholds}\n")

    recordings = discover_recordings(cfg)

    # peaks[cond][key] -> (n_win,) whole-window peaks ; medians[cond][key] -> (18,)
    peaks: dict[str, dict[str, np.ndarray]] = {COND_EARS: {}, COND_ICA: {}}
    medians: dict[str, dict[str, np.ndarray]] = {COND_PZ: {}, COND_EARS: {}, COND_ICA: {}}
    ica_reports: list[ICAReport] = []

    scalp_labels = [cfg.channels.rename_from_pattern.format(name=n) for n in channels]

    for rec in recordings:
        print(f"\n[eeg_reref_ica] === {rec.key} ===")

        # ---- linked-ears reref + filter (shared by BEFORE sweep and ICA) ----
        raw_ref = _filtered_raw(cfg, rec, do_reref=True)
        # Reduce to the 18 scalp electrodes before ICA. The dropped channels
        # (accelerometers, CM, aux X-sensors, Trigger/Event, and the degenerate
        # A1/A2/Pz) are non-scalp or degenerate and NEVER enter the decomposition,
        # so this is byte-identical to fitting ICA on the full montage -- it only
        # bounds memory. ICA thus sees every usable scalp channel.
        raw_ref.pick(scalp_labels)

        # BEFORE ICA: linked-ears only. Extract stats, then free before fitting.
        pre_before = _to_preprocessed(cfg, select_channels(cfg, raw_ref.copy(), rec.key), rec)
        stack_before = _window_stack(cfg, pre_before)
        peaks[COND_EARS][rec.key] = _whole_window_peaks(stack_before)
        medians[COND_EARS][rec.key] = _per_channel_median_peak(stack_before)
        n_before = stack_before.shape[0]
        del stack_before, pre_before
        gc.collect()

        # ICA: clean, then window
        cleaned, report = clean_with_ica(cfg, raw_ref, rec.key, report_dir=plot_dir)
        ica_reports.append(report)
        del raw_ref
        gc.collect()
        pre_after = _to_preprocessed(cfg, cleaned, rec)
        stack_after = _window_stack(cfg, pre_after)
        if stack_after.shape[0] != n_before:
            raise AssertionError(
                f"{rec.key}: window count differs before/after ICA "
                f"({n_before} vs {stack_after.shape[0]})")
        peaks[COND_ICA][rec.key] = _whole_window_peaks(stack_after)
        medians[COND_ICA][rec.key] = _per_channel_median_peak(stack_after)
        del cleaned, stack_after, pre_after
        gc.collect()

        # ---- original Pz reference (amplitude comparison only, no sweep) ----
        raw_pz = _filtered_raw(cfg, rec, do_reref=False)
        raw_pz.pick(scalp_labels)
        pre_pz = _to_preprocessed(cfg, select_channels(cfg, raw_pz, rec.key), rec)
        stack_pz = _window_stack(cfg, pre_pz)
        medians[COND_PZ][rec.key] = _per_channel_median_peak(stack_pz)
        del raw_pz, stack_pz, pre_pz
        gc.collect()

    keys = [r.key for r in recordings]
    sweep_rows = _write_sweep_csv(out_dir, keys, thresholds, peaks)
    _write_amplitude_csv(out_dir, keys, channels, medians)
    _write_survival_plot(out_dir, keys, thresholds, peaks)
    _write_ica_reports(out_dir, ica_reports)
    _write_provenance(cfg, recordings, out_dir)
    _write_determinism_hash(out_dir, sweep_rows, ica_reports)

    _print_summary(keys, thresholds, peaks, channels, medians, ica_reports, out_dir)


# --------------------------------------------------------------------------- IO

def _write_sweep_csv(out_dir: Path, keys: list[str], thresholds: list[float],
                     peaks: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    """Write per-recording + overall survival at each threshold, before vs after."""
    rows: list[dict] = []
    for cond in (COND_EARS, COND_ICA):
        all_peaks = np.concatenate([peaks[cond][k] for k in keys]) if keys else np.empty(0)
        for scope, pk in [(k, peaks[cond][k]) for k in keys] + [("OVERALL", all_peaks)]:
            for thr in thresholds:
                kept, n = _survival(pk, thr)
                rows.append({
                    "condition": cond, "recording": scope, "threshold_uV": thr,
                    "n_windows": n, "n_kept": kept,
                    "survival_frac": round(kept / n, 6) if n else 0.0,
                    "rejection_frac": round(1 - kept / n, 6) if n else 0.0,
                })
    out = out_dir / "sweep_survival.csv"
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[eeg_reref_ica] wrote {out}")
    return rows


def _write_amplitude_csv(out_dir: Path, keys: list[str], channels: list[str],
                         medians: dict[str, dict[str, np.ndarray]]) -> None:
    """Per-channel median peak amplitude for Pz / linked-ears / linked-ears+ICA."""
    out = out_dir / "per_channel_amplitude.csv"
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["recording", "channel", "median_peak_pz_uV",
                    "median_peak_linkedears_uV", "median_peak_linkedears_ica_uV"])
        for key in keys:
            for i, ch in enumerate(channels):
                w.writerow([key, ch,
                            f"{medians[COND_PZ][key][i]:.3f}",
                            f"{medians[COND_EARS][key][i]:.3f}",
                            f"{medians[COND_ICA][key][i]:.3f}"])
    print(f"[eeg_reref_ica] wrote {out}")


def _write_survival_plot(out_dir: Path, keys: list[str], thresholds: list[float],
                         peaks: dict[str, dict[str, np.ndarray]]) -> None:
    """One figure: overall window survival vs threshold, before vs after ICA."""
    all_before = np.concatenate([peaks[COND_EARS][k] for k in keys])
    all_after = np.concatenate([peaks[COND_ICA][k] for k in keys])
    n = all_before.shape[0]
    surv_before = [100 * np.mean(all_before < t) for t in thresholds]
    surv_after = [100 * np.mean(all_after < t) for t in thresholds]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, surv_before, "o-", color="C0", label="linked-ears (before ICA)")
    ax.plot(thresholds, surv_after, "s-", color="C1", label="linked-ears + ICA (after)")
    ax.set_xlabel("Amplitude rejection threshold (uV, whole-window peak-abs)")
    ax.set_ylabel(f"Window survival (%)  [n={n} windows]")
    ax.set_title("Window survival vs threshold: effect of ICA (all recordings)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = out_dir / "sweep_survival.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[eeg_reref_ica] wrote {out}")


def _write_ica_reports(out_dir: Path, reports: list[ICAReport]) -> None:
    for r in reports:
        out = out_dir / f"ica_report_{r.key}.json"
        out.write_text(json.dumps(r.as_dict(), indent=2), encoding="utf-8")
    print(f"[eeg_reref_ica] wrote {len(reports)} ICA reports -> {out_dir}")


def _write_provenance(cfg: Config, recordings: list[Recording], out_dir: Path) -> None:
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "threshold_sweep_before_ica__ica__threshold_sweep_after_ica",
        "reference_scheme": cfg.reref.scheme,
        "ref_channels": list(cfg.reref.ref_channels),
        "git_sha": cfg.git_sha,
        "config_hash": cfg.config_hash,
        "out_dir": str(out_dir),
        "inputs": [{"key": r.key, "edf": str(r.edf_path), "csv": str(r.csv_path)}
                   for r in recordings],
        "config": cfg.model_dump(mode="json"),
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[eeg_reref_ica] wrote {out_dir / 'provenance.json'}")


def _write_determinism_hash(out_dir: Path, sweep_rows: list[dict],
                            reports: list[ICAReport]) -> None:
    """Hash the numeric outputs; compare to a previous run to prove determinism."""
    payload = json.dumps(
        {"sweep": sweep_rows,
         "ica": [(r.key, r.excluded_idx, r.labels, [round(p, 6) for p in r.probabilities])
                 for r in reports]},
        sort_keys=True).encode("utf-8")
    h = hashlib.sha256(payload).hexdigest()[:16]
    hash_file = out_dir / "determinism_hash.txt"
    if hash_file.exists():
        prev = hash_file.read_text(encoding="utf-8").strip()
        status = "MATCHES" if prev == h else f"DIFFERS FROM previous {prev}"
        print(f"[eeg_reref_ica] determinism hash={h} ({status})")
    else:
        print(f"[eeg_reref_ica] determinism hash={h} (first run; saved)")
    hash_file.write_text(h, encoding="utf-8")


# ----------------------------------------------------------------------- report

def main_survival_table(keys: list[str], thresholds: list[float],
                        peaks: dict[str, dict[str, np.ndarray]]) -> None:
    all_before = np.concatenate([peaks[COND_EARS][k] for k in keys])
    all_after = np.concatenate([peaks[COND_ICA][k] for k in keys])
    n_total = all_before.shape[0]
    print("\n=== 1. WINDOW SURVIVAL vs THRESHOLD (overall) : linked-ears vs linked-ears+ICA ===")
    hdr = f"{'threshold uV':>12}{'survive BEFORE':>18}{'survive AFTER':>18}{'delta':>9}"
    print(hdr); print("-" * len(hdr))
    for thr in thresholds:
        kb, _ = _survival(all_before, thr)
        ka, _ = _survival(all_after, thr)
        sb, sa = 100 * kb / n_total, 100 * ka / n_total
        print(f"{thr:>12.0f}{f'{kb}/{n_total} ({sb:.1f}%)':>18}"
              f"{f'{ka}/{n_total} ({sa:.1f}%)':>18}{sa - sb:>+8.1f}%")


def _print_summary(keys, thresholds, peaks, channels, medians, reports, out_dir) -> None:
    main_survival_table(keys, thresholds, peaks)

    print("\n=== 2. PER-CHANNEL MEDIAN PEAK AMPLITUDE (uV): Pz vs linked-ears vs +ICA ===")
    print("(median over channels of each recording's per-channel median peak)")
    hdr = f"{'recording':<14}{'Pz':>10}{'linkedears':>12}{'+ICA':>10}"
    print(hdr); print("-" * len(hdr))
    for key in keys:
        mp = np.nanmedian(medians[COND_PZ][key])
        me = np.nanmedian(medians[COND_EARS][key])
        mi = np.nanmedian(medians[COND_ICA][key])
        print(f"{key:<14}{mp:>10.1f}{me:>12.1f}{mi:>10.1f}")

    print("\n=== 3. ICA COMPONENTS REMOVED PER RECORDING ===")
    for r in reports:
        if r.excluded_idx:
            eog_set = set(r.eog_excluded_idx)
            detail = ", ".join(
                f"IC{i}={lbl}({p:.2f})"
                + (" [eog-frontal-corr]" if i in eog_set else " [iclabel]")
                for i, lbl, p in zip(r.excluded_idx, r.excluded_labels, r.excluded_probs))
        else:
            detail = "none"
        print(f"  {r.key}: removed {len(r.excluded_idx)}/{r.n_components} -> {detail}")

    print(f"\n[eeg_reref_ica] all outputs in {out_dir}")
    print("[eeg_reref_ica] No final threshold chosen (per task) -- decide with supervisor.")


if __name__ == "__main__":
    main()
