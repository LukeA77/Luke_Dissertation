"""Three-way, PER-CHANNEL spectral (Welch PSD) comparison for the supervisor
sign-off on signal quality, now that ICA blink+artifact removal is fixed (see
PROGRESS.md Problem 2).

Display choices confirmed with Dr Haar (2026-07-09 email):
  1. Individual channels (NOT a whole-scalp average).
  2. Three-way comparison: raw / pre-ICA / post-ICA.
  3. No difference ("removed power") plot needed.

No threshold-tuning judgement is made here -- this script only produces the
before/after spectra; the supervisor decides if the result is good enough to
continue (per user decision 2026-07-08, amplitude-rejection tuning is parked).

Three conditions per recording, ALL on the same 18 canonical EEG channels
(cfg.channels.keep) and the same original sampling rate (no resample yet):

  1. RAW        -- straight off the EDF: original hardware (Pz) reference,
                   unfiltered.
  2. PRE-ICA     -- linked-ears reref -> bandpass(0.3-75 Hz) -> notch(50 Hz).
                   Exactly the input clean_with_ica receives.
  3. POST-ICA    -- PRE-ICA + the fixed ICA cleaning (src/livinglab_prep/ica.py
                   ::clean_with_ica): ICLabel winning-label rule for non-ocular
                   artifacts, unioned with frontal-correlation blink removal.

Each recording gets one figure: an 18-panel grid laid out to approximate the
10-20 montage (front row Fp1/Fp2 at the top, occipital O1/O2 at the bottom),
each panel showing all three PSD curves for that one channel. This is the
layout that actually shows the blink-removal effect (which lives almost
entirely in frontal channels and disappears under a whole-scalp average).

Read-only: does not touch the pipeline or its published outputs. Writes PNGs +
a CSV of the underlying per-channel PSD values (so the supervisor's call can be
revisited without recomputation) to the project root.

Run:
    python spectral_before_after.py               # uses config/pipeline.yaml
    python spectral_before_after.py --config <path>
"""
from __future__ import annotations

import argparse
import csv
import gc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

from src.livinglab_prep.config import Config, load_config
from src.livinglab_prep.discovery import Recording, discover_recordings
from src.livinglab_prep.ica import clean_with_ica, eeg_scalp_raw
from src.livinglab_prep.preprocess import load_raw
from src.livinglab_prep.reference import apply_reference

mne.set_log_level("ERROR")

FMIN, FMAX = 0.0, 100.0          # covers the 50/100 Hz mains lines and the 75 Hz bandpass edge
CONDITIONS = ["raw", "pre_ica", "post_ica"]
LABELS = {"raw": "1. RAW (Pz ref, unfiltered)",
          "pre_ica": "2. PRE-ICA (linked-ears + bandpass + notch)",
          "post_ica": "3. POST-ICA (blinks + artifacts removed)"}
COLORS = {"raw": "#999999", "pre_ica": "#1f77b4", "post_ica": "#d62728"}

# Approximate 10-20 montage grid position (row, col) for each of the 18 canonical
# channels, so the panel layout reads like the scalp (front at top, occipital at
# bottom; left hemisphere on the left). 5 rows x 5 cols; unused cells are blank.
CHANNEL_GRID = {
    "Fp1": (0, 1), "Fp2": (0, 3),
    "F7": (1, 0), "F3": (1, 1), "Fz": (1, 2), "F4": (1, 3), "F8": (1, 4),
    "T3": (2, 0), "C3": (2, 1), "Cz": (2, 2), "C4": (2, 3), "T4": (2, 4),
    "T5": (3, 0), "P3": (3, 1), "P4": (3, 3), "T6": (3, 4),
    "O1": (4, 1), "O2": (4, 3),
}
GRID_ROWS, GRID_COLS = 5, 5


def _welch_psd_dB_per_channel(raw18: mne.io.BaseRaw) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-channel Welch PSD in dB (10*log10 uV^2/Hz). Returns (freqs, psd_db[ch,freq], ch_names)."""
    sfreq = raw18.info["sfreq"]
    n_fft = int(min(4 * sfreq, raw18.n_times))   # ~4s window -> 0.25 Hz resolution
    spectrum = raw18.compute_psd(method="welch", fmin=FMIN, fmax=FMAX,
                                 n_fft=n_fft, n_overlap=n_fft // 2, verbose="ERROR")
    psd_v2 = spectrum.get_data()                 # (n_channels, n_freqs), V^2/Hz
    freqs = spectrum.freqs
    psd_uv2 = psd_v2 * 1e12                       # V^2 -> uV^2
    return freqs, 10 * np.log10(psd_uv2), list(raw18.ch_names)


def _condition_psds(cfg: Config, rec: Recording) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
    out = {}

    raw_pz_full = load_raw(cfg, rec)
    raw_pz_18 = eeg_scalp_raw(cfg, raw_pz_full, rec.key)
    out["raw"] = _welch_psd_dB_per_channel(raw_pz_18)
    del raw_pz_full, raw_pz_18
    gc.collect()

    raw_filt_full = load_raw(cfg, rec)
    raw_filt_full = apply_reference(cfg, raw_filt_full, rec.key)
    bp = cfg.signal.bandpass
    raw_filt_full.filter(l_freq=bp.l_freq, h_freq=bp.h_freq, picks="eeg",
                         method="fir", fir_design=bp.fir_design, phase=bp.phase, verbose="ERROR")
    raw_filt_full.notch_filter(freqs=[cfg.signal.notch_freq], picks="eeg", verbose="ERROR")

    pre_ica_18 = eeg_scalp_raw(cfg, raw_filt_full, rec.key)
    out["pre_ica"] = _welch_psd_dB_per_channel(pre_ica_18)
    del pre_ica_18
    gc.collect()

    cleaned, report = clean_with_ica(cfg, raw_filt_full, rec.key, report_dir=None)
    del raw_filt_full
    gc.collect()
    out["post_ica"] = _welch_psd_dB_per_channel(cleaned)
    print(f"[spectral_before_after] {rec.key}: ICA removed {len(report.excluded_idx)}/"
          f"{report.n_components} components (iclabel={report.iclabel_excluded_idx}, "
          f"eog-frontal-corr={report.eog_excluded_idx})")
    del cleaned
    gc.collect()

    return out


def _plot_recording(key: str, psds: dict[str, tuple[np.ndarray, np.ndarray, list[str]]], out_path) -> None:
    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(17, 15), sharex=True, sharey=True)
    for ax in axes.flat:
        ax.axis("off")

    ch_names = psds["raw"][2]
    handles, labels = None, None
    for ch in ch_names:
        row, col = CHANNEL_GRID[ch]
        ax = axes[row, col]
        ax.axis("on")
        ci = ch_names.index(ch)
        for cond in CONDITIONS:
            freqs, psd_db, cond_ch_names = psds[cond]
            cj = cond_ch_names.index(ch)
            line, = ax.plot(freqs, psd_db[cj], label=LABELS[cond], color=COLORS[cond], linewidth=1.0)
        for f in (50, 75):
            ax.axvline(f, color="black", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.set_title(ch, fontsize=10, fontweight="bold")
        ax.set_xlim(FMIN, FMAX)
        ax.tick_params(labelsize=7)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    for row in range(GRID_ROWS):
        ax = axes[row, 0]
        if ax.get_visible() and ax.axison:
            ax.set_ylabel("PSD (dB)", fontsize=8)
    for col in range(GRID_COLS):
        ax = axes[GRID_ROWS - 1, col]
        if ax.axison:
            ax.set_xlabel("Frequency (Hz)", fontsize=8)

    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"{key}: per-channel spectral comparison (Welch PSD, 10-20 layout)",
                fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _write_csv(all_psds: dict[str, dict[str, tuple[np.ndarray, np.ndarray, list[str]]]], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["recording", "condition", "channel", "freq_hz", "psd_dB_uV2_per_Hz"])
        for key, psds in all_psds.items():
            for cond in CONDITIONS:
                freqs, psd_db, ch_names = psds[cond]
                for ci, ch in enumerate(ch_names):
                    for f, p in zip(freqs, psd_db[ci]):
                        w.writerow([key, cond, ch, round(float(f), 3), round(float(p), 4)])
    print(f"[spectral_before_after] wrote {out_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="spectral_before_after")
    parser.add_argument("--config", default=None, help="path to pipeline.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    recordings = discover_recordings(cfg)

    all_psds: dict[str, dict[str, tuple[np.ndarray, np.ndarray, list[str]]]] = {}
    for rec in recordings:
        print(f"\n[spectral_before_after] === {rec.key} ===")
        psds = _condition_psds(cfg, rec)
        all_psds[rec.key] = psds
        _plot_recording(rec.key, psds, f"spectral_before_after_{rec.key}.png")
        print(f"[spectral_before_after] wrote spectral_before_after_{rec.key}.png")

    _write_csv(all_psds, "spectral_before_after_data.csv")
    print("\n[spectral_before_after] done")


if __name__ == "__main__":
    main()
