"""Time-resolved spectral (spectrogram) comparison -- the temporal-context
companion to spectral_before_after.py's whole-recording-averaged PSD.

A single PSD curve per channel (spectral_before_after.py) collapses the entire
~50-minute recording into one flat average, hiding WHEN in the recording the
spectral content changes (e.g. during a specific task, or around a blink
cluster). This script keeps the same raw/pre-ICA/post-ICA three-way comparison
but as a time x frequency heatmap per channel, so the full recording's temporal
structure stays visible.

Channels shown: Fp1, Fp2 (frontal -- where the ICA blink-removal effect lives)
and Cz (a control channel that should barely change, confirming the cleaning is
spatially targeted, not a blanket power reduction). ICA itself is still fit on
all 18 scalp channels (src/livinglab_prep/ica.py::clean_with_ica) -- only the
DISPLAY is restricted to these 3, for a readable figure.

Per recording: one figure, a 3 (channel) x 3 (condition) grid of spectrograms,
frequency 0-40 Hz (where EEG/blink activity lives), full recording duration on
the time axis, shared color scale within the figure for a fair before/after
comparison.

Read-only: does not touch the pipeline or its published outputs. Writes one PNG
per recording to the project root.

Run:
    python spectrogram_before_after.py               # uses config/pipeline.yaml
    python spectrogram_before_after.py --config <path>
"""
from __future__ import annotations

import argparse
import gc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram

from src.livinglab_prep.config import Config, load_config
from src.livinglab_prep.discovery import Recording, discover_recordings
from src.livinglab_prep.ica import clean_with_ica, eeg_scalp_raw
from src.livinglab_prep.preprocess import load_raw
from src.livinglab_prep.reference import apply_reference

mne.set_log_level("ERROR")

DISPLAY_CHANNELS = ["Fp1", "Fp2", "Cz"]     # 2 frontal (blink-affected) + 1 control
FMIN, FMAX = 0.0, 40.0                       # EEG/blink activity range
CONDITIONS = ["raw", "pre_ica", "post_ica"]
TITLES = {"raw": "RAW (Pz ref, unfiltered)",
          "pre_ica": "PRE-ICA (linked-ears + bandpass + notch)",
          "post_ica": "POST-ICA (blinks + artifacts removed)"}


def _spectrogram_dB(data_v: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (freqs, times_min, psd_db[freq, time]) for one channel's 1-D V trace."""
    nperseg = int(4 * sfreq)              # 4s window -> 0.25 Hz resolution
    noverlap = nperseg // 2               # 50% overlap -> 2s time resolution
    freqs, times_s, Sxx_v2 = scipy_spectrogram(
        data_v, fs=sfreq, nperseg=nperseg, noverlap=noverlap, scaling="density")
    Sxx_uv2 = Sxx_v2 * 1e12                # V^2/Hz -> uV^2/Hz
    fmask = (freqs >= FMIN) & (freqs <= FMAX)
    return freqs[fmask], times_s / 60.0, 10 * np.log10(Sxx_uv2[fmask] + 1e-12)


def _condition_spectrograms(cfg: Config, rec: Recording) -> dict[str, dict[str, tuple]]:
    """Returns {condition: {channel: (freqs, times_min, psd_db)}}."""
    out: dict[str, dict[str, tuple]] = {}

    raw_pz_full = load_raw(cfg, rec)
    raw_pz_18 = eeg_scalp_raw(cfg, raw_pz_full, rec.key)
    sfreq = raw_pz_18.info["sfreq"]
    data = raw_pz_18.get_data(picks=DISPLAY_CHANNELS)
    out["raw"] = {ch: _spectrogram_dB(data[i], sfreq) for i, ch in enumerate(DISPLAY_CHANNELS)}
    del raw_pz_full, raw_pz_18, data
    gc.collect()

    raw_filt_full = load_raw(cfg, rec)
    raw_filt_full = apply_reference(cfg, raw_filt_full, rec.key)
    bp = cfg.signal.bandpass
    raw_filt_full.filter(l_freq=bp.l_freq, h_freq=bp.h_freq, picks="eeg",
                         method="fir", fir_design=bp.fir_design, phase=bp.phase, verbose="ERROR")
    raw_filt_full.notch_filter(freqs=[cfg.signal.notch_freq], picks="eeg", verbose="ERROR")

    pre_ica_18 = eeg_scalp_raw(cfg, raw_filt_full, rec.key)
    data = pre_ica_18.get_data(picks=DISPLAY_CHANNELS)
    out["pre_ica"] = {ch: _spectrogram_dB(data[i], sfreq) for i, ch in enumerate(DISPLAY_CHANNELS)}
    del pre_ica_18, data
    gc.collect()

    cleaned, report = clean_with_ica(cfg, raw_filt_full, rec.key, report_dir=None)
    del raw_filt_full
    gc.collect()
    data = cleaned.get_data(picks=DISPLAY_CHANNELS)
    out["post_ica"] = {ch: _spectrogram_dB(data[i], sfreq) for i, ch in enumerate(DISPLAY_CHANNELS)}
    print(f"[spectrogram_before_after] {rec.key}: ICA removed {len(report.excluded_idx)}/"
          f"{report.n_components} components (iclabel={report.iclabel_excluded_idx}, "
          f"eog-frontal-corr={report.eog_excluded_idx})")
    del cleaned, data
    gc.collect()

    return out


def _plot_recording(key: str, spec: dict[str, dict[str, tuple]], out_path: str) -> None:
    all_db = np.concatenate([
        spec[cond][ch][2].ravel() for cond in CONDITIONS for ch in DISPLAY_CHANNELS])
    vmin, vmax = np.percentile(all_db, [2, 98])

    fig, axes = plt.subplots(len(DISPLAY_CHANNELS), len(CONDITIONS),
                             figsize=(15, 9), sharex=True, sharey=True)
    im = None
    for row, ch in enumerate(DISPLAY_CHANNELS):
        for col, cond in enumerate(CONDITIONS):
            ax = axes[row, col]
            freqs, times_min, psd_db = spec[cond][ch]
            im = ax.pcolormesh(times_min, freqs, psd_db, shading="auto",
                               cmap="viridis", vmin=vmin, vmax=vmax)
            if row == 0:
                ax.set_title(TITLES[cond], fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{ch}\nFrequency (Hz)", fontsize=9)
            if row == len(DISPLAY_CHANNELS) - 1:
                ax.set_xlabel("Time (min)", fontsize=9)

    fig.suptitle(f"{key}: spectrogram comparison (Fp1/Fp2 blink-affected, Cz control)",
                fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.92, 0.96))
    cbar_ax = fig.add_axes((0.94, 0.15, 0.015, 0.7))
    fig.colorbar(im, cax=cbar_ax, label="PSD (dB re 1 uV^2/Hz)")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="spectrogram_before_after")
    parser.add_argument("--config", default=None, help="path to pipeline.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    recordings = discover_recordings(cfg)

    for rec in recordings:
        print(f"\n[spectrogram_before_after] === {rec.key} ===")
        spec = _condition_spectrograms(cfg, rec)
        out_path = f"spectrogram_before_after_{rec.key}.png"
        _plot_recording(rec.key, spec, out_path)
        print(f"[spectrogram_before_after] wrote {out_path}")

    print("\n[spectrogram_before_after] done")


if __name__ == "__main__":
    main()
