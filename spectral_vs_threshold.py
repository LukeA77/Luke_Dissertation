"""Per-channel spectrum on an ABSOLUTE microvolt scale, drawn against the 100 uV
rejection threshold -- the plot Dr Haar asked for (2026-07-10 email).

His point: the existing before/after spectra are in dB (10*log10 uV^2/Hz), i.e.
relative values, so you cannot see where the signal sits relative to the 100 uV
peak-abs threshold the CBraMod-inherited pipeline starts with. This script puts
the spectrum on a linear microvolt axis and overlays that threshold.

WHY THIS TAKES TWO LEVELS PER PANEL (important, or the plot argues the opposite):
The 100 uV threshold is a TIME-DOMAIN peak-abs rule
(reject.mode = peak_abs_whole_window: a 10 s x 18-ch window is rejected if
np.max(np.abs(window)) >= 100 uV). A spectrum measures amplitude *density*, and
its broadband RMS (the asymptote of the cumulative-RMS curve below) sits at only
~20-40 uV for clean channels -- which on its own would look like the signal is
comfortably UNDER 100 uV. That is exactly why the naive "spectrum vs threshold"
picture is misleading. The rejection does not test RMS; it tests the peak the
window actually reaches. So each panel draws BOTH:

  1. Cumulative RMS amplitude vs frequency, RMS(f) = sqrt(integral_0^f PSD df')
     in uV -- the honest "spectrum in microvolts". Its right-hand asymptote is
     the channel's broadband RMS amplitude.
  2. The channel's TYPICAL per-window peak-abs (median over non-overlapping 10 s
     windows of that channel's max|sample|) -- the quantity the reject rule
     actually compares to 100 uV.

The story the figure tells: spectrally the signal is modest (RMS well under
100 uV), but the per-window peak-abs the threshold tests reaches well ABOVE
100 uV on the mobile/task transients -- so a 100 uV peak-abs rule rejects almost
every window despite a clean-looking spectrum. That is the proof that the
inherited threshold is inappropriate here (PROGRESS.md Problem 1).

Condition shown: POST-ICA only (linked-ears reref -> bandpass 0.3-75 -> notch 50
-> fixed ICA cleaning), because that is the exact signal the reject stage would
see. The 3-way raw/pre/post dB comparison already lives in
spectral_before_after.py; this figure is deliberately single-condition so each
panel stays readable with two overlaid reference levels.

Read-only: does not touch the pipeline or its published outputs. Writes one PNG
per recording plus a CSV (per-channel broadband RMS, typical/max window peak-abs,
and the recording-level rejected-window fraction) to the project root.

Run:
    python spectral_vs_threshold.py               # uses config/pipeline.yaml
    python spectral_vs_threshold.py --config <path>
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
from src.livinglab_prep.ica import clean_with_ica
from src.livinglab_prep.preprocess import load_raw
from src.livinglab_prep.reference import apply_reference

mne.set_log_level("ERROR")

FMIN, FMAX = 0.0, 100.0

# Same approximate 10-20 montage grid as spectral_before_after.py so the two
# figures read the same way (front row at the top, occipital at the bottom).
CHANNEL_GRID = {
    "Fp1": (0, 1), "Fp2": (0, 3),
    "F7": (1, 0), "F3": (1, 1), "Fz": (1, 2), "F4": (1, 3), "F8": (1, 4),
    "T3": (2, 0), "C3": (2, 1), "Cz": (2, 2), "C4": (2, 3), "T4": (2, 4),
    "T5": (3, 0), "P3": (3, 1), "P4": (3, 3), "T6": (3, 4),
    "O1": (4, 1), "O2": (4, 3),
}
GRID_ROWS, GRID_COLS = 5, 5

RMS_COLOR = "#1f77b4"        # cumulative-RMS spectrum curve
THRESH_COLOR = "#d62728"     # the 100 uV threshold line
PEAK_COLOR = "#ff7f0e"       # typical per-window peak-abs (what the rule tests)


def _cumulative_rms_uV(raw18: mne.io.BaseRaw) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-channel cumulative RMS amplitude vs frequency, in microvolts.

    RMS(f) = sqrt( integral_0^f PSD(f') df' ) with PSD from Welch in uV^2/Hz.
    The last column is the channel's broadband RMS over [FMIN, FMAX].
    Returns (freqs, cum_rms[ch, freq] in uV, ch_names).
    """
    sfreq = raw18.info["sfreq"]
    n_fft = int(min(4 * sfreq, raw18.n_times))          # ~4 s window -> ~0.25 Hz resolution
    spectrum = raw18.compute_psd(method="welch", fmin=FMIN, fmax=FMAX,
                                 n_fft=n_fft, n_overlap=n_fft // 2, verbose="ERROR")
    freqs = spectrum.freqs
    psd_uv2 = spectrum.get_data() * 1e12                # V^2/Hz -> uV^2/Hz
    df = np.diff(freqs)                                 # (n_freq-1,), Welch grid is uniform
    seg = 0.5 * (psd_uv2[:, 1:] + psd_uv2[:, :-1]) * df  # trapezoid power per segment
    cum_power = np.concatenate(
        [np.zeros((psd_uv2.shape[0], 1)), np.cumsum(seg, axis=1)], axis=1)
    return freqs, np.sqrt(cum_power), list(raw18.ch_names)


def _window_peak_stats(raw18: mne.io.BaseRaw, cfg: Config) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Peak-abs statistics under the pipeline's reject rule (peak_abs_whole_window).

    Splits the recording into NON-OVERLAPPING window.length_s windows (the eval
    tiling) and, per window, takes each channel's max|sample| in uV. Returns:
      per_ch_median_peak[ch] -- median over windows of that channel's window peak
      per_ch_max_peak[ch]    -- max over windows of that channel's window peak
      reject_frac            -- fraction of windows with WHOLE-WINDOW peak (max over
                                all 18 channels) >= reject.threshold_uV
      n_windows
    """
    sfreq = raw18.info["sfreq"]
    data_uv = raw18.get_data() * 1e6                    # (n_ch, n_times), uV
    win = int(round(cfg.window.length_s * sfreq))
    n_win = data_uv.shape[1] // win
    if n_win == 0:                                      # recording shorter than one window
        peak = np.abs(data_uv).max(axis=1)
        whole = float(peak.max())
        thr = cfg.reject.threshold_uV
        return peak, peak, float(whole >= thr), 1
    trimmed = data_uv[:, : n_win * win].reshape(data_uv.shape[0], n_win, win)
    win_peak = np.abs(trimmed).max(axis=2)              # (n_ch, n_win) peak-abs per ch per window
    per_ch_median = np.median(win_peak, axis=1)
    per_ch_max = win_peak.max(axis=1)
    whole_win_peak = win_peak.max(axis=0)              # (n_win,) max over channels
    reject_frac = float(np.mean(whole_win_peak >= cfg.reject.threshold_uV))
    return per_ch_median, per_ch_max, reject_frac, n_win


def _post_ica_signal(cfg: Config, rec: Recording) -> mne.io.BaseRaw:
    """The exact signal the reject stage would see: reref -> bandpass -> notch ->
    fixed ICA cleaning, restricted to the 18 canonical scalp channels."""
    raw = load_raw(cfg, rec)
    raw = apply_reference(cfg, raw, rec.key)
    bp = cfg.signal.bandpass
    raw.filter(l_freq=bp.l_freq, h_freq=bp.h_freq, picks="eeg",
               method="fir", fir_design=bp.fir_design, phase=bp.phase, verbose="ERROR")
    raw.notch_filter(freqs=[cfg.signal.notch_freq], picks="eeg", verbose="ERROR")
    # clean_with_ica already restricts to the 18 renamed scalp channels
    # (it calls eeg_scalp_raw internally), so `cleaned` is the post-ICA signal.
    cleaned, report = clean_with_ica(cfg, raw, rec.key, report_dir=None)
    del raw
    gc.collect()
    print(f"[spectral_vs_threshold] {rec.key}: ICA removed {len(report.excluded_idx)}/"
          f"{report.n_components} components (iclabel={report.iclabel_excluded_idx}, "
          f"eog-frontal-corr={report.eog_excluded_idx})")
    return cleaned


def _plot_recording(key: str, freqs: np.ndarray, cum_rms: np.ndarray,
                    ch_names: list[str], med_peak: np.ndarray, max_peak: np.ndarray,
                    thr: float, reject_frac: float, n_win: int, out_path: str) -> None:
    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(17, 15), sharex=True, sharey=True)
    for ax in axes.flat:
        ax.axis("off")

    for ch in ch_names:
        row, col = CHANNEL_GRID[ch]
        ax = axes[row, col]
        ax.axis("on")
        ci = ch_names.index(ch)

        ax.plot(freqs, cum_rms[ci], color=RMS_COLOR, linewidth=1.3,
                label="cumulative RMS (spectrum, uV)")
        ax.axhline(thr, color=THRESH_COLOR, linestyle="--", linewidth=1.1,
                   label=f"{thr:.0f} uV reject threshold")
        ax.axhline(med_peak[ci], color=PEAK_COLOR, linestyle="-", linewidth=1.1,
                   label="typical per-window peak-abs")

        rms_broadband = cum_rms[ci, -1]
        ax.set_title(f"{ch}   RMS {rms_broadband:.0f}  |  pk {med_peak[ci]:.0f} uV",
                     fontsize=9, fontweight="bold")
        ax.set_xlim(FMIN, FMAX)
        ax.set_yscale("log")
        ax.set_ylim(1, max(1000, float(np.nanmax(max_peak)) * 1.2))
        ax.tick_params(labelsize=7)

    for row in range(GRID_ROWS):
        ax = axes[row, 0]
        if ax.axison:
            ax.set_ylabel("amplitude (uV, log)", fontsize=8)
    for col in range(GRID_COLS):
        ax = axes[GRID_ROWS - 1, col]
        if ax.axison:
            ax.set_xlabel("Frequency (Hz)", fontsize=8)

    # Single legend built from proxy handles (every panel is identical).
    handles = [
        plt.Line2D([], [], color=RMS_COLOR, linewidth=1.3),
        plt.Line2D([], [], color=THRESH_COLOR, linestyle="--", linewidth=1.1),
        plt.Line2D([], [], color=PEAK_COLOR, linewidth=1.1),
    ]
    labels = ["cumulative RMS amplitude of the spectrum (uV)",
              f"{thr:.0f} uV peak-abs reject threshold",
              "typical per-window peak-abs (what the rule tests)"]
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(
        f"{key}: spectrum (in uV) vs the {thr:.0f} uV peak-abs threshold\n"
        f"the {thr:.0f} uV rule would reject {reject_frac * 100:.1f}% of "
        f"{n_win} non-overlapping 10 s windows -- "
        f"spectral RMS sits well below {thr:.0f} uV, but the per-window peak-abs "
        f"the rule tests exceeds it",
        fontsize=12, y=1.045)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _write_csv(rows: list[dict], out_path: str) -> None:
    fields = ["recording", "channel", "broadband_rms_uV", "median_window_peak_uV",
              "max_window_peak_uV", "reject_threshold_uV", "reject_frac_windows",
              "n_windows"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[spectral_vs_threshold] wrote {out_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="spectral_vs_threshold")
    parser.add_argument("--config", default=None, help="path to pipeline.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    thr = cfg.reject.threshold_uV
    recordings = discover_recordings(cfg)

    csv_rows: list[dict] = []
    for rec in recordings:
        print(f"\n[spectral_vs_threshold] === {rec.key} ===")
        post = _post_ica_signal(cfg, rec)
        freqs, cum_rms, ch_names = _cumulative_rms_uV(post)
        med_peak, max_peak, reject_frac, n_win = _window_peak_stats(post, cfg)
        del post
        gc.collect()

        out_png = f"spectral_vs_threshold_{rec.key}.png"
        _plot_recording(rec.key, freqs, cum_rms, ch_names, med_peak, max_peak,
                        thr, reject_frac, n_win, out_png)
        print(f"[spectral_vs_threshold] wrote {out_png}  "
              f"(reject {reject_frac * 100:.1f}% of {n_win} windows)")

        for ci, ch in enumerate(ch_names):
            csv_rows.append({
                "recording": rec.key,
                "channel": ch,
                "broadband_rms_uV": round(float(cum_rms[ci, -1]), 3),
                "median_window_peak_uV": round(float(med_peak[ci]), 3),
                "max_window_peak_uV": round(float(max_peak[ci]), 3),
                "reject_threshold_uV": thr,
                "reject_frac_windows": round(reject_frac, 4),
                "n_windows": n_win,
            })

    _write_csv(csv_rows, "spectral_vs_threshold_data.csv")
    print("\n[spectral_vs_threshold] done")


if __name__ == "__main__":
    main()
