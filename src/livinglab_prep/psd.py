"""STAGE 4 - PSD sanity checks (README.md §8).

Compares the RAW EDF against the processed continuous signal (the only meaningful
way to verify a filter fingerprint: 1/f EEG power dominates the low band even
after a clean high-pass, so absolute stop-band power is not a valid test --
attenuation relative to raw is). Writes a per-recording PSD plot to
processed/_reports/psd/<key>.png and runs automated checks:
  - passband (1-40 Hz) power preserved   (processed/raw ~ 1)
  - below bandpass low  strongly attenuated (processed/raw << 1)
  - above bandpass high strongly attenuated (processed/raw ~ 0)
  - mains notch removed  (processed peak at notch ~ neighbours)
Failures are LOUD warnings; the plots are the primary artifact.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import welch

from .config import Config
from .preprocess import PreprocessedRecording, load_reref_select


@dataclass
class PSDCheck:
    key: str
    pass_ratio: float         # processed/raw power in 1-40 Hz -> want ~ 1
    low_atten: float          # processed/raw below bandpass low -> want << 1
    high_atten: float         # processed/raw above bandpass high -> want ~ 0
    notch_peak_ratio: float   # processed peak@notch / neighbours -> want <~ 1
    ok: bool


def _welch_mean(data: np.ndarray, fs: float, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    f, p = welch(data, fs=fs, nperseg=int(nperseg), axis=1)
    return f, p.mean(0)


def _median_ratio(f: np.ndarray, ratio: np.ndarray, lo: float, hi: float) -> float:
    m = (f >= lo) & (f <= hi)
    return float(np.nanmedian(ratio[m])) if m.any() else float("nan")


def _raw_scalp_uV(cfg: Config, pre: PreprocessedRecording) -> tuple[np.ndarray, float]:
    """Load the raw EDF (pre-filter) and return the 18 scalp channels in uV.

    Uses the SAME re-reference + channel selection as the processed path, so the
    processed/raw PSD ratio isolates the FILTER fingerprint rather than mixing in
    the reference change.
    """
    raw = load_reref_select(cfg, pre.rec)  # reref + rename + ordered pick, still at orig_sfreq
    return raw.get_data(units=cfg.signal.extract_units), float(raw.info["sfreq"])


def compute_psd_check(cfg: Config, pre: PreprocessedRecording) -> PSDCheck:
    bp = cfg.signal.bandpass
    seg_s = cfg.window.length_s

    raw_data, fs_raw = _raw_scalp_uV(cfg, pre)
    fr, praw = _welch_mean(raw_data, fs_raw, fs_raw * seg_s)
    fp, ppre = _welch_mean(pre.data_uV, pre.sfreq, pre.sfreq * seg_s)

    # Interpolate raw PSD onto the processed grid for direct ratios.
    praw_i = np.interp(fp, fr, praw)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = ppre / np.where(praw_i > 0, praw_i, np.nan)

    pass_ratio = _median_ratio(fp, ratio, 1.0, min(40.0, bp.h_freq))
    low_atten = _median_ratio(fp, ratio, 0.1, max(0.25, bp.l_freq - 0.05))
    high_lo = bp.h_freq + 5.0
    high_atten = _median_ratio(fp, ratio, high_lo, min(high_lo + 15.0, pre.sfreq / 2.0))

    notch = cfg.signal.notch_freq
    peak = ppre[(fp >= notch - 0.6) & (fp <= notch + 0.6)]
    neigh = ppre[((fp >= notch - 2) & (fp <= notch - 1)) | ((fp >= notch + 1) & (fp <= notch + 2))]
    notch_ratio = float(peak.max() / neigh.mean()) if (peak.size and neigh.size and neigh.mean() > 0) else float("nan")

    ok = (0.5 <= pass_ratio <= 2.0) and (low_atten < 0.5) and (high_atten < 0.1) and (notch_ratio < 2.0)
    check = PSDCheck(pre.rec.key, pass_ratio, low_atten, high_atten, notch_ratio, ok)

    _plot(cfg, fr, praw, fp, ppre, check)
    _log(check, cfg)
    return check


def _plot(cfg: Config, fr, praw, fp, ppre, check: PSDCheck) -> None:
    psd_dir = cfg.reports_dir / "psd"
    psd_dir.mkdir(parents=True, exist_ok=True)
    bp = cfg.signal.bandpass
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a in ax:
        a.semilogy(fr, praw, color="0.6", lw=1, label=f"raw ({cfg.signal.orig_sfreq} Hz)")
        a.semilogy(fp, ppre, color="C0", lw=1.2, label=f"processed ({cfg.signal.target_sfreq} Hz)")
        for x, c in [(bp.l_freq, "g"), (cfg.signal.notch_freq, "r"), (bp.h_freq, "m")]:
            a.axvline(x, color=c, ls="--", lw=0.9, alpha=0.7)
        a.set_xlabel("Frequency (Hz)")
        a.set_ylabel("PSD (uV^2/Hz)")
        a.legend(loc="upper right", fontsize=8)
    ax[0].set_xlim(0, cfg.signal.target_sfreq / 2)
    ax[0].set_title(f"{check.key}: full spectrum")
    ax[1].set_xlim(bp.h_freq - 35, cfg.signal.target_sfreq / 2)
    ax[1].set_title(f"zoom: notch @{cfg.signal.notch_freq}, cutoff @{bp.h_freq}")
    fig.tight_layout()
    fig.savefig(psd_dir / f"{check.key}.png", dpi=110)
    plt.close(fig)


def _log(check: PSDCheck, cfg: Config) -> None:
    status = "OK" if check.ok else "WARN"
    print(f"[stage4] {check.key}: PSD {status} | "
          f"pass1-40={check.pass_ratio:.2f} "
          f"low<{cfg.signal.bandpass.l_freq}={check.low_atten:.3f} "
          f"high>{cfg.signal.bandpass.h_freq}={check.high_atten:.4f} "
          f"notch@{cfg.signal.notch_freq}={check.notch_peak_ratio:.2f}")
    if not check.ok:
        print(f"[stage4] WARNING: {check.key} PSD fingerprint off expected "
              f"(check {cfg.reports_dir / 'psd' / (check.key + '.png')})")
