"""Diagnose why ICLabel reports only ~1-2 eye-blink components in a ~50-min
recording (physiologically there should be hundreds of BLINKS, ideally captured
by one component). Read-only; no pipeline files modified.

Three independent checks per recording:

  (A) FULL 7-class ICLabel probability matrix per component. The pipeline only
      inspects each component's WINNING label; here we expose the eye-blink-class
      probability for EVERY component, so a component that is (say) 45% "other"
      but 40% "eye blink" -- invisible to any winning-label threshold sweep --
      becomes visible. Classes/order (ICLabel):
      brain, muscle artifact, eye blink, heart beat, line noise, channel noise, other.

  (B) BLINK COUNT inside candidate components. For the components ranked highest
      on the eye-blink CLASS probability, count blink-like deflections across the
      whole recording via robust (MAD) peak detection. Tells us whether "1 eye
      blink component" actually captures the true blink rate (hundreds) or not.

  (C) GROUND TRUTH, ICA-free. Count blink deflections directly on the frontal
      channels Fp1/Fp2 (where blinks are largest, slow, symmetric). This is the
      target number ICA should be reproducing; it does not depend on ICA/ICLabel.

Fits ICA once per recording (same settings as src/livinglab_prep/ica.py).

Run:
    python ica_blink_diagnosis.py
"""
from __future__ import annotations

import csv
import gc

import numpy as np
from scipy.signal import find_peaks

import mne
from mne.preprocessing import ICA
from mne_icalabel.iclabel import iclabel_label_components

from src.livinglab_prep.config import load_config
from src.livinglab_prep.discovery import discover_recordings
from src.livinglab_prep.ica import eeg_scalp_raw
from src.livinglab_prep.preprocess import load_raw
from src.livinglab_prep.reference import apply_reference

mne.set_log_level("ERROR")

# ICLabel output column order (from mne_icalabel source).
ICLABEL_CLASSES = ["brain", "muscle artifact", "eye blink", "heart beat",
                   "line noise", "channel noise", "other"]
EYE_COL = ICLABEL_CLASSES.index("eye blink")

# Blink physiology: refractory ~0.2-0.3 s (hard cap ~5 blinks/s); a normal rate
# is ~10-25/min at rest, often higher during tasks -> hundreds over ~50 min.
MIN_BLINK_SEP_S = 0.3
Z_THRESHOLDS = [4.0, 5.0, 6.0]        # robustness band for the deflection count


def _filtered_raw(cfg, rec) -> mne.io.BaseRaw:
    raw = load_raw(cfg, rec)
    raw = apply_reference(cfg, raw, rec.key)
    bp = cfg.signal.bandpass
    raw.filter(l_freq=bp.l_freq, h_freq=bp.h_freq, picks="eeg",
               method="fir", fir_design=bp.fir_design, phase=bp.phase, verbose="ERROR")
    raw.notch_filter(freqs=[cfg.signal.notch_freq], picks="eeg", verbose="ERROR")
    return raw


def _count_deflections(sig: np.ndarray, sfreq: float, z_thresh: float) -> int:
    """Robust (median/MAD) count of large monophasic excursions in a 1-D signal.

    Sign-agnostic (abs): blinks may project +ve or -ve into a component. Enforces
    a physiological refractory gap so a single blink isn't double-counted.
    """
    x = np.asarray(sig, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    z = np.abs(x - med) / (1.4826 * mad)
    dist = max(1, int(round(MIN_BLINK_SEP_S * sfreq)))
    peaks, _ = find_peaks(z, height=z_thresh, distance=dist)
    return int(peaks.size)


def _count_str(sig, sfreq) -> str:
    return " / ".join(f"{_count_deflections(sig, sfreq, z)}@z{z:.0f}" for z in Z_THRESHOLDS)


def main() -> None:
    cfg = load_config()
    recordings = discover_recordings(cfg)
    dur_min_map = {}
    matrix_rows = []      # full per-component probability rows -> CSV

    for rec in recordings:
        print(f"\n[blink_diag] ===== {rec.key} =====")
        raw_filt = _filtered_raw(cfg, rec)
        raw_eeg = eeg_scalp_raw(cfg, raw_filt, rec.key)      # 18 scalp ch, 0.3-75 Hz
        del raw_filt
        gc.collect()

        ic = cfg.ica
        raw_fit = raw_eeg.copy().filter(l_freq=ic.fit_highpass_hz, h_freq=None,
                                        picks="eeg", verbose="ERROR")
        if ic.fit_sfreq < raw_fit.info["sfreq"]:
            raw_fit.resample(ic.fit_sfreq, verbose="ERROR")
        fit_sfreq = raw_fit.info["sfreq"]
        dur_min = raw_fit.times[-1] / 60.0
        dur_min_map[rec.key] = dur_min

        ica = ICA(n_components=ic.n_components, method=ic.method,
                  fit_params=dict(extended=ic.extended), max_iter=ic.max_iter,
                  random_state=ic.random_state)
        ica.fit(raw_fit, decim=ic.decim, verbose="ERROR")

        # (A) FULL probability matrix (n_components, 7)
        proba = iclabel_label_components(raw_fit, ica, inplace=False)
        proba = np.asarray(proba, dtype=float)

        # Component source time-courses (whole recording) for blink counting.
        sources = ica.get_sources(raw_fit).get_data()       # (n_comp, n_times)

        winning_idx = np.argmax(proba, axis=1)
        eye_prob = proba[:, EYE_COL]

        # (C) GROUND TRUTH on frontal channels, ICA-free.
        print(f"[blink_diag] {rec.key}: recording length {dur_min:.1f} min "
              f"(expected blinks at 15-25/min ~ {int(dur_min*15)}-{int(dur_min*25)})")
        for fch in ("Fp1", "Fp2"):
            fsig = raw_eeg.copy().pick([fch]).get_data(units="uV")[0]
            # Emphasise slow blink band so fast EEG/EMG doesn't inflate the count.
            fsig_bp = mne.filter.filter_data(fsig, raw_eeg.info["sfreq"], 1.0, 10.0,
                                             verbose="ERROR")
            print(f"[blink_diag]   GROUND-TRUTH {fch} blink deflections: "
                  f"{_count_str(fsig_bp, raw_eeg.info['sfreq'])}")

        # (A)+(B) per-component report, ranked by eye-blink CLASS probability.
        order = np.argsort(-eye_prob)
        print(f"[blink_diag] {rec.key}: components ranked by EYE-BLINK-class prob "
              f"(winning label shown; blink count in the component source):")
        for rank, i in enumerate(order):
            win_lbl = ICLABEL_CLASSES[winning_idx[i]]
            win_p = proba[i, winning_idx[i]]
            tag = "<-- winning=eye blink" if winning_idx[i] == EYE_COL else ""
            # Count blinks in the source only for the plausible top candidates.
            if rank < 4 or winning_idx[i] == EYE_COL:
                bc = _count_str(sources[i], fit_sfreq)
                cnt = f"blinks_in_source={bc}"
            else:
                cnt = ""
            print(f"    IC{i:02d}  eye_p={eye_prob[i]:.3f}  win={win_lbl:<15}({win_p:.3f})  "
                  f"{cnt} {tag}")

            matrix_rows.append({
                "recording": rec.key, "component": i,
                "winning_label": win_lbl, "winning_prob": round(win_p, 4),
                **{f"p_{c.replace(' ', '_')}": round(float(proba[i, j]), 4)
                   for j, c in enumerate(ICLABEL_CLASSES)},
            })

        del raw_eeg, raw_fit, ica, sources, proba
        gc.collect()

    out = "ica_blink_diagnosis_components.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(matrix_rows[0].keys()))
        w.writeheader()
        w.writerows(matrix_rows)
    print(f"\n[blink_diag] wrote {out} (full 7-class probability matrix per component)")


if __name__ == "__main__":
    main()
