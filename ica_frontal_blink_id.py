"""Identify blink components by FRONTAL-CHANNEL CORRELATION instead of ICLabel's
label -- the standard EOG-proxy method (mne ICA.find_bads_eog), which does not
depend on ICLabel's training distribution.

Motivation (diagnosed in ica_blink_diagnosis.py): blinks are demonstrably
present (frontal ground truth ~hundreds/recording) and ICA captures them, but
ICLabel -- run off-distribution on linked-ears, Parkinsonian, mobile data --
often labels the blink component 'other'/'brain' rather than 'eye blink', so the
pipeline's winning-label rule misses them (esp. sub-134).

This uses Fp1/Fp2 as an EOG proxy (blinks are largest and slowest there) and
flags components whose source time-course correlates with the frontal blink
signal above a z-score threshold. Reports, per recording:
  * every component's frontal-correlation z-score, its ICLabel winning label and
    its ICLabel eye-blink-class probability, side by side
  * which components each method (frontal-corr vs ICLabel winning-label) flags
So we can see directly whether frontal correlation recovers the blink components
ICLabel missed.

Read-only. Fits ICA once per recording (same settings as src/livinglab_prep/ica.py).

Run:
    python ica_frontal_blink_id.py
"""
from __future__ import annotations

import csv
import gc

import numpy as np

import mne
from mne.preprocessing import ICA
from mne_icalabel.iclabel import iclabel_label_components

from src.livinglab_prep.config import load_config
from src.livinglab_prep.discovery import discover_recordings
from src.livinglab_prep.ica import eeg_scalp_raw
from src.livinglab_prep.preprocess import load_raw
from src.livinglab_prep.reference import apply_reference

mne.set_log_level("ERROR")

ICLABEL_CLASSES = ["brain", "muscle artifact", "eye blink", "heart beat",
                   "line noise", "channel noise", "other"]
EYE_COL = ICLABEL_CLASSES.index("eye blink")

# Absolute Pearson-correlation thresholds (measure='correlation'): a component is
# an ocular/blink component if |r| with the frontal blink signal exceeds this.
# (The earlier measure='zscore' adaptive threshold flagged nothing despite raw
# r up to 0.98, so we use a transparent absolute-correlation rule instead.)
EOG_R_THRESHOLDS = [0.5, 0.6, 0.7]


def _filtered_raw(cfg, rec) -> mne.io.BaseRaw:
    raw = load_raw(cfg, rec)
    raw = apply_reference(cfg, raw, rec.key)
    bp = cfg.signal.bandpass
    raw.filter(l_freq=bp.l_freq, h_freq=bp.h_freq, picks="eeg",
               method="fir", fir_design=bp.fir_design, phase=bp.phase, verbose="ERROR")
    raw.notch_filter(freqs=[cfg.signal.notch_freq], picks="eeg", verbose="ERROR")
    return raw


def main() -> None:
    cfg = load_config()
    recordings = discover_recordings(cfg)
    ic = cfg.ica
    rows = []

    for rec in recordings:
        print(f"\n[frontal_blink] ===== {rec.key} =====")
        raw_filt = _filtered_raw(cfg, rec)
        raw_eeg = eeg_scalp_raw(cfg, raw_filt, rec.key)
        del raw_filt
        gc.collect()

        raw_fit = raw_eeg.copy().filter(l_freq=ic.fit_highpass_hz, h_freq=None,
                                        picks="eeg", verbose="ERROR")
        if ic.fit_sfreq < raw_fit.info["sfreq"]:
            raw_fit.resample(ic.fit_sfreq, verbose="ERROR")

        ica = ICA(n_components=ic.n_components, method=ic.method,
                  fit_params=dict(extended=ic.extended), max_iter=ic.max_iter,
                  random_state=ic.random_state)
        ica.fit(raw_fit, decim=ic.decim, verbose="ERROR")

        # ICLabel (for side-by-side comparison with the frontal-corr method).
        proba = np.asarray(iclabel_label_components(raw_fit, ica, inplace=False), float)
        win_idx = np.argmax(proba, axis=1)
        eye_prob = proba[:, EYE_COL]

        # Frontal-correlation method: raw Pearson correlation of each component
        # source with the 1-10 Hz frontal (blink-band) signal at Fp1 and Fp2.
        # measure='correlation' -> returned scores ARE the raw |r| in [0, 1].
        eog_scores = {}
        for fch in ("Fp1", "Fp2"):
            _, scores = ica.find_bads_eog(raw_fit, ch_name=fch, threshold=0.5,
                                          l_freq=1, h_freq=10, measure="correlation",
                                          verbose="ERROR")
            eog_scores[fch] = np.abs(np.asarray(scores, float))   # |Pearson r|

        # Independent hand-computed Pearson r (verifies find_bads_eog semantics):
        # correlate each IC source with the mean frontal blink-band signal.
        sources = ica.get_sources(raw_fit).get_data()            # (n_comp, n_times)
        frontal = raw_fit.copy().pick(["Fp1", "Fp2"]).get_data()  # (2, n_times)
        frontal_bb = mne.filter.filter_data(frontal.mean(axis=0), raw_fit.info["sfreq"],
                                             1.0, 10.0, verbose="ERROR")
        manual_r = np.array([abs(np.corrcoef(sources[i], frontal_bb)[0, 1])
                             for i in range(sources.shape[0])])

        # Combined frontal score = max |r| over Fp1/Fp2 (blink projects to both).
        combined_z = np.maximum(eog_scores["Fp1"], eog_scores["Fp2"])
        print(f"[frontal_blink] {rec.key}: sanity check max|find_bads_eog r - manual r| "
              f"= {np.max(np.abs(combined_z - manual_r)):.3f} (should be small)")

        # Report: rank by frontal correlation.
        order = np.argsort(-combined_z)
        print(f"[frontal_blink] {rec.key}: components ranked by FRONTAL blink "
              f"correlation (|z|), with ICLabel for comparison:")
        print(f"    {'IC':>4} {'frontal|z|':>11} {'ICLabel win':>16} {'eye_p':>7}  flags")
        for i in order:
            flags = []
            for z in EOG_R_THRESHOLDS:
                if combined_z[i] >= z:
                    flags.append(f"r>={z}")
            if win_idx[i] == EYE_COL and eye_prob[i] >= ic.iclabel_min_prob:
                flags.append("ICLABEL-removes")
            win_lbl = ICLABEL_CLASSES[win_idx[i]]
            print(f"    IC{i:02d} {combined_z[i]:>11.2f} {win_lbl:>16} {eye_prob[i]:>7.3f}  "
                  f"{' '.join(flags)}")
            rows.append({
                "recording": rec.key, "component": i,
                "frontal_z_Fp1": round(float(eog_scores['Fp1'][i]), 3),
                "frontal_z_Fp2": round(float(eog_scores['Fp2'][i]), 3),
                "frontal_z_combined": round(float(combined_z[i]), 3),
                "iclabel_winning": win_lbl,
                "iclabel_eye_prob": round(float(eye_prob[i]), 4),
            })

        # Summary counts per method for this recording.
        n_iclabel = int(np.sum((win_idx == EYE_COL) & (eye_prob >= ic.iclabel_min_prob)))
        print(f"[frontal_blink] {rec.key}: ICLabel winning-label rule removes {n_iclabel} "
              f"component(s).")
        for z in EOG_R_THRESHOLDS:
            n = int(np.sum(combined_z >= z))
            print(f"[frontal_blink] {rec.key}: frontal-corr |r|>={z} flags {n} component(s).")

        del raw_eeg, raw_fit, ica, proba, sources
        gc.collect()

    out = "ica_frontal_blink_id.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[frontal_blink] wrote {out}")


if __name__ == "__main__":
    main()
