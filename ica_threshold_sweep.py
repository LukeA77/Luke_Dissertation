"""ICLabel classification-threshold sweep (read-only, no pipeline files touched).

Fits ICA (extended Infomax + ICLabel) ONCE per recording -- identical settings
to src/livinglab_prep/ica.py::clean_with_ica (same fit order, n_components,
seed, fit_highpass_hz, fit_sfreq) -- then, instead of applying a single fixed
iclabel_min_prob cutoff, sweeps a grid of candidate cutoffs against the SAME
per-component labels/probabilities ICLabel already produced. This answers "at
what threshold does the excluded-component count change" without re-fitting
ICA per threshold (the fit is the expensive part; the cutoff is just a filter
over already-computed probabilities).

Currently only the sub-131/sub-134 pilot recordings are available locally.

Run:
    python ica_threshold_sweep.py
"""
from __future__ import annotations

import csv
import gc

import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components

from src.livinglab_prep.config import load_config
from src.livinglab_prep.discovery import discover_recordings
from src.livinglab_prep.ica import eeg_scalp_raw
from src.livinglab_prep.preprocess import load_raw
from src.livinglab_prep.reference import apply_reference

mne.set_log_level("ERROR")

THRESHOLD_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def _filtered_raw(cfg, rec) -> mne.io.BaseRaw:
    """load -> linked-ears reref -> bandpass -> notch, ALL channels, orig_sfreq."""
    raw = load_raw(cfg, rec)
    raw = apply_reference(cfg, raw, rec.key)
    bp = cfg.signal.bandpass
    raw.filter(l_freq=bp.l_freq, h_freq=bp.h_freq, picks="eeg",
               method="fir", fir_design=bp.fir_design, phase=bp.phase, verbose="ERROR")
    raw.notch_filter(freqs=[cfg.signal.notch_freq], picks="eeg", verbose="ERROR")
    return raw


def _fit_and_label(cfg, raw_filt: mne.io.BaseRaw, key: str) -> tuple[list[str], list[float]]:
    """Fit ICA once, return (labels, probabilities) for every component."""
    ic = cfg.ica
    raw_eeg = eeg_scalp_raw(cfg, raw_filt, key)

    raw_fit = raw_eeg.copy().filter(
        l_freq=ic.fit_highpass_hz, h_freq=None, picks="eeg", verbose="ERROR")
    if ic.fit_sfreq < raw_fit.info["sfreq"]:
        raw_fit.resample(ic.fit_sfreq, verbose="ERROR")

    ica = ICA(n_components=ic.n_components, method=ic.method,
              fit_params=dict(extended=ic.extended), max_iter=ic.max_iter,
              random_state=ic.random_state)
    ica.fit(raw_fit, decim=ic.decim, verbose="ERROR")

    labelling = label_components(raw_fit, ica, method="iclabel")
    labels = list(labelling["labels"])
    probs = [float(p) for p in labelling["y_pred_proba"]]

    del raw_eeg, raw_fit, ica
    gc.collect()
    return labels, probs


def main() -> None:
    cfg = load_config()
    recordings = discover_recordings(cfg)
    exclude_labels = set(cfg.ica.exclude_labels)

    per_recording: dict[str, tuple[list[str], list[float]]] = {}

    for rec in recordings:
        print(f"\n[ica_threshold_sweep] === {rec.key} === (fitting ICA, this takes a while)")
        raw_filt = _filtered_raw(cfg, rec)
        labels, probs = _fit_and_label(cfg, raw_filt, rec.key)
        per_recording[rec.key] = (labels, probs)
        del raw_filt
        gc.collect()

        print(f"[ica_threshold_sweep] {rec.key}: {len(labels)} components labelled")
        artifact_eligible = sorted(
            [(lbl, p) for lbl, p in zip(labels, probs) if lbl in exclude_labels],
            key=lambda x: -x[1])
        print(f"[ica_threshold_sweep] {rec.key}: artifact-eligible components "
              f"(label in {sorted(exclude_labels)}), sorted by probability:")
        if artifact_eligible:
            for lbl, p in artifact_eligible:
                print(f"    {lbl:<16} p={p:.3f}")
        else:
            print("    (none)")

    # ---- Threshold sweep table: excluded count per recording per threshold ----
    print("\n=== EXCLUDED-COMPONENT COUNT vs ICLABEL_MIN_PROB THRESHOLD ===")
    keys = sorted(per_recording)
    hdr = f"{'threshold':>10}" + "".join(f"{k:>14}" for k in keys) + f"{'TOTAL':>10}"
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for thr in THRESHOLD_GRID:
        counts = {}
        for k in keys:
            labels, probs = per_recording[k]
            n = sum(1 for lbl, p in zip(labels, probs) if lbl in exclude_labels and p >= thr)
            counts[k] = n
        total = sum(counts.values())
        print(f"{thr:>10.2f}" + "".join(f"{counts[k]:>14}" for k in keys) + f"{total:>10}")
        row = {"threshold": thr, **{k: counts[k] for k in keys}, "TOTAL": total}
        rows.append(row)

    out_path = "ica_threshold_sweep_results.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[ica_threshold_sweep] wrote {out_path}")

    # ---- Full per-component detail (every recording, every component, sorted) ----
    detail_path = "ica_threshold_sweep_components.csv"
    with open(detail_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["recording", "component_idx", "label", "probability", "artifact_eligible"])
        for k in keys:
            labels, probs = per_recording[k]
            for i, (lbl, p) in enumerate(zip(labels, probs)):
                w.writerow([k, i, lbl, round(p, 4), lbl in exclude_labels])
    print(f"[ica_threshold_sweep] wrote {detail_path}")


if __name__ == "__main__":
    main()
