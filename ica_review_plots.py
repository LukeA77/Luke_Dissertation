"""Read-only ICA component review: full topomap grids + per-component property
plots for all four recordings (including the two with zero exclusions, so
nothing missed by ICLabel goes unchecked), for human sanity-checking before the
exclusions already reported are relied on.

This script changes NOTHING about the pipeline: it does not touch
processed/reref-<scheme>_ica/ica_report_*.json, sweep_survival.csv,
per_channel_amplitude.csv, determinism_hash.txt, or the existing
_reports/ica_plots/ crop plots written by eeg_reref_ica.py. Everything here is
written to a brand-new processed/reref-<scheme>_ica_review/ directory, and the
script refuses to run if that directory already exists and is non-empty.

No saved ICA solution (.fif) exists on disk, so each recording's ICA is REFIT
here, using the identical recipe in src/livinglab_prep/ica.py::clean_with_ica
(same config, same seed, same fit_highpass_hz/fit_sfreq/decim). The refit
labels/probabilities/excluded_idx are then asserted -- field by field -- against
the already-published ica_report_<key>.json; any disagreement aborts loudly
rather than silently plotting a different decomposition than the one already
reported. (This is a narrower, targeted check than recomputing the pipeline's
combined determinism_hash.txt, which also folds in the window-survival sweep --
unrelated to what is being plotted here -- and would require re-running the
full windowing/threshold-sweep pipeline to reproduce byte-for-byte.)

Run:
    python ica_review_plots.py               # uses config/pipeline.yaml
    python ica_review_plots.py --config <path>
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.preprocessing import ICA
from mne_icalabel import label_components

from src.livinglab_prep.config import Config, load_config
from src.livinglab_prep.discovery import Recording, discover_recordings
from src.livinglab_prep.ica import eeg_scalp_raw
from src.livinglab_prep.preprocess import load_raw
from src.livinglab_prep.reference import apply_reference

mne.set_log_level("ERROR")


def _filtered_raw(cfg: Config, rec: Recording) -> mne.io.BaseRaw:
    """Load -> linked-ears reref -> bandpass -> notch, ALL channels, orig sfreq.

    Mirrors eeg_reref_ica.py::_filtered_raw(do_reref=True) exactly: this is the
    input clean_with_ica received when the published exclusions were produced.
    """
    raw = load_raw(cfg, rec)
    raw = apply_reference(cfg, raw, rec.key)
    bp = cfg.signal.bandpass
    raw.filter(l_freq=bp.l_freq, h_freq=bp.h_freq, picks="eeg",
               method="fir", fir_design=bp.fir_design, phase=bp.phase, verbose="ERROR")
    raw.notch_filter(freqs=[cfg.signal.notch_freq], picks="eeg", verbose="ERROR")
    return raw


def _refit_ica(cfg: Config, raw_eeg: mne.io.BaseRaw):
    """Refit ICA + ICLabel exactly as src/livinglab_prep/ica.py::clean_with_ica.

    Deliberately duplicated rather than imported, so this review script cannot
    silently inherit a future change to the production fit recipe without the
    determinism assertion in _assert_matches_published catching the divergence.
    """
    ic = cfg.ica
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
    return ica, raw_fit, labels, probs


def _refit_eog(ic, ica: ICA, raw_fit: mne.io.BaseRaw) -> tuple[list[float], list[int]]:
    """Refit the frontal-correlation blink exclusion exactly as clean_with_ica.

    Duplicated (not imported) for the same reason as ``_refit_ica``: a future
    change to the production rule must be caught by the determinism assertion,
    not silently inherited.
    """
    per_channel_scores = []
    for ch in ic.eog_channels:
        _, scores = ica.find_bads_eog(
            raw_fit, ch_name=ch, threshold=ic.eog_corr_threshold,
            l_freq=ic.eog_l_freq, h_freq=ic.eog_h_freq,
            measure="correlation", verbose="ERROR")
        per_channel_scores.append(np.abs(np.asarray(scores, float)))
    eog_scores = np.maximum.reduce(per_channel_scores)
    eog_exclude_idx = [i for i in range(ic.n_components) if eog_scores[i] >= ic.eog_corr_threshold]
    return eog_scores.tolist(), eog_exclude_idx


def _assert_matches_published(key: str, labels: list[str], probs: list[float],
                              eog_exclude_idx: list[int],
                              ic, published_path: Path) -> tuple[list[int], list[int]]:
    """Fail loudly if the refit decomposition disagrees with the published report."""
    if not published_path.exists():
        raise FileNotFoundError(
            f"{key}: no published ICA report at {published_path} -- run "
            f"eeg_reref_ica.py first; refusing to plot an unpublished decomposition.")
    published = json.loads(published_path.read_text(encoding="utf-8"))

    iclabel_exclude_idx = [i for i, (lbl, p) in enumerate(zip(labels, probs))
                           if lbl in ic.exclude_labels and p >= ic.iclabel_min_prob]
    exclude_idx = sorted(set(iclabel_exclude_idx) | set(eog_exclude_idx))
    flagged_idx = [i for i, p in enumerate(probs) if p >= ic.iclabel_min_prob]

    mismatches = []
    if labels != published["all_labels"]:
        mismatches.append(f"labels differ: refit={labels} published={published['all_labels']}")
    pub_probs = published["all_probs"]
    if len(probs) != len(pub_probs) or any(abs(a - b) > 1e-4 for a, b in zip(probs, pub_probs)):
        mismatches.append(f"probabilities differ: refit={probs} published={pub_probs}")
    if eog_exclude_idx != published["eog_excluded_idx"]:
        mismatches.append(
            f"eog_excluded_idx differ: refit={eog_exclude_idx} "
            f"published={published['eog_excluded_idx']}")
    if exclude_idx != published["excluded_idx"]:
        mismatches.append(f"excluded_idx differ: refit={exclude_idx} published={published['excluded_idx']}")

    if mismatches:
        raise AssertionError(
            f"{key}: refit ICA does NOT reproduce the published decomposition in "
            f"{published_path} -- refusing to generate review plots for a different "
            f"result than what was already reported.\n  " + "\n  ".join(mismatches))

    print(f"[ica_review_plots] {key}: refit MATCHES published report "
          f"({published_path.name}) -- {len(exclude_idx)} excluded "
          f"(iclabel={iclabel_exclude_idx}, eog-frontal-corr={eog_exclude_idx}), "
          f"{len(flagged_idx)} flagged (prob >= {ic.iclabel_min_prob})")
    return exclude_idx, flagged_idx


def _plot_recording(cfg: Config, rec: Recording, review_dir: Path) -> dict:
    ic = cfg.ica
    raw_filt = _filtered_raw(cfg, rec)
    raw_eeg = eeg_scalp_raw(cfg, raw_filt, rec.key)
    del raw_filt
    gc.collect()

    ica, raw_fit, labels, probs = _refit_ica(cfg, raw_eeg)
    del raw_eeg
    gc.collect()

    _, eog_exclude_idx = _refit_eog(ic, ica, raw_fit)

    published_path = cfg.ica_run_dir / f"ica_report_{rec.key}.json"
    exclude_idx, flagged_idx = _assert_matches_published(
        rec.key, labels, probs, eog_exclude_idx, ic, published_path)

    rec_dir = review_dir / rec.key
    rec_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full topomap grid -- all n_components, spatial patterns only (cheap:
    #    no per-sample data needed), same call as production (ica.py:148).
    topo_files = []
    figs = ica.plot_components(show=False)
    for i, fig in enumerate(figs if isinstance(figs, list) else [figs]):
        out = rec_dir / f"{rec.key}_components_grid_{i:02d}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        topo_files.append(out.name)

    # 2. Per-component property plots on the SAME 1 Hz high-passed fit
    #    instance, cropped to ic.plot_crop_s for memory -- identical crop
    #    safeguard to production (ica.py:138-159), documented there as
    #    memory-prohibitive at full length on this machine.
    crop = raw_fit.copy().crop(tmax=min(ic.plot_crop_s, raw_fit.times[-1]))
    prop_files: dict[int, str] = {}
    for comp in range(ica.n_components_):
        lbl, p = labels[comp], probs[comp]
        tag = "EXCLUDED" if comp in exclude_idx else ("flagged" if comp in flagged_idx else "kept")
        figs = ica.plot_properties(crop, picks=[comp], show=False, verbose="ERROR")
        out = rec_dir / f"{rec.key}_property_IC{comp:02d}.png"
        for fig in (figs if isinstance(figs, list) else [figs]):
            fig.suptitle(f"{rec.key}  IC{comp:02d}  label={lbl}  p={p:.3f}  [{tag}]", fontsize=10)
            fig.savefig(out, dpi=100)
            plt.close(fig)
        prop_files[comp] = out.name
    del crop, raw_fit
    gc.collect()

    rows = []
    for comp in range(ica.n_components_):
        rows.append({
            "component": comp, "label": labels[comp], "probability": probs[comp],
            "flagged": comp in flagged_idx, "excluded": comp in exclude_idx,
            "property_plot": prop_files[comp],
        })

    del ica
    gc.collect()

    return {"key": rec.key, "topomap_grids": topo_files, "components": rows,
            "n_excluded": len(exclude_idx), "n_flagged": len(flagged_idx),
            "n_components": len(rows)}


def _write_index(cfg: Config, review_dir: Path, results: list[dict]) -> None:
    ic = cfg.ica
    lines = [
        "# ICA component review (read-only)",
        "",
        "Generated for manual review. This run did not change any exclusions, "
        "thresholds, or cleaned outputs -- see ica_review_plots.py for the "
        "determinism check performed before any plot was drawn.",
        "",
        f"- ICLabel probability cutoff (`ica.iclabel_min_prob`): **{ic.iclabel_min_prob}**",
        f"- Non-ocular classes eligible for exclusion (`ica.exclude_labels`): {ic.exclude_labels}",
        "- `brain` and `other` are always kept regardless of probability.",
        f"- Eye blinks are identified separately by frontal-channel correlation "
        f"(`ica.eog_channels`={ic.eog_channels}, `ica.eog_corr_threshold`="
        f"{ic.eog_corr_threshold}), not by ICLabel's winning label -- see "
        f"PROGRESS.md Problem 2.",
        f"- Property plots are rendered on the first {ic.plot_crop_s:.0f}s of the "
        "1 Hz high-passed fit instance (same crop as production, for memory).",
        "- Every recording's refit was asserted to match its published "
        "`ica_report_<key>.json` before any plot was generated for it.",
    ]

    for r in results:
        lines += [
            "",
            f"## {r['key']}",
            f"{r['n_excluded']}/{r['n_components']} excluded, "
            f"{r['n_flagged']}/{r['n_components']} flagged (prob >= {ic.iclabel_min_prob})",
            "",
            f"Topomap grid: {', '.join(r['topomap_grids'])}",
            "",
            "| IC | label | probability | flagged | excluded | property plot |",
            "|---|---|---|---|---|---|",
        ]
        for row in r["components"]:
            lines.append(
                f"| {row['component']:02d} | {row['label']} | {row['probability']:.3f} | "
                f"{'YES' if row['flagged'] else ''} | {'YES' if row['excluded'] else ''} | "
                f"{row['property_plot']} |")

    out = review_dir / "index.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ica_review_plots] wrote {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ica_review_plots")
    parser.add_argument("--config", default=None, help="path to pipeline.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if not cfg.ica.enabled:
        raise SystemExit("[ica_review_plots] ica.enabled is false; nothing to review.")

    review_dir = cfg.processed_dir / f"reref-{cfg.reref.scheme}_ica_review"
    if review_dir.exists() and any(review_dir.iterdir()):
        raise SystemExit(
            f"[ica_review_plots] {review_dir} already exists and is non-empty; "
            f"refusing to overwrite. Remove it first if you want a fresh review.")
    review_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ica_review_plots] out={review_dir}")

    recordings = discover_recordings(cfg)
    results = []
    for rec in recordings:
        print(f"\n[ica_review_plots] === {rec.key} ===")
        results.append(_plot_recording(cfg, rec, review_dir))

    _write_index(cfg, review_dir, results)
    print(f"\n[ica_review_plots] done -> {review_dir}")


if __name__ == "__main__":
    main()
