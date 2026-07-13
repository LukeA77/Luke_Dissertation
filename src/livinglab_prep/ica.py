"""STAGE 2b - ICA artifact removal (mne + mne-icalabel).

Pipeline position (STRICT):
    reference (linked ears) -> bandpass 0.3-75 -> notch 50 -> [ICA here] ->
    channel-select (18) -> resample 300->200 -> window -> amplitude-reject.

ICA runs AFTER filtering and BEFORE channel selection / resampling, on the EEG
scalp channels only. Design choices (all config-driven, see config.Ica):

  * Fitted PER RECORDING (each session independently); sessions/patients are never
    concatenated.
  * EEG channels are chosen EXPLICITLY from config (channels.keep, the 18 canonical
    10-20 scalp electrodes). This EDF's channel *types* are unreliable -- MNE tags
    the accelerometers, CM, Event and aux sensors all as 'eeg' -- so we do NOT trust
    types to select channels; we derive the set from config and then VERIFY each is
    eeg-typed, failing loudly on any mismatch. The ear channels A1/A2 (degenerate
    after linked-ears re-referencing) and the old Pz reference are excluded.
  * ICA is sensitive to slow drift, so it is fit on a high-passed COPY (fit_highpass_hz)
    and the resulting unmixing is applied to the 0.3-75 Hz data.
  * Extended infomax (method='infomax', extended=True) -- required by ICLabel.
  * Two independent, unioned exclusion rules:
    - Non-ocular artifacts (muscle, heart beat, line noise, channel noise) are
      removed iff ICLabel's WINNING label is in ica.exclude_labels with winning
      probability >= ica.iclabel_min_prob.
    - Eye-blink components are NOT identified via ICLabel's label. ICLabel runs
      off-distribution on this linked-ears/mobile/Parkinsonian data and often
      mislabels real blink components 'brain'/'other' (see PROGRESS.md Problem 2:
      sub-134's blink components got eye-blink probability 0.10-0.25). Instead,
      blink components are identified by frontal-channel correlation --
      ``ICA.find_bads_eog`` with ``measure='correlation'`` against ica.eog_channels
      (Fp1/Fp2), flagging any component whose |r| >= ica.eog_corr_threshold. This
      is the standard EOG-proxy method and is independent of ICLabel's training
      distribution.
    'brain' and 'other' are always kept; ica.exclude_labels must not contain
    'eye blink' (enforced in config._check) since that path is handled here.

Reference caveat: ICLabel was trained on common-average-referenced data; we keep
the supervisor's linked-ears reference, so it runs slightly off-distribution. For
that reason the component topomaps and properties are saved to disk for human review.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.preprocessing import ICA
from mne_icalabel import label_components

from .config import Config


@dataclass
class ICAReport:
    """Per-recording record of what ICA did (provenance + human review)."""
    key: str
    method: str
    extended: bool
    n_components: int
    random_state: int
    decim: int
    fit_highpass_hz: float
    eeg_channels: list[str]          # channels the ICA was fit on (canonical order)
    labels: list[str]                # winning ICLabel class per component
    probabilities: list[float]       # winning probability per component
    exclude_labels: list[str]        # non-ocular ICLabel classes eligible for removal
    iclabel_min_prob: float
    iclabel_excluded_idx: list[int]  # components removed via ICLabel winning-label rule
    eog_channels: list[str]          # frontal channels used as the blink-correlation proxy
    eog_corr_threshold: float        # |r| cutoff for a component to count as ocular
    eog_scores: list[float]          # per-component max|r| across eog_channels
    eog_excluded_idx: list[int]      # components removed via frontal-correlation rule
    excluded_idx: list[int]          # union of iclabel_excluded_idx and eog_excluded_idx
    excluded_labels: list[str]       # their winning ICLabel labels (provenance only)
    excluded_probs: list[float]      # their winning ICLabel probabilities (provenance only)

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "method": self.method,
            "extended": self.extended,
            "n_components": self.n_components,
            "random_state": self.random_state,
            "decim": self.decim,
            "fit_highpass_hz": self.fit_highpass_hz,
            "eeg_channels": self.eeg_channels,
            "n_excluded": len(self.excluded_idx),
            "excluded_idx": self.excluded_idx,
            "excluded_labels": self.excluded_labels,
            "excluded_probs": [round(p, 4) for p in self.excluded_probs],
            "iclabel_min_prob": self.iclabel_min_prob,
            "exclude_labels": self.exclude_labels,
            "iclabel_excluded_idx": self.iclabel_excluded_idx,
            "eog_channels": self.eog_channels,
            "eog_corr_threshold": self.eog_corr_threshold,
            "eog_scores": [round(s, 4) for s in self.eog_scores],
            "eog_excluded_idx": self.eog_excluded_idx,
            "all_labels": self.labels,
            "all_probs": [round(p, 4) for p in self.probabilities],
        }


def eeg_scalp_raw(cfg: Config, raw_filt: mne.io.BaseRaw, key: str) -> mne.io.BaseRaw:
    """Return an EEG-only copy (18 scalp channels, standard names, 10-20 montage).

    The channel set is taken EXPLICITLY from ``cfg.channels.keep`` (not from MNE's
    unreliable channel types). Each selected channel is verified to exist and to be
    eeg-typed; anything else fails loudly.

    Parameters
    ----------
    cfg : Config
        Loaded pipeline config.
    raw_filt : mne.io.BaseRaw
        Re-referenced + band-passed + notched Raw with ALL channels still present.
    key : str
        Recording key, for error messages.

    Returns
    -------
    mne.io.BaseRaw
        A new Raw with exactly the 18 scalp channels, renamed to standard names in
        canonical order, with a standard_1020 montage attached.

    Raises
    ------
    ValueError
        If any expected scalp channel is missing or is not eeg-typed.
    """
    pattern = cfg.channels.rename_from_pattern
    raw_labels = [pattern.format(name=std) for std in cfg.channels.keep]

    present = set(raw_filt.ch_names)
    missing = [lbl for lbl in raw_labels if lbl not in present]
    if missing:
        raise ValueError(
            f"{key}: EEG scalp channels missing for ICA {missing}. "
            f"Available: {list(raw_filt.ch_names)}")

    # Verify types are unambiguously EEG for exactly the channels we will fit on.
    types = {ch: t for ch, t in zip(raw_filt.ch_names, raw_filt.get_channel_types())}
    non_eeg = [lbl for lbl in raw_labels if types.get(lbl) != "eeg"]
    if non_eeg:
        raise ValueError(
            f"{key}: scalp channels not typed 'eeg' (types ambiguous): "
            f"{[(lbl, types.get(lbl)) for lbl in non_eeg]}. Refusing to guess.")

    raw_eeg = raw_filt.copy().pick(raw_labels)
    raw_eeg.rename_channels({pattern.format(name=std): std for std in cfg.channels.keep})
    if list(raw_eeg.ch_names) != list(cfg.channels.keep):
        raise ValueError(f"{key}: EEG channel order mismatch after pick: {raw_eeg.ch_names}")
    raw_eeg.set_montage("standard_1020")
    return raw_eeg


def _save_plots(ica: ICA, raw_fit: mne.io.BaseRaw, report_dir: Path, key: str,
                crop_s: float) -> None:
    """Save the component topomap grid and per-component property figures.

    Topomaps cover all components. Property figures (PSD/ERP-image/time course) are
    rendered on the first ``crop_s`` seconds of the recording -- a representative
    review segment -- because rendering them over a full ~50-min recording is
    memory-prohibitive on this machine. The decomposition itself is unaffected.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    figs = ica.plot_components(show=False)
    for i, fig in enumerate(figs if isinstance(figs, list) else [figs]):
        fig.savefig(report_dir / f"{key}_components_{i:02d}.png", dpi=110)
        plt.close(fig)
    crop = raw_fit.copy().crop(tmax=min(crop_s, raw_fit.times[-1]))
    for comp in range(ica.n_components_):
        figs = ica.plot_properties(crop, picks=[comp], show=False, verbose="ERROR")
        for fig in (figs if isinstance(figs, list) else [figs]):
            fig.savefig(report_dir / f"{key}_property_IC{comp:02d}.png", dpi=100)
            plt.close(fig)
    del crop
    gc.collect()


def clean_with_ica(cfg: Config, raw_filt: mne.io.BaseRaw, key: str,
                   report_dir: Path | None = None) -> tuple[mne.io.BaseRaw, ICAReport]:
    """Fit ICLabel-guided ICA on one recording and return the cleaned EEG signal.

    Parameters
    ----------
    cfg : Config
        Loaded pipeline config (``cfg.ica``).
    raw_filt : mne.io.BaseRaw
        Re-referenced + band-passed + notched Raw with ALL channels present, at the
        original sampling rate (pre-resample). NOT modified in place.
    key : str
        Recording key, for logging, plot filenames and the report.
    report_dir : pathlib.Path or None
        If given and ``cfg.ica.save_plots`` is true, component topomaps/properties
        are written here.

    Returns
    -------
    cleaned : mne.io.BaseRaw
        EEG-only Raw (18 scalp channels, standard names, canonical order) with the
        excluded ICLabel components removed, still at the original sampling rate.
    report : ICAReport
        What was removed and why (labels, probabilities, seed, channels).

    Raises
    ------
    ValueError
        On channel-type ambiguity (see ``eeg_scalp_raw``) or a misconfigured
        n_components.
    """
    ic = cfg.ica
    raw_eeg = eeg_scalp_raw(cfg, raw_filt, key)          # 0.3-75 Hz, EEG-only
    n_eeg = len(raw_eeg.ch_names)
    if ic.n_components > n_eeg:
        raise ValueError(
            f"{key}: ica.n_components ({ic.n_components}) > number of EEG "
            f"channels ({n_eeg}); reduce n_components in config.")

    # Fit on a high-passed copy (ICA is sensitive to slow drift), resampled to
    # ICLabel's native 100 Hz. The unmixing is spatial (channel-space), so it is
    # sampling-rate independent and is applied below to the full-rate 0.3-75 data;
    # resampling here only makes the fit + ICLabel features lighter and on-model.
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

    # Non-ocular artifacts: ICLabel winning-label rule.
    iclabel_exclude_idx = [i for i, (lbl, p) in enumerate(zip(labels, probs))
                            if lbl in ic.exclude_labels and p >= ic.iclabel_min_prob]

    # Eye blinks: frontal-channel correlation (find_bads_eog), NOT ICLabel's
    # label -- see module docstring / PROGRESS.md Problem 2. measure='correlation'
    # returns |r| in [0, 1] against a blink-band-filtered template built from each
    # eog_channel; a component is ocular if it clears eog_corr_threshold on ANY
    # frontal channel.
    per_channel_scores = []
    for ch in ic.eog_channels:
        _, scores = ica.find_bads_eog(
            raw_fit, ch_name=ch, threshold=ic.eog_corr_threshold,
            l_freq=ic.eog_l_freq, h_freq=ic.eog_h_freq,
            measure="correlation", verbose="ERROR")
        per_channel_scores.append(np.abs(np.asarray(scores, float)))
    eog_scores = np.maximum.reduce(per_channel_scores)
    eog_exclude_idx = [i for i in range(ic.n_components)
                        if eog_scores[i] >= ic.eog_corr_threshold]

    exclude_idx = sorted(set(iclabel_exclude_idx) | set(eog_exclude_idx))
    ica.exclude = exclude_idx

    if ic.save_plots and report_dir is not None:
        _save_plots(ica, raw_fit, report_dir, key, ic.plot_crop_s)

    eeg_channels = list(raw_eeg.ch_names)
    # Free the high-passed fit copy before the (memory-heavy) apply step.
    del raw_fit
    gc.collect()

    # Apply the unmixing to the 0.3-75 Hz data (NOT the high-passed fit copy).
    cleaned = raw_eeg.copy()
    del raw_eeg
    gc.collect()
    ica.apply(cleaned, verbose="ERROR")

    report = ICAReport(
        key=key, method=ic.method, extended=ic.extended, n_components=ic.n_components,
        random_state=ic.random_state, decim=ic.decim, fit_highpass_hz=ic.fit_highpass_hz,
        eeg_channels=eeg_channels, labels=labels, probabilities=probs,
        exclude_labels=list(ic.exclude_labels), iclabel_min_prob=ic.iclabel_min_prob,
        iclabel_excluded_idx=iclabel_exclude_idx,
        eog_channels=list(ic.eog_channels), eog_corr_threshold=ic.eog_corr_threshold,
        eog_scores=[float(s) for s in eog_scores], eog_excluded_idx=eog_exclude_idx,
        excluded_idx=exclude_idx,
        excluded_labels=[labels[i] for i in exclude_idx],
        excluded_probs=[probs[i] for i in exclude_idx],
    )
    print(f"[stage2b] {key}: ICA {ic.method}(extended={ic.extended}) "
          f"n_comp={ic.n_components} seed={ic.random_state} -> removed "
          f"{len(exclude_idx)}/{ic.n_components} total "
          f"(iclabel={iclabel_exclude_idx}, eog-frontal-corr={eog_exclude_idx})")
    return cleaned, report
