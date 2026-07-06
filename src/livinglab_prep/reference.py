"""STAGE 1a - re-referencing (README.md §2.5, supervisor requirement).

Re-referencing is the FIRST transformation applied to a recording: it runs on the
raw, preloaded Raw object BEFORE channel selection, filtering, resampling, or
windowing. Ordering is a hard constraint for two reasons:

  1. The reference channels (e.g. the ears A1/A2) must still be present, so it
     must precede the channel selection that drops them.
  2. The supervisor requires re-referencing on the raw data before filtering
     (followed literally even though the operations are linear).

Operation: for every channel, ``new_ch = ch - mean(ref_channels)``. On input that
was recorded against a single scalp electrode (Pz), each channel is ``V(ch) - V(Pz)``
and each ear channel is ``V(ear) - V(Pz)``; subtracting the ear mean cancels the
common ``V(Pz)`` term algebraically, leaving ``V(ch) - mean(V(ears))`` -- i.e. the
recorded Pz reference is replaced by the quiet, off-scalp linked-ears reference.

Channel-name safety (fail loudly): DSI systems do not always label the ears
A1/A2. The reference channel names are read from config; if any is absent from the
recording, this raises a clear error listing every available channel and STOPS.
No guessing, no substitution.
"""
from __future__ import annotations

import mne

from .config import Config


def resolve_ref_labels(cfg: Config, raw_ch_names: list[str], key: str) -> list[str]:
    """Map config reference names to the raw EDF channel labels actually present.

    Parameters
    ----------
    cfg : Config
        Loaded pipeline config; ``cfg.reref.ref_channels`` holds the standard
        electrode names and ``cfg.channels.rename_from_pattern`` the raw-label
        template (e.g. ``"EEG {name}-Pz"``).
    raw_ch_names : list of str
        The channel labels present in the loaded recording (pre-rename).
    key : str
        Recording key, for error messages.

    Returns
    -------
    list of str
        The raw labels (in config order) to pass to ``set_eeg_reference``.

    Raises
    ------
    ValueError
        If any requested reference channel cannot be found, with the full list
        of available channels so the correct label can be identified.
    """
    pattern = cfg.channels.rename_from_pattern
    available = list(raw_ch_names)
    resolved: list[str] = []
    missing: list[str] = []

    for std in cfg.reref.ref_channels:
        # Accept either the exact label as given or the patterned raw label
        # (e.g. std 'A1' -> 'EEG A1-Pz'), so the config can carry standard names.
        patterned = pattern.format(name=std)
        if std in available:
            resolved.append(std)
        elif patterned in available:
            resolved.append(patterned)
        else:
            missing.append(std)

    if missing:
        raise ValueError(
            f"{key}: reference channel(s) {missing} not found "
            f"(also tried the '{pattern}' pattern). Re-referencing needs these "
            f"channels present in the raw recording. Available channels "
            f"({len(available)}): {available}"
        )
    return resolved


def apply_reference(cfg: Config, raw: mne.io.BaseRaw, key: str) -> mne.io.BaseRaw:
    """Re-reference the raw recording in place, as the first transformation.

    Parameters
    ----------
    cfg : Config
        Loaded pipeline config (``cfg.reref``).
    raw : mne.io.BaseRaw
        Preloaded Raw with ALL channels still present (pre channel-selection).
    key : str
        Recording key, for logging and error messages.

    Returns
    -------
    mne.io.BaseRaw
        The same Raw object, re-referenced to the configured channels. If
        ``cfg.reref.ref_channels`` is empty, the recording is returned unchanged
        (keeps the recorded reference).
    """
    if not cfg.reref.enabled:
        print(f"[stage1a] {key}: re-referencing disabled (scheme='{cfg.reref.scheme}', "
              f"ref_channels=[]); keeping recorded reference")
        return raw

    ref_labels = resolve_ref_labels(cfg, list(raw.ch_names), key)
    # ref_channels=<names> subtracts the MEAN of the named channels from every
    # channel (no projection, applied directly on the preloaded data).
    raw.set_eeg_reference(ref_channels=ref_labels, verbose="ERROR")
    print(f"[stage1a] {key}: re-referenced to scheme='{cfg.reref.scheme}' "
          f"using {ref_labels}")
    return raw
