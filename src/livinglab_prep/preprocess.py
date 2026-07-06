"""STAGES 1-3 - load, re-reference, channel selection, filter/resample, uV
extraction (README.md §8). Operates on the intact continuous signal; NO sample
removal (splicing/dropping samples would desync the EEG from the CSV clock -- §5.2).

Order (STRICT): load raw (all channels, preloaded) -> re-reference (Stage 1a,
the FIRST transformation, while the reference channels are still present) ->
channel-select 18 in canonical order -> resample(200) -> zero-phase FIR
bandpass(0.3,75) -> notch(50). Re-referencing MUST precede channel selection and
filtering (see reference.py). The resample/bandpass/notch sub-order follows the
reference convention (§2.5). Scale is raw uV, unscaled -- the /100 lives only in
the (deferred) Dataset loader (§2.2).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import mne
import numpy as np

from .config import Config
from .discovery import Recording
from .reference import apply_reference


@dataclass
class PreprocessedRecording:
    rec: Recording
    data_uV: np.ndarray          # (18, n_samples) float32, microvolts
    ch_names: list[str]          # standard names, canonical order
    sfreq: int                   # target_sfreq (200)
    n_samples: int
    meas_date: datetime | None   # recording start wall-clock (for Stage 5)


def _rename_map(cfg: Config, raw_ch_names: list[str]) -> dict[str, str]:
    """Map raw EDF labels -> standard names for the channels we keep.

    rename_from_pattern (e.g. 'EEG {name}-Pz') defines the raw label for a given
    standard name; we invert it to find each keep-channel's raw label.
    """
    pattern = cfg.channels.rename_from_pattern
    raw_to_std: dict[str, str] = {}
    for std in cfg.channels.keep:
        raw_label = pattern.format(name=std)
        if raw_label in raw_ch_names:
            raw_to_std[raw_label] = std
    return raw_to_std


def load_raw(cfg: Config, rec: Recording) -> mne.io.BaseRaw:
    """STAGE 1: load read-only (preloaded), verify sfreq. ALL channels retained.

    No channel selection here: the reference channels (Stage 1a) must still be
    present, so selection happens only AFTER re-referencing.
    """
    raw = mne.io.read_raw_edf(rec.edf_path, preload=True, verbose="ERROR")

    if int(round(raw.info["sfreq"])) != cfg.signal.orig_sfreq:
        raise ValueError(
            f"{rec.key}: sfreq {raw.info['sfreq']} != expected orig_sfreq "
            f"{cfg.signal.orig_sfreq}")
    return raw


def select_channels(cfg: Config, raw: mne.io.BaseRaw, key: str) -> mne.io.BaseRaw:
    """STAGE 1b: rename, select the 18 scalp channels in canonical order.

    Runs AFTER re-referencing, so it drops the (now-consumed) reference channels
    A1/A2 along with Pz and the auxiliary sensors.
    """
    raw_to_std = _rename_map(cfg, raw.ch_names)
    missing = [std for std in cfg.channels.keep
               if cfg.channels.rename_from_pattern.format(name=std) not in raw.ch_names]
    if missing:
        raise ValueError(f"{key}: missing required channels {missing}")

    raw.rename_channels(raw_to_std)
    # Ordered pick -> guarantees identical channel order for every patient (§5.5).
    raw.pick(cfg.channels.keep)
    if list(raw.ch_names) != list(cfg.channels.keep):
        raise ValueError(f"{key}: channel order mismatch after pick: {raw.ch_names}")
    if "Pz" in raw.ch_names:
        raise ValueError(f"{key}: Pz present in kept channels (must be excluded)")
    if len(raw.ch_names) != len(cfg.channels.keep):
        raise ValueError(f"{key}: expected {len(cfg.channels.keep)} channels, got {len(raw.ch_names)}")
    return raw


def load_reref_select(cfg: Config, rec: Recording) -> mne.io.BaseRaw:
    """STAGE 1 -> 1a -> 1b: load, re-reference (first), then select 18 channels.

    Shared by the pipeline and the PSD baseline so both see the identical
    reference and channel set.
    """
    raw = load_raw(cfg, rec)
    raw = apply_reference(cfg, raw, rec.key)   # FIRST transformation (Stage 1a)
    return select_channels(cfg, raw, rec.key)


def filter_resample(cfg: Config, raw: mne.io.BaseRaw, key: str) -> mne.io.BaseRaw:
    """STAGE 2: resample -> zero-phase FIR bandpass -> notch. No sample removal."""
    bp = cfg.signal.bandpass
    raw.resample(cfg.signal.target_sfreq, verbose="ERROR")
    raw.filter(l_freq=bp.l_freq, h_freq=bp.h_freq, picks="eeg",
               method="fir", fir_design=bp.fir_design, phase=bp.phase,
               verbose="ERROR")
    raw.notch_filter(freqs=[cfg.signal.notch_freq], picks="eeg", verbose="ERROR")

    if int(round(raw.info["sfreq"])) != cfg.signal.target_sfreq:
        raise ValueError(f"{key}: post-resample sfreq {raw.info['sfreq']} != {cfg.signal.target_sfreq}")
    return raw


def extract_uV(cfg: Config, raw: mne.io.BaseRaw, key: str) -> np.ndarray:
    """STAGE 3: get_data(units='uV') -> float32 (18, n_samples), finite."""
    data = raw.get_data(units=cfg.signal.extract_units).astype(np.float32)
    if data.shape[0] != len(cfg.channels.keep):
        raise ValueError(f"{key}: extracted {data.shape[0]} channels, expected {len(cfg.channels.keep)}")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{key}: non-finite values after filtering (NaN/Inf)")
    return data


def preprocess_recording(cfg: Config, rec: Recording) -> PreprocessedRecording:
    raw = load_reref_select(cfg, rec)          # Stages 1 -> 1a (reref) -> 1b
    meas_date = raw.info.get("meas_date")       # preserved through reref/select
    raw = filter_resample(cfg, raw, rec.key)
    data = extract_uV(cfg, raw, rec.key)
    return PreprocessedRecording(
        rec=rec, data_uV=data, ch_names=list(raw.ch_names),
        sfreq=cfg.signal.target_sfreq, n_samples=data.shape[1],
        meas_date=meas_date,
    )
