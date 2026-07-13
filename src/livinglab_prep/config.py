"""Config system (README.md Stage/§6): the single source of truth.

Loads config/pipeline.yaml into a validated, frozen pydantic model. Unknown keys
are rejected (extra='forbid'); types and ranges are checked; and every load
records a deterministic config hash + the current git SHA so each output artifact
is traceable (README.md §5.9 determinism/provenance).

No parameter defined here may also appear as a literal in the rest of the code.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


# Project root = two levels up from this file (src/livinglab_prep/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _Strict(BaseModel):
    """Base: reject unknown keys, freeze after construction."""
    model_config = ConfigDict(extra="forbid", frozen=True)


class Paths(_Strict):
    raw_dir: str
    csv_dir: str
    processed_dir: str
    scratch_dir: str


class Bandpass(_Strict):
    l_freq: float = Field(gt=0)
    h_freq: float = Field(gt=0)
    fir_design: str
    phase: str

    @model_validator(mode="after")
    def _check_band(self) -> "Bandpass":
        if self.l_freq >= self.h_freq:
            raise ValueError(f"bandpass l_freq ({self.l_freq}) must be < h_freq ({self.h_freq})")
        return self


class Channels(_Strict):
    keep: list[str]
    rename_from_pattern: str
    reference_note: str

    @model_validator(mode="after")
    def _check_keep(self) -> "Channels":
        if len(self.keep) != len(set(self.keep)):
            raise ValueError("channels.keep contains duplicates")
        if "Pz" in self.keep:
            raise ValueError("Pz must NOT be in channels.keep (degenerate reference)")
        if "{name}" not in self.rename_from_pattern:
            raise ValueError("channels.rename_from_pattern must contain '{name}'")
        return self


class Reref(_Strict):
    """Re-referencing scheme applied as the FIRST transformation (before channel
    selection and filtering).

    `ref_channels` are STANDARD electrode names (e.g. [A1, A2] for linked ears);
    an empty list means "no re-referencing" (keep the recorded Pz reference).
    `scheme` is a filesystem-safe identifier that derives the output subdirectory
    (`processed/reref-<scheme>/`), so different reference choices can never
    overwrite each other's outputs.
    """
    scheme: str
    ref_channels: list[str]

    @model_validator(mode="after")
    def _check(self) -> "Reref":
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.scheme):
            raise ValueError(
                f"reref.scheme must be a non-empty slug [A-Za-z0-9_-], got {self.scheme!r}")
        if len(self.ref_channels) != len(set(self.ref_channels)):
            raise ValueError("reref.ref_channels contains duplicates")
        return self

    @property
    def enabled(self) -> bool:
        """True iff an explicit re-reference is requested (non-empty ref set)."""
        return len(self.ref_channels) > 0


class Signal(_Strict):
    orig_sfreq: int = Field(gt=0)
    target_sfreq: int = Field(gt=0)
    bandpass: Bandpass
    notch_freq: float = Field(gt=0)
    extract_units: Literal["uV"]

    @model_validator(mode="after")
    def _check(self) -> "Signal":
        nyq = self.target_sfreq / 2
        if self.bandpass.h_freq >= nyq:
            raise ValueError(
                f"bandpass h_freq ({self.bandpass.h_freq}) must be < target Nyquist ({nyq})")
        if self.notch_freq >= nyq:
            raise ValueError(f"notch_freq ({self.notch_freq}) must be < target Nyquist ({nyq})")
        return self


class Window(_Strict):
    length_s: int = Field(gt=0)
    train_stride_s: int = Field(gt=0)
    eval_stride_s: int = Field(gt=0)
    samples_per_patch: int = Field(gt=0)

    @model_validator(mode="after")
    def _check(self) -> "Window":
        if self.eval_stride_s != self.length_s:
            raise ValueError("eval_stride_s must equal length_s (non-overlapping eval subset)")
        if self.train_stride_s > self.length_s:
            raise ValueError("train_stride_s must be <= length_s")
        return self

    def samples_per_window(self, target_sfreq: int) -> int:
        return self.length_s * target_sfreq

    def n_patches(self) -> int:
        return self.length_s


class Reject(_Strict):
    enabled: bool
    mode: Literal["peak_abs_whole_window"]
    threshold_uV: float = Field(gt=0)
    warn_drop_frac: float = Field(ge=0, le=1)


class Ica(_Strict):
    """Independent Component Analysis artifact removal (mne + mne-icalabel).

    Fitted PER RECORDING on the EEG scalp channels only, on a high-passed copy,
    then applied to the band-passed data. Extended infomax is required by ICLabel.
    All numeric knobs are here so the decomposition is fully reproducible.
    """
    enabled: bool
    n_components: int = Field(gt=0)
    method: Literal["infomax"]              # ICLabel requires (extended) infomax
    extended: bool
    max_iter: int = Field(gt=0)
    fit_highpass_hz: float = Field(gt=0)    # high-pass for the fit copy (ICA hates drift)
    fit_sfreq: float = Field(gt=0)          # resample the fit/label copy to this (ICLabel is a 100 Hz model)
    decim: int = Field(ge=1)                # decimation for the fit (memory/determinism)
    random_state: int
    iclabel_min_prob: float = Field(ge=0, le=1)
    exclude_labels: list[str]               # ICLabel classes to remove (never brain/other)
    eog_channels: list[str]                 # frontal channels used as the blink-correlation proxy
    eog_corr_threshold: float = Field(ge=0, le=1)  # |r| cutoff for find_bads_eog(measure='correlation')
    eog_l_freq: float = Field(gt=0)         # blink-band low edge for find_bads_eog's internal filter
    eog_h_freq: float = Field(gt=0)         # blink-band high edge for find_bads_eog's internal filter
    save_plots: bool
    plot_crop_s: float = Field(gt=0)        # segment length for per-component property plots

    @model_validator(mode="after")
    def _check(self) -> "Ica":
        allowed = {"brain", "muscle artifact", "eye blink", "heart beat",
                   "line noise", "channel noise", "other"}
        bad = [lbl for lbl in self.exclude_labels if lbl not in allowed]
        if bad:
            raise ValueError(f"ica.exclude_labels has unknown ICLabel classes {bad}; "
                             f"allowed: {sorted(allowed)}")
        for keep in ("brain", "other"):
            if keep in self.exclude_labels:
                raise ValueError(f"ica.exclude_labels must never contain '{keep}' "
                                 "(brain and other are always kept)")
        if "eye blink" in self.exclude_labels:
            raise ValueError(
                "ica.exclude_labels must not contain 'eye blink': ICLabel's winning "
                "label misses blink components off-distribution (see PROGRESS.md "
                "Problem 2); blinks are identified separately via eog_channels/"
                "eog_corr_threshold (frontal-correlation, find_bads_eog).")
        if not self.eog_channels:
            raise ValueError("ica.eog_channels must list at least one frontal channel")
        return self


class Sweep(_Strict):
    """Amplitude-threshold sweep grid for the before/after-ICA survival tables."""
    thresholds_uV: list[float]

    @model_validator(mode="after")
    def _check(self) -> "Sweep":
        if not self.thresholds_uV:
            raise ValueError("sweep.thresholds_uV must be a non-empty list")
        if any(t <= 0 for t in self.thresholds_uV):
            raise ValueError("sweep.thresholds_uV values must all be > 0")
        if len(self.thresholds_uV) != len(set(self.thresholds_uV)):
            raise ValueError("sweep.thresholds_uV contains duplicates")
        return self


class Labels(_Strict):
    condition_map: dict[str, int]
    continuous_target: str
    continuous_reference: Literal["session_start", "task_start"]
    continuous_normalize: Literal["none", "per_session_minmax"]


class Align(_Strict):
    max_reconcile_error_s: float = Field(gt=0)
    min_tasks_fit_frac: float = Field(ge=0, le=1)


class Split(_Strict):
    strategy: str
    shuffle_train_windows: bool
    seed: int


class Run(_Strict):
    seed: int


class Config(_Strict):
    paths: Paths
    channels: Channels
    reref: Reref
    signal: Signal
    ica: Ica
    sweep: Sweep
    window: Window
    reject: Reject
    labels: Labels
    align: Align
    split: Split
    run: Run

    # Populated at load time (excluded from the hash input).
    config_hash: str = ""
    git_sha: str = ""

    # --- resolved absolute paths (relative entries are resolved against root) ---
    def _resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (PROJECT_ROOT / path)

    @property
    def raw_dir(self) -> Path:
        return self._resolve(self.paths.raw_dir)

    @property
    def csv_dir(self) -> Path:
        return self._resolve(self.paths.csv_dir)

    @property
    def processed_dir(self) -> Path:
        return self._resolve(self.paths.processed_dir)

    @property
    def run_dir(self) -> Path:
        """Reference-scheme-scoped output root: processed/reref-<scheme>/.

        Every writable artifact (windows, manifest, reports, provenance) lives
        under here, so two reference schemes can never overwrite each other.
        """
        return self.processed_dir / f"reref-{self.reref.scheme}"

    @property
    def reports_dir(self) -> Path:
        return self.run_dir / "_reports"

    @property
    def windows_dir(self) -> Path:
        return self.run_dir / "windows"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.parquet"

    @property
    def ica_run_dir(self) -> Path:
        """ICA-cleaned derivatives root: processed/reref-<scheme>_ica/.

        A sibling of run_dir so ICA-cleaned outputs never overwrite the
        reref-only (or original Pz) outputs.
        """
        return self.processed_dir / f"reref-{self.reref.scheme}_ica"


def _compute_config_hash(raw: dict) -> str:
    """Deterministic hash of the raw config dict (sorted keys, no provenance fields)."""
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
        return "no-commit"
    except Exception:
        return "no-git"


def load_config(path: str | Path | None = None) -> Config:
    """Load, validate, and stamp the pipeline config."""
    cfg_path = Path(path) if path else (PROJECT_ROOT / "config" / "pipeline.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = Config(**raw)
    # Re-create with provenance stamped (model is frozen).
    return cfg.model_copy(update={
        "config_hash": _compute_config_hash(raw),
        "git_sha": _git_sha(),
    })


if __name__ == "__main__":
    c = load_config()
    print("Config OK")
    print("  config_hash:", c.config_hash)
    print("  git_sha    :", c.git_sha)
    print("  raw_dir    :", c.raw_dir)
    print("  channels   :", len(c.channels.keep), "->", c.channels.keep)
    print("  samples/win:", c.window.samples_per_window(c.signal.target_sfreq))
