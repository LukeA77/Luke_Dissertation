"""STAGE 7 - dual labelling (README.md §8).

Per window: (a) binary condition from labels.condition_map (A->0, C->1); (b)
continuous target = elapsed seconds from continuous_reference to the window
CENTRE. continuous_reference is session_start (confirmed), i.e. EEG t=0, so the
elapsed time is simply the centre sample / sfreq -- exact and independent of the
CSV offset. continuous_label_valid is carried from Stage 5 (masked, not dropped,
if alignment was unresolved). Optional fixed normalization per config.
"""
from __future__ import annotations

from dataclasses import dataclass

from .align import AlignmentResult
from .config import Config
from .preprocess import PreprocessedRecording
from .window import Window


@dataclass
class LabeledWindow:
    win: Window
    y_cond: int
    t_cont: float
    continuous_label_valid: bool


def _continuous_target(cfg: Config, pre: PreprocessedRecording, win: Window) -> float:
    ref = cfg.labels.continuous_reference
    centre_sample = (win.start_sample + win.end_sample) / 2.0
    if ref == "session_start":
        elapsed = centre_sample / pre.sfreq
    else:  # task_start (not selected for this build, but supported by the contract)
        raise NotImplementedError(
            "continuous_reference=task_start not implemented (config selects session_start)")

    if cfg.labels.continuous_normalize == "none":
        return float(elapsed)
    if cfg.labels.continuous_normalize == "per_session_minmax":
        total = pre.n_samples / pre.sfreq
        return float(elapsed / total) if total > 0 else 0.0
    raise ValueError(f"unknown continuous_normalize: {cfg.labels.continuous_normalize}")


def label_windows(cfg: Config, pre: PreprocessedRecording,
                  align: AlignmentResult, windows: list[Window]) -> list[LabeledWindow]:
    y_cond = cfg.labels.condition_map[pre.rec.condition]
    labeled = [
        LabeledWindow(
            win=w,
            y_cond=y_cond,
            t_cont=_continuous_target(cfg, pre, w),
            continuous_label_valid=align.continuous_label_valid,
        )
        for w in windows
    ]

    # Acceptance: continuous target monotonic within the session (sample order).
    ts = [lw.t_cont for lw in labeled]
    if any(b < a for a, b in zip(ts, ts[1:])):
        raise AssertionError(f"{align.key}: continuous target not monotonic within session")

    print(f"[stage7] {align.key}: y_cond={y_cond} "
          f"({pre.rec.condition}), t_cont {ts[0]:.1f}..{ts[-1]:.1f}s, "
          f"cont_valid={align.continuous_label_valid}" if ts else
          f"[stage7] {align.key}: no windows to label")
    return labeled
