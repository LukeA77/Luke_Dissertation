"""STAGE 6 - task-locked windowing (README.md §8).

For each aligned task segment, cut windows of window.length_s that lie ENTIRELY
within the (recording-clamped) segment, at the finest stride (train_stride_s);
the trailing remainder is discarded and windows NEVER cross a task boundary
(§5.4). Each window is tagged is_nonoverlap_subset when it sits on the every-
length_s grid from the task start (the subset used by eval roles later).

Windows carry raw uV data, unscaled (the /100 lives only in the deferred loader).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .align import AlignmentResult
from .config import Config
from .preprocess import PreprocessedRecording


@dataclass
class Window:
    X: np.ndarray            # (18, samples_per_window) float32, uV
    task_id: str
    window_idx: int          # sequential within the recording
    start_sample: int
    end_sample: int          # exclusive
    stride_s: int
    is_nonoverlap_subset: bool


def make_windows(cfg: Config, pre: PreprocessedRecording,
                 align: AlignmentResult) -> list[Window]:
    sfreq = pre.sfreq
    spw = cfg.window.samples_per_window(sfreq)          # samples per window
    stride = cfg.window.train_stride_s * sfreq          # finest stride
    data = pre.data_uV
    windows: list[Window] = []
    idx = 0

    for task in align.aligned_tasks:
        lo, hi = task.usable_start, task.usable_end     # already clamped to [0, n]
        # Every start that keeps the whole window inside [lo, hi).
        start = lo
        while start + spw <= hi:
            end = start + spw
            X = np.ascontiguousarray(data[:, start:end], dtype=np.float32)
            # Boundary integrity assertion (§10.2): window fully inside the segment.
            if not (start >= lo and end <= hi):
                raise AssertionError(
                    f"{align.key} '{task.task_id}': window [{start},{end}] escapes "
                    f"segment [{lo},{hi}]")
            offset = start - lo
            is_nonoverlap = (offset % spw) == 0
            windows.append(Window(
                X=X, task_id=task.task_id, window_idx=idx,
                start_sample=start, end_sample=end,
                stride_s=cfg.window.train_stride_s,
                is_nonoverlap_subset=is_nonoverlap,
            ))
            idx += 1
            start += stride

    n_sub = sum(w.is_nonoverlap_subset for w in windows)
    print(f"[stage6] {align.key}: {len(windows)} windows "
          f"({n_sub} non-overlap subset) from {len(align.aligned_tasks)} tasks, "
          f"shape (18, {spw})")
    return windows
