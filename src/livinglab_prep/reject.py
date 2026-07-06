"""STAGE 8 - per-window amplitude rejection (README.md §8).

Non-adaptive, single global threshold (the ONLY rejection in the reference repos,
from CBraMod pretraining): keep a window iff its whole-window peak absolute value
is below reject.threshold_uV. Applied AFTER filtering/resampling/windowing, in uV.
Logs the per-recording drop fraction and warns loudly if it exceeds warn_drop_frac
(likely interacts with the Pz reference -- recorded in limitations).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .label import LabeledWindow


@dataclass
class RejectStats:
    key: str
    n_in: int
    n_kept: int
    drop_frac: float


def reject_windows(cfg: Config, key: str,
                   labeled: list[LabeledWindow]) -> tuple[list[LabeledWindow], RejectStats]:
    if not cfg.reject.enabled:
        stats = RejectStats(key, len(labeled), len(labeled), 0.0)
        print(f"[stage8] {key}: rejection disabled, kept all {len(labeled)}")
        return labeled, stats

    thr = cfg.reject.threshold_uV
    kept = [lw for lw in labeled if float(np.max(np.abs(lw.win.X))) < thr]

    n_in = len(labeled)
    drop_frac = (n_in - len(kept)) / n_in if n_in else 0.0
    stats = RejectStats(key, n_in, len(kept), drop_frac)

    # Retained windows must stay finite and correct shape.
    for lw in kept:
        if not np.all(np.isfinite(lw.win.X)):
            raise ValueError(f"{key}: retained window {lw.win.window_idx} has non-finite values")

    print(f"[stage8] {key}: kept {len(kept)}/{n_in} "
          f"(drop {drop_frac:.1%}, threshold {thr} uV)")
    if drop_frac > cfg.reject.warn_drop_frac:
        print(f"[stage8] WARNING: {key} drop fraction {drop_frac:.1%} > "
              f"{cfg.reject.warn_drop_frac:.0%} -- inspect (may interact with Pz reference)")
    return kept, stats
