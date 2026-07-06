"""Pipeline orchestrator: runs Stages 1-9 per recording, then reports.

Stages are coupled in memory (the continuous signal flows straight from load to
windowing), so the whole per-recording chain runs in one pass. Stage 0 (preflight)
and Stage 12 (validation) are separate entry points in the CLI. Determinism
(§5.9): recordings are processed in a stable discovery order and windows are
written in sequential order, so identical config+data -> identical manifest.
"""
from __future__ import annotations

import random

import numpy as np

from .align import align_recording
from .config import Config
from .discovery import discover_recordings
from .label import label_windows
from .preprocess import preprocess_recording
from .provenance import write_provenance
from .psd import compute_psd_check
from .reject import reject_windows
from .report import RecordingSummary, write_reports
from .serialize import Serializer
from .window import make_windows


def run_pipeline(cfg: Config) -> list[RecordingSummary]:
    random.seed(cfg.run.seed)
    np.random.seed(cfg.run.seed)

    recordings = discover_recordings(cfg)
    print(f"[pipeline] discovered {len(recordings)} recordings: "
          f"{[r.key for r in recordings]}")
    print(f"[pipeline] reference scheme='{cfg.reref.scheme}' "
          f"(ref_channels={cfg.reref.ref_channels or 'recorded'}) -> {cfg.run_dir}")
    write_provenance(cfg, recordings)

    serializer = Serializer(cfg)
    summaries: list[RecordingSummary] = []

    for rec in recordings:
        print(f"\n[pipeline] === {rec.key} ({rec.condition}) ===")
        pre = preprocess_recording(cfg, rec)          # Stages 1-3
        psd = compute_psd_check(cfg, pre)             # Stage 4
        align = align_recording(cfg, pre)            # Stage 5
        windows = make_windows(cfg, pre, align)       # Stage 6
        labeled = label_windows(cfg, pre, align, windows)  # Stage 7
        kept, rstats = reject_windows(cfg, rec.key, labeled)  # Stage 8
        n = serializer.write_recording(pre, kept)     # Stage 9

        summaries.append(RecordingSummary(
            key=rec.key, patient_id=rec.patient_id, condition=rec.condition,
            align=align, reject=rstats, psd=psd, n_windows=n))

    serializer.finalize()
    write_reports(cfg, summaries)
    return summaries
