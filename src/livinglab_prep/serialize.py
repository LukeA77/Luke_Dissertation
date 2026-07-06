"""STAGE 9 - metadata + serialization (README.md §8/§9).

Each retained window -> <run_dir>/windows/<patient>/<key>_<taskslug>_<idx>.pkl as
a dict {X (uV, unscaled), y_cond, t_cont, meta{...}}, where <run_dir> is the
reference-scheme-scoped output root (processed/reref-<scheme>/). Every window row
is also appended to a single <run_dir>/manifest.parquet. config_hash + git_sha are
stamped into every window's meta and are the provenance anchor (§5.9).
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import pandas as pd

from .config import Config
from .label import LabeledWindow
from .preprocess import PreprocessedRecording

_MANIFEST_COLS = [
    "path", "patient_id", "condition", "y_cond", "t_cont",
    "continuous_label_valid", "task_id", "window_idx", "stride_s",
    "is_nonoverlap_subset", "start_sample", "end_sample", "sfreq",
]


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")


class Serializer:
    """Accumulates window files + manifest rows across recordings, one parquet out."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rows: list[dict] = []
        cfg.windows_dir.mkdir(parents=True, exist_ok=True)

    def write_recording(self, pre: PreprocessedRecording,
                        kept: list[LabeledWindow]) -> int:
        cfg = self.cfg
        rec = pre.rec
        out_dir = cfg.windows_dir / rec.patient_id
        out_dir.mkdir(parents=True, exist_ok=True)

        for lw in kept:
            w = lw.win
            fname = f"{rec.key}_{_slug(w.task_id)}_{w.window_idx:05d}.pkl"
            fpath = out_dir / fname
            meta = {
                "patient_id": rec.patient_id,
                "condition": rec.condition,
                "task_id": w.task_id,
                "window_idx": w.window_idx,
                "stride_s": w.stride_s,
                "is_nonoverlap_subset": w.is_nonoverlap_subset,
                "continuous_label_valid": lw.continuous_label_valid,
                "sfreq": pre.sfreq,
                "units": cfg.signal.extract_units,
                "config_hash": cfg.config_hash,
                "git_sha": cfg.git_sha,
            }
            record = {"X": w.X, "y_cond": int(lw.y_cond),
                      "t_cont": float(lw.t_cont), "meta": meta}
            with open(fpath, "wb") as fh:
                pickle.dump(record, fh, protocol=pickle.HIGHEST_PROTOCOL)

            self.rows.append({
                "path": str(fpath.relative_to(cfg.run_dir).as_posix()),
                "patient_id": rec.patient_id,
                "condition": rec.condition,
                "y_cond": int(lw.y_cond),
                "t_cont": float(lw.t_cont),
                "continuous_label_valid": bool(lw.continuous_label_valid),
                "task_id": w.task_id,
                "window_idx": int(w.window_idx),
                "stride_s": int(w.stride_s),
                "is_nonoverlap_subset": bool(w.is_nonoverlap_subset),
                "start_sample": int(w.start_sample),
                "end_sample": int(w.end_sample),
                "sfreq": int(pre.sfreq),
            })

        print(f"[stage9] {rec.key}: wrote {len(kept)} window files -> {out_dir}")
        return len(kept)

    def finalize(self) -> Path:
        cfg = self.cfg
        df = pd.DataFrame(self.rows, columns=_MANIFEST_COLS)
        # No NaNs allowed in required columns (§9 acceptance).
        req = [c for c in _MANIFEST_COLS if c != "t_cont"]  # t_cont always finite too
        if df[req].isna().any().any() or df["t_cont"].isna().any():
            raise ValueError("manifest contains NaN in required columns")
        out = cfg.manifest_path
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"[stage9] manifest: {len(df)} rows -> {out}")
        return out
