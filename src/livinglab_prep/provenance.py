"""Provenance manifest (README.md §5.9 determinism/provenance).

Writes <run_dir>/provenance.json capturing everything needed to reproduce or
audit a run: the full validated config, the reference scheme, the git commit
hash (if available), a UTC timestamp, and the exact list of input files. One file
per reference scheme (isolated by run_dir), so a linked-ears run can never be
confused with, or overwrite, a differently-referenced run.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .config import Config
from .discovery import Recording


def write_provenance(cfg: Config, recordings: list[Recording]) -> Path:
    """Write the run's provenance manifest as JSON into ``cfg.run_dir``.

    Parameters
    ----------
    cfg : Config
        The loaded, provenance-stamped config for this run.
    recordings : list of Recording
        The recordings processed in this run (their EDF/CSV inputs are recorded).

    Returns
    -------
    pathlib.Path
        Path to the written provenance.json.
    """
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "reference_scheme": cfg.reref.scheme,
        "reref_enabled": cfg.reref.enabled,
        "ref_channels": list(cfg.reref.ref_channels),
        "git_sha": cfg.git_sha,
        "config_hash": cfg.config_hash,
        "run_dir": str(cfg.run_dir),
        "inputs": [
            {
                "key": rec.key,
                "patient_id": rec.patient_id,
                "condition": rec.condition,
                "edf": str(rec.edf_path),
                "csv": str(rec.csv_path),
            }
            for rec in recordings
        ],
        # Full validated config (single source of truth) for exact reproduction.
        "config": cfg.model_dump(mode="json"),
    }

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.run_dir / "provenance.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[provenance] wrote {out}")
    return out
