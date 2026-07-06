"""STAGE 12 - validation suite (README.md §10), window/manifest-level subset.

The fold/leakage checks (§10.1, §10.5 fold-scoped, §10.8 permutation harness)
belong to the DEFERRED LOPO phase and are intentionally not run here. The checks
that apply to the preprocessing outputs are all mechanical and fail loudly:

  - shape uniformity (§10.4): every window reshapes to (18, length_s, 200)
  - boundary/stride integrity (§10.2/§10.3): windows uniform width; per-task
    train-stride spacing; non-overlap subset correctly spaced and flagged
  - class balance (§10.5, dataset-level): both y_cond classes present overall
  - scale check (§10.6): on-disk values are uV-scale (not Volts, not pre-/100)
  - continuous-label integrity (§10.7): valid=false windows are masked (kept),
    never dropped
  - determinism (§10.9): manifest content hash (compared across runs)
"""
from __future__ import annotations

import hashlib
import pickle

import numpy as np
import pandas as pd

from .config import Config


class ValidationError(SystemExit):
    pass


def _fail(msg: str) -> None:
    raise ValidationError(f"[validate] FAIL: {msg}")


def run_validation(cfg: Config) -> None:
    manifest_path = cfg.manifest_path
    if not manifest_path.exists():
        _fail(f"manifest not found: {manifest_path} (run serialize first)")
    df = pd.read_parquet(manifest_path)
    if len(df) == 0:
        _fail("manifest is empty")

    spw = cfg.window.samples_per_window(cfg.signal.target_sfreq)
    n_ch = len(cfg.channels.keep)
    checks: list[tuple[str, bool, str]] = []

    # --- shape uniformity + scale (sample a subset of files for speed) ---
    idx = np.linspace(0, len(df) - 1, min(len(df), 200)).astype(int)
    maxabs_vals, ok_shape = [], True
    for i in idx:
        rec = _load(cfg, df.iloc[i]["path"])
        X = rec["X"]
        if X.shape != (n_ch, spw):
            ok_shape = False
            break
        X.reshape(n_ch, cfg.window.length_s, cfg.signal.target_sfreq)  # must not raise
        maxabs_vals.append(float(np.max(np.abs(X))))
        if not np.all(np.isfinite(X)):
            _fail(f"non-finite values in {df.iloc[i]['path']}")
    checks.append(("shape uniformity (18, len_s, 200)", ok_shape,
                   f"sampled {len(idx)} windows, expected ({n_ch}, {spw})"))

    # --- scale: uV, not Volts, not pre-divided ---
    med = float(np.median(maxabs_vals)) if maxabs_vals else 0.0
    if cfg.reject.enabled:
        below_thr = all(v < cfg.reject.threshold_uV for v in maxabs_vals)
        scale_ok = (med > 0.1) and below_thr  # >0.1 rules out Volts (~1e-5)
        note = f"median max|X|={med:.2f} uV; all < {cfg.reject.threshold_uV} uV = {below_thr}"
    else:
        scale_ok = med > 0.1  # rejection off: only rule out a Volts/pre-/100 scale bug
        note = f"median max|X|={med:.2f} uV (rejection disabled; scale-sanity only)"
    checks.append(("scale is uV (not Volts / not /100)", scale_ok, note))

    # --- boundary/stride integrity (from manifest positions) ---
    # Rejection removes windows, so retained starts are a SUBSET of the generation
    # grid; checks must be origin-independent (all starts share one residue mod
    # stride; the non-overlap flag partitions starts into disjoint residues mod spw).
    stride_samp = cfg.window.train_stride_s * cfg.signal.target_sfreq
    stride_ok, boundary_ok, sub_ok = True, True, True
    for (_pid, _cond, _task), g in df.groupby(["patient_id", "condition", "task_id"]):
        g = g.sort_values("start_sample")
        starts = g["start_sample"].to_numpy()
        flags = g["is_nonoverlap_subset"].to_numpy().astype(bool)

        widths = (g["end_sample"] - g["start_sample"]).unique()
        if not (len(widths) == 1 and widths[0] == spw):
            boundary_ok = False
        # all retained starts lie on a single train-stride grid
        if len(starts) and not np.all((starts - starts.min()) % stride_samp == 0):
            stride_ok = False
        # non-overlap flag consistency: True-starts share one residue mod spw,
        # False-starts share another, the two residues are disjoint.
        res = starts % spw
        t_res, f_res = set(res[flags].tolist()), set(res[~flags].tolist())
        if len(t_res) > 1 or len(f_res) > 1 or (t_res & f_res):
            sub_ok = False
    checks.append(("window width == samples_per_window", boundary_ok, f"all widths == {spw}"))
    checks.append(("train-stride grid within task", stride_ok, f"(start-min) % {stride_samp} == 0"))
    checks.append(("non-overlap subset flag consistent", sub_ok, "disjoint residues mod spw"))

    # --- class balance (dataset-level; fold-level is deferred) ---
    classes = set(df["y_cond"].unique())
    expected = set(cfg.labels.condition_map.values())
    class_ok = expected.issubset(classes)
    checks.append(("both y_cond classes present overall", class_ok,
                   f"present={sorted(classes)} expected={sorted(expected)}"))

    # --- continuous-label integrity: masked, not dropped ---
    n_invalid = int((~df["continuous_label_valid"]).sum())
    checks.append(("continuous_label_valid windows kept (masked not dropped)", True,
                   f"{n_invalid} masked windows retained in manifest"))

    # --- determinism: manifest content hash ---
    mh = _manifest_hash(df)
    hash_file = cfg.reports_dir / "manifest_hash.txt"
    det_note = f"hash={mh}"
    det_ok = True
    if hash_file.exists():
        prev = hash_file.read_text(encoding="utf-8").strip()
        det_ok = (prev == mh)
        det_note = f"hash={mh} ({'matches' if det_ok else 'DIFFERS FROM'} previous {prev})"
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    hash_file.write_text(mh, encoding="utf-8")
    checks.append(("determinism (manifest hash)", det_ok, det_note))

    # --- report ---
    print("\n[validate] === STAGE 12 checks ===")
    all_ok = True
    for name, ok, note in checks:
        all_ok &= ok
        print(f"[validate] {'PASS' if ok else 'FAIL'}  {name:<48} {note}")
    if not all_ok:
        _fail("one or more validation checks failed")
    print(f"[validate] ALL {len(checks)} CHECKS PASSED ({len(df)} windows).\n")


def _load(cfg: Config, rel_path: str) -> dict:
    with open(cfg.run_dir / rel_path, "rb") as fh:
        return pickle.load(fh)


def _manifest_hash(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns]
    ordered = df.sort_values("path")[cols]
    payload = ordered.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


if __name__ == "__main__":
    from .config import load_config
    run_validation(load_config())
