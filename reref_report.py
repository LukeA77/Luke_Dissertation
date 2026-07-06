"""Before/after re-referencing diagnostic (validation for the linked-ears change).

Measures, in isolation, how much the linked-ears re-reference recovers -- WITHOUT
touching the amplitude threshold or adding ICA. For each recording it builds the
identical pipeline output twice:

  * BEFORE : the recorded single-Pz reference (no re-referencing)
  * AFTER  : re-referenced to the configured ear channels (linked ears)

...running the SAME channel selection, filtering, resampling, task alignment and
task-locked windowing for both (so the two differ ONLY in the reference), then
reports:

  1. Per-channel median peak amplitude (median over windows of per-window
     max|x|), before vs after, with the inflation-drop factor.
  2. Window survival: total windows and % that would be rejected at the config's
     100 uV whole-window peak-abs threshold, per recording and overall, versus
     the prior ~98% rejection under the Pz reference.

No files are written and no randomness is involved (deterministic). Run:

    python reref_report.py            # uses config/pipeline.yaml
    python reref_report.py --config <path>
"""
from __future__ import annotations

import argparse

import numpy as np

from src.livinglab_prep.align import align_recording
from src.livinglab_prep.config import Config, load_config
from src.livinglab_prep.discovery import Recording, discover_recordings
from src.livinglab_prep.preprocess import (
    PreprocessedRecording,
    extract_uV,
    filter_resample,
    load_raw,
    select_channels,
)
from src.livinglab_prep.reference import apply_reference
from src.livinglab_prep.window import make_windows


def _preprocess(cfg: Config, rec: Recording, *, do_reref: bool) -> PreprocessedRecording:
    """Run stages 1-3 with re-referencing toggled on/off (else identical)."""
    raw = load_raw(cfg, rec)
    meas_date = raw.info.get("meas_date")
    if do_reref:
        raw = apply_reference(cfg, raw, rec.key)
    raw = select_channels(cfg, raw, rec.key)
    raw = filter_resample(cfg, raw, rec.key)
    data = extract_uV(cfg, raw, rec.key)
    return PreprocessedRecording(
        rec=rec, data_uV=data, ch_names=list(raw.ch_names),
        sfreq=cfg.signal.target_sfreq, n_samples=data.shape[1], meas_date=meas_date,
    )


def _window_stack(cfg: Config, pre: PreprocessedRecording) -> np.ndarray:
    """Task-locked windows as one array (n_windows, n_channels, samples_per_window).

    Uses the exact pipeline windowing so counts match the pipeline manifest.
    """
    align = align_recording(cfg, pre)
    windows = make_windows(cfg, pre, align)
    if not windows:
        n_ch = len(cfg.channels.keep)
        spw = cfg.window.samples_per_window(cfg.signal.target_sfreq)
        return np.empty((0, n_ch, spw), dtype=np.float32)
    return np.stack([w.X for w in windows], axis=0)


def _per_channel_median_peak(stack: np.ndarray) -> np.ndarray:
    """Median over windows of per-window per-channel max|x| -> (n_channels,)."""
    if stack.shape[0] == 0:
        return np.full(stack.shape[1], np.nan)
    per_win_peak = np.max(np.abs(stack), axis=2)      # (n_windows, n_channels)
    return np.median(per_win_peak, axis=0)


def _reject_fraction(stack: np.ndarray, threshold_uV: float) -> tuple[int, int]:
    """(n_rejected, n_total) at whole-window peak-abs >= threshold (reject.py rule)."""
    if stack.shape[0] == 0:
        return 0, 0
    whole_peak = np.max(np.abs(stack), axis=(1, 2))   # (n_windows,)
    n_reject = int(np.sum(whole_peak >= threshold_uV))
    return n_reject, stack.shape[0]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="reref_report")
    parser.add_argument("--config", default=None, help="path to pipeline.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    thr = cfg.reject.threshold_uV
    channels = list(cfg.channels.keep)
    print(f"[reref_report] config_hash={cfg.config_hash} git_sha={cfg.git_sha}")
    print(f"[reref_report] scheme='{cfg.reref.scheme}' ref_channels={cfg.reref.ref_channels} "
          f"threshold={thr} uV\n")

    if not cfg.reref.enabled:
        raise SystemExit("[reref_report] reref.ref_channels is empty -- nothing to compare "
                         "(set the linked-ears reference in config).")

    recordings = discover_recordings(cfg)
    per_channel_rows: dict[str, list[tuple[float, float]]] = {}  # key -> [(before, after)]
    survival_rows: list[tuple[str, int, int, int]] = []          # key, n, rej_before, rej_after
    logged_channels = False

    for rec in recordings:
        # Channel-name safety visibility: log the raw labels once.
        if not logged_channels:
            raw0 = load_raw(cfg, rec)
            print(f"[reref_report] raw channels in {rec.key} ({len(raw0.ch_names)}): "
                  f"{list(raw0.ch_names)}\n")
            del raw0
            logged_channels = True

        before = _preprocess(cfg, rec, do_reref=False)
        after = _preprocess(cfg, rec, do_reref=True)

        stack_before = _window_stack(cfg, before)
        stack_after = _window_stack(cfg, after)
        if stack_before.shape[0] != stack_after.shape[0]:
            raise AssertionError(
                f"{rec.key}: window count differs before/after reref "
                f"({stack_before.shape[0]} vs {stack_after.shape[0]})")

        med_before = _per_channel_median_peak(stack_before)
        med_after = _per_channel_median_peak(stack_after)
        per_channel_rows[rec.key] = list(zip(med_before, med_after))

        rej_b, n = _reject_fraction(stack_before, thr)
        rej_a, _ = _reject_fraction(stack_after, thr)
        survival_rows.append((rec.key, n, rej_b, rej_a))

    _print_channel_tables(channels, per_channel_rows)
    _print_survival_table(survival_rows, thr)


def _fmt(x: float) -> str:
    return "  nan" if not np.isfinite(x) else f"{x:7.1f}"


def _print_channel_tables(channels: list[str],
                          rows: dict[str, list[tuple[float, float]]]) -> None:
    print("\n=== 1. PER-CHANNEL MEDIAN PEAK AMPLITUDE (uV): BEFORE (Pz) vs AFTER (linked ears) ===")
    for key, pairs in rows.items():
        print(f"\n{key}")
        print(f"  {'ch':<5}{'before':>9}{'after':>9}{'drop x':>9}")
        print("  " + "-" * 32)
        befores, afters = [], []
        for ch, (b, a) in zip(channels, pairs):
            factor = (b / a) if (np.isfinite(a) and a > 0) else np.nan
            fac = "   nan" if not np.isfinite(factor) else f"{factor:5.2f}"
            print(f"  {ch:<5}{_fmt(b):>9}{_fmt(a):>9}{fac:>9}")
            befores.append(b); afters.append(a)
        mb, ma = np.nanmedian(befores), np.nanmedian(afters)
        fac_all = (mb / ma) if (np.isfinite(ma) and ma > 0) else np.nan
        print("  " + "-" * 32)
        print(f"  {'med':<5}{_fmt(mb):>9}{_fmt(ma):>9}"
              f"{('   nan' if not np.isfinite(fac_all) else f'{fac_all:5.2f}'):>9}")


def _print_survival_table(rows: list[tuple[str, int, int, int]], thr: float) -> None:
    print(f"\n\n=== 2. WINDOW SURVIVAL at {thr:.0f} uV whole-window peak-abs ===")
    print(f"(prior baseline under Pz reference: 23/1352 survived -> ~98.3% rejected)\n")
    hdr = f"{'recording':<14}{'windows':>9}{'rej BEFORE (Pz)':>20}{'rej AFTER (ears)':>20}"
    print(hdr); print("-" * len(hdr))
    tot_n = tot_rb = tot_ra = 0
    for key, n, rb, ra in rows:
        tot_n += n; tot_rb += rb; tot_ra += ra
        pb = f"{rb}/{n} ({100*rb/n:.1f}%)" if n else "0/0 (n/a)"
        pa = f"{ra}/{n} ({100*ra/n:.1f}%)" if n else "0/0 (n/a)"
        print(f"{key:<14}{n:>9}{pb:>20}{pa:>20}")
    print("-" * len(hdr))
    pb = f"{tot_rb}/{tot_n} ({100*tot_rb/tot_n:.1f}%)" if tot_n else "n/a"
    pa = f"{tot_ra}/{tot_n} ({100*tot_ra/tot_n:.1f}%)" if tot_n else "n/a"
    print(f"{'OVERALL':<14}{tot_n:>9}{pb:>20}{pa:>20}")
    if tot_n:
        surv_before = 100 * (tot_n - tot_rb) / tot_n
        surv_after = 100 * (tot_n - tot_ra) / tot_n
        print(f"\nSurvival: {surv_before:.1f}% (before) -> {surv_after:.1f}% (after linked-ears).")


if __name__ == "__main__":
    main()
