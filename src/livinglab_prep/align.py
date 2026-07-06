"""STAGE 5 - timestamp alignment (README.md §8): map task-PC clock -> EEG sample.

Preference order (§8): (1) raw.info['meas_date'] vs CSV start; (2) EDF+
annotations; (3) trigger/stim channel. For this dataset only (1) is available:
there are NO annotations and NO usable trigger markers (see memory:
subject-131-clock-offset), so wall-clock is the sole anchor.

meas_date gives EEG t=0 as a wall-clock epoch; a task at epoch time T maps to
EEG time (T - offset). The base offset is meas_date's epoch. Because the task-PC
and EEG-amp clocks can disagree by a whole-hour timezone/DST misconfiguration
(sub-131 was recorded in BST but stamped as if UTC -> ~1 h gap), we test a small
set of WHOLE-HOUR corrections and pick the one under which the most task
segments land inside the recording. Whole-hour only: fractional "corrections"
would be fabricating alignment, which §5.8 forbids.

Offset-vs-drift (§8): a single constant offset must reconcile BOTH the first and
last usable task (both inside the recording). We have no independent event pairs
to fit a linear drift term (wall-clock forces slope=1), so if a constant offset
cannot reconcile first+last the mapping is declared unresolved and the recording
is flagged continuous_label_valid=false (never fabricate).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .preprocess import PreprocessedRecording
from .tasks import TaskSegment, parse_task_csv, usable_duration

# Whole-hour candidate corrections (seconds) motivated by timezone/DST misconfig.
_HOUR_CANDIDATES = [0, -3600, 3600, -7200, 7200]


@dataclass
class AlignedTask:
    task_id: str
    start_sample: int
    end_sample: int
    # Clamped-to-recording sample bounds actually usable for windowing.
    usable_start: int
    usable_end: int


@dataclass
class AlignmentResult:
    key: str
    method: str
    offset_s: float               # subtracted: t_eeg = csv_s - offset_s
    correction_s: int             # whole-hour correction chosen
    n_fit: int                    # usable tasks fully inside recording at chosen offset
    n_usable: int                 # usable task segments considered
    fit_frac: float
    reconciled: bool              # constant offset places first & last usable task inside
    continuous_label_valid: bool
    note: str
    aligned_tasks: list[AlignedTask] = field(default_factory=list)


def _epoch_s(pre: PreprocessedRecording) -> float:
    if pre.meas_date is None:
        raise ValueError(f"{pre.rec.key}: EDF has no meas_date; cannot align via wall-clock")
    return pre.meas_date.timestamp()


def _count_inside(tasks: list[TaskSegment], offset_s: float, rec_len_s: float) -> int:
    n = 0
    for t in tasks:
        s, e = t.start_s - offset_s, t.end_s - offset_s
        if s >= 0 and e <= rec_len_s:
            n += 1
    return n


def align_recording(cfg: Config, pre: PreprocessedRecording) -> AlignmentResult:
    rec_len_s = pre.n_samples / pre.sfreq
    base = _epoch_s(pre)
    min_len = float(cfg.window.length_s)

    all_tasks = parse_task_csv(pre.rec.csv_path)
    usable_tasks = [t for t in all_tasks
                    if usable_duration(t, rec_len_s, min_len)[0]]
    if not usable_tasks:
        raise ValueError(f"{pre.rec.key}: no usable task segments to align")

    # Pick the whole-hour correction that maximises tasks landing inside recording.
    best_corr, best_hits = 0, -1
    for c in _HOUR_CANDIDATES:
        hits = _count_inside(usable_tasks, base + c, rec_len_s)
        if hits > best_hits:
            best_hits, best_corr = hits, c
    offset_s = base + best_corr

    # Offset-vs-drift: does one constant offset reconcile first AND last usable task?
    def inside(t: TaskSegment) -> bool:
        return (t.start_s - offset_s) >= 0 and (t.end_s - offset_s) <= rec_len_s
    first_ok, last_ok = inside(usable_tasks[0]), inside(usable_tasks[-1])
    reconciled = first_ok and last_ok

    fit_frac = best_hits / len(usable_tasks)
    continuous_valid = fit_frac >= cfg.align.min_tasks_fit_frac

    if not continuous_valid:
        note = (f"UNRESOLVED: only {fit_frac:.0%} of usable tasks fit at best whole-hour "
                f"offset (correction={best_corr}s); continuous target masked.")
    elif not reconciled:
        note = (f"partial: constant offset (correction={best_corr}s) fits {fit_frac:.0%} "
                f"but first/last not both inside -> some tasks outside recording, clamped.")
    else:
        note = f"resolved: constant offset via meas_date + {best_corr}s correction."

    # Build per-task sample mappings, clamped to [0, n_samples] (README fix: a task
    # window may extend past the recording; intersect before windowing).
    aligned: list[AlignedTask] = []
    for t in usable_tasks:
        s = round((t.start_s - offset_s) * pre.sfreq)
        e = round((t.end_s - offset_s) * pre.sfreq)
        us, ue = max(0, s), min(pre.n_samples, e)
        aligned.append(AlignedTask(t.task_id, s, e, us, ue))

    result = AlignmentResult(
        key=pre.rec.key, method="meas_date_wholehour", offset_s=offset_s,
        correction_s=best_corr, n_fit=best_hits, n_usable=len(usable_tasks),
        fit_frac=fit_frac, reconciled=reconciled,
        continuous_label_valid=continuous_valid, note=note, aligned_tasks=aligned,
    )
    _log(result)
    return result


def _log(r: AlignmentResult) -> None:
    flag = "cont_valid" if r.continuous_label_valid else "CONT_MASKED"
    print(f"[stage5] {r.key}: {r.method} corr={r.correction_s:+d}s "
          f"fit={r.n_fit}/{r.n_usable} ({r.fit_frac:.0%}) "
          f"reconciled={r.reconciled} [{flag}]")
    print(f"[stage5]   {r.note}")


if __name__ == "__main__":
    from .config import load_config
    from .discovery import discover_recordings
    from .preprocess import preprocess_recording
    cfg = load_config()
    for rec in discover_recordings(cfg):
        align_recording(cfg, preprocess_recording(cfg, rec))
