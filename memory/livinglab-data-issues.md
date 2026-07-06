---
name: livinglab-data-issues
description: Known data-quality defects in the 2 pilot LivingLab CSVs that windowing must handle
metadata:
  type: project
---

Data-quality defects found in the pilot task-timestamp CSVs (verified 2026-07-02). Windowing (Stage 6) and label prep must handle these without crashing:

- **Corrupt timestamp:** `sub-134-c` **UPDRS** row has End−Start ≈ 1.76×10⁹ s (malformed epoch, Start near zero). Must be detected (duration > recording length ⇒ reject that segment) and skipped loudly, not converted to a sample index.
- **Zero-duration tasks:** "Suspine Measure" and "Standing Still" in **both `-c` sessions** have Start == End (0.0 s). Yield zero windows; skip with a log line. ~4/40 segments.
- **Tasks outside the recording:** `sub-134-a`'s last task runs past EEG end; `sub-131-a` has ~2 pre-recording tasks. Stage 6 must **intersect each task segment with [0, n_samples]** and skip/trim — the spec text omits this clamp, so it was added.

**Why:** these are logging artifacts, not spec bugs; the spec's "fail loudly" ethos means skip-with-reason for bad rows, hard-raise only for truly unexpected states.

**How to apply:** treat a task segment as usable iff `0 < duration ≤ recording_len` and it overlaps `[0, n_samples]`; record dropped segments in the run report. Alignment offsets per file are in [[subject-131-clock-offset]].
