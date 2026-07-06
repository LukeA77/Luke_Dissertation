---
name: subject-131-clock-offset
description: LivingLab EDF device clock for subject 131 is ~1h off from CSV task logs (BST bug)
metadata:
  type: project
---

In the LivingLab dataset, the EEG device clock does not always match the task-log CSV timestamps (which are true Unix epoch microseconds), so aligning task labels to the EEG requires a per-file time shift.

- **sub-131 (recorded Aug 2025, BST):** device/EDF start time is ~1 hour ahead of real time. A constant shift of ≈ −3600 s aligns the CSV tasks to the recording (sub-131-a: 8/10 tasks fit, the 2 misses are pre-recording setup; sub-131-c: 8/8 fit).
- **sub-134 (recorded Nov 2025, GMT):** aligns with only a small shift (~−56 s for -c → 7/7; -a → 7/10, last task runs past recording end). No DST issue.
- There are **no** trigger/event/annotation markers in any EDF, so wall-clock is the only alignment anchor.

**Why:** UK daylight-saving — August recordings are BST (UTC+1) but the clock was effectively treated as UTC, producing the 1h offset; November is GMT so no gap.

**How to apply:** This is irrelevant to the CodeBrain preprocessing (filter/notch/resample/10s windows on continuous signal — see [[preprocess.py]]). It only matters if/when task labels are attached to windows: apply the per-file shift before mapping CSV Start/End into the recording. See preprocess.py header note and the timing analysis.
