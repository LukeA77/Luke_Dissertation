---
name: pipeline-scope-decisions
description: Settled build decisions for the LivingLab CodeBrain preprocessing pipeline (window len, ref, deferred stages)
metadata:
  type: project
---

Settled decisions for the LivingLab preprocessing build (README.md spec, confirmed by user 2026-07-02):

- **Window length = 10 s** (confirmed after Stage-0 duration check; shortest *real* task ~12 s "Sit to Stand", so 10 s is safe; do not drop to 5).
- **`continuous_reference` = session_start** (not task_start).
- **Stages 10–11 (LOPO fold generator + Dataset loader) are DEFERRED** to the 30-patient phase. This build implements **Stages 0–9 only**: config → preflight → channel-select/filter/resample/µV/PSD → alignment → task-locked windowing → dual labels → rejection → serialization + reports. 2-patient LOPO is degenerate (no distinct val patient), so folds/loader wait until the full cohort.
- **sub-131 continuous-mask fallback accepted** as a safety net, but per [[subject-131-clock-offset]] it will not trigger (clean constant −3600 s offset resolves it).

**Why:** these are §14 human-decision items from the spec; recorded so the build doesn't re-ask.

**How to apply:** build only Stages 0–9; validation suite (Stage 12) runs only the window/manifest-level checks (boundary integrity, shape uniformity, µV scale, continuous-label integrity, determinism), not the fold/leakage checks which belong to the deferred phase.
