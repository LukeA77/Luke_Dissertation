---
name: livinglab-amplitude-rejection
description: CBraMod 100uV reject threshold destroys LivingLab data; rejection disabled for pilot; sub-134 has bad channels; rejection tuning parked pending supervisor spectral review, ICA now fixed for blinks
metadata:
  type: project
---

Key pilot finding (2026-07-02): CBraMod's **100 µV whole-window peak-abs rejection threshold** (from TUEG pretraining) is inappropriate for the LivingLab data and discards **97–100%** of windows (23 of 1352 survive; sub-134-c → 0). Verified this is genuine, not a scale bug: after the 0.3 Hz high-pass the clean subject (sub-131) sits at ~35 µV median|sample| (textbook EEG); the huge raw medians were DC/drift, correctly removed. The cause is that a 10 s × 18-ch window = 36,000 samples, so whole-window peak-abs almost always catches one transient >100 µV in task-based **mobile, Pz-referenced** recordings (walking, kettle/toast tasks).

**Decision (user):** `reject.enabled: false` for the pilot — retain all windows as a correctness harness; defer artifact handling to the 30-patient cohort phase. This is a config change only.

**sub-134 bad channels:** O1 ~404 µV and Fz ~256 µV median|sample| in sub-134-a (chronic, not transient) — likely a recording-quality issue to raise with Dr Haar. The fixed-18-channel montage (spec) forbids dropping/interpolating them in-pipeline.

**Why:** the mismatch is itself a documented deliverable; the pilot's goal is pipeline correctness, not final artifact rejection. See [[pipeline-scope-decisions]] and [[livinglab-data-issues]].

**Linked-ears re-referencing result (2026-07-03):** re-referencing to linked ears (mean of A1/A2) as the FIRST step — implemented in `reref.ref_channels` config; outputs isolated in `processed/reref-linkedears/`; validate with `reref_report.py` — was tested in isolation (no ICA, threshold unchanged at 100 µV). It **lowered per-channel median peak amplitude ~1.4–2×** (e.g. sub-131-a median 321→158 µV) confirming the shared-Pz-noise mechanism, **but did NOT recover window survival**: 98.3%→**98.4% rejected** (survival 1.7%→1.6%, essentially flat). Whole-window peak-abs over 18ch×2000 samples is still dominated by transients, and sub-134's chronic bad channels (Fz ~2500, O1 ~2800 µV) barely move under reref (drop ≈1.0×) because that noise is local, not the common Pz term. Cross-check: the "before" path exactly reproduces the prior 23/1352 survivors. **Conclusion: re-referencing alone is insufficient to fix rejection; the threshold/rejection strategy (and likely channel-quality screening + artifact handling e.g. ICA) still need work.** Decision deferred pending these numbers.

**ICA + threshold-sweep result (2026-07-03):** ICLabel-guided ICA (mne + mne-icalabel, extended infomax, per recording, fit on the 18 scalp channels only, `n_components=18`=full rank so removing 0 = identity) was bracketed by a threshold sweep before/after — run via `eeg_reref_ica.py` (`clean_with_ica` in `src/livinglab_prep/ica.py`); outputs isolated in `processed/reref-linkedears_ica/`. Config: fit on 1 Hz-highpassed copy resampled to 100 Hz (ICLabel's native rate; apply unmixing back to 300 Hz data), `iclabel_min_prob=0.5`, exclude {muscle, eye blink, heart, line noise, channel noise}. **Off-distribution ICLabel (linked-ears ≠ its common-avg training) barely flags artifacts:** removed **1 eye-blink each from sub-131-a/c, 0 from sub-134** (sub-134's chronic bad channels were NOT labelled artifacts). **Survival gain from ICA is small and mid-range only:** overall window survival before→after ICA — 100 µV 1.6→1.8%, 150 µV 4.1→5.8%, 200 µV 8.2→10.5% (peak effect +2.3 pp), ≥500 µV ≈0. Per-channel median peak: sub-131 dropped a further ~15–21% (321→158→134; 226→136→108), sub-134 unchanged (0 comps removed = identity, confirming no rank confound). Determinism verified (identical hash on re-run, seed=1). **Conclusion: ICA alone does not rescue the ~98% rejection at 100 µV; its benefit is a few pp of survival in the 150–300 µV band from ocular-transient removal. No final threshold chosen — that's a supervisor decision from the sweep.** Key gotchas: this machine has ~1.5–2 GB free RAM, so ICA/ICLabel on ~50-min recordings needs the 100 Hz fit copy + freeing intermediates or it segfaults/OOMs; MNE mis-types the EDF's accelerometers/CM/aux as `eeg`, so the EEG set is taken explicitly from `channels.keep` (never `pick('eeg')`).

**Frontal-correlation blink fix (2026-07-08):** diagnosed that ICLabel's *winning-label*
rule was the actual bug for eye blinks, not ICA's separation -- ICLabel runs
off-distribution here (linked-ears/mobile/PD vs its common-average/seated/non-PD
training) and often labels real blink components "brain"/"other" (sub-134 eye-blink
prob only 0.10-0.25), so the winning-label rule removed 0 components even though
topomaps clearly showed frontal blink components. Fixed in `src/livinglab_prep/ica.py`:
blink components are now identified by frontal-channel correlation
(`ICA.find_bads_eog(measure='correlation', ch_name='Fp1'/'Fp2', threshold=0.5)`),
unioned with the (unchanged) ICLabel winning-label rule for the other artifact classes
(muscle/heart/line-noise/channel-noise). `ica.exclude_labels` in config no longer
contains `eye blink`. Smoke-tested against the real EDFs: sub-134-a/-c now correctly
remove their IC16/IC17 blink components (eog r~0.9), matching the diagnosed values.

**Decision (2026-07-08, user call): amplitude-rejection threshold work is PARKED, not
being pursued further right now.** Not chasing a recalibrated number off 2 patients.
Instead: ship the ICA fix above (rejection stays disabled), then produce a before/after
spectral (PSD) comparison for sub-131/sub-134 and send it to the supervisor -- he
decides if signal quality is good enough to proceed. Revisit rejection only if/when he
asks for it to be turned back on.

**How to apply:** rejection stays `enabled: false`. Do not spend more effort guessing a
threshold or running the cohort sweep until the supervisor's spectral-quality call comes
back. ICA now does real, correct artifact removal (blinks + the other ICLabel classes);
if the supervisor asks about residual noise, the answer is "amplitude rejection is
deliberately deferred," not "ICA doesn't work."
