# PROGRESS — LivingLab EEG preprocessing (re-entry notes)

Working log so you can pick this back up without re-reading everything. Newest
investigation (ICA / eye-blink) is at the top because that's where the live work is.
Written 2026-07-07. Updated 2026-07-08: frontal-correlation blink fix implemented.

---

## START HERE NEXT TIME (the one thing to do)

**The Problem 2 fix is now IMPLEMENTED** (see Problem 2 below for the full
diagnosis). `src/livinglab_prep/ica.py::clean_with_ica` now excludes ICA
components via TWO unioned rules instead of one:
  * non-ocular artifacts (muscle/heart/line-noise/channel-noise) — ICLabel
    winning-label rule, unchanged;
  * eye blinks — frontal-channel correlation (`ICA.find_bads_eog(...,
    measure='correlation', ch_name in cfg.ica.eog_channels, threshold=
    cfg.ica.eog_corr_threshold)`), independent of ICLabel's label.

Config: `ica.exclude_labels` no longer contains `eye blink`; new keys
`ica.eog_channels: [Fp1, Fp2]`, `ica.eog_corr_threshold: 0.5`, `ica.eog_l_freq/
eog_h_freq: 1.0/10.0` in `config/pipeline.yaml`. `ICAReport` now carries
`iclabel_excluded_idx`, `eog_excluded_idx`, `eog_scores` separately (as well as
the unioned `excluded_idx` as before) so provenance is auditable per component.

Smoke-tested directly against the real EDFs (not just the diagnostic scripts):
`clean_with_ica` on sub-134-a now removes IC16/IC17 (eog r=0.94/0.89, iclabel=[])
and sub-131-a removes IC14/15/16 via eog-corr — both match the confirmed table
below exactly. `ica_review_plots.py`'s refit-vs-published determinism check was
updated to also refit and assert the eog exclusion, so it won't false-flag on
the new logic next time it's run.

**Decision (2026-07-08, user call): do NOT chase an amplitude-rejection
threshold number (Problem 1 below is deliberately parked, not solved).**
Rejection stays disabled. The plan instead: run the full pipeline with this ICA
fix, produce spectral-power comparisons (before vs after preprocessing) for
sub-131/sub-134, and send those to the supervisor — he decides whether the
signal quality is good enough to keep going. No further internal
threshold-tuning work until that call comes back.

**Next actions:**
1. Re-run `eeg_reref_ica.py` (and/or the full Stage 0–9 pipeline) with the new
   ICA code so the published `ica_report_<key>.json` / plots reflect the fix.
2. Produce the before/after spectral comparison (PSD plots per recording/
   condition) for the supervisor. Nothing like this exists yet as a script —
   write one, or check whether `eeg_reref_ica.py`'s per-channel amplitude table
   is enough or a proper PSD (Welch) plot is wanted.
3. Send that to the supervisor and wait for the go/no-go call before touching
   amplitude rejection again.

To reproduce the original diagnostic evidence: `python ica_frontal_blink_id.py`
→ `ica_frontal_blink_id.csv` (unchanged, read-only, still valid).

---

## The project in one paragraph

Preprocess mobile Parkinson's EEG (LivingLab: patients doing UPDRS motor + ADL
tasks in two medication states, A=ON / C=OFF) into windowed, labelled tensors to
fine-tune **CBraMod** (an EEG foundation model pretrained on seated clinical TUEG
data). This is a **2-patient pilot** (sub-131, sub-134) to prove pipeline
correctness before scaling to an 18-patient cohort. Pipeline = Stages 0–9:
load → linked-ears reref → channel-select(18) → resample → bandpass → notch →
align tasks → window → label → (reject) → serialize. Stages 10–11 (LOPO folds,
loader) are deferred to the cohort phase.

## Exact signal parameters (from config/pipeline.yaml)

| Step | Value |
|---|---|
| Re-reference (FIRST step) | linked ears: `ch − mean(A1, A2)` — verified bit-exact vs supervisor's `ch − (A1+A2)/2` |
| Channels kept | 18: Fp1 Fp2 F7 F3 Fz F4 F8 T3 C3 Cz C4 T4 T5 P3 P4 T6 O1 O2 (Pz=recorded ref, dropped) |
| Resample | 300 → 200 Hz |
| Bandpass | 0.3–75 Hz, FIR firwin, zero-phase |
| Notch | 50 Hz (UK mains; CBraMod used 60 Hz US) |
| Window | 10 s (2000 samples), train stride 5 s, eval stride 10 s, patch = 200 samples (1 s) |
| Reject | 100 µV whole-window peak-abs — **currently DISABLED** for the pilot |

---

## PROBLEM 1 — Amplitude rejection (parked; NOT being pursued further right now)

CBraMod's inherited **100 µV whole-window peak-abs** rejection discards
**97–100%** of windows (23/1352 survive; sub-134-c → 0). Verified genuine, not a
scale bug (clean subject sits at ~35 µV median|sample| post-filter). Cause: a 10 s
× 18-ch window = 36,000 samples, so on mobile/task data one movement transient
anywhere fails the whole window. The rule was calibrated for seated clinical EEG.

Things tried that do NOT fix it:
- **Linked-ears reref**: lowers baseline amplitude ~1.4–2× but survival stays flat
  (transients, not baseline, trip the threshold).
- **ICA**: recovers only a few % of survival, in the 150–300 µV band only.
- **Patch-level rejection** (reject only bad 1 s patches, keep clean seconds):
  helps ~5× (1.6%→8.4% survival at 100 µV) BUT the collateral-damage histogram
  shows **78% of windows have all 10/10 patches bad** — so the dominant problem is
  that 100 µV is simply too strict for this data, not that one bad second nukes
  nine good ones. Only 2.4% of windows are the "mostly-clean, one bad patch" case.

**Conclusion for supervisor:** the lever is *threshold recalibration for this
recording style* (what does "clean" empirically look like here), not a smarter
rejection granularity. The cohort sweep (`cohort_reject_sweep.py`) is meant to
tell us the right threshold from 18 patients rather than guessing from 2.

**2026-07-08 update: not pursuing this right now.** Decided not to spend more
effort guessing/tuning a threshold number. Instead: ship the ICA blink+artifact
cleaning fix (Problem 2), keep rejection disabled, and hand the supervisor a
before/after spectral comparison — he'll say whether it's good enough to
proceed. Revisit this problem only if/when he asks for rejection to be turned
back on. See `memory/livinglab-amplitude-rejection.md`.

---

## PROBLEM 2 — ICA finds "only 1–2 eye-blink components" (FIXED 2026-07-08)

**The worry:** nobody blinks only twice in a 50-min recording, so something looked
broken.

**What we learned (this is the important correction):**

1. **"1 eye-blink component" ≠ "1 blink".** ICA splits the whole recording into 18
   spatial patterns; one blink *component* holds ALL the blinks in its time-course.
   So 1–2 components is expected — the question is whether that component actually
   contains the hundreds of real blinks.

2. **Blinks are definitely present** (ground truth, ICA-free, peak-counting on
   frontal Fp1/Fp2, 1–10 Hz):
   - sub-131-a: ~1000–1400 blinks / 50 min (~20–27/min — textbook)
   - sub-131-c: ~680–960 / 36 min (~19–27/min — textbook)
   - sub-134-a: ~190–700 / 53 min (~4–13/min — LOW)
   - sub-134-c: ~140–600 / 54 min (~3–11/min — LOW)
   > sub-134's low blink rate may itself be a Parkinsonian sign (reduced
   > spontaneous blinking / hypomimia) — worth mentioning to Dr Haar.

3. **ICA IS capturing them.** sub-131-a's ICLabel "eye blink" component (IC16) has
   ~625–1147 blink deflections in its source — matches ground truth. So the
   pipeline WAS removing ~1000 blinks when it excluded that one component.

4. **The real problem = ICLabel's LABELLING, not ICA's separation.** ICLabel runs
   off-distribution here (trained on common-average, seated, non-PD data; we use
   linked-ears + mobile + PD). For **sub-134 it labels the blink components
   "brain"/"other"** (eye-blink prob only 0.10–0.25), so the winning-label rule
   removes **0** components — even though topomaps clearly show frontal blink
   components (sub-134-a IC16/IC17, sub-134-c IC16/IC17).

5. **Why threshold-sweeping ICLabel can't fix it:** the pipeline only inspects each
   component's *winning* label. If "eye blink" is a component's 2nd-place guess it
   is invisible at any cutoff. Lowering `iclabel_min_prob` from 0.5→0.4 only
   recovers 1 extra component (a 0.42 eye-blink in sub-131-c); ≥0.65 removes
   nothing; sub-134 stays at 0 for the whole 0.3–0.9 range.

**The promising fix — frontal-correlation identification (find_bads_eog method):**
Identify blink components by raw Pearson correlation of each component with the
frontal blink signal (Fp1/Fp2), NOT by ICLabel's label. Standard, transparent,
published method that sidesteps ICLabel's distribution problem. Raw correlations
already computed:
   - sub-134-a: IC16 r=0.94, IC17 r=0.89 (rest <0.17) ← ICLabel MISSED these
   - sub-134-c: IC17 r=0.98, IC16 r=0.93 (rest <0.09) ← ICLabel MISSED these
   - sub-131-c: IC13 r=0.98 (the ICLabel blink) ✓
   - sub-131-a: frontal activity splits across IC14/15/16 (r 0.63–0.83)

**CONFIRMED RESULT** (`measure='correlation'`, |r| ≥ 0.5):

| Recording | ICLabel removes | Frontal-corr flags (\|r\|≥0.5) |
|---|---|---|
| sub-131-a | 1 | 3 (IC15/14/16) |
| sub-131-c | 1 | 2 (IC13/14) |
| **sub-134-a** | **0** | **2 (IC16/17)** ← recovered |
| **sub-134-c** | **0** | **2 (IC16/17)** ← recovered |

Frontal correlation recovers exactly the sub-134 blink components ICLabel missed,
with a clean gap to the rest, and agrees with ICLabel on sub-131. This is the fix.

**Two honest caveats to remember:**
- First attempt used `find_bads_eog(measure='zscore', threshold=3.0)` and flagged
  0 components (the adaptive z-score failed despite obvious raw r). The working
  rule is `measure='correlation'` with an absolute cutoff |r| ≥ ~0.5. Don't use
  the zscore mode here.
- The hand-computed whole-recording Pearson r did NOT match find_bads_eog's score
  (differed by 0.2–0.39). Reason: find_bads_eog builds an internal blink-event
  template and correlates against that, not a naive whole-timecourse correlation.
  Both methods agree on WHICH components (ranking + topomaps), so identification is
  robust, but the two scalar r's are not interchangeable — cite find_bads_eog's.

---

## What to decide / do next (in priority order)

1. **DONE** ~~Confirm and implement the frontal-correlation rule~~ — implemented
   in `src/livinglab_prep/ica.py`, smoke-tested against real EDFs, matches the
   confirmed table above exactly.
2. **Re-run the full pipeline** with the fixed ICA code so published reports/
   plots reflect it, then **produce a before/after spectral (PSD) comparison**
   for sub-131/sub-134 and send it to the supervisor for a go/no-go call.
3. **Amplitude-rejection threshold: parked, not being worked on.** Rejection
   stays disabled; no cohort sweep, no threshold-picking, until the supervisor's
   spectral-quality call comes back (see Problem 1).
4. **sub-134 bad channels** O1 (~404 µV) / Fz (~256 µV) chronic noise — raise with
   Dr Haar; the fixed-18 montage forbids dropping/interpolating them in-pipeline.
5. Only after the supervisor's go-ahead: build deferred Stages 10–11 (LOPO folds
   + loader) for the cohort phase.

## Helper scripts written during this investigation (all read-only, none modify the pipeline)

| Script | What it does |
|---|---|
| `patch_reject_check.py` | whole-window vs patch-level rejection survival + collateral-damage histogram → `patch_reject_check_results.csv` |
| `ica_threshold_sweep.py` | excluded-component count vs `iclabel_min_prob` (0.3–0.9) → `ica_threshold_sweep_*.csv` |
| `ica_blink_diagnosis.py` | full 7-class ICLabel prob matrix + per-component blink counts + frontal ground-truth blink counts → `ica_blink_diagnosis_components.csv` |
| `ica_frontal_blink_id.py` | frontal-correlation blink ID vs ICLabel, side by side → `ica_frontal_blink_id.csv` |
| `cohort_reject_sweep.py` | (pre-existing) amplitude sweep across the 18-patient drive cohort |

Saved ICA topomaps/properties for visual review live in
`processed/reref-linkedears_ica/_reports/ica_plots/` (deterministic, seed=1, so
component indices match the scripts above).

## Key gotchas (don't re-learn these the hard way)

- **Never `pick('eeg')`** — MNE mistypes this EDF's accelerometers/CM/Trigger as
  `eeg`. The 18 EEG channels are taken explicitly from `channels.keep`.
- **Trigger channel** must be excluded from any re-reference / amplitude check (it
  caused a false 9829 µV "mismatch" in one verification — real EEG channels match
  the manual reref to 1e-10 µV).
- **Memory:** this machine has ~1.5–2 GB free; ICA on ~50-min recordings needs the
  100 Hz fit copy + freeing intermediates or it OOMs. Fit copy = 1 Hz highpass,
  100 Hz, extended infomax, n_components=18, seed=1.
- **Reject numbers vs run_summary:** `processed/_reports/run_summary.md` shows
  drop%=0 because rejection is DISABLED; the 97–100% figures come from separate
  sweep experiments, not that run.

See also `memory/` for the durable decisions: `pipeline-scope-decisions`,
`livinglab-amplitude-rejection`, `livinglab-data-issues`, `subject-131-clock-offset`.
