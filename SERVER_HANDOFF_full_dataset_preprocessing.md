# Handoff: apply the LivingLab EEG preprocessing to the FULL cohort

**Audience:** the Claude instance running on the server that holds the full LivingLab_PD
dataset.
**Author:** Claude on the pilot machine, at Luke's request.
**Date written:** 2026-07-13.

Read this whole file before touching anything. It tells you *what* to do, *how* to do
it, the good practices to follow, exactly *how to check* you did it right, and *what
outputs to produce* so Luke can prove correctness to his supervisor (Dr Haar).

If anything here contradicts what you find in the repo, **stop and ask Luke** rather than
guessing. Several details below are safety-critical (which patients to include, not doing
amplitude rejection, not writing to raw data).

---

## 0. Mission in one paragraph

There is a validated 2-patient pilot preprocessing pipeline for mobile Parkinson's EEG
(LivingLab). It turns raw EDF recordings into windowed, labelled tensors to fine-tune
**CBraMod** (an EEG foundation model). The pilot proved the method is correct — including
a **fixed ICA artifact/blink-removal step**. Your job is to **apply that exact
preprocessing to the entire cohort (the 18 patients listed below, both medication
states = up to 36 recordings)** and produce (a) the serialized training tensors and (b)
evidence that proves it ran correctly. **Do NOT perform amplitude rejection** — it has
been removed from the pipeline by an explicit decision (details in §5).

---

## 1. Background: the pipeline and what "the preprocessing" is

The project is a Python package `src/livinglab_prep/` driven by a single config file
`config/pipeline.yaml` (the single source of truth — no signal parameter is hard-coded
anywhere else). The per-recording signal chain that the pilot validated is:

```
load raw EDF (all channels, 300 Hz)
  -> re-reference to linked ears (ch - mean(A1, A2))         [FIRST transformation]
  -> band-pass 0.3-75 Hz (zero-phase FIR firwin) + notch 50 Hz
  -> ICA artifact + blink removal  (the fixed step - see §5.2)
  -> select the 18 canonical scalp channels
  -> resample 300 -> 200 Hz
  -> task-locked windowing (10 s windows, 5 s train stride)
  -> label each window (condition + continuous elapsed-time target)
  -> serialize to per-window .pkl + one manifest.parquet
```

The exact numbers are in §5. **Every one of these values must come from
`config/pipeline.yaml`, not from literals you write.**

---

## 2. THE CRITICAL GAP YOU MUST CLOSE (read carefully)

The production pipeline and the validated ICA step are, right now, **two separate code
paths**. This is the single most important thing to understand:

- **`python -m src.livinglab_prep.cli run`** runs the production Stages 1-9
  (`src/livinglab_prep/pipeline.py`) and **produces the serialized tensors + manifest —
  but it does NOT run ICA.** It currently does reref -> select -> resample -> filter ->
  window -> serialize.
- **The fixed ICA lives in `src/livinglab_prep/ica.py::clean_with_ica`** and is only
  invoked by the *diagnostic* scripts `eeg_reref_ica.py` and `spectral_before_after.py`.
  Those scripts prove ICA works but **never serialize windows.**

So no ICA-cleaned training tensors have ever been produced. **Your integration task is to
wire `clean_with_ica` into the serialization pipeline at the correct position, then run
the whole cohort through it.** §6 specifies exactly how.

There is also a **data-layout gap**: the production discovery (`discovery.py`,
`discover_recordings`) expects the *flat* pilot folder. The full cohort on the shared
drive uses a *nested* layout, already handled by `src/livinglab_prep/cohort_discovery.py`.
You must make discovery return the correct recordings for the real layout (see §4).

You may modify the code to close these two gaps. Keep changes minimal, config-driven, and
consistent with the existing style. Commit them on a branch (see §8).

---

## 3. Environment setup

1. Get the repository onto the server (Luke will tell you how / where it already is).
2. Create a clean Python environment and install pinned deps:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
   The pins that matter for reproducibility (from `requirements.txt`):
   `mne==1.12.1`, `mne-icalabel==0.9.0`, `numpy==2.5.0`, `scipy==1.18.0`,
   `matplotlib==3.11.0`, `pyyaml==6.0.3`, `pydantic==2.13.3`.
   **Use these exact versions** — ICA/ICLabel results are version-sensitive and §9 asks you
   to reproduce known-good component numbers.
3. Sanity-check the config loads:
   ```bash
   python -m src.livinglab_prep.cli config
   ```
   It prints `config_hash=...` and `git_sha=...`. Record both; they stamp every output.
4. The server has generous RAM (confirmed), so you may process recordings in one pass.
   Still free large objects between recordings (the existing scripts already `gc.collect()`);
   a 50-minute recording's ICA fit is the memory peak.

---

## 4. Find the data — and DO NOT sweep in the wrong patients

**This is safety-critical.** The shared drive also hosts a *different* sub-study on
DSI-only patients that must **never** be included, plus some patients with incomplete
ON/OFF pairs. Use only this explicit allowlist (the 18-patient cohort from the 2026-07-06
audit; it is already encoded in `cohort_reject_sweep.py::COHORT_PATIENTS`):

```
sub-34, sub-42, sub-48, sub-51, sub-53,
sub-111, sub-114, sub-115, sub-116, sub-117, sub-119, sub-121,
sub-123, sub-125, sub-129, sub-131, sub-134, sub-135
```

Explicitly **excluded** and must stay excluded: incomplete-pair patients `sub-70`,
`sub-71`, `sub-130`, and **all** DSI-only patients. Do not `rglob` the whole drive blindly.

**Layout.** Luke has the exact data location. Confirm the on-disk layout first (read-only)
and pick the matching discovery:

- **Nested drive layout** — `sub-<ID>-<cond>/eeg/<livinglab|living-lab-tasks>/` each
  holding `<key>_raw.edf` + `<key>_raw.csv`. Use
  `src/livinglab_prep/cohort_discovery.py::discover_cohort_recordings(root, COHORT_PATIENTS)`.
  It returns `(recordings, missing)` — **print `missing` and confirm with Luke** before
  processing (a missing pair can mean DSI-only or a genuine gap, never silently skip it).
- **Flat layout** — all `<key>_raw.edf` + `<key>-<cond>_<date>.csv` in one dir. Use the
  existing `discover_recordings` and point `paths.raw_dir` / `paths.csv_dir` at it.

Whichever you use, the rest of the pipeline is layout-agnostic once it has a list of
`Recording` objects. The cleanest integration is to make discovery layout-aware (or add a
small config/CLI switch) so `preflight`, `run`, and `validate` all see the same recordings.

**Known data defects the run must tolerate (not crash on)** — the pilot found these and
windowing already handles them; verify they're still handled and reported, not fatal:
- corrupt task timestamps (duration > recording length) → skip that segment loudly;
- zero-duration tasks (Start == End) → yield no windows, log and continue;
- tasks partly/fully outside the recording → intersect each task with `[0, n_samples]`,
  trim/skip.
Wrap per-recording processing so **one bad recording is logged and skipped, not allowed to
abort the other 35** (see `cohort_reject_sweep.py` for the try/except pattern to mirror).

---

## 5. The exact preprocessing spec (all values live in `config/pipeline.yaml`)

### 5.1 Signal parameters (confirm these are what the config holds; do not invent)

| Step | Value |
|---|---|
| Re-reference (FIRST step) | linked ears: `ch − mean(A1, A2)`, applied while A1/A2 still present |
| Channels kept | 18: `Fp1 Fp2 F7 F3 Fz F4 F8 T3 C3 Cz C4 T4 T5 P3 P4 T6 O1 O2` (Pz = recorded ref, dropped) |
| Resample | 300 → 200 Hz |
| Band-pass | 0.3–75 Hz, FIR `firwin`, zero-phase |
| Notch | 50 Hz (UK mains) |
| Window | 10 s (2000 samples), train stride 5 s, eval stride 10 s, patch = 200 samples (1 s) |
| Continuous label | `elapsed_time_s` from `session_start`, no normalization |
| Condition label | A → 0 (ON/peak), C → 1 (OFF/wearing-off) |

### 5.2 ICA (the fixed step — this is the whole point of the exercise)

Fitted **per recording** on the 18 scalp channels only. Fit on a high-passed (1.0 Hz),
100 Hz-resampled copy using **extended infomax**, `n_components=18`, **`random_state=1`**
(fixed seed → reproducible components); apply the unmixing to the 0.3–75 Hz data. Two
**unioned** exclusion rules:

1. **Non-ocular artifacts** — remove a component iff ICLabel's *winning* label is in
   `muscle artifact | heart beat | line noise | channel noise` with winning
   probability ≥ 0.5.
2. **Eye blinks** — identified NOT by ICLabel (it runs off-distribution on this
   linked-ears / mobile / Parkinsonian data and mislabels blink components) but by
   **frontal-channel correlation**: `ICA.find_bads_eog(measure='correlation')` against
   `Fp1`/`Fp2`, flag any component with `|r| ≥ 0.5` (blink band 1–10 Hz).

`brain` and `other` are always kept. This is all already implemented in
`ica.py::clean_with_ica`; **use it as-is, do not reimplement it.** `config.ica` has all the
knobs. Do not add `eye blink` to `ica.exclude_labels` (the config validator forbids it).

### 5.3 Amplitude rejection — **DO NOT DO IT**

Amplitude rejection is **disabled by explicit decision** and must stay disabled. In
`config/pipeline.yaml`, `reject.enabled` must be `false`. The CBraMod 100 µV whole-window
peak-abs threshold discarded 97–100 % of windows on this mobile/task data and is not
appropriate here. **All windows are retained.** Do not enable rejection, do not run any
threshold sweep as part of the deliverable, do not "helpfully" filter windows by
amplitude. If you think rejection is needed, that is a question for Luke/Dr Haar, not a
thing to turn on.

---

## 6. Integration task: put ICA into the serialization pipeline

Goal: the serialized tensors must be produced from the **ICA-cleaned** signal, following
the validated order in §1. The reference implementation for the correct chain already
exists in `eeg_reref_ica.py` (it does reref → filter → pick-18 → `clean_with_ica` →
resample → window). You are extending that chain to also **label + serialize**, which
`pipeline.py` already knows how to do.

**Required per-recording chain (this exact order):**

```
load_raw
  -> apply_reference (linked ears)                       # A1/A2 present
  -> band-pass 0.3-75 + notch 50   (at 300 Hz, picks='eeg')
  -> clean_with_ica(cfg, raw_filt, key, report_dir=...)  # returns 18-ch cleaned Raw @300 Hz
  -> resample 300 -> 200
  -> extract_uV  -> PreprocessedRecording
  -> align_recording -> make_windows -> label_windows
  -> reject_windows (NO-OP: reject.enabled=false)
  -> serialize
```

Notes and requirements:

- **Reuse existing functions.** `clean_with_ica` returns an 18-channel, canonically
  ordered, cleaned `Raw` at 300 Hz. Feed that into a resample→`extract_uV` step to build a
  `PreprocessedRecording`, then hand it to the existing `align_recording` / `make_windows`
  / `label_windows` / `Serializer`. `eeg_reref_ica.py::_to_preprocessed` shows the
  resample+extract packaging; `pipeline.py` shows the label+serialize half. You are
  essentially stitching those two together.
- **Do not** apply the band-pass *after* resampling for the ICA build — the validated
  pilot filtered at 300 Hz *before* ICA and resampled *after*. Match that so the serialized
  data equals what the spectral evidence was computed on.
- **Preserve `meas_date`** through reref/filter/ICA (Stage 5 alignment needs it). ICA on an
  EEG-only Raw can drop the measurement date; carry it forward explicitly if so.
- **Output location — keep it unambiguous.** Route the ICA-cleaned serialized outputs to an
  ICA-scoped directory so they can never be confused with the older non-ICA reref-only
  outputs. `config.py` already exposes `cfg.ica_run_dir` = `processed/reref-<scheme>_ica/`
  as a sibling of `cfg.run_dir`. Make the `Serializer`, manifest, and reports for this run
  write under that ICA-scoped root (e.g. `processed/reref-linkedears_ica/windows/`,
  `.../manifest.parquet`, `.../_reports/`). Do not overwrite `processed/reref-linkedears/`.
- **Stamp provenance in every window's `meta`.** The serializer already writes
  `config_hash`, `git_sha`, `sfreq`, `units`. Also record that ICA cleaning was applied and
  which components were removed (e.g. add `ica_excluded_idx` to `meta`, sourced from the
  recording's `ICAReport`), so any window is traceable back to its cleaning.
- Write one `ica_report_<key>.json` per recording (as `eeg_reref_ica.py` already does) into
  the run's `_reports/` so the removals are auditable.

Keep the diff small and in-style. If closing the gap cleanly needs a config flag (e.g. to
select nested vs flat discovery, or to point the run at `ica_run_dir`), add it to
`pipeline.yaml` with a validating field in `config.py` — never a bare literal.

---

## 7. How to run

Run the three stages in order and keep every console log (redirect to a file):

```bash
# 0. Pre-flight: task-duration safety gate + task_durations.csv
python -m src.livinglab_prep.cli preflight  2>&1 | tee logs/preflight.log

# 1. The run: Stages 1-9 with ICA, all recordings -> tensors + manifest + reports
python -m src.livinglab_prep.cli run        2>&1 | tee logs/run.log

# 2. Validate: Stage 12 mechanical checks on the outputs
python -m src.livinglab_prep.cli validate   2>&1 | tee logs/validate.log
```

(`... cli all` chains all three, halting on the first failure — fine to use once each stage
works individually.) Expect this to take a while: ICA is fit per recording on ~50-minute
signals × up to 36 recordings.

---

## 8. Good practices / guardrails

- **Raw data is READ-ONLY.** Never write into the raw EDF/CSV directories. All outputs go
  under `processed/` (the ICA-scoped root). The EDFs are the irreplaceable source.
- **Work on a branch, commit the integration.** e.g. `git checkout -b cohort-ica-run`.
  Commit the code changes and the config with a clear message so the run is reproducible
  from a known SHA. Do not force-push or touch `master` without Luke's say-so.
- **Determinism is a requirement, not a nicety.** Seeds are fixed (`run.seed=1`,
  `ica.random_state=1`). Identical config + data must give an identical manifest. The
  pipeline writes a `manifest_hash.txt`; a re-run must reproduce it (see §9).
- **Fail loudly on the unexpected, skip-with-reason on known data defects.** Genuinely
  corrupt/zero-duration/out-of-range task rows are logged and skipped (§4). Anything else
  unexpected should raise, not be silently swallowed.
- **Don't trust MNE channel types.** This EDF mistypes accelerometers/CM/Trigger as `eeg`.
  The 18 channels are taken explicitly from `channels.keep` and each is verified — keep it
  that way; never `pick('eeg')`.
- **Never re-reference or amplitude-check the Trigger/aux channels** (it produced a false
  ~9829 µV "mismatch" once; real EEG channels match the manual reref to ~1e-10 µV).
- **One config, no literals.** If you need a new parameter, add it to `pipeline.yaml` +
  `config.py`. Record `config_hash` and `git_sha` with the results.
- **Preserve the pilot outputs.** Don't delete or overwrite `processed/reref-linkedears/`
  or the pilot's evidence PNGs/CSVs — the ICA run writes to its own `_ica` root.

---

## 9. How to CHECK you did it correctly

Do all of these and capture the output. The first is the strongest single proof.

### 9.1 Cross-check against the pilot's known-good ICA results (the correctness anchor)

`sub-131` and `sub-134` were fully analysed in the pilot; the *correct* ICA component
removals are known. Because the seed and versions are fixed, **your run MUST reproduce these
exactly.** If it doesn't, something is wrong (wrong data, wrong versions, wrong step order,
seed not applied) — stop and diagnose before trusting the other 16 patients.

| Recording | ICLabel removes | Frontal-corr (blink) flags (\|r\|≥0.5) | Notes |
|---|---|---|---|
| sub-131-a | 1 | 3 → IC14/IC15/IC16 | blink energy split across three comps |
| sub-131-c | 1 | 2 → IC13/IC14 | |
| **sub-134-a** | **0** | **2 → IC16 (r≈0.94) / IC17 (r≈0.89)** | ICLabel MISSED these; frontal-corr recovers them |
| **sub-134-c** | **0** | **2 → IC16 / IC17** | ICLabel MISSED these; recovered |

Read these back from your `ica_report_<key>.json` files (`iclabel_excluded_idx`,
`eog_excluded_idx`, `eog_scores`). The sub-134 recovery is the headline result of the fix —
confirm it reproduces.

### 9.2 Validation suite (Stage 12) must pass

`python -m src.livinglab_prep.cli validate` checks, and must PASS:
- **shape uniformity** — every window is `(18, 2000)` and reshapes to `(18, 10, 200)`;
- **scale is µV** — median `max|X|` > 0.1 (rules out a Volts or pre-/100 scale bug);
- **window width == samples_per_window**, **train-stride grid** consistent per task,
  **non-overlap subset flag** consistent;
- **both `y_cond` classes present** overall (0 and 1);
- **continuous-label integrity** — `continuous_label_valid=false` windows are *masked and
  kept*, never dropped;
- **determinism** — manifest content hash; re-run and confirm it matches.

### 9.3 Manifest sanity (spot-check yourself, beyond the automated suite)

Load `processed/reref-linkedears_ica/manifest.parquet` and confirm:
- row count = total windows; per-patient and per-condition counts are non-zero and roughly
  plausible (both A and C present for most patients);
- no NaNs in required columns; `sfreq` all 200; `t_cont` finite and ≥ 0;
- number of distinct `patient_id` == number of patients that survived discovery (≤ 18), and
  none of them is an excluded/DSI patient.

### 9.4 Determinism re-run

Re-run the pipeline on a 2-recording subset (e.g. `sub-131`) into a scratch output dir and
confirm the manifest hash and the `ica_report` component lists are **identical** to the
first run. Fixed seeds mean identical → identical.

### 9.5 Spectral before/after (also a supervisor deliverable — see §10)

Run `python spectral_before_after.py` for at least sub-131/sub-134 (ideally a spread across
the cohort). In the frontal channels (Fp1/Fp2) the post-ICA PSD should show **reduced
low-frequency (blink-band) power** versus pre-ICA, while posterior channels are largely
unchanged. That visual is both a correctness check and the evidence Dr Haar asked for.

---

## 10. Outputs to produce for the supervisor (proof of correct work)

Collect these into one place (e.g. `processed/reref-linkedears_ica/_reports/` +
a top-level `COHORT_RUN_REPORT.md` you write). Luke will show them to Dr Haar.

1. **The serialized dataset** — `processed/reref-linkedears_ica/windows/**.pkl` and
   `manifest.parquet` (the actual CBraMod training tensors). Report total window count and
   per-patient / per-condition breakdown.
2. **Per-recording ICA reports** — `ica_report_<key>.json` for every recording, plus a
   summary table: recording, #components removed (ICLabel vs frontal-corr), which ICs, the
   `eog_scores`. Highlight that sub-134 blink components are recovered (§9.1).
3. **Spectral before/after PNGs** — `spectral_before_after_<key>.png` (raw / pre-ICA /
   post-ICA, per-channel, 10-20 layout). At minimum sub-131 + sub-134; ideally the cohort.
4. **`run_summary.md`** — per-recording window counts, alignment method/offset/drift,
   `continuous_label_valid`, PSD OK/WARN (auto-written by `report.py`).
5. **`limitations.md`** — auto-written caveats (reference montage, 50 vs 60 Hz notch,
   amplitude rejection disabled + why, alignment anchor).
6. **`validate` output** — the full PASS list from §9.2 (paste `logs/validate.log`).
7. **Determinism evidence** — the `manifest_hash.txt` and the matching re-run hash (§9.4).
8. **A short `COHORT_RUN_REPORT.md` you write**, ~1 page, stating: config_hash + git_sha,
   how many of the 18 patients / 36 recordings were processed (and any missing/skipped, with
   reasons), the total windows produced, that ICA reproduced the pilot's known-good results
   for sub-131/sub-134, that all Stage-12 checks passed, and that amplitude rejection was
   (correctly) not applied.

---

## 11. Final checklist before you tell Luke it's done

- [ ] Used ONLY the 18-patient allowlist; no DSI-only or incomplete-pair patients included.
- [ ] Raw data untouched; all outputs under `processed/reref-linkedears_ica/`.
- [ ] ICA is actually in the serialization chain (not just the diagnostic scripts), in the
      correct order (filter → ICA → resample), reusing `clean_with_ica`.
- [ ] `reject.enabled=false` — no amplitude rejection performed anywhere.
- [ ] sub-131 / sub-134 ICA removals reproduce the §9.1 table exactly.
- [ ] `preflight`, `run`, `validate` all completed; validate PASSED every check.
- [ ] Manifest sane (counts, no NaNs, sfreq 200, shapes (18, 2000)).
- [ ] Determinism re-run reproduced the manifest hash.
- [ ] Deliverables in §10 gathered, including the `COHORT_RUN_REPORT.md`.
- [ ] Code changes committed on a branch with config_hash + git_sha recorded.

If any box can't be ticked, report exactly which and why — do not paper over it.

---

### Appendix: where things live

| Thing | File |
|---|---|
| Config (single source of truth) | `config/pipeline.yaml`, validated by `src/livinglab_prep/config.py` |
| Production pipeline (Stages 1-9) | `src/livinglab_prep/pipeline.py`, CLI `src/livinglab_prep/cli.py` |
| The fixed ICA step | `src/livinglab_prep/ica.py::clean_with_ica` |
| Reference impl. of the correct reref→filter→ICA→resample→window chain | `eeg_reref_ica.py` |
| Flat (pilot) discovery | `src/livinglab_prep/discovery.py` |
| Nested (cohort drive) discovery + the 18-patient allowlist | `src/livinglab_prep/cohort_discovery.py`, `cohort_reject_sweep.py` |
| Spectral before/after evidence | `spectral_before_after.py` |
| Validation suite (Stage 12) | `src/livinglab_prep/validate.py` |
| Reports (run_summary / limitations) | `src/livinglab_prep/report.py` |
| Full re-entry notes / rationale | `PROGRESS.md`, `memory/` |
