"""
CodeBrain-style EEG preprocessing for the LivingLab EDF recordings.

Steps (per the CodeBrain paper):
  1. Band-pass filter 0.3-75 Hz   (remove low- and high-frequency artifacts)
  2. 50 Hz notch filter           (remove power-line interference)
  3. Resample to 200 Hz           (native rate is 300 Hz)
  4. Segment into consecutive, non-overlapping 10 s windows (= 2000 samples)

Applied to the 18 standard 10-20 scalp channels (Pz is the reference, so it is
not a usable signal; A1/A2 ear refs, CM, X-aux sensors, Trigger/Event,
accelerometer and annotation channels are all dropped).

Output: one .npz per recording in ./preprocessed/, containing
  - windows: float32 array, shape (n_windows, n_channels, 2000)
  - ch_names: the ordered channel labels
  - sfreq: 200
No task labels / CSV alignment are used -- this is pure preprocessing.
"""

import glob
import os
import numpy as np
import mne

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LivingLabData")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocessed")

L_FREQ = 0.3      # band-pass low cutoff (Hz)
H_FREQ = 75.0     # band-pass high cutoff (Hz)
NOTCH = 50.0      # power-line notch (Hz)
TARGET_SFREQ = 200.0
WINDOW_SEC = 10.0
WINDOW_SAMPLES = int(WINDOW_SEC * TARGET_SFREQ)   # 2000

# The 18 usable 10-20 scalp electrodes (electrode name only; EDF labels are
# stored as "EEG <name>-Pz").
SCALP = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3",
         "Cz", "C4", "T4", "T5", "P3", "P4", "T6", "O1", "O2"]


def electrode_name(label):
    """Extract the electrode name from an EDF channel label.

    e.g. 'EEG Fp1-Pz' -> 'Fp1', 'CM' -> 'CM'.
    """
    name = label
    if name.upper().startswith("EEG "):
        name = name[4:]
    if "-" in name:
        name = name.split("-")[0]
    return name.strip()


def pick_scalp(raw):
    """Return the EDF channel labels (in SCALP order) for the scalp electrodes."""
    lookup = {electrode_name(ch): ch for ch in raw.ch_names}
    picks, missing = [], []
    for e in SCALP:
        if e in lookup:
            picks.append(lookup[e])
        else:
            missing.append(e)
    return picks, missing


def preprocess_file(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")

    picks, missing = pick_scalp(raw)
    if missing:
        print(f"    WARNING: missing scalp channels {missing}")
    raw.pick(picks)

    # 1. band-pass 0.3-75 Hz  (FIR, zero-phase)
    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, picks="eeg",
               method="fir", phase="zero", verbose="ERROR")

    # 2. 50 Hz notch
    raw.notch_filter(freqs=[NOTCH], picks="eeg", verbose="ERROR")

    # 3. resample 300 -> 200 Hz
    raw.resample(TARGET_SFREQ, verbose="ERROR")

    # 4. segment into consecutive non-overlapping 10 s windows
    data = raw.get_data()                       # (n_channels, n_times), volts
    data = data * 1e6                            # back to microvolts
    n_ch, n_times = data.shape
    n_windows = n_times // WINDOW_SAMPLES
    trimmed = data[:, : n_windows * WINDOW_SAMPLES]
    # -> (n_windows, n_channels, WINDOW_SAMPLES)
    windows = trimmed.reshape(n_ch, n_windows, WINDOW_SAMPLES).transpose(1, 0, 2)
    windows = np.ascontiguousarray(windows, dtype=np.float32)

    return windows, raw.ch_names, n_times


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    edf_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_raw.edf")))
    if not edf_files:
        raise SystemExit(f"No EDF files found in {DATA_DIR}")

    print(f"Output -> {OUT_DIR}")
    print(f"{'file':<20}{'windows':>9}{'channels':>10}{'samples/win':>13}")
    print("-" * 52)

    total = 0
    for edf in edf_files:
        name = os.path.basename(edf).replace("_raw.edf", "")
        windows, ch_names, _ = preprocess_file(edf)
        out = os.path.join(OUT_DIR, f"{name}_preprocessed.npz")
        np.savez_compressed(out, windows=windows,
                            ch_names=np.array(ch_names),
                            sfreq=TARGET_SFREQ)
        total += windows.shape[0]
        print(f"{name:<20}{windows.shape[0]:>9}{windows.shape[1]:>10}"
              f"{windows.shape[2]:>13}")

    print("-" * 52)
    print(f"{'TOTAL':<20}{total:>9} windows of {WINDOW_SEC:.0f}s "
          f"@ {TARGET_SFREQ:.0f} Hz")


if __name__ == "__main__":
    main()
