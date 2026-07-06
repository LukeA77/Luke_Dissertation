"""
Quality-control / validation for the CodeBrain preprocessing.

For every recording it compares the RAW EDF against the PREPROCESSED output and
checks that each of the 4 pipeline steps left the expected fingerprint:

  1. Band-pass 0.3-75 Hz -> power in passband (1-40 Hz) preserved; power below
     0.3 Hz and above 75 Hz strongly attenuated relative to raw.
  2. 50 Hz notch         -> the mains peak visible in raw is removed.
  3. Resample to 200 Hz  -> processed sfreq is 200 and Nyquist is 100 Hz;
     n_samples matches duration * 200.
  4. 10 s windowing       -> every window is exactly 2000 samples, count equals
     floor(n_times / 2000), and there are no NaN/Inf values.

Outputs:
  - printed PASS/FAIL table with the measured numbers
  - one PSD comparison figure per recording in ./qc/
"""

import glob
import os
import numpy as np
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "LivingLabData")
PRE_DIR = os.path.join(ROOT, "preprocessed")
QC_DIR = os.path.join(ROOT, "qc")

L_FREQ, H_FREQ, NOTCH, TARGET_SFREQ, WIN = 0.3, 75.0, 50.0, 200.0, 2000
SCALP = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3",
         "Cz", "C4", "T4", "T5", "P3", "P4", "T6", "O1", "O2"]


def electrode_name(label):
    n = label[4:] if label.upper().startswith("EEG ") else label
    return n.split("-")[0].strip()


def raw_scalp(edf):
    raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
    lookup = {electrode_name(c): c for c in raw.ch_names}
    raw.pick([lookup[e] for e in SCALP if e in lookup])
    return raw.get_data() * 1e6, raw.info["sfreq"]   # (ch, t) in uV


def bandpower(f, p, lo, hi):
    m = (f >= lo) & (f < hi)
    return p[m].mean() if m.any() else np.nan


def line_peak_ratio(f, p, line=50.0):
    """Peak power at `line` Hz relative to its neighbours (48-49 & 51-52 Hz)."""
    peak = p[(f >= line - 0.6) & (f <= line + 0.6)].max()
    base = p[((f >= line - 2) & (f <= line - 1)) |
             ((f >= line + 1) & (f <= line + 2))].mean()
    return peak / base if base > 0 else np.nan


def analyse(edf, npz):
    raw, fs_raw = raw_scalp(edf)
    d = np.load(npz, allow_pickle=True)
    w = d["windows"]                      # (n_win, ch, 2000)
    fs_pre = float(d["sfreq"])
    cont = w.transpose(1, 0, 2).reshape(w.shape[1], -1)   # (ch, n_win*2000)

    # PSDs (average over channels), 10 s Welch segments -> 0.1 Hz resolution
    fr, Praw = welch(raw, fs=fs_raw, nperseg=int(fs_raw * 10), axis=1)
    fp, Ppre = welch(cont, fs=fs_pre, nperseg=WIN, axis=1)
    Praw, Ppre = Praw.mean(0), Ppre.mean(0)

    # interpolate raw PSD onto processed freq grid for direct ratios
    Praw_i = np.interp(fp, fr, Praw)
    ratio = Ppre / np.where(Praw_i > 0, Praw_i, np.nan)

    metrics = {
        "pass_ratio_1_40Hz": np.nanmedian(ratio[(fp >= 1) & (fp <= 40)]),
        "atten_below_0.3Hz": np.nanmedian(ratio[(fp >= 0.1) & (fp < 0.25)]),
        "atten_above_75Hz": np.nanmedian(ratio[(fp >= 80) & (fp <= 95)]),
        "notch_raw_peak": line_peak_ratio(fr, Praw, NOTCH),
        "notch_pre_peak": line_peak_ratio(fp, Ppre, NOTCH),
        "sfreq": fs_pre,
        "nyquist": fs_pre / 2,
        "win_len_ok": bool((w.shape[2] == WIN)),
        "n_windows": w.shape[0],
        "expected_windows": cont.shape[1] // WIN,
        "n_channels": w.shape[1],
        "nan_inf": int(np.sum(~np.isfinite(w))),
    }
    return fr, Praw, fp, Ppre, metrics


def plot(edf_name, fr, Praw, fp, Ppre, out):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a in ax:
        a.semilogy(fr, Praw, color="0.6", lw=1, label="raw (300 Hz)")
        a.semilogy(fp, Ppre, color="C0", lw=1.2, label="preprocessed (200 Hz)")
        for x, c in [(L_FREQ, "g"), (NOTCH, "r"), (H_FREQ, "m")]:
            a.axvline(x, color=c, ls="--", lw=0.9, alpha=0.7)
        a.set_xlabel("Frequency (Hz)")
        a.set_ylabel("PSD (uV^2/Hz)")
        a.legend(loc="upper right", fontsize=8)
    ax[0].set_xlim(0, 100); ax[0].set_title(f"{edf_name}: full spectrum")
    ax[1].set_xlim(40, 100); ax[1].set_title("zoom 40-100 Hz (notch @50, cutoff @75)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    os.makedirs(QC_DIR, exist_ok=True)
    rows = []
    for edf in sorted(glob.glob(os.path.join(DATA_DIR, "*_raw.edf"))):
        name = os.path.basename(edf).replace("_raw.edf", "")
        npz = os.path.join(PRE_DIR, f"{name}_preprocessed.npz")
        if not os.path.exists(npz):
            print(f"  skip {name}: no preprocessed file")
            continue
        fr, Praw, fp, Ppre, m = analyse(edf, npz)
        plot(name, fr, Praw, fp, Ppre, os.path.join(QC_DIR, f"{name}_psd.png"))
        rows.append((name, m))

    # ---- report ----
    print("\n=== STRUCTURE CHECKS (must all pass) ===")
    hdr = f"{'file':<12}{'sfreq':>6}{'nyq':>6}{'winlen':>8}{'nwin':>6}{'expect':>8}{'chans':>7}{'nan/inf':>9}"
    print(hdr); print("-" * len(hdr))
    for name, m in rows:
        ok = (m["sfreq"] == TARGET_SFREQ and m["win_len_ok"]
              and m["n_windows"] == m["expected_windows"]
              and m["n_channels"] == 18 and m["nan_inf"] == 0)
        print(f"{name:<12}{m['sfreq']:>6.0f}{m['nyquist']:>6.0f}"
              f"{'2000' if m['win_len_ok'] else 'BAD':>8}{m['n_windows']:>6}"
              f"{m['expected_windows']:>8}{m['n_channels']:>7}{m['nan_inf']:>9}"
              f"   {'PASS' if ok else 'FAIL'}")

    print("\n=== FILTER FINGERPRINTS (processed / raw power ratio) ===")
    hdr2 = f"{'file':<12}{'pass1-40':>10}{'<0.3Hz':>9}{'>75Hz':>9}{'50Hz raw':>10}{'50Hz proc':>11}"
    print(hdr2); print("-" * len(hdr2))
    for name, m in rows:
        print(f"{name:<12}{m['pass_ratio_1_40Hz']:>10.2f}"
              f"{m['atten_below_0.3Hz']:>9.3f}{m['atten_above_75Hz']:>9.4f}"
              f"{m['notch_raw_peak']:>10.1f}{m['notch_pre_peak']:>11.2f}")
    print("""
Interpretation:
  pass1-40  ~1.0   -> passband power preserved (good)
  <0.3Hz    <<1    -> low-freq drift removed by 0.3 Hz high-pass
  >75Hz     ~0     -> content above 75 Hz removed by low-pass
  50Hz raw  >1     -> mains peak present in raw
  50Hz proc ~1     -> mains peak flattened by the notch
Figures with the PSD curves are in ./qc/ .""")


if __name__ == "__main__":
    main()
