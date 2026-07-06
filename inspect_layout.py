"""Read-only reconnaissance for an unfamiliar data location.

Run this FIRST against any new data directory (a different server/drive/folder)
before touching config/pipeline.yaml. It answers three questions:

  1. What files are actually there, and do the .edf/.csv names match the
     '<patient>-<cond>_raw.edf' / '<patient>-<cond>_<date>.csv' convention
     src/livinglab_prep/discovery.py expects?
  2. Do the EDF channel labels match the 'EEG {name}-Pz' pattern and the
     canonical 18-scalp-electrode set (+ A1/A2 ears) that config/pipeline.yaml
     (channels.keep, reref.ref_channels) assumes?
  3. Does every file share the original sampling rate config/pipeline.yaml
     assumes (signal.orig_sfreq)?

Nothing is written and no full recording is loaded: EDF headers are read with
preload=False, which is cheap regardless of file size or channel count.

Run:
    python inspect_layout.py <root_dir>
    python inspect_layout.py <root_dir> --max-inspect 500
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import mne

mne.set_log_level("ERROR")

# Mirrors src/livinglab_prep/discovery.py's naming convention.
_EDF_RE = re.compile(r"^(?P<patient>.+)-(?P<cond>[a-zA-Z])_raw\.edf$")

# Mirrors config/pipeline.yaml: channels.keep + reref.ref_channels.
EXPECTED_SCALP = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3",
                  "Cz", "C4", "T4", "T5", "P3", "P4", "T6", "O1", "O2"]
EXPECTED_REF = ["A1", "A2"]
EXPECTED_ORIG_SFREQ = 300


def electrode_name(label: str) -> str:
    """Mirrors preprocess.py / qc_preprocessing.py's channel-label parsing."""
    name = label[4:] if label.upper().startswith("EEG ") else label
    return name.split("-")[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="directory to scan (searched recursively)")
    parser.add_argument("--max-inspect", type=int, default=200,
                        help="max EDF headers to open (default 200)")
    args = parser.parse_args()
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"root does not exist: {root}")

    print(f"Scanning {root} ...\n")

    # ---- 1. file inventory ----
    all_files = [p for p in root.rglob("*") if p.is_file()]
    by_ext = Counter(p.suffix.lower() for p in all_files)
    print("=== FILE INVENTORY (by extension) ===")
    for ext, n in sorted(by_ext.items(), key=lambda x: -x[1]):
        print(f"  {ext or '(no ext)':<10} {n:>6}")
    print(f"  {'TOTAL':<10} {len(all_files):>6}\n")

    edf_files = [p for p in all_files if p.suffix.lower() == ".edf"]

    # ---- 2. naming convention check ----
    print("=== EDF NAMING CONVENTION ('<patient>-<cond>_raw.edf') ===")
    matched, unmatched = [], []
    for p in edf_files:
        (matched if _EDF_RE.match(p.name) else unmatched).append(p)
    print(f"  {len(matched)}/{len(edf_files)} EDF filenames match the expected pattern")
    if unmatched:
        print("  Examples that do NOT match (showing up to 10):")
        for p in unmatched[:10]:
            print(f"    {p.relative_to(root)}")
    print()

    # ---- 3. paired CSV check ----
    print("=== PAIRED TASK CSVs ===")
    paired, unpaired = 0, []
    for p in matched:
        m = _EDF_RE.match(p.name)
        patient, cond = m.group("patient"), m.group("cond")
        candidates = list(p.parent.glob(f"{patient}-{cond}_*.csv"))
        if candidates:
            paired += 1
        else:
            unpaired.append(p.name)
    print(f"  {paired}/{len(matched)} matched EDFs have a paired '<patient>-<cond>_*.csv'")
    if unpaired:
        print(f"  Missing CSV for (showing up to 10): {unpaired[:10]}")
    print()

    # ---- 4. EDF header inspection (channels, sfreq, duration) ----
    n_to_check = min(args.max_inspect, len(edf_files))
    print(f"=== EDF HEADER DETAILS (first {n_to_check} of {len(edf_files)}) ===")
    sfreqs: Counter = Counter()
    chan_counts: Counter = Counter()
    scalp_complete = 0
    ref_complete = 0
    rows = []
    for p in edf_files[:n_to_check]:
        try:
            raw = mne.io.read_raw_edf(p, preload=False, verbose="ERROR")
        except Exception as e:
            rows.append((p.name, f"FAILED TO READ: {e}"))
            continue
        sfreq = raw.info["sfreq"]
        n_ch = len(raw.ch_names)
        dur_min = raw.n_times / sfreq / 60
        sfreqs[round(sfreq, 3)] += 1
        chan_counts[n_ch] += 1

        lookup = {electrode_name(c): c for c in raw.ch_names}
        missing_scalp = [e for e in EXPECTED_SCALP if e not in lookup]
        missing_ref = [e for e in EXPECTED_REF if e not in lookup]
        if not missing_scalp:
            scalp_complete += 1
        if not missing_ref:
            ref_complete += 1

        rows.append((p.name, f"sfreq={sfreq} n_ch={n_ch} dur={dur_min:.1f}min "
                              f"missing_scalp={missing_scalp} missing_ref={missing_ref}"))

    for name, info in rows[:20]:
        print(f"  {name}: {info}")
    if len(rows) > 20:
        print(f"  ... ({len(rows) - 20} more; rerun with --max-inspect to see more/fewer)")

    print(f"\n  sfreq distribution: {dict(sfreqs)}  (pipeline expects {EXPECTED_ORIG_SFREQ})")
    print(f"  channel-count distribution: {dict(chan_counts)}")
    print(f"  {scalp_complete}/{len(rows)} files have all 18 expected scalp electrodes")
    print(f"  {ref_complete}/{len(rows)} files have both A1 and A2 (needed for linked-ears reref)")

    if edf_files:
        raw0 = mne.io.read_raw_edf(edf_files[0], preload=False, verbose="ERROR")
        print(f"\n  Full channel list from {edf_files[0].name}:")
        print(f"    {raw0.ch_names}")


if __name__ == "__main__":
    main()
