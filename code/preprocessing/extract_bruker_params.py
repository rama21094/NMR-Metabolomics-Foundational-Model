#!/usr/bin/env python3
"""Extract Bruker acquisition/processing parameters WITHOUT copying any FIDs.

RUN THIS ON THE MACHINE WITH THE HARD DISK. It needs only Python 3 (standard
library only -- no numpy, no nmrglue), reads a few kilobytes per experiment, and
writes one CSV. Transfer that CSV, not the archive.

    python3 extract_bruker_params.py /path/to/bruker/root -o bruker_params.csv

Why this is small: a Bruker `acqus` is ~9 KB and a `procs` ~5 KB, while a single
128K-point `fid` is ~1 MB. For ~10,000 experiments that is roughly 140 MB of
parameter files against >10 GB of raw data, and the CSV this produces is a few
megabytes.

WHAT THESE PARAMETERS UNBLOCK, all of which are currently open in this project:

  SFO1 / BF1        Observe and basic frequency in MHz -> THE FIELD STRENGTH,
                    per experiment. The GISSMO basis is built at 600 MHz; if
                    part of the corpus was acquired at 500 or 700, one basis is
                    wrong for those spectra and that alone could explain a large
                    part of the missing variance in the LC fit gate.

  SF / OFFSET / SR  Processing-side spectral reference. THIS IS THE FIX FOR THE
                    ppm-AXIS OFFSET. The corpus was aligned to its own rightmost
                    peak rather than to a chemical standard, so its ppm labels
                    carry an arbitrary absolute offset of order 0.15-0.3 ppm.
                    That is invisible within the corpus but blocks any
                    comparison against an external reference library such as
                    GISSMO. SF vs BF1 gives the true referencing.

  DATE              Acquisition timestamp (Unix seconds). This is the GOLD
                    STANDARD for the run-order/batch audit. That audit currently
                    uses sample-numbering as a proxy and could only prove
                    MTBLS326 confounded; real timestamps would upgrade the
                    verdicts for MTBLS563 and BrC-T2D cancer, which are
                    presently "order caveat, unresolved".

  PULPROG           Pulse programme -> confirms CPMG vs NOESY per experiment,
                    rather than trusting the folder naming.

  NS / RG / TE / P1 Scans, receiver gain, temperature, pulse width. NS and RG set
                    the SNR, and SNR is the technical feature that rowMinMax
                    normalisation was found to leak; having them measured lets
                    that be tested directly instead of inferred.

  PROBHD / INSTRUM  Probe and instrument identity -> a real batch variable.

The `relpath` column is the experiment directory relative to the scan root, so
these rows can be joined back to the existing `folder_path` columns in the
per-cohort metadata mapping CSVs.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

# JCAMP-DX style: ##$KEY= value   (values may be <strings> or (arrays))
_PAT = re.compile(r"^##\$?([A-Za-z0-9_]+)=\s*(.*)$")

ACQUS_KEYS = ["SFO1", "SFO2", "BF1", "BF2", "NUC1", "SW", "SW_h", "SWH", "TD",
              "NS", "DS", "RG", "PULPROG", "DATE", "TE", "D", "P", "SOLVENT",
              "INSTRUM", "PROBHD", "AQ_mod", "DIGMOD", "DECIM", "DSPFVS", "O1"]
PROCS_KEYS = ["SF", "SI", "OFFSET", "SW_p", "WDW", "LB", "SSB", "PHC0", "PHC1",
              "BC_mod", "ABSF1", "ABSF2", "NC_proc"]


def parse_jcamp(path):
    out = {}
    try:
        with open(path, "r", errors="ignore") as fh:
            for line in fh:
                m = _PAT.match(line.strip())
                if not m:
                    continue
                k, v = m.group(1), m.group(2).strip()
                if v.startswith("<") and v.endswith(">"):
                    v = v[1:-1]
                out[k] = v
    except OSError:
        pass
    return out


def find_procs(exp_dir):
    """First available pdata/<n>/procs, numerically lowest n."""
    pdata = os.path.join(exp_dir, "pdata")
    if not os.path.isdir(pdata):
        return None
    subs = []
    for d in os.listdir(pdata):
        p = os.path.join(pdata, d, "procs")
        if os.path.isfile(p):
            subs.append((int(d) if d.isdigit() else 1 << 30, p))
    return sorted(subs)[0][1] if subs else None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="top of the Bruker tree on the hard disk")
    ap.add_argument("-o", "--out", default="bruker_params.csv")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        sys.exit(f"not a directory: {root}")

    rows, n_seen = [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        if "acqus" not in filenames:
            continue
        n_seen += 1
        a = parse_jcamp(os.path.join(dirpath, "acqus"))
        pth = find_procs(dirpath)
        p = parse_jcamp(pth) if pth else {}

        rec = {"relpath": os.path.relpath(dirpath, root)}
        for k in ACQUS_KEYS:
            rec["acqu_" + k] = a.get(k, "")
        for k in PROCS_KEYS:
            rec["proc_" + k] = p.get(k, "")

        # convenience derivations
        try:
            sfo1 = float(a.get("SFO1", "nan"))
            rec["field_mhz_rounded"] = str(int(round(sfo1 / 50.0) * 50)) if sfo1 == sfo1 else ""
        except ValueError:
            rec["field_mhz_rounded"] = ""
        try:
            sf, bf1 = float(p.get("SF", "nan")), float(a.get("BF1", "nan"))
            # SR is the reference shift in Hz: how far the referencing moved the axis
            rec["SR_hz"] = f"{(sf - bf1) * 1e6:.3f}" if sf == sf and bf1 == bf1 else ""
        except ValueError:
            rec["SR_hz"] = ""
        rows.append(rec)
        if not args.quiet and n_seen % 500 == 0:
            print(f"  {n_seen} experiments ...", flush=True)

    if not rows:
        sys.exit(f"no Bruker experiments (directories containing 'acqus') under {root}")

    cols = ["relpath", "field_mhz_rounded", "SR_hz"] + \
           ["acqu_" + k for k in ACQUS_KEYS] + ["proc_" + k for k in PROCS_KEYS]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\nwrote {args.out}  ({len(rows)} experiments, {size_mb:.1f} MB)")
    fields = {}
    for r in rows:
        fields[r["field_mhz_rounded"]] = fields.get(r["field_mhz_rounded"], 0) + 1
    print("field strengths seen (rounded to 50 MHz):")
    for k, v in sorted(fields.items(), key=lambda kv: -kv[1]):
        print(f"   {k or '(unknown)':>10} MHz : {v}")
    pulses = {}
    for r in rows:
        pulses[r["acqu_PULPROG"]] = pulses.get(r["acqu_PULPROG"], 0) + 1
    print("top pulse programmes:")
    for k, v in sorted(pulses.items(), key=lambda kv: -kv[1])[:8]:
        print(f"   {k or '(unknown)':>24} : {v}")


if __name__ == "__main__":
    main()
