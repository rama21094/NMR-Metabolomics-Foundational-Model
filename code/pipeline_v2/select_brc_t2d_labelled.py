#!/usr/bin/env python3
"""Select the labelled BrC-T2D spectra from a no-dedupe build.

BrC-T2D was acquired in-house with three experiments per sample: a zgpr, a short
CPMG, and a full CPMG. Only one of them corresponds to a row in the label table,
and the label table is the authority on which -- so the cohort is defined by
BC_T2D_metadata_mapping.csv, not by anything measurable in the spectra.

This matters concretely. Running the corpus dedupe on this cohort dropped expnos
34 and 82 as near-duplicates of 33 and 81 (r = 0.999997 and 0.999995, the same
patient scanned ~19 minutes apart). Both dropped rows were the LABELLED ones.
So this cohort is built with --dedupe 0 and subset here instead.

    python build_corpus.py "<BrC-T2D project>" --out-prefix brc_t2d_all --out-dir rebuild/cohorts
    python select_brc_t2d_labelled.py

Patient names appear in the Bruker title files and in the mapping CSV. They stay
in the source tree; nothing written here carries them.
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np

LABEL_COLS = ["ID", "Group", "cancer_status", "diabetes_status", "combined_status"]


def expno(path: str):
    m = re.search(r"/(\d+)/pdata/\d+$", path)
    return int(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-prefix", default="rebuild/cohorts/brc_t2d_all")
    ap.add_argument("--mapping", default="fromSSD/BC_T2D_metadata_mapping.csv")
    ap.add_argument("--out-prefix", default="rebuild/cohorts/brc_t2d")
    args = ap.parse_args()

    meta = {int(r["ExpNo"]): r for r in csv.DictReader(open(args.mapping))}
    prov = list(csv.DictReader(open(f"{args.build_prefix}_provenance.csv")))
    built = {expno(r["pdata_path"]): int(r["row"]) for r in prov}

    keep = sorted(set(meta) & set(built))
    missing = sorted(set(meta) - set(built))
    print(f"labelled in mapping: {len(meta)}")
    print(f"present in build:    {len(keep)}")
    if missing:
        print(f"  !! labelled but absent from the build: {missing}")
        print("     (check the PULPROG filter -- a labelled sample should be CPMG)")

    X = np.load(f"{args.build_prefix}_spectra.npy", mmap_mode="r")
    rows = [built[e] for e in keep]
    out = np.lib.format.open_memmap(f"{args.out_prefix}_spectra.npy", mode="w+",
                                    dtype=np.float32, shape=(len(rows), X.shape[1]))
    for j, r in enumerate(rows):
        out[j] = X[r]
    out.flush(); del out

    np.save(f"{args.out_prefix}_ppm_axis.npy",
            np.load(f"{args.build_prefix}_ppm_axis.npy"))

    src = {int(r["row"]): r for r in prov}
    with open(f"{args.out_prefix}_labels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "expno", "pdata_path", "sw_ppm"] + LABEL_COLS)
        for j, e in enumerate(keep):
            p = src[built[e]]
            w.writerow([j, e, p["pdata_path"], p["sw_ppm"]] +
                       [meta[e].get(c, "") for c in LABEL_COLS])

    counts = {}
    for e in keep:
        counts[meta[e]["Group"]] = counts.get(meta[e]["Group"], 0) + 1
    print(f"\nwrote {args.out_prefix}_spectra.npy  ({len(rows)} x {X.shape[1]:,})")
    print(f"wrote {args.out_prefix}_labels.csv")
    print("\nclass balance:")
    for k in sorted(counts):
        print(f"   {k:<10} {counts[k]:>4}")


if __name__ == "__main__":
    main()
