#!/usr/bin/env python3
"""Attach class labels to the rebuilt evaluation cohorts.

Each cohort's labels live in a mapping CSV keyed by the folder layout of the
ORIGINAL download, while the rebuild records the pdata path it actually read.
This joins the two and writes <cohort>_labels.csv, one row per array row, in
array order. A row that cannot be matched is reported rather than dropped
silently: an unlabelled spectrum in an evaluation cohort is a defect, not a
detail.

BrC-T2D is not handled here -- its cohort membership IS its label table, so it
is built by select_brc_t2d_labelled.py.
"""
import csv
import re
from pathlib import Path

COHORTS = {
    # name: (mapping csv, key column, label column, how to derive the key)
    "mtbls563": ("fromSSD/MTBLS563_metadata_mapping.csv", "folder_name",
                 "Factor Value[Diagnosis]", "joined"),
    "mtbls326": ("fromSSD/MTBLS326_metadata_mapping.csv", "folder_name",
                 "Factor Value[IP3R expression]", "joined"),
    "barth":    ("data/Barth/Workbench_Barth_Syndrome_metadata.csv", "sample_folder",
                 "Phenotype", "component"),
}


def joined_key(path: str) -> str:
    """.../MTBLS563_RAW_FILES/main/1/1/pdata/1  ->  MTBLS563_RAW_FILES_main_1_1"""
    parts = Path(path).parts
    i = next(i for i, p in enumerate(parts) if "_RAW_FILES" in p)
    return "_".join(parts[i:-2]).replace("%", "_")


def main() -> None:
    for name, (mapping, keycol, labcol, how) in COHORTS.items():
        prov = list(csv.DictReader(open(f"rebuild/cohorts/{name}_provenance.csv")))
        meta = list(csv.DictReader(open(mapping)))
        by_key = {r[keycol]: r for r in meta}

        out, missing = [], []
        for r in prov:
            p = r["pdata_path"]
            if how == "joined":
                k = joined_key(p)
            else:
                k = next((s for s in by_key if f"/{s}/" in p), None)
            m = by_key.get(k)
            if m is None:
                missing.append(p)
                continue
            out.append({"row": r["row"], "key": k, "label": m[labcol].strip(),
                        "pdata_path": p, "sw_ppm": r["sw_ppm"]})

        with open(f"rebuild/cohorts/{name}_labels.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["row", "key", "label", "pdata_path", "sw_ppm"])
            w.writeheader(); w.writerows(out)

        counts = {}
        for r in out:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        print(f"{name:<10} matched {len(out)}/{len(prov)}"
              f"{'  MISSING ' + str(len(missing)) if missing else ''}")
        for k in sorted(counts):
            print(f"              {k or '(blank)':<28} {counts[k]:>4}")
        for p in missing[:3]:
            print(f"              unmatched: {p}")


if __name__ == "__main__":
    main()
