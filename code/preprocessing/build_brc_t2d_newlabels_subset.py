"""Extract the BrC/T2D rows covered by the revised label file (data/BrC_T2D/BrC-T2D_newlabels.csv)
from the raw (pre-suppression) spectra array, and build a matching metadata CSV.

The existing data/BrC_T2D/BC_T2D_aligned_spectra_WS625to680Zero.npy has 115 rows, one per
sample in BC_T2D_metadata_mapping.csv (matched by that file's "ID" column, e.g. "SM27",
"KM112"). The new label file only covers 78 of those 115 samples (the rest were apparently
dropped/reclassified) and gives one row per sample as "Sample,Label" where Label is one of
BC-D / BC-ND / NC-D / NC-ND (BC=Breast Cancer, NC=No Cancer; D=Diabetes, ND=No Diabetes).

Three sample IDs (SM19, SM23, SM31) have two rows in the old metadata -- an original
acquisition and a "... repeat" acquisition of the same sample with the same cancer/diabetes
status. We keep only the original (non-"repeat") row for those.

Output:
  data/BrC_T2D/BC_T2D_newlabels_aligned_spectra.npy       -- (78, 131072) raw subset, pre-suppression
  data/BrC_T2D/BC_T2D_newlabels_metadata_mapping.csv       -- npy_row, cancer_status, diabetes_status, ...
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

RAW_INPUT = "data/BrC_T2D/BC_T2D_aligned_spectra_WS625to680Zero.npy"
OLD_METADATA = "data/BrC_T2D/BC_T2D_metadata_mapping.csv"
NEW_LABELS = "data/BrC_T2D/BrC-T2D_newlabels.csv"
SUBSET_OUTPUT = "data/BrC_T2D/BC_T2D_newlabels_aligned_spectra.npy"
METADATA_OUTPUT = "data/BrC_T2D/BC_T2D_newlabels_metadata_mapping.csv"

LABEL_TO_STATUS = {
    "BC-D": ("Cancer", "Diabetes"),
    "BC-ND": ("Cancer", "No Diabetes"),
    "NC-D": ("No Cancer", "Diabetes"),
    "NC-ND": ("No Cancer", "No Diabetes"),
}


def load_old_metadata(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_new_labels(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["Sample"] = row["Sample"].strip()
        row["Label"] = row["Label"].strip()
    return rows


def match_rows(old_rows: list[dict], new_rows: list[dict]) -> list[dict]:
    by_id: dict[str, list[dict]] = {}
    for row in old_rows:
        by_id.setdefault(row["ID"].strip(), []).append(row)

    matched = []
    for new_row in new_rows:
        sample_id = new_row["Sample"]
        candidates = by_id.get(sample_id)
        if not candidates:
            raise ValueError(f"New-label sample {sample_id!r} not found in {OLD_METADATA}")
        if len(candidates) > 1:
            non_repeat = [c for c in candidates if "repeat" not in c["Sample name/ID"].lower()]
            chosen = non_repeat[0] if non_repeat else candidates[0]
        else:
            chosen = candidates[0]

        label = new_row["Label"]
        if label not in LABEL_TO_STATUS:
            raise ValueError(f"Unexpected label {label!r} for sample {sample_id!r}")
        cancer_status, diabetes_status = LABEL_TO_STATUS[label]

        matched.append({
            "orig_npy_row": int(chosen["npy_row"]),
            "ID": sample_id,
            "sample_name": chosen["Sample name/ID"],
            "new_label": label,
            "cancer_status": cancer_status,
            "diabetes_status": diabetes_status,
            "combined_status": f"{cancer_status} {diabetes_status}",
            "old_combined_status": chosen["combined_status"],
        })
    return matched


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--raw-input", default=RAW_INPUT)
    parser.add_argument("--old-metadata", default=OLD_METADATA)
    parser.add_argument("--new-labels", default=NEW_LABELS)
    parser.add_argument("--subset-output", default=SUBSET_OUTPUT)
    parser.add_argument("--metadata-output", default=METADATA_OUTPUT)
    args = parser.parse_args()

    old_rows = load_old_metadata(Path(args.old_metadata))
    new_rows = load_new_labels(Path(args.new_labels))
    matched = match_rows(old_rows, new_rows)

    n_conflicts = sum(1 for m in matched if m["combined_status"] != m["old_combined_status"])
    print(f"Matched {len(matched)}/{len(new_rows)} new-label samples to old metadata rows.")
    print(f"Label conflicts with old combined_status: {n_conflicts}")
    if n_conflicts:
        for m in matched:
            if m["combined_status"] != m["old_combined_status"]:
                print(f"  {m['ID']}: new={m['combined_status']!r} vs old={m['old_combined_status']!r}")

    matched.sort(key=lambda m: m["orig_npy_row"])
    indices = np.asarray([m["orig_npy_row"] for m in matched], dtype=np.int64)

    spectra = np.load(args.raw_input, mmap_mode="r")
    subset = np.asarray(spectra[indices], dtype=np.float64)
    Path(args.subset_output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.subset_output, subset)
    print(f"Wrote {subset.shape} to {args.subset_output}")

    fieldnames = ["npy_row", "orig_npy_row", "ID", "sample_name", "new_label",
                  "cancer_status", "diabetes_status", "combined_status", "old_combined_status"]
    with open(args.metadata_output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for new_row, m in enumerate(matched):
            writer.writerow({"npy_row": new_row, **m})
    print(f"Wrote metadata for {len(matched)} rows to {args.metadata_output}")

    counts_cancer = {}
    counts_diabetes = {}
    for m in matched:
        counts_cancer[m["cancer_status"]] = counts_cancer.get(m["cancer_status"], 0) + 1
        counts_diabetes[m["diabetes_status"]] = counts_diabetes.get(m["diabetes_status"], 0) + 1
    print(f"Cancer label counts: {counts_cancer}")
    print(f"Diabetes label counts: {counts_diabetes}")


if __name__ == "__main__":
    main()
