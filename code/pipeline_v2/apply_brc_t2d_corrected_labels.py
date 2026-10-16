#!/usr/bin/env python3
"""Restrict BrC-T2D to the corrected label file and apply its labels (2026-10-15).

data/BrC_T2D/BrC-T2D-newlabels_latest_corrected.csv lists 78 patients and is the
authority. The previous cohort (115 spectra, 112 patients, labels from the old
metadata mapping) is archived in rebuild/cohorts/archive_brc_t2d_115/ and is the
INPUT here, so this script is idempotent.

Patients absent from the corrected file are dropped (decision 2026-10-15). The
spectra array is not rewritten: the label CSV keeps each spectrum's original
"row" index, which is what load_cohort() uses.

SM19, SM23 and SM31 each have a second "repeat" acquisition (expnos 361, 364,
358; flagged "repeat" in the sample-ID column of BC_T2D_metadata_mapping.csv).
Only the original is kept, so the cohort is one spectrum per patient: a repeat
in the training fold and its original in the test fold would leak. (The
115-spectrum cohort kept both -- a small leak now removed.)
"""
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "rebuild/cohorts/archive_brc_t2d_115/brc_t2d_labels.csv"
NEW = ROOT / "data/BrC_T2D/BrC-T2D-newlabels_latest_corrected.csv"
OUT = ROOT / "rebuild/cohorts/brc_t2d_labels.csv"
STATUS = {"BC": "Cancer", "NC": "No Cancer", "D": "Diabetes", "ND": "No Diabetes"}

new = {r["Sample"].strip(): r["Label"].strip()
       for r in csv.DictReader(open(NEW, encoding="utf-8-sig"))}
assert set(new.values()) <= {"BC-D", "BC-ND", "NC-D", "NC-ND"}, set(new.values())
rows = list(csv.DictReader(open(SRC)))
missing = set(new) - {r["ID"] for r in rows}
assert not missing, f"labelled patients with no spectrum: {missing}"

META = ROOT / "data/BrC_T2D/BC_T2D_metadata_mapping.csv"
repeats = {m["ExpNo"].strip() for m in
           csv.DictReader(open(META, encoding="utf-8-sig"))
           if "repeat" in m.get("Sample name/ID", "").lower()}

kept, changed = [], []
for r in rows:
    lab = new.get(r["ID"])
    if lab is None or r["expno"] in repeats:
        continue
    if lab != r["Group"]:
        changed.append((r["ID"], r["Group"], lab))
    c, d = lab.split("-")
    r.update(Group=lab, cancer_status=STATUS[c], diabetes_status=STATUS[d],
             combined_status=f"{STATUS[c]} {STATUS[d]}")
    kept.append(r)

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(kept)
print(f"{len(rows)} spectra -> {len(kept)} kept ({len({r['ID'] for r in kept})} patients)")
for c in changed:
    print("  relabelled", *c)
print("  classes", dict(sorted(Counter(r["Group"] for r in kept).items())))
