#!/usr/bin/env python3
"""Recover each corpus row's source study and spectral width, by content matching.

WHY NOT INFER THE WIDTH FROM THE SPECTRUM. Five attempts to recover spectral width
from spectra alone have now failed on a case whose answer we knew (see
code/analysis/cohort_axes.py). The fifth was a constrained classifier over only the
six widths actually present in the data: it put BrC-T2D at 30.025 ppm, in 25 of 25
rows with a confident margin, when the truth is 20.024. The metabolite region is
dense enough that a wrong scale plus a shift beats the right one. So the width has
to come from acquisition metadata, which means knowing which source row each
corpus row came from.

The corpus was built by SELECTING rows from source arrays, so the spectra
themselves are the join key. Matching a 600-point fingerprint from 0.5-1.9 ppm
(clear of the zeroed water window and of the EDTA region) recovers the source row
at correlation > 0.999 for a clean match. row_mapping_Plasma_NoSuppress.csv then
supplies acqu_SW, proc_OFFSET and the field per source row.

Usage:
    python code/analysis/map_corpus_rows_to_studies.py
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CORPUS = "data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v4.npy"
SOURCES = [
    ("Plasma_NoSuppress", "fromSSD/aligned_128K_Plasma_NoSuppress.npy",
     "fromSSD/row_mapping_Plasma_NoSuppress.csv"),
    ("WSNoise", "fromSSD/aligned_nmr_spectra_128K_WSNoise.npy", None),
    ("Workbench_CPMG", "fromSSD/aligned_128K_Workbench_CPMG_serum_plasma.npy", None),
]
COLS = np.linspace(85000, 93500, 600).astype(int)
OUT = ROOT / "results/analysis/corpus_row_study_map.csv"


def fingerprint(X, rows):
    M = np.array(X[rows][:, COLS], dtype=np.float32)
    M -= M.mean(axis=1, keepdims=True)
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    return M


def main() -> None:
    V = np.load(ROOT / CORPUS, mmap_mode="r")
    n = V.shape[0]
    A = fingerprint(V, np.arange(n))
    best = np.full(n, -2.0)
    src = np.full(n, "", dtype=object)
    srow = np.full(n, -1, dtype=np.int64)
    for name, path, _ in SOURCES:
        f = ROOT / path
        if not f.exists():
            print(f"  missing source, skipped: {path}")
            continue
        S = np.load(f, mmap_mode="r")
        print(f"  matching against {name} {S.shape}")
        for i in range(0, S.shape[0], 2000):
            j = min(i + 2000, S.shape[0])
            sim = A @ fingerprint(S, np.arange(i, j)).T
            k = sim.argmax(axis=1)
            v = sim[np.arange(n), k]
            m = v > best
            best[m] = v[m]; src[m] = name; srow[m] = i + k[m]
    print(f"\nmatch quality: >0.999 {int((best > 0.999).sum()):,}/{n:,}  "
          f">0.99 {int((best > 0.99).sum()):,}  >0.95 {int((best > 0.95).sum()):,}")

    params = {}
    pf = ROOT / SOURCES[0][2]
    if pf.exists():
        for r in csv.DictReader(open(pf)):
            params[int(r["row_index"])] = r

    OUT.parent.mkdir(parents=True, exist_ok=True)
    got = 0
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["corpus_row", "match_corr", "source", "source_row",
                    "npy_path", "acqu_SW", "proc_OFFSET", "field_mhz", "proc_SI"])
        for i in range(n):
            p = params.get(int(srow[i])) if src[i] == "Plasma_NoSuppress" else None
            if p and best[i] > 0.99:
                got += 1
                w.writerow([i, f"{best[i]:.5f}", src[i], srow[i], p.get("npy_path", ""),
                            p.get("acqu_SW", ""), p.get("proc_OFFSET", ""),
                            p.get("field_mhz_rounded", ""), p.get("proc_SI", "")])
            else:
                w.writerow([i, f"{best[i]:.5f}", src[i], srow[i], "", "", "", "", ""])
    print(f"rows with a confident match AND parameters: {got:,}/{n:,}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
