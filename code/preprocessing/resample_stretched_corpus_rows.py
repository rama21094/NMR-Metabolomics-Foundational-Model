#!/usr/bin/env python3
"""Resample the corpus rows whose acquisition spectral width is not ~20 ppm.

read1D_align.py resampled every spectrum to 131,072 points across ITS OWN ppm
range (5v), so ppm-per-point varies by source study and a peak occupies 33 points
at 20 ppm but 55 at 12 ppm. No translation can align those rows; they have to be
put back on a common ppm-per-point.

Which rows: from results/analysis/corpus_row_study_map.csv, which recovers each
corpus row's source row by content matching and joins the per-row Bruker
acqu_SW (5w). Only rows whose width is known AND differs from the reference are
touched.

PARTIAL BY CONSTRUCTION. 2,923 corpus rows have no width recovered yet, and the
two 11.988 ppm serum studies (MTBLS10958, MTBLS11188, ~185 spectra) are among
them. This script therefore fixes the rows it can prove are wrong and leaves a
known population of stretched rows in place. The output name says v5partial for
that reason: it is better than v4 and it is not a corrected corpus.

ORDER OF OPERATIONS. This works on the already re-referenced corpus rather than
rebuilding from v4. Re-referencing is a per-row translation and resampling is a
per-row scale; doing scale-then-translate instead of translate-then-scale changes
only the translation, which is recomputed here anyway by re-anchoring exactly the
rows that were resampled, with the same anchor the rest of the corpus used.

Usage:
    python code/preprocessing/resample_stretched_corpus_rows.py
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NPT = 131_072
REF_SW = 20.0236          # the width 72% of the corpus was acquired at
PIVOT = 82_000
TOL = 0.05                # ppm; widths within this of REF_SW are left alone
WATER = (62_500, 68_000)
ZERO_IDX = 97_032


def fwhm_points(y, apex):
    half = y[apex] / 2.0
    l = apex
    while l > 0 and y[l] > half:
        l -= 1
    r = apex
    while r < len(y) - 1 and y[r] > half:
        r += 1
    return r - l


def locate(win, min_frac, gmax, max_fwhm=200):
    y = win - np.median(win)
    cand = []
    for i in range(2, len(y) - 2):
        if y[i] > 0 and y[i] >= y[i - 1] and y[i] > y[i + 1]:
            w = fwhm_points(y, i)
            if w <= max_fwhm:
                cand.append((float(y[i]), i))
    if not cand:
        return None
    h, apex = max(cand)
    if h / gmax < min_frac:
        return None
    a, b, c = y[apex - 1], y[apex], y[apex + 1]
    den = a - 2.0 * b + c
    d = 0.0 if den == 0 else 0.5 * (a - c) / den
    return apex + float(np.clip(d, -1, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/combined/corpus_v4_ref0ppm_clean.npy")
    ap.add_argument("--out", default="data/combined/corpus_v5partial_ref0ppm_clean.npy")
    ap.add_argument("--map", default="results/analysis/corpus_row_study_map.csv")
    ap.add_argument("--rowmap", default="results/analysis/atypical/row_map_clean.csv")
    args = ap.parse_args()

    sw_by_v4 = {}
    for r in csv.DictReader(open(ROOT / args.map)):
        if r["acqu_SW"]:
            sw_by_v4[int(r["corpus_row"])] = float(r["acqu_SW"])
    new_of_old = {int(r["old_row"]): int(r["new_row"])
                  for r in csv.DictReader(open(ROOT / args.rowmap))}

    todo = {}
    for v4row, sw in sw_by_v4.items():
        if abs(sw - REF_SW) > TOL and v4row in new_of_old:
            todo[new_of_old[v4row]] = sw
    print(f"rows with a known width: {len(sw_by_v4):,}")
    print(f"rows to resample (width differs, and survived the clean step): {len(todo):,}")
    from collections import Counter
    for sw, n in Counter(round(v, 3) for v in todo.values()).most_common():
        print(f"    SW {sw:<9} n={n:<5} scale {REF_SW / sw:.5f}")

    X = np.load(ROOT / args.src, mmap_mode="r")
    n = X.shape[0]
    out = np.lib.format.open_memmap(ROOT / args.out, mode="w+",
                                    dtype=X.dtype, shape=X.shape)
    grid = np.arange(NPT, dtype=np.float64)
    CH = 400
    for i in range(0, n, CH):
        j = min(i + CH, n)
        out[i:j] = X[i:j]
        print(f"\r  copying {j}/{n}", end="", flush=True)
    print()

    done = 0
    for row, sw in sorted(todo.items()):
        y = np.array(X[row], dtype=np.float64)
        i_src = PIVOT + (grid - PIVOT) / (REF_SW / sw)
        z = np.interp(i_src, grid, y, left=0.0, right=0.0)
        # re-anchor with the corpus's own convention, since the rescale moved it
        w = z.copy(); w[WATER[0]:WATER[1]] = 0.0
        gmax = float(np.abs(w).max()) + 1e-30
        half = 2618                       # +/-0.4 ppm
        apex = locate(w[ZERO_IDX - half:ZERO_IDX + half], 0.02, gmax)
        if apex is not None:
            s = int(round(ZERO_IDX - (ZERO_IDX - half + apex)))
            if s > 0:
                z = np.concatenate([np.zeros(s), z[:-s]])
            elif s < 0:
                z = np.concatenate([z[-s:], np.zeros(-s)])
        out[row] = z
        done += 1
    out.flush()
    print(f"resampled and re-anchored {done:,} rows")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
