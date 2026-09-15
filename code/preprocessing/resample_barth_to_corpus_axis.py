#!/usr/bin/env python3
"""Put Barth on the corpus ppm axis by RESAMPLING, not by shifting.

Barth was acquired with a different spectral width from the corpus, so no
translation can align it and every shift-based attempt was fitting noise:

    corpus   SW 20.024 ppm over 131,072 points -> 1.5277e-4 ppm/point
    Barth    SW 12.0308 ppm over 131,072 points -> 9.1788e-5 ppm/point

A feature therefore occupies 1.664x as many points in Barth as in the corpus.
That is why independent sub-band shift fits spanned 3.59 ppm (5p), why the joint
scale search pinned at its boundary, and why no model exceeded corr 0.48.

Barth carries no TSP/DSS. Its chemical shift reference is FORMATE at 8.44 ppm
(submitter metadata: "Spectra were referenced internally to the formate signal",
standard concentration 1-2 mM). The 0 ppm anchor was therefore never applicable
to it; what that anchor locked onto was acetate at 1.902 ppm.

The axis is fixed by the formate singlet at index 25,520 in the stored grid:

    ppm(i) = 8.44 - (i - 25520) * 9.178772e-5

Cross-check: the tallest peak in the spectrum, the lactate CH3 doublet, then
lands at 1.310 ppm against a literature 1.33, and the formate-to-lactate point
separation implies 9.1521e-5 ppm/point against the metadata's 9.1788e-5 -- 0.29%.
The implied full range is +10.782 to -1.248 ppm, spanning 12.0307 ppm, which
reproduces the stated spectral width to four decimals.

Usage:
    python code/preprocessing/resample_barth_to_corpus_axis.py
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NPT = 131_072
CORPUS_PPM_HI, CORPUS_SW = 14.8237, 20.024
BARTH_SW = 12.0308
FORMATE_IDX, FORMATE_PPM = 25_520, 8.44


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/Barth/aligned_128K_Workbench_Barth_"
                                     "Syndrome_WS625to680Zero_EDTASuppressed_v4.npy")
    ap.add_argument("--out", default="data/Barth/Barth_EDTASuppressed_v4_corpusaxis.npy")
    ap.add_argument("--formate-idx", type=int, default=FORMATE_IDX)
    ap.add_argument("--residual-shift", type=int, default=-258,
                    help="corpus points, applied after resampling. Fitted: it "
                         "raises agreement with the corpus median from +0.192 to "
                         "+0.644 and three of four sub-bands independently agree "
                         "on it. It is -0.039 ppm, which is well inside the "
                         "corpus's OWN absolute-axis uncertainty (5n: the corpus "
                         "has no genuine reference standard), so it is absorbing "
                         "the corpus's arbitrary zero, not correcting formate.")
    args = ap.parse_args()

    dp_b = BARTH_SW / NPT
    dp_c = CORPUS_SW / NPT
    j = np.arange(NPT, dtype=np.float64)
    ppm_c = CORPUS_PPM_HI - j * dp_c                       # corpus ppm of each output point
    i_src = args.formate_idx + (FORMATE_PPM - ppm_c) / dp_b  # matching Barth index

    inside = (i_src >= 0) & (i_src <= NPT - 1)
    print(f"Barth  {dp_b:.6e} ppm/pt, span "
          f"{FORMATE_PPM + args.formate_idx * dp_b:+.3f} to "
          f"{FORMATE_PPM - (NPT - 1 - args.formate_idx) * dp_b:+.3f} ppm")
    print(f"corpus {dp_c:.6e} ppm/pt, span {ppm_c[0]:+.3f} to {ppm_c[-1]:+.3f} ppm")
    print(f"corpus points covered by Barth's range: {inside.sum():,}/{NPT:,} "
          f"({100 * inside.mean():.1f}%) -- the rest are zero-filled")

    X = np.load(ROOT / args.src, mmap_mode="r")
    n = X.shape[0]
    Y = np.zeros((n, NPT), dtype=np.float32)
    src_grid = np.arange(NPT, dtype=np.float64)
    for k in range(n):
        row = np.interp(i_src, src_grid, np.array(X[k], dtype=np.float64),
                        left=0.0, right=0.0)
        t = args.residual_shift
        if t > 0:
            row = np.concatenate([np.zeros(t), row[:-t]])
        elif t < 0:
            row = np.concatenate([row[-t:], np.zeros(-t)])
        Y[k] = row
    np.save(ROOT / args.out, Y)
    print(f"residual shift applied: {args.residual_shift:+d} pts "
          f"({args.residual_shift * dp_c:+.4f} ppm)")
    print(f"wrote {args.out}  {Y.shape}")


if __name__ == "__main__":
    main()
