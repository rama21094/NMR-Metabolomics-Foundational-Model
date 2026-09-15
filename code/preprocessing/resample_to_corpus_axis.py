#!/usr/bin/env python3
"""Put a cohort on the corpus's ppm-per-point by resampling. Generalises the Barth fix.

A cohort acquired with a different spectral width cannot be aligned to the corpus
by any translation: a feature simply occupies a different number of points. The
symptom is that independent shift fits in different sub-bands disagree
(code/analysis/check_spectral_width.py).

    corpus     SW 20.024  ppm  -> 1.52771e-4 ppm/point
    Barth      SW 12.0308 ppm  -> 9.17877e-5   (ratio 0.601, gross)
    BrC-T2D    SW 20.1587 ppm  -> 1.53799e-4   (ratio 1.0067, subtle but real)

WHAT THE BRUKER `OFFSET` DOES AND DOES NOT GIVE US. `procs:OFFSET` is the ppm of
the first processed point and would fix each cohort's ABSOLUTE axis, which 5n
records as unrecoverable. It does not survive this pipeline: the stored .npy files
were put through a rightmost-peak alignment that re-referenced every spectrum, so
OFFSET describes the original data, not the arrays we hold. Resampling from SW is
therefore sound; positioning from OFFSET is not, and the residual offset must
still be fixed by an anchor or by cross-correlation afterwards.

Usage:
    python code/preprocessing/resample_to_corpus_axis.py \\
        --src data/BrC_T2D/BC_T2D_newlabels_WS625to680Zero_EDTASuppressed_v4.npy \\
        --out data/BrC_T2D/BC_T2D_corpusaxis.npy --sw 20.1587
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NPT = 131_072
CORPUS_SW, CORPUS_PPM_HI = 20.024, 14.8237


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sw", type=float, required=True,
                    help="the cohort's spectral width in ppm, from acqus "
                         "(SW, or SW_h/SFO1)")
    ap.add_argument("--pivot", type=int, default=None,
                    help="stored index held fixed by the resampling. Defaults to "
                         "the middle of the metabolite region, which keeps the "
                         "residual shift small; the exact choice does not matter "
                         "because a residual shift is fitted afterwards anyway.")
    args = ap.parse_args()

    pivot = args.pivot if args.pivot is not None else 82_000
    dp_c, dp_x = CORPUS_SW / NPT, args.sw / NPT
    j = np.arange(NPT, dtype=np.float64)
    i_src = pivot + (j - pivot) * (dp_c / dp_x)
    print(f"cohort SW {args.sw:.4f} ppm ({dp_x:.6e} ppm/pt), corpus "
          f"{CORPUS_SW:.4f} ({dp_c:.6e}); ratio {dp_c / dp_x:.6f}")
    inside = (i_src >= 0) & (i_src <= NPT - 1)
    print(f"corpus points covered: {inside.sum():,}/{NPT:,} "
          f"({100 * inside.mean():.1f}%)")

    X = np.load(ROOT / args.src, mmap_mode="r")
    n = X.shape[0]
    Y = np.zeros((n, NPT), dtype=np.float32)
    grid = np.arange(NPT, dtype=np.float64)
    for k in range(n):
        Y[k] = np.interp(i_src, grid, np.array(X[k], dtype=np.float64),
                         left=0.0, right=0.0)
    np.save(ROOT / args.out, Y)
    print(f"wrote {args.out}  {Y.shape}")


if __name__ == "__main__":
    main()
