#!/usr/bin/env python3
"""Does a cohort share the corpus's spectral width? Sub-band shift spread says so.

Barth did not: SW 12.0308 ppm against the corpus's 20.024, so a feature occupies
1.664x as many points and NO translation can align it. The symptom, before we had
the metadata, was that independent shift fits in different sub-bands disagreed
wildly -- 23,517 points across the spectrum -- while each fitted well alone.

That symptom is a general test, and it needs no metadata. If a cohort shares the
corpus's ppm-per-point, one shift fits everywhere and the spread is small. If its
spectral width differs by a factor s, the required shift drifts linearly across
the spectrum and the spread is large.

A large spread here means REQUEST THE SUBMITTER METADATA for that cohort; it
cannot be fixed by shifting.

Usage:
    python code/analysis/check_spectral_width.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NPT, PPM_HI, SW = 131_072, 14.8237, 20.024
PPT = SW / NPT
SUB = [(70000, 76000), (76000, 82000), (82000, 88000), (88000, 94000)]

COHORTS = [
    ("Barth (as stored)", "data/Barth/aligned_128K_Workbench_Barth_Syndrome_WS625to680Zero_EDTASuppressed_v4.npy"),
    ("Barth (resampled)", "data/Barth/Barth_EDTASuppressed_v4_corpusaxis.npy"),
    ("MTBLS326", "data/mtbls326/MTBLS326_EDTASuppressed_v4_ref0ppm.npy"),
    ("MTBLS563", "data/mtbls563/MTBLS563_EDTASuppressed_v4_ref0ppm.npy"),
    ("BrC-T2D", "data/BrC_T2D/BC_T2D_newlabels_EDTASuppressed_v4_ref0ppm.npy"),
    ("TBI", "data/tbi_tirupati/TBI_Tirupati_ref0ppm.npy"),
]


def bc(a, b, lo, hi):
    x = a[lo:hi] - a[lo:hi].mean()
    y = b[lo:hi] - b[lo:hi].mean()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30))


def sh(y, s):
    w = np.zeros_like(y)
    if s > 0:
        w[s:] = y[:-s]
    elif s < 0:
        w[:s] = y[-s:]
    else:
        w = y.copy()
    return w


def main() -> None:
    C = np.load(ROOT / "data/combined/corpus_v4_ref0ppm_clean.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    ref = np.median(np.array(C[np.sort(rng.choice(C.shape[0], 400, replace=False))],
                             dtype=np.float64), axis=0)
    print(f"{'cohort':20s} {'sub-band shifts (pts)':>34s} {'spread':>8s} {'ppm':>7s}  verdict")
    for name, path in COHORTS:
        f = ROOT / path
        if not f.exists():
            print(f"{name:20s}  [missing]")
            continue
        med = np.median(np.array(np.load(f, mmap_mode="r"), dtype=np.float64), axis=0)
        # +/-1500 points is 0.23 ppm, a generous bound on a referencing error.
        # A wider search is NOT more thorough: the sugar region 3.2-4.1 ppm is
        # dense with similar glucose peaks, so a wide scan locks onto the wrong
        # one. Resampled Barth wants -255 at +/-1500 and a spurious +4440 at
        # +/-6000.
        bs = [max(range(-1500, 1501, 5), key=lambda q: bc(sh(med, q), ref, a, z))
              for a, z in SUB]
        spread = max(bs) - min(bs)
        verdict = ("consistent -- one axis fits" if spread <= 400 else
                   "SUSPECT -- request metadata" if spread <= 1200 else
                   "INCONSISTENT -- request metadata, do not shift")
        print(f"{name:20s} {str(bs):>34s} {spread:8d} {spread * PPT:7.3f}  {verdict}")


if __name__ == "__main__":
    main()
