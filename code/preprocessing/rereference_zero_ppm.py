#!/usr/bin/env python3
"""Re-reference every spectrum on its own 0 ppm singlet, replacing the edge anchor.

WHAT WAS WRONG. code/preprocessing/alignSpectra.py anchored each spectrum on its
rightmost detectable feature, assuming that feature is the 0 ppm reference
resonance. code/analysis/reference_peak_check.py showed it is not: these spectra
extend ~5 ppm past 0, and the far right edge holds a median 0.03% of each
spectrum's global maximum -- it is the noise floor. Meanwhile a genuine sharp
singlet sits near 0 ppm in 97.5% of spectra at a median 8.4% of global maximum.
The corpus was aligned on noise while a usable reference sat 5 ppm away.

WHAT THIS DOES. For each spectrum: locate that singlet, and translate the
spectrum so it lands on one canonical 0 ppm index. Nothing else is altered --
no rescaling, no smoothing, no interpolation onto a new grid.

WHY A PEAK AND NOT THE INSTRUMENT FILES. Both work for the 8,892 MetaboLights
spectra, but the 778 Workbench spectra have no parameter export at all, and the
row -> study mapping was never recorded (docs/PI_outline.md 5h). The singlet is
in the data, so this route needs neither.

DETECTION, and why each guard is there:
  - Search +/-0.4 ppm around the nominal 0 ppm index. Wider admits real
    metabolite signal; narrower misses the +0.12 ppm population.
  - Require the candidate to be NARROW (FWHM below --max-fwhm points), to reject
    macromolecule tails. The threshold is set from MEASURED widths, not assumed
    ones. Metabolite peaks in this corpus have a median FWHM of 35 points
    (0.0054 ppm, 3.2 Hz at 600 MHz); the feature at 0 ppm has a median FWHM of 98
    points, roughly 2.8x broader, which is what a reference standard partly bound
    to serum albumin looks like. A first version of this script set the cut at 60
    points from an idealised 1.2 Hz linewidth and rejected 97% of spectra -- the
    guard was wrong, not the data. 200 points admits the observed feature with
    margin while still excluding true macromolecule humps, which run past 500.
  - Require height above --min-height of the spectrum's own global maximum.
  - Sub-point position by parabolic interpolation on the three points around the
    apex, then round. Without it the correction quantises to whole points and
    leaves a systematic residual.
  - Spectra failing any guard fall back to the cross-correlation displacement
    from code/analysis/alignment_qc.py, and are recorded as fallback so the
    proportion is visible rather than hidden.

WHAT THIS DOES NOT FIX. Three studies (~383 spectra) were acquired over a
narrower spectral window and are STRETCHED by 1.54-1.67x relative to the rest.
A translation cannot correct a scale difference. They are shifted like everything
else, flagged in the output, and still need resampling or exclusion -- a decision
deliberately deferred.

Usage:
    python code/preprocessing/rereference_zero_ppm.py
    python code/preprocessing/rereference_zero_ppm.py --dry-run   # report only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ("data/combined/combine_unique_MetaboLights_Workbench_"
          "Water_EDTA_Suppressed.npy")
OUT_NPY = ("data/combined/combine_unique_MetaboLights_Workbench_"
           "Water_EDTA_Suppressed_ref0ppm.npy")
LAGS = "results/analysis/alignment_qc/lag_all.npy"
REPORT = ROOT / "results/analysis/rereference"

NPT = 131_072
PPM_HI = 14.8237                 # MTBLS798 axis, the corpus majority
SW_PPM = 20.024
PPM_PER_PT = SW_PPM / NPT
ZERO_IDX = int(round(PPM_HI / PPM_PER_PT))     # canonical 0 ppm index
WATER = (62500, 68000)


def fwhm_points(y: np.ndarray, apex: int) -> int:
    """Full width at half maximum around `apex`, in points, walking outward."""
    half = y[apex] * 0.5
    i = apex
    while i > 0 and y[i] > half:
        i -= 1
    j = apex
    while j < len(y) - 1 and y[j] > half:
        j += 1
    return j - i


def locate(win: np.ndarray, min_height: float, max_fwhm: int):
    """Return (sub-point apex index within win, height, fwhm) or None."""
    y = win - np.median(win)
    apex = int(np.argmax(y))
    if apex <= 1 or apex >= len(y) - 2:
        return None
    h = float(y[apex])
    if h <= 0:
        return None
    w = fwhm_points(y, apex)
    if w > max_fwhm:
        return None
    # parabolic refinement on the apex and its two neighbours
    a, b, c = y[apex - 1], y[apex], y[apex + 1]
    denom = a - 2.0 * b + c
    delta = 0.0 if denom == 0 else 0.5 * (a - c) / denom
    return apex + float(np.clip(delta, -1, 1)), h, w


def shift_row(y: np.ndarray, s: int) -> np.ndarray:
    """Translate by s points with zero fill (no wraparound)."""
    if s == 0:
        return y
    out = np.zeros_like(y)
    if s > 0:
        out[s:] = y[:-s]
    else:
        out[:s] = y[-s:]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-ppm", type=float, default=0.4)
    ap.add_argument("--min-height", type=float, default=0.02,
                    help="as a fraction of the spectrum's own global maximum")
    ap.add_argument("--max-fwhm", type=int, default=200,
                    help="points. Measured: metabolite peaks 35, the 0 ppm "
                         "feature 98, macromolecule humps >500.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--corpus", default=CORPUS,
                    help="input .npy; the evaluation cohorts must be put on the "
                         "SAME canonical axis as the pretraining corpus or any "
                         "transfer result measures the axis mismatch instead of "
                         "the representation")
    ap.add_argument("--lags", default=LAGS,
                    help="per-row cross-correlation displacements used as the "
                         "fallback where no 0 ppm singlet is found. Omit with "
                         "--no-fallback for cohorts that have none computed.")
    ap.add_argument("--no-fallback", action="store_true",
                    help="leave undetected spectra unshifted rather than falling "
                         "back, and record them as such")
    ap.add_argument("--tag", default="",
                    help="label for the report files, so cohorts do not overwrite "
                         "each other")
    ap.add_argument("--consensus-tol", type=int, default=0,
                    help="points. If >0, run a second pass: rows whose located "
                         "apex sits further than this from the dataset's median "
                         "are re-searched in a narrow window around that median, "
                         "and if nothing is found there they take the median "
                         "shift. A 0.4 ppm search window is wide enough to "
                         "contain neighbouring peaks, and locate() has no "
                         "cross-row consistency check, so a third of MTBLS563 "
                         "locked onto the wrong peak (offsets clustered at "
                         "discrete values -1970, -1770, +2000). Rows of one "
                         "study share an axis; requiring them to agree removes "
                         "that failure without assuming what the shift is.")
    ap.add_argument("--out", default=OUT_NPY)
    args = ap.parse_args()

    REPORT.mkdir(parents=True, exist_ok=True)
    X = np.load(ROOT / args.corpus, mmap_mode="r")
    n = X.shape[0]
    if args.no_fallback:
        lag = np.zeros(n, dtype=np.int64)
    else:
        lag = np.load(ROOT / args.lags)
        if len(lag) != n:
            raise SystemExit(f"lag file has {len(lag)} rows, corpus has {n}; "
                             "pass --no-fallback or the right --lags")
    half = int(round(args.search_ppm / PPM_PER_PT))
    lo, hi = ZERO_IDX - half, ZERO_IDX + half
    print(f"corpus {n:,} x {X.shape[1]:,}")
    print(f"canonical 0 ppm index {ZERO_IDX}, search +/-{half} pts "
          f"({args.search_ppm} ppm)\n")

    shift = np.zeros(n, dtype=np.int64)
    method = np.empty(n, dtype=object)
    height = np.zeros(n)
    width = np.zeros(n, dtype=np.int64)

    for i in range(0, n, 500):
        j = min(i + 500, n)
        B = np.array(X[i:j], dtype=np.float64)
        B[:, WATER[0]:WATER[1]] = 0.0
        gmax = np.abs(B).max(axis=1) + 1e-30
        for k in range(j - i):
            r = locate(B[k, lo:hi], args.min_height, args.max_fwhm)
            row = i + k
            if r is None or r[1] / gmax[k] < args.min_height:
                shift[row] = -int(lag[row])      # fallback, same sign convention
                method[row] = "unshifted" if args.no_fallback else "xcorr_fallback"
                continue
            apex, h, w = r
            shift[row] = int(round(ZERO_IDX - (lo + apex)))
            method[row] = "zero_peak"
            height[row] = h / gmax[k]
            width[row] = w
        print(f"\r  located {j}/{n}", end="", flush=True)
    print()

    ok = method == "zero_peak"

    if args.consensus_tol > 0 and ok.sum() >= 5:
        tol = args.consensus_tol
        cons = int(round(np.median(shift[ok])))
        stray = ok & (np.abs(shift - cons) > tol)
        print(f"\n  consensus pass: dataset median shift {cons:+d} pts; "
              f"{stray.sum():,} of {ok.sum():,} located rows sit >{tol} pts away")
        clo, chi = ZERO_IDX - cons - tol, ZERO_IDX - cons + tol + 1
        recovered = forced = 0
        for row in np.where(stray)[0]:
            y = np.array(X[row], dtype=np.float64)
            y[WATER[0]:WATER[1]] = 0.0
            g = np.abs(y).max() + 1e-30
            r = locate(y[clo:chi], args.min_height, args.max_fwhm)
            if r is not None and r[1] / g >= args.min_height:
                shift[row] = int(round(ZERO_IDX - (clo + r[0])))
                height[row], width[row] = r[1] / g, r[2]
                recovered += 1
            else:
                shift[row] = cons
                method[row] = "consensus_median"
                forced += 1
        print(f"    re-located in the narrow window: {recovered:,}")
        print(f"    no peak there, took the median shift: {forced:,}")
        # Rows where no peak was found at all are the other source of a thrown
        # tail: with --no-fallback they stay put while every other row moves, so
        # a cohort that was internally tight acquires a handful of rows a full
        # shift away. They belong at the dataset consensus.
        if args.no_fallback:
            undetected = method == "unshifted"
            if undetected.any():
                shift[undetected] = cons
                method[undetected] = "consensus_median"
                print(f"    undetected rows moved to the median shift: "
                      f"{int(undetected.sum()):,}")
        ok = method == "zero_peak"

    print(f"\n  reference singlet found: {ok.sum():,}/{n:,} ({100 * ok.mean():.1f}%)")
    print(f"  no singlet found: {(~ok).sum():,} ({100 * (~ok).mean():.1f}%)"
          f"  [{'left unshifted' if args.no_fallback else 'xcorr fallback'}]")
    print(f"  singlet height, fraction of global max: median {np.median(height[ok]):.4f}")
    print(f"  singlet FWHM, points: median {int(np.median(width[ok]))}")
    q = np.percentile(shift, [5, 25, 50, 75, 95])
    print("\n  applied shift, points: p5 {:.0f}  p25 {:.0f}  med {:.0f} "
          " p75 {:.0f}  p95 {:.0f}".format(*q))
    print(f"  max |shift| {int(np.abs(shift).max())}")
    stretched = np.abs(shift) > 1500
    print(f"  |shift| > 1500 (the stretched studies; translation cannot fix "
          f"these): {stretched.sum():,}")

    tag = ("_" + args.tag) if args.tag else ""
    pd.DataFrame({"row": np.arange(n), "shift": shift, "method": method,
                  "peak_height_frac": height, "peak_fwhm_pts": width,
                  "xcorr_lag": lag,
                  "stretched_flag": stretched}).to_csv(
        REPORT / f"rereference_per_spectrum{tag}.csv", index=False)
    (REPORT / f"summary{tag}.json").write_text(json.dumps({
        "n": int(n), "zero_idx": ZERO_IDX,
        "found_pct": float(100 * ok.mean()),
        "median_height_frac": float(np.median(height[ok])),
        "median_fwhm_pts": int(np.median(width[ok])),
        "n_stretched": int(stretched.sum()),
    }, indent=2))

    if args.dry_run:
        print("\n  --dry-run: no corpus written")
        return

    out = ROOT / args.out
    print(f"\nwriting {out.name}")
    Y = np.lib.format.open_memmap(out, mode="w+", dtype=np.float64,
                                  shape=X.shape)
    for i in range(0, n, 200):
        j = min(i + 200, n)
        B = np.array(X[i:j], dtype=np.float64)
        for k in range(j - i):
            Y[i + k] = shift_row(B[k], int(shift[i + k]))
        print(f"\r  written {j}/{n}", end="", flush=True)
    Y.flush()
    del Y
    print(f"\n  wrote {out}")
    print("\nnext: python code/analysis/alignment_qc.py --corpus " + args.out)


if __name__ == "__main__":
    main()
