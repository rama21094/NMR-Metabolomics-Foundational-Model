#!/usr/bin/env python3
"""Align a cohort to the corpus axis by cross-correlation, for cohorts with no 0 ppm peak.

code/preprocessing/rereference_zero_ppm.py anchors each spectrum on its own
reference singlet. MTBLS563 has one in 0.7% of spectra, so that route is not
available and it must be aligned against the corpus consensus instead.

WHY THIS IS A SCRIPT AND NOT AN INLINE ONE-OFF. The first attempt at this WAS an
inline one-off, and it applied the shift with the wrong sign: it left the cohort
median at correlation -0.020 with the corpus where +0.581 was available, and the
error surfaced only later as an unexplained fall in the corpus-cohort overlap
diagnostic. Cross-correlation sign conventions are easy to invert and impossible
to eyeball, so this script DETERMINES the sign empirically -- it tries both and
keeps whichever actually raises the correlation -- and refuses to write output if
neither does.

It also checks for a scale difference before assuming translation is enough. Three
of the corpus's own source studies are stretched 1.5-1.7x and cannot be aligned by
shifting at all; a cohort could be the same. For MTBLS563 the fitted scale is
1.000, so translation is appropriate there.

Usage:
    python code/preprocessing/rereference_by_xcorr.py \\
        --src data/mtbls563/MTBLS563_aligned_spectra_WS625to680Zero_EDTASuppressed_v4.npy \\
        --out data/mtbls563/MTBLS563_EDTASuppressed_v4_ref0ppm.npy --tag mtbls563
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ("data/combined/combine_unique_MetaboLights_Workbench_"
          "Water_EDTA_Suppressed_ref0ppm_clean.npy")
BAND = (72000, 84000)          # aliphatic, clear of the zeroed water window
# The scale search needs a MUCH wider window than the shift search. A scale
# factor is only constrained by how far apart distant features sit, so judging it
# from a 12,000-point band gives an unstable answer -- the first version of this
# script reported 0.800 there against 1.000 over the full range. Scale is
# estimated over SCALE_BAND and reported with the spread across sub-bands, so an
# unreliable estimate is visible rather than asserted.
SCALE_BAND = (30000, 120000)
REPORT = ROOT / "results/analysis/rereference"


def band_corr(a, b, lo, hi):
    x = a[lo:hi] - a[lo:hi].mean()
    y = b[lo:hi] - b[lo:hi].mean()
    return float(x @ y / ((np.linalg.norm(x) + 1e-12) * (np.linalg.norm(y) + 1e-12)))


def shift_row(y, s):
    out = np.zeros_like(y)
    if s > 0:
        out[s:] = y[:-s]
    elif s < 0:
        out[:s] = y[-s:]
    else:
        out = y.copy()
    return out


def xcorr_lag(W, r, max_shift):
    m = W.shape[1]
    nfft = 1 << (2 * m - 1).bit_length()
    cc = np.fft.irfft(np.fft.rfft(W, nfft, axis=1) * np.fft.rfft(r[::-1], nfft),
                      nfft, axis=1)[:, :2 * m - 1]
    lags = np.arange(-(m - 1), m)
    ok = np.abs(lags) <= max_shift
    cc[:, ~ok] = -np.inf
    return lags[cc.argmax(axis=1)], cc.max(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--max-shift", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rigid", action="store_true",
                    help="apply ONE shift, the cohort median, to every spectrum "
                         "instead of a per-spectrum lag. Use when the cohort is "
                         "already internally consistent and only its offset "
                         "against the corpus is wrong. Per-spectrum lags on a "
                         "weak correlation (MTBLS563's median xcorr peak is "
                         "0.499) scatter rows that were tight: it widened "
                         "MTBLS563's p5-p95 spread from 233 to 2764 points "
                         "while the bulk IQR improved, i.e. it threw a tail of "
                         "rows onto neighbouring peaks. A rigid shift cannot do "
                         "that. It is only valid when the fitted scale is ~1.")
    args = ap.parse_args()

    REPORT.mkdir(parents=True, exist_ok=True)
    C = np.load(ROOT / args.corpus, mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    ref = np.median(np.array(C[np.sort(rng.choice(C.shape[0], 400, replace=False))],
                             dtype=np.float64), axis=0)

    X = np.load(ROOT / args.src, mmap_mode="r")
    n = X.shape[0]
    B = np.array(X, dtype=np.float64)
    med = np.median(B, axis=0)
    lo, hi = BAND

    # ---- is a translation even the right model? ---------------------------
    g = np.arange(X.shape[1], dtype=np.float64)
    def fit_scale(blo, bhi):
        bs, bv = 1.0, -2.0
        rr = ref[blo:bhi] - ref[blo:bhi].mean()
        rr /= np.linalg.norm(rr) + 1e-12
        for sc in np.arange(0.5, 2.01, 0.01):
            src = np.interp(g / sc, g, med, left=0, right=0)
            w = src[blo:bhi] - src[blo:bhi].mean()
            w /= np.linalg.norm(w) + 1e-12
            _, v = xcorr_lag(w[None, :], rr, args.max_shift)
            if v[0] > bv:
                bs, bv = sc, v[0]
        return bs, bv

    best_scale, best_v = fit_scale(*SCALE_BAND)
    halves = [fit_scale(SCALE_BAND[0], (SCALE_BAND[0] + SCALE_BAND[1]) // 2)[0],
              fit_scale((SCALE_BAND[0] + SCALE_BAND[1]) // 2, SCALE_BAND[1])[0]]
    print(f"best-fitting scale {best_scale:.3f} over {SCALE_BAND} "
          f"(corr {best_v:.3f}); per-half estimates {halves[0]:.3f}, {halves[1]:.3f}")
    spread = max(halves) - min(halves)
    if spread > 0.05:
        print(f"  estimates disagree by {spread:.3f} -- the scale is not well "
              "determined by this data; treating it as 1.000 and proceeding with "
              "translation only")
        best_scale = 1.0
    elif abs(best_scale - 1.0) > 0.02:
        print("  !! a scale difference this large cannot be fixed by shifting; "
              "this cohort needs resampling, not translation")

    # ---- per-spectrum lag, then determine the sign EMPIRICALLY -------------
    r = ref[lo:hi] - ref[lo:hi].mean()
    r /= np.linalg.norm(r) + 1e-12
    W = B[:, lo:hi] - B[:, lo:hi].mean(axis=1, keepdims=True)
    W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
    lag, peak = xcorr_lag(W, r, args.max_shift)

    base = band_corr(med, ref, lo, hi)
    cands = {}
    for name, sgn in (("+lag", +1), ("-lag", -1)):
        m2 = np.median(np.stack([shift_row(B[k], int(sgn * lag[k]))
                                 for k in range(min(n, 60))]), axis=0)
        cands[name] = (sgn, band_corr(m2, ref, lo, hi))
    print(f"\ncohort median vs corpus, correlation over {lo}-{hi}:")
    print(f"  unshifted        {base:+.3f}")
    for name, (sgn, v) in cands.items():
        print(f"  shifted by {name:5s} {v:+.3f}")

    name = max(cands, key=lambda k: cands[k][1])
    sgn, val = cands[name]
    if val <= base:
        raise SystemExit(f"\nNeither sign improves on unshifted ({base:+.3f}); "
                         "refusing to write. The cohort may need resampling, a "
                         "different band, or it may genuinely not resemble the "
                         "corpus.")
    print(f"\nusing {name}: {base:+.3f} -> {val:+.3f}")

    if args.rigid:
        # Scan the cohort median directly rather than taking the median of the
        # per-row lags. Those lags are multimodal -- rows lock onto different
        # neighbouring peaks -- so their median is not the cohort's offset. For
        # MTBLS563 the median of lags was +786 (corr -0.046) while a direct scan
        # found -4017 (corr +0.499).
        grid = range(-args.max_shift, args.max_shift + 1, 25)
        coarse = max(grid, key=lambda t: band_corr(shift_row(med, t), ref, lo, hi))
        s_one = max(range(coarse - 30, coarse + 31),
                    key=lambda t: band_corr(shift_row(med, t), ref, lo, hi))
        print(f"rigid scan: best single shift {s_one:+d} pts, corr "
              f"{band_corr(shift_row(med, s_one), ref, lo, hi):+.3f} "
              f"(median-of-lags would have given {int(np.median(sgn * lag)):+d} at "
              f"{band_corr(shift_row(med, int(np.median(sgn * lag))), ref, lo, hi):+.3f})")
        applied = np.full(n, s_one, dtype=np.int64)
        print(f"rigid: applying the cohort median shift {s_one:+d} to all {n} rows")
    else:
        applied = (sgn * lag).astype(np.int64)
    Y = np.stack([shift_row(B[k], int(applied[k])) for k in range(n)])
    np.save(ROOT / args.out, Y)
    pd.DataFrame({"row": np.arange(n), "shift": applied, "xcorr_peak": peak,
                  "method": f"xcorr_to_corpus_median({name})"}).to_csv(
        REPORT / f"rereference_per_spectrum_{args.tag}.csv", index=False)
    (REPORT / f"summary_{args.tag}.json").write_text(json.dumps({
        "src": args.src, "out": args.out, "n": int(n),
        "scale_fitted": float(best_scale), "sign": name,
        "median_corr_before": base, "median_corr_after": val,
        "rigid": bool(args.rigid),
        "median_shift": int(np.median(applied))}, indent=2))
    print(f"wrote {args.out}")
    print(f"  median shift {int(np.median(applied)):+d} pts, "
          f"per-spectrum xcorr peak median {np.median(peak):.3f}")


if __name__ == "__main__":
    main()
