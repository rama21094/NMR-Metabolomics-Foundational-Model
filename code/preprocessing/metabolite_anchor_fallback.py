#!/usr/bin/env python3
"""A second anchor for the 4.1% of spectra with no detectable 0 ppm reference.

code/preprocessing/rereference_zero_ppm.py locates a 0 ppm singlet in 95.9% of
spectra. The remaining 399 currently fall back to whole-band cross-correlation
against the corpus median. This script builds and TESTS an alternative: anchor on
a strong, stable metabolite landmark instead.

The candidate landmark is the sharp pair near 2.6 and 2.72 ppm. It was chosen
because code/analysis/inspect_residual_shifts.py showed it sitting on the corpus
median even in spectra whose crowded 3.1-3.6 region is visibly displaced -- i.e.
it is one of the least pH-mobile features available. Its chemical identity is NOT
confirmed here (citrate's AB quartet is the obvious candidate at this shift and
pH) and nothing below depends on the identity, only on the stability.

WHY THIS IS A TEST AND NOT AN ASSUMPTION. A second anchor is only worth adopting
if it beats the fallback already in place. We have ground truth for the 95.9% of
spectra where the reference singlet IS found, so both candidate fallbacks can be
scored against it on exactly those rows:

    ground truth      = shift from the 0 ppm singlet
    candidate 1       = whole-band cross-correlation (the current fallback)
    candidate 2       = cross-correlation restricted to the 2.5-2.8 ppm landmark

Whichever tracks ground truth more closely is the one the 399 should get. If the
current fallback wins, this script says so and changes nothing -- that is a
legitimate outcome, not a failure.

Cross-correlation over the landmark window is used rather than a bare argmax:
argmax picks whichever of the two peaks happens to be taller in a given spectrum,
which introduces a spurious ~0.12 ppm bimodality of its own.

Usage:
    python code/preprocessing/metabolite_anchor_fallback.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ("data/combined/combine_unique_MetaboLights_Workbench_"
          "Water_EDTA_Suppressed.npy")
RR = ROOT / "results/analysis/rereference"
LAGS = "results/analysis/alignment_qc/lag_all.npy"

NPT = 131_072
PPM_HI, SW = 14.8237, 20.024
PPM_PER_PT = SW / NPT
LINEWIDTH_PTS = 35.0

# 2.5-2.8 ppm on the canonical axis.
LM_LO = int(round((PPM_HI - 2.80) / PPM_PER_PT))
LM_HI = int(round((PPM_HI - 2.50) / PPM_PER_PT))


def xcorr_shift(B: np.ndarray, ref: np.ndarray, lo: int, hi: int,
                max_shift: int) -> np.ndarray:
    """Shift, in points, that best maps each row of B onto ref over [lo, hi)."""
    Y = B[:, lo:hi].astype(np.float64)
    Y = Y - Y.mean(axis=1, keepdims=True)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12
    r = ref[lo:hi] - ref[lo:hi].mean()
    r = r / (np.linalg.norm(r) + 1e-12)
    m = hi - lo
    nfft = 1 << (2 * m - 1).bit_length()
    cc = np.fft.irfft(np.fft.rfft(Y, nfft, axis=1) * np.fft.rfft(r[::-1], nfft),
                      nfft, axis=1)[:, :2 * m - 1]
    lags = np.arange(-(m - 1), m)
    ok = np.abs(lags) <= max_shift          # reject implausible alignments
    cc[:, ~ok] = -np.inf
    return -lags[cc.argmax(axis=1)]         # sign: shift to APPLY


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-shift", type=int, default=1500)
    ap.add_argument("--n-validate", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X = np.load(ROOT / CORPUS, mmap_mode="r")
    d = pd.read_csv(RR / "rereference_per_spectrum.csv")
    truth_ok = (d["method"] == "zero_peak").to_numpy()
    truth = d["shift"].to_numpy()
    xlag = np.load(ROOT / LAGS)

    print(f"landmark window {LM_LO}-{LM_HI} "
          f"({PPM_HI - LM_HI * PPM_PER_PT:.2f}-{PPM_HI - LM_LO * PPM_PER_PT:.2f} ppm)")
    print(f"ground truth available for {truth_ok.sum():,} spectra; "
          f"{(~truth_ok).sum():,} need a fallback\n")

    # Reference: median of well-referenced spectra, built on the CORRECTED axis.
    rng = np.random.default_rng(args.seed)
    base = np.sort(rng.choice(np.where(truth_ok & (np.abs(truth) < 50))[0],
                              400, replace=False))
    ref = np.median(np.array(X[base], dtype=np.float64), axis=0)

    # ---- validate both fallbacks on rows where truth is known ---------------
    val = np.sort(rng.choice(np.where(truth_ok)[0],
                             min(args.n_validate, int(truth_ok.sum())),
                             replace=False))
    B = np.array(X[val], dtype=np.float64)
    cand_lm = xcorr_shift(B, ref, LM_LO, LM_HI, args.max_shift)
    cand_xc = -xlag[val]                    # current fallback, same convention

    print("agreement with the 0 ppm ground truth, on "
          f"{len(val):,} spectra where both are available:")
    rows = []
    for name, cand in (("whole-band xcorr (current)", cand_xc),
                       ("2.5-2.8 ppm landmark", cand_lm)):
        err = np.abs(cand - truth[val])
        rows.append((name, np.median(err),
                     100 * (err <= LINEWIDTH_PTS).mean(),
                     100 * (err <= 2 * LINEWIDTH_PTS).mean()))
        print(f"  {name:28s} median |error| {rows[-1][1]:6.1f} pts   "
              f"within 1 lw {rows[-1][2]:5.1f}%   within 2 lw {rows[-1][3]:5.1f}%")

    better = max(rows, key=lambda r: r[3])
    print(f"\n  better fallback: {better[0]}")
    if better[0].startswith("whole-band"):
        print("  -> the landmark anchor does NOT beat what is already in place.")
        print("     Keep the current fallback; no change recommended.")
    else:
        print("  -> the landmark anchor is the better fallback for the 399.")

    # ---- emit shifts for the spectra that need them -------------------------
    need = np.where(~truth_ok)[0]
    Bn = np.array(X[need], dtype=np.float64)
    lm_need = xcorr_shift(Bn, ref, LM_LO, LM_HI, args.max_shift)
    out = pd.DataFrame({
        "row": need,
        "shift_xcorr_current": -xlag[need],
        "shift_landmark": lm_need,
        "disagreement_pts": np.abs(lm_need + xlag[need]),
    })
    out.to_csv(RR / "fallback_candidates.csv", index=False)
    print(f"\n  the two fallbacks disagree by: median "
          f"{out['disagreement_pts'].median():.0f} pts, "
          f"p90 {out['disagreement_pts'].quantile(0.9):.0f} pts")
    print(f"  wrote {(RR / 'fallback_candidates.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
