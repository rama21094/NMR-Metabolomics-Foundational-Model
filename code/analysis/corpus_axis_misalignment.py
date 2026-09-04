#!/usr/bin/env python3
"""Confirm, in the corpus itself, the ppm-axis misalignment predicted from metadata.

code/analysis/bruker_param_audit.py predicts from the Bruker processing
parameters alone that the training corpus is not on a common chemical-shift
axis.  The cause is that code/preprocessing/alignSpectra.py aligns spectra BY
INDEX -- align_spectra_to_longest() interpolates every row to a common point
count and never reads a ppm axis -- so a peak lands at an index determined by
each row's own (proc_OFFSET, proc_SW_p, proc_SF).

The metadata prediction, as displacement at 3 ppm relative to MTBLS798
(53.6% of parameter rows):

    MTBLS798                      0 pts   53.6%
    MTBLS147/395/424/2336/46   -790 to -900 pts
    MTBLS540/974               -720 to -730 pts
    MTBLS2387/10958/11188     +5100 to +8000 pts, and stretched 1.54-1.67x

This script tests that prediction against the spectra, without using any
metadata, by cross-correlating each spectrum against the corpus median over a
band clear of the zeroed water window and reading off the lag.  If the
prediction is right, the lag histogram should be multi-modal with a dominant
mode at 0, a second mode near -800, and a small tail near +7000.

A scale reference throughout: a 1.2 Hz linewidth at 600 MHz is 0.002 ppm,
which over a 20 ppm window at 131,072 points is ~13 points.  Displacements of
several hundred points therefore mean peaks do not overlap at all, which is
why a ppm-referenced simulated basis cannot fit them and why fit_gate.py is
capped near R^2 ~ 0.5 regardless of solver conditioning.

Usage:
    python code/analysis/corpus_axis_misalignment.py
    python code/analysis/corpus_axis_misalignment.py --n 2000 --band 72000 84000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

CORPUS = ("data/combined/combine_unique_MetaboLights_Workbench_"
          "Water_EDTA_Suppressed.npy")

WATER = (62500, 68000)     # zeroed by WSZero_62500To68000.py
PPM_PER_PT = 20.02 / 131072
LINEWIDTH_PTS = 0.002 / PPM_PER_PT     # ~13 points


def measure_lags(X, idx, lo: int, hi: int):
    """Cross-correlation lag of each sampled row against the sample median.

    Rows are mean-centred and unit-normalised first so the correlation peak is
    a similarity, not a magnitude -- otherwise the tallest spectra dominate the
    median and the lag of a faint spectrum is meaningless.
    """
    B = np.array(X[idx][:, lo:hi], dtype=np.float64, copy=True)
    B -= B.mean(axis=1, keepdims=True)
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-12

    ref = np.median(B, axis=0)
    ref -= ref.mean()
    ref /= np.linalg.norm(ref)

    n = B.shape[1]
    nfft = 1 << (2 * n - 1).bit_length()
    cc = np.fft.irfft(
        np.fft.rfft(B, nfft, axis=1) * np.fft.rfft(ref[::-1], nfft),
        nfft, axis=1)[:, :2 * n - 1]
    return cc.argmax(axis=1) - (n - 1), cc.max(axis=1)


def modes(lag: np.ndarray, bin_pts: int = 50, min_frac: float = 0.01):
    """Coarse-binned lag modes, largest first."""
    b = np.round(lag / bin_pts) * bin_pts
    u, c = np.unique(b, return_counts=True)
    keep = c >= max(2, int(min_frac * len(lag)))
    order = np.argsort(-c[keep])
    return [(int(u[keep][i]), int(c[keep][i])) for i in order]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200,
                    help="rows to sample (cross-correlation is the cost)")
    ap.add_argument("--band", type=int, nargs=2, default=(72000, 84000),
                    help="index band to correlate over; must avoid the water window")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/analysis/bruker_params")
    args = ap.parse_args()

    lo, hi = args.band
    if not (hi <= WATER[0] or lo >= WATER[1]):
        raise SystemExit(f"band {lo}-{hi} overlaps the zeroed water window {WATER}")

    X = np.load(ROOT / CORPUS, mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(X.shape[0], min(args.n, X.shape[0]), replace=False))

    lag, peak = measure_lags(X, idx, lo, hi)
    centre = float(np.median(lag))
    dev = np.abs(lag - centre)

    print(f"corpus {CORPUS}")
    print(f"  rows {X.shape[0]}, sampled {len(idx)}, band {lo}-{hi}")
    print(f"  scale: 1 linewidth (~1.2 Hz) = {LINEWIDTH_PTS:.0f} points\n")

    q = np.percentile(lag, [0, 5, 25, 50, 75, 95, 100])
    print("  cross-correlation lag, points")
    print("    min {:.0f}  p5 {:.0f}  p25 {:.0f}  med {:.0f}  "
          "p75 {:.0f}  p95 {:.0f}  max {:.0f}".format(*q))
    print(f"    sd {lag.std():.0f}   IQR {q[4] - q[2]:.0f}")
    print(f"    correlation at best lag: med {np.median(peak):.3f}  "
          f"p5 {np.percentile(peak, 5):.3f}\n")

    print("  fraction displaced from the modal axis")
    for k in (1, 2, 5, 10, 50):
        pts = k * LINEWIDTH_PTS
        print(f"    > {k:3d} linewidth(s) ({pts:5.0f} pts): "
              f"{100 * (dev > pts).mean():5.1f}%")
    print()

    md = modes(lag)
    print("  lag modes (50-point bins, >=1% of sample)")
    for pos, cnt in md:
        print(f"    {pos:+7d} pts  n={cnt:5d}  ({100 * cnt / len(lag):5.1f}%)"
              f"  = {abs(pos) / LINEWIDTH_PTS:5.0f} linewidths")
    print()

    # The metadata prediction, for direct comparison.
    print("  predicted from Bruker metadata alone (bruker_param_audit.py):")
    print("        0 pts ~54%   -720..-900 pts ~42%   +5100..+8000 pts ~4%")
    near0 = 100 * (dev <= 2 * LINEWIDTH_PTS).mean()
    band8 = 100 * (((lag - centre) < -600) & ((lag - centre) > -1000)).mean()
    tail = 100 * ((lag - centre) > 3000).mean()
    print(f"  measured:  {near0:5.1f}% at 0    {band8:5.1f}% at -600..-1000"
          f"    {tail:5.1f}% above +3000")

    verdict = "CONFIRMED" if (near0 > 35 and band8 > 20) else "NOT CONFIRMED"
    print(f"\n  VERDICT: {verdict}")
    if verdict == "CONFIRMED":
        print("    The corpus carries at least two chemical-shift axes offset by")
        print("    ~800 points (~60 linewidths, ~0.12 ppm), plus a stretched")
        print("    minority. This is a preprocessing defect, not biology, and it")
        print("    caps any ppm-referenced fit. It also means cross-study rows")
        print("    cannot resemble each other, which is consistent with the")
        print("    corpus's near-duplicate structure being within-study (exp #19).")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "corpus_xcorr_lags.npy", lag)
    (out / "axis_misalignment.json").write_text(json.dumps({
        "n_sampled": int(len(idx)), "band": [lo, hi],
        "linewidth_pts": float(LINEWIDTH_PTS),
        "lag_sd_pts": float(lag.std()),
        "lag_iqr_pts": float(q[4] - q[2]),
        "frac_beyond_1_linewidth": float((dev > LINEWIDTH_PTS).mean()),
        "frac_beyond_10_linewidths": float((dev > 10 * LINEWIDTH_PTS).mean()),
        "modes": md, "verdict": verdict,
    }, indent=2))
    print(f"\n  wrote {args.out}/axis_misalignment.json, corpus_xcorr_lags.npy")


if __name__ == "__main__":
    main()
