#!/usr/bin/env python3
"""Look at the spectra still 2-10 linewidths off after 0 ppm re-referencing.

After code/preprocessing/rereference_zero_ppm.py, ~10% of spectra sit between 2
and 10 measured linewidths (70-350 points) from the corpus consensus. Two
readings are possible and they have opposite consequences:

  (a) RESIDUAL MISALIGNMENT -- the referencing is imperfect for these rows and
      should be improved. They would then be a defect still to fix.
  (b) REAL CHEMICAL SHIFT VARIATION -- pH, ionic strength and protein binding
      move each metabolite by a DIFFERENT amount, so no single translation can
      align them. They would then be legitimate biological variants and should
      be kept, and the residual is the thing per-metabolite shift fitting exists
      to model.

The two are distinguishable without any human judgement. A rigid misalignment
displaces EVERY region of the spectrum by the SAME amount; per-metabolite shift
displaces DIFFERENT regions by DIFFERENT amounts. So: measure the displacement
independently in several well-separated windows and look at the spread across
windows within each spectrum.

    small spread across windows  -> rigid  -> reading (a), fixable
    large spread across windows  -> per-metabolite -> reading (b), keep them

The script reports that statistic for the residual group, for the aligned group
as a control, and renders a panel of individual spectra overlaid on the corpus
median so the eye can be applied to the same question.

Usage:
    python code/analysis/inspect_residual_shifts.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REF = ("data/combined/combine_unique_MetaboLights_Workbench_"
       "Water_EDTA_Suppressed_ref0ppm.npy")
LAGS = "results/analysis/alignment_qc/lag_all_ssed_ref0ppm.npy"
FIGS = ROOT / "results/figures"

NPT = 131_072
PPM_HI, SW = 14.8237, 20.024
PPM_PER_PT = SW / NPT
LINEWIDTH_PTS = 35.0          # measured, see alignment_qc.py

# Well-separated windows, each holding real serum signal, none in the water gap.
WINDOWS = [(72000, 76000), (76000, 80000), (80000, 84000),
           (86000, 90000), (92000, 96000)]

INK, ACC, WARN, MUTED = "#1f2933", "#2f6f8f", "#b04a3a", "#6b7480"


def ppm_of(i):
    return PPM_HI - np.asarray(i) * PPM_PER_PT


def lag_in_window(B, ref, lo, hi):
    """Cross-correlation lag of each row against `ref`, restricted to a window."""
    Y = B[:, lo:hi].astype(np.float64)
    Y = Y - Y.mean(axis=1, keepdims=True)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12
    r = ref[lo:hi] - ref[lo:hi].mean()
    r = r / (np.linalg.norm(r) + 1e-12)
    m = hi - lo
    nfft = 1 << (2 * m - 1).bit_length()
    cc = np.fft.irfft(np.fft.rfft(Y, nfft, axis=1) * np.fft.rfft(r[::-1], nfft),
                      nfft, axis=1)[:, :2 * m - 1]
    return cc.argmax(axis=1) - (m - 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-show", type=int, default=8)
    ap.add_argument("--n-test", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X = np.load(ROOT / REF, mmap_mode="r")
    lag = np.load(ROOT / LAGS)
    dev = np.abs(lag - np.median(lag))

    resid = np.where((dev > 2 * LINEWIDTH_PTS) & (dev <= 10 * LINEWIDTH_PTS))[0]
    aligned = np.where(dev <= 2 * LINEWIDTH_PTS)[0]
    print(f"residual group (2-10 linewidths, 70-350 pts): {len(resid):,}"
          f"  ({100 * len(resid) / len(lag):.1f}%)")
    print(f"aligned control (<=2 linewidths):             {len(aligned):,}\n")

    rng = np.random.default_rng(args.seed)
    ref = np.median(np.array(X[np.sort(rng.choice(aligned, 400, replace=False))],
                             dtype=np.float64), axis=0)

    # ---- the discriminating test ------------------------------------------
    print("displacement measured independently in 5 separate windows")
    print("  spread ACROSS windows, within each spectrum (sd, points):")
    out = {}
    for name, pool in (("aligned control", aligned), ("residual group", resid)):
        pick = np.sort(rng.choice(pool, min(args.n_test, len(pool)),
                                  replace=False))
        B = np.array(X[pick], dtype=np.float64)
        L = np.stack([lag_in_window(B, ref, lo, hi) for lo, hi in WINDOWS])
        spread = L.std(axis=0)
        out[name] = (L, spread, pick)
        print(f"    {name:18s} median {np.median(spread):6.1f}   "
              f"p25 {np.percentile(spread, 25):6.1f}   "
              f"p75 {np.percentile(spread, 75):6.1f}")
    sa = np.median(out["aligned control"][1])
    sr = np.median(out["residual group"][1])
    print(f"\n  ratio residual/aligned = {sr / max(sa, 1e-9):.2f}")
    if sr > 2 * sa and sr > LINEWIDTH_PTS:
        print("  -> the residual group shifts DIFFERENTLY in different regions.")
        print("     That is per-metabolite behaviour, not leftover misalignment:")
        print("     no single translation can align them, and they are most")
        print("     likely genuine chemical variation worth KEEPING.")
    elif sr <= 2 * sa:
        print("  -> the residual group shifts RIGIDLY, like the aligned control.")
        print("     That points to leftover referencing error, which is fixable")
        print("     and should be fixed rather than kept.")
    else:
        print("  -> intermediate; both effects are probably present.")

    # ---- the visual panel --------------------------------------------------
    show = np.sort(rng.choice(resid, args.n_show, replace=False))
    lo, hi = 76000, 82000
    x = ppm_of(np.arange(lo, hi))
    rn = ref[lo:hi] / (ref[lo:hi].max() + 1e-12)

    fig, axes = plt.subplots(args.n_show // 2, 2,
                             figsize=(14, 1.65 * args.n_show), sharex=True)
    for ax, row in zip(axes.ravel(), show):
        y = np.array(X[row, lo:hi], dtype=np.float64)
        y = y / (y.max() + 1e-12)
        ax.plot(x, rn, lw=1.0, color=MUTED, label="corpus median")
        ax.plot(x, y, lw=1.0, color=ACC, label=f"row {row}")
        ax.set_title(f"row {row}   global displacement "
                     f"{int(lag[row] - np.median(lag)):+d} pts "
                     f"({(lag[row] - np.median(lag)) / LINEWIDTH_PTS:+.1f} linewidths)",
                     fontsize=9, loc="left", color=INK)
        ax.set_yticks([])
        ax.invert_xaxis()
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    axes.ravel()[0].legend(fontsize=8, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("chemical shift (ppm)")
    fig.suptitle("Spectra 2-10 linewidths from consensus, overlaid on the "
                 "corpus median — are these misaligned, or different?",
                 fontsize=12, y=0.999)
    fig.tight_layout()
    p = FIGS / "fig_residual_shift_inspection.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
