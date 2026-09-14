#!/usr/bin/env python3
"""What separates the spectra the forward model CAN reproduce from those it cannot?

At n=60 with per-metabolite shift fitting the gate reaches a median R2 of 0.668,
but the spread is wide: 3 spectra clear 0.90 and 10 clear 0.75, while the bulk sit
near 0.65. A flat ceiling would mean one missing ingredient; a wide spread means
whatever limits the model affects some spectra much more than others, which is a
far more tractable question.

Candidate explanations, each measurable directly from the spectrum:

  LIPID / MACROMOLECULE CONTENT. The envelope block has 7 empirical components
    against 87 metabolites, yet explains comparable variance. If R2 falls as lipid
    content rises, the envelope -- not the metabolite panel -- is the weak part.
  SIGNAL-TO-NOISE. A noisy spectrum cannot be fitted well by anything; if this
    dominates, the ceiling is partly measurement quality and not model error.
  LINEWIDTH. Broad lines blur multiplets together, and GISSMO simulates narrow
    ones. Broadening is applied globally, so spectra far from that value fit worse.
  TOTAL FITTED SHIFT. If the well-fitting spectra are the ones needing least
    shifting, the shift model is patching a problem rather than describing one.
  ATYPICALITY. Do the spectra that fit well just look like the corpus median?
  RESIDUAL DISPLACEMENT. Leftover misalignment after re-referencing.

Correlation is computed for each, and a small multivariate fit reports how much of
the R2 spread the whole set accounts for -- so "we found a correlate" is not
confused with "we explained the spread".

Usage:
    python code/analysis/why_some_spectra_fit.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ("data/combined/combine_unique_MetaboLights_Workbench_"
          "Water_EDTA_Suppressed_ref0ppm_clean.npy")
FIT = "results/synthesis/gate_n60_shift05"
LAGS = "results/analysis/alignment_qc/lag_all_ssed_ref0ppm.npy"
ROWMAP = "results/analysis/atypical/row_map_clean.csv"
SCORES = "results/analysis/atypical/atypical_scores.csv"
FIGS = ROOT / "results/figures"

PPM_HI, SW = 14.8237, 20.024
PPM_PER_PT = SW / 131072
INK, ACC, WARN, MUTED = "#1f2933", "#2f6f8f", "#b04a3a", "#6b7480"


def idx(ppm):
    return int(round((PPM_HI - ppm) / PPM_PER_PT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-dir", default=FIT)
    args = ap.parse_args()

    fit = pd.read_csv(ROOT / args.fit_dir / "fit_per_spectrum.csv")
    X = np.load(ROOT / CORPUS, mmap_mode="r")
    rows = fit["row"].to_numpy()
    r2 = fit["r2_full"].to_numpy()
    print(f"{len(rows)} fitted spectra; R2 median {np.median(r2):.3f}, "
          f"{int((r2 >= 0.90).sum())} at/above 0.90\n")

    B = np.array(X[rows], dtype=np.float64)
    B[:, 62500:68000] = 0.0
    gmax = np.abs(B).max(axis=1) + 1e-30

    # lipid CH2 (1.3 ppm) and CH3 (0.9 ppm) envelope, relative to total signal
    lip = (np.abs(B[:, idx(1.45):idx(1.15)]).sum(1)
           + np.abs(B[:, idx(1.02):idx(0.75)]).sum(1))
    total = np.abs(B[:, idx(9.5):idx(0.4)]).sum(1) + 1e-30
    lipid_frac = lip / total

    # noise from an empty region; signal from the tallest metabolite peak
    noise = B[:, idx(-1.0):idx(-2.0)].std(axis=1) + 1e-30
    snr = gmax / noise

    # linewidth: FWHM of each spectrum's tallest aliphatic peak
    from scipy.signal import find_peaks, peak_widths
    lw = np.full(len(rows), np.nan)
    seg = B[:, 74000:84000]
    for k in range(len(rows)):
        y = seg[k] - np.median(seg[k])
        if y.max() <= 0:
            continue
        pk, _ = find_peaks(y, height=0.3 * y.max(), distance=20)
        if len(pk):
            lw[k] = np.median(peak_widths(y, pk, rel_height=0.5)[0])

    feats = {"lipid_fraction": lipid_frac, "SNR": snr, "linewidth_pts": lw,
             "metabolite_signal_frac": fit["metabolite_signal_frac"].to_numpy(),
             "n_metabolites_used": fit["n_metabolites_used"].to_numpy(),
             "r2_envelope_only": fit["r2_envelope_only"].to_numpy()}

    sp = ROOT / args.fit_dir / "fitted_shifts_ppm.npy"
    if sp.exists():
        S = np.load(sp)
        feats["median_abs_shift_ppm"] = np.median(np.abs(S), axis=1)

    lag = np.load(ROOT / LAGS)
    m = pd.read_csv(ROOT / ROWMAP)
    old = m.set_index("new_row")["old_row"].to_dict()
    feats["residual_displacement"] = np.array(
        [abs(int(lag[old[r]]) - int(np.median(lag))) for r in rows])
    sc = pd.read_csv(ROOT / SCORES).set_index("row")
    feats["corr_to_corpus_median"] = np.array(
        [sc.loc[old[r], "corr_to_median"] for r in rows])

    print(f"{'feature':26s} {'Spearman rho':>13s} {'p':>10s}")
    recs = []
    for k, v in feats.items():
        ok = np.isfinite(v)
        rho, p = spearmanr(v[ok], r2[ok])
        recs.append((k, rho, p))
        star = "  <-- " if p < 0.01 else ""
        print(f"{k:26s} {rho:13.3f} {p:10.2e}{star}")

    # How much of the spread do they explain TOGETHER?
    M = np.column_stack([np.nan_to_num(v, nan=np.nanmedian(v))
                         for v in feats.values()])
    M = (M - M.mean(0)) / (M.std(0) + 1e-12)
    A = np.column_stack([M, np.ones(len(M))])
    beta, *_ = np.linalg.lstsq(A, r2, rcond=None)
    pred = A @ beta
    r2_of_r2 = 1 - ((r2 - pred) ** 2).sum() / ((r2 - r2.mean()) ** 2).sum()
    print(f"\nall {len(feats)} features together explain {100 * r2_of_r2:.0f}% "
          f"of the variance in fit quality")
    print(f"  (n={len(rows)} with {len(feats)} predictors, so this is optimistic; "
          "treat it as an upper bound)")

    best = max(recs, key=lambda t: abs(t[1]))
    print(f"\nstrongest single correlate: {best[0]}  rho={best[1]:.3f}")

    # ---- figure ------------------------------------------------------------
    show = sorted(recs, key=lambda t: -abs(t[1]))[:6]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, (k, rho, p) in zip(axes.ravel(), show):
        v = feats[k]
        ok = np.isfinite(v)
        col = np.where(r2[ok] >= 0.90, WARN, ACC)
        ax.scatter(v[ok], r2[ok], s=26, c=col, alpha=0.8, edgecolors="none")
        ax.axhline(0.90, color=INK, ls="--", lw=1)
        ax.set_xlabel(k, fontsize=9)
        ax.set_ylabel("R² of the fit", fontsize=9)
        ax.set_title(f"rho = {rho:+.2f}   p = {p:.1e}", fontsize=10, loc="left",
                     color=INK if p < 0.01 else MUTED)
        for spn in ("top", "right"):
            ax.spines[spn].set_visible(False)
    fig.suptitle("What separates the spectra the forward model can reproduce? "
                 "(red = reaches the 0.90 bar)", fontsize=12, y=1.0)
    fig.tight_layout()
    fp = FIGS / "fig_why_some_fit.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    print(f"\nwrote {fp}")


if __name__ == "__main__":
    main()
