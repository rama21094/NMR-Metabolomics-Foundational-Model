#!/usr/bin/env python3
"""Whole-corpus alignment QC: measure every spectrum, then look at all of them at once.

The PI's request is to plot original vs aligned and verify visually. Verifying
9,670 spectra one at a time is neither feasible (at 5 s each it is ~13 hours of
undivided attention) nor reliable -- the defect we are looking for is a uniform
translation, and the human eye is poor at judging absolute position but excellent
at judging alignment of things placed side by side. So this script does the part
a machine does better, and sets up the part an eye does better:

  1. MEASURE every spectrum's displacement against a common reference, by
     cross-correlation. One number per spectrum, all 9,670 of them.
  2. RENDER the whole corpus as a single image (a spectral heatmap), where one
     row of pixels is one spectrum. Misalignment shows up as a visible step --
     9,670 spectra genuinely fit on one page this way, which a stack of line
     plots does not.
  3. OVERLAY the group medians in a landmark region, which is the Bruker/Sparky
     view, but of 11 traces instead of 9,670.
  4. FLAG the outliers for individual inspection in the existing Streamlit
     viewer (code/analysis/spectra_viewer_app.py), so human attention is spent
     on the few hundred spectra that actually need a decision.

Step 1 is the deliverable that matters: it turns "verify the corpus" from an
opinion into a number per spectrum that can be re-run after any fix.

Usage:
    python code/analysis/alignment_qc.py                 # measure + plot
    python code/analysis/alignment_qc.py --skip-measure  # re-plot from cache
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ("data/combined/combine_unique_MetaboLights_Workbench_"
          "Water_EDTA_Suppressed.npy")
OUT = ROOT / "results/analysis/alignment_qc"
FIGS = ROOT / "results/figures"

WATER = (62500, 68000)
BAND = (72000, 84000)          # aliphatic, clear of the zeroed water window
PPM_HI, PPM_LO = 14.82, -5.20  # majority axis (MTBLS798)
NPT = 131_072
PPM_PER_PT = (PPM_HI - PPM_LO) / NPT
# MEASURED, not assumed. An earlier version used an idealised 1.2 Hz line
# (0.002 ppm, ~13 points). Fitting FWHM to metabolite peaks in the aliphatic
# region across 300 spectra gives a median of 35 points = 0.0054 ppm = 3.2 Hz at
# 600 MHz, which is a realistic serum CPMG linewidth. Everything expressed "in
# linewidths" was therefore overstated by a factor of 2.7; an 800-point
# displacement is ~23 linewidths, not ~62. Still far more than enough for peaks
# to miss each other entirely, but the corrected figure is the one to quote.
LINEWIDTH_PTS = 35.0

INK, ACC, WARN, MUTED = "#1f2933", "#2f6f8f", "#b04a3a", "#6b7480"


def ppm_of(idx):
    return PPM_HI - np.asarray(idx) * PPM_PER_PT


def normed_band(X, rows, lo, hi):
    B = np.array(X[rows, lo:hi], dtype=np.float64, copy=True)
    B -= B.mean(axis=1, keepdims=True)
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
    return B


def measure_all(X, lo, hi, chunk=400, seed=0):
    """Cross-correlation lag of every spectrum against a robust corpus reference.

    The reference is the median of a random subsample -- using the median of ALL
    rows would be circular only in a mild sense, but a subsample is cheaper and
    makes the reference reproducible from the seed alone.
    """
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    ref_rows = np.sort(rng.choice(n, min(800, n), replace=False))
    ref = np.median(normed_band(X, ref_rows, lo, hi), axis=0)
    ref -= ref.mean()
    ref /= np.linalg.norm(ref)

    m = hi - lo
    nfft = 1 << (2 * m - 1).bit_length()
    Fr = np.fft.rfft(ref[::-1], nfft)

    lag = np.empty(n, dtype=np.int64)
    peak = np.empty(n, dtype=np.float64)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        B = normed_band(X, np.arange(i, j), lo, hi)
        cc = np.fft.irfft(np.fft.rfft(B, nfft, axis=1) * Fr, nfft,
                          axis=1)[:, :2 * m - 1]
        lag[i:j] = cc.argmax(axis=1) - (m - 1)
        peak[i:j] = cc.max(axis=1)
        print(f"\r  measured {j}/{n}", end="", flush=True)
    print()
    return lag, peak


def fig_heatmap(X, lag, lo, hi, path):
    """All spectra as one image, sorted by measured displacement.

    Every one of the N spectra enters the image -- no rows are dropped. Note the
    distinction the title now makes explicitly: the raster is ~2,400 pixels tall,
    so roughly four spectra share a pixel row and the renderer averages them.
    That is fine for seeing WHERE the population sits, and it is not the same as
    "one pixel per spectrum". An earlier version of this figure both subsampled
    to 2,400 rows AND claimed one row per spectrum; the subsampling is gone and
    the claim is corrected.

    Columns are subsampled too, which is harmless here because we are looking at
    peak POSITION, not lineshape.
    """
    n = X.shape[0]
    order = np.argsort(lag)
    cols = np.linspace(lo, hi - 1, 1600).astype(int)

    # Row-chunked to bound memory. Fancy-indexing a memmap needs ascending
    # row order, so each chunk is read sorted and then put back into lag order.
    M = np.empty((n, len(cols)), dtype=np.float32)
    blk = 500
    for i in range(0, n, blk):
        j = min(i + blk, n)
        rows = order[i:j]
        srt = np.argsort(rows)
        M[i:j] = np.array(X[rows[srt]][:, cols], dtype=np.float32)[np.argsort(srt)]

    M -= np.median(M, axis=1, keepdims=True)
    sc = np.percentile(np.abs(M), 99.5, axis=1, keepdims=True) + 1e-12
    M = np.clip(M / sc, 0, 1)
    lag_sorted = lag[order]

    h = max(7.0, n / 620.0)                        # ~1 pixel row per spectrum
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(13, h), gridspec_kw={"width_ratios": [4.2, 1]})
    a.imshow(M, aspect="auto", cmap="magma_r", norm=PowerNorm(0.45),
             extent=[ppm_of(cols[0]), ppm_of(cols[-1]), n, 0],
             interpolation="nearest")
    a.set_xlabel("chemical shift (ppm)")
    a.set_ylabel("spectra, sorted by measured displacement")
    a.set_title(f"All {n:,} spectra, none omitted "
                f"(~{n / 2400:.0f} per pixel row)\n"
                "vertical ridges = the same metabolite; a sideways step = misalignment",
                fontsize=11, loc="left")
    a.invert_xaxis()

    b.plot(lag_sorted, np.arange(n), lw=1.2, color=ACC)
    b.axvline(0, color=INK, lw=1, ls="--")
    b.axvspan(-LINEWIDTH_PTS, LINEWIDTH_PTS, color=INK, alpha=0.20)
    b.set_ylim(n, 0)
    b.set_xlabel("displacement (points)")
    b.set_title(f"1 linewidth = {LINEWIDTH_PTS:.0f} pts (measured)", fontsize=10,
                loc="left", color=MUTED)
    b.set_yticks([])
    for sp in ("top", "right"):
        b.spines[sp].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}  ({n} rows, {h:.1f} in tall)")


def fig_overlay(X, lag, path, centre=78000, half=3000, n_per=60, seed=0):
    """Bruker/Sparky overlay of displacement groups, before and after correction.

    The bins below PARTITION the corpus -- their counts sum to n, and the legend
    states every count including the ones a translation cannot fix. An earlier
    version plotted only two bins and silently omitted 846 rows; a figure that
    does not account for its own population invites exactly that mistake.

    The stretched groups are drawn dashed and are NOT shift-corrected in the
    lower panel, because a translation cannot correct a scale difference.
    """
    rng = np.random.default_rng(seed)
    bins = [(-400, 400, "aligned (+/-400 pts)", INK, False),
            (-1500, -400, "-1500..-400", ACC, False),
            (400, 1500, "+400..+1500", "#7a5195", False),
            (-10 ** 9, -1500, "< -1500 (stretched)", WARN, True),
            (1500, 10 ** 9, "> +1500 (stretched)", "#ef7a3a", True)]

    lo, hi = centre - half, centre + half
    x = ppm_of(np.arange(lo, hi))
    n = len(lag)

    fig, (a, b) = plt.subplots(2, 1, figsize=(12.5, 7.0), sharex=True)
    counted = 0
    for e0, e1, lab, c, stretched in bins:
        idx = np.where((lag >= e0) & (lag < e1))[0]
        counted += len(idx)
        if len(idx) < 3:
            a.plot([], [], color=c, lw=1.4,
                   label=f"{lab}  (n={len(idx):,}, too few to plot)")
            continue
        pick = np.sort(rng.choice(idx, min(n_per, len(idx)), replace=False))
        ls = "--" if stretched else "-"

        raw = np.array(X[pick, lo:hi], dtype=np.float64)
        raw /= raw.max(axis=1, keepdims=True) + 1e-12
        a.plot(x, np.median(raw, axis=0), lw=1.4, color=c, ls=ls,
               label=f"{lab}  (n={len(idx):,}, median {int(np.median(lag[idx])):+d} pts)")

        if stretched:
            b.plot(x, np.median(raw, axis=0), lw=1.4, color=c, ls=ls)
            continue
        sh = int(np.median(lag[idx]))
        cor = np.array(X[pick, lo + sh:hi + sh], dtype=np.float64)
        cor /= cor.max(axis=1, keepdims=True) + 1e-12
        b.plot(x, np.median(cor, axis=0), lw=1.4, color=c)

    assert counted == n, f"bins cover {counted} of {n} rows"

    a.set_title(f"AS STORED — every one of the {n:,} spectra falls in one of "
                "these bins", fontsize=11, loc="left", color=WARN)
    a.legend(fontsize=8.5, frameon=False, ncol=2)
    b.set_title("AFTER each group's median shift. Dashed = stretched studies, "
                "shown uncorrected: a translation cannot fix a scale difference.",
                fontsize=11, loc="left", color=ACC)
    b.set_xlabel("chemical shift (ppm)")
    for ax in (a, b):
        ax.invert_xaxis()
        ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS,
                    help="corpus .npy to audit; point this at the re-referenced "
                         "file to verify the fix")
    ap.add_argument("--skip-measure", action="store_true")
    ap.add_argument("--flag-beyond", type=float, default=2.0,
                    help="flag spectra beyond this many linewidths for review")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    X = np.load(ROOT / args.corpus, mmap_mode="r")
    tag = "" if args.corpus == CORPUS else "_" + Path(args.corpus).stem[-12:]

    lag_p = OUT / f"lag_all{tag}.npy"
    peak_p = OUT / f"corr_all{tag}.npy"
    if args.skip_measure and lag_p.exists():
        lag, peak = np.load(lag_p), np.load(peak_p)
    else:
        print(f"measuring all {X.shape[0]:,} spectra "
              f"(band {BAND[0]}-{BAND[1]})")
        lag, peak = measure_all(X, *BAND)
        np.save(lag_p, lag)
        np.save(peak_p, peak)

    dev = np.abs(lag - np.median(lag))
    flag = dev > args.flag_beyond * LINEWIDTH_PTS
    np.save(OUT / f"flagged_rows{tag}.npy", np.where(flag)[0])

    print(f"\ndisplacement, all {len(lag):,} spectra")
    q = np.percentile(lag, [0, 5, 25, 50, 75, 95, 100])
    print("  min {:.0f}  p5 {:.0f}  p25 {:.0f}  med {:.0f}  p75 {:.0f} "
          " p95 {:.0f}  max {:.0f}".format(*q))
    for k in (1, 2, 10, 50):
        print(f"  beyond {k:3d} linewidth(s): "
              f"{100 * (dev > k * LINEWIDTH_PTS).mean():5.1f}%")
    print(f"\n  flagged for human review: {flag.sum():,} of {len(lag):,} "
          f"({100 * flag.mean():.1f}%)")
    print(f"  low-confidence correlations (<0.5): {(peak < 0.5).sum():,}")

    (OUT / f"summary{tag}.json").write_text(json.dumps({
        "n": int(len(lag)), "band": list(BAND),
        "linewidth_pts": float(LINEWIDTH_PTS),
        "median_lag": float(np.median(lag)), "sd_lag": float(lag.std()),
        "frac_beyond_1_lw": float((dev > LINEWIDTH_PTS).mean()),
        "n_flagged": int(flag.sum()),
    }, indent=2))

    fig_heatmap(X, lag, *BAND, path=FIGS / f"fig_alignment_heatmap{tag}.png")
    fig_overlay(X, lag, FIGS / f"fig_alignment_overlay{tag}.png")
    print(f"\nper-spectrum lags: {lag_p.relative_to(ROOT)}")
    print("inspect flagged rows with:  streamlit run code/analysis/spectra_viewer_app.py")


if __name__ == "__main__":
    main()
