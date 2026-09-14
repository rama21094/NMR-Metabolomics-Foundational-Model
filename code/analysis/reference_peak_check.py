#!/usr/bin/env python3
"""Is there a 0 ppm reference peak in this corpus, and did the corpus align on it?

The PI's question, made testable. The alignment in
code/preprocessing/alignSpectra.py anchored each spectrum on its rightmost
detectable feature, on the assumption that this is the 0 ppm reference resonance
(DSS or TSP) that samples normally carry. This script looks at three places and
reports what is actually there:

  A. THE 0 PPM WINDOW. If a reference standard were present, every spectrum would
     show a sharp singlet here and the corpus would be tightly aligned on it.
  B. THE FAR RIGHT EDGE. Whatever the pipeline actually anchored on. If the
     assumption failed, this is where we see what it locked onto instead.
  C. A REAL METABOLITE REGION, as a positive control -- a place where we KNOW
     there is signal, so the plots can be read against something.

RESULT (2026-09-14), and it overturns what this project previously believed:

  - A reference singlet IS present, in 97.5% of spectra, at a median 8.4% of each
    spectrum's own global maximum. An earlier reading of normaliser_comparison.py
    put this at 0.67% and concluded "no usable reference peak"; that figure came
    from sampling one fixed index rather than searching a window, and on a corpus
    that is itself misaligned the fixed index mostly missed the peak. The
    conclusion drawn from it was wrong.
  - Its position is BIMODAL: 0.00 ppm in 49.1% of spectra and +0.11 to +0.13 ppm
    in ~35%. That gap is the same ~800 points measured by alignment_qc.py, now
    visible on a single sharp landmark instead of a whole-spectrum correlation.
  - The far right edge, which alignSpectra.py anchored on, holds a median 0.03%
    of the global maximum. It is noise. The pipeline aligned the corpus on noise
    while a perfectly good reference sat 5 ppm away.

The practical consequence is large: the corpus can be re-aligned from the spectra
themselves, by locating this singlet per spectrum, with no instrument files and
no rebuild from the raw archive.

The 0 ppm index differs per source study, because each recorded a slightly
different window; both candidate positions are marked.

Usage:
    python code/analysis/reference_peak_check.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ("data/combined/combine_unique_MetaboLights_Workbench_"
          "Water_EDTA_Suppressed.npy")
QC = ROOT / "results/analysis/alignment_qc"
FIGS = ROOT / "results/figures"

NPT = 131_072
SW_PPM = 20.024
PPM_PER_PT = SW_PPM / NPT
INK, ACC, WARN, MUTED = "#1f2933", "#2f6f8f", "#b04a3a", "#6b7480"

# 0 ppm lands at OFFSET/ppm_per_pt, and the two dominant studies disagree.
ZERO_798 = int(round(14.8237 / PPM_PER_PT))      # MTBLS798, 54% of corpus
ZERO_147 = int(round(14.7056 / PPM_PER_PT))      # the -800 group

PANELS = [
    ("A  the 0 ppm window — where a DSS/TSP reference would be",
     ZERO_798 - 3500, ZERO_798 + 3500),
    ("B  the far right edge — what the pipeline actually anchored on",
     NPT - 7000, NPT),
    ("C  positive control: a real metabolite region (~2.0–3.0 ppm)",
     76000, 83000),
]


def ppm_of(i):
    return 14.8237 - np.asarray(i) * PPM_PER_PT


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    X = np.load(ROOT / CORPUS, mmap_mode="r")
    n = X.shape[0]
    lag = np.load(QC / "lag_all.npy")
    order = np.argsort(lag)

    # ---- quantify before plotting -----------------------------------------
    print(f"corpus {n:,} spectra\n")
    gmax = np.empty(n)
    zmax = np.empty(n)
    emax = np.empty(n)
    zwin = slice(ZERO_798 - 600, ZERO_798 + 600)       # ±0.09 ppm around 0
    ewin = slice(NPT - 3000, NPT)
    for i in range(0, n, 500):
        j = min(i + 500, n)
        B = np.array(X[i:j], dtype=np.float64)
        B[:, 62500:68000] = 0.0
        gmax[i:j] = np.abs(B).max(axis=1)
        zmax[i:j] = np.abs(B[:, zwin]).max(axis=1)
        emax[i:j] = np.abs(B[:, ewin]).max(axis=1)
    rz, re = zmax / (gmax + 1e-30), emax / (gmax + 1e-30)
    print("peak height as a fraction of each spectrum's own global maximum")
    print(f"  0 ppm window (±0.09 ppm): median {np.median(rz):.4f}  "
          f"p90 {np.percentile(rz, 90):.4f}  "
          f"fraction above 0.05: {100 * (rz > 0.05).mean():.1f}%")
    print(f"  far right edge (last 3k pts): median {np.median(re):.4f}  "
          f"p90 {np.percentile(re, 90):.4f}  "
          f"fraction above 0.05: {100 * (re > 0.05).mean():.1f}%")
    # ---- where is each spectrum's sharpest peak near 0 ppm? ---------------
    # This is the question: if a reference standard is present AND the corpus
    # was aligned on it, every spectrum's peak sits at the same index.
    search = slice(ZERO_798 - 2600, ZERO_798 + 2600)     # +/- 0.4 ppm
    pos = np.empty(n, dtype=np.int64)
    hgt = np.empty(n)
    for i in range(0, n, 500):
        j = min(i + 500, n)
        W = np.array(X[i:j, search], dtype=np.float64)
        W -= np.median(W, axis=1, keepdims=True)
        pos[i:j] = W.argmax(axis=1) + search.start
        hgt[i:j] = W.max(axis=1) / (gmax[i:j] + 1e-30)
    ppm_pos = ppm_of(pos)
    strong = hgt > 0.02
    print(f"\n  sharpest feature within +/-0.4 ppm of 0, per spectrum:")
    print(f"    detectable (height > 2% of global max): "
          f"{100 * strong.mean():.1f}% of spectra")
    q = np.percentile(ppm_pos[strong], [5, 25, 50, 75, 95])
    print("    position ppm: p5 {:.4f}  p25 {:.4f}  med {:.4f}  p75 {:.4f} "
          " p95 {:.4f}".format(*q))
    print(f"    spread p5-p95 = {q[4] - q[0]:.4f} ppm "
          f"= {(q[4] - q[0]) / PPM_PER_PT:.0f} points")
    # modes
    b = np.round(ppm_pos[strong] / 0.01) * 0.01
    u, c = np.unique(b, return_counts=True)
    o = np.argsort(-c)[:5]
    print("    position modes (0.01 ppm bins):")
    for t in o:
        print(f"      {u[t]:+.2f} ppm   n={c[t]:5d}  ({100 * c[t] / strong.sum():5.1f}%)")
    np.save(QC / "zero_ppm_peak_index.npy", pos)

    print("\n  For reference, a DSS/TSP standard is normally set to be one of the "
          "\n  tallest features in the spectrum -- a ratio near 1, not near 0.08.")

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9),
                             gridspec_kw={"height_ratios": [2.6, 1]})
    for k, (title, lo, hi) in enumerate(PANELS):
        cols = np.linspace(lo, hi - 1, 1100).astype(int)
        M = np.empty((n, len(cols)), dtype=np.float32)
        blk = 500
        for i in range(0, n, blk):
            j = min(i + blk, n)
            rows = order[i:j]
            srt = np.argsort(rows)
            M[i:j] = np.array(X[rows[srt]][:, cols],
                              dtype=np.float32)[np.argsort(srt)]
        M -= np.median(M, axis=1, keepdims=True)
        # scale by each spectrum's GLOBAL max, not the window max, so an empty
        # window renders as empty instead of being stretched into fake contrast
        M = np.clip(M / (gmax[order][:, None] + 1e-30), 0, 1)

        a = axes[0, k]
        a.imshow(M, aspect="auto", cmap="magma_r", norm=PowerNorm(0.35),
                 extent=[ppm_of(cols[0]), ppm_of(cols[-1]), n, 0],
                 interpolation="nearest")
        # extent already runs high -> low ppm left to right, which IS the NMR
        # convention. An invert_xaxis() here (as in the first version) mirrors
        # the heatmaps against the line plots below them.
        a.set_title(title, fontsize=10.5, loc="left",
                    color=WARN if k < 2 else INK, pad=18)
        if k == 0:
            a.set_ylabel(f"all {n:,} spectra, sorted by displacement")
            for z, lab, c in ((ZERO_798, "0 ppm (MTBLS798 axis)", ACC),
                              (ZERO_147, "0 ppm (−800 group axis)", WARN)):
                a.axvline(ppm_of(z), color=c, lw=1.1, ls="--")
                a.text(ppm_of(z), -n * 0.02, lab, color=c, fontsize=8,
                       ha="center", va="bottom")

        b = axes[1, k]
        med = np.median(M, axis=0)
        p90 = np.percentile(M, 90, axis=0)
        b.fill_between(ppm_of(cols), 0, p90, color=ACC, alpha=0.25,
                       label="90th percentile")
        b.plot(ppm_of(cols), med, lw=1.2, color=INK, label="median")
        b.set_xlabel("chemical shift (ppm)")
        b.invert_xaxis()
        b.set_ylim(0, max(0.05, float(p90.max()) * 1.15))
        if k == 0:
            b.set_ylabel("fraction of\nglobal maximum")
            b.legend(fontsize=8, frameon=False)
        for sp in ("top", "right"):
            b.spines[sp].set_visible(False)

    fig.suptitle("There IS a 0 ppm reference peak — in 97.5% of spectra. The "
                 "pipeline anchored on the far right edge, which is pure noise.",
                 fontsize=13, y=0.995, color=WARN)
    fig.tight_layout()
    p = FIGS / "fig_reference_peak_check.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
