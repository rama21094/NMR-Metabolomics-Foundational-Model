#!/usr/bin/env python3
"""Corpus and the four evaluation cohorts on one image, before and after alignment.

The whole-corpus heatmap (code/analysis/alignment_qc.py) made the corpus's
internal split visible at a glance. The same view across datasets answers the
question scalar overlap scores keep failing to settle: where do the cohorts sit
relative to the corpus, and to each other?

Each dataset is a band of pixel rows; within a band, rows are ordered by their
measured displacement. Vertical ridges are metabolites. A band whose ridges step
sideways relative to the corpus is misaligned against it.

Left  = as stored (the files every published result used).
Right = after the 0 ppm peak anchor.

Read the right panel sceptically. docs/PI_outline.md 5n establishes that the
corpus's near-zero feature is a 15 Hz hump and a 0.6 Hz digital spike of nearly
equal height, not a reference standard, so "aligned" here means aligned to that
feature -- internally consistent, but not absolute ppm. MTBLS326 and BrC-T2D do
have genuine standards (2.8 and 4.8 Hz) and this panel moves them onto the
corpus's arbitrary zero, which for MTBLS326 demonstrably makes agreement worse.

Usage:
    python code/plotting/plot_corpus_and_cohorts.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "results/figures"

NPT = 131_072
PPM_HI, SW = 14.8237, 20.024
PPM_PER_PT = SW / NPT
BAND = (70000, 92000)          # ~4.1 to 0.8 ppm, clear of the zeroed water gap
ZBAND = (95100, 98950)         # ~+0.30 to -0.29 ppm, the reference region
INK, ACC, WARN, MUTED = "#1f2933", "#2f6f8f", "#b04a3a", "#6b7480"

DATASETS = [
    ("corpus (9,623)", "data/combined/corpus_v4_ref0ppm_clean.npy",
     "data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v4.npy", 900),
    ("Barth (40)", "data/Barth/Barth_EDTASuppressed_v4_corpusaxis.npy",
     "data/Barth/aligned_128K_Workbench_Barth_Syndrome_WS625to680Zero_EDTASuppressed_v4.npy", None),
    ("MTBLS326 (42)", "data/mtbls326/MTBLS326_EDTASuppressed_v4_ref0ppm.npy",
     "data/mtbls326/MTBLS326_aligned_spectra_WS625to680Zero_EDTASuppressed_v4.npy", None),
    ("MTBLS563 (142)", "data/mtbls563/MTBLS563_EDTASuppressed_v4_ref0ppm.npy",
     "data/mtbls563/MTBLS563_aligned_spectra_WS625to680Zero_EDTASuppressed_v4.npy", None),
    ("BrC-T2D (78)", "data/BrC_T2D/BC_T2D_newlabels_EDTASuppressed_v4_ref0ppm.npy",
     "data/BrC_T2D/BC_T2D_corpusaxis_v4.npy", None),
    # TBI is excluded from the evaluation (no acquisition metadata, so its
    # spectral width is unknown and it cannot be put on the corpus axis). It was
    # also the largest cohort at 231 rows, crowding the bands that matter.
]


def ppm_of(i):
    return PPM_HI - np.asarray(i) * PPM_PER_PT


def pick_rows(path, cap, seed=0):
    n = np.load(ROOT / path, mmap_mode="r").shape[0]
    rows = np.arange(n)
    if cap and n > cap:
        rows = np.sort(np.random.default_rng(seed).choice(n, cap, replace=False))
    return rows


def load_block(path, cols, rows, order=None):
    """Rows of `path` at `cols`, normalised for display.

    `order` lets a second panel reuse the first panel's row order. The 0 ppm
    strip MUST do this: it is only readable next to the metabolite panel if row
    k is the same spectrum in both, and a strip ordered by its own displacement
    would silently pair different spectra side by side.
    """
    X = np.load(ROOT / path, mmap_mode="r")
    M = np.array(X[rows][:, cols], dtype=np.float32)
    if order is None:
        # order within the dataset by its own displacement, so a band that is
        # internally inconsistent shows as a slanting ridge rather than noise
        ref = np.median(M, axis=0)
        r = ref - ref.mean(); r /= np.linalg.norm(r) + 1e-12
        Z = M - M.mean(axis=1, keepdims=True)
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
        m = len(cols); nfft = 1 << (2 * m - 1).bit_length()
        cc = np.fft.irfft(np.fft.rfft(Z, nfft, axis=1) * np.fft.rfft(r[::-1], nfft),
                          nfft, axis=1)[:, :2 * m - 1]
        order = np.argsort(cc.argmax(axis=1) - (m - 1))
    M = M[order]
    M = M - np.median(M, axis=1, keepdims=True)
    sc = np.percentile(np.abs(M), 99.5, axis=1, keepdims=True) + 1e-12
    return np.clip(M / sc, 0, 1), order


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cols", type=int, default=1500)
    args = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)

    lo, hi = BAND
    cols = np.linspace(lo, hi - 1, args.n_cols).astype(int)

    zcols = np.linspace(ZBAND[0], ZBAND[1] - 1, 420).astype(int)

    panels, zpanel = {}, None
    for which, key in (("as stored", 2), ("after alignment", 1)):
        blocks, zblocks, labels, edges, y = [], [], [], [], 0
        for entry in DATASETS:
            name, cap = entry[0], entry[3]
            path = entry[key]
            if not (ROOT / path).exists():
                print(f"  missing, skipped: {path}")
                continue
            rows = pick_rows(path, cap)
            B, order = load_block(path, cols, rows)
            blocks.append(B); labels.append(name)
            if key == 1:      # the 0 ppm strip follows the aligned panel's order
                Zb, _ = load_block(path, zcols, rows, order=order)
                zblocks.append(Zb)
            y += len(B); edges.append(y)
        panels[which] = (np.vstack(blocks), labels, edges)
        if zblocks:
            zpanel = (np.vstack(zblocks), labels, edges)
        print(f"{which}: {panels[which][0].shape[0]} rows, {len(labels)} datasets")

    fig, axes = plt.subplots(1, 3, figsize=(23, 11), sharey=True,
                             gridspec_kw={"width_ratios": [4, 4, 1.7]})
    view = [("as stored", cols, panels["as stored"], INK),
            ("after alignment", cols, panels["after alignment"], WARN),
            ("0 ppm region, after alignment", zcols, zpanel, WARN)]
    for ax, (which, cc_, (M, labels, edges), col) in zip(axes, view):
        ax.imshow(M, aspect="auto", cmap="magma_r", norm=PowerNorm(0.45),
                  extent=[ppm_of(cc_[0]), ppm_of(cc_[-1]), M.shape[0], 0],
                  interpolation="nearest")
        starts = [0] + edges[:-1]
        for e in edges[:-1]:
            ax.axhline(e, color=ACC, lw=1.4)
        if which != "0 ppm region, after alignment":
            for s_, e, lab in zip(starts, edges, labels):
                ax.text(ppm_of(cc_[0]) - 0.03, (s_ + e) / 2, lab, fontsize=10,
                        ha="left", va="center", color=INK,
                        bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
        else:
            ax.axvline(0.0, color="#1a6b1a", lw=1.2, ls="--", zorder=5)
        ax.set_xlabel("chemical shift (ppm)")
        ax.set_title(which, fontsize=12, loc="left", color=col)
        ax.set_yticks([])
    axes[0].set_ylabel("corpus and the four evaluation cohorts, "
                       "one row per spectrum", labelpad=10)

    fig.suptitle("Corpus and the four evaluation cohorts on one axis — "
                 "a sideways step between bands is a misalignment between datasets",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    p = FIGS / "fig_corpus_and_cohorts.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
