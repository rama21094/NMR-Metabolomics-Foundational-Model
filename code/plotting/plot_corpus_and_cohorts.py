#!/usr/bin/env python3
"""Corpus and all five evaluation cohorts on one image, before and after alignment.

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
    ("TBI (231)", "data/tbi_tirupati/TBI_Tirupati_ref0ppm.npy",
     "data/tbi_tirupati/aligned_128K_TBI_Tirupati_WS625to680Zero.npy", None),
]


def ppm_of(i):
    return PPM_HI - np.asarray(i) * PPM_PER_PT


def load_block(path, cols, cap, seed=0):
    X = np.load(ROOT / path, mmap_mode="r")
    n = X.shape[0]
    rows = np.arange(n)
    if cap and n > cap:
        rows = np.sort(np.random.default_rng(seed).choice(n, cap, replace=False))
    M = np.array(X[rows][:, cols], dtype=np.float32)
    # order within the dataset by its own displacement, so a band that is
    # internally inconsistent shows as a slanting ridge rather than noise
    ref = np.median(M, axis=0)
    r = ref - ref.mean(); r /= np.linalg.norm(r) + 1e-12
    Z = M - M.mean(axis=1, keepdims=True)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    m = len(cols); nfft = 1 << (2 * m - 1).bit_length()
    cc = np.fft.irfft(np.fft.rfft(Z, nfft, axis=1) * np.fft.rfft(r[::-1], nfft),
                      nfft, axis=1)[:, :2 * m - 1]
    lag = cc.argmax(axis=1) - (m - 1)
    M = M[np.argsort(lag)]
    M = M - np.median(M, axis=1, keepdims=True)
    sc = np.percentile(np.abs(M), 99.5, axis=1, keepdims=True) + 1e-12
    return np.clip(M / sc, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cols", type=int, default=1500)
    args = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)

    lo, hi = BAND
    cols = np.linspace(lo, hi - 1, args.n_cols).astype(int)

    panels = {}
    for which, key in (("as stored", 2), ("after the 0 ppm anchor", 1)):
        blocks, labels, edges, y = [], [], [], 0
        for entry in DATASETS:
            name, cap = entry[0], entry[3]
            path = entry[key]
            if not (ROOT / path).exists():
                print(f"  missing, skipped: {path}")
                continue
            B = load_block(path, cols, cap)
            blocks.append(B); labels.append(name)
            y += len(B); edges.append(y)
        panels[which] = (np.vstack(blocks), labels, edges)
        print(f"{which}: {panels[which][0].shape[0]} rows, {len(labels)} datasets")

    fig, axes = plt.subplots(1, 2, figsize=(17, 11), sharey=True)
    for ax, (which, (M, labels, edges)) in zip(axes, panels.items()):
        ax.imshow(M, aspect="auto", cmap="magma_r", norm=PowerNorm(0.45),
                  extent=[ppm_of(cols[0]), ppm_of(cols[-1]), M.shape[0], 0],
                  interpolation="nearest")
        starts = [0] + edges[:-1]
        for e in edges[:-1]:
            ax.axhline(e, color=ACC, lw=1.4)
        for s, e, lab in zip(starts, edges, labels):
            # inside the panel: an outside label collides with the y-axis title
            ax.text(ppm_of(cols[0]) - 0.03, (s + e) / 2, lab, fontsize=10,
                    ha="left", va="center", color=INK,
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
        ax.set_xlabel("chemical shift (ppm)")
        ax.set_title(which, fontsize=12, loc="left",
                     color=INK if which == "as stored" else WARN)
        ax.set_yticks([])
    axes[0].set_ylabel("corpus and the five evaluation cohorts, "
                       "one row per spectrum", labelpad=10)

    fig.suptitle("Corpus and all five evaluation cohorts on one axis — "
                 "a sideways step between bands is a misalignment between datasets",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    p = FIGS / "fig_corpus_and_cohorts.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
