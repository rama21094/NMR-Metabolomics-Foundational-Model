#!/usr/bin/env python3
"""Plot real vs windowed-synthetic spectra (route b).

Two figures:
  fig_synth_overview.png -- full spectral range, one real spectrum and two
      quilted_base synthetics, vertically offset. Shows whether the synthetic
      output looks like a serum CPMG spectrum at all.
  fig_synth_methods.png  -- a zoom on the crowded 1.0-4.5 ppm region comparing
      the three methods against a real spectrum, so the seams that independent
      window sampling produces are visible rather than described.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/analysis/synthetic_windowed"
OUT = ROOT / "results/plots/synthetic_windowed"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 13,
    "axes.titlesize": 15, "axes.labelsize": 13.5,
    "figure.constrained_layout.use": True, "savefig.dpi": 170,
})

NAVY, CORAL, TEAL, GOLD = "#21295C", "#C1435B", "#1C7293", "#B8860B"


def load_real():
    import importlib.util
    sp = importlib.util.spec_from_file_location(
        "cn", ROOT / "code/analysis/corpus_normalisation.py")
    cn = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(cn)
    corpus = cn.open_corpus(
        ROOT / "data/combined/combine_unique_MetaboLights_Workbench_"
        "WaterFixed_EDTAMagnitude_v4.npy", "unit_area", verbose=False)
    rows = np.load(SRC / "donor_rows.npy")
    return np.asarray(corpus[rows[:6]], dtype=np.float64)


def main():
    ppm = np.load(ROOT / "data/mtbls326/MTBLS326_common_ppm_axis.npy")
    real = load_real()
    qb = np.load(SRC / "synthetic_quilted_base.npy").astype(np.float64)
    ql = np.load(SRC / "synthetic_quilted.npy").astype(np.float64)
    ind = np.load(SRC / "synthetic_independent.npy").astype(np.float64)

    # ---------------- figure 1: overview ----------------
    lo, hi = 0.5, 9.0
    m = (ppm >= lo) & (ppm <= hi)
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    series = [("Real spectrum (corpus)", real[0], NAVY),
              ("Synthetic 1 — quilted_base", qb[0], CORAL),
              ("Synthetic 2 — quilted_base", qb[3], TEAL)]
    for i, (lab, y, col) in enumerate(series):
        yy = y[m]
        off = i * 1.15 * np.percentile(np.abs(yy), 99.9)
        ax.plot(ppm[m], yy + off, lw=0.7, color=col)
        ax.text(hi - 0.12, off + 1.02 * np.percentile(np.abs(yy), 99.9), lab,
                color=col, fontweight="bold", fontsize=13.5, ha="left", va="bottom")
    ax.set_xlim(hi, lo)
    ax.set_xlabel("chemical shift (ppm)")
    ax.set_ylabel("intensity (unit-area normalised, offset)")
    ax.set_title("Windowed synthesis: real vs two synthetic serum CPMG spectra\n"
                 "window 2048 pts, hop 1024 (50% overlap), Hann crossfade",
                 fontweight="bold")
    ax.text(0.005, -0.175, "Looking plausible is NOT validation — the naive variant below also "
            "looks fine, yet destroys the population correlation structure. See validation.csv.",
            transform=ax.transAxes, fontsize=11.5, color="0.35", style="italic")
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.25, ls="--")
    ax.set_axisbelow(True)
    p = OUT / "fig_synth_overview.png"
    fig.savefig(p)
    print("wrote", p)
    plt.close(fig)

    # ---------------- figure 2: method comparison, zoomed ----------------
    zlo, zhi = 1.0, 4.5
    z = (ppm >= zlo) & (ppm <= zhi)
    fig, axes = plt.subplots(4, 1, figsize=(12.5, 8.4), sharex=True)
    panels = [("Real spectrum", real[1], NAVY),
              ("quilted_base — donors restricted to a globally similar pool", qb[1], CORAL),
              ("quilted — best-matching overlap, unrestricted pool", ql[1], TEAL),
              ("independent — a fresh random donor per window (the naive version)",
               ind[1], GOLD)]
    for ax, (lab, y, col) in zip(axes, panels):
        ax.plot(ppm[z], y[z], lw=0.75, color=col)
        ax.set_title(lab, fontweight="bold", fontsize=13.5, loc="left")
        ax.grid(axis="x", alpha=0.25, ls="--")
        ax.set_axisbelow(True)
        ax.set_yticks([])
    axes[-1].set_xlim(zhi, zlo)
    axes[-1].set_xlabel("chemical shift (ppm)")
    axes[0].text(0.0, 1.30, "y-scale is independent per panel", transform=axes[0].transAxes,
                 fontsize=11.5, color="0.35", style="italic")
    fig.supylabel("intensity (unit-area normalised)", fontsize=13.5)
    p = OUT / "fig_synth_methods.png"
    fig.savefig(p)
    print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
