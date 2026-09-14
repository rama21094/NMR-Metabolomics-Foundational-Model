#!/usr/bin/env python3
"""Find spectra that are genuinely a different kind of thing, like row 6200.

Prompted by inspect_residual_shifts.py, where one panel (row 6200) was obviously
not a shifted version of the corpus median but a different spectrum altogether --
large extra signal near 2.8 ppm, strong peaks at 2.4 the median does not have.

This is a DIFFERENT question from alignment, and needs different machinery. A
misaligned spectrum is the same chemistry in the wrong place; an atypical
spectrum is different chemistry, a different matrix, or a failed acquisition.

Three scores, deliberately chosen to fail independently, so that agreement
between them means something:

  1. SHAPE CORRELATION to the corpus median. Catches gross departures.
  2. PCA RECONSTRUCTION ERROR. The corpus is low-rank (86% of variance in five
     components, SSL doc 19), so a spectrum the leading components cannot
     rebuild carries structure the population does not have. This is the score
     that should catch a genuinely novel metabolite profile.
  3. ISOLATION FOREST on the PCA scores. Non-parametric, catches clusters that
     are self-consistent but sit away from the bulk -- e.g. a whole study with a
     different matrix, which the first two scores may rate as merely mediocre.

A spectrum is flagged when at least two of the three rank it in their worst
`--contamination` fraction. Requiring two suppresses the single-method failure
modes: shape correlation alone punishes any residual misalignment, and PCA error
alone punishes low-SNR spectra.

IMPORTANT -- this flags, it does not delete. Being unusual is not the same as
being wrong: a real disease phenotype should look unusual against a mostly
healthy corpus, and dropping it would remove exactly the signal the project
exists to detect. The output is a ranked list with the three scores and a figure,
for a human to decide on.

A separate, older detector exists at code/analysis/detect_outlier_spectra.py; it
works from the 60 extracted canonical peaks and needs that pipeline re-run after
re-referencing. This one works directly on the spectra and needs nothing else.

Usage:
    python code/analysis/flag_atypical_spectra.py
    python code/analysis/flag_atypical_spectra.py --contamination 0.01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[2]
REF = ("data/combined/combine_unique_MetaboLights_Workbench_"
       "Water_EDTA_Suppressed_ref0ppm.npy")
OUT = ROOT / "results/analysis/atypical"
FIGS = ROOT / "results/figures"

NPT = 131_072
PPM_HI, SW = 14.8237, 20.024
PPM_PER_PT = SW / NPT
SIGNAL = (26000, 115000)        # ~0.6 to 10.9 ppm, excluding the dead edges
WATER = (62500, 68000)
NFEAT = 4000

INK, ACC, WARN, MUTED = "#1f2933", "#2f6f8f", "#b04a3a", "#6b7480"


def ppm_of(i):
    return PPM_HI - np.asarray(i) * PPM_PER_PT


def build_features(X, cols):
    n = X.shape[0]
    F = np.empty((n, len(cols)), dtype=np.float32)
    for i in range(0, n, 500):
        j = min(i + 500, n)
        F[i:j] = np.array(X[i:j][:, cols], dtype=np.float32)
        print(f"\r  features {j}/{n}", end="", flush=True)
    print()
    F -= F.mean(axis=1, keepdims=True)
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
    return F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contamination", type=float, default=0.02)
    ap.add_argument("--n-components", type=int, default=30)
    ap.add_argument("--n-show", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    X = np.load(ROOT / REF, mmap_mode="r")
    n = X.shape[0]

    cols = np.linspace(SIGNAL[0], SIGNAL[1] - 1, NFEAT).astype(int)
    cols = cols[(cols < WATER[0]) | (cols >= WATER[1])]
    F = build_features(X, cols)

    med = np.median(F, axis=0)
    med = (med - med.mean()) / (np.linalg.norm(med) + 1e-12)
    s_corr = F @ med                                    # high = typical

    pca = PCA(n_components=args.n_components, random_state=args.seed).fit(F)
    recon = pca.inverse_transform(pca.transform(F))
    s_pca = np.linalg.norm(F - recon, axis=1)           # high = atypical

    iso = IsolationForest(contamination=args.contamination,
                          random_state=args.seed).fit(pca.transform(F))
    s_iso = -iso.score_samples(pca.transform(F))        # high = atypical

    k = max(1, int(round(args.contamination * n)))
    worst_corr = set(np.argsort(s_corr)[:k])
    worst_pca = set(np.argsort(-s_pca)[:k])
    worst_iso = set(np.argsort(-s_iso)[:k])
    votes = np.array([sum(i in w for w in (worst_corr, worst_pca, worst_iso))
                      for i in range(n)])
    flagged = np.where(votes >= 2)[0]

    print(f"\ncorpus {n:,}; worst {args.contamination:.1%} per method = {k} each")
    print(f"  flagged by >=2 of 3 methods: {len(flagged):,} "
          f"({100 * len(flagged) / n:.2f}%)")
    print(f"  flagged by all 3:            {(votes == 3).sum():,}")
    print(f"\n  pairwise overlap: corr&pca {len(worst_corr & worst_pca)}, "
          f"corr&iso {len(worst_corr & worst_iso)}, "
          f"pca&iso {len(worst_pca & worst_iso)}")
    print("  (low overlap between methods is expected and is why 2 of 3 is the "
          "bar; high overlap would mean they are not independent)")

    df = pd.DataFrame({"row": np.arange(n), "votes": votes,
                       "corr_to_median": s_corr,
                       "pca_resid": s_pca, "iso_score": s_iso}
                      ).sort_values(["votes", "pca_resid"],
                                    ascending=[False, False])
    df.to_csv(OUT / "atypical_scores.csv", index=False)

    # Sanity check on the case that prompted this.
    r6200 = int(votes[6200]) if n > 6200 else -1
    print(f"\n  row 6200 (the case that prompted this): {r6200}/3 votes, "
          f"corr {s_corr[6200]:.3f} (median {np.median(s_corr):.3f}), "
          f"pca resid rank {int((s_pca > s_pca[6200]).sum())} of {n}")

    # ---- figure -----------------------------------------------------------
    show = df[df["votes"] >= 2]["row"].to_numpy()[:args.n_show]
    if len(show) == 0:
        print("\n  nothing flagged; no figure written")
        return
    lo, hi = 74000, 90000
    x = ppm_of(np.arange(lo, hi))
    ref_med = np.median(np.array(X[np.sort(np.argsort(-s_corr)[:400])][:, lo:hi],
                                 dtype=np.float64), axis=0)
    ref_med /= ref_med.max() + 1e-12

    rows_n = int(np.ceil(len(show) / 2))
    fig, axes = plt.subplots(rows_n, 2, figsize=(14, 1.7 * len(show)),
                             sharex=True, squeeze=False)
    for ax, row in zip(axes.ravel(), show):
        y = np.array(X[row, lo:hi], dtype=np.float64)
        y /= y.max() + 1e-12
        ax.plot(x, ref_med, lw=1.0, color=MUTED, label="typical corpus median")
        ax.plot(x, y, lw=1.0, color=WARN, label=f"row {row}")
        ax.set_title(f"row {row}   votes {int(votes[row])}/3   "
                     f"corr {s_corr[row]:.2f}", fontsize=9, loc="left")
        ax.set_yticks([]); ax.invert_xaxis()
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    for ax in axes.ravel()[len(show):]:
        ax.set_visible(False)
    axes.ravel()[0].legend(fontsize=8, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("chemical shift (ppm)")
    fig.suptitle("Spectra flagged as atypical by >=2 of 3 independent methods — "
                 "for inspection, NOT automatic removal", fontsize=12, y=0.999)
    fig.tight_layout()
    p = FIGS / "fig_atypical_spectra.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"\n  wrote {p}")
    print(f"  wrote {(OUT / 'atypical_scores.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
