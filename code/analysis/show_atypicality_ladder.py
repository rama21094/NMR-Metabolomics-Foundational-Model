#!/usr/bin/env python3
"""Show the moderately-atypical spectra, ordered, so a human can draw the line.

code/analysis/flag_atypical_spectra.py flags only 35 spectra as broken. Row 6200
-- the one that looked obviously different by eye -- is not among them: it sits
at the 19th percentile of shape correlation and the ~11th percentile of PCA
reconstruction residual. So there is a large population, on the order of 1,000
spectra, that is unusual but not broken.

Whether those belong in the corpus is a judgement, not a computation:

  - KEEP: a genuine disease phenotype SHOULD look unusual against a mostly
    healthy background. Removing it deletes the signal the project exists to
    find, and makes the near-duplication problem of SSL doc section 19 worse.
  - DROP: a different sample matrix, a different preparation protocol, or an
    acquisition that partly failed is not biology and adds noise.

Only a person looking at them can tell those apart, so this script does not
decide. It renders a LADDER: spectra drawn at fixed percentiles of atypicality
so the eye can see where the character changes, with row 6200 included as the
reference case that prompted the question.

Usage:
    python code/analysis/show_atypicality_ladder.py
    python code/analysis/show_atypicality_ladder.py --score corr_to_median
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REF = ("data/combined/combine_unique_MetaboLights_Workbench_"
       "Water_EDTA_Suppressed_ref0ppm.npy")
SCORES = "results/analysis/atypical/atypical_scores.csv"
FIGS = ROOT / "results/figures"

PPM_HI, SW = 14.8237, 20.024
PPM_PER_PT = SW / 131072
INK, ACC, WARN, MUTED = "#1f2933", "#2f6f8f", "#b04a3a", "#6b7480"

# Percentiles of atypicality to sample, most typical first.
PCTS = [50, 25, 15, 12, 10, 8, 6, 4, 2, 1]


def ppm_of(i):
    return PPM_HI - np.asarray(i) * PPM_PER_PT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", default="pca_resid",
                    choices=["pca_resid", "corr_to_median"],
                    help="pca_resid ranks by structure the corpus cannot "
                         "rebuild; corr_to_median by gross shape departure")
    ap.add_argument("--band", type=int, nargs=2, default=(74000, 92000))
    ap.add_argument("--reference-row", type=int, default=6200)
    args = ap.parse_args()

    X = np.load(ROOT / REF, mmap_mode="r")
    d = pd.read_csv(ROOT / SCORES).sort_values("row").reset_index(drop=True)
    n = len(d)

    # Atypicality, high = more atypical, for whichever score was chosen.
    s = d[args.score].to_numpy()
    atyp = s if args.score == "pca_resid" else -s
    rank = np.argsort(np.argsort(-atyp))          # 0 = most atypical
    pct_atyp = 100.0 * rank / n                   # 0 = most atypical

    ref_pct = float(pct_atyp[args.reference_row])
    print(f"score: {args.score}")
    print(f"row {args.reference_row} sits at the {ref_pct:.1f}th percentile "
          f"of atypicality (0 = most atypical)\n")

    picks, labels = [], []
    for p in PCTS:
        j = int(np.argmin(np.abs(pct_atyp - p)))
        picks.append(j)
        labels.append(f"{p}th pct of atypicality")
    picks.append(args.reference_row)
    labels.append(f"row {args.reference_row}  (the case that prompted this, "
                  f"{ref_pct:.0f}th pct)")

    # How many spectra are more atypical than each rung -- the size of the
    # population a cut at that rung would remove.
    print("if you cut at each rung, you would remove:")
    for p in PCTS:
        print(f"   {p:3d}th percentile -> {int(round(p * n / 100)):5,d} spectra "
              f"({p}% of the corpus)")

    lo, hi = args.band
    x = ppm_of(np.arange(lo, hi))
    typical = np.median(
        np.array(X[np.sort(np.argsort(atyp)[:400])][:, lo:hi],
                 dtype=np.float64), axis=0)
    typical /= typical.max() + 1e-12

    rows_n = len(picks)
    fig, axes = plt.subplots(rows_n, 1, figsize=(14, 1.55 * rows_n), sharex=True)
    for ax, row, lab in zip(axes, picks, labels):
        y = np.array(X[row, lo:hi], dtype=np.float64)
        y /= y.max() + 1e-12
        ax.plot(x, typical, lw=0.9, color=MUTED)
        is_ref = row == args.reference_row
        ax.plot(x, y, lw=0.9, color=WARN if is_ref else ACC)
        ax.set_title(f"{lab}    row {row}    corr {d.corr_to_median[row]:.2f}",
                     fontsize=9, loc="left",
                     color=WARN if is_ref else INK)
        ax.set_yticks([])
        ax.invert_xaxis()
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    axes[0].plot([], [], color=MUTED, label="typical corpus median")
    axes[0].legend(fontsize=8, frameon=False)
    axes[-1].set_xlabel("chemical shift (ppm)")
    fig.suptitle("Atypicality ladder — most typical at top. Where does the "
                 "character change from 'unusual patient' to 'bad data'?",
                 fontsize=12, y=0.999)
    fig.tight_layout()
    p = FIGS / f"fig_atypicality_ladder_{args.score}.png"
    fig.savefig(p, dpi=145, bbox_inches="tight")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
