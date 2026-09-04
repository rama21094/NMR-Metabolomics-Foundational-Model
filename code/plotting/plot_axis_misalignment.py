#!/usr/bin/env python3
"""Figure for the corpus ppm-axis misalignment finding.

Left  : lag histogram measured from the spectra (no metadata used).
Right : the same displacement predicted from Bruker metadata, per study.

Run code/analysis/corpus_axis_misalignment.py and
code/analysis/bruker_param_audit.py first.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results/analysis/bruker_params"
OUT = ROOT / "results/figures"

INK, ACC, WARN = "#1f2933", "#2f6f8f", "#b04a3a"
LINEWIDTH_PTS = 13.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lag = np.load(RES / "corpus_xcorr_lags.npy")
    per = pd.concat([pd.read_csv(RES / "axis_displacement.csv")])

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    # ---- measured -------------------------------------------------------
    a.hist(lag, bins=np.arange(-1500, 1501, 25), color=ACC, edgecolor="none")
    a.axvline(0, color=INK, lw=1, ls="--")
    a.axvspan(-LINEWIDTH_PTS, LINEWIDTH_PTS, color=INK, alpha=0.18,
              label="1 linewidth (13 pts)")
    a.set_xlabel("cross-correlation lag vs corpus median (points)")
    a.set_ylabel("spectra")
    a.set_title("Measured from the spectra\n(no metadata used)", fontsize=11)
    a.legend(fontsize=8, frameon=False)
    n_out = int((np.abs(lag) > 1500).sum())
    a.text(0.02, 0.95, f"{n_out} rows beyond ±1500 pts\n(stretched studies)",
           transform=a.transAxes, va="top", fontsize=8, color=WARN)

    # ---- predicted ------------------------------------------------------
    per = per.sort_values("n", ascending=True)
    y = np.arange(len(per))
    cols = [WARN if abs(v) > 1500 else ACC for v in per["dx@3.0ppm_pts"]]
    b.barh(y, per["dx@3.0ppm_pts"], color=cols, height=0.7)
    b.set_yticks(y)
    b.set_yticklabels([f"{s}  (n={n})" for s, n in
                       zip(per["study"], per["n"])], fontsize=8)
    b.axvline(0, color=INK, lw=1)
    b.set_xlabel("predicted displacement at 3 ppm (points)")
    b.set_title("Predicted from Bruker OFFSET / SW / SF\nper study", fontsize=11)
    b.set_xscale("symlog", linthresh=100)

    for ax in (a, b):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.suptitle("The training corpus is not on a common chemical-shift axis",
                 fontsize=12.5, y=1.0)
    fig.tight_layout()
    p = OUT / "fig_axis_misalignment.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
