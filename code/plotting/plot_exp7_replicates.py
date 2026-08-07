#!/usr/bin/env python3
"""Experiment #7 follow-up: replicates, the corpus gap, and the noise floor.

Left   -- three independent runs of ONE configuration (v4 ps1024 baseline) per
          target. The within-arm scatter is large (sd 0.026-0.046) and the v3
          reference sits clearly above the whole v4 cluster: the §5f corpus gap
          is real, not noise.
Middle -- peak weighting judged WITHOUT a corpus confound. Trained on v3 and
          compared to the v3 baseline it LOSES 0.039 held-out and collapses on
          BrC-T2D diabetes. Arm B's +0.011 was an artifact of a depressed v4
          reference.
Right   -- recalibration. Every effect this document has claimed, against the
          0.020 floor for a held-out-mean claim. Half of them do not clear it.
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from pathlib import Path

df = pd.read_csv("results/analysis/exp7_replicates/patch_size_results.csv")
FL = df[df.pooling == "flatten"]

HELD = ["barth", "mtbls326", "brc_t2d_cancer"]
ALL5 = HELD + ["mtbls563", "brc_t2d_diabetes"]
LBL = {"barth": "Barth", "mtbls326": "MTBLS326", "brc_t2d_cancer": "BrC-T2D\n(cancer)",
       "mtbls563": "MTBLS563\n(sel)", "brc_t2d_diabetes": "BrC-T2D\n(diab, sel)"}
REPS = ["exp7_D_baseline_v4", "exp7_D_v4_seed101", "exp7_D_v4_seed202"]
V3REF = "ps1024_nhead4_true"
PEAK = ["exp7_v3_pk025_r1", "exp7_v3_pk025_r2"]
CLASSICAL = {"barth": .705, "mtbls326": 1.0, "brc_t2d_cancer": .937,
             "mtbls563": .721, "brc_t2d_diabetes": .829}


def v(arm, ds):
    return float(FL[(FL.arm == arm) & (FL.dataset == ds)].balanced_accuracy.iloc[0])


fig, (axl, axm, axr) = plt.subplots(1, 3, figsize=(17, 5.1), dpi=300,
                                    gridspec_kw={"width_ratios": [1.5, 1.35, 1.5]})

# ---------------- LEFT: replicate scatter ----------------
for j, d in enumerate(ALL5):
    vals = [v(a, d) for a in REPS]
    axl.scatter([j] * 3, vals, s=52, color="#e34948", zorder=4,
                label="v4 runs (n=3, same config)" if j == 0 else None)
    axl.vlines(j, min(vals), max(vals), color="#e34948", lw=1.2, alpha=.5, zorder=3)
    axl.scatter([j], [v(V3REF, d)], s=95, marker="*", color="#2a9d55", zorder=5,
                label="v3 reference" if j == 0 else None)
    axl.annotate(f"±{np.std(vals, ddof=1):.3f}", xy=(j, min(vals)), xytext=(0, -13),
                 textcoords="offset points", ha="center", fontsize=6.8, color="#a03030")
axl.axvspan(2.5, 4.5, color="#f2f2f2", zorder=0)
axl.set_xticks(range(5)); axl.set_xticklabels([LBL[d] for d in ALL5], fontsize=8.3)
axl.set_ylabel("Balanced accuracy (frozen probe, flatten)")
axl.set_ylim(0.50, 1.06)
axl.set_title("One config, three runs: the scatter is large\n"
              "— v3 is at or above every v4 draw on all 3 held-out", fontsize=10.5, loc="left")
axl.legend(frameon=False, fontsize=8, loc="lower left")
axl.grid(axis="y", alpha=.25)

# ---------------- MIDDLE: matched peak comparison ----------------
w = 0.26
groups = [(V3REF, "v3 baseline", "#5a6b7c"), (PEAK[0], "v3 + top-25% (r1)", "#2a78d6"),
          (PEAK[1], "v3 + top-25% (r2)", "#9dc3ea")]
for i, (arm, lab, col) in enumerate(groups):
    xs = np.arange(len(ALL5)) + (i - 1) * w
    vals = [v(arm, d) for d in ALL5]
    bars = axm.bar(xs, vals, width=w * .9, color=col, label=lab, zorder=3)
    for b, val in zip(bars, vals):
        axm.annotate(f"{val:.3f}", xy=(b.get_x() + b.get_width() / 2, val), xytext=(0, 2),
                     textcoords="offset points", ha="center", fontsize=6.0, rotation=90)
for j, d in enumerate(ALL5):
    axm.hlines(CLASSICAL[d], j - .42, j + .42, color="black", lw=1.7, zorder=5,
               label="classical LogReg" if j == 0 else None)
axm.axvspan(2.5, 4.5, color="#f2f2f2", zorder=0)
axm.set_xticks(range(5)); axm.set_xticklabels([LBL[d] for d in ALL5], fontsize=8.3)
axm.set_ylim(0.50, 1.10)
axm.set_ylabel("Balanced accuracy")
axm.set_title("Peak weighting, matched corpus:\nit LOSES (−0.039 held-out)", fontsize=10.5, loc="left", pad=18)
# Above the axes -- bars run to the axis floor, so an in-axes legend covers them.
axm.legend(frameon=False, fontsize=7.4, loc="lower left", bbox_to_anchor=(0, 1.0),
           ncol=4, columnspacing=0.9, handletextpad=0.4)
axm.grid(axis="y", alpha=.25)

# ---------------- RIGHT: recalibration ----------------
FLOOR = 0.0200
claims = [("v3 vs v4 corpus (§5f)", +0.0687),
          ("patch 128 vs 1024 (§5b)", -0.0416),
          ("peak weighting, v3 matched (§7b)", -0.0391),
          ("patch 256 vs 1024 (§5b)", -0.0335),
          ("block masking (§7)", -0.0298),
          ("ps2048 vs ps1024 (§5d)", +0.0202),
          ("peak weighting, v4 unmatched (§7)", +0.0108),
          ("d256L6 vs ps1024 (§5d)", +0.0057)]
names = [c[0] for c in claims]
vals = np.array([c[1] for c in claims])
ys = np.arange(len(claims))[::-1]
cols = ["#2a9d55" if abs(x) >= 2 * FLOOR else "#eda100" if abs(x) >= FLOOR else "#e34948"
        for x in vals]
axr.barh(ys, np.abs(vals), color=cols, zorder=3, height=.62)
axr.axvline(FLOOR, color="#333", ls="--", lw=1.3, zorder=4)
axr.axvline(2 * FLOOR, color="#333", ls=":", lw=1.1, zorder=4)
axr.text(FLOOR, -1.05, "noise\nfloor 0.020", fontsize=7.2, color="#333", ha="center", va="bottom")
axr.text(2 * FLOOR, -1.05, "2x\nfloor", fontsize=7.2, color="#333", ha="center", va="bottom")
axr.set_ylim(-1.25, len(claims) - 0.35)
for y, val in zip(ys, vals):
    axr.annotate(f"{val:+.3f}", xy=(abs(val), y), xytext=(4, -2.5), textcoords="offset points",
                 fontsize=7.4)
axr.set_yticks(ys); axr.set_yticklabels(names, fontsize=8)
axr.set_xlim(0, 0.084)
axr.set_xlabel("|Δ balanced accuracy| (held-out mean)")
axr.set_title("Half of this document's claims do not\nclear the noise floor", fontsize=10.5, loc="left")
axr.grid(axis="x", alpha=.25)

fig.text(0.005, -0.06,
         "Noise floor: the three v4 replicates have per-target sd 0.026-0.046 (mean 0.035). Their held-out MEANS agree to sd 0.0037, but that is luck, not precision "
         "— Barth falls\nwhile cancer rises across the three draws (r = -0.92), so errors cancel inside the mean. With three draws that cancellation cannot be relied on, so the floor for a "
         "held-out-mean\nclaim is the independence figure 0.035/sqrt(3) = 0.020. Separately, --seed does NOT make GPU runs reproducible: the two v3 peak runs used --seed 101 and still "
         "differ (max|dW| = 5.3e-2),\nbecause cudnn.benchmark autotuning and AMP vary kernel selection and reduction order between processes.",
         fontsize=7.4, ha="left", va="top", color="#333")

out = Path("results/plots/all_datasets_summary_v4"); out.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out / "fig13_exp7_replicates.png", bbox_inches="tight", facecolor="white")
print(f"Wrote {out / 'fig13_exp7_replicates.png'}")
