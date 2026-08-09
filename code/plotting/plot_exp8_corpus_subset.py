#!/usr/bin/env python3
"""Experiment #8: does the 1.7% of corpus rows that differ between v3/v4
explain the §5f/§7b gap? Inconclusive -- but not in the way it first looks.

Left   -- the four reference points, held-out mean, with a noise-floor band
          around v3. common9506 (drop the 164 differing rows) and the
          v3rand9506 control (drop 164 DIFFERENT unchanged rows, same size)
          land in almost the same place: close to v4, below v3.
Middle -- the decisive comparison. common vs control, per target. Every gap is
          inside the per-target noise floor -- dropping the SPECIFIC 164 rows
          is indistinguishable from dropping an arbitrary 164.
Right   -- where that leaves the two live hypotheses: row CONTENT (refuted by
          the middle panel) vs corpus SIZE (unconfirmed -- a 1.7% cut moving
          accuracy 0.05 would be disproportionate next to every capacity
          experiment, which moved it <=0.02 for up to 2.9x more parameters).
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from pathlib import Path

df = pd.read_csv("results/analysis/exp8_corpus_subset/patch_size_results.csv")
FL = df[df.pooling == "flatten"]

HELD = ["barth", "mtbls326", "brc_t2d_cancer"]
ALL5 = HELD + ["mtbls563", "brc_t2d_diabetes"]
LBL = {"barth": "Barth", "mtbls326": "MTBLS326", "brc_t2d_cancer": "BrC-T2D\n(cancer)",
       "mtbls563": "MTBLS563\n(sel)", "brc_t2d_diabetes": "BrC-T2D\n(diab, sel)"}
V4_REPS = ["exp7_D_baseline_v4", "exp7_D_v4_seed101", "exp7_D_v4_seed202"]
V3_REF = "ps1024_nhead4_true"
COMMON, CONTROL = "exp8_common9506", "exp8_v3rand9506_control"
FLOOR_MEAN, FLOOR_TGT = 0.020, 0.035


def v(arm, ds):
    return float(FL[(FL.arm == arm) & (FL.dataset == ds)].balanced_accuracy.iloc[0])


def held_mean(arm):
    return float(np.mean([v(arm, d) for d in HELD]))


fig, (axl, axm, axr) = plt.subplots(1, 3, figsize=(16.5, 5.1), dpi=300,
                                    gridspec_kw={"width_ratios": [1.15, 1.5, 1.35]})

# ---------------- LEFT: the four reference points ----------------
v4_means = [held_mean(a) for a in V4_REPS]
v4_mean = float(np.mean(v4_means))
v3_mean = held_mean(V3_REF)
common_mean, control_mean = held_mean(COMMON), held_mean(CONTROL)

points = [("v4 replicates\n(n=3)", v4_means, "#e34948"),
          ("v3 reference\n(n=1)", [v3_mean], "#2a9d55"),
          ("common9506\n(drop the 164)", [common_mean], "#2a78d6"),
          ("v3rand control\n(drop 164 random)", [control_mean], "#9dc3ea")]
for i, (lab, vals, col) in enumerate(points):
    axl.scatter([i] * len(vals), vals, s=90, color=col, zorder=4, edgecolor="white", linewidth=0.8)
    if len(vals) > 1:
        axl.vlines(i, min(vals), max(vals), color=col, lw=1.3, alpha=.6, zorder=3)
axl.axhspan(v3_mean - FLOOR_MEAN, v3_mean + FLOOR_MEAN, color="#2a9d55", alpha=.08, zorder=0)
axl.axhspan(v4_mean - FLOOR_MEAN, v4_mean + FLOOR_MEAN, color="#e34948", alpha=.08, zorder=0)
axl.set_xticks(range(4)); axl.set_xticklabels([p[0] for p in points], fontsize=8.3)
axl.set_ylabel("Held-out mean balanced accuracy (flatten)")
axl.set_ylim(0.78, 0.91)
axl.set_title("Both ablation arms land near v4,\nnot v3", fontsize=10.8, loc="left")
axl.grid(axis="y", alpha=.25)
axl.text(2.5, v3_mean + FLOOR_MEAN + .003, "±floor around v3/v4", fontsize=7, color="#555", ha="center")

# ---------------- MIDDLE: common vs control, per target ----------------
w = 0.32
xs = np.arange(len(ALL5))
cvals = [v(COMMON, d) for d in ALL5]
kvals = [v(CONTROL, d) for d in ALL5]
b1 = axm.bar(xs - w / 2, cvals, width=w * .92, color="#2a78d6", label="common9506 (drop the 164)", zorder=3)
b2 = axm.bar(xs + w / 2, kvals, width=w * .92, color="#9dc3ea", label="v3rand control (drop 164 random)", zorder=3)
for bars in (b1, b2):
    for bx in bars:
        axm.annotate(f"{bx.get_height():.3f}", xy=(bx.get_x() + bx.get_width() / 2, bx.get_height()),
                     xytext=(0, 2), textcoords="offset points", ha="center", fontsize=6.6)
for i, d in enumerate(ALL5):
    diff = abs(cvals[i] - kvals[i])
    axm.annotate(f"Δ={diff:.3f}", xy=(i, max(cvals[i], kvals[i]) + 0.045), ha="center",
                 fontsize=7, color="#a03030" if diff > FLOOR_TGT else "#555")
axm.axvspan(2.5, 4.5, color="#f2f2f2", zorder=0)
axm.set_xticks(xs); axm.set_xticklabels([LBL[d] for d in ALL5], fontsize=8.3)
axm.set_ylim(0.50, 1.08)
axm.set_ylabel("Balanced accuracy")
axm.set_title("Decisive test: dropping the SPECIFIC 164 rows\n"
              "≈ dropping an ARBITRARY 164 (all Δ < 0.035 floor)", fontsize=10.8, loc="left", pad=20)
axm.legend(frameon=False, fontsize=7.8, loc="lower left", bbox_to_anchor=(0, 1.0),
           ncol=2, columnspacing=1.0, handletextpad=0.4)
axm.grid(axis="y", alpha=.25)

# ---------------- RIGHT: the two live hypotheses ----------------
axr.axis("off")
axr.set_title("Where this leaves the two hypotheses", fontsize=10.8, loc="left")
boxes = [
    ("Row CONTENT\n(the 164 specific rows carry the effect)", "REFUTED", "#e34948",
     "common ≈ control (max Δ 0.028,\nfloor 0.035). Dropping different\nrows gives the same answer."),
    ("Corpus SIZE\n(9506 vs 9670, a 1.7% cut)", "UNCONFIRMED", "#b36a00",
     "Both arms sit near v4 (Δ 0.018,\ninside floor). But a 1.7% cut moving\naccuracy 0.05 would dwarf every\ncapacity experiment (≤0.02 for\nup to 2.9× more params)."),
    ("Next step", "→", "#2a78d6",
     "≥2 more replicates of common9506\nand the control before either\nhypothesis is reported as established."),
]
y0 = 0.92
for title, tag, col, body in boxes:
    axr.add_patch(plt.Rectangle((0.02, y0 - 0.27), 0.96, 0.25, transform=axr.transAxes,
                                facecolor=col, alpha=.10, edgecolor=col, linewidth=1.1))
    axr.text(0.06, y0 - 0.045, title, transform=axr.transAxes, fontsize=9.6, fontweight="bold", va="top")
    axr.text(0.94, y0 - 0.045, tag, transform=axr.transAxes, fontsize=9, fontweight="bold",
             color=col, va="top", ha="right")
    axr.text(0.06, y0 - 0.13, body, transform=axr.transAxes, fontsize=8.4, va="top", linespacing=1.35)
    y0 -= 0.34

fig.text(0.005, -0.06,
         "n=1 per ablation arm against a 0.035 per-target / 0.020 held-out-mean noise floor (§7b). common9506 = 9,506 rows identical in v3 and v4 (the 164 differing "
         "rows removed). v3rand control = v3\nwith 164 DIFFERENT (always-unchanged) rows dropped at random -- same size, keeps all 164 special rows -- isolating corpus SIZE from CONTENT. "
         "The mechanism originally suspected (v4 leaves a residual\nEDTA artefact that compresses those 164 spectra) was separately tested and refuted: only 7/164 rows have their row-max "
         "inside the EDTA window in either version, and v4's rows are slightly BRIGHTER outside it.",
         fontsize=7.3, ha="left", va="top", color="#333")

out = Path("results/plots/all_datasets_summary_v4"); out.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out / "fig14_exp8_corpus_subset.png", bbox_inches="tight", facecolor="white")
print(f"Wrote {out / 'fig14_exp8_corpus_subset.png'}")
