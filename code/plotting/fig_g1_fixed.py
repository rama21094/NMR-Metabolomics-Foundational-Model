#!/usr/bin/env python3
"""Slide 2, redrawn from the corrected G1 evaluation (phase3_fixed_eval.py).

y is (pretrained - standard ML) on identical splits, with 95% intervals; zero is
parity. Few-shot budgets are plotted only where the cohort supports them with at
least 5 test patients per group, so the x-axis means what it says. The full-data
result (LOOCV for Barth, 10x5-fold CV otherwise) sits at the right, separated,
because it is a different design rather than a larger budget.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
GROUND, RED, INK = "#FBF6E9", "#C0272D", "#1E2761"
COL = {"masking": "#2E5E9E", "joint": "#7FA6D0", "jigsaw": "#A9A9B8"}
MK = {"masking": "o", "joint": "^", "jigsaw": "s"}
TITLES = {"brc_t2d_cancer": "BrC-T2D cancer", "mtbls563_binary": "MTBLS563 binary",
          "barth_case_control": "Barth case/control", "brc_t2d_diabetes": "BrC-T2D diabetes",
          "mtbls563_3class": "MTBLS563 3-class", "brc_t2d_4class": "BrC-T2D 4-class"}
ORDER = list(TITLES)
R = json.load(open(ROOT / "results/phase3_fixed/g1_fixed.json"))
BUD = [5, 10, 20, 40]; FULL = 5.0   # x position for the full-data point

fig, axes = plt.subplots(2, 3, figsize=(13.2, 5.6), sharey=True)
for ax, t in zip(axes.ravel(), ORDER):
    ax.set_facecolor(GROUND)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.axhline(0, color=RED, lw=1.8, ls="--")
    ax.axvline(4.5, color="#bbb", lw=0.8)
    rows = [r for r in R if r["target"] == t]
    for j, obj in enumerate(("jigsaw", "joint", "masking")):
        off = (j - 1) * 0.12
        fs = sorted([r for r in rows if r["objective"] == obj and r["design"] == "few-shot"],
                    key=lambda r: r["k_per_class"])
        xs = [BUD.index(r["k_per_class"]) + 1 + off for r in fs]
        ax.errorbar(xs, [r["diff"] for r in fs],
                    yerr=[[r["diff"] - r["lo"] for r in fs], [r["hi"] - r["diff"] for r in fs]],
                    color=COL[obj], marker=MK[obj], ms=6, lw=1.6 if obj == "masking" else 1.1,
                    capsize=3, label=obj)
        full = [r for r in rows if r["objective"] == obj and r["design"] != "few-shot"]
        for r in full:
            ax.errorbar([FULL + off], [r["diff"]], yerr=[[r["diff"] - r["lo"]], [r["hi"] - r["diff"]]],
                        color=COL[obj], marker=MK[obj], ms=7, capsize=3, ls="none", elinewidth=1.4)
    design = next((r["design"] for r in rows if r["design"] != "few-shot"), "")
    n = {"loocv": "LOOCV", "cv10x5": "5-fold CV"}.get(design, "")
    ax.set_xticks([1, 2, 3, 4, FULL]); ax.set_xticklabels(["5", "10", "20", "40", f"all\n({n})"])
    ax.set_xlim(0.5, 5.5)
    ax.set_title(TITLES[t], fontsize=11.5, color=INK, weight="bold")
    ax.grid(axis="y", alpha=0.3, lw=0.6)
for ax in axes[:, 0]: ax.set_ylabel("our model minus\nstandard ML", fontsize=10)
for ax in axes[1, :]: ax.set_xlabel("labelled patients per group", fontsize=10)
axes[0, 0].legend(fontsize=9, loc="lower left", framealpha=0.9, title="pretraining", title_fontsize=9)
fig.tight_layout()
fig.savefig(ROOT / "docs/figures/fig7_G1_fixed.png", dpi=200, facecolor="white")
print("wrote docs/figures/fig7_G1_fixed.png")
