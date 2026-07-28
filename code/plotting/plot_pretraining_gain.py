#!/usr/bin/env python3
"""Experiment #3: what does SSL pretraining actually contribute?

Compares each family's frozen embedding against a TRUE random-init backbone of
the identical architecture (no pretrained weights anywhere -- patch embedding
and positional encoding included), with the classifier held fixed at a
converged LogReg (C=1) on frozen features. Holding the head fixed removes the
head-underfitting confound that made the original --reinit-unfrozen-xavier
ablation unreadable; that flag also only reset the layers being fine-tuned.
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from pathlib import Path

df = pd.read_csv("results/linear_probe/exp3_pretrained_vs_random.csv")
DS = ["barth","mtbls326","brc_t2d_cancer","brc_t2d_diabetes","mtbls563"]
LBL = {"barth":"Barth","mtbls326":"MTBLS326","brc_t2d_cancer":"BrC-T2D\n(cancer)",
       "brc_t2d_diabetes":"BrC-T2D\n(diabetes)","mtbls563":"MTBLS563"}
FAM = ["masking","jigsaw","joint"]
COL = {"masking":"#1baf7a","jigsaw":"#eda100","joint":"#e34948"}

fig, ax = plt.subplots(figsize=(10.5,4.8), dpi=300)
w = 0.26
for i,f in enumerate(FAM):
    xs = np.arange(len(DS)) + (i-1)*w
    vals = [float(df[(df.dataset==d)&(df.family==f)]["pretraining_gain"].iloc[0]) for d in DS]
    bars = ax.bar(xs, vals, width=w*0.9, color=COL[f], label=f, zorder=3)
    for b,v in zip(bars,vals):
        ax.annotate(f"{v:+.3f}", xy=(b.get_x()+b.get_width()/2, v),
                    xytext=(0, 3 if v>=0 else -11), textcoords="offset points",
                    ha="center", fontsize=7.5)
ax.axhline(0, color="#52514e", linewidth=1.3, zorder=4)
ax.set_xticks(np.arange(len(DS))); ax.set_xticklabels([LBL[d] for d in DS], fontsize=9)
ax.set_ylabel("Δ balanced accuracy\n(pretrained − random init)")
ax.set_title("Experiment #3: what SSL pretraining actually contributes",
             fontsize=12, loc="left", color="#0b0b0b")
for spine in ("top","right"): ax.spines[spine].set_visible(False)
ax.spines["left"].set_color("#c3c2b7"); ax.spines["bottom"].set_color("#c3c2b7")
ax.yaxis.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0); ax.set_axisbelow(True)
ax.legend(title="SSL family", frameon=False, fontsize=9, title_fontsize=9, loc="upper right")
ax.text(0.005, -0.19,
  "Above zero = pretraining beats an untrained backbone of the same architecture. Head fixed (LogReg C=1 on frozen features) in both arms.\n"
  "Masked pretraining helps on 5/5 targets (mean +0.117). Jigsaw is worthless (mean -0.011) and joint is actively harmful (mean -0.025),\n"
  "each losing to random init on 3/5. A random transformer is a legitimately strong random-projection baseline, so 'random wins' means the\n"
  "objective adds nothing over a random projection -- not that the architecture is useless.",
  transform=ax.transAxes, fontsize=7.5, color="#898781", va="top")
fig.tight_layout()
out = Path("results/plots/all_datasets_summary_v4/fig8_pretraining_gain.png")
fig.savefig(out, bbox_inches="tight"); print("Wrote", out)
