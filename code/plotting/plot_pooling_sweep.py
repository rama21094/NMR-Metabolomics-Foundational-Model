#!/usr/bin/env python3
"""How much chemical-shift position does the classifier need? (answers the
attention-pooling question empirically)

Regional pooling splits the T encoder tokens into G contiguous groups,
mean-pools within each, and concatenates -- so G controls how much positional
detail survives, at G*d_model features. G=1 IS mean-pooling; G=T IS flatten.
Sweeping G traces the whole curve between the two extremes at controlled
dimensionality.
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from pathlib import Path

df = pd.read_csv("results/analysis/pooling_sweep/pooling_sweep_results.csv")
DS = ["barth","mtbls326","brc_t2d_cancer","brc_t2d_diabetes","mtbls563"]
LBL = {"barth":"Barth","mtbls326":"MTBLS326","brc_t2d_cancer":"BrC-T2D (cancer)",
       "brc_t2d_diabetes":"BrC-T2D (diabetes)","mtbls563":"MTBLS563"}
COL = {"barth":"#2a78d6","mtbls326":"#1baf7a","brc_t2d_cancer":"#eda100",
       "brc_t2d_diabetes":"#e34948","mtbls563":"#8b5cf6"}
fig,(axl,axr) = plt.subplots(1,2,figsize=(13,4.8),dpi=300,gridspec_kw={"width_ratios":[1.25,1]})

for ds in DS:
    s = df[df.dataset==ds].sort_values("groups")
    axl.plot(s.groups, s.balanced_accuracy, "o-", color=COL[ds], label=LBL[ds],
             linewidth=1.8, markersize=4.5, zorder=3)
    b = s.loc[s.balanced_accuracy.idxmax()]
    axl.scatter([b.groups],[b.balanced_accuracy],s=130,facecolor="none",
                edgecolor=COL[ds],linewidth=2,zorder=4)
axl.set_xscale("log", base=2)
gs = sorted(df.groups.unique())
axl.set_xticks(gs); axl.set_xticklabels([str(g) for g in gs])
axl.set_xlabel("pooling groups G   (1 = mean-pool  →  128 = flatten)")
axl.set_ylabel("Balanced accuracy (frozen linear probe)")
axl.set_title("Intermediate pooling beats both extremes",fontsize=11,loc="left")
axl.legend(frameon=False,fontsize=8,loc="lower right")
axl.text(0.0,-0.215,"Circled = per-dataset best. On 4 of 5 targets the optimum is STRICTLY between mean-pool and flatten.\n"
  "Feature dimension is G x 128, so G=32 costs 4096 features vs flatten's 16384 — better accuracy at 4x lower dimension,\n"
  "which matters at n=37..113 samples.",
  transform=axl.transAxes,fontsize=7.4,color="#898781",va="top")

mean = df.groupby("groups").balanced_accuracy.mean()
bars = axr.bar(range(len(mean)), mean.values, color="#c3c2b7", zorder=3)
best = int(np.argmax(mean.values))
bars[best].set_color("#1baf7a")
bars[0].set_color("#898781")
for i,(g,v) in enumerate(zip(mean.index, mean.values)):
    axr.annotate(f"{v:.3f}",xy=(i,v),xytext=(0,3),textcoords="offset points",
                 ha="center",fontsize=7.5)
axr.set_xticks(range(len(mean)))
axr.set_xticklabels([f"{g}\nd={g*128}" for g in mean.index],fontsize=6.8,rotation=0)
axr.set_ylim(0.75,0.85)
axr.set_xlabel("pooling groups G")
axr.set_ylabel("mean balanced accuracy (5 targets)")
axr.set_title("Mean across datasets",fontsize=11,loc="left")
axr.text(0.0,-0.235,"Grey = mean-pool (current, 0.793). Green = best (G=32, 0.816). Flatten (G=128) reaches 0.813.\n"
  "G=16 ties G=32 at 0.816 with only 2048 features. Gain over mean-pool +0.023; over flatten +0.003 (noise).\n"
  "Caveat: G chosen by looking at these same numbers, so treat the exact optimum as indicative — for an\n"
  "unbiased estimate G should be selected by nested CV inside each training fold.",
  transform=axr.transAxes,fontsize=7.4,color="#898781",va="top")

for ax in (axl,axr):
    for s_ in ("top","right"): ax.spines[s_].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7"); ax.spines["bottom"].set_color("#c3c2b7")
    ax.yaxis.grid(True,color="#e1e0d9",linewidth=0.8,zorder=0); ax.set_axisbelow(True)
fig.suptitle("Pooling sweep: mean-pool vs regional vs flatten (masking, patch 1024)",
             fontsize=12.5,x=0.02,ha="left",y=1.03)
fig.tight_layout()
out=Path("results/plots/all_datasets_summary_v4/fig10_pooling_sweep.png")
fig.savefig(out,bbox_inches="tight"); print("Wrote",out)
