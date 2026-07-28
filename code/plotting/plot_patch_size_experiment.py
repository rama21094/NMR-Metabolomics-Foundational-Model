#!/usr/bin/env python3
"""Experiment #4 result: patch size vs pooling, on the frozen linear probe.

Left  -- shrinking patch_size (1024 -> 256 -> 128) does NOT lift the
         representation ceiling; it hurts. The resolution hypothesis is refuted.
Right -- the actual win was in pooling: replacing mean-pool over patches with
         position-preserving flatten helps on all five targets.
All arms use nhead=4 (the value training actually used) except where noted.
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from pathlib import Path

df = pd.read_csv("results/analysis/patch_size_comparison/patch_size_results.csv")
DS = ["barth","mtbls326","brc_t2d_cancer","brc_t2d_diabetes","mtbls563"]
LBL = {"barth":"Barth","mtbls326":"MTBLS326","brc_t2d_cancer":"BrC-T2D\n(cancer)",
       "brc_t2d_diabetes":"BrC-T2D\n(diabetes)","mtbls563":"MTBLS563"}
fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.5,4.8), dpi=300)

# ---- LEFT: patch size, flatten pooling ----
ARMS = [("ps1024_nhead4_true","patch 1024 (128 tok)","#2a78d6"),
        ("ps256","patch 256 (512 tok)","#eda100"),
        ("ps128","patch 128 (1024 tok)","#e34948")]
w=0.26
for i,(arm,lab,col) in enumerate(ARMS):
    xs=np.arange(len(DS))+(i-1)*w
    vals=[float(df[(df.dataset==d)&(df.arm==arm)&(df.pooling=="flatten")].balanced_accuracy.iloc[0]) for d in DS]
    bars=axl.bar(xs,vals,width=w*0.9,color=col,label=lab,zorder=3)
    for b,v in zip(bars,vals):
        axl.annotate(f"{v:.3f}",xy=(b.get_x()+b.get_width()/2,v),xytext=(0,2),
                     textcoords="offset points",ha="center",fontsize=6.8)
axl.set_xticks(np.arange(len(DS))); axl.set_xticklabels([LBL[d] for d in DS],fontsize=8.5)
axl.set_ylabel("Balanced accuracy (frozen linear probe)")
axl.set_ylim(0.4,1.06)
axl.set_title("Smaller patches HURT — hypothesis refuted",fontsize=11,loc="left")
axl.legend(frameon=False,fontsize=8,loc="lower left")
axl.text(0.0,-0.235,"Mean Δ vs patch 1024: patch 256 −0.072, patch 128 −0.077 (0 of 5 wins).\n"
  "Reconstruction loss FELL as patches shrank (9.3e-5 → 5.6e-5 → 4.4e-5): a masked 128-point\n"
  "patch is interpolable from its neighbours, so the pretext task got easier, not more informative.\n"
  "Caveat: the small-patch models also have ~3× fewer parameters (0.63/0.66M vs 1.89M).",
  transform=axl.transAxes,fontsize=7.3,color="#898781",va="top")

# ---- RIGHT: pooling ----
sub = df[df.arm=="ps1024_nhead4_true"]
w2=0.36
for i,(pool,lab,col) in enumerate([("mean_pool","mean-pool over patches (current)","#898781"),
                                   ("flatten","flatten / position-preserving","#1baf7a")]):
    xs=np.arange(len(DS))+(i-0.5)*w2
    vals=[float(sub[(sub.dataset==d)&(sub.pooling==pool)].balanced_accuracy.iloc[0]) for d in DS]
    bars=axr.bar(xs,vals,width=w2*0.9,color=col,label=lab,zorder=3)
    for b,v in zip(bars,vals):
        axr.annotate(f"{v:.3f}",xy=(b.get_x()+b.get_width()/2,v),xytext=(0,2),
                     textcoords="offset points",ha="center",fontsize=7)
CLASSICAL={"barth":0.705,"mtbls326":1.000,"brc_t2d_cancer":0.937,"brc_t2d_diabetes":0.829,"mtbls563":0.721}
for j,d in enumerate(DS):
    axr.plot([j-0.42,j+0.42],[CLASSICAL[d]]*2,color="#0b0b0b",linestyle="--",linewidth=1.4,
             zorder=5,label="classical LogReg" if j==0 else None)
axr.set_xticks(np.arange(len(DS))); axr.set_xticklabels([LBL[d] for d in DS],fontsize=8.5)
axr.set_ylim(0.4,1.06)
axr.set_title("The real win: position-preserving pooling",fontsize=11,loc="left")
axr.legend(frameon=False,fontsize=8,loc="lower left")
axr.text(0.0,-0.235,"Flatten beats mean-pool on all 5 targets (+0.030 to +0.129). Combined with the LogReg head this\n"
  "gains +0.078 mean over the originally reported DNN-head numbers, and flips Barth to an SSL win\n"
  "(0.806 vs classical 0.705) while tying MTBLS326 at 1.000. Dashed line = classical LogReg baseline.",
  transform=axr.transAxes,fontsize=7.3,color="#898781",va="top")

for ax in (axl,axr):
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7"); ax.spines["bottom"].set_color("#c3c2b7")
    ax.yaxis.grid(True,color="#e1e0d9",linewidth=0.8,zorder=0); ax.set_axisbelow(True)
fig.suptitle("Experiment #4: patch size and pooling",fontsize=13,x=0.02,ha="left",y=1.02)
fig.tight_layout()
out=Path("results/plots/all_datasets_summary_v4/fig9_patch_size_and_pooling.png")
fig.savefig(out,bbox_inches="tight"); print("Wrote",out)
