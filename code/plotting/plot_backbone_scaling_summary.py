#!/usr/bin/env python3
"""Final verdict on the backbone axis: patch size and capacity both exhausted.

Five masking backbones pretrained on the same v4 corpus, compared through the
frozen linear probe. Left: no new configuration beats the original small
patch-1024 model at any pooling. Right: reconstruction loss does not predict
downstream utility -- the best-reconstructing model is among the worst
downstream.
"""
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/matplotlib")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from pathlib import Path

df = pd.read_csv("results/analysis/patch_size_comparison/patch_size_results.csv")
HELD=["barth","mtbls326","brc_t2d_cancer"]
ARMS=[("ps128","patch 128\n0.63M"),("ps256","patch 256\n0.66M"),
      ("ps1024_nhead4_true","patch 1024\n1.89M"),("ps2048","patch 2048\n5.42M"),
      ("ps1024_d256_L6","patch 1024\nd256 L6\n5.17M")]
RECON={"ps128":4.36e-5,"ps256":5.56e-5,"ps1024_nhead4_true":9.26e-5,
       "ps2048":1.020e-4,"ps1024_d256_L6":3.95e-5}
fig,(axl,axr)=plt.subplots(1,2,figsize=(13,4.8),dpi=300,gridspec_kw={"width_ratios":[1.2,1]})

w=0.38
for i,(pool,lab,col) in enumerate([("mean_pool","mean-pool","#898781"),("flatten","flatten","#1baf7a")]):
    xs=np.arange(len(ARMS))+(i-0.5)*w
    vals=[df[(df.arm==a)&(df.pooling==pool)&(df.dataset.isin(HELD))].balanced_accuracy.mean() for a,_ in ARMS]
    bars=axl.bar(xs,vals,width=w*0.9,color=col,label=lab,zorder=3)
    for b,v in zip(bars,vals):
        axl.annotate(f"{v:.3f}",xy=(b.get_x()+b.get_width()/2,v),xytext=(0,2),
                     textcoords="offset points",ha="center",fontsize=7.5)
axl.axhline(0.881,color="#0b0b0b",linestyle="--",linewidth=1.5,zorder=5,label="classical LogReg (0.881)")
axl.set_xticks(np.arange(len(ARMS))); axl.set_xticklabels([l for _,l in ARMS],fontsize=8)
axl.set_ylabel("mean balanced accuracy\n(held-out 3: Barth, MTBLS326, cancer)")
axl.set_ylim(0.6,0.95)
axl.set_title("No new backbone beats the original small one",fontsize=11,loc="left")
axl.legend(frameon=False,fontsize=8,loc="lower left",ncol=1)
axl.text(0.0,-0.28,"Four new pretraining runs — patch 128, 256, 2048, and a 2.7x-capacity model — all fail to beat\n"
 "patch 1024 at 1.89M params. Patch size and capacity are both exhausted as axes of improvement.\n"
 "ps2048 carries 2.9x the baseline's parameters and still loses, so this is not a capacity limit.",
 transform=axl.transAxes,fontsize=7.4,color="#898781",va="top")

xs=[RECON[a] for a,_ in ARMS]
ys=[df[(df.arm==a)&(df.pooling=="flatten")&(df.dataset.isin(HELD))].balanced_accuracy.mean() for a,_ in ARMS]
axr.scatter(xs,ys,s=90,color="#2a78d6",zorder=3)
for (a,lab),x,y in zip(ARMS,xs,ys):
    axr.annotate(lab.replace("\n"," "),xy=(x,y),xytext=(6,-3),textcoords="offset points",fontsize=7)
axr.set_xlabel("best validation reconstruction loss  (lower = 'better' pretraining)")
axr.set_ylabel("mean held-out balanced accuracy")
axr.set_title("Reconstruction loss does not predict utility",fontsize=11,loc="left")
axr.text(0.0,-0.28,"The BEST-reconstructing model (d256 L6, 3.95e-5 — 2.3x better than baseline) is among the WORST\n"
 "downstream. The baseline reconstructs 2.3x worse and transfers best. Spearman(recon, accuracy) = +0.60,\n"
 "i.e. if anything HIGHER reconstruction loss goes with better transfer (n=5, not significant).\n"
 "Consequence: never select checkpoints or architectures on reconstruction loss.",
 transform=axr.transAxes,fontsize=7.4,color="#898781",va="top")

for ax in (axl,axr):
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7"); ax.spines["bottom"].set_color("#c3c2b7")
    ax.yaxis.grid(True,color="#e1e0d9",linewidth=0.8,zorder=0); ax.set_axisbelow(True)
fig.suptitle("Backbone scaling: patch size and capacity are both exhausted",fontsize=12.5,x=0.02,ha="left",y=1.03)
fig.tight_layout()
out=Path("results/plots/all_datasets_summary_v4/fig11_backbone_scaling.png")
fig.savefig(out,bbox_inches="tight"); print("Wrote",out)
