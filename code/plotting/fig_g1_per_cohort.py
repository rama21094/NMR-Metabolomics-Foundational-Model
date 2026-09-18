#!/usr/bin/env python3
"""Gate G1 per cohort, with error bars -- the panel form of the slide-4 table.

The table collapses six targets into one mean, which hides that the gap is not
uniform: it is wide on the large cohorts and ambiguous on the small ones. One
panel per target, balanced accuracy against label budget.

ERROR BARS are the sd ACROSS PRETRAINING SEEDS (n=5), not across episodes. That
is the quantity that matters here: episode-to-episode spread is shared by every
representation because the episodes are paired, so it says nothing about whether
one representation beats another. Seed spread says how much of a reported gap is
just which checkpoint you happened to pick -- which is exactly the selection
error retracted twice on this project. Classical ML has no seed, so it is drawn
as a line with a band of +/- its episode sd for reference.
"""
import json, glob
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

GROUND, CLASSICAL, MASK, JIG, JOINT = "#FBF6E9", "#C0272D", "#2E5E9E", "#7FA6D0", "#94B8A0"
TITLES = {"brc_t2d_cancer":"BrC-T2D cancer","mtbls563_binary":"MTBLS563 binary",
          "barth_case_control":"Barth case/control","brc_t2d_diabetes":"BrC-T2D diabetes",
          "mtbls563_3class":"MTBLS563 3-class","brc_t2d_4class":"BrC-T2D 4-class"}
ORDER = ["brc_t2d_cancer","mtbls563_binary","barth_case_control",
         "brc_t2d_diabetes","mtbls563_3class","brc_t2d_4class"]
OBJ = [("masking",MASK,"o"),("jigsaw",JIG,"s"),("joint",JOINT,"^")]

rows=[]
for f in glob.glob("results/phase3/transfer*.json"): rows+=json.load(open(f))

# target -> budget -> {objective: [seed means], 'classical': (mean, sd)}
D=defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for r in rows:
    rep=r["representation"]; b=r["k_per_class"]
    if rep.startswith("classical"):
        D[r["target"]][b]["classical"]=(r["mean"], r["sd"])
    elif "_seed" in rep:
        D[r["target"]][b][rep.split("_")[0]].append(r["mean"])

BUD=[5,10,20,40,0]; XL=["5","10","20","40","all"]; X=np.arange(len(BUD))
fig,axes=plt.subplots(2,3,figsize=(13.2,6.6),sharex=True)
for ax,t in zip(axes.ravel(),ORDER):
    ax.set_facecolor(GROUND)
    c=np.array([D[t][b]["classical"][0] for b in BUD])
    cs=np.array([D[t][b]["classical"][1] for b in BUD])
    ax.fill_between(X,c-cs,c+cs,color=CLASSICAL,alpha=0.13,lw=0)
    ax.plot(X,c,color=CLASSICAL,lw=2.4,marker="D",ms=5,label="classical ML",zorder=5)
    for name,col,mk in OBJ:
        m=np.array([np.mean(D[t][b][name]) for b in BUD])
        e=np.array([np.std(D[t][b][name],ddof=1) for b in BUD])
        ax.errorbar(X,m,yerr=e,color=col,lw=1.9,marker=mk,ms=4.5,capsize=3,
                    elinewidth=1.1,label=name)
    ax.axhline(0.5,color="#999",lw=0.8,ls=":")
    ax.set_title(TITLES[t],fontsize=11.5,color="#1E2761",weight="bold")
    ax.set_xticks(X); ax.set_xticklabels(XL); ax.grid(alpha=0.25,lw=0.6)
    ax.set_ylim(0.30,0.95)
for ax in axes[:,0]: ax.set_ylabel("balanced accuracy",fontsize=10)
for ax in axes[1,:]: ax.set_xlabel("labelled examples per class",fontsize=10)
axes[0,0].legend(fontsize=9,loc="lower right",framealpha=0.9)
fig.suptitle("Gate G1 by cohort - classical ML leads at every label budget, in every cohort",
             fontsize=13.5,color="#1E2761",weight="bold",y=0.985)
fig.text(0.5,0.008,"Error bars: sd across 5 pretraining seeds. Classical band: +/- episode sd. "
         "50 paired episodes per point.",ha="center",fontsize=9,color="#666",style="italic")
fig.tight_layout(rect=[0,0.025,1,0.96])
fig.savefig("docs/figures/fig7_G1_per_cohort.png",dpi=200,facecolor="white")
print("wrote docs/figures/fig7_G1_per_cohort.png")

for t in ORDER:
    c=D[t][0]["classical"][0]; m=np.mean(D[t][0]["masking"]); s=np.std(D[t][0]["masking"],ddof=1)
    print(f"{t:<22} k=all  classical {c:.3f}   masking {m:.3f} +/- {s:.3f}   gap {c-m:+.3f}")

# --- slim variant for the deck: no suptitle/footer (the slide carries both) ---
fig.suptitle(""); fig.texts[-1].set_text("")
fig.set_size_inches(13.2, 4.9)
fig.tight_layout(rect=[0, 0, 1, 1])
fig.savefig("docs/figures/fig7_G1_per_cohort_slim.png", dpi=200, facecolor="white")
print("wrote docs/figures/fig7_G1_per_cohort_slim.png")
