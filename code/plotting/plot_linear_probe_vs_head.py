import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from pathlib import Path

df = pd.read_csv("results/analysis/linear_probe_frozen/linear_probe_results.csv")
df["delta_vs_frozen"] = df.linear_probe_bal_acc - df.ssl_head_frozen
DS = ["barth","mtbls326","brc_t2d_cancer","brc_t2d_diabetes","mtbls563"]
LBL = {"barth":"Barth","mtbls326":"MTBLS326","brc_t2d_cancer":"BrC-T2D\n(cancer)",
       "brc_t2d_diabetes":"BrC-T2D\n(diabetes)","mtbls563":"MTBLS563"}
FAM = ["masking","jigsaw","joint"]
COL = {"masking":"#1baf7a","jigsaw":"#eda100","joint":"#e34948"}

fig, ax = plt.subplots(figsize=(10,4.6), dpi=300)
w = 0.26
for i,f in enumerate(FAM):
    xs = np.arange(len(DS)) + (i-1)*w
    vals = [float(df[(df.dataset==d)&(df.family==f)]["delta_vs_frozen"].iloc[0]) for d in DS]
    bars = ax.bar(xs, vals, width=w*0.9, color=COL[f], label=f, zorder=3)
    for b,v in zip(bars,vals):
        ax.annotate(f"{v:+.3f}", xy=(b.get_x()+b.get_width()/2, v),
                    xytext=(0, 3 if v>=0 else -11), textcoords="offset points",
                    ha="center", fontsize=7.5)
ax.axhline(0, color="#52514e", linewidth=1.2, zorder=4)
ax.set_xticks(np.arange(len(DS))); ax.set_xticklabels([LBL[d] for d in DS], fontsize=9)
ax.set_ylabel("Δ balanced accuracy\n(LogReg probe − trained MLP head)")
ax.set_title("Experiment #2: same frozen features, different classifier fit",
             fontsize=12, loc="left", color="#0b0b0b")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#c3c2b7"); ax.spines["bottom"].set_color("#c3c2b7")
ax.yaxis.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0); ax.set_axisbelow(True)
ax.legend(title="SSL family", frameon=False, fontsize=9, title_fontsize=9, loc="upper right")
ax.text(0.005, -0.20,
  "Above zero = the trained head is leaving accuracy on the table. Backbone, pooling, features and CV folds are identical in both arms;\n"
  "only how the final linear map is fitted differs (converged L2 LogReg vs Adam ~50 epochs + dropout + early stopping).\n"
  "The masking head is underfit on all 5 of 5 targets (mean +0.120). jigsaw and joint are essentially unaffected (mean +0.009).",
  transform=ax.transAxes, fontsize=7.5, color="#898781", va="top")
fig.tight_layout()
out = Path("results/plots/all_datasets_summary_v4/fig7_linear_probe_vs_head.png")
fig.savefig(out, bbox_inches="tight"); print("Wrote", out)
