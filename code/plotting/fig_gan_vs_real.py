#!/usr/bin/env python3
"""Where GAN-generated spectra differ from real ones -- one figure for the PI."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
GROUND, BLUE, RED, GREY, INK = "#FBF6E9", "#2E5E9E", "#C0272D", "#8A8A8A", "#1E2761"
F = json.load(open(ROOT / "results/phase5/gan_vs_real_features.json"))
A = json.load(open(ROOT / "results/phase5/gan_auc_by_region.json"))
c = np.array(F["centres"]); d = np.array(F["delta_syn"]); thr = F["null_99"]
med = np.array(F["real_median_binned"])


def style(ax):
    ax.set_facecolor(GROUND)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


fig = plt.figure(figsize=(13.2, 7.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.42, wspace=0.28)

# (a) where along the spectrum
ax = fig.add_subplot(gs[0, :]); style(ax)
cols = [RED if v > 0 else BLUE for v in d]
ax.bar(c, d, width=0.018, color=cols, alpha=0.85)
ax.axhspan(-thr, thr, color=GREY, alpha=0.25, lw=0)
ax.axhline(0, color="#555", lw=0.6)
for lo, hi in ((-0.55, 0.2), (9.4, 10.05)):
    ax.axvspan(lo, hi, color="#E8C9A0", alpha=0.35, lw=0)
ax2 = ax.twinx(); ax2.fill_between(c, 0, med / med.max(), color=INK, alpha=0.12, lw=0)
ax2.set_ylim(0, 3.5); ax2.set_yticks([])
ax.set_xlim(10.05, -0.55); ax.set_ylim(-0.75, 0.75)
ax.set_xlabel("chemical shift (ppm)")
ax.set_ylabel("how different\n(Cliff's delta)")
ax.set_title("(a)  Where GAN spectra differ from real ones, window by window",
             loc="left", color=INK, weight="bold", fontsize=12)
ax.text(9.97, 0.6, "red = GAN higher\nblue = GAN lower", fontsize=9.5, color="#444", va="top")
ax.text(4.9, -0.68, "grey band: real-vs-real difference  |  shaded: spectrum edges  |  broad blue is partly an area-scaling side effect of the edge excess",
        fontsize=9, color="#555", ha="center")

# (b) AUC by region
ax = fig.add_subplot(gs[1, 0]); style(ax)
labels = ["all", "edges\nremoved", "edges\nonly"]
keys = ["all windows", "edges removed (0.2-9.4 ppm)", "edges only (<0.2 or >9.4 ppm)"]
x = np.arange(3); w = 0.36
for i, (tag, col) in enumerate((("GAN v2", BLUE), ("GAN v2 study-mixing", GREY))):
    v = [A[f"{tag} | {k}"] for k in keys]
    ax.bar(x + (i - 0.5) * w, v, w * 0.92, color=col, label=tag.replace("GAN v2 ", ""))
    for xi, vi in zip(x, v):
        ax.text(xi + (i - 0.5) * w, vi + 0.01, f"{vi:.2f}", ha="center", fontsize=9)
ax.axhline(0.5, color=RED, ls="--", lw=1.6); ax.text(-0.55, 0.465, "0.5 = can't tell", color=RED, fontsize=9, ha="left")
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0.4, 1.06); ax.set_xlim(-0.6, 2.6)
ax.set_ylabel("how easily a simple test\nspots the fakes (AUC)")
ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
ax.set_title("(b)  The edges give it away", loc="left", color=INK, weight="bold", fontsize=12)

# (c) PCA
ax = fig.add_subplot(gs[1, 1]); style(ax)
Zr = np.array(F["pc_scores"]["real"]); Zs = np.array(F["pc_scores"]["synth"])
ax.scatter(Zr[:, 0], Zr[:, 1], s=6, color=INK, alpha=0.25, label="real", lw=0)
ax.scatter(Zs[:, 0], Zs[:, 1], s=6, color=RED, alpha=0.25, label="GAN v2", lw=0)
ax.set_xlabel(f"PC1 ({F['pcs'][0]['var']:.0%} of real variation)")
ax.set_ylabel(f"PC2 ({F['pcs'][1]['var']:.0%})")
ax.legend(fontsize=9, markerscale=3, frameon=False)
ax.set_title("(c)  Inside the real cloud", loc="left", color=INK, weight="bold", fontsize=12)

# (d) what the edge looks like
ax = fig.add_subplot(gs[1, 2]); style(ax)
ppm = np.load(ROOT / "rebuild/pretrain/ppm_axis.npy")
X = np.load(ROOT / "rebuild/pretrain/corpus_train.npy", mmap_mode="r")
S = np.load(ROOT / "rebuild/pretrain/synthetic/gan_v2.npy", mmap_mode="r")
m = (ppm <= 10.0) & (ppm >= 9.3)
for k, (M, col, lab) in enumerate(((X, INK, "real"), (S, RED, "GAN v2"))):
    for j in range(3):
        y = np.asarray(M[[11, 2222, 5555][j] if M is X else [3, 300, 3000][j]], dtype=np.float64)
        y = y / np.abs(y).max()
        ax.plot(ppm[m], y[m] * 1e3 + (0 if M is X else 0.0), color=col, lw=0.8, alpha=0.8,
                label=lab if j == 0 else None)
ax.set_xlim(10.0, 9.3); ax.set_xlabel("chemical shift (ppm)")
ax.set_ylabel("intensity (x 1/1000 of max)")
ax.legend(fontsize=9, frameon=False)
ax.set_title("(d)  Real edges are flat", loc="left", color=INK, weight="bold", fontsize=12)

fig.savefig(ROOT / "docs/figures/fig_gan_vs_real.png", dpi=200, facecolor="white", bbox_inches="tight")
print("wrote docs/figures/fig_gan_vs_real.png")
