#!/usr/bin/env python3
"""PCA view of GAN v2 vs real spectra for the PI deck (results/phase5/gan_vs_real_features.json).

Left: scores on the first two principal components of REAL spectra (PCA fitted
on real only, 2,000 spectra each). Right: per spectrum, the share of its
variation that the 20 real components do not capture (median over spectra).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
F = json.load(open(ROOT / "results/phase5/gan_vs_real_features.json"))
INK, ORANGE = "#1E2761", "#C2571A"
Zr, Zs = np.array(F["pc_scores"]["real"]), np.array(F["pc_scores"]["synth"])
plt.rcParams.update({"font.size": 17, "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(16, 6.4), gridspec_kw={"width_ratios": [1.6, 1]})
a.scatter(Zr[:, 0], Zr[:, 1], s=10, color=INK, alpha=0.35, lw=0, label="real (2,000 spectra)")
a.scatter(Zs[:, 0], Zs[:, 1], s=10, color=ORANGE, alpha=0.35, lw=0, label="GAN v2 (2,000 spectra)")
a.set_xlabel(f"main pattern 1 in real spectra (PC1, {F['pcs'][0]['var']:.0%} of real variation)")
a.set_ylabel(f"main pattern 2 (PC2, {F['pcs'][1]['var']:.0%})")
a.legend(markerscale=3, frameon=False, loc="upper left")
a.set_title("GAN spectra fall inside the real cloud", loc="left", color=INK, weight="bold")
v = [F["unexplained"]["real"], F["unexplained"]["synth"]]
b.bar([0, 1], v, color=[INK, ORANGE], width=0.6)
for i, x in enumerate(v):
    b.text(i, x + 0.02, f"{x:.0%}", ha="center", fontsize=22, weight="bold")
b.set_xticks([0, 1]); b.set_xticklabels(["real", "GAN v2"]); b.set_ylim(0, 1)
b.set_yticks([0, 0.25, 0.5, 0.75, 1]); b.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
b.set_ylabel("share of each spectrum's variation\nnot explained by real patterns")
b.set_title("…but vary in new ways", loc="left", color=INK, weight="bold")
fig.tight_layout(); fig.savefig(ROOT / "docs/figures/pi16_gan_pca.png", dpi=170, facecolor="white")
print("wrote docs/figures/pi16_gan_pca.png")
