#!/usr/bin/env python3
"""Bar chart of GAN-vs-real separability by region (results/phase5/gan_auc_by_region_full.json)."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
A = json.load(open(ROOT / "results/phase5/gan_auc_by_region_full.json"))
rows = [("Far edges only\n(below -0.2 and above 9.4 ppm)", "far edges only (<-0.2 or >9.4)", "#C2571A"),
        ("Area around TSP\n(-0.2 to 0.2 ppm)", "TSP area only (-0.2 to 0.2)", "#C2571A"),
        ("Metabolite region\n(0.2 to 9.4 ppm)", "metabolite region only (0.2 to 9.4)", "#2E5E9E"),
        ("Whole spectrum", "whole spectrum", "#8A8A8A")]
plt.rcParams.update({"font.size": 17, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(13, 5.6))
for i, (lab, k, col) in enumerate(rows):
    ax.barh(i, A[k] - 0.5, left=0.5, color=col, height=0.62)
    ax.text(A[k] + 0.008, i, f"{A[k]:.2f}", va="center", fontsize=18, weight="bold")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
ax.set_xlim(0.5, 1.02); ax.axvline(0.5, color="#555", lw=1)
ax.set_xlabel("how easily a simple test tells GAN from real (AUC)\n0.5 = cannot tell apart,  1.0 = always can")
fig.tight_layout(); fig.savefig(ROOT / "docs/figures/pi16_gan_region_auc.png", dpi=170, facecolor="white")
print("wrote docs/figures/pi16_gan_region_auc.png")
