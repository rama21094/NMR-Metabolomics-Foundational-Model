#!/usr/bin/env python3
"""Experiment #7 result: 2x2 factorial over the masking pretext task.

Left   -- per-target balanced accuracy for the four arms, flatten pooling, with
          classical LogReg as the line to beat. Nothing reaches it.
Middle -- the factorial read as main effects. Block masking is negative on both
          the selection and held-out splits; peak weighting is a small positive
          well inside the baseline uncertainty.
Right  -- the confound the experiment actually uncovered (§5f): the v3 vs v4
          ps1024 baselines differ by -0.069 held-out despite byte-identical
          config, which is larger than either factorial effect. Every bar in the
          middle panel has to be read against that error bar.
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from pathlib import Path

df = pd.read_csv("results/analysis/exp7_objective_comparison/patch_size_results.csv")
FL = df[df.pooling == "flatten"]

DS = ["barth", "mtbls326", "brc_t2d_cancer", "mtbls563", "brc_t2d_diabetes"]
LBL = {"barth": "Barth", "mtbls326": "MTBLS326", "brc_t2d_cancer": "BrC-T2D\n(cancer)",
       "mtbls563": "MTBLS563\n(sel)", "brc_t2d_diabetes": "BrC-T2D\n(diab, sel)"}
CLASSICAL = {"barth": 0.705, "mtbls326": 1.000, "brc_t2d_cancer": 0.937,
             "mtbls563": 0.721, "brc_t2d_diabetes": 0.829}
HELD = ["barth", "mtbls326", "brc_t2d_cancer"]
SEL = ["mtbls563", "brc_t2d_diabetes"]

ARMS = [("exp7_D_baseline_v4", "D  sparse + uniform (ref)", "#5a6b7c"),
        ("exp7_A_blk8", "A  block8 + uniform", "#e34948"),
        ("exp7_B_pk025", "B  sparse + top-25%", "#2a78d6"),
        ("exp7_C_blk8_pk025", "C  block8 + top-25%", "#8e5ab8")]


def val(arm, ds, pooling="flatten"):
    r = df[(df.arm == arm) & (df.dataset == ds) & (df.pooling == pooling)]
    return float(r.balanced_accuracy.iloc[0])


fig, (axl, axm, axr) = plt.subplots(1, 3, figsize=(16.5, 5.0), dpi=300,
                                    gridspec_kw={"width_ratios": [2.15, 1.15, 1.0]})

# ---------------- LEFT: per-target bars ----------------
w = 0.2
for i, (arm, lab, col) in enumerate(ARMS):
    xs = np.arange(len(DS)) + (i - 1.5) * w
    vals = [val(arm, d) for d in DS]
    bars = axl.bar(xs, vals, width=w * 0.9, color=col, label=lab, zorder=3)
    for b, v in zip(bars, vals):
        axl.annotate(f"{v:.3f}", xy=(b.get_x() + b.get_width() / 2, v), xytext=(0, 2),
                     textcoords="offset points", ha="center", fontsize=6.2, rotation=90)
for j, d in enumerate(DS):
    axl.hlines(CLASSICAL[d], j - 0.44, j + 0.44, color="black", lw=1.9, zorder=5,
               label="classical LogReg" if j == 0 else None)
axl.axvspan(2.5, 4.5, color="#f0f0f0", zorder=0)
axl.text(3.5, 1.065, "pre-committed selection subset", ha="center", va="top",
         fontsize=7.5, color="#777")
axl.set_xticks(np.arange(len(DS)))
axl.set_xticklabels([LBL[d] for d in DS], fontsize=8.5)
axl.set_ylabel("Balanced accuracy (frozen linear probe, flatten)")
axl.set_ylim(0.42, 1.08)
axl.set_title("No arm reaches classical LogReg", fontsize=11, loc="left", pad=20)
# Legend above the axes: bars always run to the axis floor, so any in-axes
# placement sits on top of them and the values behind it become unreadable.
axl.legend(frameon=False, fontsize=7.6, loc="lower left", bbox_to_anchor=(0, 1.0),
           ncol=5, columnspacing=1.0, handletextpad=0.45)
axl.grid(axis="y", alpha=0.25, zorder=0)

# ---------------- MIDDLE: main effects ----------------
def cell(arm, datasets):
    return float(np.mean([val(arm, d) for d in datasets]))


def effects(datasets):
    blk = (cell("exp7_A_blk8", datasets) + cell("exp7_C_blk8_pk025", datasets)) / 2 \
        - (cell("exp7_D_baseline_v4", datasets) + cell("exp7_B_pk025", datasets)) / 2
    pk = (cell("exp7_B_pk025", datasets) + cell("exp7_C_blk8_pk025", datasets)) / 2 \
        - (cell("exp7_D_baseline_v4", datasets) + cell("exp7_A_blk8", datasets)) / 2
    inter = (cell("exp7_C_blk8_pk025", datasets) - cell("exp7_A_blk8", datasets)
             - cell("exp7_B_pk025", datasets) + cell("exp7_D_baseline_v4", datasets))
    return blk, pk, inter


eh, es = effects(HELD), effects(SEL)
names = ["block\nmasking", "peak\nweighting", "inter-\naction"]
x = np.arange(3)
axm.bar(x - 0.19, eh, width=0.36, color="#2a78d6", label="held-out (3)", zorder=3)
axm.bar(x + 0.19, es, width=0.36, color="#9dc3ea", label="selection (2)", zorder=3)
for xi, (a, b) in enumerate(zip(eh, es)):
    for dx, v in ((-0.19, a), (0.19, b)):
        axm.annotate(f"{v:+.3f}", xy=(xi + dx, v), xytext=(0, 3 if v >= 0 else -11),
                     textcoords="offset points", ha="center", fontsize=7.4)
# The uncertainty band from §5f -- the v3/v4 baseline discrepancy.
axm.axhspan(-0.069, 0.069, color="#e34948", alpha=0.11, zorder=1)
axm.axhline(0, color="black", lw=1)
axm.text(2.46, 0.062, "baseline uncertainty\n(§5f, ±0.069)", ha="right", va="top",
         fontsize=7.4, color="#a03030")
axm.set_xticks(x); axm.set_xticklabels(names, fontsize=8.5)
axm.set_ylabel("Δ balanced accuracy")
axm.set_ylim(-0.09, 0.09)
axm.set_title("Block masking hurts; peak weighting\nis inside the noise", fontsize=11, loc="left")
axm.legend(frameon=False, fontsize=8, loc="lower left")

# ---------------- RIGHT: the v3/v4 baseline confound ----------------
pairs = [(d, val("ps1024_nhead4_true", d), val("exp7_D_baseline_v4", d)) for d in DS]
ys = np.arange(len(pairs))[::-1]
for y, (d, v3, v4) in zip(ys, pairs):
    axr.plot([v3, v4], [y, y], color="#bbb", lw=1.4, zorder=2)
    axr.scatter([v3], [y], s=42, color="#2a9d55", zorder=3, label="v3 corpus" if y == ys[0] else None)
    axr.scatter([v4], [y], s=42, color="#e34948", zorder=3, label="v4 corpus" if y == ys[0] else None)
    axr.annotate(f"{v4 - v3:+.3f}", xy=(max(v3, v4), y), xytext=(6, -2.5),
                 textcoords="offset points", fontsize=7.4,
                 color="#a03030" if v4 < v3 else "#2a7a45")
axr.set_yticks(ys); axr.set_yticklabels([LBL[d].replace("\n", " ") for d, _, _ in pairs], fontsize=8)
axr.set_xlim(0.50, 1.10)
axr.set_xlabel("Balanced accuracy (flatten)")
axr.set_title("Same config, different corpus:\n−0.069 held-out", fontsize=11, loc="left")
axr.legend(frameon=False, fontsize=8, loc="lower left")
axr.grid(axis="x", alpha=0.25)

fig.text(0.005, -0.055,
         "Four arms, identical geometry (ps1024, d128, L3, nhead4, 1.89M params) and identical v4 corpus; only the pretext task differs. "
         "All early-stopped.\nBlock masking DID make the task harder as intended (val loss 7.10e-5 → 1.00e-4, +41%, best epoch +450) and still lost "
         "0.030 held-out — harder is not sufficient.\nPeak weighting's +0.011 has the same sign on both splits but is ~6x smaller than the baseline "
         "discrepancy in the right panel, so it is not established.",
         fontsize=7.6, ha="left", va="top", color="#333")

out = Path("results/plots/all_datasets_summary_v4")
out.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out / "fig12_exp7_factorial.png", bbox_inches="tight", facecolor="white")
print(f"Wrote {out / 'fig12_exp7_factorial.png'}")
