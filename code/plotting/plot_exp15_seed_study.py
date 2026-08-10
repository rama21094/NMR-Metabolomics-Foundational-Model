#!/usr/bin/env python3
"""Experiment #15: the corpus effect was a sampling artifact.

Left   -- every seed run for both corpora, with the n=5 means. The v3 reference
          that §5f's +0.069 rests on is the topmost point of its own
          distribution; the distributions themselves overlap almost completely.
Middle -- the gap as reported (v3 n=1 vs v4 mean) against the gap with error
          bars on both sides. It does not shrink, it vanishes.
Right  -- recalibration: every single-run claim in the project against the
          MEASURED single-run sd, rather than the 0.020 that was assumed.
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from pathlib import Path

df = pd.read_csv("results/analysis/exp15_seed_study/patch_size_results.csv")
FL = df[df.pooling == "flatten"]
HELD = ["barth", "mtbls326", "brc_t2d_cancer"]
present = set(FL.arm)

V3 = [("ps1024_nhead4_true", "original"), ("exp15_v3_seed202", "202"),
      ("exp15_v3_seed303", "303"), ("exp15_v3_seed404", "404"),
      ("exp15_v3_seed505", "505")]
V4 = [("exp7_D_baseline_v4", "original"), ("exp7_D_v4_seed101", "101"),
      ("exp7_D_v4_seed202", "202"), ("exp15_v4_seed303", "303"),
      ("exp15_v4_seed404", "404")]


def held(arm):
    return float(FL[FL.arm == arm].set_index("dataset")["balanced_accuracy"].reindex(HELD).mean())


v3 = [(l, held(a)) for a, l in V3 if a in present]
v4 = [(l, held(a)) for a, l in V4 if a in present]
v3v, v4v = np.array([v for _, v in v3]), np.array([v for _, v in v4])
orig3 = held("ps1024_nhead4_true")

fig, (axl, axm, axr) = plt.subplots(1, 3, figsize=(17, 5.2), dpi=300,
                                    gridspec_kw={"width_ratios": [1.15, 1.0, 1.6]})

# ---------------- LEFT: the two distributions ----------------
rng = np.random.default_rng(0)
for i, (vals, labs, col, name) in enumerate([
        (v3v, [l for l, _ in v3], "#2a9d55", f"v3  (n={len(v3v)})"),
        (v4v, [l for l, _ in v4], "#e34948", f"v4  (n={len(v4v)})")]):
    x = i + (rng.random(len(vals)) - 0.5) * 0.18
    axl.scatter(x, vals, s=70, color=col, zorder=4, edgecolor="white", linewidth=0.8, label=name)
    axl.hlines(vals.mean(), i - 0.30, i + 0.30, color=col, lw=2.4, zorder=5)
    axl.add_patch(plt.Rectangle((i - 0.30, vals.mean() - vals.std(ddof=1)), 0.60,
                                2 * vals.std(ddof=1), facecolor=col, alpha=.12, zorder=1))
    axl.annotate(f"{vals.mean():.3f}\n±{vals.std(ddof=1):.3f}", xy=(i + 0.34, vals.mean()),
                 fontsize=8.5, color=col, va="center", fontweight="bold")
# mark the original v3 reference as the outlier it is
axl.annotate("the §5f reference\n(n=1, highest of 5)", xy=(0, orig3), xytext=(0.42, orig3 + 0.004),
             fontsize=8, color="#1b5e33", va="center",
             arrowprops=dict(arrowstyle="->", color="#1b5e33", lw=1.2))
axl.set_xticks([0, 1]); axl.set_xticklabels(["v3 corpus", "v4 corpus"], fontsize=10)
axl.set_xlim(-0.85, 1.75)
axl.set_ylabel("Held-out mean balanced accuracy (flatten)")
axl.set_title("The two distributions overlap;\nthe reference was the top draw", fontsize=10.8, loc="left", pad=14)
axl.grid(axis="y", alpha=.25)

# ---------------- MIDDLE: the gap, before and after ----------------
se = np.sqrt(v3v.var(ddof=1) / len(v3v) + v4v.var(ddof=1) / len(v4v))
gap_now = v3v.mean() - v4v.mean()
axm.bar([0], [0.0687], width=0.5, color="#e34948", zorder=3, label="as reported")
axm.bar([1], [gap_now], width=0.5, color="#2a78d6", zorder=3, label="with error bars")
axm.errorbar([1], [gap_now], yerr=[se], fmt="none", ecolor="#1a4a80", capsize=7, lw=2, zorder=4)
axm.axhline(0, color="black", lw=1.1, zorder=2)
axm.annotate("+0.0687", xy=(0, 0.0687), xytext=(0, 6), textcoords="offset points",
             ha="center", fontsize=10, fontweight="bold", color="#a03030")
axm.annotate(f"{gap_now:+.4f}\n({abs(gap_now)/se:.1f} se)", xy=(1, gap_now),
             xytext=(0, -34), textcoords="offset points", ha="center", fontsize=9.5,
             fontweight="bold", color="#1a4a80")
axm.set_xticks([0, 1])
axm.set_xticklabels([f"v3 n=1\nvs v4 n=3", f"v3 n={len(v3v)}\nvs v4 n={len(v4v)}"], fontsize=9)
axm.set_ylabel("v3 − v4 held-out mean")
axm.set_ylim(-0.045, 0.095)
axm.set_title("The gap does not shrink —\nit vanishes", fontsize=10.8, loc="left")
axm.grid(axis="y", alpha=.25)

# ---------------- RIGHT: recalibration ----------------
sd_single = v3v.std(ddof=1)
claims = [("§6b masked pretraining vs random", +0.1170),
          ("§5f v3 vs v4 corpus (as reported)", +0.0687),
          ("§7b peak weighting (v3, matched)", -0.0423),
          ("§5b patch 128 vs 1024", -0.0416),
          ("§5b patch 256 vs 1024", -0.0335),
          ("§7 block masking", -0.0298),
          ("§5d ps2048 vs ps1024", +0.0202),
          ("§7 peak weighting (v4, unmatched)", +0.0108),
          ("§5d d256L6 vs ps1024", +0.0057)]
names = [c[0] for c in claims]
vals = np.array([abs(c[1]) for c in claims])
ys = np.arange(len(claims))[::-1]
cols = ["#2a9d55" if v >= 2 * sd_single else "#b36a00" if v >= sd_single else "#e34948" for v in vals]
axr.barh(ys, vals, color=cols, height=.62, zorder=3)
axr.axvline(sd_single, color="#333", ls="--", lw=1.3, zorder=4)
axr.axvline(2 * sd_single, color="#333", ls=":", lw=1.1, zorder=4)
for y, (n, v) in zip(ys, claims):
    axr.annotate(f"{v:+.3f}", xy=(abs(v), y), xytext=(4, -3), textcoords="offset points", fontsize=8)
axr.text(sd_single, -1.15, f"1 sd\n{sd_single:.3f}", fontsize=7.6, ha="center", va="bottom", color="#333")
axr.text(2 * sd_single, -1.15, "2 sd", fontsize=7.6, ha="center", va="bottom", color="#333")
axr.set_ylim(-1.35, len(claims) - 0.35)
axr.set_yticks(ys); axr.set_yticklabels(names, fontsize=8.4)
axr.set_xlim(0, 0.135)
axr.set_xlabel("|Δ held-out mean|")
axr.set_title("Only ONE single-run claim clears 2 sd", fontsize=10.8, loc="left")
axr.grid(axis="x", alpha=.25)

fig.text(0.005, -0.055,
         f"Each point is one pretraining run, identical configuration, differing only by seed. Measured single-run sd on the held-out mean: {sd_single:.3f} -- more than double the 0.020 "
         f"previously assumed, which\nhad been derived from three v4 runs whose per-target errors happened to cancel inside the mean. Per-target sd is larger still (Barth 0.076, MTBLS326 0.045). "
         f"PAIRED within-checkpoint results are\nunaffected and still stand -- §4b head fix (+0.120), §5c pooling (+0.03..+0.13), §14 jigsaw/joint pooling -- because they compare transforms on a "
         f"FIXED checkpoint and carry no training variance.",
         fontsize=7.3, ha="left", va="top", color="#333")

out = Path("results/plots/all_datasets_summary_v4"); out.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out / "fig16_exp15_seed_study.png", bbox_inches="tight", facecolor="white")
print(f"Wrote {out / 'fig16_exp15_seed_study.png'}")
