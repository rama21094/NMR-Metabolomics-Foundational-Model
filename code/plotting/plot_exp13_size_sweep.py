#!/usr/bin/env python3
"""Experiment #13: corpus-size sweep -- does dropping rows from v3 reproduce
experiment #8's ~0.05 held-out drop, in proportion to how many rows are cut?

Left   -- held-out mean vs. fraction of v3 dropped. The 1/5/10% sweep (uniform
          random drops, decoupled from the 164 differing rows) against v3 at
          0% and experiment #8's common9506/control pair at 1.7%.
Middle -- per-target detail across the sweep, so a single-target swing isn't
          hidden inside the mean.
Right  -- the verdict: does the 1.7% point look like a smooth trend or an
          outlier relative to 1/5/10%?
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from pathlib import Path

df = pd.read_csv("results/analysis/exp13_size_sweep/patch_size_results.csv")
FL = df[df.pooling == "flatten"]

HELD = ["barth", "mtbls326", "brc_t2d_cancer"]
ALL5 = HELD + ["mtbls563", "brc_t2d_diabetes"]
LBL = {"barth": "Barth", "mtbls326": "MTBLS326", "brc_t2d_cancer": "BrC-T2D\n(cancer)",
       "mtbls563": "MTBLS563\n(sel)", "brc_t2d_diabetes": "BrC-T2D\n(diab, sel)"}
V3_REF = "ps1024_nhead4_true"
COMMON, CONTROL = "exp8_common9506", "exp8_v3rand9506_control"
SWEEP = [("exp13_v3drop1pct", 1), ("exp13_v3drop5pct", 5), ("exp13_v3drop10pct", 10)]
FLOOR_MEAN, FLOOR_TGT = 0.020, 0.035


def v(arm, ds):
    return float(FL[(FL.arm == arm) & (FL.dataset == ds)].balanced_accuracy.iloc[0])


def held_mean(arm):
    return float(np.mean([v(arm, d) for d in HELD]))


fig, (axl, axm, axr) = plt.subplots(1, 3, figsize=(17, 5.1), dpi=300,
                                    gridspec_kw={"width_ratios": [1.3, 1.55, 1.15]})

# ---------------- LEFT: held-out mean vs. fraction dropped ----------------
v3_mean = held_mean(V3_REF)
exp8_mean = float(np.mean([held_mean(COMMON), held_mean(CONTROL)]))
sweep_x = [0.0] + [p for _, p in SWEEP]
sweep_y = [v3_mean] + [held_mean(a) for a, _ in SWEEP]

axl.plot(sweep_x, sweep_y, "-o", color="#2a78d6", lw=1.8, ms=7, zorder=4,
        label="1/5/10% sweep\n(random, decoupled from the 164)")
axl.scatter([1.7], [exp8_mean], s=120, marker="D", color="#e34948", zorder=5,
           label="exp #8 common+control\nmean (the 164 rows, 1.7%)")
axl.scatter([1.7], [held_mean(COMMON)], s=40, color="#e34948", alpha=.5, zorder=3)
axl.scatter([1.7], [held_mean(CONTROL)], s=40, color="#e34948", alpha=.5, zorder=3)
for x, y in zip(sweep_x, sweep_y):
    axl.annotate(f"{y:.3f}", xy=(x, y), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=8)
axl.annotate(f"{exp8_mean:.3f}", xy=(1.7, exp8_mean), xytext=(10, -3), textcoords="offset points",
            fontsize=8, color="#a03030")
axl.axhspan(v3_mean - FLOOR_MEAN, v3_mean + FLOOR_MEAN, color="#2a9d55", alpha=.08, zorder=0)
axl.set_xlabel("% of v3 rows dropped")
axl.set_ylabel("Held-out mean balanced accuracy (flatten)")
axl.set_xlim(-0.5, 11)
axl.set_ylim(0.78, 0.91)
axl.set_title("Does the 1.7% point sit on a trend,\nor off one?", fontsize=10.8, loc="left")
axl.legend(frameon=False, fontsize=7.6, loc="lower left")
axl.grid(alpha=.25)

# ---------------- MIDDLE: per-target detail ----------------
w = 0.15
xs = np.arange(len(ALL5))
arms = [(V3_REF, "v3 (0%)", "#2a9d55"), (COMMON, "exp8 common (1.7%)", "#e34948"),
        (CONTROL, "exp8 control (1.7%)", "#f0a3a1")] + \
       [(a, f"{p}% dropped", c) for (a, p), c in zip(SWEEP, ["#9dc3ea", "#5a9bd8", "#1c4f80"])]
for i, (arm, lab, col) in enumerate(arms):
    off = (i - (len(arms) - 1) / 2) * w
    axm.bar(xs + off, [v(arm, d) for d in ALL5], width=w * .92, color=col, label=lab, zorder=3)
axm.axvspan(2.5, 4.5, color="#f2f2f2", zorder=0)
axm.set_xticks(xs); axm.set_xticklabels([LBL[d] for d in ALL5], fontsize=8.2)
axm.set_ylim(0.40, 1.08)
axm.set_ylabel("Balanced accuracy")
axm.set_title("Per-target detail across the sweep", fontsize=10.8, loc="left", pad=34)
axm.legend(frameon=False, fontsize=6.6, loc="lower left", bbox_to_anchor=(0, 1.02),
          ncol=3, columnspacing=0.8, handletextpad=0.35)
axm.grid(axis="y", alpha=.25)

# ---------------- RIGHT: verdict ----------------
axr.axis("off")
axr.set_title("Verdict", fontsize=10.8, loc="left")
max_sweep_drop = v3_mean - min(sweep_y[1:])
exp8_drop = v3_mean - exp8_mean
# A genuine size effect predicts monotone-or-flat degradation as more rows are
# dropped. 5% -> 10% RECOVERING by more than the floor is the opposite -- the
# signature of n=1-per-point sampling noise, not a real effect.
step_5_to_10 = sweep_y[3] - sweep_y[2]
recovers = step_5_to_10 > FLOOR_MEAN
verdict_confirmed = max_sweep_drop >= FLOOR_MEAN and not recovers
boxes = [
    ("Experiment #8 drop\n(1.7%, the 164 rows)", f"−{exp8_drop:.3f}", "#e34948"),
    ("5% → 10% step\n(more dropped, less bad)", f"+{step_5_to_10:.3f}", "#e34948" if recovers else "#2a9d55"),
]
y0 = 0.86
for title, val, col in boxes:
    axr.add_patch(plt.Rectangle((0.02, y0 - 0.22), 0.96, 0.20, transform=axr.transAxes,
                                facecolor=col, alpha=.10, edgecolor=col, linewidth=1.1))
    axr.text(0.08, y0 - 0.05, title, transform=axr.transAxes, fontsize=9.2, fontweight="bold", va="top")
    axr.text(0.88, y0 - 0.10, val, transform=axr.transAxes, fontsize=13, fontweight="bold",
             color=col, va="top", ha="right")
    y0 -= 0.28
if recovers:
    concl = ("NON-MONOTONIC in the wrong direction:\ngoing from 5% to 10% dropped RECOVERS\n"
            "accuracy rather than degrading it further.\nThat is what n=1-per-point noise looks\n"
            "like, not a real size effect. SIZE is NOT\nsupported; row content was already\n"
            "refuted (§8). The v3-vs-v4 gap is real\nbut its cause remains open.")
elif verdict_confirmed:
    concl = "Accuracy degrades with size across\nthe sweep -- SIZE is supported as\n(part of) the explanation."
else:
    concl = ("Cuts several times larger than 1.7%\ndon't reproduce anything close --\n"
            "SIZE is NOT confirmed. The #8 result\nwas most likely an unlucky n=1 draw.")
axr.add_patch(plt.Rectangle((0.02, y0 - 0.40), 0.96, 0.38, transform=axr.transAxes,
                            facecolor="#2a78d6", alpha=.08, edgecolor="#2a78d6", linewidth=1.1))
axr.text(0.08, y0 - 0.05, "Conclusion", transform=axr.transAxes, fontsize=9.6, fontweight="bold", va="top")
axr.text(0.08, y0 - 0.13, concl, transform=axr.transAxes, fontsize=8.4, va="top", linespacing=1.35)

fig.text(0.005, -0.06,
         "n=1 per size point (same status as experiment #8's arms). 1/5/10% arms drop rows uniformly at random from ALL of v3, independent of the 164 rows that differ from v4. "
         "The exp8 pair drops exactly\nthe 164 differing rows (common) or 164 different unchanged rows (control) -- both ~1.7% of the corpus, shown for reference on the same axis "
         "though built by a different selection rule.",
         fontsize=7.3, ha="left", va="top", color="#333")

out = Path("results/plots/all_datasets_summary_v4"); out.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out / "fig15_exp13_size_sweep.png", bbox_inches="tight", facecolor="white")
print(f"Wrote {out / 'fig15_exp13_size_sweep.png'}")
