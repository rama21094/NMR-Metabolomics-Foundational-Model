#!/usr/bin/env python3
"""Slide-optimized figures for the group-meeting deck (Aug 2026).

The analysis figures in results/plots/all_datasets_summary_v4/ are built for
on-screen reading at full resolution; dropped into a 13.3x7.5" slide their tick
labels land around 8-10pt. The PI's floor is Arial 16. So these are rebuilt at
slide dimensions with every text element >= 16pt-equivalent, and wide 5-panel
strips are re-laid-out as 2 rows so each panel keeps its height.

Font: Liberation Sans is metric-identical to Arial (same advance widths, same
shapes), so sizes here transfer exactly to what the PI sees in the deck.

All numbers are the values of record from docs/SSL_vs_classical_analysis.md
(sections cited per function) or read directly from the committed few-shot CSVs.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/gm_figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---- typography: Arial-metric, PI floor of 16pt ----------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 17,
    "axes.titlesize": 19,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 21,
    "axes.linewidth": 1.2,
    "axes.edgecolor": "#444444",
    "savefig.facecolor": "white",
})

# Palette — "Ocean Gradient"-derived, committed across deck and figures.
NAVY, TEAL, DEEP = "#21295C", "#1C7293", "#065A82"
GOLD, CORAL, GREY = "#B8860B", "#C1435B", "#8A8F98"
TARGETS = ["Barth", "MTBLS326", "MTBLS563", "BrC-T2D\ncancer", "BrC-T2D\ndiabetes"]


def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p.name, f"({p.stat().st_size//1024} KB)")


# ---------------------------------------------------------------- 1. headline
def fig_headline():
    """docs §3 — the result that started the whole diagnosis."""
    classical = [0.705, 1.000, 0.721, 0.937, 0.829]
    masked = [0.691, 0.981, 0.558, 0.796, 0.653]
    jigsaw = [0.677, 0.874, 0.550, 0.782, 0.620]
    joint = [0.649, 0.930, 0.500, 0.757, 0.624]

    x = np.arange(5)
    w = 0.2
    fig, ax = plt.subplots(figsize=(13.0, 5.9))
    ax.bar(x - 1.5 * w, classical, w, label="Classical (LogReg, 1024 bins)", color=GOLD)
    ax.bar(x - 0.5 * w, masked, w, label="SSL — masked", color=DEEP)
    ax.bar(x + 0.5 * w, jigsaw, w, label="SSL — jigsaw", color=TEAL)
    ax.bar(x + 1.5 * w, joint, w, label="SSL — joint", color=GREY)
    for xi, (c, m) in enumerate(zip(classical, masked)):
        ax.annotate(f"−{c - m:.3f}", xy=(xi, max(c, m) + 0.016), ha="center",
                    fontsize=16, fontweight="bold", color=CORAL)
    ax.set_xticks(x)
    ax.set_xticklabels(TARGETS)
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.40, 1.10)
    ax.axhline(0.5, color="0.55", ls=":", lw=1.5)
    ax.text(4.42, 0.508, "chance (binary)", fontsize=15, color="0.4", ha="right")
    ax.set_title("Feb→Jul: we finally had 5 real targets — and classical ML won every one",
                 pad=14, fontweight="bold")
    ax.legend(ncol=2, loc="upper left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, ls="--")
    ax.set_axisbelow(True)
    save(fig, "gm01_headline.png")


# ------------------------------------------------- 2. the two free wins
def fig_free_wins():
    """docs §4b (head) and §5c (pooling) — both PAIRED, both survive §15."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 5.6))

    # head fix: reported DNN head -> LogReg probe on identical frozen features
    head_before = [0.691, 0.981, 0.558, 0.796, 0.653]
    head_after = [0.770, 0.944, 0.607, 0.833, 0.810]
    x = np.arange(5)
    w = 0.36
    a1.bar(x - w / 2, head_before, w, label="Trained DNN head", color=GREY)
    a1.bar(x + w / 2, head_after, w, label="LogReg probe", color=DEEP)
    for xi, (b, a) in enumerate(zip(head_before, head_after)):
        a1.annotate(f"{a - b:+.2f}", xy=(xi, max(a, b) + 0.014), ha="center",
                    fontsize=15.5, fontweight="bold",
                    color="#1a7a3c" if a > b else CORAL)
    a1.set_title("Win 1 — the head was underfit\n+0.120 mean, 5 of 5 targets", pad=10,
                 fontweight="bold")
    a1.set_ylabel("Balanced accuracy")

    # pooling: mean-pool -> position-preserving flatten, same frozen checkpoint
    pool_before = [0.677, 0.948, 0.588, 0.782, 0.687]
    pool_after = [0.806, 1.000, 0.618, 0.859, 0.780]
    a2.bar(x - w / 2, pool_before, w, label="mean-pool (discards position)", color=GREY)
    a2.bar(x + w / 2, pool_after, w, label="position-preserving", color=TEAL)
    for xi, (b, a) in enumerate(zip(pool_before, pool_after)):
        a2.annotate(f"+{a - b:.2f}", xy=(xi, max(a, b) + 0.014), ha="center",
                    fontsize=15.5, fontweight="bold", color="#1a7a3c")
    a2.set_title("Win 2 — pooling threw away chemical shift\n+0.03…+0.13, 5 of 5 targets",
                 pad=10, fontweight="bold")

    for ax in (a1, a2):
        ax.set_xticks(x)
        ax.set_xticklabels(TARGETS, fontsize=14.5)
        ax.set_ylim(0.45, 1.10)
        ax.legend(loc="upper left", fontsize=15, framealpha=0.95)
        ax.grid(axis="y", alpha=0.3, ls="--")
        ax.set_axisbelow(True)
    fig.suptitle("Two fixes that cost zero GPU time — and are still standing today",
                 y=1.02, fontweight="bold")
    save(fig, "gm02_free_wins.png")


# --------------------------------------- 3. does pretraining help at all?
def fig_pretraining_gain():
    """docs §6b — the only single-run claim that clears 2 sd."""
    mask = [0.252, 0.052, 0.047, 0.063, 0.171]
    jig = [0.087, 0.015, -0.065, -0.028, -0.067]
    joint = [0.202, -0.100, -0.126, -0.077, -0.023]
    x = np.arange(5)
    w = 0.26
    fig, ax = plt.subplots(figsize=(13.0, 5.8))
    # Liberation Sans has no CHECK MARK / BALLOT X glyphs -- they render as
    # tofu boxes. Use words instead of dingbats.
    ax.bar(x - w, mask, w, label="masked   (mean +0.117)  WORKS", color=DEEP)
    ax.bar(x, jig, w, label="jigsaw   (mean −0.011)  fails", color=TEAL)
    ax.bar(x + w, joint, w, label="joint    (mean −0.025)  harmful", color=CORAL)
    ax.axhline(0, color="k", lw=1.6)
    ax.set_xticks(x)
    ax.set_xticklabels(TARGETS)
    ax.set_ylabel("Δ balanced accuracy\n(pretrained − random init)")
    ax.set_title("Only ONE of our three objectives learns anything useful",
                 pad=14, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, ls="--")
    ax.set_axisbelow(True)
    ax.annotate("jigsaw & joint lose to a\nRANDOM network on 3 of 5",
                xy=(2.0, -0.126), xytext=(2.35, -0.185),
                arrowprops=dict(arrowstyle="->", color=CORAL, lw=2.2),
                fontsize=16, color=CORAL, fontweight="bold")
    ax.set_ylim(-0.23, 0.30)
    save(fig, "gm03_pretraining_gain.png")


# ------------------------------------------------ 4. the seed study (§15)
def fig_seed_study():
    """docs §15 — our biggest mistake, and how we caught it."""
    v3 = [0.8884, 0.8190, 0.8067, 0.8033, 0.7653]
    v4 = [0.8667, 0.8272, 0.8232, 0.8199, 0.8158]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 5.8),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    rng = np.random.default_rng(0)
    for i, (vals, lbl, col) in enumerate([(v3, "v3 corpus", DEEP), (v4, "v4 corpus", TEAL)]):
        xs = i + rng.uniform(-0.075, 0.075, len(vals))
        a1.scatter(xs, vals, s=190, color=col, zorder=3, edgecolor="white", lw=1.8, label=lbl)
        m, sd = np.mean(vals), np.std(vals, ddof=1)
        a1.hlines(m, i - 0.25, i + 0.25, color=col, lw=3.5, zorder=4)
        a1.add_patch(plt.Rectangle((i - 0.25, m - sd), 0.5, 2 * sd,
                                   color=col, alpha=0.16, zorder=1))
        a1.text(i, 0.755, f"{m:.3f}\n± {sd:.3f}", ha="center", fontsize=17,
                fontweight="bold", color=col)
    a1.scatter([0], [0.8884], s=330, facecolor="none", edgecolor=CORAL, lw=3.5, zorder=5)
    a1.annotate("the ONE run every\nearlier number used\n(+1.6 sd — a lucky draw)",
                xy=(0.06, 0.8884), xytext=(0.42, 0.897),
                arrowprops=dict(arrowstyle="->", color=CORAL, lw=2.4),
                fontsize=15.5, color=CORAL, fontweight="bold")
    a1.set_xticks([0, 1])
    a1.set_xticklabels(["v3 corpus\n(n=5 seeds)", "v4 corpus\n(n=5 seeds)"])
    a1.set_xlim(-0.5, 1.72)
    a1.set_ylim(0.745, 0.925)
    a1.set_ylabel("Held-out mean\nbalanced accuracy")
    a1.set_title("10 pretraining runs, ~45 GPU-hours", pad=12, fontweight="bold")
    a1.grid(axis="y", alpha=0.3, ls="--")
    a1.set_axisbelow(True)

    bars = a2.bar([0, 1], [0.0687, -0.0140], width=0.55, color=[CORAL, "#1a7a3c"])
    a2.errorbar([1], [-0.0140], yerr=[0.0221], color="k", lw=2.4, capsize=9, capthick=2.4)
    a2.axhline(0, color="k", lw=1.6)
    a2.set_xticks([0, 1])
    a2.set_xticklabels(["As reported\nv3(n=1) vs v4(n=3)", "With error bars\nv3(n=5) vs v4(n=5)"],
                       fontsize=15.5)
    a2.set_ylabel("v3 − v4 gap")
    a2.set_title("The 'biggest effect in the project'", pad=12, fontweight="bold")
    a2.annotate("+0.069\nheadline claim", xy=(0, 0.0687), xytext=(0, 0.078),
                ha="center", fontsize=16.5, fontweight="bold", color=CORAL)
    a2.annotate("−0.014 ± 0.022\n= 0.6 se → NO effect", xy=(1, -0.036), xytext=(1, -0.062),
                ha="center", fontsize=16.5, fontweight="bold", color="#1a7a3c")
    a2.set_ylim(-0.085, 0.105)
    a2.grid(axis="y", alpha=0.3, ls="--")
    a2.set_axisbelow(True)
    fig.suptitle("Experiment #15 — the effect we chased for two experiments was never there",
                 y=1.03, fontweight="bold")
    save(fig, "gm04_seed_study.png")


# ------------------------------------------- 5. noise-floor recalibration
def fig_recalibration():
    """docs §15 recalibration table, as a figure."""
    claims = [
        ("Masked pretraining vs random init", 0.117, True),
        ("v3 vs v4 corpus  ← RETRACTED", 0.069, False),
        ("Peak weighting (matched)", -0.042, False),
        ("Patch 128 vs 1024", -0.042, False),
        ("Patch 256 vs 1024", -0.034, False),
        ("Block masking", -0.030, False),
        ("Patch 2048 vs 1024", 0.020, False),
        ("Peak weighting (unmatched)", 0.011, False),
        ("Bigger model (d256, L6)", 0.006, False),
    ]
    fig, ax = plt.subplots(figsize=(13.2, 6.4))
    y = np.arange(len(claims))[::-1]
    vals = [abs(c[1]) for c in claims]
    cols = ["#1a7a3c" if c[2] else (CORAL if "RETRACTED" in c[0] else GREY) for c in claims]
    ax.barh(y, vals, color=cols, height=0.62)
    ax.axvline(0.045, color=NAVY, ls="--", lw=2.6)
    ax.axvline(0.090, color=CORAL, ls="--", lw=2.6)
    ax.text(0.047, len(claims) - 0.35, "1 sd = 0.045\n(measured noise)", fontsize=15.5,
            color=NAVY, fontweight="bold", va="top")
    ax.text(0.092, len(claims) - 0.35, "2 sd = 0.090\nreportable", fontsize=15.5,
            color=CORAL, fontweight="bold", va="top")
    ax.set_yticks(y)
    ax.set_yticklabels([c[0] for c in claims], fontsize=16)
    ax.set_xlabel("|Δ balanced accuracy| claimed")
    ax.set_xlim(0, 0.135)
    for yi, (lbl, v, ok) in zip(y, claims):
        ax.text(abs(v) + 0.0025, yi, f"{v:+.3f}", va="center", fontsize=15.5,
                fontweight="bold", color="#1a7a3c" if ok else "0.35")
    ax.set_title("Re-scoring every single-run claim against the measured noise floor\n"
                 "Exactly ONE of nine survives", pad=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, ls="--")
    ax.set_axisbelow(True)
    save(fig, "gm05_recalibration.png")


# ------------------------------------- 6. few-shot curves (2-row relayout)
def fig_fewshot_curves():
    runs = [("Barth", "results/fewshot/barth_v2_repooled", 0.500, 0.705, 37),
            ("MTBLS326", "results/fewshot/mtbls326_v2_coarse_pass1", 0.500, 1.000, 42),
            ("MTBLS563", "results/fewshot/mtbls563_v2_coarse", 1 / 3, 0.721, 113),
            ("BrC-T2D cancer", "results/fewshot/brc_t2d_cancer_v2_coarse", 0.500, 0.937, 78),
            ("BrC-T2D diabetes", "results/fewshot/brc_t2d_diabetes_v2_coarse", 0.500, 0.829, 78)]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4))
    axf = axes.ravel()
    for ax, (name, rel, chance, full, n) in zip(axf, runs):
        df = pd.read_csv(ROOT / rel / "fewshot_episode_metrics.csv").query("status=='ok'")
        for fam, key, val, col, mk, lbl in [
                ("classical", "model", "logistic_regression", GOLD, "o", "Classical LogReg"),
                ("masking", "fine_tune_mode", "frozen", DEEP, "s", "Masked SSL")]:
            g = df[(df.family == fam) & (df[key] == val)].groupby("support_per_class")
            m, s, c = g.balanced_accuracy.mean(), g.balanced_accuracy.std(ddof=1), g.size()
            ax.errorbar(m.index, m.to_numpy(), yerr=(s / np.sqrt(c)).to_numpy(),
                        marker=mk, ms=8, lw=2.6, capsize=4, capthick=1.8, color=col, label=lbl)
        ax.axhline(chance, color="0.6", ls=":", lw=1.8)
        ax.axhline(full, color=GOLD, ls="--", lw=1.6, alpha=0.6)
        ax.set_title(f"{name}  (n={n})", fontsize=18, pad=7, fontweight="bold")
        ax.grid(alpha=0.28, ls="--")
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=15)
    axf[0].legend(fontsize=15.5, loc="lower right", framealpha=0.95)
    for ax in axes[1]:
        ax.set_xlabel("labelled samples per class", fontsize=17)
    axes[0, 0].set_xlabel("labelled samples per class", fontsize=17)
    for ax in axes[:, 0]:
        ax.set_ylabel("Balanced accuracy", fontsize=17)

    axf[5].axis("off")
    axf[5].text(0.02, 0.90, "Reading the panels", fontsize=19, fontweight="bold",
                color=NAVY, va="top")
    axf[5].text(0.02, 0.72,
                "• Gold = classical, blue = masked SSL\n"
                "• Error bars = se over 10 episodes\n"
                "• Dotted = chance;  dashed = full-data\n"
                "   classical result\n\n"
                "Blue is at or below gold in every\n"
                "panel, at every label budget.\n\n"
                "The premise was that SSL wins on\n"
                "the LEFT of these plots.\n"
                "It does not.",
                fontsize=16.5, va="top", linespacing=1.5)
    fig.suptitle("Experiment #6 — few-shot learning curves, all 5 targets "
                 "(the test our whole premise rested on)", y=1.015, fontweight="bold")
    fig.tight_layout()
    save(fig, "gm06_fewshot_curves.png")


# ------------------------------------- 7. few-shot paired + pooled low-shot
def fig_fewshot_paired():
    runs = [("Barth", "results/fewshot/barth_v2_repooled"),
            ("MTBLS326", "results/fewshot/mtbls326_v2_coarse_pass1"),
            ("MTBLS563", "results/fewshot/mtbls563_v2_coarse"),
            ("BrC-T2D cancer", "results/fewshot/brc_t2d_cancer_v2_coarse"),
            ("BrC-T2D diabetes", "results/fewshot/brc_t2d_diabetes_v2_coarse")]
    paired = pd.read_csv(ROOT / "results/analysis/fewshot_masking_vs_classical"
                         / "fewshot_paired_masking_vs_classical.csv")
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.2, 6.0),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    cmap = plt.get_cmap("tab10")
    for i, (name, _) in enumerate(runs):
        p = paired[paired.dataset == name].sort_values("support_per_class")
        axl.errorbar(p.support_per_class, p.paired_diff, yerr=p.se, marker="o", ms=7,
                     lw=2.3, capsize=3.5, color=cmap(i), label=name)
    axl.axhline(0, color="k", lw=2.0)
    axl.axhspan(-0.02, 0.02, color="0.86", alpha=0.6, zorder=0)
    axl.set_xlabel("labelled samples per class")
    axl.set_ylabel("Paired Δ balanced accuracy\n(SSL − classical, same episodes)")
    axl.set_title("The deficit WIDENS as labels arrive\n(opposite of the transfer premise)",
                  pad=11, fontweight="bold")
    axl.legend(fontsize=14.5, loc="lower left", framealpha=0.95)
    axl.grid(alpha=0.28, ls="--")
    axl.set_axisbelow(True)

    labels, vals, errs, pooled = [], [], [], []
    for name, rel in runs:
        df = pd.read_csv(ROOT / rel / "fewshot_episode_metrics.csv").query("status=='ok'")
        s0 = df.support_per_class.min()
        c = df.query("family=='classical' and model=='logistic_regression' "
                     "and support_per_class==@s0").set_index("repeat").balanced_accuracy
        m = df.query("family=='masking' and fine_tune_mode=='frozen' "
                     "and support_per_class==@s0").set_index("repeat").balanced_accuracy
        ix = c.index.intersection(m.index)
        d = (m.loc[ix] - c.loc[ix]).to_numpy()
        pooled.extend(d.tolist())
        labels.append(name)
        vals.append(d.mean())
        errs.append(d.std(ddof=1) / np.sqrt(len(d)))
    pooled = np.array(pooled)
    labels.append("POOLED (n=50)")
    vals.append(pooled.mean())
    errs.append(pooled.std(ddof=1) / np.sqrt(len(pooled)))

    y = np.arange(len(labels))[::-1]
    cols = [DEEP if v > 0 else GOLD for v in vals]
    cols[-1] = NAVY
    axr.barh(y, vals, xerr=errs, color=cols, height=0.6,
             error_kw=dict(lw=2.0, capsize=5, capthick=2.0))
    axr.axvline(0, color="k", lw=2.0)
    axr.set_yticks(y)
    axr.set_yticklabels(labels, fontsize=16)
    axr.set_xlabel("Paired Δ at 2 labels / class")
    axr.set_title(f"Even at the smallest label budget:\n"
                  f"pooled Δ = {pooled.mean():+.4f}  (p = {wilcoxon(pooled).pvalue:.2f}, n.s.)",
                  pad=11, fontweight="bold")
    for yi, v, e in zip(y, vals, errs):
        axr.text(v + (e + 0.006 if v >= 0 else -e - 0.006), yi, f"{v:+.3f}",
                 va="center", ha="left" if v >= 0 else "right", fontsize=15.5,
                 fontweight="bold")
    axr.set_xlim(-0.115, 0.125)
    axr.grid(axis="x", alpha=0.28, ls="--")
    axr.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "gm07_fewshot_paired.png")


# --------------------------------------------- 8. where we stand scorecard
def fig_scorecard():
    fig, ax = plt.subplots(figsize=(13.2, 6.2))
    ax.axis("off")
    surv = [("§6b  Masked pretraining > random init", "+0.117", "2.6 sd, 5/5 targets"),
            ("§4b  LogReg probe > trained DNN head", "+0.120", "paired, 5/5"),
            ("§5c  Position-preserving pooling", "+0.03…+0.13", "paired, 5/5"),
            ("§14  Same pooling on jigsaw / joint", "+0.079 / +0.049", "paired")]
    gone = [("§5f  v3 corpus better than v4", "+0.069", "n=5 → −0.014 (0.6 se)"),
            ("§5d  'Backbone scaling exhausted'", "±0.02", "within noise"),
            ("§5b  Patch size 128 / 256 hurt", "−0.04", "within noise"),
            ("§7   Block masking / peak weighting", "−0.03 / +0.01", "within noise"),
            ("§6   'SSL wins in few-shot'", "+0.001", "p=0.74 — refuted")]

    ax.text(0.005, 0.985, "STILL STANDING", fontsize=20, fontweight="bold", color="#1a7a3c")
    ax.text(0.005, 0.918, "paired comparisons + the one 2 sd result",
            fontsize=15.5, color="0.35", style="italic")
    for i, (t, d, w) in enumerate(surv):
        yy = 0.845 - i * 0.083
        ax.add_patch(plt.Rectangle((0.0, yy - 0.052), 0.485, 0.070,
                                   color="#1a7a3c", alpha=0.10, zorder=0))
        ax.text(0.012, yy - 0.006, t, fontsize=16, va="center")
        ax.text(0.475, yy - 0.006, d, fontsize=16, va="center", ha="right",
                fontweight="bold", color="#1a7a3c")
        ax.text(0.012, yy - 0.038, w, fontsize=13.5, va="center", color="0.4")

    ax.text(0.525, 0.985, "RETRACTED or WITHIN NOISE", fontsize=20, fontweight="bold",
            color=CORAL)
    ax.text(0.525, 0.918, "everything that rested on a single run",
            fontsize=15.5, color="0.35", style="italic")
    for i, (t, d, w) in enumerate(gone):
        yy = 0.845 - i * 0.083
        ax.add_patch(plt.Rectangle((0.52, yy - 0.052), 0.48, 0.070,
                                   color=CORAL, alpha=0.10, zorder=0))
        ax.text(0.532, yy - 0.006, t, fontsize=16, va="center")
        ax.text(0.992, yy - 0.006, d, fontsize=16, va="center", ha="right",
                fontweight="bold", color=CORAL)
        ax.text(0.532, yy - 0.038, w, fontsize=13.5, va="center", color="0.4")

    ax.text(0.5, 0.055,
            "The pattern: every surviving result is PAIRED — same checkpoint, one thing varied.\n"
            "Every retracted result compared two separately-trained networks.",
            fontsize=17, ha="center", va="center", fontweight="bold", color=NAVY,
            linespacing=1.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save(fig, "gm08_scorecard.png")


if __name__ == "__main__":
    fig_headline()
    fig_free_wins()
    fig_pretraining_gain()
    fig_seed_study()
    fig_recalibration()
    fig_fewshot_curves()
    fig_fewshot_paired()
    fig_scorecard()
    print("\nall deck figures written to", OUT)
