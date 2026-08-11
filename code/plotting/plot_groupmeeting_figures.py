#!/usr/bin/env python3
"""Slide-optimized figures for the group-meeting deck (Aug 2026).

TYPOGRAPHY CONTRACT (the PI's floor is Arial 16)
------------------------------------------------
A figure's on-slide point size is its authored point size times
(displayed_width / authored_width). A 13.9in-wide figure dropped into an 11in
slide box therefore renders every label at 0.79x -- a 17pt tick becomes 13.4pt.
The first version of this script did exactly that and everything landed under
the floor.

So: every figure is authored at FIGW = 11.0in, the exact width of its image box
in build_group_meeting_deck.js, and saved WITHOUT bbox_inches="tight" (which
crops to the ink and silently changes output size, breaking the scale
calculation). Scale on slide is 1.0, so authored points are delivered points.
Layout uses constrained_layout instead of a tight bbox.

Font: Liberation Sans is metric-identical to Arial (same advance widths), so
these sizes transfer exactly to what the PI sees.

Numbers are the values of record from docs/SSL_vs_classical_analysis.md
(sections cited per function) or read from the committed few-shot CSVs.
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

FIGW = 11.0          # inches -- MUST equal the image-box width in the deck script
DPI = 200

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 17,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 19,
    "axes.linewidth": 1.2,
    "axes.edgecolor": "#444444",
    "savefig.facecolor": "white",
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.10,
    "figure.constrained_layout.w_pad": 0.10,
})

NAVY, TEAL, DEEP = "#21295C", "#1C7293", "#065A82"
GOLD, CORAL, GREY = "#B8860B", "#C1435B", "#8A8F98"
GREEN = "#1A7A3C"
TARGETS = ["Barth", "MTBLS\n326", "MTBLS\n563", "BrC-T2D\ncancer", "BrC-T2D\ndiabetes"]


def save(fig, name):
    """Save at exactly figsize -- no tight bbox, so the deck's scale stays 1.0."""
    p = OUT / name
    fig.savefig(p, dpi=DPI, facecolor="white")
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(p).size
    print(f"wrote {name:30s} {w}x{h}px = {w/DPI:5.2f} x {h/DPI:4.2f} in  AR={w/h:.4f}")


# ---------------------------------------------------------------- 1. headline
def fig_headline():
    """docs §3."""
    classical = [0.705, 1.000, 0.721, 0.937, 0.829]
    masked = [0.691, 0.981, 0.558, 0.796, 0.653]
    jigsaw = [0.677, 0.874, 0.550, 0.782, 0.620]
    joint = [0.649, 0.930, 0.500, 0.757, 0.624]

    x = np.arange(5)
    w = 0.2
    fig, ax = plt.subplots(figsize=(FIGW, 4.60))
    ax.bar(x - 1.5 * w, classical, w, label="Classical (LogReg, 1024 bins)", color=GOLD)
    ax.bar(x - 0.5 * w, masked, w, label="SSL — masked", color=DEEP)
    ax.bar(x + 0.5 * w, jigsaw, w, label="SSL — jigsaw", color=TEAL)
    ax.bar(x + 1.5 * w, joint, w, label="SSL — joint", color=GREY)
    for xi, (c, m) in enumerate(zip(classical, masked)):
        ax.annotate(f"−{c - m:.3f}", xy=(xi, max(c, m) + 0.018), ha="center",
                    fontsize=16, fontweight="bold", color=CORAL)
    ax.set_xticks(x)
    ax.set_xticklabels(TARGETS)
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.40, 1.075)
    ax.set_xlim(-0.95, 4.45)          # left margin to hold the chance label
    ax.axhline(0.5, color="0.55", ls=":", lw=1.5)
    ax.text(-0.90, 0.512, "chance", fontsize=15, color="0.35", ha="left")
    # legend BELOW the axes: at upper-left it covered MTBLS326's bars and hid
    # that group's delta label completely.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.155), ncol=4,
              fontsize=15.5, frameon=False, columnspacing=1.3, handlelength=1.5)
    ax.grid(axis="y", alpha=0.3, ls="--")
    ax.set_axisbelow(True)
    save(fig, "gm01_headline.png")


# ------------------------------------------------- 2. the two free wins
def fig_free_wins():
    """docs §4b (head) and §5c (pooling) -- both paired."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIGW, 4.6))
    x = np.arange(5)
    w = 0.36

    head_before = [0.691, 0.981, 0.558, 0.796, 0.653]
    head_after = [0.770, 0.944, 0.607, 0.833, 0.810]
    a1.bar(x - w / 2, head_before, w, label="Trained DNN head", color=GREY)
    a1.bar(x + w / 2, head_after, w, label="LogReg probe", color=DEEP)
    for xi, (b, a) in enumerate(zip(head_before, head_after)):
        a1.annotate(f"{a - b:+.2f}", xy=(xi, max(a, b) + 0.015), ha="center",
                    fontsize=15.5, fontweight="bold", color=GREEN if a > b else CORAL)
    a1.set_title("Win 1 — the head was underfit\n+0.120 mean, 5 of 5 targets", fontweight="bold")
    a1.set_ylabel("Balanced accuracy")

    pool_before = [0.677, 0.948, 0.588, 0.782, 0.687]
    pool_after = [0.806, 1.000, 0.618, 0.859, 0.780]
    a2.bar(x - w / 2, pool_before, w, label="mean-pool (discards position)", color=GREY)
    a2.bar(x + w / 2, pool_after, w, label="position-preserving", color=TEAL)
    for xi, (b, a) in enumerate(zip(pool_before, pool_after)):
        a2.annotate(f"+{a - b:.2f}", xy=(xi, max(a, b) + 0.015), ha="center",
                    fontsize=15.5, fontweight="bold", color=GREEN)
    a2.set_title("Win 2 — pooling discarded position\n+0.03…+0.13, 5 of 5 targets",
                 fontweight="bold")

    for ax in (a1, a2):
        ax.set_xticks(x)
        ax.set_xticklabels(TARGETS, fontsize=15)
        ax.set_ylim(0.45, 1.06)      # was 1.18 -- left a dead band across the top
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1,
                  fontsize=15, frameon=False)
        ax.grid(axis="y", alpha=0.3, ls="--")
        ax.set_axisbelow(True)
    save(fig, "gm02_free_wins.png")


# --------------------------------------- 3. does pretraining help at all?
def fig_pretraining_gain():
    """docs §6b."""
    mask = [0.252, 0.052, 0.047, 0.063, 0.171]
    jig = [0.087, 0.015, -0.065, -0.028, -0.067]
    joint = [0.202, -0.100, -0.126, -0.077, -0.023]
    x = np.arange(5)
    w = 0.26
    fig, ax = plt.subplots(figsize=(FIGW, 4.6))
    # words, not dingbats: Liberation Sans has no CHECK MARK / BALLOT X glyph.
    ax.bar(x - w, mask, w, label="masked  (mean +0.117)  WORKS", color=DEEP)
    ax.bar(x, jig, w, label="jigsaw  (mean −0.011)  fails", color=TEAL)
    ax.bar(x + w, joint, w, label="joint  (mean −0.025)  harmful", color=CORAL)
    ax.axhline(0, color="k", lw=1.6)
    ax.set_xticks(x)
    ax.set_xticklabels(TARGETS)
    ax.set_ylabel("Δ balanced accuracy\n(pretrained − random init)")
    ax.set_ylim(-0.20, 0.30)
    # legend below: inside the axes it clipped the top off the diabetes bar.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.145), ncol=3,
              fontsize=15, frameon=False, columnspacing=1.1, handlelength=1.4)
    ax.grid(axis="y", alpha=0.3, ls="--")
    ax.set_axisbelow(True)
    # arrow now terminates ON the MTBLS563 joint bar it refers to
    ax.annotate("jigsaw & joint lose to a\nRANDOM network on 3 of 5",
                xy=(2 + w, -0.118), xytext=(2.56, -0.175),
                arrowprops=dict(arrowstyle="->", color=CORAL, lw=2.2),
                fontsize=15.5, color=CORAL, fontweight="bold")
    save(fig, "gm03_pretraining_gain.png")


# ------------------------------------------------ 4. the seed study (§15)
def fig_seed_study():
    """docs §15."""
    v3 = [0.8884, 0.8190, 0.8067, 0.8033, 0.7653]
    v4 = [0.8667, 0.8272, 0.8232, 0.8199, 0.8158]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIGW, 4.60),
                                 gridspec_kw={"width_ratios": [1.3, 1]})
    rng = np.random.default_rng(0)
    for i, (vals, col) in enumerate([(v3, DEEP), (v4, TEAL)]):
        xs = i + rng.uniform(-0.07, 0.07, len(vals))
        a1.scatter(xs, vals, s=160, color=col, zorder=3, edgecolor="white", lw=1.8)
        m, sd = np.mean(vals), np.std(vals, ddof=1)
        a1.hlines(m, i - 0.24, i + 0.24, color=col, lw=3.5, zorder=4)
        a1.add_patch(plt.Rectangle((i - 0.24, m - sd), 0.48, 2 * sd,
                                   color=col, alpha=0.16, zorder=1))
        a1.text(i + 0.345, m, f"{m:.3f}\n± {sd:.3f}", ha="left", va="center",
                fontsize=16, fontweight="bold", color=col)
    a1.scatter([0], [0.8884], s=300, facecolor="none", edgecolor=CORAL, lw=3.2, zorder=5)
    a1.annotate("the ONE run every earlier\nnumber used (+1.6 sd)",
                xy=(0.055, 0.8884), xytext=(0.30, 0.8925), va="top",
                arrowprops=dict(arrowstyle="->", color=CORAL, lw=2.2),
                fontsize=14.5, color=CORAL, fontweight="bold")
    a1.set_xticks([0, 1])
    a1.set_xticklabels(["v3 corpus\n(n=5 seeds)", "v4 corpus\n(n=5 seeds)"])
    a1.set_xlim(-0.42, 2.20)
    a1.set_ylim(0.752, 0.918)
    a1.set_ylabel("Held-out mean\nbalanced accuracy")
    a1.set_title("10 pretraining runs, ~45 GPU-hours", fontweight="bold")
    a1.grid(axis="y", alpha=0.3, ls="--")
    a1.set_axisbelow(True)

    a2.bar([0, 1], [0.0687, -0.0140], width=0.5, color=[CORAL, GREEN])
    a2.errorbar([1], [-0.0140], yerr=[0.0221], color="k", lw=2.4, capsize=9, capthick=2.4)
    a2.axhline(0, color="k", lw=1.6)
    a2.set_xticks([0, 1])
    a2.set_xticklabels(["As reported\n(n=1 vs n=3)", "With error bars\n(n=5 vs n=5)"],
                       fontsize=15)
    # No y-label: between the panels it was struck through lengthwise by the
    # left panel's right spine. The title carries the quantity instead.
    a2.set_title("The 'biggest effect in the project'\n(v3 − v4 gap)", fontweight="bold")
    a2.set_xlim(-0.72, 1.72)      # room for the annotations below
    a2.annotate("+0.069\nheadline claim", xy=(0, 0.0715), xytext=(0, 0.0775),
                ha="center", fontsize=15.5, fontweight="bold", color=CORAL)
    a2.annotate("−0.014 ± 0.022\n(0.6 se) → NO effect", xy=(1, -0.052), xytext=(1, -0.062),
                ha="center", fontsize=15, fontweight="bold", color=GREEN)
    a2.set_ylim(-0.098, 0.112)
    a2.grid(axis="y", alpha=0.3, ls="--")
    a2.set_axisbelow(True)
    save(fig, "gm04_seed_study.png")


# ------------------------------------------- 5. noise-floor recalibration
def fig_recalibration():
    """docs §15 recalibration table."""
    claims = [
        ("Masked pretraining vs random init", 0.117, True),
        ("v3 vs v4 corpus   ← RETRACTED", 0.069, False),
        ("Peak weighting (matched)", -0.042, False),
        ("Patch 128 vs 1024", -0.042, False),
        ("Patch 256 vs 1024", -0.034, False),
        ("Block masking", -0.030, False),
        ("Patch 2048 vs 1024", 0.020, False),
        ("Peak weighting (unmatched)", 0.011, False),
        ("Bigger model (d256, L6)", 0.006, False),
    ]
    fig, ax = plt.subplots(figsize=(FIGW, 4.60))
    y = np.arange(len(claims))[::-1]
    vals = [abs(c[1]) for c in claims]
    cols = [GREEN if c[2] else (CORAL if "RETRACTED" in c[0] else GREY) for c in claims]
    ax.barh(y, vals, color=cols, height=0.62)
    ax.axvline(0.045, color=NAVY, ls="--", lw=2.4)
    ax.axvline(0.090, color=CORAL, ls="--", lw=2.4)
    ax.set_yticks(y)
    ax.set_yticklabels([c[0] for c in claims], fontsize=15.5)
    ax.set_xlabel("|Δ balanced accuracy| claimed")
    ax.set_xlim(0, 0.145)
    # Threshold captions go BELOW the bars, each under its own line: placed in
    # the plot body they landed on the -0.034 / -0.030 value labels.
    ax.set_ylim(-1.70, len(claims) - 0.35)
    # Captions sit between the last bar and the spine, offset to the RIGHT of
    # their own line so the dashed line never strikes through the text (an
    # earlier white backing box fixed that but masked part of the x-axis spine).
    ax.text(0.048, -0.92, "1 sd = 0.045\nmeasured noise", fontsize=15, color=NAVY,
            fontweight="bold", va="center", ha="left")
    ax.text(0.093, -0.92, "2 sd = 0.090\nreportable", fontsize=15, color=CORAL,
            fontweight="bold", va="center", ha="left")
    for yi, (lbl, v, ok) in zip(y, claims):
        xv = abs(v)
        # A label placed outside a bar that ends just short of 0.045 gets crossed
        # by the 1 sd line, so those go inside the bar instead.
        if 0.022 < xv < 0.058:
            ax.text(xv - 0.0035, yi, f"{v:+.3f}", va="center", ha="right",
                    fontsize=15, fontweight="bold", color="white")
        else:
            ax.text(xv + 0.005, yi, f"{v:+.3f}", va="center", ha="left",
                    fontsize=15.5, fontweight="bold", color=GREEN if ok else "0.30")
    # No figure title: the slide's own title and subtitle already say this.
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
    fig, axes = plt.subplots(2, 3, figsize=(FIGW, 4.62))
    axf = axes.ravel()
    handles = labels = None
    for ax, (name, rel, chance, full, n) in zip(axf, runs):
        df = pd.read_csv(ROOT / rel / "fewshot_episode_metrics.csv").query("status=='ok'")
        for fam, key, val, col, mk, lbl in [
                ("classical", "model", "logistic_regression", GOLD, "o", "Classical LogReg"),
                ("masking", "fine_tune_mode", "frozen", DEEP, "s", "Masked SSL")]:
            g = df[(df.family == fam) & (df[key] == val)].groupby("support_per_class")
            m, s, c = g.balanced_accuracy.mean(), g.balanced_accuracy.std(ddof=1), g.size()
            ax.errorbar(m.index, m.to_numpy(), yerr=(s / np.sqrt(c)).to_numpy(),
                        marker=mk, ms=7, lw=2.4, capsize=3.5, capthick=1.6, color=col, label=lbl)
        ax.axhline(chance, color="0.6", ls=":", lw=1.7)
        ax.axhline(full, color=GOLD, ls="--", lw=1.5, alpha=0.6)
        ax.set_title(f"{name}  (n={n})", fontsize=17, fontweight="bold")
        ax.grid(alpha=0.28, ls="--")
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=15)
        ax.set_xlabel("labels per class", fontsize=16)   # every panel, not just some
        handles, labels = ax.get_legend_handles_labels()
    # One shared y-label: per-column labels on the top-left panel collided with
    # the panel title row and got clipped at the figure edge.
    fig.supylabel("Balanced accuracy", fontsize=16.5)

    # Slot 6 holds ONLY the colour key and the reading notes. The interpretation
    # lives on the slide as a caption card -- keeping it here forced the figure
    # taller than the panels needed and left dead space under the bottom row.
    # No legend object: anchored in-axes it overlapped this text wherever the
    # text began, so the series are named in their own colours.
    axf[5].axis("off")
    axf[5].text(0.0, 0.92, "Classical LogReg", fontsize=17.5, fontweight="bold",
                color=GOLD, va="top", ha="left", transform=axf[5].transAxes)
    axf[5].text(0.0, 0.72, "Masked SSL", fontsize=17.5, fontweight="bold",
                color=DEEP, va="top", ha="left", transform=axf[5].transAxes)
    axf[5].text(0.0, 0.45,
                "Error bars = se, 10 episodes\n"
                "Dotted = chance\n"
                "Dashed = full-data classical",
                fontsize=16, va="top", ha="left", linespacing=1.6,
                transform=axf[5].transAxes)

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
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(FIGW, 4.60),
                                   gridspec_kw={"width_ratios": [1.28, 1]})
    cmap = plt.get_cmap("tab10")
    for i, (name, _) in enumerate(runs):
        p = paired[paired.dataset == name].sort_values("support_per_class")
        axl.errorbar(p.support_per_class, p.paired_diff, yerr=p.se, marker="o", ms=6.5,
                     lw=2.2, capsize=3, color=cmap(i), label=name)
    axl.axhline(0, color="k", lw=2.0)
    axl.axhspan(-0.02, 0.02, color="0.86", alpha=0.6, zorder=0)
    axl.set_xlabel("labelled samples per class")
    axl.set_ylabel("Paired Δ balanced accuracy\n(SSL − classical, same episodes)", fontsize=16)
    axl.set_title("The deficit WIDENS as labels arrive", fontweight="bold", fontsize=17)
    # legend below the axes: inside, it sat on top of the curves it describes.
    axl.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=3,
               fontsize=14.5, frameon=False, columnspacing=1.0, handlelength=1.3)
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
    axr.barh(y, vals, xerr=errs, color=cols, height=0.58,
             error_kw=dict(lw=1.9, capsize=4.5, capthick=1.9))
    axr.axvline(0, color="k", lw=2.0)
    axr.set_yticks(y)
    axr.set_yticklabels(labels, fontsize=15.5)
    axr.set_xlabel("Paired Δ at 2 labels / class")
    axr.set_title(f"At the smallest label budget\npooled Δ = {pooled.mean():+.4f}  "
                  f"(p = {wilcoxon(pooled).pvalue:.2f})",
                  fontweight="bold", fontsize=16.5)
    # Value labels go on the OPPOSITE side of zero from their bar. Right-aligning
    # a negative bar's label at its own end put "-0.039" on top of the
    # "MTBLS326" tick label.
    for yi, v in zip(y, vals):
        axr.text(0.182, yi, f"{v:+.3f}", va="center", ha="right",
                 fontsize=15, fontweight="bold", color="0.25")
    axr.set_xlim(-0.125, 0.186)
    axr.set_xticks([-0.10, 0.0, 0.05])   # 4 ticks ran into each other
    axr.grid(axis="x", alpha=0.28, ls="--")
    axr.set_axisbelow(True)
    save(fig, "gm07_fewshot_paired.png")


if __name__ == "__main__":
    fig_headline()
    fig_free_wins()
    fig_pretraining_gain()
    fig_seed_study()
    fig_recalibration()
    fig_fewshot_curves()
    fig_fewshot_paired()
    print("\nall deck figures written to", OUT)


# ------------------------- 8. masked reconstruction, regenerated -------------
def fig_reconstruction(n_eval=50):
    """Re-make February's masked-reconstruction demo with slide-legible type.

    Two changes from the carried figure, both deliberate:

    1. Authored at FIGW so the axes are readable next to text (the original was
       built full-bleed and its ticks land near 8pt when scaled to fit).
    2. Correlation is reported on the MASKED bins only, and over `n_eval`
       spectra rather than one. February's "r = 0.999" was computed across the
       whole spectrum -- which includes the ~75% of bins the model was handed
       and simply copies, so it mostly measures the copy. On the bins the model
       actually had to predict, r is materially lower and much more variable.
       That gap is the whole point of the slide, so it should be measured
       correctly rather than restated.
    """
    import sys
    import torch
    for sub in ("code/evaluation", "code/training"):
        if str(ROOT / sub) not in sys.path:
            sys.path.insert(0, str(ROOT / sub))
    from trainer_revised import NMRMaskedAutoencoder
    from barth_all_models_loocv import infer_mae_config

    CKPT = ("models/masked_ssl/combine_unique_MetaboLights_Workbench_Water_EDTA_"
            "Suppressed_rowMinMax_v3_20260725_085527_bs32_mr0.20-0.60_ps1024_best.pth")
    ck = torch.load(ROOT / CKPT, map_location="cpu", weights_only=False)
    state = ck["model_state_dict"]
    corpus = np.load(ROOT / "data/combined/combine_unique_MetaboLights_Workbench_"
                     "Water_EDTA_Suppressed_rowMinMax_v4.npy", mmap_mode="r")
    L = corpus.shape[1]
    model = NMRMaskedAutoencoder(spectrum_length=L, **infer_mae_config(state, 4, 0.0))
    model.load_state_dict(state, strict=True)
    model.eval()
    ps = model.patch_size
    npatch = L // ps

    def run(row, seed):
        x = np.asarray(corpus[row], dtype=np.float32)
        rng = np.random.default_rng(seed)
        mask = np.zeros(npatch, dtype=bool)
        mask[rng.choice(npatch, size=int(0.25 * npatch), replace=False)] = True
        masked = x.copy()
        for i in np.flatnonzero(mask):
            masked[i * ps:(i + 1) * ps] = 0.0
        with torch.no_grad():
            rec, _ = model(torch.from_numpy(masked).unsqueeze(0),
                           mask=torch.from_numpy(mask).unsqueeze(0))
        rec = rec.squeeze(0).numpy().reshape(-1)[:L]
        sel = np.zeros(L, dtype=bool)
        for i in np.flatnonzero(mask):
            sel[i * ps:(i + 1) * ps] = True
        return x, rec, mask, sel

    rows = np.linspace(0, len(corpus) - 1, n_eval, dtype=int)
    r_whole, r_mask = [], []
    for j, row in enumerate(rows):
        x, rec, _m, sel = run(int(row), 7 + j)
        r_whole.append(np.corrcoef(rec, x)[0, 1])
        r_mask.append(np.corrcoef(rec[sel], x[sel])[0, 1])
    r_whole, r_mask = np.array(r_whole), np.array(r_mask)
    # plot the spectrum whose masked-only r is closest to the median
    pick = int(rows[np.argmin(np.abs(r_mask - np.median(r_mask)))])
    pick_seed = 7 + int(np.argmin(np.abs(r_mask - np.median(r_mask))))
    x, rec, mask, sel = run(pick, pick_seed)
    this_r = np.corrcoef(rec[sel], x[sel])[0, 1]

    lo, hi = 60000, 100000
    fig, ax = plt.subplots(figsize=(FIGW, 4.30))
    for i in np.flatnonzero(mask):
        a, b = i * ps, (i + 1) * ps
        if b > lo and a < hi:
            ax.axvspan(max(a, lo), min(b, hi), color="#FFE9A8", zorder=0)
    ax.plot(np.arange(lo, hi), x[lo:hi], lw=1.9, color=NAVY, label="Original", zorder=3)
    ax.plot(np.arange(lo, hi), rec[lo:hi], lw=1.4, color=CORAL,
            label="Reconstructed", zorder=4)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Spectral point")
    ax.set_ylabel("Normalised intensity")
    ax.set_title(f"25% of patches hidden (yellow bands) — a median example, "
                 f"r = {this_r:.2f} on hidden bins", fontweight="bold", fontsize=17)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2,
              fontsize=16, frameon=False, columnspacing=1.6, handlelength=1.8)
    ax.grid(alpha=0.25, ls="--")
    ax.set_axisbelow(True)
    save(fig, "gm08_reconstruction.png")
    print(f"   [recon over n={n_eval} spectra]  whole-spectrum r = {r_whole.mean():.3f} "
          f"± {r_whole.std(ddof=1):.3f}   masked-only r = {r_mask.mean():.3f} "
          f"± {r_mask.std(ddof=1):.3f}  (range {r_mask.min():.2f}–{r_mask.max():.2f})")
