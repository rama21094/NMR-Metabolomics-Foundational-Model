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
    """docs §4b (head) and §5c (pooling) -- both paired.

    LEFT PANEL CORRECTION (2026-08-21). This panel used to plot the §4a probe
    table: `ssl_head_best` (the best FINE-TUNED head) against the LogReg probe
    on frozen features. That is not a paired comparison -- the two bars had
    different backbones, because one of them had been fine-tuned -- and it made
    the panel disagree with its own title: it came out +0.057 and 4 of 5, with
    MTBLS326 negative, while the title claimed §4b's +0.120 and 5 of 5.

    The panel now plots §4b's actual paired test from
    results/analysis/linear_probe_frozen/linear_probe_results.csv:
    `ssl_head_frozen` vs `linear_probe_bal_acc` -- identical backbone,
    identical pooling, identical folds, only the fitting of the final linear
    map differs. That is +0.120 mean, positive on 5 of 5, matching the title.

    NHEAD CAVEAT. Both bars here read the v3 ps1024 checkpoint at nhead=8,
    which is NOT the value it trained with (nhead=4); the checkpoint does not
    record nhead and loads silently under either. The comparison stays valid
    because it is paired -- both bars share the same (mis-read) backbone and
    only the head differs -- but the absolute heights are not comparable with
    the right panel, which reads the same checkpoint at the true nhead=4. This
    is why the left panel's "after" is NOT the right panel's "before".
    """
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIGW, 4.6))
    x = np.arange(5)
    w = 0.36

    # §4b, masking family, TARGETS order. Frozen head vs LogReg probe on the
    # identical frozen features -- the strictly paired test.
    head_before = [0.655, 0.796, 0.530, 0.730, 0.653]
    head_after = [0.770, 0.944, 0.607, 0.833, 0.810]
    a1.bar(x - w / 2, head_before, w, label="Trained DNN head (frozen backbone)", color=GREY)
    a1.bar(x + w / 2, head_after, w, label="LogReg probe, same features", color=DEEP)
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
    # NOTE: the two panels are separate paired experiments read at different
    # nhead (8 left, 4 right), so their bars deliberately do not chain --
    # the left panel's "after" is NOT the right panel's "before".

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
    XP = [0.0, 1.45]        # group centres, spread so captions do not collide
    for i, (vals, col) in enumerate([(v3, DEEP), (v4, TEAL)]):
        i = XP[i]
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
    a1.set_xticks(XP)
    a1.set_xticklabels(["v3 corpus\n(n=5 seeds)", "v4 corpus\n(n=5 seeds)"])
    a1.set_xlim(-0.40, 2.42)
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


# ------------------------------- 9. Barth: the one SSL win was a lucky seed
def fig_barth_seeds():
    """Experiment #18. Barth SSL across 5 v3 + 5 v4 seeds vs classical LogReg.

    Data: results/analysis/barth_seeds_vs_classical/barth_seed_runs.csv, built
    by code/analysis/summarize_barth_ssl_vs_classical_seeds.py.
    """
    from matplotlib.lines import Line2D

    runs = pd.read_csv(ROOT / "results/analysis/barth_seeds_vs_classical/barth_seed_runs.csv")
    groups = pd.read_csv(ROOT / "results/analysis/barth_seeds_vs_classical/barth_seed_groups.csv")
    CLASSICAL = 0.704969

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIGW, 4.62),
                                 gridspec_kw={"width_ratios": [1.32, 1.0]})

    # ---------- left: every seed, flatten pooling (what the 0.806 was quoted at)
    f = runs[runs.pooling == "flatten"]
    a1.axhspan(CLASSICAL - 0.045, CLASSICAL + 0.045, color=NAVY, alpha=0.09, zorder=0)
    a1.axhline(CLASSICAL, color=NAVY, lw=2.4, zorder=2)
    centres, tick_labels = {}, []
    for i, corpus in enumerate(("v3", "v4")):
        g = f[f.corpus == corpus].reset_index(drop=True)
        base = i * 1.50
        xs = base + np.arange(len(g)) * 0.24
        centres[corpus] = xs.mean()
        col = DEEP if corpus == "v3" else TEAL
        for x, (_, r) in zip(xs, g.iterrows()):
            if r.is_quoted_arm:
                a1.scatter([x], [r.balanced_accuracy], s=300, marker="*", color=CORAL,
                           edgecolor="white", linewidth=1.3, zorder=5)
            else:
                a1.scatter([x], [r.balanced_accuracy], s=130, color=col, zorder=4)
        gm = groups[(groups.pooling == "flatten") & (groups.corpus == corpus)].iloc[0]
        a1.plot([xs[0] - 0.11, xs[-1] + 0.11], [gm["mean"]] * 2,
                color=GOLD, lw=2.6, ls="--", zorder=3)
        tick_labels.append(f"{corpus} corpus\nmean {gm['mean']:.3f}\n"
                           f"{gm.n_beating_classical}/5 beat classical")

    # label gutter on the right, so nothing sits on top of the lines
    a1.text(2.66, 0.716, "classical LogReg\n0.705", fontsize=14, ha="left",
            va="bottom", fontweight="bold", color=NAVY, linespacing=1.35)
    a1.annotate("the single run every\nheadline was quoted from",
                xy=(0.0, 0.806), xytext=(0.60, 0.845), fontsize=14,
                fontweight="bold", color=CORAL, ha="left",
                arrowprops=dict(arrowstyle="->", color=CORAL, lw=2.0))
    a1.set_xlim(-0.30, 3.62)
    a1.set_ylim(0.50, 0.885)
    a1.set_xticks([centres["v3"], centres["v4"]])
    a1.set_xticklabels(tick_labels, fontsize=12.5, linespacing=1.4)
    a1.set_ylabel("Barth balanced accuracy")
    a1.set_title("Barth, one point per pretraining seed (flatten pooling)",
                 fontweight="bold", fontsize=16)
    a1.legend(handles=[
        Line2D([], [], marker="*", ls="none", ms=16, color=CORAL, label="quoted run"),
        Line2D([], [], marker="o", ls="none", ms=8, color=DEEP, label="other seeds"),
        Line2D([], [], ls="--", lw=2.2, color=GOLD, label="mean (n=5)"),
    ], loc="lower right", fontsize=12.5, frameon=False, handletextpad=0.5,
        borderaxespad=0.4, labelspacing=0.3)
    a1.grid(axis="y", alpha=0.3, ls="--")
    a1.set_axisbelow(True)

    # ---------- right: group deltas vs classical, all four groups
    order = [("flatten", "v4"), ("flatten", "v3"), ("mean_pool", "v4"), ("mean_pool", "v3")]
    ys = np.arange(len(order))
    vals, errs, names, wins = [], [], [], []
    for pool, corpus in order:
        g = groups[(groups.pooling == pool) & (groups.corpus == corpus)].iloc[0]
        vals.append(g.delta_vs_classical)
        errs.append(g.se)
        names.append(f"{pool}\n{corpus}")
        wins.append(f"{g.n_beating_classical}/5")
    a2.axvspan(-0.045, 0.045, color="0.86", alpha=0.75, zorder=0)
    a2.axvline(0, color="k", lw=1.8, zorder=2)
    a2.barh(ys, vals, height=0.58, xerr=errs, capsize=6,
            color=[GREEN if v > 0 else CORAL for v in vals],
            error_kw=dict(elinewidth=1.9, capthick=1.7), zorder=3)
    for y, w in zip(ys, wins):
        a2.text(0.148, y, w, fontsize=15, fontweight="bold", va="center", ha="right")
    a2.text(0.148, -0.62, "seeds >\nclassical", fontsize=12.5, ha="right",
            va="center", color="0.35", linespacing=1.3)
    a2.set_yticks(ys)
    a2.set_yticklabels(names, fontsize=14)
    a2.set_xlim(-0.135, 0.155)
    a2.set_ylim(3.75, -1.05)
    a2.set_xlabel("Δ vs classical LogReg")
    a2.set_title("Every group sits inside\nthe noise band", fontweight="bold", fontsize=16)
    a2.text(-0.130, 3.52, "grey band = ±0.045 noise floor (§15)",
            fontsize=13, color="0.30", ha="left", va="center")
    a2.grid(axis="x", alpha=0.3, ls="--")
    a2.set_axisbelow(True)
    for ax in (a1, a2):
        ax.tick_params(labelsize=14.5)
    save(fig, "gm09_barth_seeds.png")


# ------------------------------ 10. reconstruction vs non-learned baselines
def fig_recon_baselines():
    """Experiment #19. Is masked reconstruction hard? Data from
    code/analysis/reconstruction_baselines.py."""
    from matplotlib.patches import Patch

    summ = pd.read_csv(ROOT / "results/analysis/reconstruction_baselines/recon_baselines_summary.csv")
    red = pd.read_csv(ROOT / "results/analysis/reconstruction_baselines/corpus_redundancy.csv")
    rv = dict(zip(red.metric, red.value))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIGW, 4.62),
                                 gridspec_kw={"width_ratios": [1.42, 1.0]})

    preds = [("linear_interp", "linear\ninterp.", GREY),
             ("corpus_mean", "corpus\nmean", GOLD),
             ("pca50_from_visible", "PCA-50", TEAL),
             ("nn_copy", "copy a\nneighbour", DEEP),
             ("dnn", "trained\nDNN", CORAL)]
    ratios = sorted(summ.mask_ratio.unique())
    alphas = [0.42, 0.68, 1.0]
    x = np.arange(len(preds))
    w = 0.26
    for i, ratio in enumerate(ratios):
        g = summ[summ.mask_ratio == ratio].set_index("predictor")
        vals = [g.loc[k, "r_masked_mean"] for k, _, _ in preds]
        a1.bar(x + (i - 1) * w, vals, w, color=[c for _, _, c in preds],
               alpha=alphas[i], edgecolor="white", linewidth=0.8, zorder=3)
        if ratio == ratios[-1]:
            for xi, v in zip(x, vals):
                a1.text(xi + (i - 1) * w, v + 0.02, f"{v:.2f}", ha="center",
                        fontsize=13.5, fontweight="bold")
    a1.set_xticks(x)
    a1.set_xticklabels([lab for _, lab, _ in preds], fontsize=14)
    a1.set_ylabel("Pearson r on the hidden bins")
    a1.set_ylim(0, 1.16)
    a1.set_title("Copying another spectrum already scores 0.90",
                 fontweight="bold", fontsize=16)
    dnn60 = summ[(summ.mask_ratio == ratios[-1]) & (summ.predictor == "dnn")].r_masked_mean.iloc[0]
    nn60 = summ[(summ.mask_ratio == ratios[-1]) & (summ.predictor == "nn_copy")].r_masked_mean.iloc[0]
    a1.text(-0.42, 1.09, f"At {ratios[-1]:.0%} hidden, the network beats\n"
            f"'copy a neighbour' by only {dnn60 - nn60:+.3f}",
            fontsize=14.5, fontweight="bold", color=CORAL, va="top", ha="left",
            linespacing=1.4)
    a1.legend(handles=[Patch(facecolor=GREY, alpha=a, label=f"{r:.0%} hidden")
                       for a, r in zip(alphas, ratios)],
              loc="upper center", bbox_to_anchor=(0.5, -0.155), ncol=3,
              fontsize=14, frameon=False, handlelength=1.6, columnspacing=1.6)
    a1.grid(axis="y", alpha=0.3, ls="--")
    a1.set_axisbelow(True)
    a1.tick_params(labelsize=14)

    # ---------- right: why it is easy
    a2.axis("off")
    T = a2.transAxes
    a2.text(0.0, 1.0, "Why it is easy", fontsize=18, fontweight="bold",
            color=NAVY, va="top", transform=T)
    pc5 = rv["pca_cum_var_5pc"]
    a2.text(0.0, 0.880, f"{100 * pc5:.0f}% of all corpus variance sits\nin just 5 principal components",
            fontsize=15, va="top", transform=T, linespacing=1.45)
    a2.text(0.0, 0.695,
            f"20 PCs → {100 * rv['pca_cum_var_20pc']:.0f}%    "
            f"50 PCs → {100 * rv['pca_cum_var_50pc']:.0f}%",
            fontsize=14.5, va="top", color=DEEP, fontweight="bold", transform=T)
    a2.text(0.0, 0.578, f"The median spectrum's nearest\nneighbour correlates at "
            f"r = {rv['median_best_match_r']:.3f}",
            fontsize=15, va="top", transform=T, linespacing=1.45)
    a2.text(0.0, 0.408,
            f"{100 * rv['frac_rows_with_neighbour_r_gt_0.95']:.0f}% of spectra have a neighbour\n"
            f"at r > 0.95;  {100 * rv['frac_rows_with_neighbour_r_gt_0.9999']:.1f}% are near-duplicates",
            fontsize=14.5, va="top", color=CORAL, fontweight="bold", transform=T,
            linespacing=1.5)
    a2.text(0.0, 0.228, "Every serum CPMG spectrum resembles\n"
            "every other, so the hidden bins are\n"
            "already implied by the visible ones. This\n"
            "task cannot force the model to learn\n"
            "anything disease-discriminative.",
            fontsize=14.5, va="top", transform=T, linespacing=1.45, color="0.25")
    save(fig, "gm10_recon_baselines.png")


# ------------------------------- 11. batch-confound audit, all five targets
def fig_batch_audit():
    """Experiment #11, all cohorts. Data from code/analysis/batch_confound_audit.py."""
    D = ROOT / "results/analysis/batch_confound_audit"
    design = pd.read_csv(D / "design_audit.csv")
    noise = pd.read_csv(D / "noise_tests_holm.csv")
    summ = pd.read_csv(D / "summary.csv")

    order = ["Barth", "MTBLS563", "BrC-T2D cancer", "BrC-T2D diabetes", "MTBLS326"]
    design = design.set_index("cohort").loc[order].reset_index()
    summ = summ.set_index("cohort").loc[order].reset_index()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIGW, 4.62),
                                 gridspec_kw={"width_ratios": [1.0, 1.12]})
    y = np.arange(len(order))
    COL = {"clean": GREEN, "caveat (order)": GOLD,
           "AMBIGUOUS (SNR)": TEAL, "CONFOUNDED": CORAL}
    cols = [COL.get(v, GREY) for v in summ.verdict]

    # ---- left: does acquisition order alone predict the label?
    a1.axvspan(0.5, 0.75, color="0.90", alpha=0.8, zorder=0)
    a1.axvline(0.5, color="k", lw=1.7, zorder=2)
    a1.barh(y, design.order_alone_auc, height=0.6, color=cols, zorder=3)
    for yi, v in zip(y, design.order_alone_auc):
        a1.text(v + 0.012, yi, f"{v:.2f}", va="center", fontsize=14.5, fontweight="bold")
    a1.set_yticks(y)
    a1.set_yticklabels(order, fontsize=14)
    a1.set_xlim(0.42, 1.16)
    a1.set_ylim(4.6, -0.6)
    a1.set_xlabel("AUC of acquisition order alone\n"
                  "0.5 = interleaved   ·   1.0 = separate blocks", fontsize=14)
    a1.set_title("Test 1 — is the design balanced?", fontweight="bold", fontsize=16)
    a1.grid(axis="x", alpha=0.3, ls="--")
    a1.set_axisbelow(True)
    a1.tick_params(labelsize=14.5)

    # ---- right: can metabolite-free regions classify the label?
    a2.axvline(0.5, color="k", lw=1.7, ls="--", zorder=2)
    for i, coh in enumerate(order):
        g = noise[noise.cohort == coh]
        for _, r in g.iterrows():
            rm = "rowMinMax" in str(r["array"])
            off = -0.17 if rm else 0.17
            sig = r.p_holm < 0.05
            a2.barh(i + off, r.balanced_accuracy, height=0.30,
                    color=cols[i] if sig else "0.78",
                    hatch="" if rm else "///", edgecolor="white", zorder=3)
            a2.text(r.balanced_accuracy + 0.012, i + off,
                    f"{r.balanced_accuracy:.2f}" + ("*" if sig else ""),
                    va="center", fontsize=13, fontweight="bold" if sig else "normal")
    a2.set_yticks(y)
    a2.set_yticklabels(order, fontsize=14)
    a2.set_xlim(0.20, 0.92)
    a2.set_ylim(4.6, -0.6)
    a2.set_xlabel("balanced accuracy, metabolite-free regions\n"
                  "upper = rowMinMax  ·  lower = un-normalised\n"
                  "dashed = chance  ·  * survives Holm (m=10)", fontsize=12.5)
    a2.set_title("Test 2 — can noise alone classify it?", fontweight="bold", fontsize=16)
    a2.grid(axis="x", alpha=0.3, ls="--")
    a2.set_axisbelow(True)
    a2.tick_params(labelsize=14.5)

    save(fig, "gm11_batch_audit.png")


# ---------------------------------------------------------------------------
# Every figure call lives here, at the END of the file. It used to sit in the
# middle, which silently skipped every function defined below it -- gm08 and
# gm07b were not being regenerated at all.
if __name__ == "__main__":
    fig_headline()
    fig_free_wins()
    fig_pretraining_gain()
    fig_seed_study()
    fig_recalibration()
    fig_fewshot_curves()
    fig_fewshot_paired()
    fig_fewshot_paired_5panel()
    fig_reconstruction()
    fig_barth_seeds()
    fig_recon_baselines()
    fig_batch_audit()
    print("\nall deck figures written to", OUT)
