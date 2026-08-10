#!/usr/bin/env python3
"""Figures for experiment #6: few-shot masked-SSL vs classical ML, 5 targets.

fig17 -- learning curves, one panel per target: classical LogReg vs masked SSL
         (frozen probe), mean +/- se over 10 shared episodes per support size,
         with each target's chance level and its known full-data LogReg value
         marked for reference.
fig18 -- the paired story: per-target paired difference (masking - classical,
         same episodes) vs support size, plus the pooled low-support test that
         refutes the "SSL wins when labels are scarce" hypothesis.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "results/analysis/fewshot_masking_vs_classical"
OUT = ROOT / "results/plots/all_datasets_summary_v4"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = [("Barth", "results/fewshot/barth_v2_repooled", 0.500, 0.705, 37),
        ("MTBLS326", "results/fewshot/mtbls326_v2_coarse_pass1", 0.500, 1.000, 42),
        ("MTBLS563", "results/fewshot/mtbls563_v2_coarse", 1 / 3, 0.721, 113),
        ("BrC-T2D cancer", "results/fewshot/brc_t2d_cancer_v2_coarse", 0.500, 0.937, 78),
        ("BrC-T2D diabetes", "results/fewshot/brc_t2d_diabetes_v2_coarse", 0.500, 0.829, 78)]

C_CLASSICAL, C_MASKING = "#B8860B", "#1f77b4"


def curve(df, family, key, value):
    sub = df[(df.family == family) & (df[key] == value)]
    g = sub.groupby("support_per_class").balanced_accuracy
    m, s, n = g.mean(), g.std(ddof=1), g.size()
    return m.index.to_numpy(), m.to_numpy(), (s / np.sqrt(n)).to_numpy()


def main():
    frames = {name: pd.read_csv(ROOT / rel / "fewshot_episode_metrics.csv").query("status=='ok'")
              for name, rel, *_ in RUNS}

    # ---------------------------------------------------------------- fig17
    fig, axes = plt.subplots(1, 5, figsize=(23, 4.6))
    for ax, (name, rel, chance, fulldata, n) in zip(axes, RUNS):
        df = frames[name]
        xc, yc, ec = curve(df, "classical", "model", "logistic_regression")
        xm, ym, em = curve(df, "masking", "fine_tune_mode", "frozen")
        ax.axhline(chance, color="0.6", ls=":", lw=1.2, zorder=1)
        ax.text(ax.get_xlim()[0], chance, " chance", va="bottom", ha="left", fontsize=7, color="0.45")
        ax.axhline(fulldata, color=C_CLASSICAL, ls="--", lw=1.0, alpha=0.55, zorder=1)
        ax.errorbar(xc, yc, yerr=ec, marker="o", ms=4.5, lw=1.9, capsize=2.5,
                    color=C_CLASSICAL, label="classical LogReg", zorder=3)
        ax.errorbar(xm, ym, yerr=em, marker="s", ms=4.5, lw=1.9, capsize=2.5,
                    color=C_MASKING, label="masked SSL (frozen probe)", zorder=3)
        ax.set_title(f"{name}  (n={n})", fontsize=11, pad=8)
        ax.set_xlabel("labelled samples per class")
        ax.grid(alpha=0.25, ls="--")
        ax.legend(fontsize=8, loc="lower right", framealpha=0.92)
    axes[0].set_ylabel("balanced accuracy")
    fig.suptitle("Few-shot learning curves — masked SSL never overtakes classical ML, at any label budget "
                 "(mean ± se, 10 shared episodes per point; dashed = full-data LogReg)",
                 fontsize=13, y=1.04)
    fig.tight_layout()
    p1 = OUT / "fig17_fewshot_learning_curves.png"
    fig.savefig(p1, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p1)

    # ---------------------------------------------------------------- fig18
    paired = pd.read_csv(ANALYSIS / "fewshot_paired_masking_vs_classical.csv")
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(15.5, 5.6),
                                   gridspec_kw={"width_ratios": [1.55, 1]})

    cmap = plt.get_cmap("tab10")
    for i, (name, *_rest) in enumerate(RUNS):
        p = paired[paired.dataset == name].sort_values("support_per_class")
        axl.errorbar(p.support_per_class, p.paired_diff, yerr=p.se, marker="o", ms=4.5,
                     lw=1.7, capsize=2.5, color=cmap(i), label=name, alpha=0.9)
    axl.axhline(0, color="k", lw=1.3, zorder=1)
    axl.axhspan(-0.02, 0.02, color="0.85", alpha=0.5, zorder=0)
    axl.text(axl.get_xlim()[1], 0.021, "  paired noise band (±0.02)", fontsize=7.5,
             color="0.4", ha="right", va="bottom")
    axl.set_xlabel("labelled samples per class")
    axl.set_ylabel("paired Δ balanced accuracy\n(masked SSL − classical, same episodes)")
    axl.set_title("The deficit widens as labels accumulate —\nthe opposite of the transfer-learning premise",
                  fontsize=11, pad=10)
    axl.grid(alpha=0.25, ls="--")
    axl.legend(fontsize=8.5, loc="lower left", framealpha=0.92)

    # right: pooled low-support test
    pooled = []
    labels, vals, errs = [], [], []
    for name, rel, *_ in RUNS:
        df = frames[name]
        s0 = df.support_per_class.min()
        c = df[(df.family == "classical") & (df.model == "logistic_regression")
               & (df.support_per_class == s0)]
        m = df[(df.family == "masking") & (df.fine_tune_mode == "frozen")
               & (df.support_per_class == s0)]
        mg = c[["repeat", "balanced_accuracy"]].merge(
            m[["repeat", "balanced_accuracy"]], on="repeat", suffixes=("_c", "_m"))
        d = (mg.balanced_accuracy_m - mg.balanced_accuracy_c).to_numpy()
        pooled.extend(d.tolist())
        labels.append(name)
        vals.append(d.mean())
        errs.append(d.std(ddof=1) / np.sqrt(len(d)))
    pooled = np.array(pooled)
    labels.append("POOLED (n=50)")
    vals.append(pooled.mean())
    errs.append(pooled.std(ddof=1) / np.sqrt(len(pooled)))

    ypos = np.arange(len(labels))[::-1]
    colors = [C_MASKING if v > 0 else C_CLASSICAL for v in vals]
    colors[-1] = "#333333"
    axr.barh(ypos, vals, xerr=errs, color=colors, alpha=0.85, height=0.62,
             error_kw=dict(lw=1.3, capsize=3.5))
    axr.axvline(0, color="k", lw=1.3)
    axr.set_yticks(ypos)
    axr.set_yticklabels(labels, fontsize=9)
    axr.set_xlabel("paired Δ balanced accuracy at the smallest label budget")
    pv = wilcoxon(pooled).pvalue
    axr.set_title(f"Even at 2 labels per class there is no SSL advantage\n"
                  f"pooled Δ = {pooled.mean():+.4f} ± {errs[-1]:.4f}  (p = {pv:.2f}, n.s.)",
                  fontsize=11, pad=10)
    axr.grid(alpha=0.25, ls="--", axis="x")
    for y, v, e in zip(ypos, vals, errs):
        axr.text(v + (0.004 if v >= 0 else -0.004) + (e if v >= 0 else -e), y,
                 f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
    axr.set_xlim(min(0, min(np.array(vals) - np.array(errs))) - 0.045,
                 max(np.array(vals) + np.array(errs)) + 0.045)

    fig.suptitle("Experiment #6 — the few-shot advantage the project was premised on does not appear",
                 fontsize=13.5, y=1.02)
    fig.tight_layout()
    p2 = OUT / "fig18_fewshot_paired_and_lowshot.png"
    fig.savefig(p2, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p2)


if __name__ == "__main__":
    main()
