#!/usr/bin/env python3
"""Visualize why LogReg-on-binned-spectra beats the SSL backbones.

Two panels per dataset row:

  LEFT -- balanced accuracy vs. spectral resolution (bin count) for the
    classical LogReg pipeline, with three reference lines overlaid:
      * the backbone's own patch resolution (131072 / patch_size = 128
        positions), marked on the x-axis -- the finest spectral detail the
        SSL encoder can represent at all;
      * LogReg trained on the frozen SSL embedding (same classifier, learned
        representation);
      * the officially reported SSL head result from summary.csv.
    Reading these together separates a REPRESENTATION ceiling (SSL embedding
    tracks the 128-bin point and can't reach the 1024-bin point) from a
    HEAD/optimization deficit (SSL head falls below LogReg on its own
    embedding).

  RIGHT -- the same quantities as a bar chart against the label-permutation
    null band, so it's visible which results clear chance for that dataset's
    sample size.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C_BINNED = "#2a78d6"
C_EMB = "#1baf7a"
C_HEAD = "#e34948"
C_PATCH = "#eda100"
C_NULL = "#898781"

DATASET_LABELS = {
    "brc_t2d_cancer": "BrC-T2D (cancer)",
    "brc_t2d_diabetes": "BrC-T2D (diabetes)",
    "mtbls563": "MTBLS563 (3-class)",
    "mtbls326": "MTBLS326",
    "barth": "Barth",
}

# Officially reported best masked-SSL head balanced accuracy (from the v4
# summary.csv files) and its frozen-backbone variant, for reference lines.
OFFICIAL_SSL = {
    "brc_t2d_cancer": dict(
        summary="results/cv10/brc_t2d_newlabels_v4/cancer_status/summary.csv", family="masking"),
    "brc_t2d_diabetes": dict(
        summary="results/cv10/brc_t2d_newlabels_v4/diabetes_status/summary.csv", family="masking"),
    "mtbls563": dict(
        summary="results/loocv/mtbls563_all_models_v4/summary.csv", family="masking"),
    "mtbls326": dict(
        summary="results/loocv/mtbls326_masking_v4/summary.csv", family="foundation"),
    "barth": dict(
        summary="results/loocv/barth_all_models_v4/summary.csv", family="masking"),
}


def official_ssl_scores(dataset: str) -> tuple[float | None, float | None]:
    """Return (best_any_mode, frozen) masked-SSL balanced accuracy."""
    cfg = OFFICIAL_SSL.get(dataset)
    if not cfg or not Path(cfg["summary"]).exists():
        return None, None
    df = pd.read_csv(cfg["summary"])
    sub = df[df["family"] == cfg["family"]]
    if not len(sub):
        return None, None
    best = float(sub["balanced_accuracy"].max())
    frozen_rows = sub[sub["model"].astype(str).str.endswith("frozen")]
    frozen = float(frozen_rows["balanced_accuracy"].iloc[0]) if len(frozen_rows) else None
    return best, frozen


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe-csv", default="results/analysis/logreg_advantage_probe/probe_results.csv")
    parser.add_argument("--patch-positions", type=int, default=128,
                        help="131072 / patch_size; the backbone's spectral resolution")
    parser.add_argument("--output", default="results/plots/all_datasets_summary_v4/fig6_logreg_advantage_probe.png")
    args = parser.parse_args()

    df = pd.read_csv(args.probe_csv)
    datasets = [d for d in DATASET_LABELS if d in set(df["dataset"])]

    fig, axes = plt.subplots(len(datasets), 2, figsize=(12.5, 3.1 * len(datasets)), dpi=300,
                             gridspec_kw={"width_ratios": [1.35, 1]})
    if len(datasets) == 1:
        axes = np.array([axes])

    for row, dataset in enumerate(datasets):
        sub = df[df["dataset"] == dataset]
        ax_l, ax_r = axes[row, 0], axes[row, 1]

        binned = sub[sub["representation"] == "binned_abs_area"].copy()
        binned["n_bins"] = binned["detail"].str.replace(" bins", "", regex=False).astype(int)
        binned = binned.sort_values("n_bins")

        emb_mp = sub[(sub.representation == "masked_ssl_embedding") & (sub.detail == "mean_pool")]
        emb_fl = sub[(sub.representation == "masked_ssl_embedding") & (sub.detail == "flatten")]
        raw = sub[sub.representation == "raw_spectrum"]
        null_row = sub[sub.representation == "permutation_null"]
        best_head, frozen_head = official_ssl_scores(dataset)

        # ---- LEFT: resolution curve ----
        ax_l.plot(binned["n_bins"], binned["balanced_accuracy"], "o-", color=C_BINNED,
                  linewidth=2, markersize=5, label="LogReg on binned area", zorder=4)
        if len(emb_mp):
            ax_l.axhline(float(emb_mp["balanced_accuracy"].iloc[0]), color=C_EMB, linestyle="-",
                         linewidth=1.8, label="LogReg on SSL embedding (mean-pool)", zorder=3)
        if len(emb_fl):
            ax_l.axhline(float(emb_fl["balanced_accuracy"].iloc[0]), color=C_EMB, linestyle="--",
                         linewidth=1.5, alpha=0.8, label="LogReg on SSL embedding (flatten)", zorder=3)
        if best_head is not None:
            ax_l.axhline(best_head, color=C_HEAD, linestyle="-", linewidth=1.8,
                         label="SSL head as reported (best mode)", zorder=3)
        ax_l.axvline(args.patch_positions, color=C_PATCH, linestyle=":", linewidth=2,
                     label=f"backbone patch resolution ({args.patch_positions})", zorder=2)
        if len(null_row):
            p95 = float(null_row["null_p95"].iloc[0])
            ax_l.axhspan(0, p95, color=C_NULL, alpha=0.13, zorder=1)
            ax_l.axhline(p95, color=C_NULL, linewidth=1, linestyle="-", alpha=0.6, zorder=2)

        ax_l.set_xscale("log", base=2)
        ax_l.set_xticks(binned["n_bins"].tolist())
        ax_l.set_xticklabels([str(v) for v in binned["n_bins"]])
        ax_l.set_xlabel("spectral resolution (number of bins)")
        ax_l.set_ylabel("Balanced accuracy")
        n = int(sub["n_samples"].iloc[0])
        ax_l.set_title(f"{DATASET_LABELS[dataset]}  (n={n})", fontsize=11, loc="left", color="#0b0b0b")
        ax_l.set_ylim(0.3, 1.03)
        style_axes(ax_l)

        # ---- RIGHT: head vs representation bars ----
        bars = []
        if frozen_head is not None:
            bars.append(("SSL head\n(frozen)", frozen_head, C_HEAD))
        if best_head is not None:
            bars.append(("SSL head\n(best mode)", best_head, C_HEAD))
        if len(emb_mp):
            bars.append(("LogReg on\nSSL emb.", float(emb_mp["balanced_accuracy"].iloc[0]), C_EMB))
        b1024 = binned[binned["n_bins"] == 1024]
        if len(b1024):
            bars.append(("LogReg\n1024 bins", float(b1024["balanced_accuracy"].iloc[0]), C_BINNED))
        if len(raw):
            bars.append(("LogReg\nraw 131k", float(raw["balanced_accuracy"].iloc[0]), C_BINNED))

        xs = np.arange(len(bars))
        rects = ax_r.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars], width=0.66, zorder=3)
        for rect, (_, val, _) in zip(rects, bars):
            ax_r.annotate(f"{val:.3f}", xy=(rect.get_x() + rect.get_width() / 2, val),
                          xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        if len(null_row):
            p95 = float(null_row["null_p95"].iloc[0])
            ax_r.axhspan(0, p95, color=C_NULL, alpha=0.13, zorder=1)
            ax_r.axhline(p95, color=C_NULL, linewidth=1, alpha=0.6, zorder=2,
                         label="permutation null (95th pct)")
            ax_r.legend(loc="lower right", fontsize=7, frameon=False)
        ax_r.set_xticks(xs)
        ax_r.set_xticklabels([b[0] for b in bars], fontsize=8)
        ax_r.set_ylim(0.3, 1.03)
        ax_r.set_title("same representation, different classifier", fontsize=10, loc="left", color="#52514e")
        style_axes(ax_r)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.015),
               ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle("Why logistic regression outperforms the SSL backbones", fontsize=13,
                 x=0.02, ha="left", y=1.045)
    fig.text(0.02, -0.008,
             "Shaded band: label-permutation null (below its 95th percentile is indistinguishable from chance). "
             "Green vs red on the right panel isolates the classifier;\ngreen vs blue isolates the representation. "
             "The backbone encodes 131072 points as 128 patches, so it cannot represent detail finer than the dotted line.",
             fontsize=7.5, color="#898781")
    fig.tight_layout(rect=(0, 0, 1, 1.0))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
