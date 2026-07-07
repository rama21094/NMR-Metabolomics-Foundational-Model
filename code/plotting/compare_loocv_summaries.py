#!/usr/bin/env python3
"""
Compare multiple `summary.csv` files across result folders and generate plots.

Usage example:
  python scripts/compare_loocv_summaries.py \
    --dirs mtbls326_loocv_results_0.30 mtbls326_loocv_results_0.20 mtbls326_loocv_results_0.40 mtbls326_loocv_results_0.50 \
    --outdir comparison_plots

This script reads `summary.csv` from each folder, aggregates them, and creates:
 - grouped bar plot of `accuracy` per model across folders
 - line plots for several metrics (accuracy, f1, precision, recall, roc_auc, pr_auc)
 - heatmap of accuracy (models x folders)
 - combined CSV `combined_summary.csv` in the output directory

The script is intentionally dependency-light (pandas + matplotlib). If seaborn is installed,
heatmap will use it for nicer styling; otherwise matplotlib's imshow is used.
"""

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns  # optional
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False


METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
]


def read_summary_csv(folder: Path):
    p = folder / "summary.csv"
    if not p.exists():
        warnings.warn(f"Missing summary.csv in {folder}")
        return None
    try:
        df = pd.read_csv(p)
    except Exception as e:
        warnings.warn(f"Failed to read {p}: {e}")
        return None
    df = df.copy()
    df["source_folder"] = str(folder)
    # try to extract numeric tag (e.g., 0.20) from folder name
    try:
        s = str(folder.name)
        num = float(s.split("_")[-1])
    except Exception:
        num = np.nan
    df["tag_val"] = num
    return df


def ensure_columns(df):
    for m in METRICS:
        if m not in df.columns:
            df[m] = np.nan
    return df


def plot_grouped_bar(df, metric, outdir: Path):
    """Grouped bar: models on x-axis, one bar per folder"""
    pivot = df.pivot_table(index=["family", "model"], columns="source_folder", values=metric)
    pivot = pivot.sort_index()
    labels = [f"{fam}\n{mod}" for fam, mod in pivot.index.tolist()]
    x = np.arange(len(labels))
    n_groups = pivot.shape[1]
    width = 0.8 / max(1, n_groups)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), 6))
    for i, col in enumerate(pivot.columns):
        vals = pivot[col].to_numpy(dtype=float)
        ax.bar(x + i * width, vals, width=width, label=Path(col).name)

    ax.set_xticks(x + width * (n_groups - 1) / 2)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by model across folders")
    ax.legend(title="folder")
    fig.tight_layout()
    out = outdir / f"grouped_bar_{metric}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_metrics_lines(df, metrics, outdir: Path):
    # pivot so rows=model, cols=tag_val
    results = []
    for metric in metrics:
        pivot = df.pivot_table(index=["family", "model"], columns="tag_val", values=metric)
        pivot = pivot.sort_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        for idx, row in pivot.iterrows():
            ax.plot(pivot.columns, row.values, marker="o", label=f"{idx[0]}|{idx[1]}")
        ax.set_xlabel("folder numeric tag")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} across folders (per-model lines)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        out = outdir / f"lines_{metric}.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        results.append(out)
    return results


def plot_heatmap(df, metric, outdir: Path):
    pivot = df.pivot_table(index=["family", "model"], columns="source_folder", values=metric)
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(max(6, pivot.shape[0] * 0.25), max(4, pivot.shape[1] * 0.6)))
    if HAS_SEABORN:
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)
    else:
        im = ax.imshow(pivot.fillna(np.nan).to_numpy(dtype=float), aspect="auto", cmap="viridis")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"{f}\n{m}" for f, m in pivot.index.tolist()])
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([Path(c).name for c in pivot.columns], rotation=45, ha="right")
        fig.colorbar(im, ax=ax)
    ax.set_title(f"Heatmap {metric} (models x folders)")
    fig.tight_layout()
    out = outdir / f"heatmap_{metric}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Compare multiple LOOCV summary.csv files and plot results")
    p.add_argument("--dirs", nargs='+', required=True, help="Result folders containing summary.csv")
    p.add_argument("--outdir", type=str, default="comparison_plots", help="Output directory for plots")
    args = p.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for d in args.dirs:
        df = read_summary_csv(Path(d))
        if df is None:
            continue
        df = ensure_columns(df)
        all_dfs.append(df)

    if not all_dfs:
        print("No valid summary.csv files found in provided dirs.")
        return 2

    big = pd.concat(all_dfs, ignore_index=True)
    # create simple combined CSV
    combined_csv = outdir / "combined_summary.csv"
    big.to_csv(combined_csv, index=False)
    print(f"Wrote combined CSV: {combined_csv}")

    # Make grouped bar for accuracy
    out_files = []
    out_files.append(plot_grouped_bar(big, "accuracy", outdir))
    # Make heatmap for accuracy
    out_files.append(plot_heatmap(big, "accuracy", outdir))
    # Per-metric line plots
    out_files.extend(plot_metrics_lines(big, METRICS, outdir))

    print("Generated plots:")
    for fpath in out_files:
        print(f"  - {fpath}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
