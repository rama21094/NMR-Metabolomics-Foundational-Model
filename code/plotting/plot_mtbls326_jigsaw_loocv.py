#!/usr/bin/env python3
"""Plot MTBLS326 jigsaw LOOCV comparison outputs."""

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
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve


METRICS = ["balanced_accuracy", "f1", "roc_auc", "pr_auc"]


def display_name(row_or_name):
    if isinstance(row_or_name, str):
        name = row_or_name
    else:
        name = f"{row_or_name['family']}_{row_or_name['model']}"
    return (
        name.replace("classical_", "")
        .replace("jigsaw_jigsaw_", "jigsaw_")
        .replace("_frozen", "")
        .replace("_", " ")
        .title()
    )


def score_file(output_dir: Path, family: str, model: str) -> Path:
    return output_dir / f"{family}_{model}_oof_score.npy"


def pred_file(output_dir: Path, family: str, model: str) -> Path:
    return output_dir / f"{family}_{model}_oof_pred.npy"


def load_outputs(output_dir: Path):
    summary_path = output_dir / "summary.csv"
    predictions_path = output_dir / "oof_predictions.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions: {predictions_path}")
    summary = pd.read_csv(summary_path)
    predictions = pd.read_csv(predictions_path)
    return summary, predictions


def plot_metric_bars(summary: pd.DataFrame, out_dir: Path):
    plot_df = summary.copy()
    plot_df["label"] = plot_df.apply(display_name, axis=1)
    x = np.arange(len(plot_df))
    width = 0.18

    fig, ax = plt.subplots(figsize=(max(12, len(plot_df) * 1.35), 6))
    for i, metric in enumerate(METRICS):
        ax.bar(x + (i - 1.5) * width, plot_df[metric], width=width, label=metric.replace("_", " ").title())
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("LOOCV Metric Comparison")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = out_dir / "metric_comparison_bars.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_jigsaw_resolution(summary: pd.DataFrame, out_dir: Path):
    jigsaw = summary[summary["family"] == "jigsaw"].copy()
    if jigsaw.empty:
        return None

    def order_value(model):
        if "multibin" in model:
            return 9999
        for size in (256, 512, 1024, 2048):
            if f"bin_{size}" in model:
                return size
        return 0

    jigsaw["order"] = jigsaw["model"].map(order_value)
    jigsaw = jigsaw.sort_values("order")
    labels = ["Multibin" if "multibin" in m else m.split("_")[2] for m in jigsaw["model"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    for metric in ["balanced_accuracy", "f1", "roc_auc"]:
        ax.plot(labels, jigsaw[metric], marker="o", linewidth=2, label=metric.replace("_", " ").title())
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Jigsaw Bin Setting")
    ax.set_ylabel("Score")
    ax.set_title("Jigsaw Resolution Trend")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "jigsaw_resolution_trend.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_roc_pr(summary: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path, out_dir: Path):
    y_true = predictions["target"].to_numpy()
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(7, 6))

    for _, row in summary.iterrows():
        path = score_file(output_dir, row["family"], row["model"])
        if not path.exists():
            continue
        scores = np.load(path)
        fpr, tpr, _ = roc_curve(y_true, scores)
        prec, rec, _ = precision_recall_curve(y_true, scores)
        label = f"{display_name(row)} (AUC {row['roc_auc']:.3f})"
        ax_roc.plot(fpr, tpr, linewidth=1.8, label=label)
        ax_pr.plot(rec, prec, linewidth=1.8, label=f"{display_name(row)} (AP {row['pr_auc']:.3f})")

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.45)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves")
    ax_roc.legend(fontsize=8)
    ax_roc.grid(alpha=0.25)

    positive_rate = float(y_true.mean())
    ax_pr.axhline(positive_rate, color="k", linestyle="--", alpha=0.45)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curves")
    ax_pr.legend(fontsize=8)
    ax_pr.grid(alpha=0.25)

    roc_path = out_dir / "roc_curves.png"
    pr_path = out_dir / "precision_recall_curves.png"
    fig_roc.tight_layout()
    fig_pr.tight_layout()
    fig_roc.savefig(roc_path, dpi=220)
    fig_pr.savefig(pr_path, dpi=220)
    plt.close(fig_roc)
    plt.close(fig_pr)
    return roc_path, pr_path


def plot_confusion_matrices(summary: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path, out_dir: Path):
    y_true = predictions["target"].to_numpy()
    n = len(summary)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 3.7 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, row) in zip(axes, summary.iterrows()):
        path = pred_file(output_dir, row["family"], row["model"])
        preds = np.load(path)
        ConfusionMatrixDisplay.from_predictions(
            y_true,
            preds,
            display_labels=["No", "Yes"],
            cmap="Blues",
            colorbar=False,
            ax=ax,
        )
        ax.set_title(display_name(row), fontsize=10)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("LOOCV Confusion Matrices", y=1.02)
    fig.tight_layout()
    path = out_dir / "confusion_matrices.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_probability_heatmap(summary: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path, out_dir: Path):
    y_true = predictions["target"].to_numpy()
    labels = []
    matrix = []
    for _, row in summary.iterrows():
        path = score_file(output_dir, row["family"], row["model"])
        if path.exists():
            labels.append(display_name(row))
            matrix.append(np.load(path))
    if not matrix:
        return None
    matrix = np.vstack(matrix)
    order = np.argsort(y_true)
    sample_labels = [
        f"{i}:{'Yes' if predictions.iloc[i]['target'] else 'No'}"
        for i in order
    ]

    fig, ax = plt.subplots(figsize=(max(12, len(order) * 0.28), max(5, len(labels) * 0.55)))
    im = ax.imshow(matrix[:, order], aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(sample_labels, rotation=90, fontsize=7)
    ax.set_title("OOF Yes Probability by Sample")
    ax.set_xlabel("Sample sorted by true class")
    fig.colorbar(im, ax=ax, label="Yes probability")
    fig.tight_layout()
    path = out_dir / "yes_probability_heatmap.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_rankings(summary: pd.DataFrame, out_dir: Path):
    ranked = summary.sort_values(["balanced_accuracy", "f1", "roc_auc"], ascending=False).copy()
    ranked["display_name"] = ranked.apply(display_name, axis=1)
    path = out_dir / "ranked_models.csv"
    ranked.to_csv(path, index=False)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/loocv/mtbls326_jigsaw")
    parser.add_argument("--plot-dir", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    plot_dir = Path(args.plot_dir) if args.plot_dir else output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    summary, predictions = load_outputs(output_dir)
    paths = [
        plot_metric_bars(summary, plot_dir),
        plot_jigsaw_resolution(summary, plot_dir),
        *plot_roc_pr(summary, predictions, output_dir, plot_dir),
        plot_confusion_matrices(summary, predictions, output_dir, plot_dir),
        plot_probability_heatmap(summary, predictions, output_dir, plot_dir),
        save_rankings(summary, plot_dir),
    ]
    print("Saved plots/tables:")
    for path in paths:
        if path is not None:
            print(f"  {path}")


if __name__ == "__main__":
    main()
