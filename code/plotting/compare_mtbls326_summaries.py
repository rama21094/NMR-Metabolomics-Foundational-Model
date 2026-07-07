#!/usr/bin/env python3
"""Compare the four MTBLS326 LOOCV ``summary.csv`` result files.

With no arguments, this script reads:

    mtbls326_loocv_results_0.20/summary.csv
    mtbls326_loocv_results_0.30/summary.csv
    mtbls326_loocv_results_0.40/summary.csv
    mtbls326_loocv_results_0.50/summary.csv

Classical-model results are treated as threshold-independent baselines. The
script verifies that repeated classical rows agree, then propagates each
classical result to thresholds where it was not rerun.

It writes several PNG plots plus a combined CSV to
``mtbls326_loocv_comparison_plots``.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

# Avoid pandas/numexpr startup errors on hosts whose OMP thread count exceeds
# numexpr's safety limit. Users can still override this explicitly.
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DIRS = [
    "mtbls326_loocv_results_0.20",
    "mtbls326_loocv_results_0.30",
    "mtbls326_loocv_results_0.40",
    "mtbls326_loocv_results_0.50",
]

SCORE_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
]
CORE_METRICS = ["balanced_accuracy", "f1", "roc_auc", "pr_auc"]
CONFUSION_COLUMNS = ["tn", "fp", "fn", "tp"]
COLORS = plt.get_cmap("tab10").colors


def pretty(text: str) -> str:
    return text.replace("_", " ").title()


def threshold_from_folder(folder: Path) -> float:
    """Extract the trailing decimal number from a result folder name."""
    match = re.search(r"(\d+(?:\.\d+)?)$", folder.name)
    if not match:
        raise ValueError(
            f"Could not determine a threshold from folder name {folder.name!r}. "
            "Folder names must end in a number, such as '_0.30'."
        )
    return float(match.group(1))


def load_summaries(folders: list[Path]) -> pd.DataFrame:
    required = {"family", "model", *SCORE_METRICS, *CONFUSION_COLUMNS}
    frames = []
    for folder in folders:
        csv_path = folder / "summary.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing input file: {csv_path}")
        frame = pd.read_csv(csv_path)
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {', '.join(missing)}")
        frame = frame.copy()
        frame["threshold"] = threshold_from_folder(folder)
        frame["source_folder"] = folder.name
        frame["result_origin"] = "observed"
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    duplicates = data.duplicated(["family", "model", "threshold"], keep=False)
    if duplicates.any():
        rows = data.loc[duplicates, ["family", "model", "threshold"]]
        raise ValueError(f"Duplicate model/threshold rows found:\n{rows.to_string(index=False)}")

    # Classical models do not depend on the foundation-model threshold. Verify
    # repeated runs agree, then use them as fixed baselines at every threshold.
    thresholds = sorted(data["threshold"].unique())
    value_columns = [*SCORE_METRICS, *CONFUSION_COLUMNS]
    propagated_rows = []
    classical = data[data["family"].eq("classical")]
    for model, rows in classical.groupby("model", sort=True):
        reference = rows.iloc[0]
        for column in value_columns:
            values = rows[column].to_numpy(dtype=float)
            if not np.allclose(values, values[0], rtol=1e-9, atol=1e-12, equal_nan=True):
                raise ValueError(
                    f"Classical model {model!r} has inconsistent {column!r} values "
                    "across folders, so it cannot be used as a fixed baseline."
                )
        existing = set(rows["threshold"])
        for threshold in thresholds:
            if threshold in existing:
                continue
            copied = reference.copy()
            copied["threshold"] = threshold
            copied["source_folder"] = "constant_classical_baseline"
            copied["result_origin"] = "propagated classical baseline"
            propagated_rows.append(copied)

    if propagated_rows:
        data = pd.concat([data, pd.DataFrame(propagated_rows)], ignore_index=True)
    data["model_label"] = data["family"] + " | " + data["model"]
    return data.sort_values(["family", "model", "threshold"]).reset_index(drop=True)


def model_order(data: pd.DataFrame) -> list[str]:
    return data[["family", "model", "model_label"]].drop_duplicates().sort_values(
        ["family", "model"]
    )["model_label"].tolist()


def plot_metric_trends(data: pd.DataFrame, output: Path) -> None:
    """One line panel per score; gaps honestly represent missing model results."""
    models = model_order(data)
    thresholds = sorted(data["threshold"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(18, 10), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for ax, metric in zip(axes_flat, SCORE_METRICS):
        pivot = data.pivot(index="threshold", columns="model_label", values=metric)
        pivot = pivot.reindex(index=thresholds, columns=models)
        for i, model in enumerate(models):
            family = data.loc[data["model_label"].eq(model), "family"].iloc[0]
            ax.plot(
                thresholds,
                pivot[model],
                marker="o",
                linewidth=2,
                linestyle="--" if family == "classical" else "-",
                color=COLORS[i % len(COLORS)],
                label=model,
            )
        ax.set_title(pretty(metric))
        ax.set_ylim(0.4, 1.02)
        ax.set_xticks(thresholds)
        ax.grid(alpha=0.25)

    axes_flat[-1].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    axes_flat[-1].legend(handles, labels, loc="center", fontsize=10, frameon=False)
    fig.supxlabel("Result-folder threshold")
    fig.supylabel("Score")
    fig.suptitle("LOOCV metric trends across thresholds", fontsize=16)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_metric_heatmaps(data: pd.DataFrame, output: Path) -> None:
    """Annotated heatmaps make exact differences easy to scan."""
    models = model_order(data)
    thresholds = sorted(data["threshold"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(17, 12), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    image = None

    for ax, metric in zip(axes_flat, SCORE_METRICS):
        pivot = data.pivot(index="model_label", columns="threshold", values=metric)
        values = pivot.reindex(index=models, columns=thresholds).to_numpy(float)
        masked = np.ma.masked_invalid(values)
        image = ax.imshow(masked, cmap="viridis", vmin=0.4, vmax=1.0, aspect="auto")
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                label = "—" if np.isnan(value) else f"{value:.3f}"
                color = "0.45" if np.isnan(value) else ("white" if value < 0.73 else "black")
                ax.text(col, row, label, ha="center", va="center", fontsize=8, color=color)
        ax.set_title(pretty(metric))
        ax.set_xticks(range(len(thresholds)), [f"{v:.2f}" for v in thresholds])
        ax.set_yticks(range(len(models)), models, fontsize=9)

    axes_flat[-1].axis("off")
    if image is not None:
        color_axis = axes_flat[-1].inset_axes([0.30, 0.18, 0.10, 0.64])
        fig.colorbar(image, cax=color_axis, label="Score")
    fig.suptitle("Metric heatmaps across all models and thresholds", fontsize=16)
    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.07, top=0.92, wspace=0.15, hspace=0.22)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_core_metrics(data: pd.DataFrame, output: Path) -> None:
    """Grouped bars for four commonly used model-selection metrics."""
    models = model_order(data)
    thresholds = sorted(data["threshold"].unique())
    x = np.arange(len(models))
    width = 0.8 / len(thresholds)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True, sharey=True)

    for ax, metric in zip(axes.ravel(), CORE_METRICS):
        pivot = data.pivot(index="model_label", columns="threshold", values=metric)
        pivot = pivot.reindex(index=models, columns=thresholds)
        for i, threshold in enumerate(thresholds):
            offset = (i - (len(thresholds) - 1) / 2) * width
            ax.bar(
                x + offset,
                pivot[threshold],
                width=width,
                label=f"{threshold:.2f}",
                color=COLORS[i],
            )
        ax.set_title(pretty(metric))
        ax.set_ylim(0.4, 1.02)
        ax.grid(axis="y", alpha=0.25)

    for ax in axes[-1]:
        ax.set_xticks(x, models, rotation=35, ha="right", fontsize=9)
    axes[0, 0].legend(title="Threshold", ncols=2)
    fig.supylabel("Score")
    fig.suptitle("Core model metrics by threshold", fontsize=16)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_counts(data: pd.DataFrame, output: Path) -> None:
    """Stacked confusion-count bars, one panel per model."""
    models = model_order(data)
    thresholds = sorted(data["threshold"].unique())
    ncols = 2
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.2 * nrows), sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()
    count_colors = {"tn": "#4C78A8", "fp": "#F58518", "fn": "#E45756", "tp": "#54A24B"}

    for ax, model in zip(axes_flat, models):
        subset = data[data["model_label"] == model].set_index("threshold")
        subset = subset.reindex(thresholds)
        bottom = np.zeros(len(thresholds))
        for column in CONFUSION_COLUMNS:
            values = subset[column].fillna(0).to_numpy(float)
            ax.bar(
                range(len(thresholds)),
                values,
                bottom=bottom,
                color=count_colors[column],
                label=column.upper(),
            )
            bottom += values
        present = subset["model"].notna().to_numpy()
        for i, is_present in enumerate(present):
            if not is_present:
                ax.text(i, 1, "not run", rotation=90, ha="center", va="bottom", color="0.45")
        ax.set_title(model)
        ax.set_xticks(range(len(thresholds)), [f"{v:.2f}" for v in thresholds])
        ax.set_xlabel("Threshold")
        ax.grid(axis="y", alpha=0.2)

    for ax in axes_flat[len(models):]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=4, bbox_to_anchor=(0.5, 0.97))
    fig.supylabel("Number of samples")
    fig.suptitle("Confusion-matrix composition", fontsize=16)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.94))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_best_models(data: pd.DataFrame, output: Path) -> None:
    """Show the best available model at each threshold for each core metric."""
    thresholds = sorted(data["threshold"].unique())
    x = np.arange(len(thresholds))
    width = 0.8 / len(CORE_METRICS)
    fig, ax = plt.subplots(figsize=(13, 7))

    for i, metric in enumerate(CORE_METRICS):
        winners = data.loc[data.groupby("threshold")[metric].idxmax()].set_index("threshold")
        winners = winners.reindex(thresholds)
        offset = (i - (len(CORE_METRICS) - 1) / 2) * width
        bars = ax.bar(x + offset, winners[metric], width, label=pretty(metric), color=COLORS[i])
        for bar, model in zip(bars, winners["model"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                str(model),
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x, [f"{value:.2f}" for value in thresholds])
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Best score")
    ax.set_ylim(0.4, 1.14)
    ax.set_title("Best available model for each core metric")
    ax.legend(ncols=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_average_model_performance(data: pd.DataFrame, output: Path) -> None:
    """Rank all models by mean core-metric performance across thresholds."""
    means = data.groupby(["family", "model"], as_index=False)[CORE_METRICS].mean()
    means["model_label"] = means["family"] + " | " + means["model"]
    means["overall_mean"] = means[CORE_METRICS].mean(axis=1)
    means = means.sort_values("overall_mean", ascending=False).reset_index(drop=True)

    x = np.arange(len(means))
    width = 0.8 / len(CORE_METRICS)
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, metric in enumerate(CORE_METRICS):
        offset = (i - (len(CORE_METRICS) - 1) / 2) * width
        ax.bar(x + offset, means[metric], width, label=pretty(metric), color=COLORS[i])

    ax.set_xticks(x, means["model_label"], rotation=30, ha="right")
    ax.set_ylabel("Mean score across thresholds")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Overall model comparison (ranked by mean core-metric score)")
    ax.legend(ncols=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=DEFAULT_DIRS,
        metavar="DIR",
        help="Folders containing summary.csv (defaults to the four MTBLS326 folders)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("mtbls326_loocv_comparison_plots"),
        help="Output directory (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    data = load_summaries([Path(folder) for folder in args.dirs])
    data.drop(columns="model_label").to_csv(args.outdir / "combined_summary.csv", index=False)

    plots = {
        "metric_trends.png": plot_metric_trends,
        "metric_heatmaps.png": plot_metric_heatmaps,
        "grouped_core_metrics.png": plot_grouped_core_metrics,
        "confusion_matrix_counts.png": plot_confusion_counts,
        "best_models_by_metric.png": plot_best_models,
        "average_model_performance.png": plot_average_model_performance,
    }
    for filename, plotting_function in plots.items():
        plotting_function(data, args.outdir / filename)

    print(f"Compared {len(data)} rows from {len(args.dirs)} summary files.")
    print(f"Plots and combined CSV written to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
