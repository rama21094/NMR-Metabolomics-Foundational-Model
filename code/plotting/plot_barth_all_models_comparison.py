#!/usr/bin/env python3
"""Create comparison visuals for the Barth all-models LOOCV summary."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "64")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except Exception:  # pragma: no cover - optional styling dependency
    sns = None


DEFAULT_SUMMARY = "results/loocv/barth_all_models/summary.csv"
DEFAULT_OUT_DIR = "results/loocv/barth_all_models/plots"

FAMILY_COLORS = {
    "classical": "#4C78A8",
    "masking": "#59A14F",
    "jigsaw": "#F28E2B",
    "joint_ssl": "#E15759",
}
FAMILY_ORDER = ["classical", "masking", "jigsaw", "joint_ssl"]
CORE_METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
AUC_METRICS = ["roc_auc", "pr_auc"]
HEATMAP_METRICS = CORE_METRICS + AUC_METRICS + ["precision", "recall"]


def metric_label(metric: str) -> str:
    return metric.replace("_", " ").replace("auc", "AUC").title().replace("Auc", "AUC")


def clean_model_name(row: pd.Series) -> str:
    model = str(row["model"])
    if row["family"] == "jigsaw":
        return model.removeprefix("jigsaw_")
    return model


def tune_mode(model: str) -> str:
    if model.endswith("_frozen") or model == "frozen":
        return "frozen"
    match = re.search(r"unfreeze_last_(\d+)$", model)
    if match:
        return f"unfreeze_last_{match.group(1)}"
    return "n/a"


def tune_order(mode: str) -> int:
    if mode == "frozen":
        return 0
    match = re.search(r"(\d+)$", mode)
    return int(match.group(1)) if match else 99


def jigsaw_backbone(model: str) -> str:
    model = model.removeprefix("jigsaw_")
    for suffix in ("_frozen", "_unfreeze_last_1", "_unfreeze_last_2", "_unfreeze_last_3"):
        if model.endswith(suffix):
            return model[: -len(suffix)]
    return model


def masking_group(model: str) -> str:
    match = re.match(r"mask_(\d+)_", model)
    return f"mask_{match.group(1)}" if match else model


def load_data(summary_path: Path) -> pd.DataFrame:
    data = pd.read_csv(summary_path)
    for metric in HEATMAP_METRICS:
        if metric not in data.columns:
            data[metric] = np.nan
        data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data["model_display"] = data.apply(clean_model_name, axis=1)
    data["row_label"] = data["family"].str.title() + ": " + data["model_display"]
    data["tune_mode"] = data["model"].map(tune_mode)
    data["family_order"] = data["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)}).fillna(99)
    return data.sort_values(["family_order", "model_display"]).reset_index(drop=True)


def available_metrics(data: pd.DataFrame, metrics: list[str]) -> list[str]:
    return [m for m in metrics if m in data.columns and data[m].notna().any()]


def plot_ranked_bar(data: pd.DataFrame, metric: str, out_dir: Path) -> Path | None:
    subset = data[data[metric].notna()].sort_values(metric, ascending=True)
    if subset.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, max(6, 0.32 * len(subset) + 1.5)))
    colors = [FAMILY_COLORS.get(f, "#999999") for f in subset["family"]]
    bars = ax.barh(subset["row_label"], subset[metric], color=colors)
    for bar, value in zip(bars, subset[metric]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=7)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel(metric_label(metric))
    ax.set_title(f"Barth LOOCV: All Models Ranked by {metric_label(metric)}")
    ax.grid(axis="x", alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[f]) for f in FAMILY_ORDER if f in set(subset["family"])]
    labels = [f.replace("_", " ").title() for f in FAMILY_ORDER if f in set(subset["family"])]
    ax.legend(handles, labels, loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    path = out_dir / f"ranked_{metric}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_best_per_family(data: pd.DataFrame, metric: str, out_dir: Path) -> Path | None:
    subset = data[data[metric].notna()].copy()
    if subset.empty:
        return None
    best = subset.sort_values(metric, ascending=False).groupby("family", as_index=False).first()
    best["family_order"] = best["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)}).fillna(99)
    best = best.sort_values("family_order")

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [FAMILY_COLORS.get(f, "#999999") for f in best["family"]]
    bars = ax.bar(best["family"].str.replace("_", " ").str.title(), best[metric], color=colors, width=0.55)
    for bar, name, value in zip(bars, best["model_display"], best[metric]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{value:.3f}\n{name}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0, min(1.2, max(1.0, float(best[metric].max()) + 0.22)))
    ax.set_ylabel(metric_label(metric))
    ax.set_title(f"Best Model Per Family: {metric_label(metric)}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = out_dir / f"best_per_family_{metric}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_core_metrics_grouped(data: pd.DataFrame, out_dir: Path, top_n: int = 12) -> Path | None:
    metrics = available_metrics(data, CORE_METRICS)
    if not metrics:
        return None
    subset = data[data["balanced_accuracy"].notna()].sort_values("balanced_accuracy", ascending=False).head(top_n)
    if subset.empty:
        return None
    subset = subset.iloc[::-1]  # best on top for barh-style top-down ordering when plotted with grouped bars

    fig, ax = plt.subplots(figsize=(11, max(6, 0.5 * len(subset) + 1.5)))
    y = np.arange(len(subset))
    n_metrics = len(metrics)
    height = 0.8 / n_metrics
    palette = plt.cm.viridis(np.linspace(0.15, 0.9, n_metrics))
    for i, metric in enumerate(metrics):
        offset = (i - (n_metrics - 1) / 2) * height
        ax.barh(y + offset, subset[metric], height=height, label=metric_label(metric), color=palette[i])
    ax.set_yticks(y, subset["row_label"])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score")
    ax.set_title(f"Top {len(subset)} Models by Balanced Accuracy: Core Metrics")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    path = out_dir / "top_models_core_metrics_grouped.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_heatmap(data: pd.DataFrame, out_dir: Path) -> Path | None:
    metrics = available_metrics(data, HEATMAP_METRICS)
    if not metrics:
        return None
    subset = data.set_index("row_label")[metrics].copy()
    subset.columns = [metric_label(m) for m in metrics]
    subset = subset.reindex(data.sort_values(["family_order", "balanced_accuracy"], ascending=[True, False])["row_label"])

    fig_h = max(8.5, 0.34 * len(subset) + 2.0)
    fig_w = max(8.5, 1.3 * len(subset.columns) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    if sns is not None:
        sns.heatmap(subset, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0.0, vmax=1.0, linewidths=0.4, linecolor="#e6e6e6", ax=ax)
    else:
        values = subset.to_numpy(dtype=float)
        image = ax.imshow(values, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(np.arange(len(subset.columns)), subset.columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(subset.index)), subset.index)
        for yy in range(values.shape[0]):
            for xx in range(values.shape[1]):
                if np.isfinite(values[yy, xx]):
                    ax.text(xx, yy, f"{values[yy, xx]:.3f}", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Barth LOOCV: All Models, All Metrics")
    fig.tight_layout()
    path = out_dir / "all_models_all_metrics_heatmap.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_roc_pr_scatter(data: pd.DataFrame, out_dir: Path) -> Path | None:
    subset = data[data["roc_auc"].notna() & data["pr_auc"].notna()]
    if subset.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 7))
    for family, group in subset.groupby("family"):
        ax.scatter(group["roc_auc"], group["pr_auc"], color=FAMILY_COLORS.get(family, "#999999"), label=family.replace("_", " ").title(), s=60, edgecolor="white", linewidth=0.5, zorder=3)
    for _, row in subset.iterrows():
        ax.annotate(row["model_display"], (row["roc_auc"], row["pr_auc"]), fontsize=6, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel(metric_label("roc_auc"))
    ax.set_ylabel(metric_label("pr_auc"))
    ax.set_title("Barth LOOCV: ROC-AUC vs PR-AUC")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = out_dir / "roc_auc_vs_pr_auc_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_finetune_trend(data: pd.DataFrame, family: str, metric: str, out_dir: Path) -> Path | None:
    subset = data[(data["family"] == family) & data[metric].notna()].copy()
    if subset.empty:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 5))
    if family == "masking":
        subset["group"] = subset["model"].map(masking_group)
        for group, gdf in subset.groupby("group"):
            gdf = gdf.sort_values("tune_mode", key=lambda s: s.map(tune_order))
            ax.plot([tune_order(m) for m in gdf["tune_mode"]], gdf[metric], marker="o", linewidth=1.8, label=group)
        ax.set_xticks([0, 1, 2, 3], ["frozen", "u1", "u2", "u3"])
    else:  # jigsaw
        subset["backbone"] = subset["model"].map(jigsaw_backbone)
        for backbone, gdf in subset.groupby("backbone"):
            gdf = gdf.sort_values("tune_mode", key=lambda s: s.map(tune_order))
            ax.plot([tune_order(m) for m in gdf["tune_mode"]], gdf[metric], marker="o", linewidth=1.8, label=backbone)
        ax.set_xticks([0, 1, 2, 3], ["frozen", "u1", "u2", "u3"])
    ax.set_xlabel("Fine-tune mode")
    ax.set_ylabel(metric_label(metric))
    ax.set_title(f"{family.title()} Fine-Tuning Trend: {metric_label(metric)}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    path = out_dir / f"{family}_finetune_trend_{metric}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(Path(args.summary))

    plot_paths: list[Path] = []
    for metric in ("balanced_accuracy", "macro_f1", "accuracy", "roc_auc"):
        path = plot_ranked_bar(data, metric, out_dir)
        if path is not None:
            plot_paths.append(path)

    path = plot_core_metrics_grouped(data, out_dir)
    if path is not None:
        plot_paths.append(path)

    for metric in ("balanced_accuracy", "macro_f1"):
        path = plot_best_per_family(data, metric, out_dir)
        if path is not None:
            plot_paths.append(path)

    path = plot_heatmap(data, out_dir)
    if path is not None:
        plot_paths.append(path)

    path = plot_roc_pr_scatter(data, out_dir)
    if path is not None:
        plot_paths.append(path)

    for family in ("masking", "jigsaw"):
        for metric in ("balanced_accuracy", "macro_f1"):
            path = plot_finetune_trend(data, family, metric, out_dir)
            if path is not None:
                plot_paths.append(path)

    print(f"Wrote {len(plot_paths)} plots to {out_dir}:")
    for path in plot_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
