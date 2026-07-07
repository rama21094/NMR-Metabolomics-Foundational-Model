#!/usr/bin/env python3
"""Create comparison visuals for BrC/T2D masking and jigsaw LOOCV summaries."""

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


DEFAULT_MASKING_DIR = "results/loocv/brc_t2d_masking"
DEFAULT_JIGSAW_DIR = "results/loocv/brc_t2d_jigsaw"
DEFAULT_OUT_DIR = "results/loocv/brc_t2d_comparison_plots"
LABEL_ORDER = ["cancer_status", "diabetes_status", "combined_status"]
MAIN_METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
EXTRA_METRICS = ["roc_auc", "pr_auc", "precision", "recall", "f1", "macro_roc_auc_ovr"]
PLOT_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "roc_auc",
    "pr_auc",
    "macro_roc_auc_ovr",
]
PAIRWISE_COMPARISONS = [
    ("jigsaw", "classical"),
    ("masking", "classical"),
    ("jigsaw", "masking"),
]
PAIRWISE_METRICS = ["balanced_accuracy", "weighted_f1"]
CLASSICAL_ORDER = {"logistic_regression": 0, "svm_rbf": 1, "xgboost": 2}


def metric_label(metric: str) -> str:
    return metric.replace("_", " ").replace("ovr", "OvR").title()


def display_label(label: str) -> str:
    return {
        "cancer_status": "Cancer",
        "diabetes_status": "Diabetes",
        "combined_status": "Combined",
    }.get(label, label.replace("_", " ").title())


def approach_from_row(source: str, family: str) -> str:
    if family == "classical":
        return "classical"
    if source == "masking" or family == "foundation":
        return "masking"
    if source == "jigsaw" or family == "jigsaw":
        return "jigsaw"
    return source


def clean_model_name(row: pd.Series) -> str:
    model = str(row["model"])
    family = str(row["family"])
    if family == "foundation":
        return model
    if family == "jigsaw":
        return model.removeprefix("jigsaw_")
    return model


def tune_mode(model: str) -> str:
    if model.endswith("_frozen") or model == "frozen":
        return "frozen"
    match = re.search(r"unfreeze_last_(\d+)$", model)
    if match:
        return f"unfreeze_last_{match.group(1)}"
    return "n/a"


def jigsaw_backbone(model: str) -> str:
    model = model.removeprefix("jigsaw_")
    for suffix in ("_frozen", "_unfreeze_last_1", "_unfreeze_last_2", "_unfreeze_last_3"):
        if model.endswith(suffix):
            return model[: -len(suffix)]
    return model


def tune_order(mode: str) -> int:
    if mode == "frozen":
        return 0
    match = re.search(r"(\d+)$", mode)
    return int(match.group(1)) if match else 99


def read_summaries(masking_dir: Path, jigsaw_dir: Path) -> pd.DataFrame:
    frames = []
    for source, root in (("masking", masking_dir), ("jigsaw", jigsaw_dir)):
        for summary_path in sorted(root.glob("*/summary.csv")):
            label = summary_path.parent.name
            df = pd.read_csv(summary_path)
            df["source"] = source
            df["label_column"] = label
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No summary.csv files found under masking or jigsaw result directories")

    data = pd.concat(frames, ignore_index=True)
    data["approach"] = [approach_from_row(s, f) for s, f in zip(data["source"], data["family"])]
    data["model_display"] = data.apply(clean_model_name, axis=1)
    data["tune_mode"] = data["model"].map(tune_mode)
    data["jigsaw_backbone"] = data["model"].map(jigsaw_backbone)

    # Classical rows are duplicated in masking and jigsaw outputs. Keep one copy.
    is_classical = data["approach"].eq("classical")
    classical = data[is_classical].sort_values(["label_column", "model", "source"]).drop_duplicates(
        ["label_column", "family", "model"], keep="first"
    )
    non_classical = data[~is_classical]
    cleaned = pd.concat([classical, non_classical], ignore_index=True)

    for metric in MAIN_METRICS + EXTRA_METRICS:
        if metric not in cleaned.columns:
            cleaned[metric] = np.nan
        cleaned[metric] = pd.to_numeric(cleaned[metric], errors="coerce")
    cleaned["label_display"] = cleaned["label_column"].map(display_label)
    cleaned["label_order"] = cleaned["label_column"].map(
        {label: idx for idx, label in enumerate(LABEL_ORDER)}
    ).fillna(99)
    return cleaned.sort_values(["label_order", "approach", "model_display"]).reset_index(drop=True)


def available_metrics(data: pd.DataFrame, metrics: list[str]) -> list[str]:
    return [metric for metric in metrics if metric in data.columns and data[metric].notna().any()]


def save_heatmap(
    data: pd.DataFrame,
    metric: str,
    row_col: str,
    out_path: Path,
    title: str,
    height_per_row: float = 0.42,
) -> None:
    pivot = data.pivot_table(index=row_col, columns="label_display", values=metric, aggfunc="max")
    pivot = pivot.reindex(columns=[display_label(label) for label in LABEL_ORDER if display_label(label) in pivot.columns])
    pivot = pivot.sort_index()
    fig_h = max(3.2, height_per_row * max(1, len(pivot)) + 1.6)
    fig_w = max(6.4, 1.45 * max(1, len(pivot.columns)) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    if sns is not None:
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0.0, vmax=1.0, linewidths=0.5, ax=ax)
    else:
        values = pivot.to_numpy(dtype=float)
        image = ax.imshow(values, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns)
        ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
        for y in range(values.shape[0]):
            for x in range(values.shape[1]):
                if np.isfinite(values[y, x]):
                    ax.text(x, y, f"{values[y, x]:.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_metric_heatmaps(data: pd.DataFrame, approach: str, out_dir: Path) -> list[Path]:
    subset = data[data["approach"].eq(approach)].copy()
    if subset.empty:
        return []
    paths = []
    for metric in available_metrics(subset, PLOT_METRICS):
        path = out_dir / f"{approach}_{metric}_heatmap.png"
        save_heatmap(
            subset,
            metric,
            "model_display",
            path,
            f"{approach.title()} Models: {metric_label(metric)}",
        )
        paths.append(path)
    return paths


def plot_best_family_comparison(data: pd.DataFrame, metric: str, out_dir: Path) -> Path | None:
    subset = data[data[metric].notna()].copy()
    if subset.empty:
        return None
    best = (
        subset.sort_values(metric, ascending=False)
        .groupby(["label_column", "approach"], as_index=False)
        .first()
        .sort_values(["label_order", "approach"])
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    labels = [display_label(label) for label in LABEL_ORDER if label in set(best["label_column"])]
    x = np.arange(len(labels))
    approaches = ["classical", "masking", "jigsaw"]
    colors = {"classical": "#4C78A8", "masking": "#59A14F", "jigsaw": "#F28E2B"}
    width = 0.24
    for idx, approach in enumerate(approaches):
        values = []
        names = []
        for label in LABEL_ORDER:
            row = best[(best["label_column"].eq(label)) & (best["approach"].eq(approach))]
            values.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
            names.append(str(row["model_display"].iloc[0]) if not row.empty else "")
        bars = ax.bar(x + (idx - 1) * width, values[: len(labels)], width, label=approach.title(), color=colors[approach])
        for bar, name in zip(bars, names[: len(labels)]):
            if np.isfinite(bar.get_height()):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    name.replace("unfreeze_last_", "u"),
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=7,
                )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, min(1.12, max(1.0, np.nanmax(best[metric]) + 0.18)))
    ax.set_ylabel(metric_label(metric))
    ax.set_title(f"Best Model Per Family: {metric_label(metric)}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / f"best_family_{metric}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_finetune_gains(data: pd.DataFrame, approach: str, metric: str, out_dir: Path) -> Path | None:
    subset = data[(data["approach"].eq(approach)) & (data[metric].notna())].copy()
    if subset.empty:
        return None
    fig, axes = plt.subplots(1, len(LABEL_ORDER), figsize=(15, 4.7), sharey=True)
    if len(LABEL_ORDER) == 1:
        axes = [axes]
    wrote_any = False
    for ax, label in zip(axes, LABEL_ORDER):
        label_df = subset[subset["label_column"].eq(label)].copy()
        if label_df.empty:
            ax.axis("off")
            continue
        if approach == "masking":
            line_df = label_df.sort_values("tune_mode", key=lambda s: s.map(tune_order))
            ax.plot(
                [tune_order(mode) for mode in line_df["tune_mode"]],
                line_df[metric],
                marker="o",
                linewidth=2,
                color="#59A14F",
            )
            ax.set_xticks([tune_order(mode) for mode in line_df["tune_mode"]], line_df["tune_mode"], rotation=35, ha="right")
        else:
            for backbone, group in label_df.groupby("jigsaw_backbone"):
                group = group.sort_values("tune_mode", key=lambda s: s.map(tune_order))
                ax.plot(
                    [tune_order(mode) for mode in group["tune_mode"]],
                    group[metric],
                    marker="o",
                    linewidth=1.8,
                    label=backbone,
                )
            ax.set_xticks([0, 1, 2, 3], ["frozen", "u1", "u2", "u3"])
        ax.set_title(display_label(label))
        ax.set_xlabel("Fine-tune mode")
        ax.grid(axis="y", alpha=0.25)
        wrote_any = True
    axes[0].set_ylabel(metric_label(metric))
    if approach == "jigsaw":
        axes[-1].legend(fontsize=8, frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.suptitle(f"{approach.title()} Fine-Tuning Trend: {metric_label(metric)}")
    fig.tight_layout()
    path = out_dir / f"{approach}_finetune_trend_{metric}.png"
    if wrote_any:
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path
    plt.close(fig)
    return None


def plot_top_rankings(data: pd.DataFrame, metric: str, out_dir: Path, top_n: int) -> Path | None:
    subset = data[data[metric].notna()].copy()
    if subset.empty:
        return None
    rows = []
    for label, group in subset.groupby("label_column"):
        top = group.sort_values(metric, ascending=False).head(top_n).copy()
        top["rank_label"] = top["approach"].str.title() + ": " + top["model_display"]
        rows.append(top)
    ranked = pd.concat(rows, ignore_index=True)
    fig, axes = plt.subplots(1, len(LABEL_ORDER), figsize=(15, 5.2), sharex=True)
    if len(LABEL_ORDER) == 1:
        axes = [axes]
    colors = {"classical": "#4C78A8", "masking": "#59A14F", "jigsaw": "#F28E2B"}
    for ax, label in zip(axes, LABEL_ORDER):
        label_df = ranked[ranked["label_column"].eq(label)].sort_values(metric)
        if label_df.empty:
            ax.axis("off")
            continue
        ax.barh(label_df["rank_label"], label_df[metric], color=[colors[a] for a in label_df["approach"]])
        ax.set_title(display_label(label))
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle(f"Top {top_n} Models Per Task: {metric_label(metric)}")
    fig.tight_layout()
    path = out_dir / f"top_{top_n}_{metric}_rankings.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def best_rows_for_metric(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    subset = data[data[metric].notna()].copy()
    if subset.empty:
        return pd.DataFrame()
    return (
        subset.sort_values(metric, ascending=False)
        .groupby(["label_column", "approach"], as_index=False)
        .first()
        .sort_values(["label_order", "approach"])
    )


def plot_pairwise_best_comparison(
    data: pd.DataFrame,
    left: str,
    right: str,
    metric: str,
    out_dir: Path,
) -> Path | None:
    best = best_rows_for_metric(data, metric)
    if best.empty:
        return None
    labels = [label for label in LABEL_ORDER if label in set(best["label_column"])]
    if not labels:
        return None

    colors = {"classical": "#4C78A8", "masking": "#59A14F", "jigsaw": "#F28E2B"}
    fig, axes = plt.subplots(1, len(labels), figsize=(5.2 * len(labels), 4.8), sharey=True)
    if len(labels) == 1:
        axes = [axes]

    wrote_any = False
    for ax, label in zip(axes, labels):
        rows = []
        for approach in (left, right):
            row = best[(best["label_column"].eq(label)) & (best["approach"].eq(approach))]
            if not row.empty:
                rows.append(row.iloc[0])
        if len(rows) < 2:
            ax.axis("off")
            continue

        values = [float(row[metric]) for row in rows]
        names = [str(row["model_display"]).replace("unfreeze_last_", "u") for row in rows]
        approaches = [str(row["approach"]) for row in rows]
        bars = ax.bar(approaches, values, color=[colors[a] for a in approaches], width=0.55)
        for bar, value, name in zip(bars, values, names):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.3f}\n{name}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        delta = values[0] - values[1]
        ax.text(
            0.5,
            0.04,
            f"Delta {left} - {right}: {delta:+.3f}",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#dddddd"},
        )
        ax.set_title(display_label(label))
        ax.set_ylim(0, min(1.16, max(1.0, max(values) + 0.2)))
        ax.grid(axis="y", alpha=0.25)
        wrote_any = True

    axes[0].set_ylabel(metric_label(metric))
    title = f"{left.title()} vs {right.title()}: Best Model {metric_label(metric)}"
    fig.suptitle(title)
    fig.tight_layout()
    path = out_dir / f"pairwise_best_{left}_vs_{right}_{metric}.png"
    if wrote_any:
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path
    plt.close(fig)
    return None


def plot_all_models_core_heatmap(data: pd.DataFrame, out_dir: Path) -> Path | None:
    metrics = ["balanced_accuracy", "weighted_f1"]
    subset = data[data[metrics].notna().any(axis=1)].copy()
    if subset.empty:
        return None

    subset["row_label"] = subset["approach"].str.title() + ": " + subset["model_display"]
    frames = []
    for metric in metrics:
        metric_frame = subset.pivot_table(
            index=["approach", "model_display", "row_label"],
            columns="label_display",
            values=metric,
            aggfunc="max",
        )
        metric_frame.columns = pd.MultiIndex.from_product(
            [[metric_label(metric)], metric_frame.columns]
        )
        frames.append(metric_frame)
    heatmap_data = pd.concat(frames, axis=1)

    ordered_columns = []
    for label in [display_label(label) for label in LABEL_ORDER]:
        for metric in [metric_label(metric) for metric in metrics]:
            column = (metric, label)
            if column in heatmap_data.columns:
                ordered_columns.append(column)
    heatmap_data = heatmap_data.reindex(columns=pd.MultiIndex.from_tuples(ordered_columns))

    row_order = {"classical": 0, "masking": 1, "jigsaw": 2}
    sort_frame = heatmap_data.copy()
    sort_frame["approach_order"] = [row_order.get(idx[0], 99) for idx in sort_frame.index]
    sort_frame["max_balanced_accuracy"] = heatmap_data[
        [column for column in heatmap_data.columns if column[0] == metric_label("balanced_accuracy")]
    ].max(axis=1)
    sort_frame = sort_frame.sort_values(
        ["approach_order", "max_balanced_accuracy"],
        ascending=[True, False],
    )
    heatmap_data = heatmap_data.loc[sort_frame.index]
    heatmap_data.index = [idx[2] for idx in heatmap_data.index]
    heatmap_data.columns = [f"{label}\n{metric}" for metric, label in heatmap_data.columns]

    fig_h = max(8.5, 0.34 * len(heatmap_data) + 2.2)
    fig_w = max(10.5, 1.55 * len(heatmap_data.columns) + 2.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    if sns is not None:
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.35,
            linecolor="#e6e6e6",
            ax=ax,
        )
    else:
        values = heatmap_data.to_numpy(dtype=float)
        image = ax.imshow(values, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(np.arange(len(heatmap_data.columns)), heatmap_data.columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(heatmap_data.index)), heatmap_data.index)
        for y in range(values.shape[0]):
            for x in range(values.shape[1]):
                if np.isfinite(values[y, x]):
                    ax.text(x, y, f"{values[y, x]:.3f}", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("All BrC/T2D Models: Balanced Accuracy and Weighted F1")
    fig.tight_layout()
    path = out_dir / "all_models_balanced_accuracy_weighted_f1_heatmap.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_tables(data: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    combined_path = out_dir / "combined_deduplicated_summary.csv"
    data.drop(columns=["label_order"]).to_csv(combined_path, index=False)
    paths.append(combined_path)

    best_rows = []
    for metric in available_metrics(data, MAIN_METRICS + ["roc_auc", "pr_auc", "macro_roc_auc_ovr"]):
        metric_data = data[data[metric].notna()].copy()
        if metric_data.empty:
            continue
        best = metric_data.sort_values(metric, ascending=False).groupby(["label_column", "approach"], as_index=False).first()
        best["metric"] = metric
        best["score"] = best[metric]
        best_rows.append(best[["label_column", "approach", "model_display", "metric", "score"]])
    if best_rows:
        best_path = out_dir / "best_by_family_and_metric.csv"
        pd.concat(best_rows, ignore_index=True).to_csv(best_path, index=False)
        paths.append(best_path)

    for rank_metric in ("balanced_accuracy", "weighted_f1"):
        ranking = data[data[rank_metric].notna()].copy()
        if ranking.empty:
            continue
        ranking["rank"] = ranking.groupby("label_column")[rank_metric].rank(ascending=False, method="min")
        ranking = ranking.sort_values(["label_order", "rank", "approach", "model_display"])
        rank_path = out_dir / f"all_models_ranked_by_{rank_metric}.csv"
        ranking[["label_column", "rank", "approach", "model_display", rank_metric, "accuracy", "macro_f1"]].to_csv(
            rank_path, index=False
        )
        paths.append(rank_path)

    pairwise_rows = []
    for metric in PAIRWISE_METRICS:
        best = best_rows_for_metric(data, metric)
        if best.empty:
            continue
        for left, right in PAIRWISE_COMPARISONS:
            for label in LABEL_ORDER:
                left_row = best[(best["label_column"].eq(label)) & (best["approach"].eq(left))]
                right_row = best[(best["label_column"].eq(label)) & (best["approach"].eq(right))]
                if left_row.empty or right_row.empty:
                    continue
                left_row = left_row.iloc[0]
                right_row = right_row.iloc[0]
                pairwise_rows.append(
                    {
                        "label_column": label,
                        "metric": metric,
                        "comparison": f"{left}_vs_{right}",
                        "left_approach": left,
                        "left_model": left_row["model_display"],
                        "left_score": left_row[metric],
                        "right_approach": right,
                        "right_model": right_row["model_display"],
                        "right_score": right_row[metric],
                        "delta_left_minus_right": left_row[metric] - right_row[metric],
                    }
                )
    if pairwise_rows:
        pairwise_path = out_dir / "pairwise_best_model_deltas.csv"
        pd.DataFrame(pairwise_rows).to_csv(pairwise_path, index=False)
        paths.append(pairwise_path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masking-dir", default=DEFAULT_MASKING_DIR)
    parser.add_argument("--jigsaw-dir", default=DEFAULT_JIGSAW_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-n", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = read_summaries(Path(args.masking_dir), Path(args.jigsaw_dir))
    table_paths = write_tables(data, out_dir)

    plot_paths: list[Path] = []
    path = plot_all_models_core_heatmap(data, out_dir)
    if path is not None:
        plot_paths.append(path)
    plot_paths.extend(plot_metric_heatmaps(data, "classical", out_dir))
    plot_paths.extend(plot_metric_heatmaps(data, "masking", out_dir))
    plot_paths.extend(plot_metric_heatmaps(data, "jigsaw", out_dir))
    for metric in ("balanced_accuracy", "macro_f1", "accuracy"):
        path = plot_best_family_comparison(data, metric, out_dir)
        if path is not None:
            plot_paths.append(path)
    for left, right in PAIRWISE_COMPARISONS:
        for metric in PAIRWISE_METRICS:
            path = plot_pairwise_best_comparison(data, left, right, metric, out_dir)
            if path is not None:
                plot_paths.append(path)
    for approach in ("masking", "jigsaw"):
        path = plot_finetune_gains(data, approach, "balanced_accuracy", out_dir)
        if path is not None:
            plot_paths.append(path)
    for metric in ("balanced_accuracy", "weighted_f1"):
        path = plot_top_rankings(data, metric, out_dir, args.top_n)
        if path is not None:
            plot_paths.append(path)

    print(f"Wrote tables to {out_dir}:")
    for path in table_paths:
        print(f"  - {path}")
    print("Wrote plots:")
    for path in plot_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
