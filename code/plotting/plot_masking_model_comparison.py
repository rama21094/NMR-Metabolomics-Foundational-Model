#!/usr/bin/env python3
"""
Plot comparative few-shot ML results across MAE masking ratios.

Expected inputs:
  results/fewshot/mask_0.2/fewshot_ml_comparison_summary.csv
  results/fewshot/mask_0.3/fewshot_ml_comparison_summary.csv
  results/fewshot/mask_0.4/fewshot_ml_comparison_summary.csv
  results/fewshot/mask_0.5/fewshot_ml_comparison_summary.csv

Each folder is treated as a separate backbone trained with the corresponding
masking ratio: 20%, 30%, 40%, and 50%.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


USE_IDE_CONFIG = True

IDE_CONFIG = {
    "result_dirs": [
        "results/fewshot/mask_0.2",
        "results/fewshot/mask_0.3",
        "results/fewshot/mask_0.4",
        "results/fewshot/mask_0.5",
    ],
    "summary_filename": "fewshot_ml_comparison_summary.csv",
    "out_dir": "results/fewshot/masking_comparison_plots",
    "metrics": ["accuracy_mean", "macro_f1_mean"],
    "include_error_bars": True,
    "top_n_methods": 10,
}


MASKING_LABELS = {
    "0.2": "20%",
    "0.3": "30%",
    "0.4": "40%",
    "0.5": "50%",
}


def parse_csv_list(text):
    if isinstance(text, list):
        return text
    return [x.strip() for x in str(text).split(",") if x.strip()]


def str2bool(v):
    if isinstance(v, bool):
        return v
    val = str(v).strip().lower()
    if val in {"1", "true", "t", "yes", "y"}:
        return True
    if val in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def infer_masking_ratio(result_dir):
    name = Path(result_dir).name
    suffix = name.rsplit("_", 1)[-1]
    try:
        ratio = float(suffix)
    except ValueError as exc:
        raise ValueError(f"Could not infer masking ratio from directory name: {result_dir}") from exc
    label = MASKING_LABELS.get(suffix, f"{int(round(ratio * 100))}%")
    return ratio, label


def clean_method_label(row):
    group = row["feature_group"]
    feature = row["feature_name"]
    clf = row["classifier"]

    if group == "direct_binned":
        return f"Direct {feature.replace('bins', '').replace('_auc', '')} AUC + {clf.upper()}"
    if group == "foundation_backbone":
        return f"MAE embeddings + {clf.upper()}"
    if group == "prototype_head":
        return "MAE prototype head"
    return f"{group} | {feature} | {clf}"


def load_results(result_dirs, summary_filename):
    frames = []
    for result_dir in result_dirs:
        path = Path(result_dir) / summary_filename
        if not path.exists():
            raise FileNotFoundError(f"Missing summary file: {path}")

        ratio, label = infer_masking_ratio(result_dir)
        df = pd.read_csv(path)
        df["masking_ratio"] = ratio
        df["masking_label"] = label
        df["source_dir"] = str(result_dir)
        df["method_id"] = df["feature_group"] + "|" + df["feature_name"] + "|" + df["classifier"]
        df["method_label"] = df.apply(clean_method_label, axis=1)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["masking_ratio", "feature_group", "feature_name", "classifier"]).reset_index(drop=True)
    return out


def metric_std_name(metric):
    if metric.endswith("_mean"):
        return metric.replace("_mean", "_std")
    return None


def plot_metric_trends(df, metric, out_dir, include_error_bars, top_n_methods):
    method_rank = (
        df.groupby("method_label", as_index=False)[metric]
        .mean()
        .sort_values(metric, ascending=False)
        .head(top_n_methods)
    )
    selected = method_rank["method_label"].tolist()
    work = df[df["method_label"].isin(selected)].copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    std_col = metric_std_name(metric)
    for method, gdf in work.groupby("method_label", sort=False):
        gdf = gdf.sort_values("masking_ratio")
        x = gdf["masking_ratio"].to_numpy() * 100
        y = gdf[metric].to_numpy()
        if include_error_bars and std_col in gdf.columns:
            yerr = gdf[std_col].fillna(0.0).to_numpy()
            ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.8, capsize=3, label=method)
        else:
            ax.plot(x, y, marker="o", linewidth=1.8, label=method)

    ax.set_title(f"{metric.replace('_', ' ').title()} Across Masking Ratios")
    ax.set_xlabel("MAE masking ratio")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xticks(sorted(df["masking_ratio"].unique() * 100))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()

    path = Path(out_dir) / f"{metric}_trend_top{top_n_methods}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_grouped_bar(df, metric, out_dir):
    work = df.copy()
    work["mask_method"] = work["masking_label"] + " | " + work["method_label"]
    work = work.sort_values(["masking_ratio", metric], ascending=[True, False])

    labels = work["mask_method"].tolist()
    x = np.arange(len(work))

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = {
        "direct_binned": "#6c8ebf",
        "foundation_backbone": "#2a9d8f",
        "prototype_head": "#e76f51",
    }
    bar_colors = [colors.get(g, "#888888") for g in work["feature_group"]]
    ax.bar(x, work[metric].to_numpy(), color=bar_colors)
    ax.set_title(f"All Methods by Masking Ratio: {metric.replace('_', ' ').title()}")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()

    path = Path(out_dir) / f"{metric}_all_methods_bar.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_heatmap(df, metric, out_dir):
    pivot = df.pivot_table(
        index="method_label",
        columns="masking_label",
        values=metric,
        aggfunc="mean",
    )
    ordered_cols = [
        MASKING_LABELS.get(str(r), f"{int(round(float(r) * 100))}%")
        for r in sorted(df["masking_ratio"].unique())
    ]
    pivot = pivot.reindex(columns=ordered_cols)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(9, max(5, 0.45 * len(pivot))))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_title(f"Heatmap: {metric.replace('_', ' ').title()}")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iat[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="white", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    fig.tight_layout()

    path = Path(out_dir) / f"{metric}_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_best_by_masking(df, metric, out_dir):
    best = (
        df.sort_values(["masking_ratio", metric], ascending=[True, False])
        .groupby("masking_label", as_index=False)
        .head(1)
        .sort_values("masking_ratio")
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(best))
    ax.bar(x, best[metric].to_numpy(), color="#264653")
    ax.set_title(f"Best Method at Each Masking Ratio: {metric.replace('_', ' ').title()}")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xticks(x)
    ax.set_xticklabels(best["masking_label"])
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for i, row in enumerate(best.itertuples(index=False)):
        ax.text(
            i,
            getattr(row, metric) + 0.01,
            row.method_label,
            rotation=30,
            ha="left",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    path = Path(out_dir) / f"{metric}_best_by_masking.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_rank_tables(df, out_dir):
    ranked = df.sort_values(["masking_ratio", "accuracy_mean"], ascending=[True, False]).copy()
    ranked["rank_accuracy_within_masking"] = ranked.groupby("masking_ratio")["accuracy_mean"].rank(
        method="first",
        ascending=False,
    ).astype(int)

    overall = (
        df.groupby(["method_id", "method_label", "feature_group", "feature_name", "classifier"], as_index=False)
        .agg(
            accuracy_mean_across_masks=("accuracy_mean", "mean"),
            macro_f1_mean_across_masks=("macro_f1_mean", "mean"),
            accuracy_std_across_masks=("accuracy_mean", "std"),
            macro_f1_std_across_masks=("macro_f1_mean", "std"),
        )
        .sort_values("accuracy_mean_across_masks", ascending=False)
        .reset_index(drop=True)
    )
    overall["overall_rank_accuracy"] = np.arange(1, len(overall) + 1)

    ranked_path = Path(out_dir) / "masking_comparison_ranked_by_ratio.csv"
    overall_path = Path(out_dir) / "masking_comparison_overall_ranking.csv"
    ranked.to_csv(ranked_path, index=False)
    overall.to_csv(overall_path, index=False)
    return ranked_path, overall_path


def build_parser():
    parser = argparse.ArgumentParser(description="Plot few-shot comparison summaries across MAE masking ratios.")
    parser.add_argument("--result-dirs", nargs="+", default=IDE_CONFIG["result_dirs"])
    parser.add_argument("--summary-filename", default=IDE_CONFIG["summary_filename"])
    parser.add_argument("--out-dir", default=IDE_CONFIG["out_dir"])
    parser.add_argument("--metrics", type=parse_csv_list, default=IDE_CONFIG["metrics"])
    parser.add_argument("--include-error-bars", type=str2bool, default=IDE_CONFIG["include_error_bars"])
    parser.add_argument("--top-n-methods", type=int, default=IDE_CONFIG["top_n_methods"])
    return parser


def args_from_ide_config():
    parser = build_parser()
    args = parser.parse_args([])
    for key, value in IDE_CONFIG.items():
        setattr(args, key, value)
    return args


def main():
    args = args_from_ide_config() if USE_IDE_CONFIG else build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.result_dirs, args.summary_filename)
    combined_path = out_dir / "masking_comparison_combined_summary.csv"
    df.to_csv(combined_path, index=False)

    plot_paths = []
    for metric in args.metrics:
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found. Available columns: {list(df.columns)}")
        plot_paths.append(plot_metric_trends(df, metric, out_dir, args.include_error_bars, args.top_n_methods))
        plot_paths.append(plot_grouped_bar(df, metric, out_dir))
        plot_paths.append(plot_heatmap(df, metric, out_dir))
        plot_paths.append(plot_best_by_masking(df, metric, out_dir))

    ranked_path, overall_path = save_rank_tables(df, out_dir)

    print("Saved combined summary:", combined_path)
    print("Saved ranked table:", ranked_path)
    print("Saved overall ranking:", overall_path)
    print("Saved plots:")
    for path in plot_paths:
        print("  -", path)


if __name__ == "__main__":
    main()
