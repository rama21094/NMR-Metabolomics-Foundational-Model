#!/usr/bin/env python3
"""
Plot support-count sweeps across MAE masking-ratio models.

Expected input layout:
  results/fewshot/support_sweep/mask_0.20/support_1/fewshot_ml_comparison_summary.csv
  results/fewshot/support_sweep/mask_0.20/support_2/fewshot_ml_comparison_summary.csv
  ...
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
    "root_dir": "results/fewshot/support_sweep",
    "summary_filename": "fewshot_ml_comparison_summary.csv",
    "out_dir": "results/fewshot/support_sweep_plots",
    "metrics": ["accuracy_mean", "macro_f1_mean"],
    "include_error_bars": True,
    "top_n_methods": 10,
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


def parse_masking_ratio(mask_dir):
    text = Path(mask_dir).name.replace("mask_", "")
    return float(text)


def parse_support_value(support_dir):
    text = Path(support_dir).name.replace("support_", "")
    return int(text)


def masking_label(ratio):
    return f"{int(round(float(ratio) * 100))}%"


def load_results(root_dir, summary_filename):
    root = Path(root_dir)
    paths = sorted(root.glob(f"mask_*/support_*/{summary_filename}"))
    if not paths:
        raise FileNotFoundError(f"No summary files found under {root}/{summary_filename}")

    frames = []
    for path in paths:
        support_dir = path.parent
        mask_dir = support_dir.parent
        ratio = parse_masking_ratio(mask_dir)
        support_value = parse_support_value(support_dir)

        df = pd.read_csv(path)
        df["masking_ratio"] = ratio
        df["masking_label"] = masking_label(ratio)
        df["support_per_class"] = support_value
        df["source_dir"] = str(support_dir)
        df["method_id"] = df["feature_group"] + "|" + df["feature_name"] + "|" + df["classifier"]
        df["method_label"] = df.apply(clean_method_label, axis=1)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(
        ["masking_ratio", "support_per_class", "feature_group", "feature_name", "classifier"]
    ).reset_index(drop=True)
    return out


def metric_std_name(metric):
    if metric.endswith("_mean"):
        return metric.replace("_mean", "_std")
    return None


def top_methods(df, metric, top_n):
    return (
        df.groupby("method_label", as_index=False)[metric]
        .mean()
        .sort_values(metric, ascending=False)
        .head(top_n)["method_label"]
        .tolist()
    )


def plot_metric_by_support(df, metric, out_dir, include_error_bars, top_n_methods):
    selected = top_methods(df, metric, top_n_methods)
    work = df[df["method_label"].isin(selected)].copy()
    masks = sorted(work["masking_ratio"].unique())
    std_col = metric_std_name(metric)

    ncols = 2
    nrows = int(np.ceil(len(masks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows), squeeze=False, sharey=True)

    for ax, ratio in zip(axes.ravel(), masks):
        gmask = work[work["masking_ratio"] == ratio]
        for method, gdf in gmask.groupby("method_label", sort=False):
            gdf = gdf.sort_values("support_per_class")
            x = gdf["support_per_class"].to_numpy()
            y = gdf[metric].to_numpy()
            if include_error_bars and std_col in gdf.columns:
                yerr = gdf[std_col].fillna(0.0).to_numpy()
                ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.7, capsize=3, label=method)
            else:
                ax.plot(x, y, marker="o", linewidth=1.7, label=method)
        ax.set_title(f"Masking {masking_label(ratio)}")
        ax.set_xlabel("Support samples per class")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    for ax in axes[:, 0]:
        ax.set_ylabel(metric.replace("_", " ").title())
    for ax in axes.ravel()[len(masks):]:
        ax.axis("off")

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle(f"{metric.replace('_', ' ').title()} vs Support Count", y=0.995)
    fig.tight_layout(rect=[0, 0, 0.82, 0.97])

    path = Path(out_dir) / f"{metric}_vs_support_by_masking.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_masking_support_heatmaps(df, metric, out_dir, top_n_methods):
    selected = top_methods(df, metric, top_n_methods)
    paths = []

    for method in selected:
        work = df[df["method_label"] == method]
        pivot = work.pivot_table(
            index="masking_label",
            columns="support_per_class",
            values=metric,
            aggfunc="mean",
        )
        ordered_rows = [masking_label(r) for r in sorted(work["masking_ratio"].unique())]
        pivot = pivot.reindex(index=ordered_rows)
        pivot = pivot.reindex(columns=sorted(work["support_per_class"].unique()))

        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
        ax.set_title(f"{method}\n{metric.replace('_', ' ').title()}")
        ax.set_xlabel("Support samples per class")
        ax.set_ylabel("Masking ratio")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iat[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="white", fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        fig.tight_layout()

        safe_name = (
            method.lower()
            .replace(" ", "_")
            .replace("|", "_")
            .replace("+", "plus")
            .replace("%", "pct")
            .replace("/", "_")
        )
        path = Path(out_dir) / f"{metric}_heatmap_{safe_name}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    return paths


def plot_best_by_support_and_masking(df, metric, out_dir):
    best = (
        df.sort_values(["masking_ratio", "support_per_class", metric], ascending=[True, True, False])
        .groupby(["masking_ratio", "support_per_class"], as_index=False)
        .head(1)
        .sort_values(["masking_ratio", "support_per_class"])
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    for ratio, gdf in best.groupby("masking_ratio"):
        ax.plot(
            gdf["support_per_class"],
            gdf[metric],
            marker="o",
            linewidth=2,
            label=f"Masking {masking_label(ratio)}",
        )
        for row in gdf.itertuples(index=False):
            ax.text(
                row.support_per_class,
                getattr(row, metric) + 0.005,
                row.method_label,
                rotation=30,
                ha="left",
                va="bottom",
                fontsize=7,
            )

    ax.set_title(f"Best Method per Support Count and Masking Ratio: {metric.replace('_', ' ').title()}")
    ax.set_xlabel("Support samples per class")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()

    path = Path(out_dir) / f"{metric}_best_by_support_and_masking.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_rank_tables(df, out_dir):
    ranked = df.sort_values(
        ["masking_ratio", "support_per_class", "accuracy_mean"],
        ascending=[True, True, False],
    ).copy()
    ranked["rank_accuracy_within_setting"] = ranked.groupby(
        ["masking_ratio", "support_per_class"]
    )["accuracy_mean"].rank(method="first", ascending=False).astype(int)

    overall = (
        df.groupby(["method_id", "method_label", "feature_group", "feature_name", "classifier"], as_index=False)
        .agg(
            accuracy_mean_across_sweep=("accuracy_mean", "mean"),
            macro_f1_mean_across_sweep=("macro_f1_mean", "mean"),
            accuracy_std_across_sweep=("accuracy_mean", "std"),
            macro_f1_std_across_sweep=("macro_f1_mean", "std"),
        )
        .sort_values("accuracy_mean_across_sweep", ascending=False)
        .reset_index(drop=True)
    )
    overall["overall_rank_accuracy"] = np.arange(1, len(overall) + 1)

    ranked_path = Path(out_dir) / "support_sweep_ranked_by_setting.csv"
    overall_path = Path(out_dir) / "support_sweep_overall_ranking.csv"
    ranked.to_csv(ranked_path, index=False)
    overall.to_csv(overall_path, index=False)
    return ranked_path, overall_path


def build_parser():
    parser = argparse.ArgumentParser(description="Plot support-count sweep summaries across masking ratios.")
    parser.add_argument("--root-dir", default=IDE_CONFIG["root_dir"])
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


def run_plotting(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.root_dir, args.summary_filename)
    combined_path = out_dir / "support_sweep_combined_summary.csv"
    df.to_csv(combined_path, index=False)

    plot_paths = []
    for metric in args.metrics:
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found. Available columns: {list(df.columns)}")
        plot_paths.append(plot_metric_by_support(df, metric, out_dir, args.include_error_bars, args.top_n_methods))
        plot_paths.extend(plot_masking_support_heatmaps(df, metric, out_dir, args.top_n_methods))
        plot_paths.append(plot_best_by_support_and_masking(df, metric, out_dir))

    ranked_path, overall_path = save_rank_tables(df, out_dir)

    print("Saved combined summary:", combined_path)
    print("Saved ranked table:", ranked_path)
    print("Saved overall ranking:", overall_path)
    print("Saved plots:")
    for path in plot_paths:
        print("  -", path)

    return {
        "combined_summary": str(combined_path),
        "ranked_table": str(ranked_path),
        "overall_ranking": str(overall_path),
        "plots": [str(p) for p in plot_paths],
    }


def main():
    args = args_from_ide_config() if USE_IDE_CONFIG else build_parser().parse_args()
    run_plotting(args)


if __name__ == "__main__":
    main()
