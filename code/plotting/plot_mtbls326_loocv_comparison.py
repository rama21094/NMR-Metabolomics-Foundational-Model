#!/usr/bin/env python3
"""Create comparison visuals for MTBLS326 masking and jigsaw LOOCV summaries."""

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


DEFAULT_MASKING_DIR = "results/loocv/mtbls326_mask_0.50_rowMinMax"
DEFAULT_JIGSAW_DIR = "results/loocv/mtbls326_jigsaw"
DEFAULT_OUT_DIR = "results/loocv/mtbls326_comparison_plots"
MAIN_METRICS = ["accuracy", "balanced_accuracy", "weighted_f1", "f1", "roc_auc", "pr_auc"]
PAIRWISE_COMPARISONS = [
    ("jigsaw", "classical"),
    ("masking", "classical"),
    ("jigsaw", "masking"),
]
PAIRWISE_METRICS = ["balanced_accuracy", "weighted_f1"]
COLORS = {"classical": "#4C78A8", "masking": "#59A14F", "jigsaw": "#F28E2B"}


def metric_label(metric: str) -> str:
    return metric.replace("_", " ").title()


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


def f1_from_counts(tp: float, fp: float, fn: float) -> float:
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0


def add_weighted_f1(data: pd.DataFrame) -> pd.DataFrame:
    if "weighted_f1" in data.columns and data["weighted_f1"].notna().any():
        return data
    required = {"tn", "fp", "fn", "tp"}
    if not required.issubset(data.columns):
        data["weighted_f1"] = np.nan
        return data

    weighted = []
    for _, row in data.iterrows():
        tn = float(row["tn"])
        fp = float(row["fp"])
        fn = float(row["fn"])
        tp = float(row["tp"])
        negative_support = tn + fp
        positive_support = tp + fn
        total = negative_support + positive_support
        negative_f1 = f1_from_counts(tn, fn, fp)
        positive_f1 = f1_from_counts(tp, fp, fn)
        weighted.append(
            (negative_f1 * negative_support + positive_f1 * positive_support) / total
            if total
            else np.nan
        )
    data["weighted_f1"] = weighted
    return data


def read_summary(path: Path, source: str) -> pd.DataFrame:
    summary_path = path / "summary.csv" if path.is_dir() else path
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.csv: {summary_path}")
    df = pd.read_csv(summary_path)
    df["source"] = source
    return df


def read_summaries(masking_dir: Path, jigsaw_dir: Path) -> pd.DataFrame:
    data = pd.concat(
        [
            read_summary(masking_dir, "masking"),
            read_summary(jigsaw_dir, "jigsaw"),
        ],
        ignore_index=True,
    )
    data["approach"] = [approach_from_row(s, f) for s, f in zip(data["source"], data["family"])]
    data["model_display"] = data.apply(clean_model_name, axis=1)
    data["tune_mode"] = data["model"].map(tune_mode)
    data["jigsaw_backbone"] = data["model"].map(jigsaw_backbone)

    is_classical = data["approach"].eq("classical")
    classical = data[is_classical].sort_values(["model", "source"]).drop_duplicates(
        ["family", "model"], keep="first"
    )
    cleaned = pd.concat([classical, data[~is_classical]], ignore_index=True)

    for metric in set(MAIN_METRICS + ["precision", "recall", "tn", "fp", "fn", "tp"]):
        if metric not in cleaned.columns:
            cleaned[metric] = np.nan
        cleaned[metric] = pd.to_numeric(cleaned[metric], errors="coerce")
    return add_weighted_f1(cleaned).sort_values(["approach", "model_display"]).reset_index(drop=True)


def available_metrics(data: pd.DataFrame, metrics: list[str]) -> list[str]:
    return [metric for metric in metrics if metric in data.columns and data[metric].notna().any()]


def plot_metric_heatmap(data: pd.DataFrame, approach: str, out_dir: Path) -> Path | None:
    subset = data[data["approach"].eq(approach)].copy()
    metrics = available_metrics(subset, MAIN_METRICS)
    if subset.empty or not metrics:
        return None
    pivot = subset.pivot_table(index="model_display", values=metrics, aggfunc="max")
    pivot = pivot[metrics].sort_values("balanced_accuracy", ascending=False)

    fig_h = max(4.0, 0.38 * len(pivot) + 1.6)
    fig, ax = plt.subplots(figsize=(9.4, fig_h))
    if sns is not None:
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0.0, vmax=1.0, linewidths=0.5, ax=ax)
    else:
        values = pivot.to_numpy(dtype=float)
        image = ax.imshow(values, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(np.arange(len(pivot.columns)), [metric_label(m) for m in pivot.columns], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
        for y in range(values.shape[0]):
            for x in range(values.shape[1]):
                if np.isfinite(values[y, x]):
                    ax.text(x, y, f"{values[y, x]:.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"MTBLS326 {approach.title()} Metrics")
    fig.tight_layout()
    path = out_dir / f"{approach}_metric_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def best_rows_for_metric(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    subset = data[data[metric].notna()].copy()
    if subset.empty:
        return pd.DataFrame()
    return subset.sort_values(metric, ascending=False).groupby("approach", as_index=False).first()


def plot_best_family_comparison(data: pd.DataFrame, metric: str, out_dir: Path) -> Path | None:
    best = best_rows_for_metric(data, metric)
    if best.empty:
        return None
    order = [approach for approach in ("classical", "masking", "jigsaw") if approach in set(best["approach"])]
    values = []
    names = []
    for approach in order:
        row = best[best["approach"].eq(approach)].iloc[0]
        values.append(float(row[metric]))
        names.append(str(row["model_display"]).replace("unfreeze_last_", "u"))

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    bars = ax.bar(order, values, color=[COLORS[a] for a in order], width=0.55)
    for bar, value, name in zip(bars, values, names):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}\n{name}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, min(1.16, max(1.0, max(values) + 0.18)))
    ax.set_ylabel(metric_label(metric))
    ax.set_title(f"MTBLS326 Best Model Per Family: {metric_label(metric)}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = out_dir / f"best_family_{metric}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_pairwise_best_comparison(data: pd.DataFrame, left: str, right: str, metric: str, out_dir: Path) -> Path | None:
    best = best_rows_for_metric(data, metric)
    if best.empty:
        return None
    rows = []
    for approach in (left, right):
        row = best[best["approach"].eq(approach)]
        if row.empty:
            return None
        rows.append(row.iloc[0])
    values = [float(row[metric]) for row in rows]
    names = [str(row["model_display"]).replace("unfreeze_last_", "u") for row in rows]

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    bars = ax.bar([left, right], values, color=[COLORS[left], COLORS[right]], width=0.55)
    for bar, value, name in zip(bars, values, names):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}\n{name}", ha="center", va="bottom", fontsize=8)
    ax.text(
        0.5,
        0.06,
        f"Delta {left} - {right}: {values[0] - values[1]:+.3f}",
        transform=ax.transAxes,
        ha="center",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#dddddd"},
    )
    ax.set_ylim(0, min(1.16, max(1.0, max(values) + 0.18)))
    ax.set_ylabel(metric_label(metric))
    ax.set_title(f"MTBLS326 {left.title()} vs {right.title()}: {metric_label(metric)}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = out_dir / f"pairwise_best_{left}_vs_{right}_{metric}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_finetune_trend(data: pd.DataFrame, approach: str, metric: str, out_dir: Path) -> Path | None:
    subset = data[(data["approach"].eq(approach)) & (data[metric].notna())].copy()
    if subset.empty:
        return None
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    if approach == "masking":
        group = subset.sort_values("tune_mode", key=lambda s: s.map(tune_order))
        ax.plot([tune_order(m) for m in group["tune_mode"]], group[metric], marker="o", linewidth=2.2, color=COLORS[approach])
        ax.set_xticks([tune_order(m) for m in group["tune_mode"]], group["tune_mode"], rotation=35, ha="right")
    else:
        for backbone, group in subset.groupby("jigsaw_backbone"):
            group = group.sort_values("tune_mode", key=lambda s: s.map(tune_order))
            ax.plot([tune_order(m) for m in group["tune_mode"]], group[metric], marker="o", linewidth=1.8, label=backbone)
        ax.set_xticks([0, 1, 2, 3], ["frozen", "u1", "u2", "u3"])
        ax.legend(fontsize=8, frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    ax.set_xlabel("Fine-tune mode")
    ax.set_ylabel(metric_label(metric))
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(f"MTBLS326 {approach.title()} Fine-Tuning Trend: {metric_label(metric)}")
    fig.tight_layout()
    path = out_dir / f"{approach}_finetune_trend_{metric}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_top_rankings(data: pd.DataFrame, metric: str, out_dir: Path, top_n: int) -> Path | None:
    subset = data[data[metric].notna()].copy()
    if subset.empty:
        return None
    ranked = subset.sort_values(metric, ascending=False).head(top_n).sort_values(metric)
    labels = ranked["approach"].str.title() + ": " + ranked["model_display"]
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.barh(labels, ranked[metric], color=[COLORS[a] for a in ranked["approach"]])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(metric_label(metric))
    ax.set_title(f"MTBLS326 Top {top_n} Models: {metric_label(metric)}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = out_dir / f"top_{top_n}_{metric}_rankings.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_tables(data: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    combined_path = out_dir / "combined_deduplicated_summary.csv"
    data.to_csv(combined_path, index=False)
    paths.append(combined_path)

    best_rows = []
    for metric in available_metrics(data, MAIN_METRICS):
        best = best_rows_for_metric(data, metric)
        if best.empty:
            continue
        best["metric"] = metric
        best["score"] = best[metric]
        best_rows.append(best[["approach", "model_display", "metric", "score"]])
    if best_rows:
        best_path = out_dir / "best_by_family_and_metric.csv"
        pd.concat(best_rows, ignore_index=True).to_csv(best_path, index=False)
        paths.append(best_path)

    for metric in ("balanced_accuracy", "weighted_f1"):
        ranking = data[data[metric].notna()].copy()
        if ranking.empty:
            continue
        ranking["rank"] = ranking[metric].rank(ascending=False, method="min")
        ranking = ranking.sort_values(["rank", "approach", "model_display"])
        rank_path = out_dir / f"all_models_ranked_by_{metric}.csv"
        ranking[["rank", "approach", "model_display", metric, "accuracy", "f1", "roc_auc", "pr_auc"]].to_csv(
            rank_path, index=False
        )
        paths.append(rank_path)

    pairwise_rows = []
    for metric in PAIRWISE_METRICS:
        best = best_rows_for_metric(data, metric)
        if best.empty:
            continue
        for left, right in PAIRWISE_COMPARISONS:
            left_row = best[best["approach"].eq(left)]
            right_row = best[best["approach"].eq(right)]
            if left_row.empty or right_row.empty:
                continue
            left_row = left_row.iloc[0]
            right_row = right_row.iloc[0]
            pairwise_rows.append(
                {
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
    for approach in ("classical", "masking", "jigsaw"):
        path = plot_metric_heatmap(data, approach, out_dir)
        if path is not None:
            plot_paths.append(path)
    for metric in ("balanced_accuracy", "weighted_f1", "accuracy", "roc_auc", "pr_auc"):
        path = plot_best_family_comparison(data, metric, out_dir)
        if path is not None:
            plot_paths.append(path)
    for left, right in PAIRWISE_COMPARISONS:
        for metric in PAIRWISE_METRICS:
            path = plot_pairwise_best_comparison(data, left, right, metric, out_dir)
            if path is not None:
                plot_paths.append(path)
    for approach in ("masking", "jigsaw"):
        path = plot_finetune_trend(data, approach, "balanced_accuracy", out_dir)
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
