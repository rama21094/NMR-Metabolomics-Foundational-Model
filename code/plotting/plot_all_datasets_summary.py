#!/usr/bin/env python3
"""Cross-dataset summary figures for the MetaboLights-corpus SSL checkpoints.

Reads the completed LOOCV/CV summary.csv files for Barth, MTBLS326, and
MTBLS563 (all evaluated on the masked/jigsaw/joint_ssl checkpoints trained on
`combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax.npy`)
and produces:

  1. fig1_balanced_accuracy.png -- grouped bar, best fine-tune mode per family,
     balanced accuracy, one group of bars per dataset.
  2. fig2_roc_auc.png -- same layout, ROC-AUC (binary) / macro ROC-AUC OVR
     (MTBLS563's 3-class problem).
  3. fig3_finetune_depth.png -- balanced accuracy vs. fine-tune depth
     (frozen -> unfreeze_last_3), one subplot per dataset, classical shown as
     a flat reference line since it has no fine-tuning modes.

The masked-model family is flagged (hatched bars, footnote) rather than
dropped: its Barth result is unreliable (see the run's README) but the number
is still shown for transparency.
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

# Validated categorical palette (dataviz skill, references/palette.md, slots 1/2/3/6).
# Assigned in a fixed order -- never remapped per-figure.
FAMILY_COLORS = {
    "classical": "#2a78d6",
    "masked": "#1baf7a",
    "jigsaw": "#eda100",
    "joint_ssl": "#e34948",
}
FAMILY_LABELS = {
    "classical": "Classical (logistic regression)",
    "masked": "Masked SSL",
    "jigsaw": "Jigsaw SSL",
    "joint_ssl": "Joint SSL",
}
FAMILY_ORDER = ["classical", "masked", "jigsaw", "joint_ssl"]

FINE_TUNE_ORDER = ["frozen", "unfreeze_last_1", "unfreeze_last_2", "unfreeze_last_3"]


def _strip_mode(model_name: str, family: str) -> str:
    for mode in FINE_TUNE_ORDER:
        if model_name.endswith(mode):
            return mode
    return "" if family == "classical" else model_name


def _is_degenerate(row: pd.Series) -> bool:
    """Flag a row whose confusion matrix shows it collapsed to predicting a
    single class (tp==0 or tn==0 in the binary case). Computed from the run's
    own numbers rather than a hardcoded list, so it never goes stale when a
    checkpoint is retrained."""
    if "tp" not in row or "tn" not in row:
        return False
    return bool(row["tp"] == 0 or row["tn"] == 0)


def load_dataset(name: str, sources: list[tuple[str, str, dict]]) -> pd.DataFrame:
    """sources: list of (csv_path, roc_auc_column, family_rename_map)."""
    rows = []
    for csv_path, roc_col, rename in sources:
        df = pd.read_csv(csv_path)
        df["family"] = df["family"].replace(rename)
        df["mode"] = [
            _strip_mode(model, family) for model, family in zip(df["model"], df["family"])
        ]
        df["roc_auc_display"] = df[roc_col]
        df["degenerate"] = df.apply(_is_degenerate, axis=1)
        cols = ["family", "model", "mode", "balanced_accuracy", "roc_auc_display", "degenerate"]
        rows.append(df[cols])
    out = pd.concat(rows, ignore_index=True)
    out["dataset"] = name
    return out


def best_per_family(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("family")["balanced_accuracy"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def add_value_labels(ax, bars, fmt="{:.2f}"):
    for bar in bars:
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#0b0b0b",
        )


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.yaxis.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def grouped_bar_figure(best_by_dataset: dict[str, pd.DataFrame], metric: str, ylabel: str, title: str, out_path: Path):
    datasets = list(best_by_dataset.keys())
    n_families = len(FAMILY_ORDER)
    width = 0.8 / n_families
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    for i, family in enumerate(FAMILY_ORDER):
        xs = np.arange(len(datasets)) + (i - (n_families - 1) / 2) * width
        heights = []
        hatches = []
        for dataset in datasets:
            df = best_by_dataset[dataset]
            match = df[df["family"] == family]
            heights.append(float(match[metric].iloc[0]) if len(match) else np.nan)
            hatches.append("////" if (len(match) and bool(match["degenerate"].iloc[0])) else None)
        bars = ax.bar(
            xs, heights, width=width * 0.92, color=FAMILY_COLORS[family],
            label=FAMILY_LABELS[family], zorder=3,
        )
        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_hatch(hatch)
                bar.set_edgecolor("#52514e")
                bar.set_linewidth(0.6)
        add_value_labels(ax, bars)

    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels(datasets)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.08)
    ax.axhline(0.5, color="#898781", linewidth=1, linestyle="--", zorder=1)
    ax.set_title(title, fontsize=12, color="#0b0b0b", loc="left")
    style_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=9)
    fig.text(
        0.01, 0.01,
        "Hatched bar: model collapsed to predicting a single class (tp==0 or tn==0 in its confusion matrix).\n"
        "Dashed line: chance level (0.5). MTBLS563 ROC-AUC is macro one-vs-rest (3-class); others are binary.",
        fontsize=7, color="#898781",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


MODE_LABELS = {"": "", "frozen": "frozen", "unfreeze_last_1": "+1 layer", "unfreeze_last_2": "+2 layers", "unfreeze_last_3": "+3 layers"}
CLASSICAL_MODEL_LABELS = {"logistic_regression": "logistic regression", "svm_rbf": "SVM (RBF)", "xgboost": "XGBoost"}


def row_key(family: str, mode: str, model: str) -> tuple[str, str, str]:
    if family == "classical":
        return (family, model, CLASSICAL_MODEL_LABELS.get(model, model))
    return (family, mode, f"{FAMILY_LABELS[family]} ({MODE_LABELS.get(mode, mode)})")


def all_rows(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    seen = {}
    for family in FAMILY_ORDER:
        sub = df[df["family"] == family]
        if family == "classical":
            order = ["logistic_regression", "svm_rbf", "xgboost"]
            for model in order:
                match = sub[sub["model"] == model]
                if len(match):
                    seen[row_key(family, "", model)] = None
        else:
            for mode in FINE_TUNE_ORDER:
                match = sub[sub["mode"] == mode]
                if len(match):
                    seen[row_key(family, mode, match["model"].iloc[0])] = None
    return list(seen.keys())


def heatmap_figure(by_dataset: dict[str, pd.DataFrame], out_path: Path):
    datasets = list(by_dataset.keys())
    # Union of every row seen across datasets, in a fixed family/mode order.
    row_order = []
    for dataset in datasets:
        for key in all_rows(by_dataset[dataset]):
            if key not in row_order:
                row_order.append(key)

    metrics = [("balanced_accuracy", "Balanced accuracy"), ("roc_auc_display", "ROC-AUC (macro OVR for MTBLS563)")]
    col_width = max(1.1, 0.9 * len(datasets) / 3)
    fig, axes = plt.subplots(
        1, len(metrics),
        figsize=(col_width * len(datasets) * len(metrics) + 2, 0.42 * len(row_order) + 2.4),
        dpi=300,
    )

    for ax, (metric, metric_label) in zip(axes, metrics):
        matrix = np.full((len(row_order), len(datasets)), np.nan)
        flags = np.zeros((len(row_order), len(datasets)), dtype=bool)
        for r, (family, mode, _) in enumerate(row_order):
            for c, dataset in enumerate(datasets):
                df = by_dataset[dataset]
                sub = df[df["family"] == family]
                sub = sub[sub["model"] == mode] if family == "classical" else sub[sub["mode"] == mode]
                if len(sub):
                    matrix[r, c] = float(sub[metric].iloc[0])
                    flags[r, c] = bool(sub["degenerate"].iloc[0])

        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for r in range(len(row_order)):
            for c in range(len(datasets)):
                value = matrix[r, c]
                if not np.isfinite(value):
                    ax.text(c, r, "--", ha="center", va="center", fontsize=8, color="#898781")
                    continue
                flagged = bool(flags[r, c])
                text_color = "#0b0b0b" if value < 0.6 else "#ffffff"
                label = f"{value:.2f}" + ("*" if flagged else "")
                ax.text(c, r, label, ha="center", va="center", fontsize=8, color=text_color, fontweight="bold" if flagged else "normal")

        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets, rotation=30, ha="right")
        ax.set_yticks(range(len(row_order)))
        ax.set_yticklabels([label for _, _, label in row_order], fontsize=8)
        ax.set_title(metric_label, fontsize=11, loc="left", color="#0b0b0b")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(datasets), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(row_order), 1), minor=True)
        ax.grid(which="minor", color="#fcfcfb", linewidth=2)
        ax.tick_params(which="minor", length=0)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    fig.suptitle("All evaluated models across datasets", fontsize=13, x=0.02, ha="left", y=1.0)
    fig.text(0.02, -0.01, "* model collapsed to predicting a single class (tp==0 or tn==0 in its confusion matrix).", fontsize=7.5, color="#898781")
    fig.tight_layout(rect=(0, 0.01, 1, 0.98))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def finetune_depth_figure(df_by_dataset: dict[str, pd.DataFrame], out_path: Path):
    datasets = list(df_by_dataset.keys())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.2), dpi=300, sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        df = df_by_dataset[dataset]
        for family in FAMILY_ORDER:
            sub = df[df["family"] == family]
            if family == "classical":
                if len(sub):
                    ax.axhline(
                        float(sub["balanced_accuracy"].max()), color=FAMILY_COLORS[family],
                        linewidth=2, linestyle=":", label=FAMILY_LABELS[family], zorder=2,
                    )
                continue
            sub = sub.set_index("mode").reindex(FINE_TUNE_ORDER)
            if sub["balanced_accuracy"].isna().all():
                continue
            style = dict(color=FAMILY_COLORS[family], marker="o", markersize=5, linewidth=2, zorder=3)
            if sub["degenerate"].fillna(False).any():
                style.update(linestyle="--", alpha=0.55)
            ax.plot(range(len(FINE_TUNE_ORDER)), sub["balanced_accuracy"].to_numpy(), label=FAMILY_LABELS[family], **style)

        ax.set_xticks(range(len(FINE_TUNE_ORDER)))
        ax.set_xticklabels(["frozen", "+1 layer", "+2 layers", "+3 layers"], rotation=20, ha="right")
        ax.axhline(0.5, color="#898781", linewidth=1, linestyle="--", zorder=1)
        ax.set_title(dataset, fontsize=11, loc="left", color="#0b0b0b")
        ax.set_ylim(0, 1.05)
        style_axes(ax)

    axes[0].set_ylabel("Balanced accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Balanced accuracy vs. backbone fine-tuning depth", fontsize=12, y=1.14, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 1.0))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--barth", default="results/loocv/barth_all_models_v4/summary.csv")
    parser.add_argument("--mtbls326-masked", default="results/loocv/mtbls326_masking_v4/summary.csv")
    parser.add_argument("--mtbls326-jigsaw", default="results/loocv/mtbls326_jigsaw_v4/summary.csv")
    parser.add_argument("--mtbls326-joint", default="results/loocv/mtbls326_joint_ssl_v4/summary.csv")
    parser.add_argument("--mtbls563", default="results/loocv/mtbls563_all_models_v4/summary.csv")
    parser.add_argument("--brc-t2d-cancer", default="results/cv10/brc_t2d_newlabels_v4/cancer_status/summary.csv")
    parser.add_argument("--brc-t2d-cancer-joint", default="results/cv10/brc_t2d_newlabels_v4_joint/cancer_status/summary.csv")
    parser.add_argument("--brc-t2d-diabetes", default="results/cv10/brc_t2d_newlabels_v4/diabetes_status/summary.csv")
    parser.add_argument("--brc-t2d-diabetes-joint", default="results/cv10/brc_t2d_newlabels_v4_joint/diabetes_status/summary.csv")
    parser.add_argument("--output-dir", default="results/plots/all_datasets_summary_v4")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    barth = load_dataset("Barth", [(args.barth, "roc_auc", {"masking": "masked"})])
    mtbls326 = load_dataset(
        "MTBLS326",
        [
            (args.mtbls326_masked, "roc_auc", {"foundation": "masked", "classical": "classical"}),
            (args.mtbls326_jigsaw, "roc_auc", {}),
            (args.mtbls326_joint, "roc_auc", {}),
        ],
    )
    # mtbls326_masked's summary.csv also contains its own classical rows; drop the
    # duplicates already carried by jigsaw/joint's separate classical-free runs.
    mtbls326 = mtbls326.drop_duplicates(subset=["family", "model"])
    mtbls563 = load_dataset("MTBLS563", [(args.mtbls563, "macro_roc_auc_ovr", {"masking": "masked"})])

    brc_cancer = load_dataset(
        "BrC-T2D (cancer)",
        [
            (args.brc_t2d_cancer, "roc_auc", {"masking": "masked"}),
            (args.brc_t2d_cancer_joint, "roc_auc", {}),
        ],
    ).drop_duplicates(subset=["family", "model"])
    brc_diabetes = load_dataset(
        "BrC-T2D (diabetes)",
        [
            (args.brc_t2d_diabetes, "roc_auc", {"masking": "masked"}),
            (args.brc_t2d_diabetes_joint, "roc_auc", {}),
        ],
    ).drop_duplicates(subset=["family", "model"])

    by_dataset = {
        "Barth": barth, "MTBLS326": mtbls326, "MTBLS563": mtbls563,
        "BrC-T2D (cancer)": brc_cancer, "BrC-T2D (diabetes)": brc_diabetes,
    }
    best_by_dataset = {name: best_per_family(df) for name, df in by_dataset.items()}

    grouped_bar_figure(
        best_by_dataset, "balanced_accuracy", "Balanced accuracy",
        "Balanced accuracy by model family (best fine-tune mode), across datasets",
        out_dir / "fig1_balanced_accuracy.png",
    )
    grouped_bar_figure(
        best_by_dataset, "roc_auc_display", "ROC-AUC (macro OVR for MTBLS563)",
        "ROC-AUC by model family (best fine-tune mode), across datasets",
        out_dir / "fig2_roc_auc.png",
    )
    finetune_depth_figure(by_dataset, out_dir / "fig3_finetune_depth.png")
    heatmap_figure(by_dataset, out_dir / "fig4_heatmap_all_models.png")


if __name__ == "__main__":
    main()
