"""Compare Xavier-reinitialized SSL backbones against the original pretrained
backbones on the LOOCV benchmarks.

For each of the three test datasets (Barth, MTBLS326, MTBLS563) this script
pairs up the "metabolights_v1" (originally pretrained) and
"metabolights_v1_xavier" (backbone re-initialized with Xavier/Glorot instead
of loading pretrained SSL weights, then trained identically) LOOCV runs,
matches rows on (family, model), and plots balanced_accuracy side by side.

Balanced accuracy is used as the primary comparison metric because it is the
only metric present in every summary.csv across both the binary (Barth,
MTBLS326) and multiclass (MTBLS563) result files.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#2a78d6"    # baseline (pretrained SSL)
ORANGE = "#eb6834"  # xavier (random re-init)
GRID = "#d8d7d2"
TEXT = "#52514e"

RESULTS_DIR = Path("results/loocv")
OUT_DIR = Path("results/analysis/xavier_ablation")

# (label, baseline_dir, xavier_dir)
DATASET_PAIRS = [
    ("Barth", "barth_all_models_metabolights_v1", "barth_all_models_metabolights_v1_xavier"),
    ("MTBLS326 (jigsaw)", "mtbls326_jigsaw_metabolights_v1", "mtbls326_jigsaw_metabolights_v1_xavier"),
    ("MTBLS326 (joint_ssl)", "mtbls326_joint_ssl_metabolights_v1", "mtbls326_joint_ssl_metabolights_v1_xavier"),
    ("MTBLS326 (masking)", "mtbls326_masked_metabolights_v1", "mtbls326_masked_metabolights_v1_xavier"),
    ("MTBLS563", "mtbls563_all_models_metabolights_v1", "mtbls563_all_models_metabolights_v1_xavier"),
]


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def load_pair(label, base_name, xav_name):
    base = pd.read_csv(RESULTS_DIR / base_name / "summary.csv")
    xav = pd.read_csv(RESULTS_DIR / xav_name / "summary.csv")
    merged = base.merge(
        xav, on=["family", "model"], suffixes=("_pretrained", "_xavier")
    )
    merged.insert(0, "dataset", label)
    merged["delta_balanced_accuracy"] = (
        merged["balanced_accuracy_xavier"] - merged["balanced_accuracy_pretrained"]
    )
    return merged


def plot_dataset_comparison(label, df, out_path):
    df = df.sort_values(["family", "model"]).reset_index(drop=True)
    n = len(df)
    x = np.arange(n)
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(6, n * 0.9), 4.5))
    ax.bar(x - width / 2, df["balanced_accuracy_pretrained"], width, label="SSL-pretrained init", color=BLUE, zorder=3)
    ax.bar(x + width / 2, df["balanced_accuracy_xavier"], width, label="Xavier (random) init", color=ORANGE, zorder=3)

    labels = [f"{f}/{m}" for f, m in zip(df["family"], df["model"])]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, color=TEXT, rotation=35, ha="right")
    ax.set_ylabel("Balanced accuracy", color=TEXT)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{label}: pretrained vs. Xavier-reinit backbone (LOOCV)", color=TEXT, fontsize=11)
    style_ax(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.subplots_adjust(bottom=0.32, top=0.90, left=0.08, right=0.98)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_delta_summary(all_df, out_path):
    df = all_df.copy()
    df["group"] = df["dataset"] + " | " + df["family"] + "/" + df["model"]
    df = df.sort_values("delta_balanced_accuracy")

    colors = [ORANGE if v >= 0 else BLUE for v in df["delta_balanced_accuracy"]]
    fig, ax = plt.subplots(figsize=(11, max(6, len(df) * 0.28)))
    y = np.arange(len(df))
    ax.barh(y, df["delta_balanced_accuracy"], color=colors, zorder=3)
    ax.axvline(0, color=TEXT, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["group"], fontsize=7)
    ax.set_xlabel("Δ balanced accuracy (Xavier − pretrained)", color=TEXT)
    ax.set_title(
        "Effect of discarding SSL pretraining (Xavier reinit) per model/dataset",
        color=TEXT, fontsize=11,
    )
    style_ax(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.yaxis.grid(False)
    fig.subplots_adjust(left=0.42, right=0.97, top=0.95, bottom=0.08)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_family_summary(all_df, out_path):
    """Average delta per (dataset, family), collapsing across fine-tune modes."""
    fam_map = {
        "foundation": "masking",
    }
    df = all_df.copy()
    df["family_norm"] = df["family"].replace(fam_map)
    grp = (
        df.groupby(["dataset", "family_norm"])["delta_balanced_accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    datasets = grp["dataset"].unique().tolist()
    families = sorted(grp["family_norm"].unique().tolist())
    palette = {"classical": "#1baf7a", "jigsaw": "#eda100", "joint_ssl": "#4a3aa7", "masking": "#e34948"}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / max(len(families), 1)
    for i, fam in enumerate(families):
        sub = grp[grp["family_norm"] == fam]
        xs = [datasets.index(d) + (i - len(families) / 2) * width + width / 2 for d in sub["dataset"]]
        ax.bar(xs, sub["mean"], width=width * 0.9, label=fam, color=palette.get(fam, "#999999"), zorder=3)
    ax.axhline(0, color=TEXT, linewidth=1)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, fontsize=9)
    ax.set_ylabel("Mean Δ balanced accuracy (Xavier − pretrained)", color=TEXT)
    ax.set_title("Average effect of dropping SSL pretraining, by model family", color=TEXT, fontsize=11)
    style_ax(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for label, base_name, xav_name in DATASET_PAIRS:
        base_path = RESULTS_DIR / base_name / "summary.csv"
        xav_path = RESULTS_DIR / xav_name / "summary.csv"
        if not base_path.exists() or not xav_path.exists():
            print(f"Skipping {label}: missing {base_path if not base_path.exists() else xav_path}")
            continue
        merged = load_pair(label, base_name, xav_name)
        all_rows.append(merged)
        safe_label = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plot_dataset_comparison(label, merged, out_dir / f"xavier_vs_pretrained_{safe_label}.png")
        print(f"{label}: {len(merged)} model rows compared")

    all_df = pd.concat(all_rows, ignore_index=True)
    keep_cols = [
        "dataset", "family", "model",
        "balanced_accuracy_pretrained", "balanced_accuracy_xavier", "delta_balanced_accuracy",
        "accuracy_pretrained", "accuracy_xavier",
    ]
    all_df[keep_cols].to_csv(out_dir / "xavier_ablation_comparison.csv", index=False)

    plot_delta_summary(all_df, out_dir / "xavier_delta_per_model.png")
    plot_family_summary(all_df, out_dir / "xavier_delta_by_family.png")

    summary = {
        "n_model_rows_compared": int(len(all_df)),
        "mean_delta_balanced_accuracy": float(all_df["delta_balanced_accuracy"].mean()),
        "median_delta_balanced_accuracy": float(all_df["delta_balanced_accuracy"].median()),
        "n_xavier_better": int((all_df["delta_balanced_accuracy"] > 0).sum()),
        "n_pretrained_better": int((all_df["delta_balanced_accuracy"] < 0).sum()),
        "n_tied": int((all_df["delta_balanced_accuracy"] == 0).sum()),
        "worst_regression": {
            "row": all_df.loc[all_df["delta_balanced_accuracy"].idxmin(), ["dataset", "family", "model"]].to_dict(),
            "delta": float(all_df["delta_balanced_accuracy"].min()),
        },
        "best_xavier_gain": {
            "row": all_df.loc[all_df["delta_balanced_accuracy"].idxmax(), ["dataset", "family", "model"]].to_dict(),
            "delta": float(all_df["delta_balanced_accuracy"].max()),
        },
    }
    with open(out_dir / "xavier_ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote outputs to {out_dir}/")


if __name__ == "__main__":
    main()
