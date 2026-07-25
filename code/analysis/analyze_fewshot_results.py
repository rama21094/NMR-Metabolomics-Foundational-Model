"""Analyze the structured few-shot benchmark (fewshot_benchmark.py output)
across all 3 test datasets (Barth, MTBLS326, MTBLS563) and all 6 models
(3 classical ML + 3 SSL backbones, each with frozen + 3 unfreeze depths).

For each dataset produces:
  - a small-multiples figure: one panel per model family, learning curves
    (balanced accuracy vs. support-per-class) with +/-1 std shaded bands,
    one line per fine-tune mode (classical: per model; SSL: frozen vs.
    unfreeze_last_1/2/3).
  - a head-to-head summary figure: best classical model vs. each SSL
    family's frozen vs. its best unfrozen (fine-tuned) mode.

Also writes a single cross-dataset CSV of the frozen-vs-best-finetuned
balanced-accuracy gain, averaged over all support levels, to quantify
whether unfreezing/fine-tuning helps in the few-shot regime.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
VIOLET = "#4a3aa7"
RED = "#e34948"
GRID = "#d8d7d2"
TEXT = "#52514e"

FEWSHOT_DIR = Path("results/fewshot")
OUT_DIR = Path("results/analysis/fewshot_analysis")

DATASETS = [
    ("Barth", "barth_v1", "barth_v1_jigsaw", "barth_v1_joint", "barth_v1_masking"),
    ("MTBLS326", "mtbls326_v1", "mtbls326_v1_jigsaw", "mtbls326_v1_joint", "mtbls326_v1_masking"),
    ("MTBLS563 (multiclass)", "mtbls563_v1", "mtbls563_v1_jigsaw", "mtbls563_v1_joint", "mtbls563_v1_masking"),
]

CLASSICAL_COLORS = {"logistic_regression": BLUE, "svm_rbf": AQUA, "xgboost": VIOLET}
MODE_COLORS = {"frozen": BLUE, "unfreeze_last_1": YELLOW, "unfreeze_last_2": ORANGE, "unfreeze_last_3": RED}
MODE_ORDER = ["frozen", "unfreeze_last_1", "unfreeze_last_2", "unfreeze_last_3"]


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_family_panel(ax, df, group_col, color_map, order=None, metric="balanced_accuracy"):
    keys = order if order is not None else sorted(df[group_col].unique())
    for key in keys:
        sub = df[df[group_col] == key].sort_values("support_per_class")
        if sub.empty:
            continue
        x = sub["support_per_class"].values
        mean = sub[f"{metric}_mean"].values
        std = sub[f"{metric}_std"].values
        color = color_map.get(key, "#999999")
        ax.plot(x, mean, marker="o", markersize=3, linewidth=1.8, color=color, label=key, zorder=4)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, zorder=2, linewidth=0)
    ax.axhline(0.5 if metric != "macro_f1" else None, color=GRID, linewidth=0) if False else None
    style_ax(ax)


def load_dataset_frames(classical_dir, jigsaw_dir, joint_dir, masking_dir):
    classical = pd.read_csv(FEWSHOT_DIR / classical_dir / "fewshot_summary.csv")
    jigsaw = pd.read_csv(FEWSHOT_DIR / jigsaw_dir / "fewshot_summary.csv")
    joint = pd.read_csv(FEWSHOT_DIR / joint_dir / "fewshot_summary.csv")
    masking = pd.read_csv(FEWSHOT_DIR / masking_dir / "fewshot_summary.csv")
    return classical, jigsaw, joint, masking


def small_multiples_figure(label, classical, jigsaw, joint, masking, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    panels = [
        ("Classical ML", classical, "model", CLASSICAL_COLORS, sorted(CLASSICAL_COLORS.keys())),
        ("Jigsaw SSL", jigsaw, "fine_tune_mode", MODE_COLORS, MODE_ORDER),
        ("Joint SSL", joint, "fine_tune_mode", MODE_COLORS, MODE_ORDER),
        ("Masking SSL", masking, "fine_tune_mode", MODE_COLORS, MODE_ORDER),
    ]
    for ax, (title, df, group_col, cmap, order) in zip(axes.flat, panels):
        plot_family_panel(ax, df, group_col, cmap, order=order)
        ax.set_title(title, fontsize=10, color=TEXT)
        ax.legend(frameon=False, fontsize=7, loc="lower right")
    for ax in axes[-1, :]:
        ax.set_xlabel("Support samples per class", color=TEXT, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Balanced accuracy", color=TEXT, fontsize=9)
    fig.suptitle(f"{label}: few-shot learning curves (mean ± 1 std over 10 episodes)", color=TEXT, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def best_classical_curve(classical):
    """Pick, per support level, the classical model with highest mean balanced accuracy,
    then return that best model's own full curve (the model with the best AUC-over-support)."""
    auc = classical.groupby("model").apply(
        lambda d: np.trapz(d.sort_values("support_per_class")["balanced_accuracy_mean"], d.sort_values("support_per_class")["support_per_class"])
    )
    best_model = auc.idxmax()
    return classical[classical["model"] == best_model], best_model


def head_to_head_figure(label, classical, jigsaw, joint, masking, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    best_classical, best_model = best_classical_curve(classical)
    sub = best_classical.sort_values("support_per_class")
    ax.plot(sub["support_per_class"], sub["balanced_accuracy_mean"], marker="s", linewidth=2,
            color="#999999", linestyle="--", label=f"Best classical ({best_model})", zorder=4)

    ssl_families = [("Jigsaw", jigsaw, VIOLET), ("Joint", joint, AQUA), ("Masking", masking, RED)]
    for fam_label, df, color in ssl_families:
        frozen = df[df["fine_tune_mode"] == "frozen"].sort_values("support_per_class")
        # best unfrozen mode by AUC-over-support among the 3 unfreeze depths
        unfrozen_modes = df[df["fine_tune_mode"] != "frozen"]
        auc = unfrozen_modes.groupby("fine_tune_mode").apply(
            lambda d: np.trapz(d.sort_values("support_per_class")["balanced_accuracy_mean"], d.sort_values("support_per_class")["support_per_class"])
        )
        best_mode = auc.idxmax()
        best_unfrozen = df[df["fine_tune_mode"] == best_mode].sort_values("support_per_class")

        ax.plot(frozen["support_per_class"], frozen["balanced_accuracy_mean"], marker="o", linewidth=1.6,
                linestyle=":", color=color, alpha=0.7, label=f"{fam_label} frozen", zorder=3)
        ax.plot(best_unfrozen["support_per_class"], best_unfrozen["balanced_accuracy_mean"], marker="o", linewidth=2.2,
                color=color, label=f"{fam_label} {best_mode}", zorder=4)

    ax.set_xlabel("Support samples per class", color=TEXT)
    ax.set_ylabel("Balanced accuracy", color=TEXT)
    ax.set_title(f"{label}: best classical vs. SSL (frozen vs. best fine-tuned)", color=TEXT, fontsize=11)
    style_ax(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right", ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def finetune_gain_rows(label, family_label, df):
    frozen = df[df["fine_tune_mode"] == "frozen"].set_index("support_per_class")["balanced_accuracy_mean"]
    rows = []
    for mode in ["unfreeze_last_1", "unfreeze_last_2", "unfreeze_last_3"]:
        sub = df[df["fine_tune_mode"] == mode].set_index("support_per_class")["balanced_accuracy_mean"]
        common = frozen.index.intersection(sub.index)
        if len(common) == 0:
            continue
        gain = (sub.loc[common] - frozen.loc[common])
        rows.append({
            "dataset": label,
            "family": family_label,
            "fine_tune_mode": mode,
            "mean_gain_over_frozen": float(gain.mean()),
            "n_support_levels": int(len(common)),
            "gain_at_min_support": float(gain.loc[common.min()]),
            "gain_at_max_support": float(gain.loc[common.max()]),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gain_rows = []
    for label, classical_dir, jigsaw_dir, joint_dir, masking_dir in DATASETS:
        classical, jigsaw, joint, masking = load_dataset_frames(classical_dir, jigsaw_dir, joint_dir, masking_dir)
        safe = label.split(" ")[0].lower()

        small_multiples_figure(label, classical, jigsaw, joint, masking, out_dir / f"fewshot_learning_curves_{safe}.png")
        head_to_head_figure(label, classical, jigsaw, joint, masking, out_dir / f"fewshot_head_to_head_{safe}.png")

        for fam_label, df in [("jigsaw", jigsaw), ("joint_ssl", joint), ("masking", masking)]:
            gain_rows.extend(finetune_gain_rows(label, fam_label, df))

        print(f"{label}: done")

    gain_df = pd.DataFrame(gain_rows)
    gain_df.to_csv(out_dir / "finetune_gain_over_frozen.csv", index=False)

    summary = {
        "mean_gain_over_frozen_overall": float(gain_df["mean_gain_over_frozen"].mean()),
        "frac_modes_positive_gain": float((gain_df["mean_gain_over_frozen"] > 0).mean()),
        "by_family_mean_gain": gain_df.groupby("family")["mean_gain_over_frozen"].mean().to_dict(),
        "by_dataset_mean_gain": gain_df.groupby("dataset")["mean_gain_over_frozen"].mean().to_dict(),
    }
    with open(out_dir / "finetune_gain_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote outputs to {out_dir}/")


if __name__ == "__main__":
    main()
