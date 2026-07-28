#!/usr/bin/env python3
"""Per-fold balanced-accuracy variability for BrC-T2D 10-fold CV.

The pooled (OOF) summary.csv numbers used elsewhere collapse each model down
to a single point estimate. For a 10-fold CV run we have per-fold metrics
too, which is the only way to tell whether one family's edge over another is
a real effect or within the run's own fold-to-fold noise.

classical/masking/jigsaw (brc_t2d_10fold_cv.py) write fold_metrics.csv
directly. joint_ssl (brc_t2d_joint_ssl_cv10.py) does not -- it only writes
pooled OOF predictions -- so its per-fold balanced accuracy is reconstructed
here from oof_predictions.csv using the fold assignment recorded in the
sibling classical/masking/jigsaw run's fold_indices.json. This is only valid
because both scripts build folds with StratifiedKFold(n_splits=10,
shuffle=True, random_state=seed) over the *same* labels array from the same
data/metadata/seed (verified against both runs' run_config.json before
relying on this) -- so the split is deterministically identical.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

FAMILY_COLORS = {
    "classical": "#2a78d6",
    "masking": "#1baf7a",
    "jigsaw": "#eda100",
    "joint_ssl": "#e34948",
}
FAMILY_LABELS = {
    "classical": "Classical",
    "masking": "Masked SSL",
    "jigsaw": "Jigsaw SSL",
    "joint_ssl": "Joint SSL",
}
FAMILY_ORDER = ["classical", "masking", "jigsaw", "joint_ssl"]


def best_model_per_family(summary: pd.DataFrame) -> dict[str, str]:
    best = {}
    for family in summary["family"].unique():
        sub = summary[summary["family"] == family]
        best[family] = sub.loc[sub["balanced_accuracy"].idxmax(), "model"]
    return best


def load_fold_metrics(fold_metrics_csv: Path, best_model: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(fold_metrics_csv)
    rows = []
    for family, model in best_model.items():
        if family not in df["family"].unique():
            continue
        sub = df[(df["family"] == family) & (df["model"] == model)]
        if len(sub):
            rows.append(sub[["family", "fold", "balanced_accuracy"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["family", "fold", "balanced_accuracy"])


def reconstruct_joint_fold_metrics(oof_csv: Path, fold_indices_json: Path, best_model: str) -> pd.DataFrame:
    oof = pd.read_csv(oof_csv)
    folds = json.load(open(fold_indices_json))
    pred_col = f"joint_ssl_{best_model}_prediction"
    if pred_col not in oof.columns:
        raise KeyError(f"{pred_col!r} not in {oof_csv}; available: {[c for c in oof.columns if 'prediction' in c]}")
    rows = []
    for fold in folds:
        idx = fold["test_idx"]
        sub = oof[oof["npy_row"].isin(idx)]
        bal_acc = balanced_accuracy_score(sub["label"], sub[pred_col])
        rows.append({"family": "joint_ssl", "fold": fold["fold"], "balanced_accuracy": bal_acc})
    return pd.DataFrame(rows)


def strip_mode(model: str) -> str:
    for mode in ["frozen", "unfreeze_last_1", "unfreeze_last_2", "unfreeze_last_3"]:
        if model.endswith(mode):
            return {"frozen": "frozen", "unfreeze_last_1": "+1 layer", "unfreeze_last_2": "+2 layers", "unfreeze_last_3": "+3 layers"}[mode]
    return model


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.yaxis.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_label(ax, per_family: pd.DataFrame, best_model: dict[str, str], title: str):
    positions, data, colors, labels = [], [], [], []
    for i, family in enumerate(FAMILY_ORDER):
        sub = per_family[per_family["family"] == family]["balanced_accuracy"].to_numpy()
        if sub.size == 0:
            continue
        positions.append(i)
        data.append(sub)
        colors.append(FAMILY_COLORS[family])
        labels.append(f"{FAMILY_LABELS[family]}\n({strip_mode(best_model[family])})")

    bp = ax.boxplot(
        data, positions=positions, widths=0.55, patch_artist=True,
        showfliers=False, medianprops=dict(color="#0b0b0b", linewidth=1.5),
        whiskerprops=dict(color="#52514e"), capprops=dict(color="#52514e"),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
    rng = np.random.default_rng(0)
    for pos, sub, color in zip(positions, data, colors):
        jitter = rng.uniform(-0.12, 0.12, size=sub.size)
        ax.scatter(np.full(sub.size, pos) + jitter, sub, color=color, s=22, zorder=3, edgecolor="white", linewidth=0.5)

    ax.axhline(0.5, color="#898781", linewidth=1, linestyle="--", zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=11, loc="left", color="#0b0b0b")
    style_axes(ax)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cancer-dir", default="results/cv10/brc_t2d_newlabels_v4/cancer_status")
    parser.add_argument("--cancer-joint-dir", default="results/cv10/brc_t2d_newlabels_v4_joint/cancer_status")
    parser.add_argument("--diabetes-dir", default="results/cv10/brc_t2d_newlabels_v4/diabetes_status")
    parser.add_argument("--diabetes-joint-dir", default="results/cv10/brc_t2d_newlabels_v4_joint/diabetes_status")
    parser.add_argument("--output", default="results/plots/all_datasets_summary_v4/fig5_brc_t2d_fold_variability.png")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=300, sharey=True)

    for ax, title, main_dir, joint_dir in [
        (axes[0], "BrC-T2D cancer_status (10-fold)", args.cancer_dir, args.cancer_joint_dir),
        (axes[1], "BrC-T2D diabetes_status (10-fold)", args.diabetes_dir, args.diabetes_joint_dir),
    ]:
        main_dir, joint_dir = Path(main_dir), Path(joint_dir)
        summary = pd.read_csv(main_dir / "summary.csv")
        joint_summary = pd.read_csv(joint_dir / "summary.csv")
        best = best_model_per_family(summary)
        best["joint_ssl"] = best_model_per_family(joint_summary)["joint_ssl"]

        per_family = load_fold_metrics(main_dir / "fold_metrics.csv", best)
        joint_fold = reconstruct_joint_fold_metrics(
            joint_dir / "oof_predictions.csv", main_dir / "fold_indices.json", best["joint_ssl"],
        )
        per_family = pd.concat([per_family, joint_fold], ignore_index=True)
        plot_label(ax, per_family, best, title)

    axes[0].set_ylabel("Balanced accuracy (per fold)")
    handles = [plt.Line2D([0], [0], color=FAMILY_COLORS[f], marker="o", linestyle="", markersize=6) for f in FAMILY_ORDER]
    fig.legend(handles, [FAMILY_LABELS[f] for f in FAMILY_ORDER], loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=4, frameon=False, fontsize=9)
    fig.suptitle(
        "Fold-to-fold spread of the best fine-tune mode per family (n=10 folds, ~8 test samples/fold)",
        fontsize=11, y=1.16, x=0.02, ha="left",
    )
    fig.text(0.01, -0.02, "Box: median/IQR across the 10 folds. Dots: individual fold balanced accuracy. Dashed line: chance level (0.5).", fontsize=7.5, color="#898781")
    fig.tight_layout(rect=(0, 0, 1, 1.0))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
