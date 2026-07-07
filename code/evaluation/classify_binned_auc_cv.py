"""
Classify Day-1 vs Day-4 using handcrafted binned spectral features (no DNN).

Feature extraction:
- Split each spectrum into equal bins (e.g., 128 or 256)
- Compute absolute area under the curve (AUC) in each bin w.r.t. zero baseline

Models:
- Logistic Regression
- SVM (RBF)
- XGBoost (if installed)

Evaluation:
- 5-fold stratified cross-validation
- OOF metrics, per-model plots, per-bin summaries, and comparison plots
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_SEED = 42


def str2bool(v):
    if isinstance(v, bool):
        return v
    val = str(v).strip().lower()
    if val in {"1", "true", "t", "yes", "y"}:
        return True
    if val in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_bins_list(text):
    vals = [x.strip() for x in str(text).split(",") if x.strip()]
    bins = []
    for v in vals:
        b = int(v)
        if b <= 1:
            raise ValueError("Each bin count must be > 1.")
        bins.append(b)
    if not bins:
        raise ValueError("No valid bins provided.")
    return bins


def map_day_binary(row):
    day_label = str(row.get("day_label", "")).strip().lower()
    day_code = str(row.get("day_code", "")).strip().upper()
    if "1" in day_label or day_label == "1st day" or day_code == "A":
        return 0
    if "4" in day_label or day_label == "4th day" or day_code == "B":
        return 1
    return np.nan


def baseline_center_known_zero_window(spectra, start_idx=62501, end_idx=68000):
    centered = spectra.astype(np.float32, copy=True)
    n_rows, n_cols = centered.shape
    s = max(0, min(int(start_idx), n_cols))
    e = max(0, min(int(end_idx), n_cols))
    if s >= e:
        return centered
    offsets = np.median(centered[:, s:e], axis=1)
    centered -= offsets[:, None]
    return centered


def load_labeled_subset(data_path, labels_csv, baseline_center):
    spectra = np.load(data_path).astype(np.float32)
    labels = pd.read_csv(labels_csv)
    if "spectrum_index" not in labels.columns:
        raise ValueError("data/tbi_tirupati/title_labels.csv must contain 'spectrum_index'.")

    labels = labels.copy()
    labels["target"] = labels.apply(map_day_binary, axis=1)
    labels = labels.dropna(subset=["target"])
    labels["target"] = labels["target"].astype(int)
    labels["spectrum_index"] = labels["spectrum_index"].astype(int)
    labels = labels[(labels["spectrum_index"] >= 0) & (labels["spectrum_index"] < spectra.shape[0])]
    labels = labels.sort_values("spectrum_index").reset_index(drop=True)

    idx = labels["spectrum_index"].to_numpy(dtype=int)
    x = spectra[idx]
    y = labels["target"].to_numpy(dtype=int)
    if baseline_center:
        x = baseline_center_known_zero_window(x)
    return x, y, labels


def extract_abs_auc_features(spectra, n_bins):
    n_samples, n_points = spectra.shape
    edges = np.linspace(0, n_points, n_bins + 1, dtype=int)
    feats = np.zeros((n_samples, n_bins), dtype=np.float32)

    for b in range(n_bins):
        s = edges[b]
        e = edges[b + 1]
        if e <= s:
            continue
        seg = np.abs(spectra[:, s:e])
        if seg.shape[1] > 1:
            area = np.trapz(seg, dx=1.0, axis=1)
        else:
            area = seg[:, 0]
        feats[:, b] = area.astype(np.float32)
    return feats


def score_vector(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    return model.predict(x)


def evaluate_model_cv(model, x, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_rows = []
    oof_pred = np.zeros_like(y)
    oof_score = np.zeros_like(y, dtype=np.float64)

    for fold, (tr, te) in enumerate(skf.split(x, y), start=1):
        clf = clone(model)
        clf.fit(x[tr], y[tr])
        pred = clf.predict(x[te])
        score = score_vector(clf, x[te])

        oof_pred[te] = pred
        oof_score[te] = score
        fold_rows.append(
            {
                "fold": fold,
                "accuracy": accuracy_score(y[te], pred),
                "precision": precision_score(y[te], pred, zero_division=0),
                "recall": recall_score(y[te], pred, zero_division=0),
                "f1": f1_score(y[te], pred, zero_division=0),
                "roc_auc": roc_auc_score(y[te], score),
                "pr_auc": average_precision_score(y[te], score),
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    summary = {}
    for m in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        summary[f"{m}_mean"] = float(fold_df[m].mean())
        summary[f"{m}_std"] = float(fold_df[m].std(ddof=1))
    return fold_df, summary, oof_pred, oof_score


def plot_model_performance(name, y_true, y_pred, y_score, fold_df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
    axes[0].set_title(f"{name} - Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_xticklabels(["Day 1", "Day 4"])
    axes[0].set_yticklabels(["Day 1", "Day 4"], rotation=0)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[1].set_title(f"{name} - ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.3)

    prec, rec, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    axes[2].plot(rec, prec, label=f"AP = {pr_auc:.3f}")
    axes[2].set_title(f"{name} - Precision-Recall")
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].legend(loc="lower left")
    axes[2].grid(alpha=0.3)

    m = fold_df.mean(numeric_only=True)
    s = fold_df.std(numeric_only=True, ddof=1)
    fig.suptitle(
        f"{name} | "
        f"Acc {m['accuracy']:.3f}±{s['accuracy']:.3f}, "
        f"F1 {m['f1']:.3f}±{s['f1']:.3f}, "
        f"ROC-AUC {m['roc_auc']:.3f}±{s['roc_auc']:.3f}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bin_level_comparison(summary_df, out_path):
    # Compare models within each bin count using ROC-AUC mean.
    pivot = summary_df.pivot(index="model", columns="bins", values="roc_auc_mean")
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", vmin=0.0, vmax=1.0)
    plt.title("ROC-AUC Mean (Model x Bin Count)")
    plt.xlabel("Bins")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Binned-abs-AUC classification (no DNN backbone).")
    parser.add_argument("--data-path", default="data/tbi_tirupati/aligned_128K_TBI_Tirupati_WS625to680Zero.npy")
    parser.add_argument("--labels-csv", default="data/tbi_tirupati/title_labels.csv")
    parser.add_argument("--output-dir", default="results/classification/binned_auc")
    parser.add_argument("--bins", default="128,256", help="Comma-separated bin counts (e.g., 128,256)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--baseline-center", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bin_list = parse_bins_list(args.bins)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    x_raw, y, labels_df = load_labeled_subset(
        data_path=args.data_path,
        labels_csv=args.labels_csv,
        baseline_center=args.baseline_center,
    )
    print(f"Loaded labeled subset: {x_raw.shape[0]} samples, {x_raw.shape[1]} points")
    print(f"Class counts -> Day1: {(y == 0).sum()}, Day4: {(y == 1).sum()}")

    models = {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, random_state=args.seed)),
            ]
        ),
        "SVM_RBF": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=args.seed)),
            ]
        ),
    }

    xgb_available = False
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=args.seed,
        )
        xgb_available = True
    except Exception:
        print("XGBoost not available. Skipping XGBoost model.")
        print("Install with: pip install xgboost")

    summary_rows = []
    all_fold_rows = []

    for n_bins in bin_list:
        print(f"\nExtracting features for {n_bins} bins...")
        x_feat = extract_abs_auc_features(x_raw, n_bins=n_bins)
        np.save(out_root / f"features_{n_bins}bins.npy", x_feat)

        pred_table = labels_df.copy()
        pred_table["sample_title"] = pred_table.get("title", "").astype(str)
        pred_table["original_label"] = np.where(y == 0, "Day1", "Day4")

        for model_name, model in models.items():
            print(f"Evaluating {model_name} ({n_bins} bins)...")
            fold_df, summary, y_pred, y_score = evaluate_model_cv(
                model=model,
                x=x_feat,
                y=y,
                n_splits=args.folds,
                seed=args.seed,
            )

            fold_df.insert(0, "bins", n_bins)
            fold_df.insert(1, "model", model_name)
            fold_df.to_csv(out_root / f"{model_name}_{n_bins}bins_fold_metrics.csv", index=False)
            all_fold_rows.append(fold_df)

            row = {"bins": n_bins, "model": model_name}
            row.update(summary)
            summary_rows.append(row)

            np.save(out_root / f"{model_name}_{n_bins}bins_oof_pred.npy", y_pred)
            np.save(out_root / f"{model_name}_{n_bins}bins_oof_score.npy", y_score)

            plot_model_performance(
                name=f"{model_name} ({n_bins} bins)",
                y_true=y,
                y_pred=y_pred,
                y_score=y_score,
                fold_df=fold_df,
                out_path=out_root / f"{model_name}_{n_bins}bins_performance.png",
            )

            pred_table[f"pred_{model_name}"] = np.where(y_pred == 0, "Day1", "Day4")
            pred_table[f"score_{model_name}"] = y_score

        ordered_cols = ["spectrum_index", "sample_title", "original_label"] + [
            c for c in pred_table.columns if c.startswith("pred_")
        ] + [c for c in pred_table.columns if c.startswith("score_")]
        pred_table[ordered_cols].to_csv(out_root / f"per_sample_predictions_{n_bins}bins.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_root / "model_summary_by_bins.csv", index=False)

    all_fold_df = pd.concat(all_fold_rows, ignore_index=True)
    all_fold_df.to_csv(out_root / "all_models_all_bins_fold_metrics.csv", index=False)

    plot_bin_level_comparison(summary_df, out_root / "roc_auc_model_bin_heatmap.png")

    run_info = {
        "data_path": args.data_path,
        "labels_csv": args.labels_csv,
        "output_dir": str(out_root),
        "n_samples_used": int(len(y)),
        "class_counts": {"Day1": int((y == 0).sum()), "Day4": int((y == 1).sum())},
        "bins": bin_list,
        "folds": int(args.folds),
        "baseline_center": bool(args.baseline_center),
        "seed": int(args.seed),
        "xgboost_available": bool(xgb_available),
        "feature_definition": "Absolute AUC per equal-width bin (trapz over abs(signal), baseline=0)",
    }
    with open(out_root / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print("\nDone.")
    print(f"Saved summary: {out_root / 'model_summary_by_bins.csv'}")
    print(f"Saved comparison heatmap: {out_root / 'roc_auc_model_bin_heatmap.png'}")


if __name__ == "__main__":
    main()
