"""
Evaluate MAE embeddings for Day-1 vs Day-4 classification with 5-fold CV.

Pipeline:
1) Load spectra (.npy) and labels (data/tbi_tirupati/title_labels.csv)
2) Filter to Day-1 and Day-4 samples
3) Extract one embedding vector per sample from the MAE encoder
4) Train/evaluate classifiers (SVM, Logistic Regression, XGBoost if available)
5) Save metrics and plots per model + cross-model comparison
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
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
from tqdm import tqdm

from trainer_revised import NMRMaskedAutoencoder


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


def parse_normalize_mode(mode):
    m = str(mode).strip().lower()
    if m not in {"auto", "true", "false"}:
        raise ValueError("normalize_input must be one of: auto, true, false")
    return m


def normalize_per_spectrum_minmax(spectra):
    out = np.zeros_like(spectra, dtype=np.float32)
    for i in range(spectra.shape[0]):
        x = spectra[i]
        mn = x.min()
        mx = x.max()
        if mx - mn > 1e-8:
            out[i] = (x - mn) / (mx - mn)
        else:
            out[i] = x
    return out


def is_unit_range(x, tol=1e-6):
    return float(np.min(x)) >= -tol and float(np.max(x)) <= (1.0 + tol)


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


def infer_model_dims(state_dict, default_nhead=4, default_dropout=0.2):
    patch_w = state_dict["encoder.patch_embedding.0.weight"]
    ff_w = state_dict["encoder.transformer.layers.0.linear1.weight"]

    patch_size = int(patch_w.shape[1])
    d_model = int(patch_w.shape[0])
    dim_feedforward = int(ff_w.shape[0])

    layer_ids = []
    prefix = "encoder.transformer.layers."
    for key in state_dict.keys():
        if key.startswith(prefix):
            suffix = key[len(prefix):]
            layer_id = suffix.split(".", 1)[0]
            if layer_id.isdigit():
                layer_ids.append(int(layer_id))
    num_layers = max(layer_ids) + 1 if layer_ids else 3

    return {
        "patch_size": patch_size,
        "d_model": d_model,
        "nhead": default_nhead,
        "num_layers": num_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": default_dropout,
    }


def map_day_binary(row):
    day_label = str(row.get("day_label", "")).strip().lower()
    day_code = str(row.get("day_code", "")).strip().upper()
    if "1" in day_label or day_label == "1st day" or day_code == "A":
        return 0
    if "4" in day_label or day_label == "4th day" or day_code == "B":
        return 1
    return np.nan


def load_labeled_subset(data_path, labels_csv, baseline_center):
    spectra = np.load(data_path).astype(np.float32)
    labels = pd.read_csv(labels_csv)

    if "spectrum_index" not in labels.columns:
        raise ValueError("data/tbi_tirupati/title_labels.csv must contain a 'spectrum_index' column.")

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


def build_model_from_checkpoint(ckpt_path, spectrum_length, device, nhead):
    checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint["model_state_dict"]
    dims = infer_model_dims(state, default_nhead=nhead)

    if dims["d_model"] % dims["nhead"] != 0:
        raise ValueError(
            f"d_model={dims['d_model']} is not divisible by nhead={dims['nhead']}. "
            "Pass a valid --nhead value."
        )

    model = NMRMaskedAutoencoder(
        spectrum_length=spectrum_length,
        patch_size=dims["patch_size"],
        d_model=dims["d_model"],
        nhead=dims["nhead"],
        num_layers=dims["num_layers"],
        dim_feedforward=dims["dim_feedforward"],
        dropout=dims["dropout"],
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, checkpoint, dims


def extract_embeddings(model, spectra, batch_size, device):
    x = torch.from_numpy(spectra).float()
    embs = []
    with torch.no_grad():
        for start in tqdm(range(0, x.shape[0], batch_size), desc="Extracting embeddings"):
            xb = x[start:start + batch_size].to(device)
            _, encoded = model(xb, mask=None)
            pooled = encoded.mean(dim=1)
            embs.append(pooled.cpu().numpy())
    return np.vstack(embs)


def score_vector(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    return model.predict(x)


def evaluate_model_cv(model, x, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics = []
    oof_pred = np.zeros_like(y)
    oof_score = np.zeros_like(y, dtype=np.float64)

    for fold, (tr, te) in enumerate(skf.split(x, y), start=1):
        clf = clone(model)
        clf.fit(x[tr], y[tr])
        pred = clf.predict(x[te])
        score = score_vector(clf, x[te])

        oof_pred[te] = pred
        oof_score[te] = score

        fold_metrics.append(
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

    fold_df = pd.DataFrame(fold_metrics)
    summary = {
        "accuracy_mean": float(fold_df["accuracy"].mean()),
        "accuracy_std": float(fold_df["accuracy"].std(ddof=1)),
        "precision_mean": float(fold_df["precision"].mean()),
        "precision_std": float(fold_df["precision"].std(ddof=1)),
        "recall_mean": float(fold_df["recall"].mean()),
        "recall_std": float(fold_df["recall"].std(ddof=1)),
        "f1_mean": float(fold_df["f1"].mean()),
        "f1_std": float(fold_df["f1"].std(ddof=1)),
        "roc_auc_mean": float(fold_df["roc_auc"].mean()),
        "roc_auc_std": float(fold_df["roc_auc"].std(ddof=1)),
        "pr_auc_mean": float(fold_df["pr_auc"].mean()),
        "pr_auc_std": float(fold_df["pr_auc"].std(ddof=1)),
    }
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


def plot_model_comparison(summary_df, out_path):
    metrics = ["accuracy_mean", "f1_mean", "roc_auc_mean", "pr_auc_mean"]
    metric_labels = ["Accuracy", "F1", "ROC-AUC", "PR-AUC"]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (_, row) in enumerate(summary_df.iterrows()):
        means = [row[m] for m in metrics]
        stds = [row[m.replace("_mean", "_std")] for m in metrics]
        ax.bar(
            x + i * width,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            label=row["model"],
            alpha=0.85,
        )

    ax.set_xticks(x + width * (len(summary_df) - 1) / 2)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Classifier Comparison on MAE Embeddings (5-fold CV)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate MAE embeddings for Day1 vs Day4 classification.")
    parser.add_argument(
        "--model-path",
        default="models/SSL_models/Itr6Rerun_20260106_041031_bs16_mr0.50_ps1024_best.pth",
    )
    # 15 - Itr6Rerun_20251209_035637_bs16_mr0.15_ps1024_20251209_035637_best_300epochs.pth
    # 25 - Itr6Rerun_20251211_084137_bs16_mr0.25_ps1024_best.pth
    # 35 - Itr6Rerun_20251212_103256_bs16_mr0.35_ps1024_best.pth
    # 50 - Itr6Rerun_20260106_041031_bs16_mr0.50_ps1024_best.pth
    parser.add_argument("--data-path", default="data/tbi_tirupati/aligned_128K_TBI_Tirupati_WS625to680Zero.npy")
    parser.add_argument("--labels-csv", default="data/tbi_tirupati/title_labels.csv")
    parser.add_argument("--output-dir", default="results/classification/embeddings")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--normalize-input", default="auto", help="auto|true|false")
    parser.add_argument("--baseline-center", type=str2bool, default=True)
    parser.add_argument("--nhead", type=int, default=4)
    args = parser.parse_args()

    normalize_mode = parse_normalize_mode(args.normalize_input)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    model_name = Path(args.model_path).stem
    out_dir = output_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    print(f"Using device: {device}")
    print(f"Output directory: {out_dir}")

    x_raw, y, label_df = load_labeled_subset(
        data_path=args.data_path,
        labels_csv=args.labels_csv,
        baseline_center=args.baseline_center,
    )
    print(f"Loaded labeled subset: {x_raw.shape[0]} samples, {x_raw.shape[1]} points")
    print(f"Class counts -> Day1: {(y == 0).sum()}, Day4: {(y == 1).sum()}")

    if normalize_mode == "auto":
        normalize_input = not is_unit_range(x_raw)
    else:
        normalize_input = normalize_mode == "true"
    print(f"Input normalization: {'enabled' if normalize_input else 'skipped'} ({normalize_mode})")

    x = normalize_per_spectrum_minmax(x_raw) if normalize_input else x_raw.astype(np.float32, copy=False)

    model, checkpoint, dims = build_model_from_checkpoint(
        ckpt_path=args.model_path,
        spectrum_length=x.shape[1],
        device=device,
        nhead=args.nhead,
    )
    print("Loaded model with inferred dimensions:")
    print(json.dumps(dims, indent=2))

    embeddings = extract_embeddings(model, x, batch_size=args.batch_size, device=device)
    emb_path = out_dir / "embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"Saved embeddings: {embeddings.shape} -> {emb_path}")

    metadata = label_df.copy()
    metadata["target"] = y
    metadata["target_name"] = np.where(y == 0, "Day1", "Day4")
    metadata.to_csv(out_dir / "sample_metadata.csv", index=False)

    models = {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_SEED)),
            ]
        ),
        "SVM_RBF": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_SEED)),
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
            random_state=RANDOM_SEED,
        )
        xgb_available = True
    except Exception:
        print("XGBoost not available in this environment. Skipping XGBoost model.")
        print("Install with: pip install xgboost")

    summary_rows = []
    all_fold_metrics = []
    model_oof_predictions = {}

    for model_name_key, model_obj in models.items():
        print(f"\nEvaluating {model_name_key}...")
        fold_df, summary, y_pred, y_score = evaluate_model_cv(
            model=model_obj,
            x=embeddings,
            y=y,
            n_splits=args.folds,
            seed=RANDOM_SEED,
        )

        fold_df.insert(0, "model", model_name_key)
        all_fold_metrics.append(fold_df)
        model_oof_predictions[model_name_key] = y_pred.copy()

        row = {"model": model_name_key}
        row.update(summary)
        summary_rows.append(row)

        fold_df.to_csv(out_dir / f"{model_name_key}_fold_metrics.csv", index=False)

        np.save(out_dir / f"{model_name_key}_oof_pred.npy", y_pred)
        np.save(out_dir / f"{model_name_key}_oof_score.npy", y_score)

        plot_model_performance(
            name=model_name_key,
            y_true=y,
            y_pred=y_pred,
            y_score=y_score,
            fold_df=fold_df,
            out_path=out_dir / f"{model_name_key}_performance.png",
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("roc_auc_mean", ascending=False)
    summary_df.to_csv(out_dir / "model_summary.csv", index=False)

    fold_metrics_df = pd.concat(all_fold_metrics, ignore_index=True)
    fold_metrics_df.to_csv(out_dir / "all_models_fold_metrics.csv", index=False)

    plot_model_comparison(summary_df, out_dir / "model_comparison.png")

    # Per-sample drill-down table: title, true label, and OOF prediction by model.
    per_sample_df = metadata.copy()
    per_sample_df["sample_title"] = per_sample_df.get("title", "").astype(str)
    per_sample_df["original_label"] = np.where(per_sample_df["target"] == 0, "Day1", "Day4")
    for model_name_key, y_pred in model_oof_predictions.items():
        col = f"pred_{model_name_key}"
        per_sample_df[col] = np.where(y_pred == 0, "Day1", "Day4")
    ordered_cols = ["spectrum_index", "sample_title", "original_label"] + [
        c for c in per_sample_df.columns if c.startswith("pred_")
    ]
    per_sample_df[ordered_cols].to_csv(out_dir / "per_sample_predictions.csv", index=False)

    run_info = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "labels_csv": args.labels_csv,
        "output_dir": str(out_dir),
        "device_used": str(device),
        "n_samples_used": int(len(y)),
        "class_counts": {"Day1": int((y == 0).sum()), "Day4": int((y == 1).sum())},
        "normalize_input_mode": normalize_mode,
        "normalize_input_effective": bool(normalize_input),
        "baseline_center": bool(args.baseline_center),
        "folds": int(args.folds),
        "xgboost_available": bool(xgb_available),
        "inferred_model_dims": dims,
        "checkpoint_hyperparameters": checkpoint.get("hyperparameters", {}),
    }
    with open(out_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print("\nDone.")
    print(f"Saved summary: {out_dir / 'model_summary.csv'}")
    print(f"Saved comparison plot: {out_dir / 'model_comparison.png'}")


if __name__ == "__main__":
    main()
