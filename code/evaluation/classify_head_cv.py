"""
5-fold CV classification using a pretrained MAE backbone + classification head.

Two training modes are evaluated:
1) frozen_backbone: only the classification head is trained
2) finetune_backbone: backbone and head are both trained

Outputs include fold metrics, summary tables, per-mode plots, comparison plot,
and per-sample out-of-fold predictions.
"""

import argparse
import copy
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
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
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from trainer_revised import NMRMaskedAutoencoder


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def is_unit_range(x, tol=1e-6):
    return float(np.min(x)) >= -tol and float(np.max(x)) <= (1.0 + tol)


def normalize_per_spectrum_minmax(spectra):
    out = np.zeros_like(spectra, dtype=np.float32)
    for i in range(spectra.shape[0]):
        s = spectra[i]
        mn = s.min()
        mx = s.max()
        if mx - mn > 1e-8:
            out[i] = (s - mn) / (mx - mn)
        else:
            out[i] = s
    return out


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


class MAEClassifier(nn.Module):
    def __init__(self, backbone, d_model, head_dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(head_dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        _, encoded = self.backbone(x, mask=None)
        pooled = encoded.mean(dim=1)
        logits = self.head(pooled).squeeze(-1)
        return logits


def build_classifier_from_checkpoint(
    checkpoint_state,
    spectrum_length,
    device,
    nhead,
    head_dropout,
):
    dims = infer_model_dims(checkpoint_state, default_nhead=nhead)
    if dims["d_model"] % dims["nhead"] != 0:
        raise ValueError(
            f"d_model={dims['d_model']} is not divisible by nhead={dims['nhead']}. "
            "Pass a valid --nhead."
        )

    backbone = NMRMaskedAutoencoder(
        spectrum_length=spectrum_length,
        patch_size=dims["patch_size"],
        d_model=dims["d_model"],
        nhead=dims["nhead"],
        num_layers=dims["num_layers"],
        dim_feedforward=dims["dim_feedforward"],
        dropout=dims["dropout"],
    )
    backbone.load_state_dict(checkpoint_state)
    backbone.to(device)
    model = MAEClassifier(backbone, d_model=dims["d_model"], head_dropout=head_dropout).to(device)
    return model, dims


def create_loaders(x_train, y_train, x_val, y_val, x_test, y_test, batch_size):
    train_ds = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
    val_ds = TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).float())
    test_ds = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def predict_loader(model, loader, device):
    model.eval()
    ys = []
    scores = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            prob = torch.sigmoid(logits).cpu().numpy()
            ys.append(yb.numpy())
            scores.append(prob)
    y_true = np.concatenate(ys).astype(int)
    y_score = np.concatenate(scores)
    y_pred = (y_score >= 0.5).astype(int)
    return y_true, y_pred, y_score


def fold_metrics(y_true, y_pred, y_score):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }


def train_one_fold(
    mode,
    checkpoint_state,
    spectrum_length,
    nhead,
    x_train,
    y_train,
    x_test,
    y_test,
    args,
    device,
):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=args.seed)
    tr_idx, va_idx = next(splitter.split(x_train, y_train))
    x_tr, y_tr = x_train[tr_idx], y_train[tr_idx]
    x_va, y_va = x_train[va_idx], y_train[va_idx]

    train_loader, val_loader, test_loader = create_loaders(
        x_tr, y_tr, x_va, y_va, x_test, y_test, args.batch_size
    )

    model, dims = build_classifier_from_checkpoint(
        checkpoint_state=checkpoint_state,
        spectrum_length=spectrum_length,
        device=device,
        nhead=nhead,
        head_dropout=args.head_dropout,
    )

    if mode == "frozen_backbone":
        for p in model.backbone.parameters():
            p.requires_grad = False
        trainable = [p for p in model.head.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr_head, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": args.lr_backbone},
                {"params": model.head.parameters(), "lr": args.lr_head},
            ],
            weight_decay=args.weight_decay,
        )

    criterion = nn.BCEWithLogitsLoss()
    best_state = None
    best_val_auc = -1.0
    best_epoch = 0
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        if mode == "frozen_backbone":
            model.backbone.eval()

        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_true, val_pred, val_score = predict_loader(model, val_loader, device)
        val_m = fold_metrics(val_true, val_pred, val_score)
        tr_loss = float(np.mean(train_losses)) if train_losses else np.nan
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_accuracy": val_m["accuracy"],
                "val_f1": val_m["f1"],
                "val_roc_auc": val_m["roc_auc"],
                "val_pr_auc": val_m["pr_auc"],
            }
        )

        improved = val_m["roc_auc"] > (best_val_auc + 1e-6)
        if improved:
            best_val_auc = val_m["roc_auc"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_true, test_pred, test_score = predict_loader(model, test_loader, device)
    test_m = fold_metrics(test_true, test_pred, test_score)
    test_m["best_epoch"] = best_epoch
    return test_m, test_pred, test_score, history, dims


def summarize_fold_df(df):
    summary = {}
    for m in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        summary[f"{m}_mean"] = float(df[m].mean())
        summary[f"{m}_std"] = float(df[m].std(ddof=1))
    return summary


def plot_mode_performance(mode_name, y_true, y_pred, y_score, fold_df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
    axes[0].set_title(f"{mode_name} - Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_xticklabels(["Day 1", "Day 4"])
    axes[0].set_yticklabels(["Day 1", "Day 4"], rotation=0)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[1].set_title(f"{mode_name} - ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.3)

    prec, rec, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    axes[2].plot(rec, prec, label=f"AP = {pr_auc:.3f}")
    axes[2].set_title(f"{mode_name} - Precision-Recall")
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].legend(loc="lower left")
    axes[2].grid(alpha=0.3)

    m = fold_df.mean(numeric_only=True)
    s = fold_df.std(numeric_only=True, ddof=1)
    fig.suptitle(
        f"{mode_name} | "
        f"Acc {m['accuracy']:.3f}±{s['accuracy']:.3f}, "
        f"F1 {m['f1']:.3f}±{s['f1']:.3f}, "
        f"ROC-AUC {m['roc_auc']:.3f}±{s['roc_auc']:.3f}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mode_comparison(summary_df, out_path):
    metrics = ["accuracy_mean", "f1_mean", "roc_auc_mean", "pr_auc_mean"]
    labels = ["Accuracy", "F1", "ROC-AUC", "PR-AUC"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (_, row) in enumerate(summary_df.iterrows()):
        means = [row[m] for m in metrics]
        stds = [row[m.replace("_mean", "_std")] for m in metrics]
        ax.bar(
            x + i * width,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            alpha=0.85,
            label=row["mode"],
        )

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Frozen vs Finetune (5-fold CV)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(histories_by_mode, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for mode_name, histories in histories_by_mode.items():
        for fold_idx, hist in enumerate(histories, start=1):
            hdf = pd.DataFrame(hist)
            axes[0].plot(hdf["epoch"], hdf["train_loss"], alpha=0.35, label=f"{mode_name} fold{fold_idx}")
            axes[1].plot(hdf["epoch"], hdf["val_roc_auc"], alpha=0.35, label=f"{mode_name} fold{fold_idx}")

    axes[0].set_title("Training Loss by Fold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Validation ROC-AUC by Fold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].grid(alpha=0.3)

    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_mode_cv(mode_name, checkpoint_state, x, y, args, device):
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_pred = np.zeros_like(y)
    oof_score = np.zeros_like(y, dtype=np.float64)
    fold_rows = []
    histories = []
    last_dims = None

    pbar = tqdm(list(skf.split(x, y)), desc=f"{mode_name} folds")
    for fold_id, (tr_idx, te_idx) in enumerate(pbar, start=1):
        metrics, pred, score, history, dims = train_one_fold(
            mode=mode_name,
            checkpoint_state=checkpoint_state,
            spectrum_length=x.shape[1],
            nhead=args.nhead,
            x_train=x[tr_idx],
            y_train=y[tr_idx],
            x_test=x[te_idx],
            y_test=y[te_idx],
            args=args,
            device=device,
        )
        last_dims = dims
        histories.append(history)
        oof_pred[te_idx] = pred
        oof_score[te_idx] = score
        row = {"fold": fold_id}
        row.update(metrics)
        fold_rows.append(row)
        pbar.set_postfix({"f1": f"{metrics['f1']:.3f}", "auc": f"{metrics['roc_auc']:.3f}"})

    fold_df = pd.DataFrame(fold_rows)
    summary = summarize_fold_df(fold_df)
    return fold_df, summary, oof_pred, oof_score, histories, last_dims


def main():
    parser = argparse.ArgumentParser(description="5-fold CV MAE-head classification (frozen + finetune).")
    parser.add_argument(
        "--model-path",
        default="models/SSL_models/aligned_nmr_spectra_128K_Plasma_WS625to680Zero_merged2_20260520_014209_bs16_mr0.35_ps1024_best.pth",
    )
    # 15 - Itr6Rerun_20251209_035637_bs16_mr0.15_ps1024_20251209_035637_best_300epochs.pth
    # 25 - Itr6Rerun_20251211_084137_bs16_mr0.25_ps1024_best.pth
    # 35 - Itr6Rerun_20251212_103256_bs16_mr0.35_ps1024_best.pth
    # 50 - Itr6Rerun_20260106_041031_bs16_mr0.50_ps1024_best.pth
    # models/SSL_models/aligned_nmr_spectra_128K_Plasma_WS625to680Zero_merged2_20260520_014209_bs16_mr0.35_ps1024_best.pth
    parser.add_argument("--data-path", default="data/tbi_tirupati/aligned_128K_TBI_Tirupati_WS625to680Zero.npy")
    parser.add_argument("--labels-csv", default="data/tbi_tirupati/title_labels.csv")
    parser.add_argument("--output-dir", default="results/classification/head")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--normalize-input", default="auto", help="auto|true|false")
    parser.add_argument("--baseline-center", type=str2bool, default=True)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    normalize_mode = parse_normalize_mode(args.normalize_input)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / Path(args.model_path).stem
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    print(f"Using device: {device}")
    print(f"Output directory: {run_dir}")

    x_raw, y, labels_df = load_labeled_subset(
        data_path=args.data_path,
        labels_csv=args.labels_csv,
        baseline_center=args.baseline_center,
    )
    print(f"Labeled samples: {len(y)} | Day1={(y==0).sum()}, Day4={(y==1).sum()}")

    if normalize_mode == "auto":
        normalize_input = not is_unit_range(x_raw)
    else:
        normalize_input = normalize_mode == "true"
    x = normalize_per_spectrum_minmax(x_raw) if normalize_input else x_raw.astype(np.float32, copy=False)
    print(f"Input normalization: {'enabled' if normalize_input else 'skipped'} ({normalize_mode})")

    checkpoint = torch.load(args.model_path, map_location="cpu")
    checkpoint_state = checkpoint["model_state_dict"]

    modes = ["frozen_backbone", "finetune_backbone"]
    summary_rows = []
    oof_store = {}
    histories_by_mode = {}
    inferred_dims = None

    for mode in modes:
        print(f"\nRunning mode: {mode}")
        fold_df, summary, oof_pred, oof_score, histories, dims = run_mode_cv(
            mode_name=mode,
            checkpoint_state=checkpoint_state,
            x=x,
            y=y,
            args=args,
            device=device,
        )
        inferred_dims = dims
        histories_by_mode[mode] = histories
        oof_store[mode] = {"pred": oof_pred, "score": oof_score}

        fold_df.insert(0, "mode", mode)
        fold_df.to_csv(run_dir / f"{mode}_fold_metrics.csv", index=False)
        np.save(run_dir / f"{mode}_oof_pred.npy", oof_pred)
        np.save(run_dir / f"{mode}_oof_score.npy", oof_score)

        row = {"mode": mode}
        row.update(summary)
        summary_rows.append(row)

        plot_mode_performance(
            mode_name=mode,
            y_true=y,
            y_pred=oof_pred,
            y_score=oof_score,
            fold_df=fold_df,
            out_path=run_dir / f"{mode}_performance.png",
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("roc_auc_mean", ascending=False)
    summary_df.to_csv(run_dir / "mode_summary.csv", index=False)
    plot_mode_comparison(summary_df, run_dir / "mode_comparison.png")
    plot_training_curves(histories_by_mode, run_dir / "training_curves.png")

    per_sample = labels_df.copy()
    per_sample["sample_title"] = per_sample.get("title", "").astype(str)
    per_sample["original_label"] = np.where(y == 0, "Day1", "Day4")
    per_sample["pred_frozen_backbone"] = np.where(oof_store["frozen_backbone"]["pred"] == 0, "Day1", "Day4")
    per_sample["pred_finetune_backbone"] = np.where(oof_store["finetune_backbone"]["pred"] == 0, "Day1", "Day4")
    per_sample["score_frozen_backbone"] = oof_store["frozen_backbone"]["score"]
    per_sample["score_finetune_backbone"] = oof_store["finetune_backbone"]["score"]
    out_cols = [
        "spectrum_index",
        "sample_title",
        "original_label",
        "pred_frozen_backbone",
        "pred_finetune_backbone",
        "score_frozen_backbone",
        "score_finetune_backbone",
    ]
    per_sample[out_cols].to_csv(run_dir / "per_sample_predictions.csv", index=False)

    run_info = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "labels_csv": args.labels_csv,
        "output_dir": str(run_dir),
        "device_used": str(device),
        "n_samples_used": int(len(y)),
        "class_counts": {"Day1": int((y == 0).sum()), "Day4": int((y == 1).sum())},
        "folds": int(args.folds),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "val_fraction": float(args.val_fraction),
        "lr_head": float(args.lr_head),
        "lr_backbone": float(args.lr_backbone),
        "weight_decay": float(args.weight_decay),
        "head_dropout": float(args.head_dropout),
        "baseline_center": bool(args.baseline_center),
        "normalize_input_mode": normalize_mode,
        "normalize_input_effective": bool(normalize_input),
        "seed": int(args.seed),
        "inferred_backbone_dims": inferred_dims,
        "checkpoint_hyperparameters": checkpoint.get("hyperparameters", {}),
    }
    with open(run_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print("\nDone.")
    print(f"Saved summary: {run_dir / 'mode_summary.csv'}")
    print(f"Saved comparison plot: {run_dir / 'mode_comparison.png'}")
    print(f"Saved per-sample predictions: {run_dir / 'per_sample_predictions.csv'}")


if __name__ == "__main__":
    main()
