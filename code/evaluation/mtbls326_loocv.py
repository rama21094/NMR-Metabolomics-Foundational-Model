#!/usr/bin/env python3
"""LOOCV classification for the 42-sample MTBLS326 dataset.

Runs two experiment families:
  1. Classical baselines without the foundation model: logistic regression,
     RBF-SVM, and XGBoost.
  2. A pretrained NMR masked-autoencoder with a two-class softmax head. The
     head is trained while either all backbone weights remain frozen or the
     last 1, 2, or 3 transformer encoder blocks are fine-tuned.

All preprocessing that learns parameters (feature scaling) is fitted inside
each LOOCV training fold. The held-out sample is never used for training or
early stopping.
"""

import os

# This workstation can advertise more threads than numexpr's default cap.
os.environ.setdefault("NUMEXPR_MAX_THREADS", "256")
# Avoid slow Matplotlib cache setup when trainer_revised is imported on a node
# where the home configuration directory is read-only.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "code" / "training"
for path in (ROOT, TRAINING_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from trainer_revised import NMRMaskedAutoencoder


# ---------------------------------------------------------------------------
# IDE CONFIGURATION
# ---------------------------------------------------------------------------
# Set this to True to run from an IDE using the values below. Set it to False
# to use command-line arguments. Only edit this block for normal experiments.
USE_IDE_CONFIG = True

IDE_CONFIG = {
    # Input/output files
    "data": "data/mtbls326/MTBLS326_aligned_spectra_WS625to680Zero_rowMinMax.npy",
    "metadata": "data/mtbls326/MTBLS326_metadata_mapping.csv",
    "checkpoint": "./models/SSL_models/combined_unique_WS625to680Zero_20260601_084533_bs32_mr0.50_ps1024_best.pth",
    "output_dir": "results/loocv/mtbls326_mask_0.50_rowMinMax",

    # Select experiments. Valid combinations are:
    #   classical_only=True,  foundation_only=False -> classical models only
    #   classical_only=False, foundation_only=True  -> foundation model only
    #   classical_only=False, foundation_only=False -> run everything
    "classical_only": False,
    "foundation_only": False,

    # Classical-model settings
    "classical_features": "binned_auc",  # "binned_auc" or "raw"
    "feature_bins": 1024,
    "xgb_jobs": 4,

    # Foundation-model training settings
    "epochs": 50,
    "batch_size": 8,
    "head_lr": 1e-3,
    "backbone_lr": 1e-5,
    "weight_decay": 1e-4,
    "head_dropout": 0.1,
    "backbone_dropout": 0.15,
    "nhead": 8,             # Must match the pretraining architecture
    "device": "auto",      # "auto", "cpu", or "cuda"
    "seed": 42,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_mtbls326(data_path, metadata_path):
    spectra = np.load(data_path).astype(np.float32)
    with open(metadata_path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if not rows or "label" not in rows[0]:
        raise ValueError(f"{metadata_path} does not contain a 'label' column")
    if "npy_row" not in rows[0]:
        raise ValueError(f"{metadata_path} does not contain an 'npy_row' column")

    used = []
    for row in rows:
        label = str(row["label"]).strip()
        if not label:
            continue
        try:
            npy_row = int(row["npy_row"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid npy_row value: {row['npy_row']!r}") from exc
        if not 0 <= npy_row < len(spectra):
            raise IndexError(f"npy_row {npy_row} is outside spectra array")
        used.append((npy_row, label, row))

    used.sort(key=lambda item: item[0])
    label_names = sorted({item[1] for item in used})
    if len(label_names) != 2:
        raise ValueError(f"Expected exactly two labels; found {label_names}")
    # Make the common MTBLS326 convention explicit and deterministic.
    if set(label_names) == {"No", "Yes"}:
        label_names = ["No", "Yes"]
    label_to_index = {name: i for i, name in enumerate(label_names)}

    indices = np.asarray([item[0] for item in used], dtype=int)
    labels = np.asarray([label_to_index[item[1]] for item in used], dtype=np.int64)
    metadata = [item[2] for item in used]
    return spectra[indices], labels, metadata, label_names


def binned_abs_area(spectra, n_bins):
    """Reduce each spectrum to absolute integrated area in equal-width bins."""
    edges = np.linspace(0, spectra.shape[1], n_bins + 1, dtype=int)
    features = np.empty((len(spectra), n_bins), dtype=np.float32)
    integrate = getattr(np, "trapezoid", np.trapz)
    for i, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        segment = np.abs(spectra[:, start:stop])
        features[:, i] = integrate(segment, axis=1) if stop - start > 1 else segment[:, 0]
    return features


def aggregate_metrics(y_true, y_pred, y_score):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "tn": int(confusion_matrix(y_true, y_pred, labels=[0, 1])[0, 0]),
        "fp": int(confusion_matrix(y_true, y_pred, labels=[0, 1])[0, 1]),
        "fn": int(confusion_matrix(y_true, y_pred, labels=[0, 1])[1, 0]),
        "tp": int(confusion_matrix(y_true, y_pred, labels=[0, 1])[1, 1]),
    }


def positive_scores(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def classical_models(seed, xgb_jobs):
    models = {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, C=1.0, random_state=seed)),
        ]),
        "svm_rbf": Pipeline([
            ("scale", StandardScaler()),
            ("model", SVC(C=1.0, kernel="rbf", gamma="scale", probability=True,
                          random_state=seed)),
        ]),
    }
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("XGBoost is required: install the 'xgboost' package") from exc
    models["xgboost"] = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=xgb_jobs,
    )
    return models


def run_classical_loocv(features, labels, seed, xgb_jobs):
    results = {}
    splitter = LeaveOneOut()
    for name, estimator in classical_models(seed, xgb_jobs).items():
        predictions = np.empty(len(labels), dtype=np.int64)
        scores = np.empty(len(labels), dtype=np.float64)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(features), 1):
            model = clone(estimator)
            model.fit(features[train_idx], labels[train_idx])
            predictions[test_idx] = model.predict(features[test_idx])
            scores[test_idx] = positive_scores(model, features[test_idx])
            print(f"\r{name}: LOOCV fold {fold}/{len(labels)}", end="", flush=True)
        print()
        results[name] = {
            "predictions": predictions,
            "scores": scores,
            "metrics": aggregate_metrics(labels, predictions, scores),
        }
    return results


def checkpoint_state(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def infer_backbone_config(state, nhead, dropout):
    patch_weight = state["encoder.patch_embedding.0.weight"]
    ff_weight = state["encoder.transformer.layers.0.linear1.weight"]
    layer_ids = {
        int(key.split(".")[3])
        for key in state
        if key.startswith("encoder.transformer.layers.") and key.split(".")[3].isdigit()
    }
    config = {
        "patch_size": int(patch_weight.shape[1]),
        "d_model": int(patch_weight.shape[0]),
        "nhead": int(nhead),
        "num_layers": max(layer_ids) + 1,
        "dim_feedforward": int(ff_weight.shape[0]),
        "dropout": float(dropout),
    }
    if config["d_model"] % config["nhead"]:
        raise ValueError("Checkpoint d_model must be divisible by --nhead")
    return config


class SoftmaxMAEClassifier(nn.Module):
    """Pretrained encoder, mean pooling, and an explicit two-class softmax."""

    def __init__(self, backbone, d_model, head_dropout):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(head_dropout),
            nn.Linear(d_model, 2),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, return_logits=False):
        _, encoded = self.backbone(x, mask=None)
        logits = self.classifier(encoded.mean(dim=1))
        return logits if return_logits else self.softmax(logits)


def build_foundation_model(state, spectrum_length, nhead, backbone_dropout,
                           head_dropout, unfreeze_layers, device):
    config = infer_backbone_config(state, nhead, backbone_dropout)
    backbone = NMRMaskedAutoencoder(spectrum_length=spectrum_length, **config)
    backbone.load_state_dict(state, strict=True)
    model = SoftmaxMAEClassifier(backbone, config["d_model"], head_dropout)

    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    layers = model.backbone.encoder.transformer.layers
    if unfreeze_layers > len(layers):
        raise ValueError(f"Cannot unfreeze {unfreeze_layers}; backbone has {len(layers)} layers")
    for layer in list(layers)[len(layers) - unfreeze_layers:] if unfreeze_layers else []:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    model.unfreeze_layers = unfreeze_layers
    return model.to(device), config


def train_one_fold(model, x_train, y_train, device, epochs, batch_size,
                   head_lr, backbone_lr, weight_decay, seed):
    set_seed(seed)
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    groups = [{"params": head_params, "lr": head_lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=min(batch_size, len(y_train)), shuffle=True, generator=generator,
        num_workers=0,
    )
    model.train()
    # Frozen encoder blocks should not update dropout state during head training.
    model.backbone.eval()
    if model.unfreeze_layers:
        for layer in list(model.backbone.encoder.transformer.layers)[-model.unfreeze_layers:]:
            layer.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb.to(device), return_logits=True)
            loss = loss_fn(logits, yb.to(device))
            loss.backward()
            optimizer.step()


def run_foundation_loocv(spectra, labels, checkpoint_path, args, device):
    state = checkpoint_state(checkpoint_path)
    results = {}
    modes = [("frozen", 0), ("unfreeze_last_1", 1),
             ("unfreeze_last_2", 2), ("unfreeze_last_3", 3)]
    for mode, unfreeze_count in modes:
        predictions = np.empty(len(labels), dtype=np.int64)
        scores = np.empty(len(labels), dtype=np.float64)
        for fold, (train_idx, test_idx) in enumerate(LeaveOneOut().split(spectra), 1):
            # Rebuild from the identical pretrained state for every fold.
            set_seed(args.seed + fold)
            model, config = build_foundation_model(
                state, spectra.shape[1], args.nhead, args.backbone_dropout,
                args.head_dropout, unfreeze_count, device,
            )
            train_one_fold(
                model, spectra[train_idx], labels[train_idx], device,
                args.epochs, args.batch_size, args.head_lr, args.backbone_lr,
                args.weight_decay, args.seed + fold,
            )
            model.eval()
            with torch.no_grad():
                probability = model(torch.from_numpy(spectra[test_idx]).to(device))[0]
            scores[test_idx] = float(probability[1].cpu())
            predictions[test_idx] = int(probability.argmax().cpu())
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"\r{mode}: LOOCV fold {fold}/{len(labels)}", end="", flush=True)
        print()
        results[mode] = {
            "predictions": predictions,
            "scores": scores,
            "metrics": aggregate_metrics(labels, predictions, scores),
            "unfrozen_transformer_layers": unfreeze_count,
            "backbone_config": config,
        }
    return results


def save_results(output_dir, metadata, labels, label_names, families, run_config):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    prediction_rows = []
    for i, row in enumerate(metadata):
        prediction_rows.append({
            "npy_row": row["npy_row"],
            "sample_name": row.get("Sample Name", ""),
            "label": label_names[labels[i]],
            "target": int(labels[i]),
        })

    for family, models in families.items():
        for model_name, result in models.items():
            summary_rows.append({"family": family, "model": model_name, **result["metrics"]})
            np.save(output_dir / f"{family}_{model_name}_oof_pred.npy", result["predictions"])
            np.save(output_dir / f"{family}_{model_name}_oof_score.npy", result["scores"])
            for i, (pred, score) in enumerate(zip(result["predictions"], result["scores"])):
                prefix = f"{family}_{model_name}"
                prediction_rows[i][f"{prefix}_prediction"] = label_names[int(pred)]
                prediction_rows[i][f"{prefix}_yes_probability"] = float(score)

    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with open(output_dir / "oof_predictions.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/mtbls326/MTBLS326_aligned_spectra_WS625to680Zero_rowMinMax.npy")
    parser.add_argument("--metadata", default="data/mtbls326/MTBLS326_metadata_mapping.csv")
    parser.add_argument("--checkpoint", help="Pretrained MAE .pth file (required unless --classical-only)")
    parser.add_argument("--output-dir", default="mtbls326_loocv_results")
    parser.add_argument("--classical-only", action="store_true")
    parser.add_argument("--foundation-only", action="store_true")
    parser.add_argument("--classical-features", choices=["binned_auc", "raw"], default="binned_auc")
    parser.add_argument("--feature-bins", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--backbone-dropout", type=float, default=0.15)
    parser.add_argument("--nhead", type=int, default=8,
                        help="Attention heads used during pretraining (not stored in checkpoint)")
    parser.add_argument("--xgb-jobs", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    if USE_IDE_CONFIG:
        unknown = set(IDE_CONFIG) - {action.dest for action in parser._actions}
        if unknown:
            parser.error(f"Unknown IDE_CONFIG entries: {sorted(unknown)}")
        args = parser.parse_args([])
        for name, value in IDE_CONFIG.items():
            setattr(args, name, value)
        print("Using settings from IDE_CONFIG")
    else:
        args = parser.parse_args()

    if args.classical_only and args.foundation_only:
        parser.error("--classical-only and --foundation-only are mutually exclusive")
    if not args.classical_only and not args.checkpoint:
        parser.error("--checkpoint is required for foundation-model experiments")
    if args.classical_features not in {"binned_auc", "raw"}:
        parser.error("classical_features must be 'binned_auc' or 'raw'")
    if args.device not in {"auto", "cpu", "cuda"}:
        parser.error("device must be 'auto', 'cpu', or 'cuda'")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    spectra, labels, metadata, label_names = load_mtbls326(args.data, args.metadata)
    if len(labels) != 42:
        print(f"Warning: expected 42 labeled samples, found {len(labels)}")
    counts = {name: int((labels == i).sum()) for i, name in enumerate(label_names)}
    print(f"Loaded {spectra.shape}; label mapping={dict(enumerate(label_names))}; counts={counts}")

    families = {}
    if not args.foundation_only:
        features = (binned_abs_area(spectra, args.feature_bins)
                    if args.classical_features == "binned_auc" else spectra)
        families["classical"] = run_classical_loocv(features, labels, args.seed, args.xgb_jobs)

    if not args.classical_only:
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        print(f"Foundation-model device: {device}")
        families["foundation"] = run_foundation_loocv(
            spectra, labels, args.checkpoint, args, device,
        )

    run_config = vars(args).copy()
    run_config.update({"n_samples": len(labels), "spectrum_length": spectra.shape[1],
                       "label_mapping": {str(i): name for i, name in enumerate(label_names)}})
    save_results(Path(args.output_dir), metadata, labels, label_names, families, run_config)
    print(f"\nResults written to {args.output_dir}/summary.csv")
    for family, models in families.items():
        for name, result in models.items():
            m = result["metrics"]
            print(f"{family}/{name}: accuracy={m['accuracy']:.3f}, "
                  f"balanced_accuracy={m['balanced_accuracy']:.3f}, "
                  f"F1={m['f1']:.3f}, ROC-AUC={m['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
