#!/usr/bin/env python3
"""LOOCV classification for MTBLS326 using jigsaw foundation checkpoints.

This is the jigsaw counterpart to mtbls326_loocv.py. It keeps the same output
shape while rebuilding JigsawNMRModel checkpoints and training a small
two-class head in each leave-one-out fold.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("NUMEXPR_MAX_THREADS", "256")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

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
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from train_jigsaw_spectra import JigsawNMRModel  # noqa: E402


FINE_TUNE_CHOICES = ("frozen", "unfreeze_last_1", "unfreeze_last_2", "unfreeze_last_3")


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
    if set(label_names) == {"No", "Yes"}:
        label_names = ["No", "Yes"]
    if len(label_names) != 2:
        raise ValueError(f"Expected exactly two labels; found {label_names}")
    label_to_index = {name: i for i, name in enumerate(label_names)}

    indices = np.asarray([item[0] for item in used], dtype=int)
    labels = np.asarray([label_to_index[item[1]] for item in used], dtype=np.int64)
    metadata = [item[2] for item in used]
    return spectra[indices], labels, metadata, label_names


def binned_abs_area(spectra, n_bins):
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
            ("model", SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=seed)),
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
            print(f"\rclassical/{name}: LOOCV fold {fold}/{len(labels)}", end="", flush=True)
        print()
        results[name] = {
            "predictions": predictions,
            "scores": scores,
            "metrics": aggregate_metrics(labels, predictions, scores),
        }
    return results


def normalize_batch(x):
    x = x.astype(np.float32, copy=True)
    finite = np.isfinite(x)
    if not np.all(finite):
        x[~finite] = 0.0
    lo = x.min(axis=1, keepdims=True)
    hi = x.max(axis=1, keepdims=True)
    denom = hi - lo
    mask = denom > 1e-8
    out = x.copy()
    out[mask[:, 0]] = (x[mask[:, 0]] - lo[mask[:, 0]]) / denom[mask[:, 0]]
    return out


def resolve_normalize_mode(mode, spectra):
    if mode == "true":
        return True
    if mode == "false":
        return False
    if mode != "auto":
        raise ValueError("--normalize-input must be auto, true, or false")
    finite = spectra[np.isfinite(spectra)]
    if finite.size == 0:
        return True
    sampled_min = float(finite.min())
    sampled_max = float(finite.max())
    resolved = sampled_min < -1e-4 or sampled_max > 1.5
    print(f"Jigsaw normalization auto-check: range [{sampled_min:.4g}, {sampled_max:.4g}] -> {resolved}")
    return resolved


def natural_bins(x, bin_size, normalize_input):
    if normalize_input:
        x = normalize_batch(x)
    else:
        x = x.astype(np.float32, copy=True)
        x[~np.isfinite(x)] = 0.0
    trimmed_length = (x.shape[1] // bin_size) * bin_size
    if trimmed_length <= 0:
        raise ValueError(f"Bin size {bin_size} is too large for spectra length {x.shape[1]}")
    return torch.from_numpy(x[:, :trimmed_length].reshape(len(x), trimmed_length // bin_size, bin_size).copy())


def checkpoint_label(path, bin_sizes):
    if len(bin_sizes) == 1:
        return f"jigsaw_bin_{bin_sizes[0]}"
    return "jigsaw_multibin"


def discover_checkpoints(pattern, max_checkpoints=None):
    paths = sorted(Path(p) for p in glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No jigsaw checkpoints matched: {pattern}")

    grouped = {}
    for path in paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        bin_sizes = [int(b) for b in ckpt["bin_sizes"]]
        label = checkpoint_label(path, bin_sizes)
        if label not in grouped or path.stat().st_mtime > grouped[label].stat().st_mtime:
            grouped[label] = path

    selected = [grouped[key] for key in sorted(grouped, key=lambda x: (x != "jigsaw_multibin", x))]
    selected = sorted(selected, key=lambda p: checkpoint_label_from_path(p))
    if max_checkpoints is not None:
        selected = selected[: int(max_checkpoints)]
    return selected


def checkpoint_label_from_path(path):
    text = str(path)
    if "multibin" in text:
        return "jigsaw_multibin"
    for size in (256, 512, 1024, 2048):
        if f"bin_{size}" in text:
            return f"jigsaw_bin_{size}"
    return path.stem


def load_jigsaw_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hp = checkpoint.get("hyperparameters", {})
    bin_sizes = [int(b) for b in checkpoint["bin_sizes"]]
    model = JigsawNMRModel(
        spectrum_length=int(checkpoint["spectrum_length"]),
        bin_sizes=bin_sizes,
        d_model=int(hp.get("d_model", 192)),
        nhead=int(hp.get("nhead", 6)),
        num_layers=int(hp.get("num_layers", 4)),
        dim_feedforward=int(hp.get("dim_feedforward", 768)),
        dropout=float(hp.get("dropout", 0.15)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device), checkpoint


class JigsawSoftmaxClassifier(nn.Module):
    def __init__(self, backbone, bin_sizes, d_model, head_dropout, normalize_input):
        super().__init__()
        self.backbone = backbone
        self.bin_sizes = [int(b) for b in bin_sizes]
        self.normalize_input = bool(normalize_input)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * len(self.bin_sizes)),
            nn.Dropout(head_dropout),
            nn.Linear(d_model * len(self.bin_sizes), 2),
        )
        self.softmax = nn.Softmax(dim=1)
        self.unfreeze_layers = 0

    def encode_one_bin_size(self, x, bin_size):
        bins = natural_bins(x.detach().cpu().numpy(), bin_size, self.normalize_input).to(x.device)
        key = str(int(bin_size))
        encoded = self.backbone.input_projections[key](bins)
        positions = torch.arange(encoded.shape[1], device=encoded.device)
        encoded = encoded + self.backbone.slot_embedding(positions).unsqueeze(0)
        encoded = self.backbone.transformer(encoded)
        return encoded.mean(dim=1)

    def encode(self, x):
        pooled = [self.encode_one_bin_size(x, bin_size) for bin_size in self.bin_sizes]
        return torch.cat(pooled, dim=1)

    def forward(self, x, return_logits=False):
        logits = self.classifier(self.encode(x))
        return logits if return_logits else self.softmax(logits)


def build_jigsaw_classifier(checkpoint_path, spectra, args, device, unfreeze_layers):
    backbone, checkpoint = load_jigsaw_checkpoint(checkpoint_path, device)
    bin_sizes = [int(b) for b in checkpoint["bin_sizes"]]
    hp = checkpoint.get("hyperparameters", {})
    normalize_input = resolve_normalize_mode(args.normalize_input, spectra)
    model = JigsawSoftmaxClassifier(
        backbone=backbone,
        bin_sizes=bin_sizes,
        d_model=int(hp.get("d_model", 192)),
        head_dropout=args.head_dropout,
        normalize_input=normalize_input,
    )

    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    layers = model.backbone.transformer.layers
    if unfreeze_layers > len(layers):
        raise ValueError(f"Cannot unfreeze {unfreeze_layers}; backbone has {len(layers)} layers")
    for layer in list(layers)[len(layers) - unfreeze_layers:] if unfreeze_layers else []:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    model.unfreeze_layers = unfreeze_layers
    return model.to(device), {
        "bin_sizes": bin_sizes,
        "spectrum_length": int(checkpoint["spectrum_length"]),
        "d_model": int(hp.get("d_model", 192)),
        "nhead": int(hp.get("nhead", 6)),
        "num_layers": int(hp.get("num_layers", 4)),
        "dim_feedforward": int(hp.get("dim_feedforward", 768)),
        "dropout": float(hp.get("dropout", 0.15)),
        "normalize_input": normalize_input,
    }


def fine_tune_count(mode):
    if mode == "frozen":
        return 0
    return int(mode.rsplit("_", 1)[-1])


def train_one_fold(model, x_train, y_train, device, epochs, batch_size, head_lr, backbone_lr, weight_decay, seed):
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
        batch_size=min(batch_size, len(y_train)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    for _ in range(epochs):
        model.classifier.train()
        model.backbone.eval()
        if model.unfreeze_layers:
            for layer in list(model.backbone.transformer.layers)[-model.unfreeze_layers:]:
                layer.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb.to(device), return_logits=True)
            loss = loss_fn(logits, yb.to(device))
            loss.backward()
            optimizer.step()


def run_jigsaw_loocv(spectra, labels, checkpoint_paths, args, device):
    results = {}
    for checkpoint_path in checkpoint_paths:
        base_label = checkpoint_label_from_path(checkpoint_path)
        for mode in args.fine_tune_modes:
            unfreeze_count = fine_tune_count(mode)
            result_name = f"{base_label}_{mode}"
            predictions = np.empty(len(labels), dtype=np.int64)
            scores = np.empty(len(labels), dtype=np.float64)
            config = None
            for fold, (train_idx, test_idx) in enumerate(LeaveOneOut().split(spectra), 1):
                set_seed(args.seed + fold)
                model, config = build_jigsaw_classifier(
                    checkpoint_path, spectra, args, device, unfreeze_count
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
                print(f"\r{result_name}: LOOCV fold {fold}/{len(labels)}", end="", flush=True)
            print()
            results[result_name] = {
                "predictions": predictions,
                "scores": scores,
                "metrics": aggregate_metrics(labels, predictions, scores),
                "checkpoint": str(checkpoint_path),
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

    if not summary_rows:
        raise ValueError("No models were evaluated; nothing to save.")
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
    parser.add_argument("--checkpoint-glob", default="models/jigsaw/*/*/*_best.pth")
    parser.add_argument("--checkpoint", action="append", default=[], help="Explicit jigsaw checkpoint path. Can be repeated.")
    parser.add_argument("--output-dir", default="results/loocv/mtbls326_jigsaw")
    parser.add_argument("--classical-only", action="store_true")
    parser.add_argument("--jigsaw-only", action="store_true")
    parser.add_argument("--classical-features", choices=["binned_auc", "raw"], default="binned_auc")
    parser.add_argument("--feature-bins", type=int, default=1024)
    parser.add_argument("--fine-tune-modes", nargs="+", choices=FINE_TUNE_CHOICES, default=["frozen"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--normalize-input", choices=["auto", "true", "false"], default="true")
    parser.add_argument("--xgb-jobs", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    args = parser.parse_args()

    if args.classical_only and args.jigsaw_only:
        parser.error("--classical-only and --jigsaw-only are mutually exclusive")
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
    if not args.jigsaw_only:
        features = binned_abs_area(spectra, args.feature_bins) if args.classical_features == "binned_auc" else spectra
        families["classical"] = run_classical_loocv(features, labels, args.seed, args.xgb_jobs)

    checkpoint_paths = []
    if not args.classical_only:
        if args.checkpoint:
            checkpoint_paths = [Path(p) for p in args.checkpoint]
            if args.max_checkpoints is not None:
                checkpoint_paths = checkpoint_paths[: args.max_checkpoints]
        else:
            checkpoint_paths = discover_checkpoints(args.checkpoint_glob, args.max_checkpoints)
        print("Jigsaw checkpoints:")
        for path in checkpoint_paths:
            print(f"  - {checkpoint_label_from_path(path)}: {path}")

        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        print(f"Jigsaw-model device: {device}")
        families["jigsaw"] = run_jigsaw_loocv(spectra, labels, checkpoint_paths, args, device)

    run_config = vars(args).copy()
    run_config.update({
        "n_samples": len(labels),
        "spectrum_length": spectra.shape[1],
        "label_mapping": {str(i): name for i, name in enumerate(label_names)},
        "checkpoint_paths": [str(p) for p in checkpoint_paths],
    })
    save_results(Path(args.output_dir), metadata, labels, label_names, families, run_config)
    print(f"\nResults written to {args.output_dir}/summary.csv")
    for family, models in families.items():
        for name, result in models.items():
            m = result["metrics"]
            print(
                f"{family}/{name}: accuracy={m['accuracy']:.3f}, "
                f"balanced_accuracy={m['balanced_accuracy']:.3f}, F1={m['f1']:.3f}, ROC-AUC={m['roc_auc']:.3f}"
            )


if __name__ == "__main__":
    main()
