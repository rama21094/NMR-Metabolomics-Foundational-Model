#!/usr/bin/env python3
"""LOOCV evaluation for the Barth Syndrome dataset across all model families.

The script compares:
  1. Classical ML baselines: logistic regression, RBF-SVM, and XGBoost.
  2. Masked autoencoder SSL checkpoints, with frozen and partial fine-tuning.
  3. Jigsaw SSL checkpoints, including single-bin and multibin models.
  4. Joint masked + jigsaw SSL checkpoints.

Edit IDE_CONFIG near the top to change checkpoint paths from an IDE, or keep
USE_IDE_CONFIG=False and pass paths on the command line.
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
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "code" / "evaluation", ROOT / "code" / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from brc_t2d_common import binned_abs_area, classical_models, probability_matrix  # noqa: E402
from joint_ssl_eval_common import (  # noqa: E402
    FINE_TUNE_CHOICES,
    build_joint_classifier,
    choose_device,
    fine_tune_count,
    maybe_normalize_eval_spectra,
    train_one_fold as train_joint_one_fold,
)
from train_jigsaw_spectra import JigsawNMRModel  # noqa: E402
from trainer_revised import NMRMaskedAutoencoder  # noqa: E402


# ---------------------------------------------------------------------------
# IDE CONFIGURATION
# ---------------------------------------------------------------------------
# Set to True to ignore command-line arguments and use the values below.
USE_IDE_CONFIG = False

IDE_CONFIG = {
    "data": "data/Barth/aligned_128K_Workbench_Barth_Syndrome.npy",
    "metadata": "data/Barth/Workbench_Barth_Syndrome_metadata.csv",
    "output_dir": "results/loocv/barth_all_models",
    "label_column": "label",
    "exclude_labels": ["Pool"],
    "families": ["classical", "masking", "jigsaw", "joint_ssl"],
    "classical_features": "binned_auc",
    "feature_bins": 1024,
    "masking_checkpoints": {
        "mask_20": "models/SSL_models/combine_unique_Water_EDTA_Suppressed_20260614_084450_bs32_mr0.20_ps1024_best.pth",
        "mask_30": "models/SSL_models/combine_unique_Water_EDTA_Suppressed_20260614_114013_bs32_mr0.30_ps1024_best.pth",
        "mask_40": "models/SSL_models/combine_unique_Water_EDTA_Suppressed_20260614_084724_bs32_mr0.40_ps1024_best.pth",
        "mask_50": "models/SSL_models/combine_unique_Water_EDTA_Suppressed_20260614_090604_bs32_mr0.50_ps1024_best.pth",
    },
    "jigsaw_checkpoints": {},
    "jigsaw_checkpoint_glob": "models/jigsaw/*/*/*_best.pth",
    "max_jigsaw_checkpoints": None,
    "joint_checkpoints": {
        "joint_ssl": "models/joint_ssl/joint_ssl_20260705_161313/joint_ssl_20260705_161313_best.pth",
    },
    "fine_tune_modes": ["frozen", "unfreeze_last_1", "unfreeze_last_2", "unfreeze_last_3"],
    "epochs": 50,
    "batch_size": 8,
    "head_lr": 1e-3,
    "backbone_lr": 1e-5,
    "weight_decay": 1e-4,
    "head_dropout": 0.1,
    "backbone_dropout": 0.15,
    "nhead": 8,
    "normalize_input": "auto",
    "joint_normalize_input": "checkpoint",
    "xgb_jobs": 4,
    "device": "auto",
    "seed": 42,
    "max_folds": None,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_barth(
    data_path: str | Path,
    metadata_path: str | Path,
    label_column: str,
    exclude_labels: list[str],
):
    spectra = np.load(data_path).astype(np.float32)
    with Path(metadata_path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No metadata rows found in {metadata_path}")
    if label_column not in rows[0]:
        raise ValueError(f"{metadata_path} does not contain label column {label_column!r}")

    excluded = {str(label).strip() for label in exclude_labels}
    used = []
    for i, row in enumerate(rows):
        label = str(row.get(label_column, "")).strip()
        if not label or label in excluded:
            continue
        npy_row = int(row.get("npy_row", i))
        if not 0 <= npy_row < len(spectra):
            raise IndexError(f"npy_row {npy_row} is outside spectra array with {len(spectra)} rows")
        row = dict(row)
        row["npy_row"] = str(npy_row)
        row.setdefault("Sample Name", row.get("sample_folder", row.get("label_source_id", "")))
        used.append((npy_row, label, row))

    if not used:
        raise ValueError("No usable Barth rows after label filtering")
    used.sort(key=lambda item: item[0])

    labels_present = sorted({label for _, label, _ in used})
    if set(labels_present) == {"Case", "Control"}:
        label_names = ["Control", "Case"]
    else:
        label_names = labels_present
    label_to_index = {name: i for i, name in enumerate(label_names)}
    if any(label not in label_to_index for _, label, _ in used):
        raise ValueError(f"Unexpected labels after mapping: {labels_present}")

    indices = np.asarray([item[0] for item in used], dtype=np.int64)
    labels = np.asarray([label_to_index[item[1]] for item in used], dtype=np.int64)
    metadata = [item[2] for item in used]
    return spectra[indices], labels, metadata, label_names


def safe_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, label_names: list[str]):
    n_classes = len(label_names)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    if n_classes == 2:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob[:, 1]))
        except ValueError:
            roc_auc = float("nan")
        try:
            pr_auc = float(average_precision_score(y_true, y_prob[:, 1]))
        except ValueError:
            pr_auc = float("nan")
        metrics.update(
            {
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            }
        )
    else:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob, average="macro", multi_class="ovr"))
        except ValueError:
            roc_auc = float("nan")
        metrics.update(
            {
                "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
                "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
                "macro_roc_auc_ovr": roc_auc,
                "confusion_matrix": json.dumps(cm.tolist()),
            }
        )
    return metrics


def finalize_result(labels, label_names, predictions, probabilities, **extra):
    evaluated = (predictions >= 0) & np.isfinite(probabilities).all(axis=1)
    if not np.any(evaluated):
        raise RuntimeError("No folds produced predictions")
    result = {
        "predictions": predictions,
        "probabilities": probabilities,
        "metrics": safe_metrics(
            labels[evaluated],
            predictions[evaluated],
            probabilities[evaluated],
            label_names,
        ),
        "n_evaluated": int(evaluated.sum()),
        **extra,
    }
    return result


def run_classical_loocv(features, labels, label_names, args):
    results = {}
    n_classes = len(label_names)
    for name, estimator in classical_models(args.seed, args.xgb_jobs, n_classes).items():
        predictions = np.full(len(labels), -1, dtype=np.int64)
        probabilities = np.full((len(labels), n_classes), np.nan, dtype=np.float64)
        for fold, (train_idx, test_idx) in enumerate(LeaveOneOut().split(features), 1):
            if args.max_folds is not None and fold > int(args.max_folds):
                break
            model = clone(estimator)
            model.fit(features[train_idx], labels[train_idx])
            predictions[test_idx] = model.predict(features[test_idx])
            probabilities[test_idx] = probability_matrix(model, features[test_idx], n_classes)
            print(f"\rclassical/{name}: LOOCV fold {fold}/{len(labels)}", end="", flush=True)
        print()
        results[name] = finalize_result(labels, label_names, predictions, probabilities)
    return results


def checkpoint_state(checkpoint_path: str | Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def infer_mae_config(state, nhead: int, dropout: float):
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
        raise ValueError(f"d_model={config['d_model']} is not divisible by nhead={config['nhead']}")
    return config


class MaskedMAEClassifier(nn.Module):
    def __init__(self, backbone, d_model: int, n_classes: int, head_dropout: float):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(head_dropout),
            nn.Linear(d_model, n_classes),
        )
        self.softmax = nn.Softmax(dim=1)
        self.unfreeze_layers = 0

    def forward(self, x, return_logits: bool = False):
        _, encoded = self.backbone(x, mask=None)
        logits = self.classifier(encoded.mean(dim=1))
        return logits if return_logits else self.softmax(logits)


def build_masked_classifier(state, spectrum_length, n_classes, args, device, unfreeze_layers):
    config = infer_mae_config(state, args.nhead, args.backbone_dropout)
    backbone = NMRMaskedAutoencoder(spectrum_length=spectrum_length, **config)
    backbone.load_state_dict(state, strict=True)
    model = MaskedMAEClassifier(backbone, config["d_model"], n_classes, args.head_dropout)
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


def train_classifier_one_fold(
    model,
    x_train,
    y_train,
    device,
    epochs,
    batch_size,
    head_lr,
    backbone_lr,
    weight_decay,
    seed,
    layer_getter,
):
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
            for layer in list(layer_getter(model))[-model.unfreeze_layers:]:
                layer.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb.to(device), return_logits=True)
            loss = loss_fn(logits, yb.to(device))
            loss.backward()
            optimizer.step()


def run_masking_loocv(spectra, labels, label_names, checkpoint_map, args, device):
    results = {}
    n_classes = len(label_names)
    for checkpoint_label, checkpoint_path in checkpoint_map.items():
        state = checkpoint_state(checkpoint_path)
        for mode in args.fine_tune_modes:
            unfreeze_count = fine_tune_count(mode)
            result_name = f"{checkpoint_label}_{mode}"
            predictions = np.full(len(labels), -1, dtype=np.int64)
            probabilities = np.full((len(labels), n_classes), np.nan, dtype=np.float64)
            config = None
            for fold, (train_idx, test_idx) in enumerate(LeaveOneOut().split(spectra), 1):
                if args.max_folds is not None and fold > int(args.max_folds):
                    break
                set_seed(args.seed + fold)
                model, config = build_masked_classifier(
                    state, spectra.shape[1], n_classes, args, device, unfreeze_count
                )
                train_classifier_one_fold(
                    model,
                    spectra[train_idx],
                    labels[train_idx],
                    device,
                    args.epochs,
                    args.batch_size,
                    args.head_lr,
                    args.backbone_lr,
                    args.weight_decay,
                    args.seed + fold,
                    lambda m: m.backbone.encoder.transformer.layers,
                )
                model.eval()
                with torch.no_grad():
                    prob = model(torch.from_numpy(spectra[test_idx]).to(device)).cpu().numpy()
                probabilities[test_idx] = prob
                predictions[test_idx] = np.argmax(prob, axis=1)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"\rmasking/{result_name}: LOOCV fold {fold}/{len(labels)}", end="", flush=True)
            print()
            results[result_name] = finalize_result(
                labels,
                label_names,
                predictions,
                probabilities,
                checkpoint=str(checkpoint_path),
                unfrozen_transformer_layers=unfreeze_count,
                backbone_config=config,
            )
    return results


def normalize_batch(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    x[~np.isfinite(x)] = 0.0
    lo = x.min(axis=1, keepdims=True)
    hi = x.max(axis=1, keepdims=True)
    denom = hi - lo
    out = x.copy()
    mask = denom[:, 0] > 1e-8
    out[mask] = (x[mask] - lo[mask]) / denom[mask]
    return out


def resolve_normalize_mode(mode: str, spectra: np.ndarray) -> bool:
    if mode == "true":
        return True
    if mode == "false":
        return False
    if mode != "auto":
        raise ValueError("--normalize-input must be auto, true, or false")
    finite = spectra[np.isfinite(spectra)]
    if finite.size == 0:
        return True
    return float(finite.min()) < -1e-4 or float(finite.max()) > 1.5


def natural_bins(x: np.ndarray, bin_size: int, normalize_input: bool):
    x = normalize_batch(x) if normalize_input else x.astype(np.float32, copy=True)
    x[~np.isfinite(x)] = 0.0
    trimmed_length = (x.shape[1] // bin_size) * bin_size
    if trimmed_length <= 0:
        raise ValueError(f"Bin size {bin_size} is too large for spectra length {x.shape[1]}")
    return torch.from_numpy(x[:, :trimmed_length].reshape(len(x), trimmed_length // bin_size, bin_size).copy())


def checkpoint_label_from_path(path: str | Path):
    text = str(path)
    if "multibin" in text:
        return "jigsaw_multibin"
    for size in (256, 512, 1024, 2048):
        if f"bin_{size}" in text:
            return f"jigsaw_bin_{size}"
    return Path(path).stem


def checkpoint_label_from_bin_sizes(bin_sizes: list[int]) -> str:
    if len(bin_sizes) == 1:
        return f"jigsaw_bin_{bin_sizes[0]}"
    return "jigsaw_multibin"


def discover_jigsaw_checkpoints(pattern: str, max_checkpoints: int | None = None):
    paths = sorted(Path(p) for p in glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No jigsaw checkpoints matched: {pattern}")
    grouped = {}
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        label = checkpoint_label_from_bin_sizes([int(b) for b in checkpoint["bin_sizes"]])
        if label not in grouped or path.stat().st_mtime > grouped[label].stat().st_mtime:
            grouped[label] = path
    selected = sorted(grouped.values(), key=lambda p: checkpoint_label_from_path(p))
    if max_checkpoints is not None:
        selected = selected[: int(max_checkpoints)]
    return selected


def load_jigsaw_checkpoint(checkpoint_path: str | Path, device):
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


class JigsawClassifier(nn.Module):
    def __init__(self, backbone, bin_sizes, d_model, n_classes, head_dropout, normalize_input):
        super().__init__()
        self.backbone = backbone
        self.bin_sizes = [int(b) for b in bin_sizes]
        self.normalize_input = bool(normalize_input)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * len(self.bin_sizes)),
            nn.Dropout(head_dropout),
            nn.Linear(d_model * len(self.bin_sizes), n_classes),
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
        return torch.cat([self.encode_one_bin_size(x, size) for size in self.bin_sizes], dim=1)

    def forward(self, x, return_logits: bool = False):
        logits = self.classifier(self.encode(x))
        return logits if return_logits else self.softmax(logits)


def build_jigsaw_classifier(checkpoint_path, spectra, n_classes, args, device, unfreeze_layers):
    backbone, checkpoint = load_jigsaw_checkpoint(checkpoint_path, device)
    hp = checkpoint.get("hyperparameters", {})
    bin_sizes = [int(b) for b in checkpoint["bin_sizes"]]
    normalize_input = resolve_normalize_mode(args.normalize_input, spectra)
    model = JigsawClassifier(
        backbone,
        bin_sizes,
        int(hp.get("d_model", 192)),
        n_classes,
        args.head_dropout,
        normalize_input,
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
    config = {
        "bin_sizes": bin_sizes,
        "spectrum_length": int(checkpoint["spectrum_length"]),
        "d_model": int(hp.get("d_model", 192)),
        "nhead": int(hp.get("nhead", 6)),
        "num_layers": int(hp.get("num_layers", 4)),
        "dim_feedforward": int(hp.get("dim_feedforward", 768)),
        "dropout": float(hp.get("dropout", 0.15)),
        "normalize_input": normalize_input,
    }
    return model.to(device), config


def run_jigsaw_loocv(spectra, labels, label_names, checkpoint_map, args, device):
    results = {}
    n_classes = len(label_names)
    for base_label, checkpoint_path in checkpoint_map.items():
        for mode in args.fine_tune_modes:
            unfreeze_count = fine_tune_count(mode)
            result_name = f"{base_label}_{mode}"
            predictions = np.full(len(labels), -1, dtype=np.int64)
            probabilities = np.full((len(labels), n_classes), np.nan, dtype=np.float64)
            config = None
            for fold, (train_idx, test_idx) in enumerate(LeaveOneOut().split(spectra), 1):
                if args.max_folds is not None and fold > int(args.max_folds):
                    break
                set_seed(args.seed + fold)
                model, config = build_jigsaw_classifier(
                    checkpoint_path, spectra, n_classes, args, device, unfreeze_count
                )
                train_classifier_one_fold(
                    model,
                    spectra[train_idx],
                    labels[train_idx],
                    device,
                    args.epochs,
                    args.batch_size,
                    args.head_lr,
                    args.backbone_lr,
                    args.weight_decay,
                    args.seed + fold,
                    lambda m: m.backbone.transformer.layers,
                )
                model.eval()
                with torch.no_grad():
                    prob = model(torch.from_numpy(spectra[test_idx]).to(device)).cpu().numpy()
                probabilities[test_idx] = prob
                predictions[test_idx] = np.argmax(prob, axis=1)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"\rjigsaw/{result_name}: LOOCV fold {fold}/{len(labels)}", end="", flush=True)
            print()
            results[result_name] = finalize_result(
                labels,
                label_names,
                predictions,
                probabilities,
                checkpoint=str(checkpoint_path),
                unfrozen_transformer_layers=unfreeze_count,
                backbone_config=config,
            )
    return results


def run_joint_ssl_loocv_multi(spectra, labels, label_names, checkpoint_map, args, device):
    results = {}
    n_classes = len(label_names)
    for checkpoint_label, checkpoint_path in checkpoint_map.items():
        for mode in args.fine_tune_modes:
            unfreeze_count = fine_tune_count(mode)
            result_name = f"{checkpoint_label}_{mode}"
            predictions = np.full(len(labels), -1, dtype=np.int64)
            probabilities = np.full((len(labels), n_classes), np.nan, dtype=np.float64)
            config = None
            normalized_spectra = None
            for fold, (train_idx, test_idx) in enumerate(LeaveOneOut().split(spectra), 1):
                if args.max_folds is not None and fold > int(args.max_folds):
                    break
                set_seed(args.seed + fold)
                model, config = build_joint_classifier(
                    checkpoint_path=checkpoint_path,
                    spectra=spectra,
                    n_classes=n_classes,
                    head_dropout=args.head_dropout,
                    normalize_input_mode=args.joint_normalize_input,
                    unfreeze_layers=unfreeze_count,
                    device=device,
                )
                if normalized_spectra is None:
                    normalized_spectra = maybe_normalize_eval_spectra(spectra, config["normalize_input"])
                train_joint_one_fold(
                    model,
                    normalized_spectra[train_idx],
                    labels[train_idx],
                    device,
                    args.epochs,
                    args.batch_size,
                    args.head_lr,
                    args.backbone_lr,
                    args.weight_decay,
                    args.seed + fold,
                )
                model.eval()
                with torch.no_grad():
                    prob = model(torch.from_numpy(normalized_spectra[test_idx]).to(device)).cpu().numpy()
                probabilities[test_idx] = prob
                predictions[test_idx] = np.argmax(prob, axis=1)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"\rjoint_ssl/{result_name}: LOOCV fold {fold}/{len(labels)}", end="", flush=True)
            print()
            results[result_name] = finalize_result(
                labels,
                label_names,
                predictions,
                probabilities,
                checkpoint=str(checkpoint_path),
                unfrozen_transformer_layers=unfreeze_count,
                backbone_config=config,
            )
    return results


def parse_key_value_paths(items: list[str]) -> dict[str, str]:
    parsed = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected LABEL=PATH, got {item!r}")
        label, path = item.split("=", 1)
        parsed[label.strip()] = path.strip()
    return parsed


def normalize_path_mapping(value, label_from_path):
    if isinstance(value, dict):
        return {str(label): str(path) for label, path in value.items()}
    if isinstance(value, (list, tuple)):
        return {label_from_path(path): str(path) for path in value}
    raise TypeError(f"Expected dict or list of paths, got {type(value).__name__}")


def selected_families(args) -> set[str]:
    families = set(args.families)
    if "all" in families:
        return {"classical", "masking", "jigsaw", "joint_ssl"}
    return families


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=IDE_CONFIG["data"])
    parser.add_argument("--metadata", default=IDE_CONFIG["metadata"])
    parser.add_argument("--output-dir", default=IDE_CONFIG["output_dir"])
    parser.add_argument("--label-column", default=IDE_CONFIG["label_column"])
    parser.add_argument("--exclude-labels", nargs="*", default=IDE_CONFIG["exclude_labels"])
    parser.add_argument(
        "--families",
        nargs="+",
        choices=["all", "classical", "masking", "jigsaw", "joint_ssl"],
        default=IDE_CONFIG["families"],
    )
    parser.add_argument("--classical-features", choices=["binned_auc", "raw"], default=IDE_CONFIG["classical_features"])
    parser.add_argument("--feature-bins", type=int, default=IDE_CONFIG["feature_bins"])
    parser.add_argument("--masking-checkpoint", action="append", default=[])
    parser.add_argument("--jigsaw-checkpoint", action="append", default=[])
    parser.add_argument("--jigsaw-checkpoint-glob", default=IDE_CONFIG["jigsaw_checkpoint_glob"])
    parser.add_argument("--max-jigsaw-checkpoints", type=int, default=IDE_CONFIG["max_jigsaw_checkpoints"])
    parser.add_argument("--joint-checkpoint", action="append", default=[])
    parser.add_argument("--fine-tune-modes", nargs="+", choices=FINE_TUNE_CHOICES, default=IDE_CONFIG["fine_tune_modes"])
    parser.add_argument("--epochs", type=int, default=IDE_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=IDE_CONFIG["batch_size"])
    parser.add_argument("--head-lr", type=float, default=IDE_CONFIG["head_lr"])
    parser.add_argument("--backbone-lr", type=float, default=IDE_CONFIG["backbone_lr"])
    parser.add_argument("--weight-decay", type=float, default=IDE_CONFIG["weight_decay"])
    parser.add_argument("--head-dropout", type=float, default=IDE_CONFIG["head_dropout"])
    parser.add_argument("--backbone-dropout", type=float, default=IDE_CONFIG["backbone_dropout"])
    parser.add_argument("--nhead", type=int, default=IDE_CONFIG["nhead"])
    parser.add_argument("--normalize-input", choices=["auto", "true", "false"], default=IDE_CONFIG["normalize_input"])
    parser.add_argument(
        "--joint-normalize-input",
        choices=["checkpoint", "auto", "true", "false"],
        default=IDE_CONFIG["joint_normalize_input"],
    )
    parser.add_argument("--xgb-jobs", type=int, default=IDE_CONFIG["xgb_jobs"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=IDE_CONFIG["device"])
    parser.add_argument("--seed", type=int, default=IDE_CONFIG["seed"])
    parser.add_argument("--max-folds", type=int, default=IDE_CONFIG["max_folds"])

    if USE_IDE_CONFIG:
        unknown = set(IDE_CONFIG) - {action.dest for action in parser._actions} - {
            "masking_checkpoints",
            "jigsaw_checkpoints",
            "joint_checkpoints",
        }
        if unknown:
            parser.error(f"Unknown IDE_CONFIG entries: {sorted(unknown)}")
        args = parser.parse_args([])
        for name, value in IDE_CONFIG.items():
            if name in {"masking_checkpoints", "jigsaw_checkpoints", "joint_checkpoints"}:
                continue
            setattr(args, name, value)
        args.masking_checkpoint = [f"{k}={v}" for k, v in IDE_CONFIG["masking_checkpoints"].items()]
        args.jigsaw_checkpoint = []
        args.joint_checkpoint = [f"{k}={v}" for k, v in IDE_CONFIG["joint_checkpoints"].items()]
        print("Using settings from IDE_CONFIG")
    else:
        args = parser.parse_args()
        if not args.masking_checkpoint:
            args.masking_checkpoint = [f"{k}={v}" for k, v in IDE_CONFIG["masking_checkpoints"].items()]
        if not args.joint_checkpoint:
            args.joint_checkpoint = [f"{k}={v}" for k, v in IDE_CONFIG["joint_checkpoints"].items()]

    args.masking_checkpoints = parse_key_value_paths(args.masking_checkpoint)
    args.joint_checkpoints = parse_key_value_paths(args.joint_checkpoint)
    if args.jigsaw_checkpoint:
        args.jigsaw_checkpoints = normalize_path_mapping(args.jigsaw_checkpoint, checkpoint_label_from_path)
    else:
        args.jigsaw_checkpoints = normalize_path_mapping(
            IDE_CONFIG["jigsaw_checkpoints"], checkpoint_label_from_path
        )
    return args


def save_results(output_dir, metadata, labels, label_names, label_column, families, run_config):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    prediction_rows = []

    for i, row in enumerate(metadata):
        prediction_rows.append(
            {
                "npy_row": row.get("npy_row", ""),
                "sample_id": row.get("label_source_id", row.get("sample_folder", "")),
                "sample_name": row.get("Sample Name", row.get("sample_folder", "")),
                "label_column": label_column,
                "label": label_names[int(labels[i])],
                "target": int(labels[i]),
            }
        )

    for family, models in families.items():
        for model_name, result in models.items():
            summary_rows.append(
                {
                    "family": family,
                    "model": model_name,
                    "n_evaluated": result.get("n_evaluated", len(labels)),
                    **result["metrics"],
                }
            )
            np.save(output_dir / f"{family}_{model_name}_oof_pred.npy", result["predictions"])
            np.save(output_dir / f"{family}_{model_name}_oof_prob.npy", result["probabilities"])
            for i, pred in enumerate(result["predictions"]):
                prefix = f"{family}_{model_name}"
                prediction_rows[i][f"{prefix}_prediction"] = (
                    label_names[int(pred)] if int(pred) >= 0 else ""
                )
                for class_idx, class_name in enumerate(label_names):
                    safe_name = class_name.lower().replace(" ", "_").replace("/", "_")
                    prob = result["probabilities"][i, class_idx]
                    prediction_rows[i][f"{prefix}_prob_{safe_name}"] = (
                        float(prob) if np.isfinite(prob) else ""
                    )

    if not summary_rows:
        raise ValueError("No models were evaluated; nothing to save.")

    fieldnames = sorted({key for row in summary_rows for key in row})
    preferred = ["family", "model", "n_evaluated", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    fieldnames = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    prediction_fields = sorted({key for row in prediction_rows for key in row})
    preferred_prediction = ["npy_row", "sample_id", "sample_name", "label_column", "label", "target"]
    prediction_fields = [key for key in preferred_prediction if key in prediction_fields] + [
        key for key in prediction_fields if key not in preferred_prediction
    ]
    with (output_dir / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(prediction_rows)

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)


def main():
    args = parse_args()
    set_seed(args.seed)
    spectra, labels, metadata, label_names = load_barth(
        args.data, args.metadata, args.label_column, args.exclude_labels
    )
    counts = {name: int((labels == i).sum()) for i, name in enumerate(label_names)}
    print(
        f"Loaded {spectra.shape}; label_column={args.label_column}; "
        f"labels={dict(enumerate(label_names))}; counts={counts}"
    )

    families_to_run = selected_families(args)
    families = {}

    if "classical" in families_to_run:
        features = binned_abs_area(spectra, args.feature_bins) if args.classical_features == "binned_auc" else spectra
        families["classical"] = run_classical_loocv(features, labels, label_names, args)

    foundation_families = {"masking", "jigsaw", "joint_ssl"} & families_to_run
    if foundation_families:
        device = choose_device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        print(f"Foundation-model device: {device}")

    if "masking" in families_to_run:
        print("Masking checkpoints:")
        for label, path in args.masking_checkpoints.items():
            print(f"  - {label}: {path}")
        families["masking"] = run_masking_loocv(
            spectra, labels, label_names, args.masking_checkpoints, args, device
        )

    if "jigsaw" in families_to_run:
        if args.jigsaw_checkpoints:
            jigsaw_paths = {label: Path(path) for label, path in args.jigsaw_checkpoints.items()}
        else:
            jigsaw_paths = {
                checkpoint_label_from_path(path): path
                for path in discover_jigsaw_checkpoints(args.jigsaw_checkpoint_glob, args.max_jigsaw_checkpoints)
            }
        print("Jigsaw checkpoints:")
        for label, path in jigsaw_paths.items():
            print(f"  - {label}: {path}")
        families["jigsaw"] = run_jigsaw_loocv(
            spectra, labels, label_names, jigsaw_paths, args, device
        )

    if "joint_ssl" in families_to_run:
        print("Joint SSL checkpoints:")
        for label, path in args.joint_checkpoints.items():
            print(f"  - {label}: {path}")
        families["joint_ssl"] = run_joint_ssl_loocv_multi(
            spectra, labels, label_names, args.joint_checkpoints, args, device
        )

    run_config = vars(args).copy()
    run_config.update(
        {
            "n_samples": int(len(labels)),
            "spectrum_length": int(spectra.shape[1]),
            "label_mapping": {str(i): name for i, name in enumerate(label_names)},
        }
    )
    save_results(args.output_dir, metadata, labels, label_names, args.label_column, families, run_config)

    print(f"\nResults written to {args.output_dir}/summary.csv")
    for family, models in families.items():
        for name, result in models.items():
            m = result["metrics"]
            details = (
                f"{family}/{name}: balanced_accuracy={m['balanced_accuracy']:.3f}, "
                f"weighted_f1={m['weighted_f1']:.3f}"
            )
            if "roc_auc" in m and np.isfinite(m["roc_auc"]):
                details += f", ROC-AUC={m['roc_auc']:.3f}"
            print(details)


if __name__ == "__main__":
    main()
