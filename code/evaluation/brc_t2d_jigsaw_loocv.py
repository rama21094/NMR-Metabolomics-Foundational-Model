#!/usr/bin/env python3
"""LOOCV classification for the BrC/T2D dataset using jigsaw foundation models.

Runs classical baselines and frozen/fine-tuned jigsaw checkpoints. The selected
metadata label can be binary (`cancer_status`, `diabetes_status`) or 4-class
(`combined_status`).
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("NUMEXPR_MAX_THREADS", "256")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import LeaveOneOut
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "code" / "training"
for path in (ROOT, TRAINING_DIR, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from brc_t2d_common import (  # noqa: E402
    LABEL_MAPPINGS,
    aggregate_metrics,
    binned_abs_area,
    default_output_dir,
    load_brc_t2d,
    run_classical_loocv,
    save_results,
)
from train_jigsaw_spectra import JigsawNMRModel  # noqa: E402


FINE_TUNE_CHOICES = ("frozen", "unfreeze_last_1", "unfreeze_last_2", "unfreeze_last_3")
DEFAULT_DATA = "data/BrC_T2D/BC_T2D_aligned_spectra_WS625to680Zero_rowMinMax.npy"
DEFAULT_METADATA = "data/BrC_T2D/BC_T2D_metadata_mapping.csv"
DEFAULT_OUTPUT_BASE = "results/loocv/brc_t2d_jigsaw"
DEFAULT_CHECKPOINT_GLOB = "models/jigsaw/*/*/*_best.pth"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_batch(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    finite = np.isfinite(x)
    if not np.all(finite):
        x[~finite] = 0.0
    lo = x.min(axis=1, keepdims=True)
    hi = x.max(axis=1, keepdims=True)
    denom = hi - lo
    mask = denom[:, 0] > 1e-8
    out = x.copy()
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
    sampled_min = float(finite.min())
    sampled_max = float(finite.max())
    resolved = sampled_min < -1e-4 or sampled_max > 1.5
    print(f"Jigsaw normalization auto-check: range [{sampled_min:.4g}, {sampled_max:.4g}] -> {resolved}")
    return resolved


def natural_bins(x: np.ndarray, bin_size: int, normalize_input: bool) -> torch.Tensor:
    if normalize_input:
        x = normalize_batch(x)
    else:
        x = x.astype(np.float32, copy=True)
        x[~np.isfinite(x)] = 0.0
    trimmed_length = (x.shape[1] // bin_size) * bin_size
    if trimmed_length <= 0:
        raise ValueError(f"Bin size {bin_size} is too large for spectra length {x.shape[1]}")
    bins = x[:, :trimmed_length].reshape(len(x), trimmed_length // bin_size, bin_size)
    return torch.from_numpy(bins.copy())


def checkpoint_label_from_path(path: str | Path) -> str:
    text = str(path)
    if "multibin" in text:
        return "jigsaw_multibin"
    for size in (256, 512, 1024, 2048):
        if f"bin_{size}" in text:
            return f"jigsaw_bin_{size}"
    return Path(path).stem


def checkpoint_label_from_bins(bin_sizes: list[int]) -> str:
    if len(bin_sizes) == 1:
        return f"jigsaw_bin_{bin_sizes[0]}"
    return "jigsaw_multibin"


def discover_checkpoints(pattern: str, max_checkpoints: int | None = None) -> list[Path]:
    paths = sorted(Path(p) for p in glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No jigsaw checkpoints matched: {pattern}")

    grouped: dict[str, Path] = {}
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        bin_sizes = [int(b) for b in checkpoint["bin_sizes"]]
        label = checkpoint_label_from_bins(bin_sizes)
        if label not in grouped or path.stat().st_mtime > grouped[label].stat().st_mtime:
            grouped[label] = path

    order = {
        "jigsaw_bin_256": 0,
        "jigsaw_bin_512": 1,
        "jigsaw_bin_1024": 2,
        "jigsaw_bin_2048": 3,
        "jigsaw_multibin": 4,
    }
    selected = sorted(grouped.values(), key=lambda p: order.get(checkpoint_label_from_path(p), 99))
    if max_checkpoints is not None:
        selected = selected[: int(max_checkpoints)]
    return selected


def load_existing_results(output_dir: str | Path):
    """Load previously saved metrics and OOF arrays so new runs can append safely."""
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.csv"
    families = {}
    completed = set()
    if not summary_path.exists():
        return families, completed

    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        family = row.pop("family")
        model = row.pop("model")
        pred_path = output_dir / f"{family}_{model}_oof_pred.npy"
        prob_path = output_dir / f"{family}_{model}_oof_prob.npy"
        if not pred_path.exists() or not prob_path.exists():
            print(f"Skipping incomplete cached result {family}/{model}: missing OOF arrays")
            continue

        metrics = {}
        for key, value in row.items():
            if value == "":
                continue
            try:
                numeric = float(value)
                metrics[key] = int(numeric) if numeric.is_integer() and key in {"tn", "fp", "fn", "tp"} else numeric
            except ValueError:
                metrics[key] = value

        families.setdefault(family, {})[model] = {
            "predictions": np.load(pred_path),
            "probabilities": np.load(prob_path),
            "metrics": metrics,
        }
        completed.add((family, model))
    return families, completed


def load_jigsaw_checkpoint(checkpoint_path: str | Path, device: torch.device):
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
    """Frozen/fine-tuned jigsaw encoder with an n-class softmax head."""

    def __init__(self, backbone, bin_sizes, d_model: int, n_classes: int, head_dropout: float, normalize_input: bool):
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

    def encode_one_bin_size(self, x: torch.Tensor, bin_size: int) -> torch.Tensor:
        bins = natural_bins(x.detach().cpu().numpy(), bin_size, self.normalize_input).to(x.device)
        key = str(int(bin_size))
        encoded = self.backbone.input_projections[key](bins)
        positions = torch.arange(encoded.shape[1], device=encoded.device)
        encoded = encoded + self.backbone.slot_embedding(positions).unsqueeze(0)
        encoded = self.backbone.transformer(encoded)
        return encoded.mean(dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pooled = [self.encode_one_bin_size(x, bin_size) for bin_size in self.bin_sizes]
        return torch.cat(pooled, dim=1)

    def forward(self, x, return_logits: bool = False):
        logits = self.classifier(self.encode(x))
        return logits if return_logits else self.softmax(logits)


def build_jigsaw_classifier(checkpoint_path, spectra, n_classes, args, device, unfreeze_layers):
    backbone, checkpoint = load_jigsaw_checkpoint(checkpoint_path, device)
    bin_sizes = [int(b) for b in checkpoint["bin_sizes"]]
    hp = checkpoint.get("hyperparameters", {})
    normalize_input = resolve_normalize_mode(args.normalize_input, spectra)
    model = JigsawSoftmaxClassifier(
        backbone=backbone,
        bin_sizes=bin_sizes,
        d_model=int(hp.get("d_model", 192)),
        n_classes=n_classes,
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


def fine_tune_count(mode: str) -> int:
    if mode == "frozen":
        return 0
    return int(mode.rsplit("_", 1)[-1])


def train_one_fold(
    model,
    x_train,
    y_train,
    device,
    epochs: int,
    batch_size: int,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
    seed: int,
) -> None:
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


def run_jigsaw_loocv(spectra, labels, label_names, checkpoint_paths, args, device):
    results = {}
    n_classes = len(label_names)
    completed = getattr(args, "_completed_results", set())

    for checkpoint_path in checkpoint_paths:
        base_label = checkpoint_label_from_path(checkpoint_path)
        for mode in args.fine_tune_modes:
            unfreeze_count = fine_tune_count(mode)
            result_name = f"{base_label}_{mode}"
            if args.skip_completed and ("jigsaw", result_name) in completed:
                print(f"jigsaw/{result_name}: already present in summary.csv; skipping")
                continue
            predictions = np.empty(len(labels), dtype=np.int64)
            probabilities = np.empty((len(labels), n_classes), dtype=np.float64)
            config = None

            for fold, (train_idx, test_idx) in enumerate(LeaveOneOut().split(spectra), 1):
                set_seed(args.seed + fold)
                model, config = build_jigsaw_classifier(
                    checkpoint_path, spectra, n_classes, args, device, unfreeze_count
                )
                train_one_fold(
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
                )
                model.eval()
                with torch.no_grad():
                    probability = model(torch.from_numpy(spectra[test_idx]).to(device))[0].cpu().numpy()
                probabilities[test_idx] = probability
                predictions[test_idx] = int(np.argmax(probability))
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"\rjigsaw/{result_name}: LOOCV fold {fold}/{len(labels)}", end="", flush=True)
            print()

            results[result_name] = {
                "predictions": predictions,
                "probabilities": probabilities,
                "metrics": aggregate_metrics(labels, predictions, probabilities, label_names),
                "checkpoint": str(checkpoint_path),
                "unfrozen_transformer_layers": unfreeze_count,
                "backbone_config": config,
            }
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--label-column", choices=sorted(LABEL_MAPPINGS), default="cancer_status")
    parser.add_argument("--checkpoint-glob", default=DEFAULT_CHECKPOINT_GLOB)
    parser.add_argument("--checkpoint", action="append", default=[], help="Explicit jigsaw checkpoint path. Can be repeated.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--classical-only", action="store_true")
    parser.add_argument("--jigsaw-only", action="store_true")
    parser.add_argument("--classical-features", choices=["binned_auc", "raw"], default="binned_auc")
    parser.add_argument("--feature-bins", type=int, default=1024)
    parser.add_argument("--fine-tune-modes", nargs="+", choices=FINE_TUNE_CHOICES, default=["frozen"])
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Load existing summary/OOF arrays and run only models not already present.",
    )
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
    if args.output_dir is None:
        args.output_dir = default_output_dir(DEFAULT_OUTPUT_BASE, args.label_column)
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    spectra, labels, metadata, label_names = load_brc_t2d(args.data, args.metadata, args.label_column)
    counts = {name: int((labels == i).sum()) for i, name in enumerate(label_names)}
    print(f"Loaded {spectra.shape}; label_column={args.label_column}; labels={dict(enumerate(label_names))}; counts={counts}")

    families = {}
    completed_results = set()
    if args.skip_completed:
        families, completed_results = load_existing_results(args.output_dir)
        args._completed_results = completed_results
        print(f"Loaded {len(completed_results)} completed result(s) from {Path(args.output_dir) / 'summary.csv'}")
    else:
        args._completed_results = set()

    if not args.jigsaw_only:
        missing_classical = [name for name in ("logistic_regression", "svm_rbf", "xgboost") if ("classical", name) not in completed_results]
        if args.skip_completed and not missing_classical:
            print("classical: all models already present in summary.csv; skipping")
        else:
            features = binned_abs_area(spectra, args.feature_bins) if args.classical_features == "binned_auc" else spectra
            families["classical"] = run_classical_loocv(features, labels, label_names, args.seed, args.xgb_jobs)

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
        new_jigsaw_results = run_jigsaw_loocv(
            spectra, labels, label_names, checkpoint_paths, args, device
        )
        if new_jigsaw_results:
            families.setdefault("jigsaw", {}).update(new_jigsaw_results)

    run_config = vars(args).copy()
    run_config.update(
        {
            "n_samples": int(len(labels)),
            "spectrum_length": int(spectra.shape[1]),
            "label_mapping": {str(i): name for i, name in enumerate(label_names)},
            "checkpoint_paths": [str(p) for p in checkpoint_paths],
        }
    )
    save_results(
        Path(args.output_dir),
        metadata,
        labels,
        label_names,
        args.label_column,
        families,
        run_config,
    )

    print(f"\nResults written to {args.output_dir}/summary.csv")
    for family, models in families.items():
        for name, result in models.items():
            m = result["metrics"]
            print(
                f"{family}/{name}: accuracy={m['accuracy']:.3f}, "
                f"balanced_accuracy={m['balanced_accuracy']:.3f}, "
                f"macro_f1={m['macro_f1']:.3f}"
            )


if __name__ == "__main__":
    main()
