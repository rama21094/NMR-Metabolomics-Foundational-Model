#!/usr/bin/env python3
"""LOOCV classification for the BrC/T2D dataset using masking foundation models.

Runs classical baselines and a pretrained masked-autoencoder backbone with a
softmax classification head. The selected metadata label can be binary
(`cancer_status`, `diabetes_status`) or 4-class (`combined_status`).
"""

from __future__ import annotations

import argparse
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
from trainer_revised import NMRMaskedAutoencoder  # noqa: E402


FINE_TUNE_CHOICES = ("frozen", "unfreeze_last_1", "unfreeze_last_2", "unfreeze_last_3")
DEFAULT_DATA = "data/BrC_T2D/BC_T2D_aligned_spectra_WS625to680Zero_rowMinMax.npy"
DEFAULT_METADATA = "data/BrC_T2D/BC_T2D_metadata_mapping.csv"
DEFAULT_CHECKPOINT = "models/SSL_models/combined_unique_WS625to680Zero_20260601_084533_bs32_mr0.50_ps1024_best.pth"
DEFAULT_OUTPUT_BASE = "results/loocv/brc_t2d_masking"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_state(checkpoint_path: str | Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def infer_backbone_config(state, nhead: int, dropout: float):
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
    """Pretrained MAE encoder, mean pooling, and n-class softmax head."""

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


def build_foundation_model(
    state,
    spectrum_length: int,
    n_classes: int,
    nhead: int,
    backbone_dropout: float,
    head_dropout: float,
    unfreeze_layers: int,
    device: torch.device,
):
    config = infer_backbone_config(state, nhead, backbone_dropout)
    backbone = NMRMaskedAutoencoder(spectrum_length=spectrum_length, **config)
    backbone.load_state_dict(state, strict=True)
    model = SoftmaxMAEClassifier(backbone, config["d_model"], n_classes, head_dropout)

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
            for layer in list(model.backbone.encoder.transformer.layers)[-model.unfreeze_layers:]:
                layer.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb.to(device), return_logits=True)
            loss = loss_fn(logits, yb.to(device))
            loss.backward()
            optimizer.step()


def run_foundation_loocv(spectra, labels, label_names, checkpoint_path, args, device):
    state = checkpoint_state(checkpoint_path)
    n_classes = len(label_names)
    results = {}

    for mode in args.fine_tune_modes:
        unfreeze_count = fine_tune_count(mode)
        predictions = np.empty(len(labels), dtype=np.int64)
        probabilities = np.empty((len(labels), n_classes), dtype=np.float64)
        config = None

        for fold, (train_idx, test_idx) in enumerate(LeaveOneOut().split(spectra), 1):
            set_seed(args.seed + fold)
            model, config = build_foundation_model(
                state=state,
                spectrum_length=spectra.shape[1],
                n_classes=n_classes,
                nhead=args.nhead,
                backbone_dropout=args.backbone_dropout,
                head_dropout=args.head_dropout,
                unfreeze_layers=unfreeze_count,
                device=device,
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
            print(f"\rfoundation/{mode}: LOOCV fold {fold}/{len(labels)}", end="", flush=True)
        print()

        results[mode] = {
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
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--classical-only", action="store_true")
    parser.add_argument("--foundation-only", action="store_true")
    parser.add_argument("--classical-features", choices=["binned_auc", "raw"], default="binned_auc")
    parser.add_argument("--feature-bins", type=int, default=1024)
    parser.add_argument("--fine-tune-modes", nargs="+", choices=FINE_TUNE_CHOICES, default=list(FINE_TUNE_CHOICES))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--backbone-dropout", type=float, default=0.15)
    parser.add_argument("--nhead", type=int, default=8, help="Attention heads used during pretraining")
    parser.add_argument("--xgb-jobs", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.classical_only and args.foundation_only:
        parser.error("--classical-only and --foundation-only are mutually exclusive")
    if not args.classical_only and not args.checkpoint:
        parser.error("--checkpoint is required unless --classical-only is used")
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
    if not args.foundation_only:
        features = binned_abs_area(spectra, args.feature_bins) if args.classical_features == "binned_auc" else spectra
        families["classical"] = run_classical_loocv(features, labels, label_names, args.seed, args.xgb_jobs)

    if not args.classical_only:
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        print(f"Foundation-model device: {device}")
        families["foundation"] = run_foundation_loocv(
            spectra, labels, label_names, args.checkpoint, args, device
        )

    run_config = vars(args).copy()
    run_config.update(
        {
            "n_samples": int(len(labels)),
            "spectrum_length": int(spectra.shape[1]),
            "label_mapping": {str(i): name for i, name in enumerate(label_names)},
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
