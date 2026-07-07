"""Shared evaluation helpers for joint masked + multibin jigsaw SSL checkpoints."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import LeaveOneOut
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "code" / "training"
for path in (ROOT, TRAINING_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_joint_ssl import JointNMRSSLModel, load_joint_checkpoint, normalize_spectrum  # noqa: E402


FINE_TUNE_CHOICES = ("frozen", "unfreeze_last_1", "unfreeze_last_2", "unfreeze_last_3")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_normalize_mode(mode: str, spectra: np.ndarray, checkpoint: dict) -> bool:
    mode = mode.lower()
    hp = checkpoint.get("hyperparameters", {})
    if mode == "checkpoint":
        return bool(hp.get("normalize_resolved", hp.get("normalize_input", False)))
    if mode == "true":
        return True
    if mode == "false":
        return False
    if mode != "auto":
        raise ValueError("--normalize-input must be checkpoint, auto, true, or false")
    return bool(float(np.nanmin(spectra)) < -1e-4 or float(np.nanmax(spectra)) > 1.5)


def maybe_normalize_eval_spectra(spectra: np.ndarray, normalize_input: bool) -> np.ndarray:
    spectra = spectra.astype(np.float32, copy=True)
    spectra[~np.isfinite(spectra)] = 0.0
    if not normalize_input:
        return spectra
    return np.stack([normalize_spectrum(row) for row in spectra], axis=0).astype(np.float32)


class JointSSLSoftmaxClassifier(nn.Module):
    """Joint SSL encoder, natural-order multibin pooling, and a softmax head."""

    def __init__(
        self,
        backbone: JointNMRSSLModel,
        bin_sizes: list[int],
        n_classes: int,
        head_dropout: float,
        include_masked_task: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.bin_sizes = [int(b) for b in bin_sizes]
        self.include_masked_task = bool(include_masked_task)
        pooled_count = len(self.bin_sizes) + (1 if self.include_masked_task else 0)
        self.classifier = nn.Sequential(
            nn.LayerNorm(backbone.d_model * pooled_count),
            nn.Dropout(head_dropout),
            nn.Linear(backbone.d_model * pooled_count, n_classes),
        )
        self.softmax = nn.Softmax(dim=1)
        self.unfreeze_layers = 0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode_spectrum(x, self.bin_sizes, include_masked_task=self.include_masked_task)

    def forward(self, x: torch.Tensor, return_logits: bool = False):
        logits = self.classifier(self.encode(x))
        return logits if return_logits else self.softmax(logits)


def fine_tune_count(mode: str) -> int:
    if mode == "frozen":
        return 0
    return int(mode.rsplit("_", 1)[-1])


def build_joint_classifier(
    checkpoint_path: str | Path,
    spectra: np.ndarray,
    n_classes: int,
    head_dropout: float,
    normalize_input_mode: str,
    unfreeze_layers: int,
    device: torch.device,
    include_masked_task: bool = True,
):
    backbone, checkpoint = load_joint_checkpoint(checkpoint_path, device)
    normalize_input = resolve_normalize_mode(normalize_input_mode, spectra, checkpoint)
    bin_sizes = [int(b) for b in checkpoint.get("jigsaw_bin_sizes", backbone.jigsaw_bin_sizes)]
    model = JointSSLSoftmaxClassifier(backbone, bin_sizes, n_classes, head_dropout, include_masked_task=include_masked_task)

    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    layers = model.backbone.encoder_layers
    if unfreeze_layers > len(layers):
        raise ValueError(f"Cannot unfreeze {unfreeze_layers}; backbone has {len(layers)} layers")
    for layer in list(layers)[len(layers) - unfreeze_layers:] if unfreeze_layers else []:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    model.unfreeze_layers = unfreeze_layers
    config = {
        "bin_sizes": bin_sizes,
        "mask_bin_size": int(checkpoint.get("mask_bin_size", backbone.mask_bin_size)),
        "spectrum_length": int(checkpoint["spectrum_length"]),
        "d_model": int(backbone.d_model),
        "nhead": int(backbone.nhead),
        "num_layers": int(backbone.num_layers),
        "dim_feedforward": int(backbone.dim_feedforward),
        "dropout": float(backbone.dropout),
        "fourier_bands": int(backbone.fourier_bands),
        "normalize_input": bool(normalize_input),
        "include_masked_task": bool(include_masked_task),
    }
    return model.to(device), config


def train_one_fold(
    model: JointSSLSoftmaxClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    device: torch.device,
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
            for layer in list(model.backbone.encoder_layers)[-model.unfreeze_layers:]:
                layer.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb.to(device), return_logits=True)
            loss = loss_fn(logits, yb.to(device))
            loss.backward()
            optimizer.step()


def run_joint_ssl_loocv(
    spectra: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    checkpoint_path: str | Path,
    args,
    device: torch.device,
    metric_fn,
) -> dict:
    n_classes = len(label_names)
    results = {}

    for mode in args.fine_tune_modes:
        unfreeze_count = fine_tune_count(mode)
        predictions = np.full(len(labels), -1, dtype=np.int64)
        probabilities = np.full((len(labels), n_classes), np.nan, dtype=np.float64)
        config = None
        normalized_spectra = None

        max_folds = getattr(args, "max_folds", None)
        for fold, (train_idx, test_idx) in enumerate(LeaveOneOut().split(spectra), 1):
            if max_folds is not None and fold > int(max_folds):
                break
            set_seed(args.seed + fold)
            model, config = build_joint_classifier(
                checkpoint_path=checkpoint_path,
                spectra=spectra,
                n_classes=n_classes,
                head_dropout=args.head_dropout,
                normalize_input_mode=args.normalize_input,
                unfreeze_layers=unfreeze_count,
                device=device,
            )
            if normalized_spectra is None:
                normalized_spectra = maybe_normalize_eval_spectra(spectra, config["normalize_input"])
            train_one_fold(
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
                probability = model(torch.from_numpy(normalized_spectra[test_idx]).to(device))[0].cpu().numpy()
            probabilities[test_idx] = probability
            predictions[test_idx] = int(np.argmax(probability))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"\rjoint_ssl/{mode}: LOOCV fold {fold}/{len(labels)}", end="", flush=True)
        print()

        evaluated_mask = np.isfinite(probabilities).all(axis=1)
        if not np.all(evaluated_mask):
            if max_folds is None:
                raise RuntimeError("Some LOOCV folds did not produce probabilities.")
            eval_labels = labels[evaluated_mask]
            eval_predictions = predictions[evaluated_mask]
            eval_probabilities = probabilities[evaluated_mask]
        else:
            eval_labels = labels
            eval_predictions = predictions
            eval_probabilities = probabilities

        result = {
            "predictions": predictions,
            "probabilities": probabilities,
            "scores": probabilities[:, 1] if n_classes == 2 else probabilities.max(axis=1),
            "metrics": metric_fn(eval_labels, eval_predictions, eval_probabilities),
            "checkpoint": str(checkpoint_path),
            "unfrozen_transformer_layers": unfreeze_count,
            "backbone_config": config,
        }
        results[mode] = result
    return results
