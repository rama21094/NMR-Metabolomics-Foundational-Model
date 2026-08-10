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
ANALYSIS_DIR = ROOT / "code" / "analysis"
for path in (ROOT, TRAINING_DIR, ANALYSIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_joint_ssl import (  # noqa: E402
    TASK_JIGSAW,
    TASK_MASKED,
    JointNMRSSLModel,
    build_joint_model_from_loaded_checkpoint,
    normalize_spectrum,
)
from linear_probe_frozen_embeddings import apply_pool, pooled_feature_dim  # noqa: E402


FINE_TUNE_CHOICES = ("frozen", "unfreeze_last_1", "unfreeze_last_2", "unfreeze_last_3")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reinit_xavier_(module: nn.Module) -> None:
    """Reinitialize a module's own weight matrices with Xavier/Glorot init.

    Used as an ablation: the classifier-building code below always starts
    every layer (frozen or unfrozen) from the SSL-pretrained checkpoint. This
    function is applied, optionally, ONLY to the layers that were just
    unfrozen -- letting a fold fine-tune those layers from a fresh random
    init instead of their pretrained values, to isolate whether the
    pretrained weights in the fine-tuned layers matter versus just having
    trainable capacity there. Call set_seed(...) beforehand for a
    reproducible reinit (nn.init draws from the global RNG).
    """
    for sub in module.modules():
        if isinstance(sub, nn.MultiheadAttention):
            if sub.in_proj_weight is not None:
                nn.init.xavier_uniform_(sub.in_proj_weight)
            if sub.in_proj_bias is not None:
                nn.init.zeros_(sub.in_proj_bias)
            nn.init.xavier_uniform_(sub.out_proj.weight)
            if sub.out_proj.bias is not None:
                nn.init.zeros_(sub.out_proj.bias)
        elif isinstance(sub, nn.Linear):
            nn.init.xavier_uniform_(sub.weight)
            if sub.bias is not None:
                nn.init.zeros_(sub.bias)
        elif isinstance(sub, nn.LayerNorm):
            nn.init.ones_(sub.weight)
            nn.init.zeros_(sub.bias)
        elif isinstance(sub, nn.Embedding):
            # e.g. the joint model's per-layer RelativePositionBias table.
            nn.init.zeros_(sub.weight)


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
        pooling: str = "mean_pool",
    ):
        super().__init__()
        self.backbone = backbone
        self.bin_sizes = [int(b) for b in bin_sizes]
        self.include_masked_task = bool(include_masked_task)
        self.pooling = pooling
        if pooling == "mean_pool":
            pooled_count = len(self.bin_sizes) + (1 if self.include_masked_task else 0)
            pooled_dim = backbone.d_model * pooled_count
        else:
            spectrum_length = backbone.spectrum_length
            pooled_dim = sum(
                pooled_feature_dim(pooling, spectrum_length // bs, backbone.d_model) for bs in self.bin_sizes
            )
            if self.include_masked_task:
                pooled_dim += pooled_feature_dim(
                    pooling, spectrum_length // backbone.mask_bin_size, backbone.d_model
                )
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Dropout(head_dropout),
            nn.Linear(pooled_dim, n_classes),
        )
        self.softmax = nn.Softmax(dim=1)
        self.unfreeze_layers = 0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean_pool":
            return self.backbone.encode_spectrum(x, self.bin_sizes, include_masked_task=self.include_masked_task)
        # Position-preserving pooling (docs §14): mirrors encode_spectrum's bin
        # construction exactly, but pools each component with apply_pool
        # instead of the hardcoded encoded.mean(dim=1).
        spectrum_length = self.backbone.spectrum_length
        x = x[:, :spectrum_length]
        pooled = []
        for bin_size in self.bin_sizes:
            trimmed = (spectrum_length // bin_size) * bin_size
            bins = x[:, :trimmed].reshape(x.shape[0], trimmed // bin_size, bin_size)
            encoded = self.backbone.encode_bins(bins, bin_size, TASK_JIGSAW, None)
            pooled.append(apply_pool(encoded, self.pooling))
        if self.include_masked_task:
            mbs = self.backbone.mask_bin_size
            trimmed = (spectrum_length // mbs) * mbs
            bins = x[:, :trimmed].reshape(x.shape[0], trimmed // mbs, mbs)
            no_mask = torch.zeros(bins.shape[0], bins.shape[1], dtype=torch.bool, device=x.device)
            encoded = self.backbone.encode_bins(bins, mbs, TASK_MASKED, no_mask)
            pooled.append(apply_pool(encoded, self.pooling))
        return torch.cat(pooled, dim=1)

    def forward(self, x: torch.Tensor, return_logits: bool = False):
        logits = self.classifier(self.encode(x))
        return logits if return_logits else self.softmax(logits)


def fine_tune_count(mode: str) -> int:
    if mode == "frozen":
        return 0
    return int(mode.rsplit("_", 1)[-1])


def build_joint_classifier(
    checkpoint: dict,
    spectra: np.ndarray,
    n_classes: int,
    head_dropout: float,
    normalize_input_mode: str,
    unfreeze_layers: int,
    device: torch.device,
    include_masked_task: bool = True,
    reinit_unfrozen: bool = False,
    pooling: str = "mean_pool",
):
    backbone = build_joint_model_from_loaded_checkpoint(checkpoint, device)
    normalize_input = resolve_normalize_mode(normalize_input_mode, spectra, checkpoint)
    bin_sizes = [int(b) for b in checkpoint.get("jigsaw_bin_sizes", backbone.jigsaw_bin_sizes)]
    model = JointSSLSoftmaxClassifier(
        backbone, bin_sizes, n_classes, head_dropout,
        include_masked_task=include_masked_task, pooling=(pooling or "mean_pool"),
    )

    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    layers = model.backbone.encoder_layers
    if unfreeze_layers > len(layers):
        raise ValueError(f"Cannot unfreeze {unfreeze_layers}; backbone has {len(layers)} layers")
    unfrozen = list(layers)[len(layers) - unfreeze_layers:] if unfreeze_layers else []
    for layer in unfrozen:
        for parameter in layer.parameters():
            parameter.requires_grad = True
        if reinit_unfrozen:
            reinit_xavier_(layer)
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


def class_balanced_weights(y_train: np.ndarray, n_classes: int, device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights (mean 1) for a small imbalanced fold.

    See the identical helper in barth_all_models_loocv.py for why this is
    needed: plain unweighted cross-entropy on a tiny imbalanced fold can
    collapse to the majority class even when the features are separable.
    """
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


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
    n_classes = model.classifier[-1].out_features
    loss_fn = nn.CrossEntropyLoss(weight=class_balanced_weights(y_train, n_classes, device))
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
    # Read the (tens-of-MB) checkpoint file once and reuse the in-memory dict
    # across every fold/fine-tune-mode combination below, instead of
    # re-reading + unpickling it on every fold (which dominated runtime under
    # CPU contention).
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

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
                checkpoint=checkpoint,
                spectra=spectra,
                n_classes=n_classes,
                head_dropout=args.head_dropout,
                normalize_input_mode=args.normalize_input,
                unfreeze_layers=unfreeze_count,
                device=device,
                reinit_unfrozen=getattr(args, "reinit_unfrozen_xavier", False),
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
