#!/usr/bin/env python3
"""Train jigsaw-style self-supervised models for NMR spectra.

The task splits each spectrum into contiguous bins, shuffles the bins, and
predicts each shuffled bin's original absolute position. The script supports:

- fixed: train one model for one bin size
- sweep: run separate fixed-bin trainings sequentially
- multibin: train one checkpoint with per-bin-size projections/heads
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nmr_jigsaw_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SUPPORTED_BIN_SIZES = (256, 512, 1024, 2048)


@dataclass
class RunConfig:
    mode: str
    data_path: list[str]
    bin_size: int
    bin_sizes: list[int]
    out_dir: str
    model_dir: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    warmup_epochs: int
    patience: int
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    label_smoothing: float
    coverage_weight: float
    train_split: float
    val_split: float
    test_split: float
    device: str
    num_workers: int | str
    seed: int
    max_samples: int | None
    normalize_input: str
    save_every: int
    min_lr: float


class SpectrumStore:
    """Memory-mapped access to one or more same-width .npy spectra arrays."""

    def __init__(self, paths: Iterable[str], max_samples: int | None = None):
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise ValueError("At least one --data-path is required.")

        self.arrays: list[np.ndarray] = []
        self.lengths: list[int] = []
        self.cumulative: list[int] = []
        self.spectrum_length: int | None = None

        total = 0
        for path in self.paths:
            if not path.exists():
                raise FileNotFoundError(f"Input .npy not found: {path}")
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            if arr.ndim != 2:
                raise ValueError(f"{path} must be a 2D array, got shape {arr.shape}")
            if self.spectrum_length is None:
                self.spectrum_length = int(arr.shape[1])
            elif int(arr.shape[1]) != self.spectrum_length:
                raise ValueError(
                    f"All arrays must have the same number of points. "
                    f"Expected {self.spectrum_length}, got {arr.shape[1]} in {path}"
                )
            self.arrays.append(arr)
            self.lengths.append(int(arr.shape[0]))
            total += int(arr.shape[0])
            self.cumulative.append(total)

        self.total_samples = total if max_samples is None else min(total, int(max_samples))
        if self.total_samples < 2:
            raise ValueError("Need at least 2 spectra after --max-samples.")

    def __len__(self) -> int:
        return self.total_samples

    def get(self, global_index: int) -> np.ndarray:
        if global_index < 0 or global_index >= self.total_samples:
            raise IndexError(global_index)
        prev = 0
        for arr, stop in zip(self.arrays, self.cumulative):
            if global_index < stop:
                return arr[global_index - prev]
            prev = stop
        raise IndexError(global_index)

    def sample_min_max(self, seed: int, n_samples: int = 64) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        count = min(len(self), n_samples)
        indices = rng.choice(len(self), size=count, replace=False)
        mins = []
        maxs = []
        for idx in indices:
            row = np.asarray(self.get(int(idx)))
            finite = row[np.isfinite(row)]
            if finite.size:
                mins.append(float(finite.min()))
                maxs.append(float(finite.max()))
        if not mins:
            raise ValueError("Could not sample finite values from input spectra.")
        return min(mins), max(maxs)


def parse_normalize_mode(mode: str, store: SpectrumStore, seed: int) -> bool:
    mode = mode.lower()
    if mode == "true":
        return True
    if mode == "false":
        return False
    if mode != "auto":
        raise ValueError("--normalize-input must be auto, true, or false")
    sampled_min, sampled_max = store.sample_min_max(seed=seed)
    use_norm = sampled_min < -1e-4 or sampled_max > 1.5
    print(
        f"Normalization auto-check: sampled range [{sampled_min:.4g}, {sampled_max:.4g}] "
        f"-> normalize={use_norm}"
    )
    return use_norm


def normalize_spectrum(row: np.ndarray) -> np.ndarray:
    row = row.astype(np.float32, copy=True)
    finite = np.isfinite(row)
    if not np.all(finite):
        row[~finite] = 0.0
    lo = float(row.min())
    hi = float(row.max())
    if hi - lo > 1e-8:
        row = (row - lo) / (hi - lo)
    return row


def split_indices(n: int, train_split: float, val_split: float, test_split: float, seed: int) -> dict[str, np.ndarray]:
    total = train_split + val_split + test_split
    if total <= 0:
        raise ValueError("Train/val/test splits must sum to a positive value.")
    train_split, val_split, test_split = train_split / total, val_split / total, test_split / total

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_test = int(round(n * test_split))
    n_val = int(round(n * val_split))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("Split leaves no training samples. Reduce val/test split or increase samples.")
    return {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }


class JigsawSpectrumDataset(Dataset):
    def __init__(
        self,
        store: SpectrumStore,
        indices: np.ndarray,
        bin_size: int,
        normalize_input: bool,
    ):
        if bin_size not in SUPPORTED_BIN_SIZES:
            raise ValueError(f"Unsupported bin size {bin_size}; choose from {SUPPORTED_BIN_SIZES}")
        self.store = store
        self.indices = np.asarray(indices, dtype=np.int64)
        self.bin_size = int(bin_size)
        self.normalize_input = bool(normalize_input)
        self.trimmed_length = (int(store.spectrum_length) // self.bin_size) * self.bin_size
        self.n_bins = self.trimmed_length // self.bin_size
        if self.n_bins < 2:
            raise ValueError(f"Bin size {bin_size} leaves fewer than 2 bins.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        global_idx = int(self.indices[idx])
        row = np.asarray(self.store.get(global_idx)[: self.trimmed_length])
        if self.normalize_input:
            row = normalize_spectrum(row)
        else:
            row = row.astype(np.float32, copy=True)
            row[~np.isfinite(row)] = 0.0

        bins = torch.from_numpy(row.reshape(self.n_bins, self.bin_size).copy())
        perm = torch.randperm(self.n_bins)
        shuffled = bins[perm]
        return {"bins": shuffled, "labels": perm.long()}


class JigsawNMRModel(nn.Module):
    def __init__(
        self,
        spectrum_length: int,
        bin_sizes: Iterable[int],
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.spectrum_length = int(spectrum_length)
        self.bin_sizes = [int(b) for b in bin_sizes]
        self.max_bins = max(self.spectrum_length // b for b in self.bin_sizes)
        self.d_model = int(d_model)

        self.input_projections = nn.ModuleDict(
            {
                str(b): nn.Sequential(
                    nn.Linear(b, d_model),
                    nn.LayerNorm(d_model),
                    nn.Dropout(dropout),
                )
                for b in self.bin_sizes
            }
        )
        self.slot_embedding = nn.Embedding(self.max_bins, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.classifiers = nn.ModuleDict(
            {str(b): nn.Linear(d_model, self.spectrum_length // b) for b in self.bin_sizes}
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.7)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, bins: torch.Tensor, bin_size: int) -> torch.Tensor:
        key = str(int(bin_size))
        x = self.input_projections[key](bins)
        positions = torch.arange(x.shape[1], device=x.device)
        x = x + self.slot_embedding(positions).unsqueeze(0)
        x = self.transformer(x)
        return self.classifiers[key](x)


def coverage_regularizer(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    expected_counts = probs.sum(dim=1)
    target = torch.ones_like(expected_counts)
    return torch.mean((expected_counts - target) ** 2)


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, loss: float) -> dict[str, float]:
    preds = logits.argmax(dim=-1)
    token_correct = preds.eq(labels)
    n_classes = logits.shape[-1]
    k = min(5, n_classes)
    topk = logits.topk(k=k, dim=-1).indices
    topk_correct = topk.eq(labels.unsqueeze(-1)).any(dim=-1)
    neighbor_correct = (preds - labels).abs().le(1)
    exact = token_correct.all(dim=1)

    duplicate_rates = []
    for row in preds:
        unique = torch.unique(row).numel()
        duplicate_rates.append(1.0 - float(unique) / float(row.numel()))

    return {
        "loss": float(loss),
        "token_accuracy": float(token_correct.float().mean().item()),
        "top5_accuracy": float(topk_correct.float().mean().item()),
        "neighbor_accuracy": float(neighbor_correct.float().mean().item()),
        "exact_order_accuracy": float(exact.float().mean().item()),
        "duplicate_position_rate": float(np.mean(duplicate_rates)),
    }


def merge_metric_batches(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {
            "loss": float("inf"),
            "token_accuracy": 0.0,
            "top5_accuracy": 0.0,
            "neighbor_accuracy": 0.0,
            "exact_order_accuracy": 0.0,
            "duplicate_position_rate": 1.0,
        }
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def choose_workers(num_workers: int | str) -> int:
    if num_workers != "auto":
        return int(num_workers)
    cpu_count = os.cpu_count() or 4
    return max(0, min(8, cpu_count // 2))


def make_grad_scaler(device: torch.device):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    return torch.cuda.amp.GradScaler(enabled=device.type == "cuda")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def lr_factor(epoch: int, warmup_epochs: int, total_epochs: int, min_lr: float, base_lr: float) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    decay_epochs = max(1, total_epochs - warmup_epochs)
    decay_step = max(0, epoch - warmup_epochs)
    min_factor = min_lr / base_lr
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * decay_step / decay_epochs))


def run_epoch(
    model: JigsawNMRModel,
    loaders: dict[int, DataLoader],
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    criterion: nn.Module,
    scaler,
    coverage_weight: float,
    epoch: int,
    phase: str,
) -> tuple[dict[str, float], dict[int, dict[str, float]]]:
    is_train = optimizer is not None
    model.train(is_train)
    per_bin_rows: dict[int, list[dict[str, float]]] = {b: [] for b in loaders}
    all_rows: list[dict[str, float]] = []

    iterable = []
    if is_train:
        max_steps = max(len(loader) for loader in loaders.values())
        iterators = {b: iter(loader) for b, loader in loaders.items()}
        for _ in range(max_steps):
            for bin_size in sorted(loaders):
                try:
                    batch = next(iterators[bin_size])
                except StopIteration:
                    iterators[bin_size] = iter(loaders[bin_size])
                    batch = next(iterators[bin_size])
                iterable.append((bin_size, batch))
    else:
        for bin_size in sorted(loaders):
            for batch in loaders[bin_size]:
                iterable.append((bin_size, batch))

    pbar = tqdm(iterable, desc=f"{phase} epoch {epoch}", leave=False)
    for bin_size, batch in pbar:
        bins = batch["bins"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            autocast_context = torch.cuda.amp.autocast() if device.type == "cuda" else nullcontext()
            with autocast_context:
                logits = model(bins, bin_size)
                ce_loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
                cov_loss = coverage_regularizer(logits) if coverage_weight > 0 else logits.new_tensor(0.0)
                loss = ce_loss + coverage_weight * cov_loss

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

        metrics = compute_metrics(logits.detach(), labels.detach(), float(loss.detach().item()))
        per_bin_rows[bin_size].append(metrics)
        all_rows.append(metrics)
        pbar.set_postfix(
            {
                "bin": bin_size,
                "loss": f"{metrics['loss']:.4f}",
                "acc": f"{metrics['token_accuracy']:.3f}",
            }
        )

    per_bin = {bin_size: merge_metric_batches(rows) for bin_size, rows in per_bin_rows.items()}
    return merge_metric_batches(all_rows), per_bin


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def append_metrics_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def plot_curves(history: list[dict], out_path: Path) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history if row["split"] == "train" and row["bin_size"] == "overall"]
    train_loss = [row["loss"] for row in history if row["split"] == "train" and row["bin_size"] == "overall"]
    val_loss = [row["loss"] for row in history if row["split"] == "val" and row["bin_size"] == "overall"]
    train_acc = [row["token_accuracy"] for row in history if row["split"] == "train" and row["bin_size"] == "overall"]
    val_acc = [row["token_accuracy"] for row in history if row["split"] == "val" and row["bin_size"] == "overall"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(epochs[: len(train_loss)], train_loss, label="train")
    if val_loss:
        axes[0].plot(epochs[: len(val_loss)], val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs[: len(train_acc)], train_acc, label="train")
    if val_acc:
        axes[1].plot(epochs[: len(val_acc)], val_acc, label="val")
    axes[1].set_title("Token accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def save_preview_plot(
    model: JigsawNMRModel,
    dataset: JigsawSpectrumDataset,
    bin_size: int,
    device: torch.device,
    out_path: Path,
) -> None:
    if len(dataset) == 0:
        return
    sample = dataset[0]
    bins = sample["bins"].unsqueeze(0).to(device)
    labels = sample["labels"]
    logits = model(bins, bin_size).cpu()
    preds = logits.argmax(dim=-1).squeeze(0)
    shuffled = sample["bins"].numpy().reshape(-1)

    n_bins = dataset.n_bins
    original_bins = torch.empty_like(sample["bins"])
    predicted_bins = torch.zeros_like(sample["bins"])
    filled = torch.zeros(n_bins, dtype=torch.bool)
    for shuffled_slot, original_pos in enumerate(labels.tolist()):
        original_bins[original_pos] = sample["bins"][shuffled_slot]
    for shuffled_slot, predicted_pos in enumerate(preds.tolist()):
        if 0 <= predicted_pos < n_bins and not filled[predicted_pos]:
            predicted_bins[predicted_pos] = sample["bins"][shuffled_slot]
            filled[predicted_pos] = True

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(original_bins.numpy().reshape(-1), linewidth=0.8)
    axes[0].set_title("Original order")
    axes[1].plot(shuffled, linewidth=0.8)
    axes[1].set_title("Shuffled input")
    axes[2].plot(predicted_bins.numpy().reshape(-1), linewidth=0.8)
    axes[2].set_title("Predicted reorder")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_datasets_for_bins(
    store: SpectrumStore,
    splits: dict[str, np.ndarray],
    bin_sizes: Iterable[int],
    normalize_input: bool,
) -> dict[int, dict[str, JigsawSpectrumDataset]]:
    datasets: dict[int, dict[str, JigsawSpectrumDataset]] = {}
    for bin_size in bin_sizes:
        datasets[int(bin_size)] = {
            split: JigsawSpectrumDataset(store, indices, int(bin_size), normalize_input)
            for split, indices in splits.items()
            if len(indices) > 0
        }
    return datasets


def save_checkpoint(
    path: Path,
    model: JigsawNMRModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    config: RunConfig,
    bin_sizes: list[int],
    history: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "bin_sizes": bin_sizes,
            "spectrum_length": model.spectrum_length,
            "hyperparameters": asdict(config),
            "history": history,
        },
        path,
    )


def train_one_run(config: RunConfig, run_name: str, bin_sizes: list[int]) -> dict:
    seed_everything(config.seed)
    store = SpectrumStore(config.data_path, max_samples=config.max_samples)
    normalize_input = parse_normalize_mode(config.normalize_input, store, config.seed)
    splits = split_indices(len(store), config.train_split, config.val_split, config.test_split, config.seed)
    print(
        f"Samples: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}; "
        f"spectrum_length={store.spectrum_length}"
    )

    datasets = build_datasets_for_bins(store, splits, bin_sizes, normalize_input)
    num_workers = choose_workers(config.num_workers)
    device = choose_device(config.device)
    print(f"Using device={device}, num_workers={num_workers}, bin_sizes={bin_sizes}")

    train_loaders = {
        b: make_loader(ds["train"], config.batch_size, True, num_workers, config.seed + b)
        for b, ds in datasets.items()
        if "train" in ds
    }
    val_loaders = {
        b: make_loader(ds["val"], config.batch_size, False, num_workers, config.seed + 1000 + b)
        for b, ds in datasets.items()
        if "val" in ds
    }
    test_loaders = {
        b: make_loader(ds["test"], config.batch_size, False, num_workers, config.seed + 2000 + b)
        for b, ds in datasets.items()
        if "test" in ds
    }

    model = JigsawNMRModel(
        spectrum_length=int(store.spectrum_length),
        bin_sizes=bin_sizes,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    scaler = make_grad_scaler(device)

    result_dir = Path(config.out_dir) / run_name
    model_dir = Path(config.model_dir) / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    config_payload = asdict(config)
    config_payload.update({"run_name": run_name, "active_bin_sizes": bin_sizes, "normalize_resolved": normalize_input})
    save_json(result_dir / "config.json", config_payload)

    history: list[dict] = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    run_file_tag = run_name.replace("/", "_")
    best_path = model_dir / f"{run_file_tag}_best.pth"

    for epoch in range(1, config.epochs + 1):
        for group in optimizer.param_groups:
            group["lr"] = config.lr * lr_factor(epoch - 1, config.warmup_epochs, config.epochs, config.min_lr, config.lr)

        train_metrics, train_per_bin = run_epoch(
            model,
            train_loaders,
            optimizer,
            device,
            criterion,
            scaler,
            config.coverage_weight,
            epoch,
            "train",
        )

        val_metrics = None
        val_per_bin = {}
        if val_loaders:
            val_metrics, val_per_bin = run_epoch(
                model,
                val_loaders,
                None,
                device,
                criterion,
                scaler,
                config.coverage_weight,
                epoch,
                "val",
            )
            monitor_loss = val_metrics["loss"]
        else:
            monitor_loss = train_metrics["loss"]

        rows = [("train", "overall", train_metrics)]
        rows.extend(("train", str(b), metrics) for b, metrics in train_per_bin.items())
        if val_metrics is not None:
            rows.append(("val", "overall", val_metrics))
            rows.extend(("val", str(b), metrics) for b, metrics in val_per_bin.items())

        for split, bin_label, metrics in rows:
            row = {
                "epoch": epoch,
                "split": split,
                "bin_size": bin_label,
                "lr": optimizer.param_groups[0]["lr"],
                **metrics,
            }
            history.append(row)
            append_metrics_csv(result_dir / "metrics.csv", row)

        improved = monitor_loss < best_val_loss - 1e-6
        if improved:
            best_val_loss = monitor_loss
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, optimizer, epoch, best_val_loss, config, bin_sizes, history)
        else:
            epochs_without_improvement += 1

        if config.save_every > 0 and epoch % config.save_every == 0:
            save_checkpoint(
                model_dir / f"{run_file_tag}_epoch_{epoch}.pth",
                model,
                optimizer,
                epoch,
                best_val_loss,
                config,
                bin_sizes,
                history,
            )

        val_text = f", val_loss={val_metrics['loss']:.4f}" if val_metrics else ""
        print(
            f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, "
            f"train_acc={train_metrics['token_accuracy']:.3f}{val_text}, "
            f"best={best_val_loss:.4f}, patience={epochs_without_improvement}/{config.patience}"
        )

        plot_curves(history, result_dir / "training_curves.png")
        if epochs_without_improvement >= config.patience:
            print(f"Early stopping after {config.patience} epochs without improvement.")
            break

    if test_loaders:
        test_metrics, test_per_bin = run_epoch(
            model,
            test_loaders,
            None,
            device,
            criterion,
            scaler,
            config.coverage_weight,
            epoch,
            "test",
        )
        save_json(result_dir / "test_metrics.json", {"overall": test_metrics, "per_bin": test_per_bin})

    for bin_size in bin_sizes:
        preview_dataset = datasets[bin_size].get("val") or datasets[bin_size]["train"]
        save_preview_plot(
            model,
            preview_dataset,
            bin_size,
            device,
            result_dir / f"preview_bin_{bin_size}.png",
        )

    summary = {
        "run_name": run_name,
        "best_val_loss": best_val_loss,
        "best_checkpoint": str(best_path),
        "epochs_completed": epoch,
        "bin_sizes": bin_sizes,
        "results_dir": str(result_dir),
    }
    save_json(result_dir / "summary.json", summary)
    return summary


def validate_bin_sizes(values: Iterable[int]) -> list[int]:
    bins = [int(v) for v in values]
    invalid = [b for b in bins if b not in SUPPORTED_BIN_SIZES]
    if invalid:
        raise ValueError(f"Unsupported bin size(s): {invalid}. Choose from {SUPPORTED_BIN_SIZES}.")
    return sorted(set(bins))


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Train NMR jigsaw self-supervised models.")
    parser.add_argument("--mode", choices=["fixed", "sweep", "multibin"], default="fixed")
    parser.add_argument("--data-path", nargs="+", default=["data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v3.npy"])
    parser.add_argument("--bin-size", type=int, default=1024, choices=SUPPORTED_BIN_SIZES)
    parser.add_argument("--bin-sizes", nargs="+", type=int, default=list(SUPPORTED_BIN_SIZES))
    parser.add_argument("--out-dir", default="results/jigsaw")
    parser.add_argument("--model-dir", default="models/jigsaw")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--coverage-weight", type=float, default=0.0)
    parser.add_argument("--train-split", type=float, default=0.80)
    parser.add_argument("--val-split", type=float, default=0.20)
    parser.add_argument("--test-split", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--normalize-input", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--save-every", type=int, default=0, help="Save a periodic checkpoint every N epochs in addition to the best one; 0 disables this.")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    args = parser.parse_args()

    num_workers: int | str
    num_workers = args.num_workers if args.num_workers == "auto" else int(args.num_workers)
    return RunConfig(
        mode=args.mode,
        data_path=args.data_path,
        bin_size=args.bin_size,
        bin_sizes=validate_bin_sizes(args.bin_sizes),
        out_dir=args.out_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        coverage_weight=args.coverage_weight,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        device=args.device,
        num_workers=num_workers,
        seed=args.seed,
        max_samples=args.max_samples,
        normalize_input=args.normalize_input,
        save_every=args.save_every,
        min_lr=args.min_lr,
    )


def main() -> None:
    config = parse_args()
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    if config.mode == "fixed":
        run_name = f"bin_{config.bin_size}/{timestamp}"
        summary = train_one_run(config, run_name, [config.bin_size])
        print(json.dumps(summary, indent=2))
    elif config.mode == "sweep":
        summaries = []
        for bin_size in config.bin_sizes:
            print(f"\n{'=' * 80}\nStarting fixed-bin sweep run for bin_size={bin_size}\n{'=' * 80}")
            run_name = f"bin_{bin_size}/{timestamp}"
            summaries.append(train_one_run(config, run_name, [bin_size]))
        save_json(Path(config.out_dir) / f"sweep_{timestamp}_summary.json", {"runs": summaries})
        print(json.dumps({"runs": summaries}, indent=2))
    elif config.mode == "multibin":
        run_name = f"multibin/{timestamp}"
        summary = train_one_run(config, run_name, config.bin_sizes)
        print(json.dumps(summary, indent=2))
    else:
        raise ValueError(f"Unknown mode: {config.mode}")


if __name__ == "__main__":
    main()
