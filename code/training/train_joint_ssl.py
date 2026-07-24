#!/usr/bin/env python3
"""Train a joint masked-reconstruction + multibin jigsaw SSL model for NMR spectra."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import random
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nmr_joint_ssl_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SUPPORTED_BIN_SIZES = (256, 512, 1024, 2048)
TASK_MASKED = 0
TASK_JIGSAW = 1


@dataclass
class RunConfig:
    data_path: list[str]
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
    fourier_bands: int
    task_embed_dim: int
    mask_bin_size: int
    jigsaw_bin_sizes: list[int]
    mask_ratio_min: float
    mask_ratio_max: float
    unmasked_recon_weight: float
    jigsaw_weight: float
    coverage_weight: float
    loss_ema_decay: float
    peak_top_fraction: float
    soft_jigsaw_sigma: float
    recon_skip_weight: float
    monitor_metric: str
    deterministic_eval: bool
    eval_seed: int
    label_smoothing: float
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
            total += int(arr.shape[0])
            self.lengths.append(int(arr.shape[0]))
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


class JointSpectrumDataset(Dataset):
    def __init__(
        self,
        store: SpectrumStore,
        indices: np.ndarray,
        spectrum_length: int,
        normalize_input: bool,
    ):
        self.store = store
        self.indices = np.asarray(indices, dtype=np.int64)
        self.spectrum_length = int(spectrum_length)
        self.normalize_input = bool(normalize_input)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = np.asarray(self.store.get(int(self.indices[idx]))[: self.spectrum_length])
        if self.normalize_input:
            row = normalize_spectrum(row)
        else:
            row = row.astype(np.float32, copy=True)
            row[~np.isfinite(row)] = 0.0
        return torch.from_numpy(row)


class FourierPositionFeatures(nn.Module):
    def __init__(self, num_bands: int, d_model: int):
        super().__init__()
        self.num_bands = int(num_bands)
        in_dim = 1 + 2 * self.num_bands
        self.proj = nn.Sequential(nn.Linear(in_dim, d_model), nn.LayerNorm(d_model))

    def forward(self, position_ids: torch.Tensor, n_tokens: int) -> torch.Tensor:
        denom = max(1, int(n_tokens) - 1)
        pos = position_ids.float().unsqueeze(-1) / float(denom)
        if self.num_bands > 0:
            freqs = torch.pow(
                torch.tensor(2.0, device=position_ids.device),
                torch.arange(self.num_bands, device=position_ids.device, dtype=torch.float32),
            )
            angles = math.pi * pos * freqs.view(1, -1)
            features = torch.cat([pos, torch.sin(angles), torch.cos(angles)], dim=-1)
        else:
            features = pos
        return self.proj(features)


class RelativePositionBias(nn.Module):
    def __init__(self, max_bins: int, nhead: int):
        super().__init__()
        self.max_bins = int(max_bins)
        self.nhead = int(nhead)
        self.bias = nn.Embedding(2 * self.max_bins - 1, self.nhead)
        nn.init.zeros_(self.bias.weight)

    def forward(self, position_ids: torch.Tensor, batch_size: int) -> torch.Tensor:
        diff = position_ids[:, None] - position_ids[None, :]
        diff = diff.clamp(-(self.max_bins - 1), self.max_bins - 1) + self.max_bins - 1
        bias = self.bias(diff.long()).permute(2, 0, 1)
        return bias.repeat_interleave(batch_size, dim=0)


class RelPosEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, max_bins: int):
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.nhead = int(nhead)
        self.head_dim = int(d_model) // int(nhead)
        self.dropout_p = float(dropout)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rel_pos = RelativePositionBias(max_bins, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm1(x)
        batch_size, n_tokens, d_model = x_norm.shape
        qkv = self.qkv(x_norm).reshape(batch_size, n_tokens, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_bias = self.rel_pos(position_ids, batch_size).reshape(batch_size, self.nhead, n_tokens, n_tokens)
        if hasattr(F, "scaled_dot_product_attention"):
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_bias,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + attn_bias
            weights = F.softmax(scores, dim=-1)
            weights = F.dropout(weights, p=self.dropout_p, training=self.training)
            attn_out = torch.matmul(weights, v)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, n_tokens, d_model)
        attn_out = self.out_proj(attn_out)
        x = residual + self.dropout1(attn_out)
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


class JointNMRSSLModel(nn.Module):
    """Shared encoder with masked reconstruction and multibin jigsaw heads."""

    def __init__(
        self,
        spectrum_length: int,
        mask_bin_size: int = 1024,
        jigsaw_bin_sizes: Iterable[int] = SUPPORTED_BIN_SIZES,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 4,
        dim_feedforward: int = 768,
        dropout: float = 0.15,
        fourier_bands: int = 8,
        task_embed_dim: int = 8,
    ):
        super().__init__()
        self.spectrum_length = int(spectrum_length)
        self.mask_bin_size = int(mask_bin_size)
        self.jigsaw_bin_sizes = sorted({int(b) for b in jigsaw_bin_sizes})
        self.bin_sizes = sorted(set(self.jigsaw_bin_sizes + [self.mask_bin_size]))
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.fourier_bands = int(fourier_bands)
        self.task_embed_dim = int(task_embed_dim)
        self.max_bins = max(self.spectrum_length // b for b in self.bin_sizes)

        for bin_size in self.bin_sizes:
            if bin_size not in SUPPORTED_BIN_SIZES:
                raise ValueError(f"Unsupported bin size {bin_size}; choose from {SUPPORTED_BIN_SIZES}")
            if self.spectrum_length // bin_size < 2:
                raise ValueError(f"Bin size {bin_size} leaves fewer than 2 bins.")

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
        self.position_features = FourierPositionFeatures(fourier_bands, d_model)
        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        # Bottlenecked task signal: a low-rank nudge rather than a full-rank,
        # freely-learned d_model vector the encoder could use to fork into two
        # nearly-disjoint per-task sub-networks. task_proj is initialized with a
        # small gain so the signal starts gentle and only grows if the training
        # signal actually needs it.
        self.task_embedding = nn.Embedding(2, self.task_embed_dim)
        self.task_proj = nn.Linear(self.task_embed_dim, d_model, bias=False)
        self.bin_embedding = nn.ParameterDict(
            {str(b): nn.Parameter(torch.zeros(1, 1, d_model)) for b in self.bin_sizes}
        )
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.encoder_layers = nn.ModuleList(
            [
                RelPosEncoderLayer(d_model, nhead, dim_feedforward, dropout, self.max_bins)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.reconstruction_heads = nn.ModuleDict(
            {str(self.mask_bin_size): nn.Linear(d_model, self.mask_bin_size)}
        )
        self.reconstruction_skips = nn.ModuleDict(
            {str(self.mask_bin_size): nn.Linear(self.mask_bin_size, self.mask_bin_size)}
        )
        self.jigsaw_heads = nn.ModuleDict(
            {str(b): nn.Linear(d_model, self.spectrum_length // b) for b in self.jigsaw_bin_sizes}
        )
        self.apply(self._init_weights)
        for skip in self.reconstruction_skips.values():
            nn.init.eye_(skip.weight)
            nn.init.zeros_(skip.bias)
        if self.task_embed_dim == d_model:
            # Legacy (pre-bottleneck) checkpoints stored a full-width task
            # embedding with no separate projection. An identity task_proj
            # reproduces their exact behavior so old checkpoints still load
            # and evaluate correctly.
            nn.init.eye_(self.task_proj.weight)
        else:
            nn.init.xavier_uniform_(self.task_proj.weight, gain=0.05)

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

    def encode_bins(
        self,
        bins: torch.Tensor,
        bin_size: int,
        task_id: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key = str(int(bin_size))
        if key not in self.input_projections:
            raise ValueError(f"Model was not configured for bin size {bin_size}")
        batch_size, n_tokens, _ = bins.shape
        position_ids = torch.arange(n_tokens, device=bins.device)

        intensity = self.input_projections[key](bins)
        if mask is not None:
            mask_tokens = self.mask_token.expand(batch_size, n_tokens, -1)
            intensity = torch.where(mask.unsqueeze(-1), mask_tokens, intensity)

        position = self.slot_embedding(position_ids).unsqueeze(0)
        position = position + self.position_features(position_ids, n_tokens).unsqueeze(0)
        x = self.fusion(torch.cat([intensity, position.expand(batch_size, -1, -1)], dim=-1))

        task_ids = torch.full((batch_size, n_tokens), int(task_id), device=bins.device, dtype=torch.long)
        x = x + self.task_proj(self.task_embedding(task_ids)) + self.bin_embedding[key]
        for layer in self.encoder_layers:
            x = layer(x, position_ids)
        return self.final_norm(x)

    def forward_masked(
        self,
        bins: torch.Tensor,
        mask: torch.Tensor,
        bin_size: int | None = None,
        skip_weight: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bin_size = self.mask_bin_size if bin_size is None else int(bin_size)
        encoded = self.encode_bins(bins, bin_size, TASK_MASKED, mask)
        reconstructed = self.reconstruction_heads[str(bin_size)](encoded)
        if skip_weight:
            skip = self.reconstruction_skips[str(bin_size)](bins)
            reconstructed = reconstructed + float(skip_weight) * torch.where(mask.unsqueeze(-1), torch.zeros_like(skip), skip)
        return reconstructed, encoded

    def forward_jigsaw(self, bins: torch.Tensor, bin_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode_bins(bins, int(bin_size), TASK_JIGSAW, None)
        logits = self.jigsaw_heads[str(int(bin_size))](encoded)
        return logits, encoded

    def encode_spectrum(
        self,
        x: torch.Tensor,
        bin_sizes: Iterable[int] | None = None,
        include_masked_task: bool = True,
    ) -> torch.Tensor:
        """Return concatenated mean-pooled natural-order embeddings.

        By default this pools both task pathways: the jigsaw path (per
        ``bin_sizes``, natural token order) and the masked-reconstruction path
        (at ``mask_bin_size``, with no tokens actually masked). Without the
        latter, downstream consumers only ever see representations produced
        under the jigsaw task id, so the masked-reconstruction objective never
        influences the features that get used -- despite being half the
        pretraining loss.
        """
        if x.shape[1] < self.spectrum_length:
            raise ValueError(f"Input spectrum length {x.shape[1]} is shorter than checkpoint length {self.spectrum_length}")
        x = x[:, : self.spectrum_length]
        pooled = []
        active = self.jigsaw_bin_sizes if bin_sizes is None else [int(b) for b in bin_sizes]
        for bin_size in active:
            trimmed_length = (self.spectrum_length // int(bin_size)) * int(bin_size)
            bins = x[:, :trimmed_length].reshape(x.shape[0], trimmed_length // int(bin_size), int(bin_size))
            encoded = self.encode_bins(bins, int(bin_size), TASK_JIGSAW, None)
            pooled.append(encoded.mean(dim=1))
        if include_masked_task:
            trimmed_length = (self.spectrum_length // self.mask_bin_size) * self.mask_bin_size
            bins = x[:, :trimmed_length].reshape(x.shape[0], trimmed_length // self.mask_bin_size, self.mask_bin_size)
            no_mask = torch.zeros(bins.shape[0], bins.shape[1], dtype=torch.bool, device=x.device)
            encoded = self.encode_bins(bins, self.mask_bin_size, TASK_MASKED, no_mask)
            pooled.append(encoded.mean(dim=1))
        return torch.cat(pooled, dim=1)


def validate_bin_sizes(values: Iterable[int]) -> list[int]:
    bins = [int(v) for v in values]
    invalid = [b for b in bins if b not in SUPPORTED_BIN_SIZES]
    if invalid:
        raise ValueError(f"Unsupported bin size(s): {invalid}. Choose from {SUPPORTED_BIN_SIZES}.")
    return sorted(set(bins))


def str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value")


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


def lr_factor(epoch: int, warmup_epochs: int, total_epochs: int, min_lr: float, base_lr: float) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    decay_epochs = max(1, total_epochs - warmup_epochs)
    decay_step = max(0, epoch - warmup_epochs)
    min_factor = min_lr / base_lr
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * decay_step / decay_epochs))


def spectra_to_bins(x: torch.Tensor, bin_size: int) -> torch.Tensor:
    trimmed_length = (x.shape[1] // int(bin_size)) * int(bin_size)
    return x[:, :trimmed_length].reshape(x.shape[0], trimmed_length // int(bin_size), int(bin_size))


def random_patch_mask(batch_size: int, n_tokens: int, ratio_min: float, ratio_max: float, device: torch.device) -> torch.Tensor:
    ratios = torch.empty(batch_size, device=device).uniform_(ratio_min, ratio_max)
    counts = torch.clamp((ratios * n_tokens).round().long(), min=1, max=max(1, n_tokens - 1))
    noise = torch.rand(batch_size, n_tokens, device=device)
    order = noise.argsort(dim=1)
    mask = torch.zeros(batch_size, n_tokens, dtype=torch.bool, device=device)
    for row, count in enumerate(counts.tolist()):
        mask[row, order[row, :count]] = True
    return mask


def make_jigsaw_batch(x: torch.Tensor, bin_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    bins = spectra_to_bins(x, bin_size)
    batch_size, n_bins, _ = bins.shape
    shuffled = torch.empty_like(bins)
    labels = torch.empty(batch_size, n_bins, dtype=torch.long, device=x.device)
    for row in range(batch_size):
        perm = torch.randperm(n_bins, device=x.device)
        shuffled[row] = bins[row, perm]
        labels[row] = perm
    return shuffled, labels


def split_batch_for_bin_sizes(batch_size: int, bin_sizes: list[int], device: torch.device) -> dict[int, torch.Tensor]:
    """Randomly partition a batch across bin sizes for this step.

    Previously every sample was run through every jigsaw bin size each step,
    giving the jigsaw objective len(bin_sizes)x as many encoder forward/backward
    passes as the masking objective. Splitting the batch instead means each
    sample sees one randomly-assigned bin size per step (a different one next
    step), so all scales still get covered over an epoch while the total
    jigsaw compute per step roughly matches the masking task's single pass.
    """
    n_groups = max(1, min(len(bin_sizes), batch_size))
    order = torch.randperm(batch_size, device=device)
    chunks = torch.chunk(order, n_groups)
    # If there are more bin sizes than samples, rotate which bin sizes get a
    # group this step so every bin size still gets covered across steps.
    bin_size_order = random.sample(bin_sizes, len(bin_sizes)) if len(bin_sizes) > n_groups else bin_sizes
    return {bin_size: chunk for bin_size, chunk in zip(bin_size_order, chunks)}


def coverage_regularizer(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    expected_counts = probs.sum(dim=1)
    target = torch.ones_like(expected_counts)
    return torch.mean((expected_counts - target) ** 2)


class EMALossNormalizer:
    """Normalize multitask losses by moving loss scale estimates."""

    def __init__(self, decay: float = 0.98, eps: float = 1e-8):
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError("EMA decay must satisfy 0 <= decay < 1")
        self.decay = float(decay)
        self.eps = float(eps)
        self.scales: dict[str, float] = {}

    def normalize(self, name: str, loss: torch.Tensor, update: bool) -> torch.Tensor:
        value = float(loss.detach().clamp_min(self.eps).item())
        if name not in self.scales:
            self.scales[name] = value
        elif update:
            self.scales[name] = self.decay * self.scales[name] + (1.0 - self.decay) * value
        denom = torch.as_tensor(self.scales[name] + self.eps, device=loss.device, dtype=loss.dtype)
        return loss / denom

    def state_dict(self) -> dict:
        return {"decay": self.decay, "eps": self.eps, "scales": dict(self.scales)}

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.eps = float(state.get("eps", self.eps))
        self.scales = {str(k): float(v) for k, v in state.get("scales", {}).items()}


def top_peak_bin_weights(target_bins: torch.Tensor, top_fraction: float) -> torch.Tensor:
    """Return a per-bin 0/1 mask selecting the top `top_fraction` bins by magnitude, per spectrum.

    Ranks bins by a peak score (blend of mean and max magnitude) and keeps only
    the highest-scoring `top_fraction` of them per spectrum; the rest (mostly
    flat baseline/noise) get weight 0 and are excluded from the reconstruction
    loss entirely. This is a hard version of relevance weighting: rather than
    softly up-weighting high-magnitude bins relative to the spectrum's single
    tallest bin (which can be dominated by one recurring artifact peak), it
    directly restricts supervision to the bins that actually carry signal.
    """
    if top_fraction >= 1.0:
        return torch.ones(target_bins.shape[:2], device=target_bins.device, dtype=target_bins.dtype)
    magnitude = target_bins.abs()
    peak_score = 0.5 * magnitude.mean(dim=2) + 0.5 * magnitude.amax(dim=2)
    n_bins = peak_score.shape[1]
    k = max(1, int(round(float(top_fraction) * n_bins)))
    threshold = peak_score.topk(k, dim=1).values.min(dim=1, keepdim=True).values
    return (peak_score >= threshold).to(target_bins.dtype)


def weighted_region_loss(se_per_bin: torch.Tensor, region_mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weighted_mask = region_mask.float() * weights
    return (se_per_bin * weighted_mask).sum() / (weighted_mask.sum() + 1e-8)


def soft_jigsaw_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, sigma: float, label_smoothing: float) -> torch.Tensor:
    """Distance-aware jigsaw CE where near-position errors are less wrong."""
    if sigma <= 0:
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            label_smoothing=label_smoothing,
        )

    n_classes = logits.shape[-1]
    positions = torch.arange(n_classes, device=logits.device, dtype=logits.dtype)
    distances = positions.view(1, 1, -1) - labels.unsqueeze(-1).to(logits.dtype)
    targets = torch.exp(-0.5 * (distances / float(sigma)) ** 2)
    targets = targets / targets.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    if label_smoothing > 0:
        targets = (1.0 - label_smoothing) * targets + label_smoothing / float(n_classes)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


def compute_joint_loss(
    model: JointNMRSSLModel,
    spectra: torch.Tensor,
    config: RunConfig,
    loss_normalizer: EMALossNormalizer,
    update_loss_normalizer: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    masked_bins = spectra_to_bins(spectra, config.mask_bin_size)
    mask = random_patch_mask(
        spectra.shape[0],
        masked_bins.shape[1],
        config.mask_ratio_min,
        config.mask_ratio_max,
        spectra.device,
    )
    masked_input = masked_bins.clone()
    masked_input[mask] = 0.0
    reconstructed, _ = model.forward_masked(
        masked_input,
        mask,
        config.mask_bin_size,
        skip_weight=config.recon_skip_weight,
    )

    se_per_bin = ((reconstructed - masked_bins) ** 2).mean(dim=2)
    peak_mask = top_peak_bin_weights(masked_bins, config.peak_top_fraction)
    masked_loss = weighted_region_loss(se_per_bin, mask, peak_mask)
    unmasked = ~mask
    unmasked_loss = weighted_region_loss(se_per_bin, unmasked, peak_mask)
    masked_peak_overlap = (mask.to(peak_mask.dtype) * peak_mask).sum(dim=1).mean()

    jigsaw_losses = []
    jigsaw_accs = []
    coverage_losses = []
    bin_groups = split_batch_for_bin_sizes(spectra.shape[0], config.jigsaw_bin_sizes, spectra.device)
    for bin_size, group_idx in bin_groups.items():
        if group_idx.numel() == 0:
            continue
        shuffled, labels = make_jigsaw_batch(spectra[group_idx], bin_size)
        logits, _ = model.forward_jigsaw(shuffled, bin_size)
        ce_loss = soft_jigsaw_cross_entropy(
            logits,
            labels,
            sigma=config.soft_jigsaw_sigma,
            label_smoothing=config.label_smoothing,
        )
        cov_loss = coverage_regularizer(logits) if config.coverage_weight > 0 else logits.new_tensor(0.0)
        jigsaw_losses.append(ce_loss)
        coverage_losses.append(cov_loss)
        with torch.no_grad():
            jigsaw_accs.append(logits.argmax(dim=-1).eq(labels).float().mean())

    jigsaw_loss = torch.stack(jigsaw_losses).mean()
    coverage_loss = torch.stack(coverage_losses).mean() if coverage_losses else jigsaw_loss.new_tensor(0.0)
    masked_norm = loss_normalizer.normalize("masked", masked_loss, update_loss_normalizer)
    unmasked_norm = loss_normalizer.normalize("unmasked", unmasked_loss, update_loss_normalizer)
    jigsaw_norm = loss_normalizer.normalize("jigsaw", jigsaw_loss, update_loss_normalizer)
    coverage_norm = loss_normalizer.normalize("coverage", coverage_loss, update_loss_normalizer) if config.coverage_weight > 0 else coverage_loss
    raw_total = (
        masked_loss
        + config.unmasked_recon_weight * unmasked_loss
        + config.jigsaw_weight * jigsaw_loss
        + config.coverage_weight * coverage_loss
    )
    total = (
        masked_norm
        + config.unmasked_recon_weight * unmasked_norm
        + config.jigsaw_weight * jigsaw_norm
        + config.coverage_weight * coverage_norm
    )
    metrics = {
        "loss": float(total.detach().item()),
        "balanced_loss": float(total.detach().item()),
        "raw_loss": float(raw_total.detach().item()),
        "masked_loss": float(masked_loss.detach().item()),
        "unmasked_loss": float(unmasked_loss.detach().item()),
        "jigsaw_loss": float(jigsaw_loss.detach().item()),
        "coverage_loss": float(coverage_loss.detach().item()),
        "masked_loss_norm": float(masked_norm.detach().item()),
        "unmasked_loss_norm": float(unmasked_norm.detach().item()),
        "jigsaw_loss_norm": float(jigsaw_norm.detach().item()),
        "coverage_loss_norm": float(coverage_norm.detach().item()) if config.coverage_weight > 0 else 0.0,
        "peak_bin_frac": float(peak_mask.detach().mean().item()),
        "masked_peak_bins_mean": float(masked_peak_overlap.detach().item()),
        "jigsaw_token_accuracy": float(torch.stack(jigsaw_accs).mean().detach().item()),
        "mask_ratio": float(mask.float().mean().detach().item()),
    }
    return total, metrics


def merge_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {
            "loss": float("inf"),
            "masked_loss": float("inf"),
            "unmasked_loss": float("inf"),
            "jigsaw_loss": float("inf"),
            "coverage_loss": 0.0,
            "jigsaw_token_accuracy": 0.0,
            "mask_ratio": 0.0,
        }
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


@contextmanager
def fixed_torch_rng(seed: int, device: torch.device):
    devices = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        yield


def run_epoch(
    model: JointNMRSSLModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler,
    config: RunConfig,
    loss_normalizer: EMALossNormalizer,
    epoch: int,
    phase: str,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    rows = []
    pbar = tqdm(loader, desc=f"{phase} epoch {epoch}", leave=False)
    rng_context = fixed_torch_rng(config.eval_seed, device) if (not is_train and config.deterministic_eval) else nullcontext()
    with rng_context:
        for spectra in pbar:
            spectra = spectra.to(device, non_blocking=True)
            with torch.set_grad_enabled(is_train):
                autocast_context = torch.cuda.amp.autocast() if device.type == "cuda" else nullcontext()
                with autocast_context:
                    loss, metrics = compute_joint_loss(
                        model,
                        spectra,
                        config,
                        loss_normalizer,
                        update_loss_normalizer=is_train,
                    )
                if is_train:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
            rows.append(metrics)
            pbar.set_postfix(
                {
                    "loss": f"{metrics['loss']:.4f}",
                    "raw": f"{metrics['raw_loss']:.4f}",
                    "mask": f"{metrics['masked_loss']:.4f}",
                    "jig": f"{metrics['jigsaw_token_accuracy']:.3f}",
                }
            )
    return merge_metrics(rows)


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
    epochs = [row["epoch"] for row in history if row["split"] == "train"]
    train_loss = [row["loss"] for row in history if row["split"] == "train"]
    val_loss = [row["loss"] for row in history if row["split"] == "val"]
    train_jig = [row["jigsaw_token_accuracy"] for row in history if row["split"] == "train"]
    val_jig = [row["jigsaw_token_accuracy"] for row in history if row["split"] == "val"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(epochs[: len(train_loss)], train_loss, label="train")
    if val_loss:
        axes[0].plot(epochs[: len(val_loss)], val_loss, label="val")
    axes[0].set_title("Joint SSL loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs[: len(train_jig)], train_jig, label="train")
    if val_jig:
        axes[1].plot(epochs[: len(val_jig)], val_jig, label="val")
    axes[1].set_title("Jigsaw token accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_checkpoint(
    path: Path,
    model: JointNMRSSLModel,
    optimizer: torch.optim.Optimizer,
    loss_normalizer: EMALossNormalizer,
    epoch: int,
    best_val_loss: float,
    config: RunConfig,
    history: list[dict],
    normalize_resolved: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_normalizer_state": loss_normalizer.state_dict(),
            "best_val_loss": best_val_loss,
            "spectrum_length": model.spectrum_length,
            "mask_bin_size": model.mask_bin_size,
            "jigsaw_bin_sizes": model.jigsaw_bin_sizes,
            "bin_sizes": model.bin_sizes,
            "hyperparameters": {
                **asdict(config),
                "normalize_resolved": bool(normalize_resolved),
                "architecture": "joint_masked_multibin_jigsaw_v1",
            },
            "history": history,
        },
        path,
    )


def infer_task_embed_dim(checkpoint: dict, d_model: int) -> int:
    """Infer task-embedding width from the checkpoint's own tensor shape.

    Checkpoints saved before the bottlenecked task embedding was introduced
    have no "task_embed_dim" hyperparameter and a full-width (d_model)
    task_embedding.weight. Reading the shape directly (rather than trusting a
    hyperparameter key that may not exist) lets both old and new checkpoints
    reconstruct the exact architecture they were trained with.
    """
    state = checkpoint.get("model_state_dict", {})
    weight = state.get("task_embedding.weight")
    if weight is not None:
        return int(weight.shape[1])
    hp = checkpoint.get("hyperparameters", {})
    return int(hp.get("task_embed_dim", d_model))


def build_model_from_checkpoint(checkpoint: dict) -> JointNMRSSLModel:
    hp = checkpoint.get("hyperparameters", {})
    d_model = int(hp.get("d_model", 192))
    return JointNMRSSLModel(
        spectrum_length=int(checkpoint["spectrum_length"]),
        mask_bin_size=int(checkpoint.get("mask_bin_size", hp.get("mask_bin_size", 1024))),
        jigsaw_bin_sizes=[int(b) for b in checkpoint.get("jigsaw_bin_sizes", hp.get("jigsaw_bin_sizes", SUPPORTED_BIN_SIZES))],
        d_model=d_model,
        nhead=int(hp.get("nhead", 6)),
        num_layers=int(hp.get("num_layers", 4)),
        dim_feedforward=int(hp.get("dim_feedforward", 768)),
        dropout=float(hp.get("dropout", 0.15)),
        fourier_bands=int(hp.get("fourier_bands", 8)),
        task_embed_dim=infer_task_embed_dim(checkpoint, d_model),
    )


def build_joint_model_from_loaded_checkpoint(checkpoint: dict, device: torch.device) -> JointNMRSSLModel:
    """Construct + load a model from an already-`torch.load`-ed checkpoint dict.

    Split out from `load_joint_checkpoint` so callers that build a fresh
    classifier per LOOCV fold (e.g. `joint_ssl_eval_common.py`) can read the
    (large) checkpoint file from disk once and reuse the in-memory dict
    across folds, instead of re-reading/unpickling it every fold.
    """
    model = build_model_from_checkpoint(checkpoint)
    state = checkpoint["model_state_dict"]
    # reconstruction_skips and task_proj were both added after this model's
    # original release; older checkpoints won't have either, but the model
    # already initializes safe (identity) fallbacks for both at construction
    # time, so a missing key there is expected, not an error.
    optional_prefixes = tuple(
        prefix
        for prefix in ("reconstruction_skips.", "task_proj.")
        if not any(key.startswith(prefix) for key in state)
    )
    strict = not optional_prefixes
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if not strict:
        unexpected_keys = list(unexpected)
        missing_keys = [key for key in missing if not key.startswith(optional_prefixes)]
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                f"Checkpoint is incompatible. Missing={missing_keys}, unexpected={unexpected_keys}"
            )
    return model.to(device)


def load_joint_checkpoint(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_joint_model_from_loaded_checkpoint(checkpoint, device)
    return model, checkpoint


def train_one_run(config: RunConfig) -> dict:
    seed_everything(config.seed)
    store = SpectrumStore(config.data_path, max_samples=config.max_samples)
    normalize_resolved = parse_normalize_mode(config.normalize_input, store, config.seed)
    all_bins = sorted(set(config.jigsaw_bin_sizes + [config.mask_bin_size]))
    spectrum_length = min(int(store.spectrum_length), min((int(store.spectrum_length) // b) * b for b in all_bins))
    if spectrum_length != int(store.spectrum_length):
        print(f"Adjusted spectrum length from {store.spectrum_length} to {spectrum_length}")

    splits = split_indices(len(store), config.train_split, config.val_split, config.test_split, config.seed)
    print(
        f"Samples: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}; "
        f"spectrum_length={spectrum_length}"
    )

    datasets = {
        split: JointSpectrumDataset(store, indices, spectrum_length, normalize_resolved)
        for split, indices in splits.items()
        if len(indices) > 0
    }
    num_workers = choose_workers(config.num_workers)
    device = choose_device(config.device)
    print(f"Using device={device}, num_workers={num_workers}")

    train_loader = make_loader(datasets["train"], config.batch_size, True, num_workers, config.seed)
    val_loader = make_loader(datasets["val"], config.batch_size, False, num_workers, config.seed + 1000) if "val" in datasets else None
    test_loader = make_loader(datasets["test"], config.batch_size, False, num_workers, config.seed + 2000) if "test" in datasets else None

    model = JointNMRSSLModel(
        spectrum_length=spectrum_length,
        mask_bin_size=config.mask_bin_size,
        jigsaw_bin_sizes=config.jigsaw_bin_sizes,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        fourier_bands=config.fourier_bands,
        task_embed_dim=config.task_embed_dim,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_normalizer = EMALossNormalizer(decay=config.loss_ema_decay)
    scaler = make_grad_scaler(device)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"joint_ssl_{timestamp}"
    result_dir = Path(config.out_dir) / run_name
    model_dir = Path(config.model_dir) / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    config_payload = asdict(config)
    config_payload.update(
        {
            "run_name": run_name,
            "normalize_resolved": bool(normalize_resolved),
            "spectrum_length_resolved": int(spectrum_length),
            "architecture": "joint_masked_multibin_jigsaw_v1",
        }
    )
    save_json(result_dir / "config.json", config_payload)

    history: list[dict] = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_path = model_dir / f"{run_name}_best.pth"
    monitor_metric = config.monitor_metric

    for epoch in range(1, config.epochs + 1):
        for group in optimizer.param_groups:
            group["lr"] = config.lr * lr_factor(epoch - 1, config.warmup_epochs, config.epochs, config.min_lr, config.lr)

        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            config,
            loss_normalizer,
            epoch,
            "train",
        )
        monitor_metrics = train_metrics
        val_metrics = None
        if val_loader is not None:
            val_metrics = run_epoch(
                model,
                val_loader,
                None,
                device,
                scaler,
                config,
                loss_normalizer,
                epoch,
                "val",
            )
            monitor_metrics = val_metrics

        for split, metrics in (("train", train_metrics), ("val", val_metrics)):
            if metrics is None:
                continue
            row = {"epoch": epoch, "split": split, "lr": optimizer.param_groups[0]["lr"], **metrics}
            history.append(row)
            append_metrics_csv(result_dir / "metrics.csv", row)

        if monitor_metric not in monitor_metrics:
            raise KeyError(f"Monitor metric {monitor_metric!r} is not available. Choose from {sorted(monitor_metrics)}")
        monitor_value = monitor_metrics[monitor_metric]
        improved = monitor_value < best_val_loss - 1e-6
        if improved:
            best_val_loss = monitor_value
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                loss_normalizer,
                epoch,
                best_val_loss,
                config,
                history,
                normalize_resolved,
            )
        else:
            epochs_without_improvement += 1

        if config.save_every > 0 and epoch % config.save_every == 0:
            save_checkpoint(
                model_dir / f"{run_name}_epoch_{epoch}.pth",
                model,
                optimizer,
                loss_normalizer,
                epoch,
                best_val_loss,
                config,
                history,
                normalize_resolved,
            )

        val_text = (
            f", val_balanced={val_metrics['loss']:.4f}, val_raw={val_metrics['raw_loss']:.4f}"
            if val_metrics
            else ""
        )
        print(
            f"Epoch {epoch}: train_balanced={train_metrics['loss']:.4f}, "
            f"train_raw={train_metrics['raw_loss']:.4f}, "
            f"masked={train_metrics['masked_loss']:.4f}, "
            f"jigsaw_acc={train_metrics['jigsaw_token_accuracy']:.3f}{val_text}, "
            f"monitor={monitor_metric}, best={best_val_loss:.4f}, "
            f"patience={epochs_without_improvement}/{config.patience}"
        )
        plot_curves(history, result_dir / "training_curves.png")

        if epochs_without_improvement >= config.patience:
            print(f"Early stopping after {config.patience} epochs without improvement.")
            break

    if test_loader is not None:
        test_metrics = run_epoch(
            model,
            test_loader,
            None,
            device,
            scaler,
            config,
            loss_normalizer,
            epoch,
            "test",
        )
        save_json(result_dir / "test_metrics.json", test_metrics)

    summary = {
        "run_name": run_name,
        "best_val_loss": best_val_loss,
        "monitor_metric": monitor_metric,
        "best_checkpoint": str(best_path),
        "epochs_completed": epoch,
        "mask_bin_size": config.mask_bin_size,
        "jigsaw_bin_sizes": config.jigsaw_bin_sizes,
        "mask_ratio_range": [config.mask_ratio_min, config.mask_ratio_max],
        "results_dir": str(result_dir),
    }
    save_json(result_dir / "summary.json", summary)
    return summary


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", nargs="+", default=["data/combined/combine_unique_Water_EDTA_Suppressed.npy"])
    parser.add_argument("--out-dir", default="results/joint_ssl")
    parser.add_argument("--model-dir", default="models/joint_ssl")
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
    parser.add_argument("--fourier-bands", type=int, default=8)
    parser.add_argument(
        "--task-embed-dim",
        type=int,
        default=8,
        help="Bottleneck width for the task-id signal, projected up to d_model. "
        "Small values (vs. d_model) discourage the encoder from forking into "
        "near-disjoint per-task sub-networks.",
    )
    parser.add_argument("--mask-bin-size", type=int, default=1024, choices=SUPPORTED_BIN_SIZES)
    parser.add_argument("--jigsaw-bin-sizes", nargs="+", type=int, default=list(SUPPORTED_BIN_SIZES))
    parser.add_argument("--mask-ratio-min", type=float, default=0.20)
    parser.add_argument("--mask-ratio-max", type=float, default=0.60)
    parser.add_argument("--unmasked-recon-weight", type=float, default=0.1)
    parser.add_argument("--jigsaw-weight", type=float, default=1.0)
    parser.add_argument("--coverage-weight", type=float, default=0.0)
    parser.add_argument("--loss-ema-decay", type=float, default=0.98)
    parser.add_argument(
        "--peak-top-fraction",
        type=float,
        default=0.175,
        help="Restrict masked-reconstruction loss to the top fraction of bins by "
        "magnitude per spectrum (the rest are near-baseline/noise and excluded). "
        "1.0 disables selection and uses all bins.",
    )
    parser.add_argument(
        "--soft-jigsaw-sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma in bins for soft jigsaw targets. 0 restores hard CE targets.",
    )
    parser.add_argument(
        "--recon-skip-weight",
        type=float,
        default=0.3,
        help="Weight for reconstruction skip from masked input bins to output unmasked bins. 0 disables it.",
    )
    parser.add_argument(
        "--monitor-metric",
        choices=[
            "raw_loss",
            "loss",
            "balanced_loss",
            "masked_loss",
            "jigsaw_loss",
            "masked_loss_norm",
            "jigsaw_loss_norm",
        ],
        default="raw_loss",
        help="Metric used for checkpoint selection and early stopping. raw_loss is comparable across epochs.",
    )
    parser.add_argument(
        "--deterministic-eval",
        type=str2bool,
        default=True,
        help="Use fixed validation/test masks and jigsaw shuffles for comparable validation loss.",
    )
    parser.add_argument("--eval-seed", type=int, default=12345)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
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

    if not 0.0 < args.mask_ratio_min <= args.mask_ratio_max < 1.0:
        parser.error("--mask-ratio-min and --mask-ratio-max must satisfy 0 < min <= max < 1")
    if args.d_model % args.nhead:
        parser.error("--d-model must be divisible by --nhead")
    if not 0.0 <= args.loss_ema_decay < 1.0:
        parser.error("--loss-ema-decay must satisfy 0 <= decay < 1")
    if not 0.0 < args.peak_top_fraction <= 1.0:
        parser.error("--peak-top-fraction must satisfy 0 < fraction <= 1")
    if args.soft_jigsaw_sigma < 0:
        parser.error("--soft-jigsaw-sigma must be non-negative")
    if args.recon_skip_weight < 0:
        parser.error("--recon-skip-weight must be non-negative")

    num_workers: int | str
    num_workers = args.num_workers if args.num_workers == "auto" else int(args.num_workers)
    return RunConfig(
        data_path=args.data_path,
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
        fourier_bands=args.fourier_bands,
        task_embed_dim=args.task_embed_dim,
        mask_bin_size=args.mask_bin_size,
        jigsaw_bin_sizes=validate_bin_sizes(args.jigsaw_bin_sizes),
        mask_ratio_min=args.mask_ratio_min,
        mask_ratio_max=args.mask_ratio_max,
        unmasked_recon_weight=args.unmasked_recon_weight,
        jigsaw_weight=args.jigsaw_weight,
        coverage_weight=args.coverage_weight,
        loss_ema_decay=args.loss_ema_decay,
        peak_top_fraction=args.peak_top_fraction,
        soft_jigsaw_sigma=args.soft_jigsaw_sigma,
        recon_skip_weight=args.recon_skip_weight,
        monitor_metric=args.monitor_metric,
        deterministic_eval=args.deterministic_eval,
        eval_seed=args.eval_seed,
        label_smoothing=args.label_smoothing,
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
    summary = train_one_run(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
