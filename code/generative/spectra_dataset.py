"""Dataset + loss-weighting utilities for the NMR spectra generative models.

Mirrors the exclusion ranges used in code/analysis/peak_extraction.py:
water and EDTA solvent-suppression windows carry no metabolite signal, so the
VAE reconstruction loss can optionally downweight them; canonical peak
positions (from a peak_extraction.py run) can optionally be upweighted so the
VAE spends its limited capacity on the sharp, information-carrying features
rather than the flat baseline that dominates the point count.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

WATER_SUPPRESSION_RANGE = (62_500, 68_000)
EDTA_SEARCH_RANGE = (72_000, 74_000)


class SpectraDataset(Dataset):
    def __init__(self, data_path, indices=None, max_samples=None):
        self.data = np.load(data_path, mmap_mode="r")
        n = self.data.shape[0]
        if indices is None:
            indices = np.arange(n)
        if max_samples is not None:
            indices = indices[:max_samples]
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = np.asarray(self.data[self.indices[i]], dtype=np.float32)
        row = np.nan_to_num(row, nan=0.0)
        return torch.from_numpy(row).unsqueeze(0)  # (1, L)


def train_val_split(n, val_frac=0.05, seed=42):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    return perm[n_val:], perm[:n_val]


def build_loss_weight_mask(
    length,
    peaks_csv=None,
    peak_halfwidth=150,
    peak_weight=5.0,
    suppression_weight=0.1,
):
    """Returns a (length,) float32 array of per-point reconstruction-loss weights."""
    weight = np.ones(length, dtype=np.float32)

    lo, hi = WATER_SUPPRESSION_RANGE
    weight[lo:hi] = suppression_weight
    lo, hi = EDTA_SEARCH_RANGE
    weight[lo:hi] = suppression_weight

    if peaks_csv is not None and Path(peaks_csv).exists():
        peaks = pd.read_csv(peaks_csv)
        for pos in peaks["point_index"].astype(int):
            lo = max(0, pos - peak_halfwidth)
            hi = min(length, pos + peak_halfwidth)
            weight[lo:hi] = np.maximum(weight[lo:hi], peak_weight)

    return weight
