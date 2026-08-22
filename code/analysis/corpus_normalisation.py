#!/usr/bin/env python3
"""Per-spectrum normalisation applied lazily over a memory-mapped corpus.

WHY THIS EXISTS. The corpus arrays are ~10 GB, so materialising a second
normalised copy per scheme is wasteful, and several readers slice *columns*
(`arr[rows, lo:hi]`) rather than whole rows. A per-spectrum normaliser must use
statistics of the FULL row even when only a column window is being read, or the
window gets scaled by its own partial statistics — which is silently wrong.

This module precomputes per-row (offset, scale) once in a single chunked pass and
then applies `(x - offset) / scale` to whatever block is requested. Every
normaliser here is of that affine, per-row form:

    none       offset 0,        scale 1
    rowminmax  offset row.min,  scale row.max - row.min
    unit_area  offset 0,        scale sum|row|

`pqn` is deliberately absent: it needs a corpus-median reference spectrum and,
as measured in `normaliser_comparison.py`, it is numerically unstable on this
data (raw range order 1e8, and it collapses MTBLS563 from 0.687 to 0.365).

CHOOSING A NORMALISER FOR *MEASUREMENT* vs *TRAINING* — these differ, see
docs/PI_outline.md §7.1:
  - Training / reported evaluation: `rowminmax`. Best-conditioned MSE targets
    ([0, 1.15] after one global scale) and best mean classical accuracy.
  - Distributions of peak intensities: `unit_area`. Under min-max the unit is
    "this spectrum's tallest peak", which occupies 21 distinct chemical
    positions across 400 spectra and sits below 0.5 ppm — where no metabolite
    resonates — in 19% of them. That is not a usable unit for an intensity
    distribution.
"""
from __future__ import annotations

import numpy as np

MODES = ("none", "rowminmax", "unit_area")


class NormalisedCorpus:
    """Read-only, lazily normalised view over an .npy memory map.

    Supports the access patterns the peak pipeline uses: `arr[i]`,
    `arr[a:b]`, `arr[index_array]`, and any of those with a trailing column
    slice. Exposes `.shape`, `.dtype`, `len()`.
    """

    def __init__(self, path, mode: str = "none", chunk_size: int = 500,
                 verbose: bool = True):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self._arr = np.load(path, mmap_mode="r")
        if self._arr.ndim != 2:
            raise ValueError(f"expected a 2-D corpus, got shape {self._arr.shape}")
        self.shape = self._arr.shape
        self.dtype = np.dtype(np.float64)
        self.mode = mode
        n = self.shape[0]

        if mode == "none":
            self._offset = np.zeros(n)
            self._scale = np.ones(n)
        else:
            self._offset = np.zeros(n)
            self._scale = np.ones(n)
            if verbose:
                print(f"[{mode}] computing per-row statistics over {n} spectra ...")
            for start in range(0, n, chunk_size):
                stop = min(start + chunk_size, n)
                block = np.asarray(self._arr[start:stop], dtype=np.float64)
                if mode == "rowminmax":
                    lo = block.min(axis=1)
                    hi = block.max(axis=1)
                    self._offset[start:stop] = lo
                    self._scale[start:stop] = hi - lo
                else:  # unit_area
                    self._scale[start:stop] = np.abs(block).sum(axis=1)
                if verbose:
                    print(f"\r  {stop}/{n}", end="", flush=True)
            if verbose:
                print()
            bad = int((~np.isfinite(self._scale)).sum() + (self._scale <= 0).sum())
            if bad:
                print(f"  WARNING: {bad} row(s) had a non-positive or non-finite "
                      f"scale; those rows are left unscaled.")
            self._scale = np.where(np.isfinite(self._scale) & (self._scale > 0),
                                   self._scale, 1.0)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        row_key = key[0] if isinstance(key, tuple) else key
        block = np.asarray(self._arr[key], dtype=np.float64)
        if self.mode == "none":
            return block
        off = self._offset[row_key]
        sc = self._scale[row_key]
        if np.ndim(off) == 0:
            return (block - off) / sc
        return (block - off[:, None]) / sc[:, None]

    @property
    def row_scale(self):
        """The per-row divisor actually applied. Useful for diagnostics."""
        return self._scale.copy()


def open_corpus(path, mode: str = "none", chunk_size: int = 500, verbose: bool = True):
    """Return a plain memory map for mode='none', else a NormalisedCorpus."""
    if mode == "none":
        return np.load(path, mmap_mode="r")
    return NormalisedCorpus(path, mode=mode, chunk_size=chunk_size, verbose=verbose)
