import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path

def max_normalize_spectra(spectra):
    """Per-spectrum max-abs normalization (preserves sign).
    spectra: numpy array (n_samples, n_points)
    returns: normalized numpy array (same shape)
    """
    spectra = np.asarray(spectra, dtype=float)
    # handle NaN/Inf by replacing them with 0
    spectra = np.nan_to_num(spectra, posinf=0.0, neginf=0.0)
    max_vals = np.max(np.abs(spectra), axis=1)
    # avoid division by zero
    nonzero = max_vals > 1e-12
    normalized = spectra.copy()
    normalized[nonzero] = normalized[nonzero] / max_vals[nonzero, None]
    # leave constant-zero spectra as-is (all zeros)
    return normalized

def quantile_normalize_spectra(spectra, percentile=99):
    """Normalize by high percentile - robust to outliers and baseline.
    spectra: numpy array (n_samples, n_points)
    returns: normalized numpy array (same shape)
    """
    normalized_spectra = np.zeros_like(spectra)
    
    for i in range(len(spectra)):
        spectrum = spectra[i]
        # Use 99th percentile instead of max (more robust)
        ref_val = np.percentile(np.abs(spectrum), percentile)
        if ref_val > 1e-8:
            normalized_spectra[i] = spectrum / ref_val
        else:
            normalized_spectra[i] = spectrum
    
    return normalized_spectra

def log_normalize_spectra(spectra):
    """Log normalization - handles large dynamic ranges.
    Accepts a numpy array (n_samples, n_points) or 1-D array (n_points).
    Returns a numpy array of the same shape (dtype float).
    """
    spectra = np.asarray(spectra, dtype=float)
    # handle NaN/Inf
    spectra = np.nan_to_num(spectra, posinf=0.0, neginf=0.0)

    if spectra.ndim == 1:
        min_val = np.min(spectra)
        shifted = spectra - min_val + 1.0
        return np.log1p(shifted)

    # vectorized per-spectrum shift then log1p
    min_vals = np.min(spectra, axis=1, keepdims=True)
    shifted = spectra - min_vals + 1.0
    return np.log1p(shifted)

def pqn_normalize_spectra(spectra, reference=None):
    """PQN normalization - accounts for dilution effects.
    spectra: numpy array (n_samples, n_points)
    reference: optional 1D numpy array (n_points), if None uses median spectrum
    returns: normalized numpy array (same shape)
    """
    spectra = np.asarray(spectra, dtype=float)
    # handle NaN/Inf
    spectra = np.nan_to_num(spectra, posinf=0.0, neginf=0.0)
    if reference is None:
        reference = np.median(spectra, axis=0)
    else:
        reference = np.asarray(reference, dtype=float)
        reference = np.nan_to_num(reference, posinf=0.0, neginf=0.0)
    
    normalized_spectra = np.zeros_like(spectra)
    ref_mask = np.abs(reference) > 1e-8

    for i in range(len(spectra)):
        spectrum = spectra[i]
        quotients = np.zeros_like(spectrum)
        quotients[ref_mask] = spectrum[ref_mask] / (reference[ref_mask] + 1e-8)
        # Use median of quotients as scaling factor (ignore zeros)
        valid_quotients = quotients[ref_mask]
        if valid_quotients.size > 0:
            median_quotient = np.median(valid_quotients)
            if np.abs(median_quotient) > 1e-8:
                normalized_spectra[i] = spectrum / median_quotient
            else:
                normalized_spectra[i] = spectrum
        else:
            normalized_spectra[i] = spectrum

    return normalized_spectra

def visualize_random_spectra(
    data,
    n_samples=9,
    figsize=(12, 8),
    save_path=None,
    random_state=None,
    save_normalized=False,
):
    """Visualize random spectra after max normalization in a memory-efficient way.

    Behavior:
    - If `data` is a file path to a .npy, the file is memory-mapped and only the
      sampled rows are loaded into memory (no full-array allocation).
    - If `data` is an in-memory numpy array, the function behaves like before.

    Parameters:
    - data: path to .npy file or numpy array (n_samples, n_points)
    - n_samples: number of random spectra to show (grid layout computed automatically)
    - figsize: matplotlib figure size tuple
    - save_path: optional path to save the figure (PNG)
    - random_state: optional int for reproducibility
    - save_normalized: if True, saves a normalized file; default False.

    Returns:
    - sampled_indices: numpy array of indices (relative to the cleaned/normalized subset)
    """

    rng = np.random.default_rng(random_state)

    # Case A: file path -> use memory-mapping and sample rows
    if isinstance(data, str):
        arr = np.load(data, mmap_mode="r")
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array in file {data}; got shape {arr.shape}")
        N_total = int(arr.shape[0])
        if N_total == 0:
            raise ValueError("No spectra in the file.")

        n = min(int(n_samples), N_total)
        chosen_idx = rng.choice(N_total, size=n, replace=False)

        # Load only selected rows into memory
        spectra = np.asarray(arr[chosen_idx], dtype=float)
        # Proceed with cleaning & normalization on the sampled rows only
        spectra = np.nan_to_num(spectra, posinf=0.0, neginf=0.0)
        non_const_mask = np.std(spectra, axis=1) > 1e-12
        if non_const_mask.sum() == 0:
            clean = spectra
            rel_indices = np.arange(spectra.shape[0], dtype=int)
        else:
            clean = spectra[non_const_mask]
            rel_indices = np.nonzero(non_const_mask)[0]

        normalized = max_normalize_spectra(clean)

        # Optionally save only the sampled normalized subset (not the full dataset)
        if save_normalized:
            outname = Path(data).stem + f"_normalized_sample_{normalized.shape[0]}.npy"
            np.save(outname, normalized)

        # For plotting we will use the normalized sampled subset; return original chosen indices if desired
        sampled_indices = chosen_idx[rel_indices]

    else:
        # Case B: in-memory array
        spectra_all = np.asarray(data, dtype=float)
        spectra_all = np.nan_to_num(spectra_all, posinf=0.0, neginf=0.0)
        non_const_mask = np.std(spectra_all, axis=1) > 1e-12
        if non_const_mask.sum() == 0:
            clean_all = spectra_all
        else:
            clean_all = spectra_all[non_const_mask]

        normalized = max_normalize_spectra(clean_all)
        N = normalized.shape[0]
        if N == 0:
            raise ValueError("No spectra available after cleaning.")

        n = min(int(n_samples), N)
        chosen_rel = rng.choice(N, size=n, replace=False)
        normalized = normalized[chosen_rel]
        sampled_indices = np.asarray(chosen_rel, dtype=int)

    # Plotting - `normalized` contains the rows we will draw (at most n_samples)
    N_plot = normalized.shape[0]
    if N_plot == 0:
        raise ValueError("No spectra available after cleaning.")

    n_plot = min(int(n_samples), N_plot)
    cols = int(np.ceil(np.sqrt(n_plot)))
    rows = int(np.ceil(n_plot / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    for i, ax in enumerate(axes.flat):
        if i < n_plot:
            s = normalized[i].flatten()
            ax.plot(s, color="tab:blue", lw=1)
            ax.set_title(f"Sample idx={sampled_indices[i]}\nmin={s.min():.3f}, max={s.max():.3f}", fontsize=9)
            ax.set_xlabel("Freq point", fontsize=8)
            ax.set_ylabel("Normalized intensity", fontsize=8)
            ax.grid(alpha=0.25)
        else:
            ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    plt.show()

    return np.asarray(sampled_indices, dtype=int)

# Example usage:
visualize_random_spectra('data/mtbls326/MTBLS326_aligned_spectra_WS625to680Zero.npy', n_samples=9, save_path='data/mtbls326/MTBLS326_WS625to680Zero.png', random_state=42)