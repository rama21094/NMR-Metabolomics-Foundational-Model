"""
Standalone testing script for NMR Masked Autoencoder
Run this after training to evaluate model performance
"""

import torch
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import pandas as pd
import json
from pathlib import Path
import os

# Ensure every process (main + spawned workers) uses writable cache dirs.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

# Import your training modules
from trainer_revised import (
    NMRMaskedAutoencoder, 
    NMRSpectrumDataset
)
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import math
import warnings

MAX_SAMPLES_PER_TASK = 1000

# Suppress repeated non-actionable Transformer nested tensor warning.
warnings.filterwarnings(
    "ignore",
    message=r".*enable_nested_tensor is True, but self.use_nested_tensor is False.*",
    category=UserWarning,
)


def enforce_zero_window_baseline(
    spectra,
    start_idx=62600,
    end_idx=67000,
    atol=1e-10,
    rtol=1e-8,
    verbose=False
):
    """
    Ensure a known zeroed window is exactly zero by removing constant row offsets.

    For each spectrum row, if values in spectra[row, start_idx:end_idx] are all
    (approximately) equal to a constant c and c is not approximately zero, the
    entire row is shifted by -c.
    """
    if spectra.ndim != 2:
        return spectra, {
            "total_rows": 0,
            "shifted_rows": 0,
            "already_zero_rows": 0,
            "non_constant_rows": 0,
            "window": [start_idx, end_idx],
        }

    n_rows, n_cols = spectra.shape
    s = max(0, int(start_idx))
    e = min(int(end_idx), n_cols)
    if s >= e:
        return spectra, {
            "total_rows": n_rows,
            "shifted_rows": 0,
            "already_zero_rows": n_rows,
            "non_constant_rows": 0,
            "window": [s, e],
        }

    shifted_rows = 0
    already_zero_rows = 0
    non_constant_rows = 0

    for i in range(n_rows):
        window_vals = spectra[i, s:e]
        ref_val = window_vals[0]
        is_constant = np.allclose(window_vals, ref_val, atol=atol, rtol=rtol)
        if not is_constant:
            non_constant_rows += 1
            continue

        if np.isclose(ref_val, 0.0, atol=atol, rtol=rtol):
            already_zero_rows += 1
            continue

        spectra[i] = spectra[i] - ref_val
        shifted_rows += 1

    info = {
        "total_rows": n_rows,
        "shifted_rows": shifted_rows,
        "already_zero_rows": already_zero_rows,
        "non_constant_rows": non_constant_rows,
        "window": [s, e],
    }
    if verbose:
        print(
            f"Zero-window baseline check [{s}:{e}] -> "
            f"shifted: {shifted_rows}, already_zero: {already_zero_rows}, "
            f"non_constant: {non_constant_rows}"
        )
    return spectra, info


def _resolve_worker_devices(requested_device, workers):
    """
    Resolve per-worker device assignment from a user-provided device string.

    Rules:
    - 'cuda:N' -> all workers use that exact GPU.
    - 'cuda'   -> workers round-robin across visible GPUs.
    - 'cpu'    -> all workers use CPU.
    """
    dev = str(requested_device).strip().lower()
    workers = max(1, int(workers))

    if not torch.cuda.is_available() or dev == "cpu":
        return ["cpu"] * workers

    n_gpus = torch.cuda.device_count()
    if dev.startswith("cuda:"):
        try:
            gpu_id = int(dev.split(":")[1])
        except Exception:
            gpu_id = 0
        gpu_id = max(0, min(gpu_id, n_gpus - 1))
        if workers == 1:
            return [f"cuda:{gpu_id}"]
        return [f"cuda:{(gpu_id + wid) % n_gpus}" for wid in range(workers)]

    if dev == "cuda":
        return [f"cuda:{wid % n_gpus}" for wid in range(workers)]

    # Fallback for unknown strings.
    return ["cpu"] * workers


def _is_data_in_unit_range(spectra, tol=1e-6):
    return float(np.min(spectra)) >= -tol and float(np.max(spectra)) <= (1.0 + tol)


_WORKER_SHARED_SPECTRA = None
_WORKER_DATASET = None
_WORKER_MODEL = None


def _worker_init(data_path, normalize_spectra, patch_size, mask_ratio):
    """Initializer for worker processes.

    Load the shared spectra buffer once per worker and create a dataset that
    is reused for all tasks in that worker process.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    global _WORKER_SHARED_SPECTRA, _WORKER_DATASET
    _WORKER_SHARED_SPECTRA = np.load(data_path, mmap_mode='r')
    _WORKER_DATASET = NMRSpectrumDataset(
        _WORKER_SHARED_SPECTRA,
        mask_ratio=mask_ratio,
        patch_size=patch_size,
        mask_strategy='sparse_random',
        normalize_input=normalize_spectra
    )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _align_original_masked_baseline(orig, masked, mask_np, patch_size, start_idx=62500, end_idx=68000, tol=1e-8):
    """
    Apply the same baseline offset to original and masked (unmasked points only)
    using unmasked values in a known zero window.
    """
    n = len(orig)
    s = max(0, min(int(start_idx), n))
    e = max(0, min(int(end_idx), n))
    if s >= e:
        return orig, masked

    point_mask = np.zeros(n, dtype=bool)
    for i, is_masked in enumerate(mask_np):
        if is_masked:
            ps = i * patch_size
            pe = min(ps + patch_size, n)
            point_mask[ps:pe] = True

    unmasked_window = ~point_mask[s:e]
    if not np.any(unmasked_window):
        return orig, masked

    offset = np.median(masked[s:e][unmasked_window])
    if np.abs(offset) <= tol:
        return orig, masked

    orig = orig - offset
    masked = masked.copy()
    masked[~point_mask] = masked[~point_mask] - offset
    return orig, masked


def _zero_baseline_signal(signal, start_idx=62500, end_idx=68000, tol=1e-8):
    """
    Shift a 1D signal so the known zero window is centered at zero.
    """
    n = len(signal)
    s = max(0, min(int(start_idx), n))
    e = max(0, min(int(end_idx), n))
    if s >= e:
        return signal
    offset = np.median(signal[s:e])
    if np.abs(offset) <= tol:
        return signal
    return signal - offset


def _worker_run(
    worker_id,
    idx_list,
    model_path,
    data_path,
    n_masks_per_sample,
    patch_size,
    mask_ratio,
    device_for_worker,
    normalize_spectra=None,
    infer_batch_size=8,
    result_dir=None
):
    """Worker function executed in a separate process."""
    import torch
    import numpy as np
    from trainer_revised import NMRMaskedAutoencoder, NMRSpectrumDataset
    from scipy import stats
    from scipy.signal import find_peaks
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    device = torch.device(device_for_worker if torch.cuda.is_available() and 'cuda' in device_for_worker else 'cpu')
    global _WORKER_DATASET, _WORKER_MODEL
    spectra = None
    if _WORKER_DATASET is None:
        # Fallback in case this worker was not initialized.
        spectra = np.load(data_path, mmap_mode='r')
        if normalize_spectra is None:
            normalize_spectra = not _is_data_in_unit_range(spectra)
        dataset = NMRSpectrumDataset(
            spectra,
            mask_ratio=mask_ratio,
            patch_size=patch_size,
            mask_strategy='sparse_random',
            normalize_input=normalize_spectra
        )
    else:
        dataset = _WORKER_DATASET
        spectra = _WORKER_SHARED_SPECTRA

    if _WORKER_MODEL is not None:
        model = _WORKER_MODEL
    else:
        assert spectra is not None, "Spectra must be loaded before model construction"
        checkpoint = torch.load(model_path, map_location=device)
        spectrum_length = spectra.shape[1]
        model = NMRMaskedAutoencoder(
            spectrum_length=spectrum_length,
            patch_size=patch_size,
            d_model=128,
            nhead=4,
            num_layers=3,
            dim_feedforward=256,
            dropout=0.2
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        del checkpoint

    local_results = {
        'sample_idx': [], 'mask_idx': [], 'mse_overall': [], 'mae_overall': [], 'rmse_overall': [], 'r2_overall': [],
        'pearson_overall': [], 'spearman_overall': [], 'mse_masked': [], 'mae_masked': [], 'pearson_masked': [],
        'peak_recovery_masked': [], 'mse_unmasked': [], 'mae_unmasked': [], 'pearson_unmasked': [],
        'n_peaks_original': [], 'n_peaks_reconstructed': [], 'peak_position_error': [], 'peak_intensity_error': [],
        'peak_f1': [], 'local_corr_mean': [], 'local_corr_std': []
    }

    infer_batch_size = max(1, int(infer_batch_size))
    if device.type == "cpu":
        infer_batch_size = 1

    def _process_one(sample_idx, mask_idx, orig, masked_np_input, mask_np, recon):
                orig, masked_np_input = _align_original_masked_baseline(
                    orig,
                    masked_np_input,
                    mask_np,
                    patch_size
                )
                orig = _zero_baseline_signal(orig, start_idx=62501, end_idx=68000)

                local_results['sample_idx'].append(sample_idx)
                local_results['mask_idx'].append(mask_idx)

                local_results['mse_overall'].append(mean_squared_error(orig, recon))
                local_results['mae_overall'].append(mean_absolute_error(orig, recon))
                local_results['rmse_overall'].append(np.sqrt(local_results['mse_overall'][-1]))
                try:
                    local_results['r2_overall'].append(r2_score(orig, recon))
                except Exception:
                    local_results['r2_overall'].append(np.nan)

                try:
                    p, _ = stats.pearsonr(orig, recon)
                except Exception:
                    p = np.nan
                try:
                    s, _ = stats.spearmanr(orig, recon)
                except Exception:
                    s = np.nan
                local_results['pearson_overall'].append(p)
                local_results['spearman_overall'].append(s)

                # Point-wise mask
                point_mask = np.zeros(len(orig), dtype=bool)
                for i, is_masked in enumerate(mask_np):
                    if is_masked:
                        start = i * patch_size
                        end = start + patch_size
                        point_mask[start:end] = True

                # Masked/unmasked metrics
                if point_mask.sum() > 0:
                    masked_orig = orig[point_mask]
                    masked_recon = recon[point_mask]
                    local_results['mse_masked'].append(mean_squared_error(masked_orig, masked_recon))
                    local_results['mae_masked'].append(mean_absolute_error(masked_orig, masked_recon))
                    try:
                        p_masked, _ = stats.pearsonr(masked_orig, masked_recon) if len(masked_orig) > 1 else (np.nan, None)
                    except Exception:
                        p_masked = np.nan
                    local_results['pearson_masked'].append(p_masked)
                    try:
                        thresh_o = np.percentile(masked_orig, 90)
                        thresh_r = np.percentile(masked_recon, 90)
                        peak_o = masked_orig > thresh_o
                        peak_r = masked_recon > thresh_r
                        intersection = (peak_o & peak_r).sum()
                        union = (peak_o | peak_r).sum()
                        local_results['peak_recovery_masked'].append(intersection / union if union > 0 else 0)
                    except Exception:
                        local_results['peak_recovery_masked'].append(np.nan)
                else:
                    local_results['mse_masked'].append(np.nan)
                    local_results['mae_masked'].append(np.nan)
                    local_results['pearson_masked'].append(np.nan)
                    local_results['peak_recovery_masked'].append(np.nan)

                if (~point_mask).sum() > 0:
                    unmasked_orig = orig[~point_mask]
                    unmasked_recon = recon[~point_mask]
                    local_results['mse_unmasked'].append(mean_squared_error(unmasked_orig, unmasked_recon))
                    local_results['mae_unmasked'].append(mean_absolute_error(unmasked_orig, unmasked_recon))
                    try:
                        p_un, _ = stats.pearsonr(unmasked_orig, unmasked_recon) if len(unmasked_orig) > 1 else (np.nan, None)
                    except Exception:
                        p_un = np.nan
                    local_results['pearson_unmasked'].append(p_un)
                else:
                    local_results['mse_unmasked'].append(np.nan)
                    local_results['mae_unmasked'].append(np.nan)
                    local_results['pearson_unmasked'].append(np.nan)

                # Peak metrics
                try:
                    height_orig = np.percentile(orig, 75)
                    height_recon = np.percentile(recon, 75)
                    peaks_o, _ = find_peaks(orig, height=height_orig, distance=10)
                    peaks_r, _ = find_peaks(recon, height=height_recon, distance=10)
                except Exception:
                    peaks_o = np.array([])
                    peaks_r = np.array([])

                local_results['n_peaks_original'].append(len(peaks_o))
                local_results['n_peaks_reconstructed'].append(len(peaks_r))

                tolerance = 20
                matched = 0
                pos_errors = []
                int_errors = []
                for po in peaks_o:
                    if len(peaks_r) > 0:
                        dists = np.abs(peaks_r - po)
                        if dists.min() <= tolerance:
                            idx = np.argmin(dists)
                            matched += 1
                            pos_errors.append(dists[idx])
                            int_errors.append(abs(orig[po] - recon[peaks_r[idx]]))

                if len(peaks_o) > 0 or len(peaks_r) > 0:
                    prec = matched / len(peaks_r) if len(peaks_r) > 0 else 0
                    rec = matched / len(peaks_o) if len(peaks_o) > 0 else 0
                    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                else:
                    f1 = 1.0

                local_results['peak_position_error'].append(np.mean(pos_errors) if pos_errors else np.nan)
                local_results['peak_intensity_error'].append(np.mean(int_errors) if int_errors else np.nan)
                local_results['peak_f1'].append(f1)

                # Local correlation
                window = 100
                local_corrs = []
                for i in range(0, len(orig) - window, window // 2):
                    w_o = orig[i:i+window]
                    w_r = recon[i:i+window]
                    # Skip invalid/near-constant windows where Pearson is undefined.
                    if len(w_o) <= 1:
                        continue
                    if not np.all(np.isfinite(w_o)) or not np.all(np.isfinite(w_r)):
                        continue
                    if np.ptp(w_o) <= 1e-8 or np.ptp(w_r) <= 1e-8:
                        continue

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=stats.ConstantInputWarning)
                        pearson_result = stats.pearsonr(w_o, w_r)
                        c = np.asarray(pearson_result[0], dtype=np.float64).item()
                    if np.isfinite(c):
                        local_corrs.append(c)

                local_results['local_corr_mean'].append(np.mean(local_corrs) if local_corrs else np.nan)
                local_results['local_corr_std'].append(np.std(local_corrs) if local_corrs else np.nan)

    with torch.no_grad():
        batch_samples = []
        batch_original = []
        batch_masked = []
        batch_mask = []

        def flush_batch():
            nonlocal batch_samples, batch_original, batch_masked, batch_mask
            if not batch_samples:
                return
            originals = torch.stack(batch_original, dim=0).to(device, non_blocking=True)
            maskeds = torch.stack(batch_masked, dim=0).to(device, non_blocking=True)
            masks = torch.stack(batch_mask, dim=0).to(device, non_blocking=True)
            try:
                recon_batch, _ = model(maskeds, masks)
            except Exception as e:
                print(f"Worker {worker_id} skipping batch with samples {[s for s, _ in batch_samples]} due to {type(e).__name__}: {e}")
                batch_samples = []
                batch_original = []
                batch_masked = []
                batch_mask = []
                return

            orig_np_batch = originals.cpu().numpy()
            masked_np_batch = maskeds.cpu().numpy()
            mask_np_batch = masks.cpu().numpy()
            recon_np_batch = recon_batch.cpu().numpy()

            for b, (sample_idx, mask_idx) in enumerate(batch_samples):
                _process_one(
                    sample_idx,
                    mask_idx,
                    orig_np_batch[b].flatten(),
                    masked_np_batch[b].flatten(),
                    mask_np_batch[b].flatten(),
                    recon_np_batch[b].flatten(),
                )
                pass

            batch_samples = []
            batch_original = []
            batch_masked = []
            batch_mask = []

        for sample_idx in idx_list:
            try:
                sample = dataset[sample_idx]
            except Exception as e:
                print(f"Worker {worker_id} skipping sample {sample_idx} due to {type(e).__name__}: {e}")
                continue
            for mask_idx in range(n_masks_per_sample):
                batch_samples.append((sample_idx, mask_idx))
                batch_original.append(sample['original'])
                batch_masked.append(sample['masked'])
                batch_mask.append(sample['mask'])
                if len(batch_samples) >= infer_batch_size:
                    flush_batch()

        flush_batch()

    if result_dir is not None:
        out_path = Path(result_dir) / f"_worker_results_{worker_id}_{os.getpid()}.csv"
        pd.DataFrame(local_results).to_csv(out_path, index=False)
        return {"result_file": str(out_path), "n_rows": int(len(local_results["sample_idx"]))}

    # Return local results (serial path)
    return local_results

def test_nmr_model(model_path, data_path, save_dir='results/testing/combined', 
                   n_samples=None, mask_realizations_per_sample=5, 
                   patch_size=1024, mask_ratio=0.15, device='cuda:0', 
                   suffix='', workers=None, normalize_spectra=None, infer_batch_size=8):
    """
    Main testing function
    
    Args:
        model_path: Path to saved model (.pth file)
        data_path: Path to NMR data (.npy file)
        save_dir: Directory to save results
        n_samples: Number of samples to test (None = all)
        mask_realizations_per_sample: Number of random masks per sample
        patch_size: Should match training
        mask_ratio: Should match training
        workers: Number of parallel worker processes (None = auto)
        normalize_spectra: True/False to force normalization, None = auto
    """
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {data_path}...")
    spectra = np.load(data_path, mmap_mode='r')
    print(f"Loaded spectra shape: {spectra.shape}")
    # Remove duplicate spectra immediately after loading to avoid redundant evaluation.
    unique_spectra = np.unique(spectra, axis=0)
    if unique_spectra.shape[0] != spectra.shape[0]:
        print(f"Removed {spectra.shape[0] - unique_spectra.shape[0]} duplicate spectra.")
        spectra = unique_spectra
    else:
        print("No duplicate spectra found.")
    if normalize_spectra is None:
        normalize_spectra = not _is_data_in_unit_range(spectra)
        mode = "enabled" if normalize_spectra else "skipped"
        print(f"Dataset normalization (auto): {mode}")
    else:
        mode = "enabled" if normalize_spectra else "skipped"
        print(f"Dataset normalization (forced): {mode}")
    print("Skipping full-array baseline rewrite for speed/memory (baseline handled per-sample during evaluation).")
    
    # Create dataset
    dataset = NMRSpectrumDataset(
        spectra,
        mask_ratio=mask_ratio,
        patch_size=patch_size,
        mask_strategy='sparse_random',
        normalize_input=normalize_spectra
    )
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get hyperparameters if saved
    if 'hyperparameters' in checkpoint:
        hyperparams = checkpoint['hyperparameters']
        print(f"Model hyperparameters: {hyperparams}")
    
    spectrum_length = spectra.shape[1]
    model = NMRMaskedAutoencoder(
        spectrum_length=spectrum_length,
        patch_size=patch_size,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.2
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"Testing on device: {device}")
    
    # Run testing
    total_samples = len(dataset)
    if n_samples is None:
        idx_list = list(range(total_samples))
    else:
        n_samples = min(n_samples, total_samples)
        if n_samples < total_samples:
            idx_list = np.random.choice(total_samples, size=n_samples, replace=False).tolist()
            np.random.shuffle(idx_list)
            print(f"Randomly selected {n_samples} samples from {total_samples} total samples.")
        else:
            idx_list = list(range(total_samples))
    n_samples = len(idx_list)

    if workers is None:
        if torch.cuda.is_available():
            workers = max(1, min(torch.cuda.device_count(), n_samples))
        else:
            workers = max(1, min(4, multiprocessing.cpu_count(), n_samples))
    WORKERS = max(1, int(workers))
    if torch.cuda.is_available() and str(device).startswith('cuda'):
        n_gpus = torch.cuda.device_count()
        if WORKERS > n_gpus:
            print(f"Warning: reducing worker count from {WORKERS} to {n_gpus} because CUDA inference is limited by GPU count.")
            WORKERS = n_gpus
    
    print(f"\n{'='*70}")
    print(f"Testing Configuration:")
    print(f"  Samples: {n_samples}")
    print(f"  Masks per sample: {mask_realizations_per_sample}")
    print(f"  Total reconstructions: {n_samples * mask_realizations_per_sample}")
    print(f"  Patch size: {patch_size}")
    print(f"  Mask ratio: {mask_ratio}")
    print(f"  Workers: {WORKERS}")
    print(f"{'='*70}\n")
    
    # Initialize results storage
    results = {
        'sample_idx': [],
        'mask_idx': [],
        # Overall
        'mse_overall': [],
        'mae_overall': [],
        'rmse_overall': [],
        'r2_overall': [],
        'pearson_overall': [],
        'spearman_overall': [],
        # Masked
        'mse_masked': [],
        'mae_masked': [],
        'pearson_masked': [],
        'peak_recovery_masked': [],
        # Unmasked
        'mse_unmasked': [],
        'mae_unmasked': [],
        'pearson_unmasked': [],
        # Peak metrics
        'n_peaks_original': [],
        'n_peaks_reconstructed': [],
        'peak_position_error': [],
        'peak_intensity_error': [],
        'peak_f1': [],
        # Spatial
        'local_corr_mean': [],
        'local_corr_std': [],
    }
    
    # Parallel execution: split sample indices across worker processes that each load model+data
    aggregated = []
    worker_result_files = []
    if WORKERS == 1:
        # True serial mode: avoid subprocess/CUDA fork interactions.
        idx_list = list(range(n_samples))
        chunks = [idx_list[i:i + MAX_SAMPLES_PER_TASK] for i in range(0, len(idx_list), MAX_SAMPLES_PER_TASK)]
        total_reconstructions = n_samples * mask_realizations_per_sample
        aggregated = []
        with tqdm(total=total_reconstructions, desc="Testing", unit="recon") as pbar:
            for chunk in chunks:
                part = _worker_run(
                    0,
                    chunk,
                    model_path,
                    data_path,
                    mask_realizations_per_sample,
                    patch_size,
                    mask_ratio,
                    str(device),
                    normalize_spectra=normalize_spectra,
                    infer_batch_size=infer_batch_size,
                    result_dir=None
                )
                aggregated.append(part)
                pbar.update(len(chunk) * mask_realizations_per_sample)
    elif WORKERS > 1:
        indices = list(range(n_samples))
        chunk_size = min(MAX_SAMPLES_PER_TASK, math.ceil(len(indices) / WORKERS))
        chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]
        total_reconstructions = n_samples * mask_realizations_per_sample

        actual_workers = min(WORKERS, len(chunks))
        if actual_workers < WORKERS:
            print(f"Reducing worker count from {WORKERS} to {actual_workers} because there are fewer chunks than workers.")
        worker_devices = _resolve_worker_devices(device, actual_workers)

        futures = []
        aggregated = []
        worker_result_files = []
        mp_ctx = multiprocessing.get_context("spawn")
        try:
            with ProcessPoolExecutor(
                max_workers=min(WORKERS, len(chunks)),
                mp_context=mp_ctx,
                initializer=_worker_init,
                initargs=(
                    data_path,
                    normalize_spectra,
                    patch_size,
                    mask_ratio,
                ),
            ) as executor:
                for wid, idx_list in enumerate(chunks):
                    if not idx_list:
                        continue
                    dev = worker_devices[wid % actual_workers]
                    futures.append(executor.submit(
                        _worker_run,
                        wid,
                        idx_list,
                        model_path,
                        data_path,
                        mask_realizations_per_sample,
                        patch_size,
                        mask_ratio,
                        dev,
                        normalize_spectra,
                        infer_batch_size,
                        save_dir
                    ))
            with tqdm(total=total_reconstructions, desc="Testing", unit="recon") as pbar:
                for fut in as_completed(futures):
                    try:
                        worker_out = fut.result()
                        worker_result_files.append(worker_out["result_file"])
                        pbar.update(worker_out["n_rows"])
                    except Exception as e:
                        print('Worker error:', e)

        finally:
            pass

    # Merge results for both serial and parallel paths
    if WORKERS == 1:
        for ar in aggregated:
            for k, v in ar.items():
                results[k].extend(v)
        df = pd.DataFrame(results)
    else:
        worker_dfs = []
        for fp in worker_result_files:
            worker_dfs.append(pd.read_csv(fp))
        if worker_dfs:
            df = pd.concat(worker_dfs, ignore_index=True)
        else:
            df = pd.DataFrame(results)
        for fp in worker_result_files:
            try:
                Path(fp).unlink(missing_ok=True)
            except Exception:
                pass
    
    # Save detailed results
    df.to_csv(f'{save_dir}/detailed_results{suffix}.csv', index=False)
    print(f"\nDetailed results saved to {save_dir}/detailed_results{suffix}.csv")
    
    # Compute summary statistics
    summary = compute_summary(df)
    
    # Save summary
    with open(f'{save_dir}/summary{suffix}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {save_dir}/summary{suffix}.json")
    
    # Print summary
    print_summary(summary)
    
    # Generate plots
    plot_results(df, dataset, model, device, save_dir, suffix=suffix)
    
    return df, summary


def compute_summary(df):
    """Compute summary statistics"""
    summary = {}
    
    metrics = [
        ('Overall', ['mse_overall', 'mae_overall', 'rmse_overall', 'r2_overall', 
                     'pearson_overall', 'spearman_overall']),
        ('Masked', ['mse_masked', 'mae_masked', 'pearson_masked', 'peak_recovery_masked']),
        ('Unmasked', ['mse_unmasked', 'mae_unmasked', 'pearson_unmasked']),
        ('Peaks', ['n_peaks_original', 'n_peaks_reconstructed', 'peak_position_error', 
                   'peak_intensity_error', 'peak_f1']),
        ('Spatial', ['local_corr_mean', 'local_corr_std']),
    ]
    
    for category, cols in metrics:
        summary[category] = {}
        for col in cols:
            summary[category][col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'median': float(df[col].median()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
            }
    
    return summary


def print_summary(summary):
    """Print formatted summary"""
    print("\n" + "="*70)
    print("TESTING SUMMARY")
    print("="*70)
    
    for category, metrics in summary.items():
        print(f"\n{category} Metrics:")
        for metric_name, stats in metrics.items():
            print(f"  {metric_name}:")
            for stat_name, value in stats.items():
                if not np.isnan(value):
                    print(f"    {stat_name:8s}: {value:.6f}")
    
    print("="*70 + "\n")


def plot_results(df, dataset, model, device, save_dir, suffix=''):
    """Generate visualization plots"""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    # 1. Distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # MSE
    axes[0, 0].hist(df['mse_overall'], bins=50, alpha=0.7, label='Overall', color='blue')
    axes[0, 0].hist(df['mse_masked'], bins=50, alpha=0.7, label='Masked', color='red')
    axes[0, 0].hist(df['mse_unmasked'], bins=50, alpha=0.7, label='Unmasked', color='green')
    axes[0, 0].set_xlabel('MSE')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('MSE Distribution')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # Pearson correlation
    axes[0, 1].hist(df['pearson_overall'], bins=50, alpha=0.7, color='blue')
    axes[0, 1].axvline(df['pearson_overall'].mean(), color='red', linestyle='--', 
                       label=f"Mean: {df['pearson_overall'].mean():.3f}")
    axes[0, 1].set_xlabel('Pearson Correlation')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Overall Correlation')
    axes[0, 1].legend()
    
    # R²
    axes[0, 2].hist(df['r2_overall'], bins=50, alpha=0.7, color='purple')
    axes[0, 2].axvline(df['r2_overall'].mean(), color='red', linestyle='--',
                       label=f"Mean: {df['r2_overall'].mean():.3f}")
    axes[0, 2].set_xlabel('R² Score')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('R² Distribution')
    axes[0, 2].legend()
    
    # Masked vs Unmasked
    axes[1, 0].scatter(df['pearson_masked'], df['pearson_unmasked'], alpha=0.5, s=10)
    axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
    axes[1, 0].set_xlabel('Masked Correlation')
    axes[1, 0].set_ylabel('Unmasked Correlation')
    axes[1, 0].set_title('Masked vs Unmasked')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Peak F1
    axes[1, 1].hist(df['peak_f1'], bins=50, alpha=0.7, color='orange')
    axes[1, 1].axvline(df['peak_f1'].mean(), color='red', linestyle='--',
                       label=f"Mean: {df['peak_f1'].mean():.3f}")
    axes[1, 1].set_xlabel('Peak Detection F1')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Peak Detection Performance')
    axes[1, 1].legend()
    
    # Local correlation
    axes[1, 2].hist(df['local_corr_mean'], bins=50, alpha=0.7, color='green')
    axes[1, 2].axvline(df['local_corr_mean'].mean(), color='red', linestyle='--',
                       label=f"Mean: {df['local_corr_mean'].mean():.3f}")
    axes[1, 2].set_xlabel('Local Correlation')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Spatial Correlation')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/distributions{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Distribution plots saved to {save_dir}/distributions{suffix}.png")
    
    # 2. Example reconstructions
    plot_examples(dataset, model, device, save_dir, n_examples=10, suffix=suffix)


def plot_examples(dataset, model, device, save_dir, n_examples=10, suffix=''):
    """Plot example reconstructions"""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(n_examples, 4, figsize=(24, 4*n_examples))
    
    indices = np.random.choice(len(dataset), min(n_examples, len(dataset)), replace=False)
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            sample = dataset[sample_idx]
            original = sample['original'].unsqueeze(0).to(device)
            masked = sample['masked'].unsqueeze(0).to(device)
            mask = sample['mask'].unsqueeze(0).to(device)
            
            reconstructed, _ = model(masked, mask)
            
            orig = original.cpu().numpy().flatten()
            mask_input = masked.cpu().numpy().flatten()
            recon = reconstructed.cpu().numpy().flatten()
            mask_np = mask.cpu().numpy().flatten()
            orig = _zero_baseline_signal(orig, start_idx=62501, end_idx=68000)
            orig, mask_input = _align_original_masked_baseline(
                orig,
                mask_input,
                mask_np,
                dataset.patch_size
            )
            
            # Point mask
            point_mask = np.zeros(len(orig), dtype=bool)
            for i, is_masked in enumerate(mask_np):
                if is_masked:
                    start = i * dataset.patch_size
                    end = start + dataset.patch_size
                    point_mask[start:end] = True
            
            # Original
            axes[idx, 0].plot(orig, 'b-', linewidth=0.5)
            axes[idx, 0].fill_between(range(len(orig)), 0, orig, 
                                     where=point_mask, alpha=0.3, color='red')
            axes[idx, 0].set_title(f'Sample {sample_idx}: Original')
            axes[idx, 0].set_ylabel('Intensity')
            
            # Masked
            axes[idx, 1].plot(mask_input, 'orange', linewidth=0.5)
            axes[idx, 1].set_title('Masked Input')
            axes[idx, 1].set_ylabel('Intensity')
            
            # Overlay
            axes[idx, 2].plot(orig, 'b-', label='Original', alpha=0.7, linewidth=0.5)
            axes[idx, 2].plot(recon, 'r-', label='Reconstructed', alpha=0.7, linewidth=0.5)
            for i, is_masked in enumerate(mask_np):
                if is_masked:
                    start = i * dataset.patch_size
                    end = start + dataset.patch_size
                    axes[idx, 2].axvspan(start, end, alpha=0.1, color='yellow')
            
            mse = mean_squared_error(orig, recon)
            corr, _ = stats.pearsonr(orig, recon)
            axes[idx, 2].set_title(f'MSE: {mse:.4f}, Corr: {corr:.3f}')
            axes[idx, 2].set_ylabel('Intensity')
            if idx == 0:
                axes[idx, 2].legend()
            
            # Error
            error = np.abs(orig - recon)
            axes[idx, 3].plot(error, 'purple', linewidth=0.5)
            axes[idx, 3].fill_between(range(len(error)), 0, error,
                                     where=point_mask, alpha=0.3, color='red',
                                     label='Masked error')
            axes[idx, 3].fill_between(range(len(error)), 0, error,
                                     where=~point_mask, alpha=0.3, color='green',
                                     label='Unmasked error')
            axes[idx, 3].set_title('Absolute Error')
            axes[idx, 3].set_ylabel('|Error|')
            if idx == 0:
                axes[idx, 3].legend()
            
            if idx == n_examples - 1:
                for j in range(4):
                    axes[idx, j].set_xlabel('Frequency Point')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/examples{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Example plots saved to {save_dir}/examples{suffix}.png")


if __name__ == "__main__":
    # CONFIGURATION - MODIFY THESE
    MODEL_PATH = "models/SSL_models/aligned_nmr_spectra_128K_Plasma_WS625to680Zero_merged2_20260520_014209_bs16_mr0.35_ps1024_best.pth"
    # "./models/SSL_models/nmr_mae_v2_WSZero_normalised_20251208_125133_bs16_mr0.15_ps1024_20251208_125133_best.pth"
    DATA_PATH = "data/plasma/aligned_nmr_spectra_128K_Plasma_NoSuppress.npy"
    SAVE_DIR = "results/testing/plasma"
    SUFFIX = "_35mask_merged2"  # Suffix for result files
    MASK_RATIO = 0.35 # Should match training
    USE_PARALLEL = True # Set to True to enable parallel processing
    NUM_WORKERS = 6  # Adjust based on your CPU cores and GPU availability (if using CUDA inference)
    WORKERS = NUM_WORKERS if USE_PARALLEL else 1
    INFER_BATCH_SIZE = 8
    NORMALIZE_SPECTRA = None  # None=auto, True=force normalize, False=skip normalize
    SAMPLE_COUNT = None  # Set to None to evaluate all samples, or an integer to randomly select that many.
    
    # Test on all data with 5 random masks each
    df, summary = test_nmr_model(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        save_dir=SAVE_DIR,
        n_samples=SAMPLE_COUNT,  # None = all samples, otherwise random sample count
        mask_realizations_per_sample=5,
        patch_size=1024,
        mask_ratio=MASK_RATIO,
        device='cuda:1',
        suffix=SUFFIX,
        workers=WORKERS,
        normalize_spectra=NORMALIZE_SPECTRA,
        infer_batch_size=INFER_BATCH_SIZE
    )
    
    print("\nTesting complete!")
    print(f"Results saved to {SAVE_DIR}/")
