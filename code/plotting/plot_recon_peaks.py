#!/usr/bin/env python3
"""
Plot original vs reconstructed spectra zoomed into a high-resolution peak region.
Generates one PNG per randomly selected sample (default 5 samples).

Usage:
    python code/plotting/plot_recon_peaks.py --model-path <checkpoint.pth> --data-path <spectra.npy>

Defaults will auto-select a checkpoint from `models/SSL_models/` if none provided.
"""

import argparse
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from trainer_revised import NMRMaskedAutoencoder, NMRSpectrumDataset
from sklearn.metrics import mean_squared_error
from scipy import stats


def auto_select_checkpoint():
    try:
        files = os.listdir('models/SSL_models')
        pths = [os.path.join('models/SSL_models', f) for f in files if f.endswith('.pth')]
        if pths:
            return pths[0]
    except Exception:
        pass
    return None


def load_model(checkpoint_path, device, spectrum_length=None, patch_size=1024):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)

    # Try to get hyperparameters from checkpoint
    hp = ckpt.get('hyperparameters', {}) if isinstance(ckpt, dict) else {}
    ps = hp.get('patch_size', patch_size)
    d_model = hp.get('d_model', 128)
    nhead = hp.get('nhead', 4)
    num_layers = hp.get('num_layers', 3)
    dim_feedforward = hp.get('dim_feedforward', 256)

    if spectrum_length is None:
        # try to infer from checkpoint name or fallback
        spectrum_length = hp.get('spectrum_length', None)

    model = NMRMaskedAutoencoder(
        spectrum_length=spectrum_length or (ps * 16),
        patch_size=ps,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=hp.get('dropout', 0.2)
    )

    try:
        model.load_state_dict(state)
    except Exception:
        # attempt nested
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            try:
                model.load_state_dict(ckpt['model_state_dict'])
            except Exception:
                pass

    model.to(device)
    model.eval()
    return model


def plot_sample(original, reconstructed, mask_np, dataset_patch_size, region_start, region_end, out_path, sample_idx):
    L = len(original)
    rs = max(0, region_start)
    re = min(L, region_end)

    x = np.arange(rs, re)
    orig_seg = original[rs:re]
    recon_seg = reconstructed[rs:re]

    mse = mean_squared_error(orig_seg, recon_seg)
    try:
        corr, _ = stats.pearsonr(orig_seg, recon_seg)
    except Exception:
        corr = np.nan

    fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
    ax.plot(x, orig_seg, label='Original', color='blue', linewidth=0.8)
    ax.plot(x, recon_seg, label='Reconstructed', color='red', linewidth=0.7)

    # Shade masked patches
    n_patches = len(mask_np)
    for i, is_masked in enumerate(mask_np):
        if not is_masked:
            continue
        start = i * dataset_patch_size
        end = start + dataset_patch_size
        # only show overlap with region
        if end < rs or start > re:
            continue
        s = max(start, rs)
        e = min(end, re)
        ax.axvspan(s, e, color='yellow', alpha=0.25)

    ax.set_title(f'Sample {sample_idx} — MSE: {mse:.6f}, Corr: {corr:.3f}')
    ax.set_xlabel('Frequency Point')
    ax.set_ylabel('Normalized Intensity')
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description='Plot reconstructions in peak region for a few samples')
    p.add_argument('--model-path', default=None, help='Checkpoint path (.pth)')
    p.add_argument('--data-path', default='data/aligned/aligned_nmr_spectra_128K_WS625to680Zero.npy', help='Numpy spectra file')
    p.add_argument('--output-dir', default='results/reconstruction/recon_peaks', help='Directory to save images')
    p.add_argument('--n-samples', type=int, default=5, help='Number of random spectra to plot')
    p.add_argument('--region-start', type=int, default=60000, help='Start index of high-resolution region')
    p.add_argument('--region-end', type=int, default=100000, help='End index of high-resolution region')
    p.add_argument('--mask-ratio', type=float, default=0.50, help='Mask ratio used to generate masked inputs')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run model on')
    p.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_path = "models/SSL_models/Itr6Rerun_20260105_062242_bs16_mr0.50_ps1024_best.pth" #args.model_path or auto_select_checkpoint()
    if model_path is None:
        raise FileNotFoundError('No checkpoint provided and none found in models/SSL_models/')
    print('Using checkpoint:', model_path)

    # Load spectra to create dataset and to get length
    spectra = np.load(args.data_path)
    print('Loaded spectra shape:', spectra.shape)

    # Use dataset to generate masks consistent with training preprocessing
    dataset = NMRSpectrumDataset(spectra, mask_ratio=args.mask_ratio, patch_size=1024, mask_strategy='sparse_random')

    device = torch.device(args.device if torch.cuda.is_available() or 'cpu' else 'cpu')
    model = load_model(model_path, device, spectrum_length=spectra.shape[1], patch_size=dataset.patch_size)

    # pick random indices
    rng = random.Random(args.seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    selected = indices[:args.n_samples]

    for idx in selected:
        sample = dataset[idx]
        original = sample['original'].unsqueeze(0)  # shape [1, L]
        masked = sample['masked'].unsqueeze(0)
        mask = sample['mask']  # shape [n_patches]

        # Move to device for inference
        masked_d = masked.to(device)
        mask_d = mask.unsqueeze(0).to(device)

        with torch.no_grad():
            reconstructed, _ = model(masked_d, mask_d)

        orig_np = original.cpu().numpy().flatten()
        recon_np = reconstructed.cpu().numpy().flatten()
        mask_np = mask.cpu().numpy().astype(bool)

        # include mask ratio in filename for clarity
        mr_str = f"{args.mask_ratio:.2f}".replace('.', '_')
        out_path = os.path.join(
            args.output_dir,
            f'sample_{idx}_region_{args.region_start}_{args.region_end}_mr{mr_str}.png'
        )
        plot_sample(orig_np, recon_np, mask_np, dataset.patch_size, args.region_start, args.region_end, out_path, idx)
        print('Saved', out_path)

    print('Done.')


if __name__ == '__main__':
    main()
