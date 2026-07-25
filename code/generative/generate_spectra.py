"""Generate synthetic NMR spectra from a trained VAE + latent-diffusion model.

Samples latents with the diffusion model's EMA weights (DDIM by default, far
fewer steps than training), decodes them through the frozen VAE, and saves
both the raw array and a comparison plot against real spectra.

Example:
    python code/generative/generate_spectra.py \\
        --vae-checkpoint results/generative/vae_v1/vae.pt \\
        --diffusion-checkpoint results/generative/diffusion_v1/diffusion.pt \\
        --num-samples 200 --out-dir results/generative/samples_v1
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from diffusion_process import GaussianDiffusion
from unet1d_diffusion import UNet1D
from vae1d import VAE1D

DEFAULT_DATA = "data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax.npy"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--vae-checkpoint", required=True)
    p.add_argument("--diffusion-checkpoint", required=True)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity, 0=deterministic")
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--no-use-ema", dest="use_ema", action="store_false")
    p.add_argument("--out-dir", default="results/generative/samples_v1")
    p.add_argument("--compare-real-data", default=DEFAULT_DATA, help="path to real .npy for the comparison plot; pass '' to skip")
    p.add_argument("--n-plot", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-threads", type=int, default=None)
    return p.parse_args()


def load_vae(vae_checkpoint, device):
    ckpt = torch.load(vae_checkpoint, map_location=device)
    vae = VAE1D(
        base_channels=ckpt["base_channels"],
        channel_mult=tuple(ckpt["channel_mult"]),
        latent_channels=ckpt["latent_channels"],
    ).to(device)
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    return vae


def load_diffusion(diffusion_checkpoint, device, use_ema):
    ckpt = torch.load(diffusion_checkpoint, map_location=device)
    model = UNet1D(
        latent_channels=ckpt["latent_channels"],
        base_channels=ckpt["base_channels"],
        channel_mult=tuple(ckpt["channel_mult"]),
        num_res_blocks=ckpt["num_res_blocks"],
        attn_resolutions_from_end=tuple(ckpt["attn_resolutions_from_end"]),
        time_dim=ckpt["time_dim"],
    ).to(device)
    model.load_state_dict(ckpt["ema"] if use_ema else ckpt["model"])
    model.eval()
    diffusion = GaussianDiffusion(timesteps=ckpt["timesteps"], device=device)
    return model, diffusion, ckpt["latent_channels"], ckpt["latent_length"], ckpt["latent_scale"]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    vae = load_vae(args.vae_checkpoint, device)
    model, diffusion, latent_channels, latent_length, scale = load_diffusion(
        args.diffusion_checkpoint, device, args.use_ema
    )
    print(f"Latent shape ({latent_channels}, {latent_length}), scale={scale:.4f}, "
          f"sampler={args.sampler}, device={device}")

    all_synth = []
    n_done = 0
    with torch.no_grad():
        while n_done < args.num_samples:
            b = min(args.batch_size, args.num_samples - n_done)
            shape = (b, latent_channels, latent_length)
            if args.sampler == "ddim":
                z = diffusion.ddim_sample(model, shape, device, num_steps=args.ddim_steps, eta=args.eta)
            else:
                z = diffusion.ddpm_sample(model, shape, device)
            z = z / scale
            x = vae.decode(z).cpu().numpy()[:, 0, :]
            all_synth.append(x)
            n_done += b
            print(f"generated {n_done}/{args.num_samples}")

    synthetic = np.concatenate(all_synth, axis=0)
    out_npy = out_dir / "synthetic_spectra.npy"
    np.save(out_npy, synthetic)
    print(f"Saved {synthetic.shape} to {out_npy}")

    if args.compare_real_data:
        real_full = np.load(args.compare_real_data, mmap_mode="r")
        n_plot = min(args.n_plot, synthetic.shape[0], real_full.shape[0])
        rng = np.random.default_rng(args.seed)
        real_idx = rng.choice(real_full.shape[0], size=n_plot, replace=False)
        real = np.asarray(real_full[real_idx])
        synth_idx = rng.choice(synthetic.shape[0], size=n_plot, replace=False)

        fig, axes = plt.subplots(n_plot, 2, figsize=(11, 2.2 * n_plot), sharex=True)
        if n_plot == 1:
            axes = axes[None, :]
        for row in range(n_plot):
            axes[row, 0].plot(real[row], color="#2a78d6", linewidth=0.7)
            axes[row, 1].plot(synthetic[synth_idx[row]], color="#eb6834", linewidth=0.7)
        axes[0, 0].set_title("real", fontsize=9)
        axes[0, 1].set_title("synthetic", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "real_vs_synthetic.png", dpi=150)
        plt.close(fig)
        print(f"Saved comparison plot to {out_dir / 'real_vs_synthetic.png'}")

    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
