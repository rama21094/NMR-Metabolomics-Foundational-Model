"""Stage 2 of the latent-diffusion pipeline: train the denoising U-Net in the
frozen VAE's latent space (analogous to training Stable Diffusion's U-Net on
top of a frozen image VAE).

Requires a VAE checkpoint from train_vae.py. Latents for the whole corpus are
encoded once (deterministically, using the posterior mean) and cached to disk
so subsequent epochs/resumes don't re-run the VAE encoder.

Example:
    python code/generative/train_diffusion.py --vae-checkpoint results/generative/vae_v1/vae.pt --epochs 100
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from diffusion_process import EMA, GaussianDiffusion
from spectra_dataset import SpectraDataset
from unet1d_diffusion import UNet1D
from vae1d import VAE1D

DEFAULT_DATA = "data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax.npy"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--vae-checkpoint", required=True)
    p.add_argument("--out-dir", default="results/generative/diffusion_v1")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--base-channels", type=int, default=128)
    p.add_argument("--channel-mult", default="1,2,4")
    p.add_argument("--num-res-blocks", type=int, default=2)
    p.add_argument("--attn-last-n-stages", type=int, default=2)
    p.add_argument("--time-dim", type=int, default=512)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--preview-steps", type=int, default=50)
    p.add_argument("--recompute-latents", action="store_true")
    p.add_argument("--resume", default=None)
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
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


@torch.no_grad()
def encode_corpus(vae, data_path, max_samples, device, batch_size=16):
    ds = SpectraDataset(data_path, max_samples=max_samples)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    means = []
    for x in loader:
        x = x.to(device)
        mean, _ = vae.encode(x)
        means.append(mean.cpu())
    return torch.cat(means, dim=0)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    vae = load_vae(args.vae_checkpoint, device)

    latents_path = out_dir / "cached_latents.npy"
    scale_path = out_dir / "latent_scale.json"
    if latents_path.exists() and scale_path.exists() and not args.recompute_latents:
        latents = torch.from_numpy(np.load(latents_path))
        scale = json.load(open(scale_path))["scale"]
        print(f"Loaded cached latents {tuple(latents.shape)} from {latents_path}")
    else:
        print("Encoding corpus through frozen VAE (one-time cost)...")
        t0 = time.time()
        latents = encode_corpus(vae, args.data, args.max_samples, device)
        print(f"Encoded {latents.shape[0]} spectra to latents {tuple(latents.shape[1:])} in {time.time() - t0:.1f}s")
        scale = float(1.0 / latents.std().item())
        np.save(latents_path, latents.numpy())
        json.dump({"scale": scale}, open(scale_path, "w"))
        print(f"Cached latents to {latents_path}, scale factor {scale:.4f}")

    latents = latents * scale  # rescale to ~unit variance, standard latent-diffusion practice
    latent_channels, latent_length = latents.shape[1], latents.shape[2]

    dataset = TensorDataset(latents)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    channel_mult = tuple(int(x) for x in args.channel_mult.split(","))
    n_stages = len(channel_mult)
    attn_from_end = tuple(range(min(args.attn_last_n_stages, n_stages)))
    model = UNet1D(
        latent_channels=latent_channels,
        base_channels=args.base_channels,
        channel_mult=channel_mult,
        num_res_blocks=args.num_res_blocks,
        attn_resolutions_from_end=attn_from_end,
        time_dim=args.time_dim,
    ).to(device)
    diffusion = GaussianDiffusion(timesteps=args.timesteps, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema = EMA(model, decay=args.ema_decay)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        ema.load_state_dict(ckpt["ema"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    config = vars(args) | {
        "latent_channels": latent_channels, "latent_length": latent_length,
        "latent_scale": scale, "channel_mult": channel_mult,
    }
    json.dump(config, open(out_dir / "run_config.json", "w"), indent=2, default=str)

    n_params = sum(p.numel() for p in model.parameters())

    log_file = open(out_dir / "train.log", "a", buffering=1)  # line-buffered: readable live with tail -f

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")

    csv_path = out_dir / "train_history.csv"
    csv_is_new = not csv_path.exists()
    csv_file = open(csv_path, "a", buffering=1)
    if csv_is_new:
        csv_file.write("epoch,loss,epoch_time_s\n")

    log(f"UNet parameters: {n_params:,}  |  latent shape ({latent_channels},{latent_length})  |  "
        f"n_train={len(dataset)}  |  device={device}")

    history = []
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        running = 0.0
        for step, (z,) in enumerate(loader):
            z = z.to(device)
            loss = diffusion.training_loss(model, z)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)
            running += loss.item()
            if step % args.log_every == 0:
                log(f"epoch {epoch} step {step}/{len(loader)} loss={loss.item():.5f} "
                    f"({time.time() - t0:.1f}s elapsed)")
        epoch_time = time.time() - t0
        mean_loss = running / len(loader)
        history.append({"epoch": epoch, "loss": mean_loss, "epoch_time_s": epoch_time})
        log(f"== epoch {epoch} done in {epoch_time:.1f}s | mean loss={mean_loss:.5f} ==")
        csv_file.write(f"{epoch},{mean_loss:.6f},{epoch_time:.2f}\n")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt_path = out_dir / "diffusion.pt"
            torch.save({
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "base_channels": args.base_channels,
                "channel_mult": channel_mult,
                "num_res_blocks": args.num_res_blocks,
                "attn_resolutions_from_end": attn_from_end,
                "time_dim": args.time_dim,
                "timesteps": args.timesteps,
                "latent_channels": latent_channels,
                "latent_length": latent_length,
                "latent_scale": scale,
            }, ckpt_path)
            log(f"Saved checkpoint to {ckpt_path}")
            make_preview(model, ema, diffusion, vae, scale, latent_channels, latent_length,
                         device, args.preview_steps, args.data,
                         out_dir / f"sample_preview_epoch{epoch}.png")

    json.dump(history, open(out_dir / "history.json", "w"), indent=2)
    log(f"\nDone. Outputs in {out_dir}/")
    log_file.close()
    csv_file.close()


@torch.no_grad()
def make_preview(model, ema, diffusion, vae, scale, latent_channels, latent_length,
                  device, steps, data_path, out_path, n=3):
    """Generate n synthetic spectra with the current EMA weights and plot them
    next to n random real spectra for a quick visual sanity check."""
    state_backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema.copy_to(model)
    model.eval()

    z = diffusion.ddim_sample(model, (n, latent_channels, latent_length), device, num_steps=steps)
    z = z / scale
    synthetic = vae.decode(z).cpu().numpy()[:, 0, :]

    model.train()
    model.load_state_dict(state_backup)

    real_full = np.load(data_path, mmap_mode="r")
    rng = np.random.default_rng(0)
    real_idx = rng.choice(real_full.shape[0], size=n, replace=False)
    real = np.asarray(real_full[real_idx])

    fig, axes = plt.subplots(n, 2, figsize=(11, 2.2 * n), sharex=True)
    for row in range(n):
        axes[row, 0].plot(real[row], color="#2a78d6", linewidth=0.7)
        axes[row, 1].plot(synthetic[row], color="#eb6834", linewidth=0.7)
    axes[0, 0].set_title("real (random)", fontsize=9)
    axes[0, 1].set_title("synthetic (generated)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
