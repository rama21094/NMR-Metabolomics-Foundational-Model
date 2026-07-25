"""Stage 1 of the latent-diffusion pipeline: train the 1D convolutional VAE
that compresses each 131072-point spectrum into a small latent grid.

Run this first; its checkpoint is required by train_diffusion.py.

Example:
    python code/generative/train_vae.py --epochs 40 --batch-size 8
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
from torch.utils.data import DataLoader

from spectra_dataset import SpectraDataset, build_loss_weight_mask, train_val_split
from vae1d import VAE1D, vae_loss

DEFAULT_DATA = "data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax.npy"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--out-dir", default="results/generative/vae_v1")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--channel-mult", default="1,2,2,4,4,6,6", help="comma-separated per-downsample-stage channel multipliers")
    p.add_argument("--latent-channels", type=int, default=8)
    p.add_argument("--kl-weight", type=float, default=1e-4)
    p.add_argument("--peak-weight-csv", default=None, help="canonical_peaks.csv from peak_extraction.py, to upweight peak regions in the loss")
    p.add_argument("--peak-halfwidth", type=int, default=150)
    p.add_argument("--peak-weight", type=float, default=5.0)
    p.add_argument("--suppression-weight", type=float, default=0.1)
    p.add_argument("--val-frac", type=float, default=0.03)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=None, help="use only the first N spectra (quick smoke tests)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-threads", type=int, default=None)
    return p.parse_args()


def make_preview_plot(model, val_ds, device, out_path, n=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(val_ds), size=min(n, len(val_ds)), replace=False)
    model.eval()
    fig, axes = plt.subplots(len(idx), 1, figsize=(10, 2.2 * len(idx)), sharex=True)
    if len(idx) == 1:
        axes = [axes]
    with torch.no_grad():
        for ax, i in zip(axes, idx):
            x = val_ds[i].unsqueeze(0).to(device)
            recon, _, _ = model(x)
            ax.plot(x[0, 0].cpu().numpy(), color="#2a78d6", linewidth=0.7, label="real")
            ax.plot(recon[0, 0].cpu().numpy(), color="#eb6834", linewidth=0.7, alpha=0.8, label="reconstructed")
            ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    model.train()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    full = np.load(args.data, mmap_mode="r")
    n_total = full.shape[0] if args.max_samples is None else min(args.max_samples, full.shape[0])
    length = full.shape[1]
    train_idx, val_idx = train_val_split(n_total, val_frac=args.val_frac, seed=args.seed)

    train_ds = SpectraDataset(args.data, indices=train_idx)
    val_ds = SpectraDataset(args.data, indices=val_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    channel_mult = tuple(int(x) for x in args.channel_mult.split(","))
    model = VAE1D(base_channels=args.base_channels, channel_mult=channel_mult, latent_channels=args.latent_channels).to(device)
    assert length % model.downsample_factor == 0, (
        f"spectrum length {length} must be divisible by downsample factor {model.downsample_factor}"
    )
    latent_length = length // model.downsample_factor
    print(f"Latent shape per sample: ({args.latent_channels}, {latent_length})  "
          f"(compression {length / (args.latent_channels * latent_length):.2f}x)")

    weight_np = build_loss_weight_mask(
        length, peaks_csv=args.peak_weight_csv, peak_halfwidth=args.peak_halfwidth,
        peak_weight=args.peak_weight, suppression_weight=args.suppression_weight,
    )
    weight = torch.from_numpy(weight_np).to(device).view(1, 1, -1)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    config = vars(args) | {"n_total": n_total, "length": length, "latent_length": latent_length, "channel_mult": channel_mult}
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    n_params = sum(p.numel() for p in model.parameters())

    log_file = open(out_dir / "train.log", "a", buffering=1)  # line-buffered: readable live with tail -f

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")

    csv_path = out_dir / "train_history.csv"
    csv_is_new = not csv_path.exists()
    csv_file = open(csv_path, "a", buffering=1)
    if csv_is_new:
        csv_file.write("epoch,loss,recon,kl,epoch_time_s\n")

    log(f"VAE parameters: {n_params:,}  |  train={len(train_ds)} val={len(val_ds)}  |  device={device}")

    history = []
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        running = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        for step, x in enumerate(train_loader):
            x = x.to(device)
            recon, mean, logvar = model(x)
            loss, recon_l, kl_l = vae_loss(recon, x, mean, logvar, weight=weight, kl_weight=args.kl_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running["loss"] += loss.item()
            running["recon"] += recon_l.item()
            running["kl"] += kl_l.item()
            if step % args.log_every == 0:
                elapsed = time.time() - t0
                log(f"epoch {epoch} step {step}/{len(train_loader)} "
                    f"loss={loss.item():.5f} recon={recon_l.item():.5f} kl={kl_l.item():.5f} "
                    f"({elapsed:.1f}s elapsed)")
        n_steps = len(train_loader)
        epoch_time = time.time() - t0
        row = {k: v / n_steps for k, v in running.items()} | {"epoch": epoch, "epoch_time_s": epoch_time}
        history.append(row)
        log(f"== epoch {epoch} done in {epoch_time:.1f}s | "
            f"mean loss={row['loss']:.5f} recon={row['recon']:.5f} kl={row['kl']:.5f} ==")
        csv_file.write(f"{epoch},{row['loss']:.6f},{row['recon']:.6f},{row['kl']:.6f},{epoch_time:.2f}\n")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt_path = out_dir / "vae.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "base_channels": args.base_channels,
                "channel_mult": channel_mult,
                "latent_channels": args.latent_channels,
            }, ckpt_path)
            make_preview_plot(model, val_ds, device, out_dir / f"reconstruction_preview_epoch{epoch}.png")
            log(f"Saved checkpoint to {ckpt_path}")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    log(f"\nDone. Outputs in {out_dir}/")
    log_file.close()
    csv_file.close()


if __name__ == "__main__":
    main()
