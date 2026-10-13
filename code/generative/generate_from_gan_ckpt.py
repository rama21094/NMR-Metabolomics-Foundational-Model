#!/usr/bin/env python3
"""Generate spectra from a saved GAN v2 checkpoint.

Needed because the conditional run diverged at epoch 175 and the divergence
guard stopped it before its own generate step. Its best checkpoint (epoch 79,
selection AUC 0.984) was saved; this produces the set from it, using the same
interpolation, water blanking and amplitude matching as the training script.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/generative"))
from train_spectrum_gan_v2 import Gen, GEN_LEN, WATER, sample_embeddings  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    dev = torch.device(a.device)
    ck = torch.load(ROOT / a.ckpt, map_location=dev, weights_only=False)
    zdim = ck["args"]["zdim"]; n_class = ck["n_class"]
    G = Gen(zdim, n_class=n_class).to(dev); G.load_state_dict(ck["G_ema"]); G.eval()
    X = np.load(ROOT / "rebuild/pretrain/corpus_train.npy", mmap_mode="r")
    ppm = np.load(ROOT / "rebuild/pretrain/ppm_axis.npy")
    amp = float(np.median(np.abs(np.asarray(X[:200], dtype=np.float32)).max(1)))
    ppm_gen = np.linspace(ppm[0], ppm[-1], GEN_LEN)
    rng = np.random.default_rng(0)
    out = np.zeros((a.n, X.shape[1]), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, a.n, 64):
            m = min(64, a.n - i)
            z = torch.randn(m, zdim, device=dev)
            e = sample_embeddings(G, m, n_class, rng, dev) if n_class else None
            y = G(z, e).cpu().numpy()
            for j in range(m):
                v = np.interp(ppm, ppm_gen[::-1], y[j][::-1])
                v[(ppm <= WATER[1]) & (ppm >= WATER[0])] = 0.0
                mx = np.abs(v).max()
                out[i + j] = (v / mx * amp) if mx > 0 else v
    np.save(ROOT / a.out, out)
    print(f"wrote {a.out} {out.shape} from epoch {ck['epoch']} (selection AUC {ck['auc']:.3f})")


if __name__ == "__main__":
    main()
