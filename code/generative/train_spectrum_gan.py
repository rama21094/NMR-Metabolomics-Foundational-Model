#!/usr/bin/env python3
"""Phase 5.1b -- a WGAN-GP over NMR spectra, the generative half of gate G3.

WHY, GIVEN THE PRIOR IS LOW. The GISSMO route failed the 5.2 validation gate on
REALISM: 87 metabolites explain a median R2 of 0.25 of a real spectrum, so
synthetic spectra were sparse where real ones are dense and a discriminator
separated them at AUC 1.000. A GAN learns the whole signal, including everything
the basis cannot model, so it addresses exactly that failure. Leaving it untested
would leave the generative half of G3 resting on argument rather than
measurement, which is a stated limitation of the current write-up.

WHAT IT PROBABLY CANNOT DO. G2 found the binding constraint is the number of
INDEPENDENT STUDIES (this corpus holds an effective 4.09), not the number of
spectra, and rows saturate below 2,000. A generator fitted to this corpus
resamples the composition it was shown, so the expected outcome is convincing
spectra that do not improve transfer.

That is a prediction, and the run is instrumented to refute it: the effective
study count of the GENERATED set is measured directly, by classifying each
generated spectrum to its nearest real study centroid and taking
exp(entropy of the resulting composition). Two outcomes are informative --

  eff(generated) ~ eff(real) = 4.09  ->  resampling, as predicted
  eff(generated) <  eff(real)        ->  mode collapse; the 5.2 gate should
                                          catch it and the data is not used
  eff(generated) >  eff(real)        ->  the prediction is WRONG; the GAN is
                                          interpolating between studies and
                                          synthetic data may genuinely help

RESOLUTION. The corpus is 131,072 points over 10.5 ppm -- roughly 10,900 points
per ppm, which is far beyond the information content (a 7 Hz coupling at 600 MHz
spans 0.0117 ppm). The GAN generates at 32,768 points (~2,730 per ppm, so a 7 Hz
splitting is still ~32 points) and is interpolated up to the corpus axis. This is
a deliberate approximation and the 5.2 discriminator is the check on whether it
is detectable.

WGAN-GP rather than vanilla GAN: the gradient penalty is the standard remedy for
the training instability and mode collapse that a plain 1-D DCGAN shows on
signals like these, and mode collapse is the specific failure that would make
this experiment uninterpretable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
GEN_LEN = 32768
WATER = (4.45, 5.10)


class Gen(nn.Module):
    def __init__(self, zdim=128, ch=512):
        super().__init__()
        self.fc = nn.Linear(zdim, ch * 64)
        self.ch = ch
        layers, c = [], ch
        for _ in range(9):                       # 64 -> 32768
            nxt = max(32, c // 2)
            layers += [nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                       nn.Conv1d(c, nxt, 9, padding=4),
                       nn.BatchNorm1d(nxt), nn.LeakyReLU(0.2, inplace=True)]
            c = nxt
        self.net = nn.Sequential(*layers)
        self.out = nn.Conv1d(c, 1, 9, padding=4)

    def forward(self, z):
        h = self.fc(z).view(z.size(0), self.ch, 64)
        return self.out(self.net(h)).squeeze(1)


class Disc(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        layers, c = [], 1
        for _ in range(9):                       # 32768 -> 64
            nxt = min(512, ch if c == 1 else c * 2)
            layers += [nn.Conv1d(c, nxt, 9, stride=2, padding=4),
                       nn.LeakyReLU(0.2, inplace=True)]
            c = nxt
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(c * 64, 1)

    def forward(self, x):
        h = self.net(x.unsqueeze(1))
        return self.fc(h.flatten(1)).squeeze(1)


def gradient_penalty(D, real, fake, device):
    a = torch.rand(real.size(0), 1, device=device)
    x = (a * real + (1 - a) * fake).requires_grad_(True)
    d = D(x)
    g = torch.autograd.grad(d, x, torch.ones_like(d), create_graph=True)[0]
    return ((g.norm(2, dim=1) - 1) ** 2).mean()


def pool(A, f):
    n = (A.shape[-1] // f) * f
    return A[..., :n].reshape(*A.shape[:-1], n // f, f).mean(axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--zdim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-critic", type=int, default=5)
    ap.add_argument("--gp", type=float, default=10.0)
    ap.add_argument("--n-generate", type=int, default=4000)
    ap.add_argument("--out", default="rebuild/pretrain/synthetic/gan.npy")
    ap.add_argument("--ckpt", default="models/gan/spectrum_wgan.pth")
    args = ap.parse_args()
    dev = torch.device(args.device)

    X = np.load(ROOT / "rebuild/pretrain/corpus_train.npy", mmap_mode="r")
    ppm = np.load(ROOT / "rebuild/pretrain/ppm_axis.npy")
    factor = X.shape[1] // GEN_LEN
    print(f"corpus {X.shape}, generating at {GEN_LEN} (pool factor {factor})")

    R = pool(np.asarray(X, dtype=np.float32), factor)
    scale = np.abs(R).max(axis=1, keepdims=True); scale[scale == 0] = 1
    R = R / scale
    T = torch.from_numpy(R)
    print(f"training tensor {tuple(T.shape)}")

    G, D = Gen(args.zdim).to(dev), Disc().to(dev)
    og = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.9))
    od = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.9))
    n = T.size(0)
    (ROOT / args.ckpt).parent.mkdir(parents=True, exist_ok=True)

    for ep in range(args.epochs):
        perm = torch.randperm(n)
        dl = gl = 0.0; nb = 0
        for i in range(0, n - args.batch, args.batch):
            real = T[perm[i:i + args.batch]].to(dev)
            for _ in range(args.n_critic):
                z = torch.randn(real.size(0), args.zdim, device=dev)
                with torch.no_grad():
                    fake = G(z)
                loss_d = (D(fake).mean() - D(real).mean()
                          + args.gp * gradient_penalty(D, real, fake, dev))
                od.zero_grad(); loss_d.backward(); od.step()
            z = torch.randn(real.size(0), args.zdim, device=dev)
            loss_g = -D(G(z)).mean()
            og.zero_grad(); loss_g.backward(); og.step()
            dl += loss_d.item(); gl += loss_g.item(); nb += 1
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:>4}  D {dl/max(nb,1):+.4f}  G {gl/max(nb,1):+.4f}", flush=True)
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "epoch": ep,
                        "args": vars(args)}, ROOT / args.ckpt)

    # generate, interpolate onto the corpus axis, blank water, match amplitude
    G.eval()
    amp = float(np.median(np.abs(np.asarray(X[:200], dtype=np.float32)).max(axis=1)))
    ppm_gen = np.linspace(ppm[0], ppm[-1], GEN_LEN)
    out = np.zeros((args.n_generate, X.shape[1]), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, args.n_generate, 64):
            m = min(64, args.n_generate - i)
            y = G(torch.randn(m, args.zdim, device=dev)).cpu().numpy()
            for j in range(m):
                v = np.interp(ppm, ppm_gen[::-1], y[j][::-1])
                v[(ppm <= WATER[1]) & (ppm >= WATER[0])] = 0.0
                mx = np.abs(v).max()
                out[i + j] = (v / mx * amp) if mx > 0 else v
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, out)
    json.dump({"gen_len": GEN_LEN, "n": args.n_generate, "epochs": args.epochs,
               "zdim": args.zdim, "note": "WGAN-GP; validate with "
               "code/synthesis/validate_synthetic.py before any training use"},
              open(p.with_suffix(".json"), "w"), indent=1)
    print(f"wrote {args.out}  {out.shape}")


if __name__ == "__main__":
    main()
