#!/usr/bin/env python3
"""WGAN-GP v2 -- the three fixes the first run needed, plus a conditional arm.

The v1 run (train_spectrum_gan.py) was stable to ~epoch 210, then its generator
loss swung between -260 and +90, and the samples came from epoch 299 because the
checkpoint was overwritten every 5 epochs. Its 5.2 result (discriminator AUC
0.945) was therefore not a fair measure of what the architecture can do.

FIXES
  1. Spectral normalisation on every critic layer, on top of the gradient
     penalty. The standard remedy for late-training critic blow-up.
  2. EMA of generator weights (decay 0.999). Samples come from the averaged
     generator, which is smoother and more reliable than any single step.
  3. Checkpoint SELECTION by the realism test, not by epoch. Every --eval-every
     epochs, 400 spectra are generated and a logistic-regression discriminator is
     fitted against a SELECTION set of real spectra. The lowest-AUC generator is
     kept. The selection set is disjoint from the spectra the final
     validate_synthetic.py uses, so the model is not graded on the test it was
     chosen by.

CONDITIONAL ARM (--conditional). An unconditional GAN resamples the study
composition it was trained on -- v1 reached an effective 4.71 studies against the
corpus's 4.09 -- so no architecture change to it can address what G2 found
binding. Conditioning on study, then generating from MIXED study embeddings,
is the one design that could in principle produce compositions no real study
has. Generation draws a random convex combination of two study embeddings per
spectrum, weighted uniformly across studies rather than by corpus frequency, so
that MTBLS798 cannot dominate the output the way it dominates the corpus.
Whether that yields realistic spectra or artefacts is what 5.2 decides.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
GEN_LEN = 32768
WATER = (4.45, 5.10)


class Gen(nn.Module):
    def __init__(self, zdim=128, ch=512, n_class=0, edim=32):
        super().__init__()
        self.n_class = n_class
        self.emb = nn.Embedding(n_class, edim) if n_class else None
        self.fc = nn.Linear(zdim + (edim if n_class else 0), ch * 64)
        self.ch = ch
        layers, c = [], ch
        for _ in range(9):
            nxt = max(32, c // 2)
            layers += [nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                       nn.Conv1d(c, nxt, 9, padding=4),
                       nn.BatchNorm1d(nxt), nn.LeakyReLU(0.2, inplace=True)]
            c = nxt
        self.net = nn.Sequential(*layers)
        self.out = nn.Conv1d(c, 1, 9, padding=4)

    def forward(self, z, e=None):
        h = torch.cat([z, e], 1) if e is not None else z
        h = self.fc(h).view(z.size(0), self.ch, 64)
        return self.out(self.net(h)).squeeze(1)


class Disc(nn.Module):
    def __init__(self, ch=32, n_class=0):
        super().__init__()
        layers, c = [], 1
        for _ in range(9):
            nxt = min(512, ch if c == 1 else c * 2)
            layers += [SN(nn.Conv1d(c, nxt, 9, stride=2, padding=4)),
                       nn.LeakyReLU(0.2, inplace=True)]
            c = nxt
        self.net = nn.Sequential(*layers)
        self.fc = SN(nn.Linear(c * 64, 1))
        # projection discriminator for the conditional arm
        self.proj = SN(nn.Embedding(n_class, c * 64)) if n_class else None

    def forward(self, x, y=None):
        h = self.net(x.unsqueeze(1)).flatten(1)
        out = self.fc(h).squeeze(1)
        if self.proj is not None and y is not None:
            out = out + (self.proj(y) * h).sum(1)
        return out


def gradient_penalty(D, real, fake, y, device):
    a = torch.rand(real.size(0), 1, device=device)
    x = (a * real + (1 - a) * fake).requires_grad_(True)
    d = D(x, y)
    g = torch.autograd.grad(d, x, torch.ones_like(d), create_graph=True)[0]
    return ((g.norm(2, dim=1) - 1) ** 2).mean()


def pool(A, f):
    n = (A.shape[-1] // f) * f
    return A[..., :n].reshape(*A.shape[:-1], n // f, f).mean(axis=-1)


def bin_feat(A, f=64):
    """Coarse features for the selection discriminator: 32,768 -> 512."""
    B = pool(A, f)
    a = np.abs(B).sum(1, keepdims=True); a[a == 0] = 1
    return B / a


def realism_auc(real_f, fake_f):
    Z = np.vstack([real_f, fake_f]); y = np.r_[np.zeros(len(real_f)), np.ones(len(fake_f))]
    aucs = []
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=0).split(Z, y):
        sc = StandardScaler().fit(Z[tr])
        m = LogisticRegression(max_iter=1000, C=0.1).fit(sc.transform(Z[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(sc.transform(Z[te]))[:, 1]))
    return float(np.mean(aucs))


def sample_embeddings(G, n, n_class, rng, device):
    """Convex mix of two study embeddings, studies drawn uniformly."""
    a = torch.as_tensor(rng.integers(0, n_class, n), device=device)
    b = torch.as_tensor(rng.integers(0, n_class, n), device=device)
    w = torch.as_tensor(rng.uniform(0, 1, (n, 1)), dtype=torch.float32, device=device)
    return w * G.emb(a) + (1 - w) * G.emb(b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--zdim", type=int, default=128)
    ap.add_argument("--lr-g", type=float, default=1e-4)
    ap.add_argument("--lr-d", type=float, default=2e-4)
    ap.add_argument("--n-critic", type=int, default=5)
    ap.add_argument("--gp", type=float, default=10.0)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--conditional", action="store_true")
    ap.add_argument("--n-generate", type=int, default=4000)
    ap.add_argument("--tag", default="gan_v2")
    args = ap.parse_args()
    dev = torch.device(args.device)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    X = np.load(ROOT / "rebuild/pretrain/corpus_train.npy", mmap_mode="r")
    ppm = np.load(ROOT / "rebuild/pretrain/ppm_axis.npy")
    split = [r for r in csv.DictReader(open(ROOT / "rebuild/pretrain/corpus_split.csv"))
             if r["split"] == "train"]
    studies = np.array([r["study"] for r in split])
    uniq = sorted(set(studies)); sid = np.array([uniq.index(s) for s in studies])
    n_class = len(uniq) if args.conditional else 0

    factor = X.shape[1] // GEN_LEN
    R = pool(np.asarray(X, dtype=np.float32), factor)
    s = np.abs(R).max(1, keepdims=True); s[s == 0] = 1
    R = R / s

    # selection set: 600 real spectra held out of the SELECTION discriminator's
    # comparison only; seeded differently from validate_synthetic.py (seed 0 on
    # the full-resolution corpus) so the two real samples are disjoint in draw
    sel = np.random.default_rng(12345).choice(len(R), 600, replace=False)
    sel_f = bin_feat(R[sel])

    T = torch.from_numpy(R); Y = torch.from_numpy(sid)
    G = Gen(args.zdim, n_class=n_class).to(dev)
    D = Disc(n_class=n_class).to(dev)
    G_ema = copy.deepcopy(G).eval()
    og = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.0, 0.9))
    od = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.0, 0.9))
    out_dir = ROOT / "models/gan"; out_dir.mkdir(parents=True, exist_ok=True)
    log = {"args": vars(args), "evals": []}
    best = (1.1, -1)
    n = T.size(0)
    print(f"{'conditional' if n_class else 'unconditional'} WGAN-GP v2, "
          f"{n} spectra at {GEN_LEN}, {n_class} classes", flush=True)

    def generate(model, m):
        with torch.no_grad():
            z = torch.randn(m, args.zdim, device=dev)
            e = sample_embeddings(model, m, n_class, rng, dev) if n_class else None
            return model(z, e).cpu().numpy()

    for ep in range(args.epochs):
        perm = torch.randperm(n)
        dl = gl = 0.0; nb = 0
        for i in range(0, n - args.batch, args.batch):
            idx = perm[i:i + args.batch]
            real = T[idx].to(dev); yr = Y[idx].to(dev) if n_class else None
            for _ in range(args.n_critic):
                z = torch.randn(real.size(0), args.zdim, device=dev)
                e = G.emb(yr) if n_class else None
                with torch.no_grad():
                    fake = G(z, e)
                loss_d = (D(fake, yr).mean() - D(real, yr).mean()
                          + args.gp * gradient_penalty(D, real, fake, yr, dev))
                od.zero_grad(); loss_d.backward(); od.step()
            z = torch.randn(real.size(0), args.zdim, device=dev)
            e = G.emb(yr) if n_class else None
            loss_g = -D(G(z, e), yr).mean()
            og.zero_grad(); loss_g.backward(); og.step()
            with torch.no_grad():
                for pe, pg in zip(G_ema.parameters(), G.parameters()):
                    pe.mul_(args.ema).add_(pg, alpha=1 - args.ema)
                for be, bg in zip(G_ema.buffers(), G.buffers()):
                    be.copy_(bg)
            dl += loss_d.item(); gl += loss_g.item(); nb += 1

        msg = f"epoch {ep:>4}  D {dl/max(nb,1):+.4f}  G {gl/max(nb,1):+.4f}"
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            auc = realism_auc(sel_f, bin_feat(generate(G_ema, 400)))
            log["evals"].append({"epoch": ep, "auc": auc})
            msg += f"   selection AUC {auc:.3f}"
            if auc < best[0]:
                best = (auc, ep)
                torch.save({"G_ema": G_ema.state_dict(), "epoch": ep, "auc": auc,
                            "n_class": n_class, "classes": uniq, "args": vars(args)},
                           out_dir / f"{args.tag}_best.pth")
                msg += "  <- best"
        print(msg, flush=True)
        json.dump(log, open(ROOT / f"results/phase5/{args.tag}_log.json", "w"), indent=1)

    print(f"\nbest selection AUC {best[0]:.3f} at epoch {best[1]}", flush=True)
    ck = torch.load(out_dir / f"{args.tag}_best.pth", map_location=dev, weights_only=False)
    G_ema.load_state_dict(ck["G_ema"])
    amp = float(np.median(np.abs(np.asarray(X[:200], dtype=np.float32)).max(1)))
    ppm_gen = np.linspace(ppm[0], ppm[-1], GEN_LEN)
    out = np.zeros((args.n_generate, X.shape[1]), dtype=np.float32)
    for i in range(0, args.n_generate, 64):
        m = min(64, args.n_generate - i)
        y = generate(G_ema, m)
        for j in range(m):
            v = np.interp(ppm, ppm_gen[::-1], y[j][::-1])
            v[(ppm <= WATER[1]) & (ppm >= WATER[0])] = 0.0
            mx = np.abs(v).max()
            out[i + j] = (v / mx * amp) if mx > 0 else v
    p = ROOT / f"rebuild/pretrain/synthetic/{args.tag}.npy"
    np.save(p, out)
    print(f"wrote {p.relative_to(ROOT)}  {out.shape}  from epoch {best[1]}", flush=True)


if __name__ == "__main__":
    main()
