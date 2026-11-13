#!/usr/bin/env python3
"""Final experiment 5 (Shyam meeting 2026-10-23): VAE and latent diffusion for
synthetic NMR spectra, built to current practice and to what the GAN taught us.

Two stages, as in latent diffusion (Rombach et al. 2022):

STAGE 1 -- a KL-regularised convolutional autoencoder ("ae").
  * Input: spectra mean-pooled by 4 to 32,768 points (the GAN's resolution, so
    the two are comparable), scaled to max |x| = 1 like the GAN, then passed
    through an invertible dynamic-range compression
        u = asinh(x / c) / asinh(1 / c),   c = 1e-3,
    so that metabolite peaks at 1/1000 of the tallest peak still carry weight in
    the loss. Plain L1/L2 on raw intensities would let a handful of large peaks
    dominate (the same reason audio models work on log-magnitude spectra).
  * Residual blocks with GroupNorm + SiLU, strided-conv downsampling and
    nearest-neighbour upsampling + conv (no transposed conv, so no checkerboard),
    self-attention at the bottleneck, 6 downsamplings: 32,768 -> 512 positions
    x 16 latent channels.
  * REFLECT padding in every convolution. The GAN's clearest artefact was
    drifting spectrum ends; zero padding at 18 layers is the leading suspect, so
    this model does not use it.
  * Loss: L1 on u, L1 on the first difference of u (preserves peak shape and
    sharp edges, which plain L1 blurs), and a KL term. --beta sets the KL weight
    per latent element. A tiny beta (1e-6) gives the near-deterministic
    autoencoder latent diffusion wants; a large beta gives a proper VAE whose
    prior can be sampled directly, which is the "VAE" arm of the experiment.

STAGE 2 -- a diffusion transformer on the latents ("ldm"), after DiT (Peebles &
  Xie 2023): latents (16 x 512) are patchified (patch 2 -> 256 tokens), a
  12-block transformer with adaLN-Zero conditioning on time step and study,
  v-prediction (Salimans & Ho 2022) on a cosine schedule, Min-SNR-5 loss
  weighting (Hang et al. 2023), EMA weights, classifier-free guidance (Ho &
  Salimans 2022) via 10% label dropout, DDIM sampling.
  Study conditioning lets generation be steered to an EVEN study mix, the one
  thing a generator could add that G2 identified as missing.

  Small-data risk: diffusion models memorise small training sets (Somepalli et
  al. 2023). Checkpoints are therefore selected on realism measured against a
  separate selection set, and `eval` reports, for every generated spectrum, its
  highest correlation with any training spectrum next to the same number for
  real held-out spectra.

Usage:
  ldm_v2.py ae     --tag ae_b1e-6 --beta 1e-6 --device cuda:0
  ldm_v2.py ldm    --ae ae_b1e-6 --tag ldm1 --device cuda:0
  ldm_v2.py sample --ldm ldm1 --n 4000 --study-mix uniform --guidance 1.5
  ldm_v2.py vae-sample --ae vae_b1e-2 --n 4000
"""
from __future__ import annotations

import argparse, copy, csv, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
GEN_LEN, C_ASINH = 32768, 1e-3
WATER = (4.45, 5.10)
MODELS = ROOT / "models/ldm_v2"; MODELS.mkdir(parents=True, exist_ok=True)
LOGS = ROOT / "results/final/ldm_v2"; LOGS.mkdir(parents=True, exist_ok=True)


# =========================================================== data ===========
def to_u(x):
    return np.arcsinh(x / C_ASINH) / np.arcsinh(1.0 / C_ASINH)


def from_u(u):
    return np.sinh(u * np.arcsinh(1.0 / C_ASINH)) * C_ASINH


def load_corpus():
    """Training corpus at 32,768 points, max-abs scaled, in u-space; study ids."""
    X = np.load(ROOT / "rebuild/pretrain/corpus_train.npy", mmap_mode="r")
    f = X.shape[1] // GEN_LEN
    R = np.empty((X.shape[0], GEN_LEN), dtype=np.float32)
    for i in range(0, X.shape[0], 500):
        b = np.asarray(X[i:i + 500], dtype=np.float32)
        R[i:i + 500] = b[:, :GEN_LEN * f].reshape(len(b), GEN_LEN, f).mean(-1)
    s = np.abs(R).max(1, keepdims=True); s[s == 0] = 1; R /= s
    split = [r for r in csv.DictReader(open(ROOT / "rebuild/pretrain/corpus_split.csv"))
             if r["split"] == "train"]
    studies = np.array([r["study"] for r in split]); assert len(studies) == len(R)
    uniq = sorted(set(studies)); sid = np.array([uniq.index(s) for s in studies])
    return to_u(R).astype(np.float32), R, sid, uniq


def bin_feat(A, f=64):
    """Coarse features for the realism check used in checkpoint selection."""
    n = (A.shape[-1] // f) * f
    B = A[..., :n].reshape(*A.shape[:-1], n // f, f).mean(-1)
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


# ===================================================== autoencoder ==========
def conv(ci, co, k=3, s=1):
    return nn.Conv1d(ci, co, k, stride=s, padding=k // 2, padding_mode="reflect")


class Res(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(32, ci), conv(ci, co)
        self.n2, self.c2 = nn.GroupNorm(32, co), conv(co, co)
        self.skip = nn.Conv1d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        return self.skip(x) + self.c2(F.silu(self.n2(h)))


class Attn(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.n = nn.GroupNorm(32, c); self.a = nn.MultiheadAttention(c, heads, batch_first=True)

    def forward(self, x):
        h = self.n(x).transpose(1, 2)
        return x + self.a(h, h, h, need_weights=False)[0].transpose(1, 2)


class AE(nn.Module):
    def __init__(self, ch=(64, 128, 128, 256, 256, 256), z=16):
        super().__init__()
        enc = [conv(1, ch[0])]
        for i, c in enumerate(ch):
            enc += [Res(ch[max(i - 1, 0)] if i else ch[0], c), Res(c, c), conv(c, c, 3, 2)]
        enc += [Res(ch[-1], ch[-1]), Attn(ch[-1]), Res(ch[-1], ch[-1]),
                nn.GroupNorm(32, ch[-1]), nn.SiLU(), conv(ch[-1], 2 * z)]
        self.enc = nn.Sequential(*enc)
        dec = [conv(z, ch[-1]), Res(ch[-1], ch[-1]), Attn(ch[-1]), Res(ch[-1], ch[-1])]
        rc = list(reversed(ch))
        for i, c in enumerate(rc):
            nxt = rc[i + 1] if i + 1 < len(rc) else ch[0]
            dec += [nn.Upsample(scale_factor=2, mode="nearest"), conv(c, c), Res(c, nxt), Res(nxt, nxt)]
        dec += [nn.GroupNorm(32, ch[0]), nn.SiLU(), conv(ch[0], 1)]
        self.dec = nn.Sequential(*dec)

    def encode(self, x):
        m, lv = self.enc(x.unsqueeze(1)).chunk(2, 1)
        return m, lv.clamp(-30, 20)

    def decode(self, z):
        return self.dec(z).squeeze(1)


def ae_loss(model, x, beta):
    m, lv = model.encode(x)
    z = m + torch.randn_like(m) * (0.5 * lv).exp()
    xh = model.decode(z)
    rec = (xh - x).abs().mean()
    d = ((xh[:, 1:] - xh[:, :-1]) - (x[:, 1:] - x[:, :-1])).abs().mean()
    kl = 0.5 * (m.pow(2) + lv.exp() - 1 - lv).mean()
    return rec + d + beta * kl, {"l1": rec.item(), "dl1": d.item(), "kl": kl.item()}


def train_ae(a):
    dev = torch.device(a.device); torch.manual_seed(a.seed)
    U, _, sid, uniq = load_corpus()
    rng = np.random.default_rng(0); val = rng.choice(len(U), 400, replace=False)
    tr = np.setdiff1d(np.arange(len(U)), val)
    model = AE().to(dev)
    print(f"AE {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, beta {a.beta}, "
          f"{len(tr)} train / {len(val)} val", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    total = a.epochs * (len(tr) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=total, pct_start=0.03)
    Ut = torch.from_numpy(U); best = 1e9; log = []
    for ep in range(a.epochs):
        model.train(); perm = np.random.permutation(tr); t0 = time.time(); acc = []
        for i in range(0, len(perm) - a.batch + 1, a.batch):
            x = Ut[perm[i:i + a.batch]].to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts = ae_loss(model, x, a.beta)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            acc.append(parts)
        model.eval(); vl = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(val), a.batch):
                x = Ut[val[i:i + a.batch]].to(dev)
                m, _ = model.encode(x); vl.append((model.decode(m) - x).abs().mean().item())
        v = float(np.mean(vl)); tr_l = {k: float(np.mean([p[k] for p in acc])) for k in acc[0]}
        log.append({"epoch": ep, "val_l1": v, **tr_l})
        msg = f"ep {ep:3d}  train {tr_l}  val L1 {v:.5f}  {time.time()-t0:.0f}s"
        if v < best:
            best = v; torch.save({"model": model.state_dict(), "beta": a.beta, "epoch": ep,
                                  "val_l1": v, "classes": uniq}, MODELS / f"{a.tag}.pth"); msg += "  <- best"
        print(msg, flush=True)
        json.dump(log, open(LOGS / f"{a.tag}_log.json", "w"), indent=1)


def load_ae(tag, dev):
    ck = torch.load(MODELS / f"{tag}.pth", map_location=dev, weights_only=False)
    m = AE().to(dev); m.load_state_dict(ck["model"]); m.eval()
    return m, ck


# ========================================================= DiT ==============
def t_embed(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * f[None]
    return torch.cat([a.cos(), a.sin()], 1)


class DiTBlock(nn.Module):
    def __init__(self, w, heads):
        super().__init__()
        self.n1 = nn.LayerNorm(w, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(w, heads, batch_first=True)
        self.n2 = nn.LayerNorm(w, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(w, 4 * w), nn.GELU(approximate="tanh"), nn.Linear(4 * w, w))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(w, 6 * w))
        nn.init.zeros_(self.ada[1].weight); nn.init.zeros_(self.ada[1].bias)   # adaLN-Zero

    def forward(self, x, c):
        s1, b1, g1, s2, b2, g2 = self.ada(c).unsqueeze(1).chunk(6, -1)
        h = self.n1(x) * (1 + s1) + b1
        x = x + g1 * self.attn(h, h, h, need_weights=False)[0]
        return x + g2 * self.mlp(self.n2(x) * (1 + s2) + b2)


class DiT1D(nn.Module):
    def __init__(self, zc=16, L=512, patch=2, w=384, depth=12, heads=6, n_class=12):
        super().__init__()
        self.p, self.zc, self.n_class = patch, zc, n_class
        self.inp = nn.Linear(zc * patch, w)
        self.pos = nn.Parameter(torch.zeros(1, L // patch, w)); nn.init.normal_(self.pos, std=0.02)
        self.temb = nn.Sequential(nn.Linear(256, w), nn.SiLU(), nn.Linear(w, w))
        self.yemb = nn.Embedding(n_class + 1, w)                      # last index = null (CFG)
        self.blocks = nn.ModuleList([DiTBlock(w, heads) for _ in range(depth)])
        self.nf = nn.LayerNorm(w, elementwise_affine=False, eps=1e-6)
        self.adaf = nn.Sequential(nn.SiLU(), nn.Linear(w, 2 * w))
        self.out = nn.Linear(w, zc * patch)
        for m in (self.adaf[1], self.out):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, z, t, y_emb):
        B, C, L = z.shape
        x = z.reshape(B, C, L // self.p, self.p).permute(0, 2, 1, 3).reshape(B, L // self.p, C * self.p)
        x = self.inp(x) + self.pos
        c = self.temb(t_embed(t, 256)) + y_emb
        for blk in self.blocks:
            x = blk(x, c)
        s, b = self.adaf(c).unsqueeze(1).chunk(2, -1)
        x = self.out(self.nf(x) * (1 + s) + b)
        return x.reshape(B, L // self.p, C, self.p).permute(0, 2, 1, 3).reshape(B, C, L)


T_STEPS = 1000


def alphas_cumprod(T=T_STEPS, s=0.008):
    t = torch.linspace(0, T, T + 1) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    return (f[1:] / f[0]).clamp(1e-5, 0.9999)


def encode_latents(ae, U, dev, bs=32):
    M, S = [], []
    with torch.no_grad():
        for i in range(0, len(U), bs):
            m, lv = ae.encode(torch.from_numpy(U[i:i + bs]).to(dev))
            M.append(m.float().cpu()); S.append((0.5 * lv).exp().float().cpu())
    return torch.cat(M), torch.cat(S)


@torch.no_grad()
def ddim(model, ab, n, y_emb_fn, dev, steps=100, guidance=1.5, zc=16, L=512):
    ts = torch.linspace(T_STEPS - 1, 0, steps).long()
    z = torch.randn(n, zc, L, device=dev)
    yc, yn = y_emb_fn(n)
    for i, t in enumerate(ts):
        tt = torch.full((n,), int(t), device=dev)
        v = model(torch.cat([z, z]), torch.cat([tt, tt]), torch.cat([yc, yn]))
        vc, vu = v.chunk(2); v = vu + guidance * (vc - vu)
        a = ab[t].to(dev); x0 = a.sqrt() * z - (1 - a).sqrt() * v
        eps = (1 - a).sqrt() * z + a.sqrt() * v
        a_prev = ab[ts[i + 1]].to(dev) if i + 1 < len(ts) else torch.tensor(1.0, device=dev)
        z = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
    return z


def y_embedder(model, mode, sid_freq, rng, dev, n_class):
    """Returns fn(n) -> (conditional, null) embeddings.
    proportional: studies drawn with corpus frequencies (reproduces the corpus mix)
    uniform:      studies drawn equally (an even mix the corpus does not have)
    mix:          a random convex blend of two uniformly drawn studies"""
    def fn(n):
        if mode == "proportional":
            y = rng.choice(n_class, n, p=sid_freq); e = model.yemb(torch.as_tensor(y, device=dev))
        elif mode == "uniform":
            e = model.yemb(torch.as_tensor(rng.integers(0, n_class, n), device=dev))
        else:
            a_, b_ = (model.yemb(torch.as_tensor(rng.integers(0, n_class, n), device=dev)) for _ in range(2))
            w = torch.as_tensor(rng.uniform(0, 1, (n, 1)), dtype=torch.float32, device=dev)
            e = w * a_ + (1 - w) * b_
        return e, model.yemb(torch.full((n,), n_class, device=dev))
    return fn


def train_ldm(a):
    dev = torch.device(a.device); torch.manual_seed(a.seed)
    U, R, sid, uniq = load_corpus(); n_class = len(uniq)
    ae, ck = load_ae(a.ae, dev)
    Mu, Sd = encode_latents(ae, U, dev)
    scale = float(1.0 / Mu.std()); print(f"latents {tuple(Mu.shape)}, scale {scale:.3f}", flush=True)
    sel = np.random.default_rng(12345).choice(len(R), 600, replace=False); sel_f = bin_feat(R[sel])
    model = DiT1D(n_class=n_class).to(dev); ema = copy.deepcopy(model).eval()
    print(f"DiT {sum(p.numel() for p in model.parameters())/1e6:.1f}M params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    ab = alphas_cumprod().to(dev); Y = torch.as_tensor(sid); freq = np.bincount(sid) / len(sid)
    best, log, t0 = (1.1, -1), [], time.time()
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, len(Mu), (a.batch,))
        z0 = ((Mu[idx] + Sd[idx] * torch.randn_like(Mu[idx])) * scale).to(dev)
        y = Y[idx].to(dev); y = torch.where(torch.rand(a.batch, device=dev) < 0.1, n_class, y)
        t = torch.randint(0, T_STEPS, (a.batch,), device=dev); al = ab[t][:, None, None]
        eps = torch.randn_like(z0); zt = al.sqrt() * z0 + (1 - al).sqrt() * eps
        v_tgt = al.sqrt() * eps - (1 - al).sqrt() * z0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(zt, t, model.yemb(y))
        snr = ab[t] / (1 - ab[t]); w = torch.clamp(snr, max=5.0) / (snr + 1)      # Min-SNR-5, v-pred
        loss = (w * ((v.float() - v_tgt) ** 2).mean((1, 2))).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        d = min(0.9999, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        if step % a.eval_every == 0:
            rng = np.random.default_rng(step)
            z = ddim(ema, ab, 400, y_embedder(ema, "proportional", freq, rng, dev, n_class), dev,
                     steps=50, guidance=1.0)
            with torch.no_grad():
                xs = from_u(ae.decode(z / scale).float().cpu().numpy())
            auc = realism_auc(sel_f, bin_feat(xs))
            log.append({"step": step, "loss": float(loss), "sel_auc": auc})
            msg = f"step {step:6d}  loss {loss.item():.4f}  selection AUC {auc:.3f}  {time.time()-t0:.0f}s"
            if auc < best[0]:
                best = (auc, step)
                torch.save({"ema": ema.state_dict(), "scale": scale, "ae": a.ae, "classes": uniq,
                            "freq": freq.tolist(), "step": step, "auc": auc}, MODELS / f"{a.tag}.pth")
                msg += "  <- best"
            print(msg, flush=True); json.dump(log, open(LOGS / f"{a.tag}_log.json", "w"), indent=1)
    print(f"best selection AUC {best[0]:.3f} at step {best[1]}", flush=True)


# ===================================================== generation ===========
def finish(xs, n_out):
    """32,768-point x-space spectra -> 131,072-point corpus format, as for the GAN."""
    ppm = np.load(ROOT / "rebuild/pretrain/ppm_axis.npy")
    X = np.load(ROOT / "rebuild/pretrain/corpus_train.npy", mmap_mode="r")
    amp = float(np.median(np.abs(np.asarray(X[:200], dtype=np.float32)).max(1)))
    ppm_gen = np.linspace(ppm[0], ppm[-1], GEN_LEN)
    out = np.zeros((n_out, len(ppm)), dtype=np.float32)
    for j in range(n_out):
        v = np.interp(ppm, ppm_gen[::-1], xs[j][::-1])
        v[(ppm <= WATER[1]) & (ppm >= WATER[0])] = 0.0
        mx = np.abs(v).max(); out[j] = v / mx * amp if mx > 0 else v
    return out


def sample(a):
    dev = torch.device(a.device)
    ck = torch.load(MODELS / f"{a.ldm}.pth", map_location=dev, weights_only=False)
    ae, _ = load_ae(ck["ae"], dev); n_class = len(ck["classes"])
    model = DiT1D(n_class=n_class).to(dev); model.load_state_dict(ck["ema"]); model.eval()
    ab = alphas_cumprod().to(dev); rng = np.random.default_rng(a.seed); xs = []
    fn = y_embedder(model, a.study_mix, np.array(ck["freq"]), rng, dev, n_class)
    for i in range(0, a.n, 100):
        z = ddim(model, ab, min(100, a.n - i), fn, dev, steps=a.steps, guidance=a.guidance)
        with torch.no_grad():
            xs.append(from_u(ae.decode(z / ck["scale"]).float().cpu().numpy()))
    out = finish(np.vstack(xs), a.n)
    p = ROOT / f"rebuild/pretrain/synthetic/{a.tag}.npy"; np.save(p, out)
    print(f"wrote {p.relative_to(ROOT)} {out.shape} (ldm {a.ldm} step {ck['step']}, "
          f"{a.study_mix}, guidance {a.guidance})", flush=True)


def vae_sample(a):
    """Sample the autoencoder's prior directly: z ~ N(0, I) -> decoder."""
    dev = torch.device(a.device); ae, ck = load_ae(a.ae, dev); torch.manual_seed(a.seed); xs = []
    with torch.no_grad():
        for i in range(0, a.n, 50):
            z = torch.randn(min(50, a.n - i), 16, 512, device=dev)
            xs.append(from_u(ae.decode(z).float().cpu().numpy()))
    out = finish(np.vstack(xs), a.n)
    p = ROOT / f"rebuild/pretrain/synthetic/{a.tag}.npy"; np.save(p, out)
    print(f"wrote {p.relative_to(ROOT)} {out.shape} (prior samples of {a.ae}, beta {ck['beta']})", flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ae"); p.add_argument("--tag", required=True); p.add_argument("--beta", type=float, default=1e-6)
    p.add_argument("--epochs", type=int, default=60); p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p = sub.add_parser("ldm"); p.add_argument("--ae", required=True); p.add_argument("--tag", required=True)
    p.add_argument("--steps", type=int, default=60000); p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--eval-every", type=int, default=2500)
    p = sub.add_parser("sample"); p.add_argument("--ldm", required=True); p.add_argument("--tag", required=True)
    p.add_argument("--n", type=int, default=4000); p.add_argument("--steps", type=int, default=100)
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--study-mix", choices=["proportional", "uniform", "mix"], default="proportional")
    p = sub.add_parser("vae-sample"); p.add_argument("--ae", required=True); p.add_argument("--tag", required=True)
    p.add_argument("--n", type=int, default=4000)
    for s in sub.choices.values():
        s.add_argument("--device", default="cuda:0"); s.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"ae": train_ae, "ldm": train_ldm, "sample": sample, "vae-sample": vae_sample}[a.cmd](a)


if __name__ == "__main__":
    main()
