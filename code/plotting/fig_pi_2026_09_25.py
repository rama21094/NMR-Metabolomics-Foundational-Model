#!/usr/bin/env python3
"""Figures for the 2026-09-25 PI update. Plain-language titles, one point each.

Palette is the one the PI preferred: pale wheat ground, blue for our model, red
for standard ML. Every series is also directly labelled and has its own marker,
so identity never depends on colour alone.
"""
import csv, json, collections, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/figures"
GROUND, BLUE, RED, GREY, INK = "#FBF6E9", "#2E5E9E", "#C0272D", "#8A8A8A", "#1E2761"
plt.rcParams.update({"font.size": 12, "axes.edgecolor": "#999", "axes.labelcolor": "#333",
                     "xtick.color": "#444", "ytick.color": "#444"})


def style(ax):
    ax.set_facecolor(GROUND)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.3, lw=0.6)


# ---- A: twelve studies, effectively four ---------------------------------
split = [r for r in csv.DictReader(open(ROOT / "rebuild/pretrain/corpus_split.csv"))
         if r["split"] == "train"]
cnt = collections.Counter(r["study"] for r in split)
names, vals = zip(*cnt.most_common())
share = np.array(vals) / sum(vals) * 100
fig, ax = plt.subplots(figsize=(10, 4.6))
style(ax); ax.grid(axis="x", alpha=0.3, lw=0.6); ax.grid(axis="y", visible=False)
y = np.arange(len(names))[::-1]
ax.barh(y, share, color=[BLUE if i == 0 else "#9DB3D1" for i in range(len(names))],
        height=0.7)
for yi, s, v in zip(y, share, vals):
    ax.text(s + 0.8, yi, f"{s:.1f}%  ({v:,} spectra)", va="center", fontsize=10.5, color="#333")
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=10.5)
ax.set_xlabel("share of the training spectra (%)"); ax.set_xlim(0, 75)
fig.tight_layout(); fig.savefig(OUT / "pi25_study_share.png", dpi=200, facecolor="white")
plt.close(fig)

# ---- B: more spectra vs more studies -------------------------------------
sc = json.load(open(ROOT / "results/phase4/scaling_eval_45.json"))
man = {}
for mf in ("manifest.json", "manifest_fixed4000.json"):
    for r in json.load(open(ROOT / "rebuild/pretrain/scaling" / mf)):
        man[r["subset"]] = r
agg = collections.defaultdict(list); cls = []
for r in sc:
    if r["k_per_class"] == 0:
        agg[(r["subset"], r["seed"])].append(r["mean"]); cls.append(r["classical"])
per = {k: np.mean(v) for k, v in agg.items()}
def pts(prefix):
    out = []
    for (sub, seed), v in per.items():
        if sub.startswith(prefix):
            m = man.get(f"{sub}_seed{seed}", man.get(sub))
            out.append((m["n_rows"], m["eff_studies"], v))
    return out
rowsP = pts("rows"); divP = pts("fixed") + pts("studies")
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax in axs: style(ax); ax.axhline(np.mean(cls), color=RED, lw=2, ls="--")
g = collections.defaultdict(list)
for n, e, v in rowsP: g[n].append(v)
xs = sorted(g); axs[0].plot(xs, [np.mean(g[x]) for x in xs], color=BLUE, lw=2.4, marker="o", ms=8)
axs[0].errorbar(xs, [np.mean(g[x]) for x in xs], yerr=[np.std(g[x], ddof=1) for x in xs],
                color=BLUE, lw=0, elinewidth=1.2, capsize=4)
axs[0].set_xlabel("number of spectra used for pretraining")
axs[0].set_ylabel("accuracy on patient cohorts")
axs[0].set_title("More spectra: no gain after ~2,000", color=INK, weight="bold", loc="left")
axs[0].text(xs[-1], np.mean(cls) + 0.006, "standard ML", color=RED, ha="right", fontsize=11)
E = np.array([p[1] for p in divP]); V = np.array([p[2] for p in divP])
axs[1].scatter(E, V, color=BLUE, s=45, alpha=0.75, edgecolor="white", lw=0.8)
b = np.polyfit(E, V, 1); xx = np.linspace(E.min(), E.max(), 50)
axs[1].plot(xx, np.polyval(b, xx), color=BLUE, lw=2)
axs[1].set_xlabel("effective number of studies")
axs[1].set_title("More studies: small but consistent gain", color=INK, weight="bold", loc="left")
axs[1].text(E.max(), np.mean(cls) + 0.006, "standard ML", color=RED, ha="right", fontsize=11)
axs[0].set_ylim(0.58, 0.74)
fig.tight_layout(); fig.savefig(OUT / "pi25_spectra_vs_studies.png", dpi=200, facecolor="white")
plt.close(fig)

# ---- C: bigger models ----------------------------------------------------
cap = json.load(open(ROOT / "results/phase6/capacity_eval.json"))
cg = collections.defaultdict(list); ccls = []
for r in cap:
    if r["k_per_class"] == 0:
        cg[(r["params_m"], r["seed"])].append(r["mean"]); ccls.append(r["classical"])
bp = collections.defaultdict(list)
for (p, s), v in cg.items(): bp[p].append(np.mean(v))
P = sorted(bp)
fig, ax = plt.subplots(figsize=(10, 4.6)); style(ax)
ax.errorbar(P, [np.mean(bp[p]) for p in P], yerr=[np.std(bp[p], ddof=1) for p in P],
            color=BLUE, lw=2.4, marker="o", ms=8, capsize=4, elinewidth=1.2)
ax.axhline(np.mean(ccls), color=RED, lw=2, ls="--")
ax.text(P[-1], np.mean(ccls) + 0.005, "standard ML", color=RED, ha="right", fontsize=11)
ax.text(P[-1], np.mean(bp[P[-1]]) - 0.014, "our model", color=BLUE, ha="right", fontsize=11)
ax.set_xscale("log"); ax.set_xticks(P); ax.set_xticklabels([f"{p:g}M" for p in P])
ax.axvline(3.99, color=GREY, lw=1, ls=":")
ax.text(3.99, 0.64, " current size", color=GREY, fontsize=10)
ax.set_xlabel("model size (millions of parameters, log scale)")
ax.set_ylabel("accuracy on patient cohorts"); ax.set_ylim(0.62, 0.74)
fig.tight_layout(); fig.savefig(OUT / "pi25_model_size.png", dpi=200, facecolor="white")
plt.close(fig)

# ---- D: what synthetic spectra look like ---------------------------------
ppm = np.load(ROOT / "rebuild/pretrain/ppm_axis.npy")
X = np.load(ROOT / "rebuild/pretrain/corpus_train.npy", mmap_mode="r")
sets = [("Real spectrum", np.asarray(X[4321], dtype=np.float32), INK),
        ("GISSMO (built from 87 known metabolites)",
         np.load(ROOT / "rebuild/pretrain/synthetic/probe_lip.npy")[3], GREY),
        ("GAN (learned from the real spectra)",
         np.load(ROOT / "rebuild/pretrain/synthetic/gan.npy")[7], BLUE)]
m = (ppm <= 4.4) & (ppm >= 0.7)
fig, axs = plt.subplots(3, 1, figsize=(12, 5.6), sharex=True)
for ax, (t, y, c) in zip(axs, sets):
    style(ax); ax.grid(visible=False)
    yy = y[m] / (np.abs(y[m]).max() + 1e-12)
    ax.plot(ppm[m], yy, color=c, lw=0.9); ax.set_yticks([])
    ax.text(0.01, 0.82, t, transform=ax.transAxes, fontsize=12, color=c, weight="bold")
axs[-1].invert_xaxis(); axs[-1].set_xlabel("chemical shift (ppm)")
fig.tight_layout(); fig.savefig(OUT / "pi25_synthetic_examples.png", dpi=200, facecolor="white")
plt.close(fig)
print("wrote 4 figures")
