#!/usr/bin/env python3
"""Slide 4 and 6 figures, redrawn as differences from standard ML.

The first versions plotted balanced accuracy averaged over six targets. That
average mixes tasks whose chance levels are 0.50 (binary), 0.33 (3-class) and
0.25 (4-class), so the pooled value is not the accuracy of any cohort. Here every
target is expressed as (pretrained - standard ML) on the same splits, so each sits
on the same zero line; zero means parity. Thin lines are the individual targets;
the bold line is their mean; error bars are the spread across training seeds.
"""
import csv, json, collections
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/figures"
GROUND, BLUE, RED, GREY, INK = "#FBF6E9", "#2E5E9E", "#C0272D", "#8A8A8A", "#1E2761"
plt.rcParams.update({"font.size": 12})


def style(ax):
    ax.set_facecolor(GROUND)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.3, lw=0.6)
    ax.axhline(0, color=RED, lw=2, ls="--")


def diffs(rows, keyfn, k=0):
    """(x, target) -> list over seeds of (ssl - classical)."""
    d = collections.defaultdict(list)
    for r in rows:
        if r["k_per_class"] == k:
            d[(keyfn(r), r["target"], r["seed"])].append(r["mean"] - r["classical"])
    out = collections.defaultdict(list)
    for (x, t, s), v in d.items():
        out[(x, t)].append(np.mean(v))
    return out


def plot_lines(ax, D, xs, xlab):
    T = sorted({t for _, t in D})
    for t in T:
        ax.plot(xs, [np.mean(D[(x, t)]) for x in xs], color=BLUE, alpha=0.28, lw=1.1)
    per_seed = collections.defaultdict(list)
    mean = [np.mean([np.mean(D[(x, t)]) for t in T]) for x in xs]
    # seed spread of the target-mean difference
    err = []
    for x in xs:
        n = min(len(D[(x, t)]) for t in T)
        sm = [np.mean([D[(x, t)][i] for t in T]) for i in range(n)]
        err.append(np.std(sm, ddof=1) if n > 1 else 0)
    ax.errorbar(xs, mean, yerr=err, color=BLUE, lw=2.6, marker="o", ms=8, capsize=4)
    ax.set_xlabel(xlab)
    return mean


# ---- slide 4: rows and effective studies ---------------------------------
sc = json.load(open(ROOT / "results/phase4/scaling_eval_45.json"))
man = {}
for mf in ("manifest.json", "manifest_fixed4000.json"):
    for r in json.load(open(ROOT / "rebuild/pretrain/scaling" / mf)):
        man[r["subset"]] = r
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax in axs: style(ax)
rows_only = [r for r in sc if r["subset"].startswith("rows")]
Dr = diffs(rows_only, lambda r: man[f"{r['subset']}_seed{r['seed']}"]["n_rows"])
xs = sorted({x for x, _ in Dr})
plot_lines(axs[0], Dr, xs, "number of spectra used for pretraining")
axs[0].set_ylabel("our model minus standard ML")
axs[0].set_title("More spectra: flat", color=INK, weight="bold", loc="left")
axs[0].text(xs[-1], 0.004, "standard ML (parity)", color=RED, ha="right", fontsize=10.5)

div = [r for r in sc if r["subset"].startswith(("fixed", "studies"))]
pts = collections.defaultdict(list)
for r in div:
    m = man.get(f"{r['subset']}_seed{r['seed']}", man.get(r["subset"]))
    pts[(r["subset"], r["seed"])].append((m["eff_studies"], r["mean"] - r["classical"]))
E = np.array([v[0][0] for v in pts.values()]); V = np.array([np.mean([p[1] for p in v]) for v in pts.values()])
axs[1].scatter(E, V, color=BLUE, s=45, alpha=0.75, edgecolor="white", lw=0.8)
b = np.polyfit(E, V, 1); xx = np.linspace(E.min(), E.max(), 50)
axs[1].plot(xx, np.polyval(b, xx), color=BLUE, lw=2)
axs[1].set_xlabel("effective number of studies (rebalanced subsets)")
axs[1].set_title("More studies: gap narrows, never closes", color=INK, weight="bold", loc="left")
axs[0].set_ylim(-0.16, 0.02)
fig.tight_layout(); fig.savefig(OUT / "pi25_spectra_vs_studies.png", dpi=200, facecolor="white")
plt.close(fig)

# ---- slide 6: model size -------------------------------------------------
cap = json.load(open(ROOT / "results/phase6/capacity_eval.json"))
Dc = diffs(cap, lambda r: r["params_m"])
P = sorted({x for x, _ in Dc})
fig, ax = plt.subplots(figsize=(10, 4.6)); style(ax)
mean = plot_lines(ax, Dc, P, "model size (millions of parameters, log scale)")
ax.set_xscale("log"); ax.set_xticks(P); ax.set_xticklabels([f"{p:g}M" for p in P])
ax.set_ylabel("our model minus standard ML")
ax.axvline(3.99, color=GREY, lw=1, ls=":"); ax.text(3.99, -0.15, " current size", color=GREY, fontsize=10)
ax.text(P[-1], 0.004, "standard ML (parity)", color=RED, ha="right", fontsize=10.5)
ax.text(P[1], -0.135, "thin lines: individual cohorts", color=BLUE, fontsize=10, alpha=0.8)
ax.set_ylim(-0.16, 0.02)
fig.tight_layout(); fig.savefig(OUT / "pi25_model_size.png", dpi=200, facecolor="white")
plt.close(fig)
print("wrote slide 4 and 6 difference figures")
