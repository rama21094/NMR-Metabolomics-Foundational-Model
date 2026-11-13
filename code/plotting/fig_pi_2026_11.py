#!/usr/bin/env python3
"""Figures for the PI update of November 2026: one per action point of the
2026-10-23 meeting. Reads only saved results; writes docs/figures/pi13_*.png.

  pi13_pca.png            item 1: PCA components vs binning, against the transformer
  pi13_seeds.png          item 2: do seeds converge? (CKA, matched weights)
  pi13_readouts.png       item 3: mean / 16-region / attention / CLS, two corpora
  pi13_generators.png     item 4: realism of every generator + example spectra
  pi13_volume.png         item 4: pretraining with 1k / 4k / 16k diffusion spectra
  pi13_realism_tests.png  correction: study-grouped vs fair realism test
  pi13_simple_vs_few.png  open point: variance held vs accuracy bought per PCA band

Set FONT_DIR to a folder with SourceSans3-*.ttf to use Source Sans 3.
"""
import collections, glob, json, os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/figures"
INK, ORANGE, GREY, LIGHT = "#1E2761", "#C2571A", "#8A8F9C", "#D9DCE3"
for f in glob.glob(os.path.join(os.environ.get("FONT_DIR", "/nonexistent"), "SourceSans3-*.ttf")):
    fm.fontManager.addfont(f)
plt.rcParams.update({"font.size": 17, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 19, "axes.titleweight": "bold", "axes.titlecolor": INK,
                     "axes.labelcolor": "#333333", "xtick.color": "#333333", "ytick.color": "#333333"})
if any("Source Sans 3" == f.name for f in fm.fontManager.ttflist):
    plt.rcParams["font.family"] = "Source Sans 3"


def load(p):
    return json.load(open(ROOT / p))


def tally(rows):
    d = np.array([r["diff"] for r in rows]); lo = np.array([r["lo"] for r in rows]); hi = np.array([r["hi"] for r in rows])
    return d, int((lo > 0).sum()), int((hi < 0).sum()), int(((lo <= 0) & (hi >= 0)).sum())


def by(rows, key):
    g = collections.defaultdict(list)
    for r in rows:
        g[r[key]].append(r)
    return g


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, facecolor="white"); plt.close(fig); print("wrote", OUT / name)


# --- item 1: PCA vs binning ---------------------------------------------------
P = load("results/final/pca_vs_binning.json")
G1 = by(load("results/final/pooling_eval_baseline.json"), "readout")
ks = [5, 10, 20, 40, 50, 100]
pca = {k: tally(by(P["results"], "features")[f"pca{k}"]) for k in ks}
fig, (a, b) = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1, 1.5]})
ev = [P["explained"][str(k)] * 100 for k in ks]
a.plot(ks, ev, "o-", color=INK, lw=2.5, ms=9)
for k, v in zip(ks, ev):
    a.annotate(f"{v:.1f}%", (k, v), textcoords="offset points", xytext=(0, -26 if k in (5, 50) else 10),
               ha="center", fontsize=15, color=INK)
a.set_xscale("log"); a.set_xticks(ks); a.set_xticklabels(ks); a.set_ylim(88, 101.2)
a.set_xlabel("number of PCA components"); a.set_ylabel("variance of the corpus explained (%)")
a.set_title("A few components hold almost everything", loc="left")
m = [pca[k][0].mean() for k in ks]
b.axhline(0, color=GREY, lw=1.5)
b.text(4.6, 0.003, "standard ML (all ~470 bins)", va="bottom", color="#555555", fontsize=14)
for lab, key, ls in [("transformer, mean pooling", "mean", "--"), ("transformer, 16-region pooling", "regional16", ":")]:
    y = tally(G1[key])[0].mean(); b.axhline(y, color=INK, ls=ls, lw=2)
    b.text(115, y - 0.003, f"{lab} ({y:+.3f})", va="top", ha="right", color=INK, fontsize=14)
b.plot(ks, m, "o-", color=ORANGE, lw=2.5, ms=10, label="PCA components + same classifier")
for k, v in zip(ks, m):
    _, bt, w, lv = pca[k]
    b.annotate(f"{v:+.3f}\n{bt}/{w}/{lv}", (k, v), textcoords="offset points", xytext=(0, -44 if k == 5 else 12),
               ha="center", fontsize=13, color=ORANGE)
b.set_xscale("log"); b.set_xticks(ks); b.set_xticklabels(ks); b.set_xlim(4.3, 118); b.set_ylim(-0.1, 0.035)
b.set_xlabel("number of PCA components"); b.set_ylabel("balanced accuracy minus standard ML\n(mean of 21 comparisons)")
b.set_title("PCA matches or beats the transformer from 10 components on", loc="left")
b.legend(frameon=False, loc="lower right", fontsize=14)
fig.tight_layout(); save(fig, "pi13_pca.png")

# --- item 2: seed convergence -----------------------------------------------
S = load("results/final/seed_similarity.json")
c = S["mean_pool|standardised"]
fig, (a, b) = plt.subplots(1, 2, figsize=(16, 5.4), gridspec_kw={"width_ratios": [1.3, 1]})
rng = np.random.default_rng(0)
groups = [("two trained models\n(different seeds)", c["cka_trained_vs_trained"], INK),
          ("a trained model vs\nan untrained one", c["cka_trained_vs_random"], GREY)]
for i, (lab, v, col) in enumerate(groups):
    a.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=70, color=col, alpha=0.8, lw=0)
    a.hlines(np.mean(v), i - 0.25, i + 0.25, color=col, lw=3)
    a.text(i + 0.3, np.mean(v), f"{np.mean(v):.2f}", va="center", fontsize=18, weight="bold", color=col)
a.set_xticks([0, 1]); a.set_xticklabels([g[0] for g in groups]); a.set_xlim(-0.5, 1.7); a.set_ylim(0.8, 1.0)
a.set_ylabel("similarity of the embeddings (CKA)\n1 = same up to rotation and scale")
a.set_title("What the models compute: the same", loc="left")
w = list(S["weight_matched_abs_corr"].values()); base = S["weight_matched_abs_corr_random_baseline"]
b.bar([0, 1], [np.mean(w), base], color=[INK, GREY], width=0.55)
b.scatter(rng.uniform(-0.1, 0.1, len(w)), w, s=30, color="white", zorder=3, lw=0)
for i, v in enumerate([np.mean(w), base]):
    b.text(i, v + 0.014, f"{v:.2f}", ha="center", fontsize=18, weight="bold", color=[INK, GREY][i])
b.set_xticks([0, 1]); b.set_xticklabels(["two trained models,\nbest-matched neurons", "two random matrices,\nbest-matched"])
b.set_ylim(0, 0.3); b.set_ylabel("correlation of last-layer weights")
b.set_title("The individual weights: different", loc="left")
fig.tight_layout(); save(fig, "pi13_seeds.png")

# --- item 3: readouts ---------------------------------------------------------
cfg = [("pooling_eval_baseline", "8,545 spectra (as before)", [("mean", "mean pooling"), ("regional16", "16-region pooling"), ("attention", "learned attention pooling")]),
       ("pooling_eval_full", "all 9,480 spectra", [("mean", "mean pooling"), ("regional16", "16-region pooling")]),
       ("pooling_eval_fullcls", "all 9,480 spectra + CLS token", [("mean", "mean pooling"), ("regional16", "16-region pooling"), ("cls", "CLS token")])]
fig, ax = plt.subplots(figsize=(16, 6.6))
y = 0; yt, yl = [], []
for fn, corpus, rs in cfg:
    R = by(load(f"results/final/{fn}.json"), "readout")
    ax.text(-0.195, y - 0.15, corpus, fontsize=16, weight="bold", color=INK, va="bottom")
    y -= 0.55
    for key, lab in rs:
        d, bt, wr, lv = tally(R[key])
        new = key in ("attention", "cls")
        col = ORANGE if new else INK
        ax.scatter(d, y + rng.uniform(-0.12, 0.12, len(d)), s=34, color=col, alpha=0.45, lw=0)
        ax.scatter([d.mean()], [y], s=230, color=col, marker="D", edgecolor="white", lw=1.5, zorder=3)
        ax.text(0.085, y, f"{d.mean():+.3f}    {bt} / {wr} / {lv}", va="center", fontsize=15, color=col)
        yt.append(y); yl.append(lab + ("  (new)" if new else "")); y -= 0.6
    y -= 0.25
ax.axvline(0, color=GREY, lw=1.5)
ax.text(0.003, 0.25, "standard ML", color="#555555", fontsize=14)
ax.text(0.085, 0.25, "mean     better / worse / level", fontsize=13, color="#555555")
ax.set_yticks(yt); ax.set_yticklabels(yl); ax.set_ylim(y + 0.3, 0.55); ax.set_xlim(-0.2, 0.2)
ax.set_xlabel("balanced accuracy minus standard ML  (each dot = one of 21 comparisons; diamond = mean)")
for t, lab in zip(ax.get_yticklabels(), yl):
    t.set_color(ORANGE if "(new)" in lab else INK)
fig.tight_layout(); save(fig, "pi13_readouts.png")

# --- item 4: generators -------------------------------------------------------
Rz = load("results/phase5/synthetic_realism.json")
order = ["GISSMO + lipids", "VAE prior samples", "GAN v1", "GAN v2", "GAN v2 study-mixing",
         "Latent diffusion, even mix", "Latent diffusion, corpus mix"]
names = {"GISSMO + lipids": "GISSMO (physics)", "VAE prior samples": "VAE (new)",
         "Latent diffusion, even mix": "Diffusion, even study mix (new)",
         "Latent diffusion, corpus mix": "Diffusion, corpus mix (new)"}
ppm = np.load(ROOT / "rebuild/pretrain/ppm_axis.npy")
fig, (a, b) = plt.subplots(1, 2, figsize=(16, 5.8), gridspec_kw={"width_ratios": [1.15, 1]})
v = [Rz[k]["auc_stratified"] for k in order]
cols = [ORANGE if "diffusion" in k.lower() else (GREY if k.startswith(("GISSMO", "VAE")) else INK) for k in order]
yy = np.arange(len(order))[::-1]
a.barh(yy, [x - 0.5 for x in v], left=0.5, color=cols, height=0.62)
for yi, x in zip(yy, v):
    a.text(x + 0.006, yi, f"{x:.3f}", va="center", fontsize=15, weight="bold")
a.set_yticks(yy); a.set_yticklabels([names.get(k, k) for k in order]); a.set_xlim(0.5, 1.07)
a.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
a.set_xlabel("realism test AUC  (0.5 = cannot be told from real; 1.0 = always caught)")
a.set_title("Diffusion is the most realistic generator", loc="left")
X = np.load(ROOT / "rebuild/pretrain/corpus_train.npy", mmap_mode="r")
L = np.load(ROOT / "rebuild/pretrain/synthetic/ldm_v2_proportional.npy", mmap_mode="r")
Gv = np.load(ROOT / "rebuild/pretrain/synthetic/gan_v2.npy", mmap_mode="r")
w = (ppm > 0.8) & (ppm < 4.3)
tr = [("real spectrum", np.asarray(X[1234]), INK), ("diffusion sample", np.asarray(L[17]), ORANGE),
      ("GAN v2 sample", np.asarray(Gv[17]), GREY)]
off = 0
for lab, s, col in tr:
    s = np.minimum(s[w] / np.percentile(s[w], 99.5), 1.05)
    b.plot(ppm[w], s + off, color=col, lw=0.8)
    off += 1.15
b.invert_xaxis(); b.spines["left"].set_visible(False); b.set_ylim(-0.1, off + 0.1)
b.set_yticks([0.2 + 1.15 * i for i in range(3)]); b.set_yticklabels([t[0] for t in tr]); b.tick_params(axis="y", length=0)
for t, (_, _, col) in zip(b.get_yticklabels(), tr):
    t.set_color(col); t.set_fontweight("bold")
b.set_xlabel("chemical shift (ppm), 0.8 to 4.3"); b.set_title("One example of each, same window", loc="left")
fig.tight_layout(); save(fig, "pi13_generators.png")

# --- item 4: synthetic volume curve -------------------------------------------
A = load("results/final/synaug_eval_regional16.json")
pairs = by(A, "a")
vol = [("baseline", "0\n(real only)"), ("aug1k", "1,000"), ("aug4k", "4,000"), ("aug16k", "16,000")]
fig, (a, b) = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1.35, 1]})
means = []
for i, (k, lab) in enumerate(vol):
    d, bt, wr, lv = tally([r for r in pairs[k] if r["b"] == "classical"])
    a.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=34, color=INK if i == 0 else ORANGE, alpha=0.45, lw=0)
    means.append(d.mean())
    a.text(i, 0.08, f"{bt} / {wr} / {lv}", ha="center", fontsize=14, color="#555555")
a.plot(range(4), means, "D-", color=INK, ms=13, lw=2, mec="white", zorder=3)
for i, m_ in enumerate(means):
    a.text(i, -0.21, f"mean {m_:+.3f}", ha="center", fontsize=15, weight="bold", color=INK)
a.axhline(0, color=GREY, lw=1.5); a.text(3.85, 0.004, "standard ML", ha="right", color="#555555", fontsize=14)
a.text(-0.4, 0.105, "better / worse / level vs standard ML", fontsize=13, color="#555555")
a.set_xticks(range(4)); a.set_xticklabels([v_[1] for v_ in vol]); a.set_ylim(-0.23, 0.12); a.set_xlim(-0.45, 3.9)
a.set_xlabel("diffusion spectra added to the 8,545 real ones")
a.set_ylabel("balanced accuracy minus standard ML")
a.set_title("The gap to standard ML does not move", loc="left")
cnt = [tally([r for r in pairs[k] if r["b"] == "baseline"]) for k, _ in vol[1:]]
x = np.arange(3)
bb = np.array([c_[1] for c_ in cnt]); ww = np.array([c_[2] for c_ in cnt]); ll = np.array([c_[3] for c_ in cnt])
b.bar(x, bb, color=ORANGE, width=0.6, label="better")
b.bar(x, ll, bottom=bb, color=LIGHT, width=0.6, label="level (no clear difference)")
b.bar(x, ww, bottom=bb + ll, color=INK, width=0.6, label="worse")
for i in range(3):
    for val, base_, col in [(bb[i], 0, "white"), (ll[i], bb[i], "#333333"), (ww[i], bb[i] + ll[i], "white")]:
        if val:
            b.text(i, base_ + val / 2, str(val), ha="center", va="center", fontsize=16, weight="bold", color=col)
    b.text(i, 21.6, f"mean {cnt[i][0].mean():+.3f}", ha="center", fontsize=14, color="#333333")
b.set_xticks(x); b.set_xticklabels(["+1,000", "+4,000", "+16,000"]); b.set_ylim(0, 25); b.set_yticks([0, 7, 14, 21])
b.set_ylabel("number of the 21 comparisons")
b.set_title("Against real-only pretraining", loc="left")
b.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=13)
fig.tight_layout(); save(fig, "pi13_volume.png")

# --- correction: grouped vs fair realism test --------------------------------
order2 = ["GAN v1", "GAN v2", "GAN v2 study-mixing", "Latent diffusion, even mix", "Latent diffusion, corpus mix"]
fig, ax = plt.subplots(figsize=(16, 5.2))
for i, k in enumerate(order2[::-1]):
    g_, f_ = Rz[k]["auc"], Rz[k]["auc_stratified"]
    ax.plot([g_, f_], [i, i], color=LIGHT, lw=6, zorder=1)
    ax.scatter([g_], [i], s=220, color=GREY, zorder=2, label="old test (folds grouped by study)" if i == 0 else None)
    ax.scatter([f_], [i], s=220, color=ORANGE, zorder=3, label="fair test (folds mixed)" if i == 0 else None)
    ax.text(g_, i + 0.28, f"{g_:.2f}", ha="center", fontsize=14, color="#555555")
    ax.text(f_, i + 0.28, f"{f_:.2f}", ha="center", fontsize=14, color=ORANGE, weight="bold")
ax.axvspan(0, 0.5, color="#F3E3D9", zorder=0)
ax.text(0.25, -0.75, "below 0.5 is impossible for a fair test:\nit would mean fakes look 'more real than real'",
        ha="center", fontsize=14, color=ORANGE)
ax.axvline(0.5, color=GREY, lw=1.5, ls="--")
ax.set_yticks(range(len(order2))); ax.set_yticklabels([names.get(k, k) for k in order2[::-1]])
ax.set_xlim(0.15, 1.05); ax.set_ylim(-1.2, len(order2) - 0.3)
ax.set_xlabel("realism test AUC  (0.5 = cannot be told from real)")
ax.legend(frameon=False, loc="lower right", fontsize=14)
fig.tight_layout(); save(fig, "pi13_realism_tests.png")

# --- open point: simple data vs too few samples -------------------------------
bands = [(5, 10), (10, 20), (20, 40), (40, 50), (50, 100)]
lab = ["6–10", "11–20", "21–40", "41–50", "51–100"]
var = [100 * (P["explained"][str(hi)] - P["explained"][str(lo)]) for lo, hi in bands]
gain = [pca[hi][0].mean() - pca[lo][0].mean() for lo, hi in bands]
fig, (a, b) = plt.subplots(1, 2, figsize=(16, 5.4))
x = np.arange(len(bands))
a.bar(x, var, color=INK, width=0.6); a.set_yscale("log"); a.set_ylim(0.05, 12)
a.set_yticks([0.1, 1, 10]); a.set_yticklabels(["0.1%", "1%", "10%"])
for i, v_ in enumerate(var):
    a.text(i, v_ * 1.12, f"{v_:.2f}%", ha="center", fontsize=15, weight="bold", color=INK)
a.set_xticks(x); a.set_xticklabels(lab); a.set_xlabel("PCA components (components 1–5 hold 90.5%)")
a.set_ylabel("share of corpus variance (log scale)")
a.set_title("Variance each group of components holds", loc="left")
b.bar(x, gain, color=[ORANGE if g_ > 0.01 else LIGHT for g_ in gain], width=0.6)
for i, g_ in enumerate(gain):
    b.text(i, g_ + (0.001 if g_ >= 0 else -0.004), f"{g_:+.3f}", ha="center", fontsize=15, weight="bold",
           color=ORANGE if g_ > 0.01 else "#555555")
b.axhline(0, color=GREY, lw=1)
b.set_xticks(x); b.set_xticklabels(lab); b.set_xlabel("PCA components added")
b.set_ylabel("accuracy gained by adding them\n(mean of 21 comparisons)"); b.set_ylim(-0.008, 0.04)
b.set_title("Accuracy each group adds", loc="left")
fig.tight_layout(); save(fig, "pi13_simple_vs_few.png")
