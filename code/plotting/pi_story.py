#!/usr/bin/env python3
"""Four figures telling the corpus-alignment story end to end, for the PI.

Fig 1  where we started        -- the corpus as the old pipeline stored it
Fig 2  the first fix           -- anchoring every spectrum at 0 ppm
Fig 3  what the anchoring hid  -- the spectral-width defect, which no amount of
                                  looking at spectra could have revealed
Fig 4  the rebuild             -- reprocessed from raw onto one real ppm axis

Run from the repository root.
"""
import csv
import collections
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)
OLD = Path("data/combined")
NEW = Path("rebuild")
RNG = np.random.default_rng(0)
NROW = 300
# The aliphatic window: lactate 1.33, alanine 1.47. Sharp, always present,
# and close enough together that misalignment is unmistakable.
ZOOM = (1.15, 1.65)      # tight: lactate + alanine, for the heat maps
WIDE = (0.55, 4.45)      # the whole informative region, for the overlays


def sample(path, k=NROW):
    X = np.load(path, mmap_mode="r")
    rows = np.sort(RNG.choice(X.shape[0], size=min(k, X.shape[0]), replace=False))
    return np.array(X[rows], dtype=np.float32), X.shape


def norm(M):
    m = np.max(np.abs(M), axis=1, keepdims=True)
    m[m == 0] = 1
    return M / m


def panel(ax, M, x, title, xlabel, zoom=None):
    if zoom is not None:
        sel = (x <= zoom[1]) & (x >= zoom[0])
        M, x = M[:, sel], x[sel]
    M = norm(M)
    for row in M:
        ax.plot(x, row, lw=0.25, alpha=0.30, color="#1f4e79")
    ax.plot(x, np.median(M, axis=0), lw=1.3, color="#c00000", label="median")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(labelsize=7)
    if zoom is not None:
        ax.set_xlim(zoom[1], zoom[0])
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc="upper right")


def heat(ax, M, x, title, zoom=None):
    if zoom is not None:
        sel = (x <= zoom[1]) & (x >= zoom[0])
        M, x = M[:, sel], x[sel]
    M = norm(M)
    ax.imshow(M, aspect="auto", origin="lower", cmap="magma", vmin=0,
              vmax=float(np.percentile(M, 99.0)) or 1.0,
              extent=[x[0], x[-1], 0, M.shape[0]])
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("ppm", fontsize=8)
    ax.set_ylabel("spectrum", fontsize=8)
    ax.tick_params(labelsize=7)


def fig1():
    """The corpus as the old pipeline produced it."""
    M, shp = sample(OLD / "combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed.npy")
    ax_nom = np.load("fromSSD/common_ppm_axis.npy")      # what the pipeline believed
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    panel(axes[0], M, ax_nom, f"{NROW} spectra of {shp[0]:,}, as stored\n"
          "(nominal axis: every row assumed to share it)", "ppm (nominal)", WIDE)
    heat(axes[1], M, ax_nom, "the same rows as an image -- each horizontal line is one spectrum", ZOOM)
    fig.suptitle("FIGURE 1   Where we started: the corpus as the original pipeline stored it",
                 fontsize=11, weight="bold")
    fig.text(0.5, 0.005,
             "The tallest line in the lactate window falls anywhere across a 0.41 ppm range "
             "(95% of spectra), and its median sits at 1.62 rather than lactate's 1.33. "
             "Nothing here is on a shared, correct chemical-shift scale.",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(OUT / "fig1_as_stored.png", dpi=150)
    plt.close(fig)
    print("fig1 done", shp)


def fig2():
    """After anchoring every spectrum on its 0 ppm reference peak."""
    M, shp = sample(OLD / "combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_ref0ppm_clean.npy")
    ax_nom = np.load("fromSSD/common_ppm_axis.npy")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    panel(axes[0], M, ax_nom, f"{NROW} spectra of {shp[0]:,}, after 0 ppm anchoring",
          "ppm (nominal)", WIDE)
    heat(axes[1], M, ax_nom, "peaks now line up -- and this is where we thought we were finished", ZOOM)
    fig.suptitle("FIGURE 2   The first fix: anchoring every spectrum at 0 ppm",
                 fontsize=11, weight="bold")
    fig.text(0.5, 0.005,
             "The spread collapses to zero -- every spectrum now agrees exactly. But they agree "
             "at 1.62 ppm, not lactate's 1.33, and agreeing says nothing about whether one ppm "
             "step means the same thing in every row. It does not. See Figure 3.",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(OUT / "fig2_after_anchor.png", dpi=150)
    plt.close(fig)
    print("fig2 done", shp)


def census():
    c = collections.Counter()
    for pre in ("serum", "plasma", "workbench"):
        f = NEW / f"{pre}_provenance.csv"
        if not f.exists():
            continue
        for r in csv.DictReader(open(f)):
            c[round(float(r["sw_ppm"]), 3)] += 1
    return c


def fig3():
    """The defect the anchoring hid: spectra were never on one ppm scale."""
    c = census()
    ws = sorted(c)
    n = [c[w] for w in ws]
    dom = max(c, key=c.get)
    fig = plt.figure(figsize=(13, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.1])

    ax = fig.add_subplot(gs[0, 0])
    # "different" means different enough to matter: >0.5% from the dominant sweep.
    # 20.017 and 20.031 are the same acquisition as 20.024 to within 0.03%.
    diff = {w: abs(w - dom) / dom > 0.005 for w in ws}
    col = ["#c00000" if diff[w] else "#1f4e79" for w in ws]
    ax.barh([f"{w:.3f}" for w in ws], n, color=col)
    for i, v in enumerate(n):
        ax.text(v, i, f" {v:,}", va="center", fontsize=7)
    ax.set_xlabel("spectra", fontsize=8)
    ax.set_ylabel("acquired spectral width (ppm)", fontsize=8)
    off = sum(v for w, v in c.items() if abs(w - dom) / dom > 0.005)
    ax.set_title(f"Acquired spectral widths in the corpus\n"
                 f"{off:,} of {sum(n):,} ({100*off/sum(n):.1f}%) were acquired at a materially\n"
                 f"different sweep from the dominant {dom:.3f} ppm (red)", fontsize=9)
    ax.tick_params(labelsize=7)

    # what that does to a peak, under the old pipeline
    ax = fig.add_subplot(gs[0, 1])
    pts = 131072
    for w, lab, cl in [(dom, f"{dom:.1f} ppm sweep", "#1f4e79"),
                       (11.988, "11.99 ppm sweep", "#c00000")]:
        ppm_per_pt = w / pts
        width_ppm = 0.0025                       # a typical 1.5 Hz line at 600 MHz
        npts = width_ppm / ppm_per_pt
        xx = np.linspace(-3, 3, 400)
        ax.plot(xx * npts, np.exp(-xx ** 2), lw=1.6, color=cl,
                label=f"{lab}: {npts:.0f} points across the line")
    ax.set_xlabel("points", fontsize=8)
    ax.set_ylabel("intensity", fontsize=8)
    ax.set_title("The same metabolite line, as the old pipeline stored it\n"
                 "(each spectrum resampled onto its OWN range)", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    axes_found = [("corpus", 10.500, -1.500), ("Barth cohort", 9.948, -2.083),
                  ("Workbench cohort", 14.704, -5.327), ("Childhood obesity", 9.955, -2.068)]
    txt = ("Each dataset was written with its own\n"
           '"common" ppm axis file:\n\n')
    for nm, a, b in axes_found:
        txt += f"   {nm:<20} {a:+7.3f} .. {b:+7.3f}\n"
    txt += ("\nThe old scripts computed a shared axis\n"
            "and then never used it. Row index i meant\n"
            "a different chemical shift in every study.\n\n"
            "This is invisible to the eye: the spectra\n"
            "look perfectly normal one at a time.")
    ax.text(0.0, 0.98, txt, va="top", family="monospace", fontsize=7.6)
    ax.set_title("Why inspecting 9,000 spectra could not have found it", fontsize=9)

    fig.suptitle("FIGURE 3   What the anchoring hid: the spectra were never on one ppm scale",
                 fontsize=11, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "fig3_the_defect.png", dpi=150)
    plt.close(fig)
    print("fig3 done", dict(sorted(c.items())))


def fig4():
    """The rebuild: reprocessed from raw onto one real ppm axis."""
    ppm = np.load(NEW / "ppm_axis_8x.npy")
    parts, total = [], 0
    for pre in ("serum", "plasma", "workbench"):
        f = NEW / f"{pre}_spectra_8x.npy"
        if not f.exists():
            continue
        X = np.load(f, mmap_mode="r")
        total += X.shape[0]
        k = max(1, int(NROW * X.shape[0] / 9480))
        rows = np.sort(RNG.choice(X.shape[0], size=min(k, X.shape[0]), replace=False))
        parts.append(np.array(X[rows], dtype=np.float32))
    M = np.vstack(parts)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    panel(axes[0], M, ppm, f"{M.shape[0]} spectra sampled across all three sources "
          f"({total:,} rows)", "ppm (real, from the Bruker parameters)", WIDE)
    for v, lab in [(1.330, "lactate"), (1.470, "alanine")]:
        axes[0].axvline(v, color="k", lw=0.6, ls="--", alpha=0.6)
        axes[0].text(v, 1.02, lab, fontsize=6.5, ha="center")
    heat(axes[1], M, ppm, "lines now have one width and one spacing; the remaining\n"
         "horizontal spread is each study's own referencing", ZOOM)
    fig.suptitle("FIGURE 4   The rebuild: every spectrum reprocessed from raw onto one ppm axis",
                 fontsize=11, weight="bold")
    fig.text(0.5, 0.005,
             "Verified: every spectral-width group peaks at scale 1.00 and carries the correct "
             "2.780 ppm lactate CH3-to-CH separation -- the geometry is now right. The residual "
             "+/-0.05 ppm spread is how the source studies referenced themselves (TSP, DSS, or "
             "nothing); it is real, quantified, and deliberately left in for the model to learn "
             "invariance to rather than corrected by inventing a reference the data lacks.",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(OUT / "fig4_rebuilt.png", dpi=150)
    plt.close(fig)
    print("fig4 done", M.shape, total)


def apexes(path, axis, k=1200, win=(1.15, 1.75)):
    """Where the tallest line in the lactate window actually falls, per spectrum."""
    X = np.load(path, mmap_mode="r")
    ppm = np.load(axis) if isinstance(axis, (str, Path)) else axis
    a = int(np.argmin(np.abs(ppm - win[1]))); b = int(np.argmin(np.abs(ppm - win[0])))
    rows = np.sort(RNG.choice(X.shape[0], size=min(k, X.shape[0]), replace=False))
    out = []
    for i in range(0, len(rows), 100):
        for y in np.array(X[rows[i:i + 100], a:b], dtype=np.float32):
            if y.max() > 0:
                out.append(ppm[a + int(np.argmax(y))])
    return np.array(out)


def fig5():
    """The same measurement at each stage: where does lactate actually land?"""
    old = apexes(OLD / "combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed.npy",
                 "fromSSD/common_ppm_axis.npy")
    anc = apexes(OLD / "combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_ref0ppm_clean.npy",
                 "fromSSD/common_ppm_axis.npy")
    ppm8 = np.load(NEW / "ppm_axis_8x.npy")
    new = np.concatenate([apexes(NEW / f"{p}_spectra_8x.npy", ppm8, k=400)
                          for p in ("serum", "plasma", "workbench")])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    bins = np.linspace(1.15, 1.75, 160)
    for ax, (nm, v, col) in zip(axes, [
            ("1. as stored", old, "#808080"),
            ("2. after 0 ppm anchoring", anc, "#c00000"),
            ("3. after the rebuild", new, "#1f4e79")]):
        ax.hist(v, bins=bins, color=col)
        ax.axvline(1.330, color="k", lw=1.2, ls="--")
        ax.text(1.330, ax.get_ylim()[1] * 0.97, " true lactate 1.330", fontsize=7.5, va="top")
        lo, hi = np.percentile(v, [2.5, 97.5])
        ax.set_title(f"{nm}\nmedian {np.median(v):.3f} ppm   "
                     f"95% span {hi - lo:.3f} ppm", fontsize=9)
        ax.set_xlabel("position of the tallest line in the lactate window (ppm)", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.set_xlim(1.75, 1.15)
    axes[0].set_ylabel("spectra", fontsize=8)
    fig.suptitle("FIGURE 5   The same measurement at each stage: where does lactate actually land?",
                 fontsize=11, weight="bold")
    fig.text(0.5, 0.005,
             "Stage 2 is the cautionary panel: anchoring drove the 95% span to 0.000 ppm, every "
             "spectrum in exact agreement -- at 1.616 ppm, which is not where lactate is.\n"
             "Agreement between spectra is not the same as correctness. Only stage 3 has both: "
             "the right chemical shift, with 0.117 ppm of genuine inter-study spread left in.",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=[0, 0.09, 1, 0.91])
    fig.savefig(OUT / "fig5_alignment_measured.png", dpi=150)
    plt.close(fig)
    print(f"fig5 done  as-stored span {np.percentile(old,97.5)-np.percentile(old,2.5):.4f}, "
          f"anchored {np.percentile(anc,97.5)-np.percentile(anc,2.5):.4f}, "
          f"rebuilt {np.percentile(new,97.5)-np.percentile(new,2.5):.4f}")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("\nwrote:", *(p.name for p in sorted(OUT.glob("fig*.png"))))
