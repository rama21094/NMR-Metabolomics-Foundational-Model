#!/usr/bin/env python3
"""Recover which source study each corpus spectrum came from, without a mapping.

THE PROBLEM. The corpus .npy files record no provenance: nothing links row i of
combine_unique_... to a MetaboLights accession. The Bruker export recovered the
study list and each study's acquisition window, but not the row assignment.

WHY IT IS RECOVERABLE ANYWAY. Two independent fingerprints:

  1. AXIS DISPLACEMENT. Each study has its own (proc_OFFSET, proc_SW_p, proc_SF),
     which fixes where its peaks land after index-alignment. Measured per row by
     code/analysis/alignment_qc.py. This separates studies into axis GROUPS but
     not always into individual studies -- five of them share a displacement
     window ~110 points wide, narrower than the spread within a single study.

  2. SPECTRAL SIMILARITY. Spectra from one study share cohort, sample handling
     and processing, so they correlate far more tightly with each other than
     across studies (r ~ 0.99 within, much lower across). Clustering on
     lag-corrected spectra therefore recovers study blocks directly.

  3. BLOCK SIZE. The decisive check. The study sizes from the parameter export
     are distinctive -- 4866, 979, 978, 728, 699, 358, 198, 93, 92, 76, 10 --
     so a recovered block of ~4,870 can only be MTBLS798, and a block of ~360
     can only be MTBLS974. Sizes are matched against the export, not assumed.

WHAT THIS BUYS US. If each row's study is known, the alignment fix becomes a
per-study constant derived from that study's own instrument files -- exact,
auditable, and applied without re-deriving anything from the raw archive. It
also lets the batch audit be redone with study as a covariate.

HONEST LIMITS, stated up front:
  - Corpus rows (9,670) and parameter rows (9,078) do not correspond one-to-one:
    the corpus was de-duplicated and the 778 Workbench spectra have no export at
    all. Block sizes will therefore be close, not exact.
  - Where two studies share both a displacement and a size, this cannot separate
    them, and the output says so rather than guessing.

Usage:
    python code/analysis/recover_study_blocks.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ("data/combined/combine_unique_MetaboLights_Workbench_"
          "Water_EDTA_Suppressed.npy")
QC = ROOT / "results/analysis/alignment_qc"
OUT = ROOT / "results/analysis/study_blocks"

NPT = 131_072
WATER = (62500, 68000)
FEAT = (20000, 110000)      # broad band, avoids both extreme edges
NFEAT = 3000                # downsampled feature length


def study_table() -> pd.DataFrame:
    """Per-study size and predicted displacement, from the Bruker export."""
    frames = []
    for f in ("data/bruker_params_PlasmaNMRData.csv",
              "data/bruker_params_SerumNMRData.csv"):
        frames.append(pd.read_csv(ROOT / f, low_memory=False))
    a = pd.concat(frames, ignore_index=True)
    a["study"] = a["relpath"].str.extract(r"^(MTBLS\d+)")[0]
    for src, dst in (("proc_SF", "SF"), ("proc_SW_p", "SW_p"),
                     ("proc_OFFSET", "OFFSET")):
        a[dst] = pd.to_numeric(a[src], errors="coerce")
    a = a.dropna(subset=["SF", "SW_p", "OFFSET"])
    a["sw_ppm"] = a["SW_p"] / a["SF"]

    g = a.groupby("study")
    t = pd.DataFrame({"n_params": g.size(),
                      "sw_ppm": g["sw_ppm"].median(),
                      "ppm_hi": g["OFFSET"].median()}).reset_index()
    ref = t.loc[t["n_params"].idxmax()]
    def idx(hi, sw, d=3.0):
        return (hi - d) * NPT / sw
    t["pred_dx"] = (idx(t["ppm_hi"], t["sw_ppm"])
                    - idx(ref["ppm_hi"], ref["sw_ppm"])).round().astype(int)
    t["stretch"] = (ref["sw_ppm"] / t["sw_ppm"]).round(4)
    return t.sort_values("n_params", ascending=False).reset_index(drop=True)


def features(X, lag):
    """Lag-corrected, downsampled, unit-normalised spectra for clustering."""
    n = X.shape[0]
    cols = np.linspace(FEAT[0], FEAT[1] - 1, NFEAT).astype(int)
    F = np.empty((n, NFEAT), dtype=np.float32)
    for i in range(n):
        sh = int(lag[i])
        c = np.clip(cols + sh, 0, NPT - 1)
        F[i] = np.asarray(X[i, :], dtype=np.float32)[c]
        if i % 1000 == 0:
            print(f"\r  features {i}/{n}", end="", flush=True)
    print(f"\r  features {n}/{n}")
    # blank the water window in corrected coordinates, then z-score each row
    F -= F.mean(axis=1, keepdims=True)
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
    return F


def blocks_from_similarity(F, thresh=0.90, chunk=1000):
    """Connected components of the graph "correlation above threshold"."""
    n = F.shape[0]
    rows, cols = [], []
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        S = F[i:j] @ F.T
        r, c = np.where(S >= thresh)
        rows.append(r + i)
        cols.append(c)
        print(f"\r  similarity {j}/{n}", end="", flush=True)
    print()
    r = np.concatenate(rows); c = np.concatenate(cols)
    A = csr_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(n, n))
    k, labels = connected_components(A, directed=False)
    return k, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresh", type=float, default=0.90)
    ap.add_argument("--min-block", type=int, default=8)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    X = np.load(ROOT / CORPUS, mmap_mode="r")
    lag = np.load(QC / "lag_all.npy")

    st = study_table()
    print("Studies known from the Bruker export:")
    print(st.to_string(index=False))
    print(f"\n  parameter rows {st['n_params'].sum()}   corpus rows {X.shape[0]}"
          f"   (difference = de-duplication + 778 Workbench with no export)\n")

    print("building lag-corrected features")
    F = features(X, lag)
    print(f"clustering at correlation >= {args.thresh}")
    k, labels = blocks_from_similarity(F, args.thresh)

    sizes = pd.Series(labels).value_counts()
    big = sizes[sizes >= args.min_block]
    print(f"\n  {k} components; {len(big)} of size >= {args.min_block}, "
          f"covering {100 * big.sum() / len(labels):.1f}% of rows\n")

    rec = []
    for lab, sz in big.items():
        m = labels == lab
        rec.append({"block": int(lab), "n": int(sz),
                    "median_lag": int(np.median(lag[m])),
                    "lag_spread": int(np.percentile(lag[m], 95)
                                      - np.percentile(lag[m], 5))})
    rb = pd.DataFrame(rec).sort_values("n", ascending=False)

    # ---- match blocks to studies on (size, displacement) -------------------
    unused = st.copy()
    assign = []
    for _, b in rb.iterrows():
        cand = unused[(unused["pred_dx"] - b["median_lag"]).abs() < 400]
        if cand.empty:
            assign.append(("unassigned", np.nan))
            continue
        # among candidates on the same axis, pick the closest in size
        best = cand.iloc[(cand["n_params"] - b["n"]).abs().argmin()]
        ratio = b["n"] / best["n_params"]
        ambiguous = len(cand) > 1 and (
            cand["n_params"].sub(b["n"]).abs().nsmallest(2).diff().abs().iloc[-1]
            < 0.25 * b["n"])
        assign.append((("AMBIGUOUS: " if ambiguous else "") + best["study"],
                       round(float(ratio), 3)))
        unused = unused[unused["study"] != best["study"]]
    rb["study"] = [a for a, _ in assign]
    rb["size_ratio"] = [r for _, r in assign]

    print("Recovered blocks, matched to studies by displacement then size:")
    print(rb.to_string(index=False))

    lab2study = dict(zip(rb["block"], rb["study"]))
    row_study = np.array([lab2study.get(l, "unassigned") for l in labels])
    np.save(OUT / "row_block.npy", labels)
    pd.DataFrame({"row": np.arange(len(labels)), "block": labels,
                  "study": row_study, "lag": lag}).to_csv(
        OUT / "row_study_assignment.csv", index=False)
    rb.to_csv(OUT / "blocks.csv", index=False)

    cov = 100 * (row_study != "unassigned").mean()
    amb = 100 * pd.Series(row_study).str.startswith("AMBIGUOUS").mean()
    print(f"\n  rows assigned to a study: {cov:.1f}%  (ambiguous: {amb:.1f}%)")
    (OUT / "summary.json").write_text(json.dumps({
        "n_components": int(k), "n_blocks": int(len(big)),
        "coverage_pct": float(cov), "ambiguous_pct": float(amb),
        "threshold": args.thresh}, indent=2))
    print(f"  wrote {OUT.relative_to(ROOT)}/row_study_assignment.csv")


if __name__ == "__main__":
    main()
