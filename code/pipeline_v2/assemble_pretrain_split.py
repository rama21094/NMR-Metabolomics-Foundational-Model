#!/usr/bin/env python3
"""Assemble the pretraining corpus with a STUDY-level held-out split.

Rows within one study are near-duplicates of each other -- the rebuilt corpus's
within-study nearest-neighbour correlation is 0.989. A random row split therefore
puts near-copies of every validation spectrum into training, and the held-out
reconstruction loss it reports is close to meaningless. Splitting by study is the
only honest way to ask whether the model generalises beyond what it has seen.

Held-out studies are chosen to be several whole small-to-mid studies rather than
one large one, so the held-out set spans multiple instruments and protocols
instead of measuring how well the model fits a single lab.

MTBLS798 (51% of the corpus) stays in training: holding it out would remove half
the data, and holding out only part of it would leak by the argument above.
"""
import csv
import re
from pathlib import Path

import numpy as np

HELD_OUT = ["ST000826", "MTBLS2387", "ST000892", "ST000306", "MTBLS10958",
            "MTBLS11188", "ST000939"]
OUT = Path("rebuild/pretrain")


def study(path: str) -> str:
    m = re.search(r"(MTBLS\d+|ST\d+)", path)
    return m.group(1) if m else "UNKNOWN"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    src, rows_meta = [], []
    for pre in ("serum", "plasma", "workbench"):
        X = np.load(f"rebuild/{pre}_spectra.npy", mmap_mode="r")
        prov = list(csv.DictReader(open(f"rebuild/{pre}_provenance.csv")))
        assert len(prov) == X.shape[0], pre
        src.append(X)
        for r in prov:
            rows_meta.append((pre, int(r["row"]), study(r["pdata_path"]),
                              r["pdata_path"], r["sw_ppm"]))

    held = set(HELD_OUT)
    split = ["heldout" if s in held else "train" for _, _, s, _, _ in rows_meta]
    npts = src[0].shape[1]
    off = {}
    n = 0
    for pre, X in zip(("serum", "plasma", "workbench"), src):
        off[pre] = n; n += X.shape[0]

    for which in ("train", "heldout"):
        idx = [i for i, s in enumerate(split) if s == which]
        out = np.lib.format.open_memmap(OUT / f"corpus_{which}.npy", mode="w+",
                                        dtype=np.float32, shape=(len(idx), npts))
        for j, i in enumerate(idx):
            pre, row, _, _, _ = rows_meta[i]
            X = src[("serum", "plasma", "workbench").index(pre)]
            out[j] = X[row]
        out.flush(); del out
        studies = sorted({rows_meta[i][2] for i in idx})
        print(f"{which:<8} {len(idx):>6,} rows   {len(studies):>2} studies")

    with open(OUT / "corpus_split.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "source", "source_row", "study", "pdata_path", "sw_ppm"])
        for (pre, row, st, path, sw), s in zip(rows_meta, split):
            w.writerow([s, pre, row, st, path, sw])

    np.save(OUT / "ppm_axis.npy", np.load("rebuild/serum_ppm_axis.npy"))
    print(f"\nheld-out studies: {', '.join(HELD_OUT)}")
    print(f"wrote {OUT}/corpus_train.npy, corpus_heldout.npy, corpus_split.csv, ppm_axis.npy")


if __name__ == "__main__":
    main()
