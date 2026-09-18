#!/usr/bin/env python3
"""Phase 4.1 -- corpus subsets for the two scaling axes.

The whiteboard asks "need for more data? Is data limiting." That question has two
different answers in this corpus and they must be separated, because 51% of the
9,480 rows come from a single study (MTBLS798) and within-study nearest-neighbour
correlation is 0.989. Adding rows from a study already present adds very little
new information; adding a study adds a new instrument, protocol and population.

So we scale along two axes and report both:

  ROW axis    all 12 training studies kept, a fraction of rows sampled within
              each study. Tests "more spectra".
  STUDY axis  whole studies dropped, all rows kept in those retained. Tests
              "more diversity" -- but CONFOUNDED, because dropping studies also
              drops rows, and the 3-study subset still holds 68% of the corpus.

  FIXED axis  the decisive one: a constant row budget drawn from k studies, k in
              {2, 4, 8, 12}, rows split as evenly as each study allows. Row count
              is held fixed so any difference is diversity alone. This is the
              experiment that separates "more data" from "more diverse data".

If performance tracks rows, the recommendation to the field is more data. If it
tracks studies, the recommendation is more DIVERSE data -- a sharper and more
useful claim, and one that changes what "we need more public data" means in the
G2 terminal branch.

Study subsets are nested (3 in 6 in 9 in 12) and chosen to span the size range
rather than taking the largest, so the 3-study subset is not simply MTBLS798
plus scraps.
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OUT = Path("rebuild/pretrain/scaling")
ROW_FRACS = [0.10, 0.25, 0.50, 0.75]
STUDY_COUNTS = [3, 6, 9]
SEEDS = [0, 1, 2]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X = np.load("rebuild/pretrain/corpus_train.npy", mmap_mode="r")
    split = [r for r in csv.DictReader(open("rebuild/pretrain/corpus_split.csv"))
             if r["split"] == "train"]
    assert len(split) == X.shape[0]
    studies = np.array([r["study"] for r in split])
    counts = Counter(studies)
    by_study = defaultdict(list)
    for i, s in enumerate(studies):
        by_study[s].append(i)
    print(f"train: {X.shape[0]:,} rows, {len(counts)} studies")
    for s, n in counts.most_common():
        print(f"   {s:<12}{n:>6}")

    # Nested study subsets spanning the size range: interleave large and small
    # so a 3-study subset is not just the giant study plus two fragments.
    ordered = [s for s, _ in counts.most_common()]
    big, small = ordered[:len(ordered) // 2], ordered[len(ordered) // 2:][::-1]
    woven = [x for pair in zip(big, small) for x in pair]
    woven += [s for s in ordered if s not in woven]

    manifest = []
    for k in STUDY_COUNTS:
        keep = set(woven[:k])
        idx = np.array(sorted(i for s in keep for i in by_study[s]))
        name = f"studies{k}"
        np.save(OUT / f"{name}.npy", np.array(X[idx], dtype=np.float32))
        manifest.append({"subset": name, "axis": "study", "n_rows": int(len(idx)),
                         "n_studies": k, "studies": sorted(keep), "seed": None})
        print(f"{name:<12}{len(idx):>7,} rows  {k} studies")

    rng_base = 12345
    for frac in ROW_FRACS:
        for seed in SEEDS:
            rng = np.random.default_rng(rng_base + int(frac * 1000) + seed)
            idx = []
            for s, rows in by_study.items():
                take = max(1, int(round(frac * len(rows))))
                idx += list(rng.choice(rows, size=take, replace=False))
            idx = np.array(sorted(idx))
            name = f"rows{int(frac * 100):03d}_seed{seed}"
            np.save(OUT / f"{name}.npy", np.array(X[idx], dtype=np.float32))
            manifest.append({"subset": name, "axis": "row", "n_rows": int(len(idx)),
                             "n_studies": len(by_study), "studies": sorted(by_study),
                             "seed": seed})
        print(f"rows{int(frac * 100):03d}     {len(idx):>7,} rows  "
              f"{len(by_study)} studies  x{len(SEEDS)} seeds")

    # ---- fixed-budget diversity axis -------------------------------------
    # 2,000 rows every time; only the number of contributing studies changes.
    BUDGET = 2000
    usable = [s for s, n in counts.most_common() if n >= 50]
    for k in (2, 4, 8, 12):
        pool = [s for s in (woven if k >= len(usable) else
                            [x for x in woven if x in usable])][:k]
        if len(pool) < k:
            pool = ordered[:k]
        for seed in SEEDS:
            rng = np.random.default_rng(777 + k * 10 + seed)
            per = BUDGET // len(pool)
            idx, short = [], 0
            for s in pool:
                rows = by_study[s]
                take = min(per, len(rows))
                short += per - take
                idx += list(rng.choice(rows, size=take, replace=False))
            # redistribute any shortfall over the studies that still have rows
            if short:
                spare = [i for s in pool for i in by_study[s] if i not in set(idx)]
                if spare:
                    idx += list(rng.choice(spare, size=min(short, len(spare)),
                                           replace=False))
            idx = np.array(sorted(set(idx)))
            name = f"fixed{BUDGET}_k{k:02d}_seed{seed}"
            np.save(OUT / f"{name}.npy", np.array(X[idx], dtype=np.float32))
            manifest.append({"subset": name, "axis": "fixed_budget",
                             "n_rows": int(len(idx)), "n_studies": len(pool),
                             "studies": sorted(pool), "seed": seed})
        print(f"fixed{BUDGET}_k{k:02d}  {len(idx):>6,} rows  {len(pool)} studies  x{len(SEEDS)} seeds")

    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=1)
    print(f"\nwrote {len(manifest)} subsets to {OUT}")


if __name__ == "__main__":
    main()
