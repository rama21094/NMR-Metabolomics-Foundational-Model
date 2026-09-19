#!/usr/bin/env python3
"""Add eff_studies to fixed-budget manifest entries written before it existed.

Replicates build_scaling_subsets.py's selection exactly -- same pool order, same
rng seeds (777 + k*10 + seed), same quota-and-redistribute logic -- so the
composition recovered here is the composition those subsets were actually built
with. Nothing is re-saved: the .npy files the trained checkpoints depend on are
never touched.

Verifies the recovered row count against the manifest's n_rows and refuses to
write if any entry disagrees, which is what would catch a drift between this copy
of the logic and the builder's.
"""
import csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rebuild/pretrain/scaling"


def compositions(budget, ks=(2, 4, 8, 12), seeds=(0, 1, 2)):
    split = [r for r in csv.DictReader(open(ROOT / "rebuild/pretrain/corpus_split.csv"))
             if r["split"] == "train"]
    studies = np.array([r["study"] for r in split])
    counts = Counter(studies)
    by = defaultdict(list)
    for i, s in enumerate(studies):
        by[s].append(i)
    ordered = [s for s, _ in counts.most_common()]
    big, small = ordered[:len(ordered) // 2], ordered[len(ordered) // 2:][::-1]
    woven = [x for pair in zip(big, small) for x in pair]
    woven += [s for s in ordered if s not in woven]
    usable = [s for s, n in counts.most_common() if n >= 50]

    out = {}
    for k in ks:
        pool = [s for s in (woven if k >= len(usable) else
                            [x for x in woven if x in usable])][:k]
        if len(pool) < k:
            pool = ordered[:k]
        for seed in seeds:
            rng = np.random.default_rng(777 + k * 10 + seed)
            per = budget // len(pool)
            idx, short = [], 0
            for s in pool:
                rows = by[s]
                take = min(per, len(rows))
                short += per - take
                idx += list(rng.choice(rows, size=take, replace=False))
            if short:
                spare = [i for s in pool for i in by[s] if i not in set(idx)]
                if spare:
                    idx += list(rng.choice(spare, size=min(short, len(spare)),
                                           replace=False))
            idx = sorted(set(idx))
            S = set(idx)
            comp = np.array([sum(1 for i in by[s2] if i in S) for s2 in pool], float)
            pr = comp / comp.sum()
            eff = float(np.exp(-(pr[pr > 0] * np.log(pr[pr > 0])).sum()))
            out[f"fixed{budget}_k{k:02d}_seed{seed}"] = (round(eff, 2), len(idx))
    return out


def analytic_eff(counts, subsets):
    """Effective studies for the row and study axes, which need no rng.

    Row subsets take round(frac * n_s) from EVERY study, so the proportions are
    preserved and effective diversity is pinned near the full corpus value.
    Study subsets keep whole studies. Both are exact from the counts alone.
    """
    def eff(c):
        p = np.array(list(c), float)
        p = p / p.sum()
        return float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))

    ordered = [s for s, _ in counts.most_common()]
    big, small = ordered[:len(ordered) // 2], ordered[len(ordered) // 2:][::-1]
    woven = [x for pair in zip(big, small) for x in pair]
    woven += [s for s in ordered if s not in woven]

    out = {}
    for name in subsets:
        if name.startswith("rows"):
            frac = int(name[4:7]) / 100
            c = [max(1, int(round(frac * n))) for n in counts.values()]
        elif name.startswith("studies"):
            k = int(name[7:].split("_")[0])
            c = [counts[s] for s in woven[:k]]
        else:
            continue
        out[name] = (round(eff(c), 2), int(sum(c)))
    return out


def main() -> None:
    mf = OUT / (sys.argv[1] if len(sys.argv) > 1 else "manifest.json")
    man = json.load(open(mf))
    budgets = {int(r["subset"].split("_")[0][5:])
               for r in man if r.get("axis") == "fixed_budget"}
    comp = {}
    for b in budgets:
        comp.update(compositions(b))

    names = [r["subset"] for r in man if r.get("axis") in ("row", "study")]
    split = [x for x in csv.DictReader(open(ROOT / "rebuild/pretrain/corpus_split.csv"))
             if x["split"] == "train"]
    comp.update(analytic_eff(Counter(x["study"] for x in split), names))

    bad, n = [], 0
    for r in man:
        if r["subset"] not in comp:
            continue
        eff, rows = comp[r["subset"]]
        if rows != r["n_rows"]:
            bad.append(f"{r['subset']}: manifest {r['n_rows']} rows, recomputed {rows}")
        r["eff_studies"] = eff
        n += 1
    if bad:
        print("REFUSING to write -- recomputed composition disagrees with manifest:")
        for b in bad:
            print("   " + b)
        raise SystemExit(1)
    json.dump(man, open(mf, "w"), indent=1)
    print(f"backfilled eff_studies on {n} entries in {mf.name}")
    for r in man:
        if r.get("axis") == "fixed_budget" and r["seed"] == 0:
            print(f"   {r['subset']:<20} {r['n_rows']:>5,} rows  "
                  f"nominal {r['n_studies']:>2}  effective {r['eff_studies']}")


if __name__ == "__main__":
    main()
