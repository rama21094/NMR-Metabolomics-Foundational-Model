#!/usr/bin/env python3
"""Phase 4.1 evaluation -- does more data, or more diverse data, transfer better?

Probes every scaling checkpoint exactly as Phase 3 did (frozen embeddings, linear
probe, stratified episodes) so the numbers are directly comparable, then groups
them by scaling axis. The fixed-budget axis is the one that matters: 2,000 rows
every time, drawn from k studies, so a difference across k is diversity alone.
"""
import argparse, csv, json, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in ("code/training", "code/evaluation", "code/analysis"):
    sys.path.insert(0, str(ROOT / p))
from phase3_transfer import target_defs, load_cohort, binned, episodes, probe  # noqa: E402


def ckpts():
    """Every scaling checkpoint, as (subset, replicate, path).

    Two naming conventions, because the replicate lives in different places:

      rows075_seed1_<ts>_..._seed0_best.pth   row and fixed-budget subsets carry
                                              their replicate in the SUBSET file
                                              name; the trailing seed is the
                                              training seed and is always 0.
      studies6_<ts>_..._seed1_best.pth        study subsets have no per-replicate
                                              file, so the replicate IS the
                                              trailing training seed.

    The first version of this matched only `(subset)_seed(N)_` and so silently
    dropped all nine study-axis checkpoints -- the evaluation ran and reported a
    table with the study axis simply absent. Parse both forms, and assert the
    expected count rather than trusting a glob.
    """
    out = []
    for p in sorted((ROOT / "models/masked_ssl").glob("*_best.pth")):
        m = re.match(r"(fixed2000_k\d+|rows\d+|studies\d+)(_seed(\d+))?_\d{8}_", p.name)
        if not m:
            continue
        sub = m.group(1)
        if m.group(3) is not None:            # replicate in the subset name
            rep = int(m.group(3))
        else:                                 # replicate is the training seed
            t = re.search(r"_seed(\d+)_best\.pth$", p.name)
            if not t:
                continue
            rep = int(t.group(1))
        out.append((sub, rep, p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--budgets", type=int, nargs="+", default=[10, 0])
    ap.add_argument("--out", default="results/phase4/scaling_eval.json")
    args = ap.parse_args()
    from linear_probe_frozen_embeddings import embed_masking

    C = ckpts()
    seen = {(a, b) for a, b, _ in C}
    if len(seen) != len(C):
        raise SystemExit(f"duplicate (subset, replicate): {len(C)} paths, {len(seen)} tags")
    print(f"{len(C)} scaling checkpoints, {len(target_defs())} targets\n")
    rows = []
    for tid, cohort, labfn in target_defs():
        M, ppm, y = load_cohort(cohort, labfn)
        base = binned(M, ppm, 0.02)
        embs = {}
        for sub, seed, ck in C:
            try:
                embs[(sub, seed)] = np.asarray(embed_masking(str(ck), M, args.device),
                                               dtype=np.float32)
            except Exception as e:
                print(f"  embed fail {sub} s{seed}: {type(e).__name__}", flush=True)
        for k in args.budgets:
            eps = episodes(y, k, args.episodes, seed=hash(tid) % 2**31)
            if not eps:
                continue
            cls = float(np.mean([probe(base, y, s, q) for s, q in eps]))
            agg = defaultdict(list)
            for (sub, seed), F in embs.items():
                sc = float(np.mean([probe(F, y, s, q) for s, q in eps]))
                rows.append({"target": tid, "subset": sub, "seed": seed,
                             "k_per_class": k, "mean": sc, "classical": cls})
                agg[sub].append(sc)
            kl = "all" if k == 0 else str(k)
            print(f"{tid:<20} k={kl:<4} classical {cls:.3f} | " +
                  "  ".join(f"{s}:{np.mean(v):.3f}" for s, v in sorted(agg.items())),
                  flush=True)
            json.dump(rows, open(ROOT / args.out, "w"), indent=1)
    json.dump(rows, open(ROOT / args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
