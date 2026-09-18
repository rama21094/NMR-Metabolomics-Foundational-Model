#!/usr/bin/env python3
"""Aggregate Phase 3 into the gate-G1 verdict.

Reads the per-host transfer_*.json files (one row per target x representation x
label budget, every episode score retained) and answers three questions:

  1. Does the best SSL representation beat classical ML, per target and budget?
  2. Does pretraining beat its own random-init backbone? -- which separates
     "pretraining helped" from "this architecture helped".
  3. Which objective is best, averaged over seeds?

Differences are PAIRED: every representation saw the same episodes, so the
per-episode difference is the right statistic and its spread is the right
uncertainty. A mean difference smaller than the project's measured single-run
sd of 0.045 is not treated as a result.

The headline comparator is each objective's MEAN OVER SEEDS, not the best of the
15 checkpoints. Taking a maximum over 15 correlated draws is a selection bias
large enough to invent a result: on Barth at k=20 the best-of-15 scores 0.827
against classical's 0.770, while the three objectives' seed means are 0.707,
0.663 and 0.635 -- all clearly below. This project has already retracted one
"SSL beats classical on Barth" claim for precisely this reason
(SSL_vs_classical_analysis.md §18). Best-of-15 is still printed, marked as
biased, because it bounds the most favourable reading.
"""
import collections
import glob
import json

import numpy as np

NOISE = 0.045
ORDER = [5, 10, 20, 40, 0]


def load():
    rows, seen = [], set()
    for f in sorted(glob.glob("results/phase3/transfer_*.json")):
        for r in json.load(open(f)):
            k = (r["target"], r["representation"], r["k_per_class"])
            if k not in seen:
                seen.add(k); rows.append(r)
    return rows


def paired(a, b):
    """mean and 95% CI of the per-episode difference a - b."""
    x = np.asarray(a["scores"]); y = np.asarray(b["scores"])
    n = min(len(x), len(y)); d = x[:n] - y[:n]
    se = d.std(ddof=1) / np.sqrt(n)
    return d.mean(), d.mean() - 1.96 * se, d.mean() + 1.96 * se


def main() -> None:
    rows = load()
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["target"], r["k_per_class"])][r["representation"]] = r
    targets = sorted({t for t, _ in by})

    print("=" * 96)
    print("GATE G1 -- best SSL vs classical ML, paired on identical episodes")
    print("=" * 96)
    print(f"{'target':<20}{'k':>5}{'classical':>10}{'bestObj':>9}{'delta':>8}"
          f"{'95% CI':>18}{'bias':>7}  verdict")
    verdicts = collections.Counter()
    for t in targets:
        for k in ORDER:
            d = by.get((t, k))
            if not d or "classical_binned0.02" not in d:
                continue
            c = d["classical_binned0.02"]
            ssl = {n: v for n, v in d.items()
                   if not n.startswith("classical") and "randinit" not in n}
            if not ssl:
                continue
            # honest comparator: the best OBJECTIVE by its mean over seeds
            best_obj, best_mean, best_members = None, -1.0, None
            for obj in ("masking", "jigsaw", "joint"):
                mem = [v for n, v in ssl.items() if n.startswith(obj)]
                if mem:
                    mu = float(np.mean([v["mean"] for v in mem]))
                    if mu > best_mean:
                        best_obj, best_mean, best_members = obj, mu, mem
            pooled = {"scores": list(np.mean([np.asarray(v["scores"])
                                              for v in best_members], axis=0)),
                      "mean": best_mean, "representation": f"{best_obj} (seed mean)"}
            b = pooled
            m, lo, hi = paired(b, c)
            biased_best = max(v["mean"] for v in ssl.values())
            if lo > 0 and m > NOISE:
                v = "SSL WINS"
            elif hi < 0 and -m > NOISE:
                v = "classical wins"
            else:
                v = "tie (within noise)"
            verdicts[v] += 1
            kl = "all" if k == 0 else str(k)
            print(f"{t:<20}{kl:>5}{c['mean']:>10.3f}{b['mean']:>9.3f}{m:>+8.3f}"
                  f"  [{lo:+.3f},{hi:+.3f}]{biased_best:>7.3f}  {v}")
        print()

    print("=" * 96)
    print("Does pretraining beat its own random-init backbone? (same architecture)")
    print("=" * 96)
    print(f"{'objective':<10}{'k':>5}{'pretrained':>12}{'randinit':>10}{'delta':>8}   n targets")
    for obj in ("masking", "jigsaw", "joint"):
        for k in ORDER:
            ds, ps, rs = [], [], []
            for t in targets:
                d = by.get((t, k), {})
                pre = [v for n, v in d.items() if n.startswith(obj) and "randinit" not in n]
                rnd = d.get(f"{obj}_randinit")
                if not pre or rnd is None:
                    continue
                best = max(pre, key=lambda z: z["mean"])
                ds.append(best["mean"] - rnd["mean"]); ps.append(best["mean"]); rs.append(rnd["mean"])
            if ds:
                kl = "all" if k == 0 else str(k)
                print(f"{obj:<10}{kl:>5}{np.mean(ps):>12.3f}{np.mean(rs):>10.3f}"
                      f"{np.mean(ds):>+8.3f}   {len(ds)}")
        print()

    print("=" * 96)
    print("Objective comparison -- mean over seeds and targets, per budget")
    print("=" * 96)
    print(f"{'k':>5}{'classical':>11}{'masking':>10}{'jigsaw':>9}{'joint':>8}")
    for k in ORDER:
        cells = {}
        for name in ("classical", "masking", "jigsaw", "joint"):
            vals = []
            for t in targets:
                d = by.get((t, k), {})
                if name == "classical":
                    if "classical_binned0.02" in d:
                        vals.append(d["classical_binned0.02"]["mean"])
                else:
                    s = [v["mean"] for n, v in d.items()
                         if n.startswith(name) and "randinit" not in n]
                    if s:
                        vals.append(np.mean(s))
            cells[name] = np.mean(vals) if vals else float("nan")
        kl = "all" if k == 0 else str(k)
        print(f"{kl:>5}{cells['classical']:>11.3f}{cells['masking']:>10.3f}"
              f"{cells['jigsaw']:>9.3f}{cells['joint']:>8.3f}")

    print(f"\nverdict tally: {dict(verdicts)}")


if __name__ == "__main__":
    main()
