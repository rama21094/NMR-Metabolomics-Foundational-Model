#!/usr/bin/env python3
"""How many effective studies would pretraining need to match standard ML?

Checks the "~20-25 studies" figure from the 2026-09-25 PI notes, which was never
computed. For each of the 45 scaling checkpoints: gap = (pretrained - standard
ML) averaged over the 6 targets, at one label budget. Fit gap against effective
studies (linear, and linear in log), solve for gap = 0, and bootstrap the 45
checkpoints for a 95% interval on that crossing. A crossing far outside the
observed range (about 2.5-8 effective studies) is an extrapolation either way.
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/analysis"))
from phase4_diversity_regression import load_eff  # noqa: E402


def crossing(x, g, log):
    X = np.log(x) if log else x
    b, a = np.polyfit(X, g, 1)
    if b <= 0:
        return np.inf, b
    c = -a / b
    return (np.exp(c) if log else c), b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default="results/phase4/scaling_eval_45.json")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--out", default="results/phase4/parity_extrapolation.json")
    a = ap.parse_args()
    eff = load_eff()
    agg = defaultdict(list)
    for r in json.load(open(ROOT / a.evals)):
        agg[(r["k_per_class"], r["subset"], r["seed"])].append(r["mean"] - r["classical"])
    out = {}
    rng = np.random.default_rng(0)
    for kb in sorted({k for k, _, _ in agg}):
        pts = []
        for (k, sub, seed), v in agg.items():
            key = f"{sub}_seed{seed}" if f"{sub}_seed{seed}" in eff else sub
            if k == kb and key in eff and eff[key][0] is not None:
                pts.append((eff[key][0], np.mean(v)))
        x, g = np.array(pts).T
        res = {"n": len(x), "eff_range": [float(x.min()), float(x.max())],
               "mean_gap": float(g.mean())}
        for log in (False, True):
            c, b = crossing(x, g, log)
            bs = []
            for _ in range(a.boot):
                i = rng.integers(0, len(x), len(x))
                bs.append(crossing(x[i], g[i], log)[0])
            bs = np.array(bs)
            fin = np.where(np.isfinite(bs), bs, np.inf)   # never-reach counts as unbounded
            hi = np.percentile(np.minimum(fin, 1e12), 97.5)
            res["log" if log else "linear"] = {
                "slope": float(b), "crossing": float(c),
                "ci95": [float(np.percentile(np.minimum(fin, 1e12), 2.5)),
                         None if hi >= 1e12 or hi > 1e4 else float(hi)],
                "share_no_crossing": float(np.mean(~np.isfinite(bs)))}
        out[f"k={kb}"] = res
        print(f"k={kb}  n={res['n']}  eff {x.min():.1f}-{x.max():.1f}  mean gap {g.mean():+.3f}")
        for m in ("linear", "log"):
            r = res[m]
            print(f"   {m:<6} slope {r['slope']:+.4f}  parity at {r['crossing']:.3g} eff. studies"
                  f"  95% CI [{r['ci95'][0]:.1f}, "
                  f"{'unbounded' if r['ci95'][1] is None else format(r['ci95'][1], '.1f')}]"
                  f"  ({r['share_no_crossing']:.0%} of resamples never reach parity)")
    json.dump(out, open(ROOT / a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
