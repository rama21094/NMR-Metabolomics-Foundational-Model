#!/usr/bin/env python3
"""Phase 6 -- does capacity help? Evaluated exactly as Phase 3 evaluated G1.

Two-sided by design: the sweep spans 1.8M to 23.5M parameters, below and above
the 4.0M baseline, because the G1 fine-tuning result predicts the baseline is
already too large for 40-142 sample cohorts. A one-sided sweep could not tell
"capacity is not the lever" from "we did not add enough".

Checkpoints are identified by the architecture recorded INSIDE them, not by
filename: the sweep writes the same corpus_train_<ts>_..._seed<N>_best.pth
pattern as the Phase 2 baseline, so 17 files match one glob and only 5 are the
baseline.
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in ("code/training", "code/evaluation", "code/analysis"):
    sys.path.insert(0, str(ROOT / p))
from phase3_transfer import target_defs, load_cohort, binned, episodes, probe  # noqa: E402
import zlib  # noqa: E402


def ckpts():
    out = []
    for p in sorted((ROOT / "models/masked_ssl").glob("corpus_train_*_seed*_best.pth")):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        hp = ck.get("hyperparameters", {})
        d, l = hp.get("d_model"), hp.get("num_layers")
        if d is None:
            continue
        n = sum(v.numel() for v in ck["model_state_dict"].values()) / 1e6
        out.append(((d, l), round(n, 2), int(p.name.split("_seed")[-1].split("_")[0]), p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--budgets", type=int, nargs="+", default=[10, 0])
    ap.add_argument("--out", default="results/phase6/capacity_eval.json")
    args = ap.parse_args()
    (ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    from linear_probe_frozen_embeddings import embed_masking

    C = ckpts()
    arches = sorted({a for a, _, _, _ in C})
    print(f"{len(C)} checkpoints, {len(arches)} architectures: " +
          ", ".join(f"d{d}/L{l}" for d, l in arches) + "\n", flush=True)

    rows = []
    for tid, cohort, labfn in target_defs():
        M, ppm, y = load_cohort(cohort, labfn)
        base = binned(M, ppm, 0.02)
        embs = {}
        for arch, nparam, seed, p in C:
            try:
                embs[(arch, nparam, seed)] = np.asarray(
                    embed_masking(str(p), M, args.device), dtype=np.float32)
            except Exception as e:
                print(f"  embed fail d{arch[0]}/L{arch[1]} s{seed}: {type(e).__name__}: {e}",
                      flush=True)
        for k in args.budgets:
            eps = episodes(y, k, args.episodes, seed=zlib.crc32(tid.encode()) % 2**31)
            if not eps:
                continue
            cls = float(np.mean([probe(base, y, s, q) for s, q in eps]))
            agg = defaultdict(list)
            for (arch, nparam, seed), F in embs.items():
                sc = float(np.mean([probe(F, y, s, q) for s, q in eps]))
                rows.append({"target": tid, "d_model": arch[0], "num_layers": arch[1],
                             "params_m": nparam, "seed": seed, "k_per_class": k,
                             "mean": sc, "classical": cls})
                agg[(nparam, arch)].append(sc)
            kl = "all" if k == 0 else str(k)
            print(f"{tid:<20} k={kl:<4} classical {cls:.3f} | " +
                  "  ".join(f"{n}M:{np.mean(v):.3f}" for (n, a), v in sorted(agg.items())),
                  flush=True)
            json.dump(rows, open(ROOT / args.out, "w"), indent=1)
    json.dump(rows, open(ROOT / args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
