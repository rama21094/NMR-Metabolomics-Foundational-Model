#!/usr/bin/env python3
"""Phase 3.1, completed -- fine-tuning, not just the frozen linear probe.

Gate G1 was decided on frozen embeddings alone. That makes its verdict a claim
about REPRESENTATIONS, not about what the architecture can do when adapted, and
the research plan (Phase 3.1) always specified four transfer modes in increasing
adaptation. Three were never run. This runs the two that matter:

    head        linear head on frozen features        (= the Phase 3 probe,
                                                        re-run here as a control
                                                        through the same code
                                                        path, so a difference
                                                        cannot be an artefact of
                                                        a different pipeline)
    last_block  unfreeze the final transformer block + head
    full        unfreeze everything

IDENTICAL READOUT ACROSS ARMS. Every arm is scored by the SAME frozen probe used
in Phase 3 -- StandardScaler + balanced logistic regression on pooled embeddings.
The arms differ only in whether, and how deeply, the ENCODER was adapted on the
support set first. A gradient-trained head was tried and abandoned: it made the
"head" arm a different estimator from the Phase 3 probe rather than the same one,
so a fine-tuning gain could not be separated from an optimiser difference. With
the shared readout, `head` is exactly the Phase 3 probe, by construction.

PAIRING. Every mode, every seed and the classical baseline are scored on THE SAME
episodes, drawn by the same stable crc32 seeding as phase3_transfer. Comparisons
are therefore paired and the episode noise -- which is large, sd 0.05-0.09 -- is
common to all arms and cancels in the differences.

THE EXPECTED FAILURE. Fine-tuning a 4-layer transformer on 5-40 labelled spectra
should overfit badly. That is not a reason to skip the experiment, it is the
result the plan needs on record: if adaptation cannot be made to pay even with
early stopping on a held-out slice of the support set, G1's frozen-probe verdict
stands for the architecture as a whole rather than only for its representations.
To give fine-tuning its best honest chance:

  * the support set is split 75/25 into train/val, and the epoch with the best
    val balanced accuracy is restored (so the mode cannot be beaten merely by
    training too long);
  * the head-only arm uses the same optimiser, schedule and early stopping, so
    the arms differ ONLY in which parameters carry gradient;
  * class weights are balanced, matching the probe's class_weight="balanced";
  * a random-init control at each mode separates "adaptation helps" from
    "pretraining helps".

At k=5 with 2 classes the support set is 10 spectra, so the val slice is 2-3.
Early stopping on that is itself noisy; the val-restore is a guard against
runaway overfitting, not a reliable model selector, and is reported as such.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[2]
for p in ("code/training", "code/evaluation", "code/analysis"):
    sys.path.insert(0, str(ROOT / p))
from phase3_transfer import target_defs, load_cohort, binned, episodes, probe  # noqa: E402

MODES = ("head", "last_block", "full")


class Classifier(nn.Module):
    """Pretrained encoder + mean-pool + linear head."""

    def __init__(self, ckpt, n_class, spectrum_length, random_init=False):
        super().__init__()
        from trainer_revised import NMRMaskedAutoencoder
        from barth_all_models_loocv import infer_mae_config
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = ck["model_state_dict"]
        nh = ck.get("hyperparameters", {}).get("nhead") or 8
        cfg = infer_mae_config(state, int(nh), 0.0)
        self.enc = NMRMaskedAutoencoder(spectrum_length=spectrum_length, **cfg)
        if not random_init:
            self.enc.load_state_dict(state, strict=True)
        # the frozen probe standardises features before logistic regression;
        # without an equivalent the head arm optimises far slower and would be a
        # crippled control for the fine-tuned arms to beat
        self.norm = nn.LayerNorm(cfg["d_model"])
        self.head = nn.Linear(cfg["d_model"], n_class)

    def forward(self, x):
        _, enc = self.enc(x, mask=None)
        return self.head(self.norm(enc.mean(dim=1)))

    def set_mode(self, mode):
        """Freeze everything, then unfreeze exactly what `mode` names."""
        for p in self.parameters():
            p.requires_grad = False
        for p in list(self.head.parameters()) + list(self.norm.parameters()):
            p.requires_grad = True
        if mode == "full":
            for p in self.enc.parameters():
                p.requires_grad = True
        elif mode == "last_block":
            # NMRMaskedAutoencoder.encoder.transformer.layers
            blocks = None
            for mod in self.enc.modules():
                if isinstance(mod, nn.TransformerEncoder):
                    blocks = mod.layers
            if blocks is None:
                raise RuntimeError("could not locate transformer blocks to unfreeze")
            for p in blocks[-1].parameters():
                p.requires_grad = True
        elif mode != "head":
            raise ValueError(mode)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def fit_episode(ckpt, M, y, sup, qry, mode, device, epochs, lr, random_init, seed):
    """Adapt the encoder on the support set, then read out with the Phase 3 probe.

    `head` performs no encoder update at all, so it reproduces the Phase 3 frozen
    probe exactly rather than approximating it.
    """
    torch.manual_seed(seed)
    classes = np.unique(y)
    cmap = {c: i for i, c in enumerate(classes)}
    model = Classifier(ckpt, len(classes), M.shape[1], random_init).to(device)
    ntrain = model.set_mode(mode)

    if mode != "head":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(sup))
        ntr = max(2, int(round(0.75 * len(sup))))
        tr, va = sup[perm[:ntr]], sup[perm[ntr:]]
        if len(va) == 0 or len(np.unique(y[tr])) < 2:
            tr, va = sup, sup

        Xtr = torch.from_numpy(np.asarray(M[tr], dtype=np.float32)).to(device)
        ytr = torch.tensor([cmap[c] for c in y[tr]], device=device)
        Xva = torch.from_numpy(np.asarray(M[va], dtype=np.float32)).to(device)
        yva = [cmap[c] for c in y[va]]

        cnt = np.array([max(1, (y[tr] == c).sum()) for c in classes], dtype=np.float32)
        w = torch.tensor(len(tr) / (len(classes) * cnt), device=device)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=lr, weight_decay=0.01)
        lossf = nn.CrossEntropyLoss(weight=w)

        best, best_state = -1.0, copy.deepcopy(model.state_dict())
        for _ in range(epochs):
            model.train()
            opt.zero_grad()
            lossf(model(Xtr), ytr).backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            model.eval()
            with torch.no_grad():
                pv = model(Xva).argmax(1).cpu().numpy()
            sc = balanced_accuracy_score(yva, pv)
            if sc > best:
                best, best_state = sc, copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)

    # shared readout: pooled embeddings -> the Phase 3 probe
    model.eval()
    F = []
    with torch.no_grad():
        for i in range(0, len(M), 8):
            x = torch.from_numpy(np.asarray(M[i:i + 8], dtype=np.float32)).to(device)
            _, enc = model.enc(x, mask=None)
            F.append(enc.mean(dim=1).cpu().numpy())
    F = np.vstack(F).astype(np.float32)
    del model
    torch.cuda.empty_cache()
    return probe(F, y, sup, qry), ntrain


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--targets", nargs="+")
    ap.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 20, 0])
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--random-init", action="store_true",
                    help="also run every mode from an untrained backbone")
    ap.add_argument("--out", default="results/phase5/finetune.json")
    args = ap.parse_args()
    (ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)

    cks = sorted((ROOT / "models/masked_ssl").glob("corpus_train_*_seed*_best.pth"))
    ck_by_seed = {int(p.name.split("_seed")[-1].split("_")[0]): p for p in cks}
    use = [(s, ck_by_seed[s]) for s in args.seeds if s in ck_by_seed]
    if not use:
        raise SystemExit(f"no masking checkpoints for seeds {args.seeds}")
    print(f"{len(use)} checkpoints, modes {args.modes}, "
          f"budgets {args.budgets}, {args.episodes} episodes\n", flush=True)

    rows = []
    for tid, cohort, labfn in target_defs():
        if args.targets and tid not in args.targets:
            continue
        M, ppm, y = load_cohort(cohort, labfn)
        base = binned(M, ppm, 0.02)
        for k in args.budgets:
            eps = episodes(y, k, args.episodes, seed=zlib.crc32(tid.encode()) % 2**31)
            if not eps:
                continue
            cls = float(np.mean([probe(base, y, s, q) for s, q in eps]))
            agg = defaultdict(list)
            for mode in args.modes:
                for seed, ck in use:
                    res = [fit_episode(ck, M, y, s, q, mode, args.device,
                                       args.epochs, args.lr, False, seed)
                           for s, q in eps]
                    sc = [r[0] for r in res]; ntrain = res[0][1]
                    agg[mode].append(float(np.mean(sc)))
                    rows.append({"target": tid, "mode": mode, "seed": seed,
                                 "k_per_class": k, "mean": float(np.mean(sc)),
                                 "sd": float(np.std(sc)), "classical": cls,
                                 "random_init": False, "n_episodes": len(eps),
                                 "trainable_params": int(ntrain)})
                if args.random_init:
                    sc = [fit_episode(ck, M, y, s, q, mode, args.device,
                                      args.epochs, args.lr, True, use[0][0])[0]
                          for s, q in eps]
                    rows.append({"target": tid, "mode": mode, "seed": use[0][0],
                                 "k_per_class": k, "mean": float(np.mean(sc)),
                                 "sd": float(np.std(sc)), "classical": cls,
                                 "random_init": True, "n_episodes": len(eps)})
                    agg[mode + "/rand"].append(float(np.mean(sc)))
            kl = "all" if k == 0 else str(k)
            print(f"{tid:<20} k={kl:<4} classical {cls:.3f} | " +
                  "  ".join(f"{m}:{np.mean(v):.3f}" for m, v in sorted(agg.items())),
                  flush=True)
            json.dump(rows, open(ROOT / args.out, "w"), indent=1)
    json.dump(rows, open(ROOT / args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
