#!/usr/bin/env python3
"""One-off verification of the peak-weighted reconstruction loss (experiment #7).

Two things are checked, because two things could silently be wrong:

  1. The WEIGHT FUNCTION does what its docstring claims -- keeps exactly the
     right number of patches, keeps the tall ones and drops flat baseline, is
     per-spectrum (not per-batch), and is a no-op at fraction 1.0.
  2. The PORT is faithful. trainer_revised.top_peak_patch_weights was copied
     from train_joint_ssl.top_peak_bin_weights; the two must agree elementwise
     on identical input, or the masking and joint families would be weighting
     reconstruction differently while claiming to share an objective.

Then the loss itself is checked end to end: peak weighting must change the loss
value when enabled and leave it bit-identical when disabled, since every
committed masking run used the unweighted path.

Run:  python code/tests/verify_top_peak_loss.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/training"))

from trainer_revised import (NMRSpectrumDataset, compute_loss,  # noqa: E402
                            top_peak_patch_weights)
from train_joint_ssl import top_peak_bin_weights  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def synthetic_patches(batch=4, n_patches=32, patch_size=16, n_peaks=8, seed=0):
    """Flat noise everywhere, tall Gaussian-ish bumps in n_peaks known patches."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(batch, n_patches, patch_size, generator=g) * 0.01
    peak_idx = torch.stack([torch.randperm(n_patches, generator=g)[:n_peaks]
                            for _ in range(batch)])
    for b in range(batch):
        for p in peak_idx[b]:
            x[b, p] += torch.linspace(0.0, 1.0, patch_size) * (0.5 + 0.5 * torch.rand(1, generator=g))
    return x, peak_idx


def main():
    print("\n=== 1. weight function semantics ===")
    x, peak_idx = synthetic_patches()
    batch, n_patches, _ = x.shape

    w = top_peak_patch_weights(x, 0.25)
    k_expected = max(1, int(round(0.25 * n_patches)))
    check("shape is (B, n_patches)", tuple(w.shape) == (batch, n_patches), str(tuple(w.shape)))
    check("values are 0/1 only", bool(((w == 0) | (w == 1)).all()))
    counts = w.sum(dim=1)
    check(f"keeps exactly k={k_expected} patches per spectrum",
          bool((counts == k_expected).all()), f"counts={counts.tolist()}")

    # The 8 planted peaks are the only non-flat patches, so the top 8 by
    # magnitude must be exactly those -- a strong test that ranking works.
    w8 = top_peak_patch_weights(x, 8 / n_patches)
    kept = [set(torch.nonzero(w8[b]).flatten().tolist()) for b in range(batch)]
    planted = [set(peak_idx[b].tolist()) for b in range(batch)]
    check("top-8 selection recovers exactly the planted peak patches",
          kept == planted, f"first row kept={sorted(kept[0])} planted={sorted(planted[0])}")

    check("fraction=1.0 returns all ones (no-op)",
          bool((top_peak_patch_weights(x, 1.0) == 1).all()))
    check("tiny fraction still keeps at least one patch",
          int(top_peak_patch_weights(x, 1e-6).sum(dim=1).min()) == 1)

    # Per-spectrum, not per-batch: scaling one row must not change any other
    # row's selection. A global threshold would fail this.
    x2 = x.clone()
    x2[0] *= 1000.0
    w2 = top_peak_patch_weights(x2, 0.25)
    check("thresholding is per-spectrum (scaling row 0 leaves rows 1..N unchanged)",
          bool(torch.equal(w2[1:], w[1:])))
    check("scaling a row does not change its own selection (rank-invariance)",
          bool(torch.equal(w2[0], w[0])))

    print("\n=== 2. port matches train_joint_ssl.top_peak_bin_weights ===")
    for frac in (0.05, 0.1, 0.25, 0.5, 0.9, 1.0):
        a = top_peak_patch_weights(x, frac)
        b = top_peak_bin_weights(x, frac)
        check(f"identical at fraction={frac}", bool(torch.equal(a, b)))
    # And on real preprocessed spectra, where the magnitude distribution is
    # heavily skewed and ties at the threshold are plausible.
    real = ROOT / "data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v4.npy"
    if not real.exists():
        real = ROOT / "data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v3.npy"
    if real.exists():
        arr = np.load(real, mmap_mode="r")[:16]
        t = torch.from_numpy(np.asarray(arr, dtype=np.float32))
        t = t.reshape(t.shape[0], t.shape[1] // 1024, 1024)
        for frac in (0.1, 0.25, 0.5):
            check(f"identical on real spectra at fraction={frac}",
                  bool(torch.equal(top_peak_patch_weights(t, frac),
                                   top_peak_bin_weights(t, frac))))
        # Sanity on real data: kept patches must be ENRICHED in signal relative
        # to picking the same number at random (share > fraction), and the
        # enrichment must fall as the fraction grows, since patches are added in
        # decreasing order of magnitude.
        #
        # The measured curve is reported rather than asserted against a guessed
        # threshold, because it is the number that decides how aggressive the
        # arm should be. At patch_size=1024 each patch spans 1024 points, so
        # nearly every patch contains SOME signal and the concentration is
        # milder than the "mostly flat baseline" framing suggests: the top 25%
        # of patches hold ~60% of total intensity (2.4x enrichment), meaning a
        # fraction of 0.25 discards ~40% of the signal. See the note in
        # docs/SSL_vs_classical_analysis.md experiment #7.
        mag = t.abs().sum(dim=2)
        shares = {}
        for frac in (0.05, 0.1, 0.25, 0.5, 0.75):
            wf = top_peak_patch_weights(t, frac)
            shares[frac] = float((mag * wf).sum() / mag.sum())
        print("         intensity share of kept patches: "
              + ", ".join(f"{f:.2f}->{s:.3f}" for f, s in shares.items()))
        check("kept patches are enriched in signal at every fraction",
              all(s > f for f, s in shares.items()),
              ", ".join(f"{f:.2f}: {s / f:.2f}x" for f, s in shares.items()))
        enrich = [s / f for f, s in shares.items()]
        check("enrichment decreases monotonically with fraction",
              all(a >= b - 1e-9 for a, b in zip(enrich, enrich[1:])),
              " > ".join(f"{e:.2f}" for e in enrich))
    else:
        print(f"  [SKIP] real-spectra checks -- {real.name} not found")

    print("\n=== 3. compute_loss integration ===")

    class TinyModel(torch.nn.Module):
        """Returns a fixed wrong reconstruction; only the loss math is under test."""

        patch_size = 16

        def __init__(self, n_points):
            super().__init__()
            self.out = torch.nn.Parameter(torch.zeros(n_points), requires_grad=True)

        def forward(self, masked, mask=None):
            return self.out.unsqueeze(0).expand(masked.shape[0], -1), None

    n_patches, patch_size = 32, 16
    n_points = n_patches * patch_size
    flat, _ = synthetic_patches(batch=4, n_patches=n_patches, patch_size=patch_size, seed=7)
    original = flat.reshape(4, n_points)
    mask = torch.zeros(4, n_patches, dtype=torch.bool)
    mask[:, :8] = True  # first 8 patches masked in every row
    batch_dict = {"original": original, "masked": original.clone(), "mask": mask}
    model = TinyModel(n_points)

    uniform = compute_loss(model, batch_dict, "cpu", peak_top_fraction=1.0)
    legacy = compute_loss(model, batch_dict, "cpu")  # default must be uniform
    check("default peak_top_fraction reproduces the uniform loss bit-exactly",
          all(float(a) == float(b) for a, b in zip(uniform, legacy)),
          f"{float(uniform[0]):.8e} vs {float(legacy[0]):.8e}")

    weighted = compute_loss(model, batch_dict, "cpu", peak_top_fraction=0.25)
    check("peak weighting changes the loss value",
          float(weighted[0]) != float(uniform[0]),
          f"uniform={float(uniform[0]):.6e} weighted={float(weighted[0]):.6e}")
    check("weighted loss is LARGER (flat easy patches excluded)",
          float(weighted[0]) > float(uniform[0]))
    check("weighted loss is finite and non-negative",
          np.isfinite(float(weighted[0])) and float(weighted[0]) >= 0)
    check("gradient flows through the weighted loss",
          torch.autograd.grad(weighted[0], model.out, retain_graph=True)[0].abs().sum().item() > 0)

    print("\n=== 4. block masking ===")
    dummy = np.random.default_rng(0).random((4, n_points)).astype(np.float32)
    ds = NMRSpectrumDataset(dummy, mask_ratio_min=0.20, mask_ratio_max=0.60,
                            patch_size=patch_size, mask_strategy='block',
                            mask_block_patches=4, normalize_input=False,
                            correct_post_mask_baseline=False)
    runs_seen, ratios_ok, exact_count_ok = [], True, True
    for _ in range(200):
        ratio = None
        m = ds.create_mask(n_patches)
        n_masked = int(m.sum())
        ratios_ok &= (max(1, int(n_patches * 0.20)) <= n_masked <= int(n_patches * 0.60) + 1)
        # Longest contiguous run of masked patches.
        run = best = 0
        for v in m.tolist():
            run = run + 1 if v else 0
            best = max(best, run)
        runs_seen.append(best)
    check("block masking respects the mask-ratio range", ratios_ok,
          f"observed n_masked range enforced against [{max(1, int(n_patches*0.20))}, {int(n_patches*0.60)+1}]")
    check("block masking produces contiguous runs of >= span length",
          min(runs_seen) >= 4, f"min longest-run over 200 draws = {min(runs_seen)}")

    sparse = NMRSpectrumDataset(dummy, patch_size=patch_size, mask_strategy='sparse_random',
                                normalize_input=False, correct_post_mask_baseline=False)
    sparse_runs = []
    for _ in range(200):
        m = sparse.create_mask(n_patches)
        run = best = 0
        for v in m.tolist():
            run = run + 1 if v else 0
            best = max(best, run)
        sparse_runs.append(best)
    check("block runs are longer than sparse_random runs",
          float(np.mean(runs_seen)) > float(np.mean(sparse_runs)),
          f"block mean={np.mean(runs_seen):.1f} vs sparse mean={np.mean(sparse_runs):.1f}")

    span_full = NMRSpectrumDataset(dummy, patch_size=patch_size, mask_strategy='block',
                                  mask_block_patches=n_patches * 4, normalize_input=False,
                                  correct_post_mask_baseline=False)
    m = span_full.create_mask(n_patches)
    check("span longer than the spectrum is clamped, not an error",
          int(m.sum()) >= 1)

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
