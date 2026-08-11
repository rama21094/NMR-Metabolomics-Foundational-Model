# Why classical ML outperforms the SSL backbones — analysis and experiment queue

**Status:** updated 2026-08-10. Experiments **#1–#8, #13, #14, #15 done**; #9–#12
queued (#9 void). **#6 is the decisive one and it is negative:** masked SSL does not beat
classical LogReg at any label budget on any of the five targets, and pooled at the smallest
budget the difference is +0.0012 ± 0.0162 (p=0.74) — see §6.

Experiment #4 **refuted** the patch-resolution hypothesis of §5 — see §5b, which is
the correction of record. It also found the actual win (pooling), which remains the only
robust positive result in this document measured on masking (§5c is a paired
within-checkpoint comparison, unaffected by everything below) — §14 now shows the same win,
smaller, on jigsaw/joint. §5d's "backbone axis is exhausted" claim is **withdrawn** (§7b) —
it compared v4 arms against a v3 reference. §7/§7b found both a new objective change (block
masking, peak weighting) fails and, more importantly, that the pretraining **corpus
version** (v3 vs v4) *appeared* worth 0.069 held-out — which looked like the largest effect in
the document, on an assumed noise floor of 0.020 (mean) / 0.035 (per-target). **Both figures
were wrong**: the measured single-run sd is **0.045** (§15). §8 and §13 both tried to localize that 0.069 (to row content, then to
corpus size) and both failed — and **§15 then showed why: there was no effect to localize.**
With n=5 seeds per corpus the gap goes from +0.069 to −0.014 (0.6 se), with v4 marginally ahead.
§5f is retracted. The measured single-run sd is **0.045**, more than double the 0.020 that
was assumed, and against it only §6b survives among single-run claims. What still stands are
the **paired within-checkpoint** results — §4b (head fix), §5c (pooling), §14 (jigsaw/joint
pooling) — plus §6b (pretraining vs a random-init control).

**§6 (2026-08-10) then tested the premise itself** — few-shot transfer on all five targets,
paired on shared episodes — and found no advantage at any label budget, with the deficit
*widening* as labels accumulate. The remaining untested explanation is corpus scale (9,670
spectra); §13 only probed size downward.

**Purpose of this document.** The v4 benchmark showed logistic regression beating all
three SSL families on all five dataset/label targets. This records *why*, with the
diagnostics that established it, so each follow-up experiment can be run one at a time
without re-deriving the reasoning. Every number here is reproducible from the scripts
named in each section.

---

## 1. TL;DR

1. The headline gap is **not one problem, it is two**, in different proportions per
   dataset: a **representation ceiling** caused by the backbone's patch size, and a
   **head-fitting deficit** in how the classifier is trained.
2. The **masking** family's head is systematically underfit. Replacing it with a
   properly-regularized linear probe on the *identical frozen features* gains
   **+0.12 balanced accuracy on average, on all 5 of 5 targets**. This is free.
3. The backbone tokenizes 131,072 points into **128 patches**, so it cannot represent
   spectral detail finer than 128 positions. LogReg on 1024 bins exploits 8× finer
   resolution, and the bin sweep shows that resolution is exactly what buys accuracy.
4. The classical results are **real, not CV artifacts** — label-permutation nulls give
   p ≤ 0.02 everywhere. They are also **not a dilution confound** — a single global
   intensity scalar explains far less.
5. **A correction to an earlier claim:** the existing Xavier ablation does *not* test
   "pretrained vs random backbone." Read correctly, pretraining **helps**. See §6.

---

## 2. What was compared, and how

| | classical track | SSL track |
|---|---|---|
| features | `binned_abs_area`: absolute integrated area in *N* equal bins (N=1024 default, 256 for `mtbls326_loocv.py`) | frozen/fine-tuned backbone embedding |
| classifier | `StandardScaler` → `LogisticRegression(C=1, max_iter=5000, class_weight="balanced")`, L-BFGS to convergence | pooled embedding → `LayerNorm` → `Dropout` → `Linear`, Adam ~50 epochs, early stopping, class-balanced CE |

Both classifiers are **linear**. That is what makes the swap in §4 a clean experiment.

Pooling per family (each replicated exactly in the probe scripts):

| family | pooling | dim |
|---|---|---|
| masking | `encoded.mean(dim=1)` | 128 |
| jigsaw | per-bin-size transformer pass → mean-pool → concat 4 bin sizes | 768 |
| joint | `encode_spectrum(bin_sizes, include_masked_task=True)` | 960 |

Protocols match the committed runs exactly: LOOCV for Barth (n=37, "Pool" QC rows
excluded) and MTBLS326 (n=42); `StratifiedKFold(10, shuffle=True, random_state=42)` for
BrC-T2D (n=78) and MTBLS563 (n=113, 3-class, "unknown" excluded).

> **Validation gate.** Before drawing any conclusion, the probe harness was checked to
> reproduce the official `summary.csv` LogReg balanced accuracy to 6 decimal places on
> all four datasets: BrC-T2D cancer 0.936842, diabetes 0.828877, MTBLS563 0.720785,
> MTBLS326 1.000000, Barth 0.704969. Without this, none of the comparisons below would
> be interpretable.

---

## 3. Headline result (v4)

Best mode per family, balanced accuracy. Figures:
`results/plots/all_datasets_summary_v4/fig1_balanced_accuracy.png`,
`fig2_roc_auc.png`, `fig3_finetune_depth.png`, `fig4_heatmap_all_models.png`.

| dataset | classical (LR) | masked | jigsaw | joint | gap (LR − best SSL) |
|---|---|---|---|---|---|
| Barth (n=37, LOOCV) | **0.705** | 0.691 | 0.677 | 0.649 | 0.014 |
| MTBLS326 (n=42, LOOCV) | **1.000** | 0.981 | 0.874 | 0.930 | 0.019 |
| MTBLS563 (n=113, 3-class) | **0.721** | 0.558 | 0.550 | 0.500 | 0.163 |
| BrC-T2D cancer (n=78) | **0.937** | 0.796 | 0.782 | 0.757 | 0.141 |
| BrC-T2D diabetes (n=78) | **0.829** | 0.653 | 0.620 | 0.624 | 0.176 |

Consistent SSL ordering: **masked > jigsaw ≈ joint**.

**Do not over-read Barth and MTBLS326.** Their gaps (0.014, 0.019) are half a sample and
one sample respectively, and LOOCV yields no fold variance, so there are no error bars.
Treat those two as ties.

---

## 4. Gap decomposition — representation vs. head

Script: `code/analysis/probe_logreg_advantage.py` →
`results/analysis/logreg_advantage_probe/probe_results.csv`
Figure: `results/plots/all_datasets_summary_v4/fig6_logreg_advantage_probe.png`

Running the **same** LogReg on the frozen masked-SSL embedding isolates the two causes:

| dataset | SSL head (reported) | LogReg on *same* embedding | LogReg @1024 bins | head deficit | representation ceiling |
|---|---|---|---|---|---|
| BrC-T2D cancer | 0.796 | 0.833 | 0.937 | 0.037 | **0.104** |
| BrC-T2D diabetes | 0.653 | **0.810** | 0.829 | **0.157** | 0.019 |
| MTBLS563 | 0.558 | 0.607 | 0.721 | 0.049 | **0.114** |
| Barth | 0.691 | **0.770** | 0.705 | **0.079** | −0.065 (embedding *beats* bins) |
| MTBLS326 | 0.981 | 0.944 | 1.000 | −0.037 | 0.056 |

Reading:
- **Diabetes is a head problem** (89% of the gap). The embedding already supports 0.810.
- **Cancer and MTBLS563 are representation problems** (~70–75% of the gap survives the
  better classifier).
- **Barth's SSL representation is genuinely better than binned features** (0.770 vs
  0.705). The head was throwing that away.

### 4b. The pure classifier test (experiment #2, DONE)

Script: `code/analysis/linear_probe_frozen_embeddings.py` →
`results/analysis/linear_probe_frozen/linear_probe_results.csv`
Figure: `results/plots/all_datasets_summary_v4/fig7_linear_probe_vs_head.png`

LogReg vs the trained MLP head on **identical frozen features** — same backbone, same
pooling, same folds. Any difference is purely how the linear map is fitted.

Δ balanced accuracy (LogReg probe − frozen head):

| dataset | masking | jigsaw | joint |
|---|---|---|---|
| Barth | **+0.115** | +0.093 | +0.022 |
| BrC-T2D cancer | **+0.103** | 0.000 | +0.013 |
| BrC-T2D diabetes | **+0.156** | −0.026 | +0.019 |
| MTBLS326 | **+0.148** | 0.000 | 0.000 |
| MTBLS563 | **+0.077** | −0.024 | −0.006 |
| **mean** | **+0.120** | +0.009 | +0.009 |

**Conclusion: the masking head is underfit on all 5 of 5 targets**, by +0.077 to +0.156.
Jigsaw and joint heads are fine.

Compared against each family's *best fine-tuned* mode (the practical question):

| family | mean Δ | interpretation |
|---|---|---|
| masking | **+0.057** | fine-tuning does *not* close the gap — switch the head |
| jigsaw | −0.022 | fine-tuning beats a frozen probe — keep fine-tuning |
| joint | −0.019 | keep fine-tuning |

**Actionable now:** for the masking family, replace the Adam-trained MLP head with a
converged, L2-regularized linear probe (or tune weight decay / epochs / LR schedule to
match). Expect ≈ +0.06 vs the current best configuration and ≈ +0.12 vs frozen.

---

## 5. Mechanism of the representation ceiling

`patch_size=1024` → 131072/1024 = **128 patches**. The encoder cannot represent detail
finer than 128 spectral positions. Compare LogReg at exactly that resolution:

| | LogReg @128 bins | SSL embedding | LogReg @1024 bins |
|---|---|---|---|
| BrC-T2D cancer | 0.885 | 0.859 (flatten) | 0.937 |
| BrC-T2D diabetes | 0.746 | 0.810 | 0.829 |
| MTBLS563 | between 64-bin 0.592 and 256-bin 0.670 | 0.607–0.621 | 0.721 |

The embedding lands at its own patch resolution. And resolution is what buys accuracy —
the bin sweep (BrC-T2D cancer):

| bins | 16 | 64 | 128 | 256 | 1024 | 4096 |
|---|---|---|---|---|---|---|
| bal. acc. | 0.836 | 0.859 | 0.885 | 0.898 | **0.937** | 0.937 |

Monotonic to 1024, then plateau. **The 8× downsampling in the tokenizer is discarding
discriminative signal.** Encouragingly, on diabetes the embedding (0.810) *beats* raw
binning at the same resolution (0.746), so the learned representation does add value —
it is just starved of resolution.

Mean-pooling is a *minor* factor by comparison: flatten (position-preserving) vs
mean-pool changed cancer +0.026 and MTBLS563 +0.014, but *hurt* diabetes −0.018.

### 5b. Experiment #4 result — the resolution hypothesis is REFUTED

Two backbones were pretrained on the v4 corpus with everything else held fixed
(`patch_size` 256 and 128 vs the 1024 baseline) and compared through the frozen linear
probe. Script `code/analysis/compare_patch_sizes.py`, figure
`fig9_patch_size_and_pooling.png`, data
`results/analysis/patch_size_comparison/patch_size_results.csv`.

All arms are read at `nhead=4`, the value training actually used, so this is a clean
resolution comparison (see the §6c note on the nhead recording bug).

Balanced accuracy, flatten pooling:

| dataset | patch 1024 (128 tok) | patch 256 (512 tok) | patch 128 (1024 tok) |
|---|---|---|---|
| Barth | **0.806** | 0.598 | 0.655 |
| MTBLS326 | **1.000** | 0.907 | 0.911 |
| BrC-T2D cancer | **0.859** | 0.832 | 0.768 |
| BrC-T2D diabetes | 0.780 | **0.783** | 0.738 |
| MTBLS563 | **0.618** | 0.581 | 0.607 |

Mean Δ vs patch 1024: **patch 256 −0.072, patch 128 −0.077 — 0 of 5 wins.** With
mean-pooling: −0.040 and −0.060. **Finer patches consistently made things worse.**

(ps128 numbers are from its final epoch-1813 checkpoint. An earlier reading used the
epoch-1632 checkpoint, before training finished; the final one scored 0.000–0.026 *lower*
downstream on 4 of 5 targets despite a 2.4% better reconstruction loss — a small extra
illustration of the same reconstruction-vs-utility disconnect described below. The
conclusion is unchanged and marginally strengthened.)

Why the prediction failed. Best validation reconstruction loss *fell* as patches shrank:
9.26e-5 (ps1024) → 5.56e-5 (ps256) → 4.36e-5 (ps128). A masked 128-point patch is largely
interpolable from its immediate neighbours, so shrinking the patch made the pretext task
**easier**, not more informative — the model can solve it by local smoothing without
learning metabolite structure. This is the standard MAE trade-off (patch size and mask
ratio jointly set task difficulty), and it evidently outweighs the resolution gain.

Confound recorded: the small-patch models also have ~3× fewer parameters (0.63M / 0.66M
vs 1.89M), because the patch embedding and reconstruction head scale with patch size. So
the negative result is not purely about resolution. The falling reconstruction loss is
independent evidence for the trivial-task explanation, but a capacity-matched rerun
(`--d-model 256 --nhead 8 --dim-feedforward 512` at ps=128) would settle it if anyone
wants to reopen this.

### 5c. What actually worked: position-preserving pooling

The same experiment found a real, cheap win. Replacing `encoded.mean(dim=1)` (which
averages away *where* in the spectrum each token came from) with a flattened
position-preserving embedding helps on **all five** targets, at patch 1024:

| dataset | mean-pool (current) | flatten | gain |
|---|---|---|---|
| Barth | 0.677 | **0.806** | +0.129 |
| BrC-T2D diabetes | 0.687 | **0.780** | +0.093 |
| BrC-T2D cancer | 0.782 | **0.859** | +0.077 |
| MTBLS326 | 0.948 | **1.000** | +0.052 |
| MTBLS563 | 0.588 | **0.618** | +0.030 |

This makes sense given §5: chemical-shift position *is* the discriminative information in
NMR, and mean-pooling discards it. Note this is a **pooling** fix, not a resolution fix —
the tokens always carried the position information; the head was throwing it away.

Combining the two cheap wins (LogReg head from #2/#5 + flatten pooling) against the
originally reported DNN-head numbers:

| dataset | reported DNN head | best probe + flatten | classical LogReg | vs classical |
|---|---|---|---|---|
| Barth | 0.691 | **0.806** | 0.705 | **+0.101 (SSL wins)** |
| MTBLS326 | 0.981 | **1.000** | 1.000 | 0.000 (tie) |
| BrC-T2D cancer | 0.796 | 0.859 | **0.937** | −0.078 |
| BrC-T2D diabetes | 0.653 | 0.783 | **0.829** | −0.046 |
| MTBLS563 | 0.558 | 0.621 | **0.721** | −0.100 |

**+0.078 mean improvement over the reported numbers, with no retraining at all.** The
SSL-vs-classical record moves from 0 wins / 0 ties / 5 losses to **1 win / 1 tie / 3
losses**. Barth now favours SSL and MTBLS326 is a tie.

### 5d. Backbone scaling is exhausted (ps2048 + capacity arms) — ⚠️ PARTIALLY RETRACTED

> **CORRECTION (experiment #7, §7).** This section's reference cell is the **v3**-pretrained
> ps1024 checkpoint, while all four comparison backbones were pretrained on **v4**. When a
> v4 ps1024 baseline was finally trained with byte-identical config (arm D of experiment
> #7, verified against the CONFIG block in force on 2026-07-25), it scored **0.820**
> held-out flatten, not 0.888. Against that same-data reference the ordering changes sign:
>
> | backbone (all v4) | held-out flatten | vs v4 ps1024 |
> |---|---|---|
> | patch 2048 (5.42M) | 0.840 | **+0.020** |
> | patch 1024 d256 L6 (5.17M) | 0.826 | **+0.006** |
> | patch 1024 (1.89M) — v4 baseline | 0.820 | — |
> | patch 256 (0.66M) | 0.786 | −0.033 |
> | patch 128 (0.63M) | 0.778 | −0.042 |
>
> **Conclusion 1 below ("stop scaling the backbone") is not supported by the data as
> corrected** — on a same-data comparison the two ~5M arms are level-to-slightly-ahead, not
> behind. Conclusion 2 (pooling, not capacity, was the bottleneck) and §5b (shrinking the
> patch size hurts) both survive: ps256/ps128 are v4 and still lose to the v4 baseline.
> Everything below is left as originally written for the record. Do not cite the 0.888
> figure as a same-data baseline again.



Two further backbones were pretrained on the v4 corpus and compared through the frozen
probe: `patch_size=2048` (64 tokens, 5.42M params) and a capacity arm holding patch 1024
fixed while raising d_model 128→256, layers 3→6, ff 256→512 (5.17M params). Both
early-stopped. Figure `fig11_backbone_scaling.png`.

Mean balanced accuracy on the **held-out three** (Barth, MTBLS326, BrC-T2D cancer):

| backbone | params | recon loss | mean-pool | flatten |
|---|---|---|---|---|
| patch 128 | 0.63M | 4.36e-5 | 0.745 | 0.778 |
| patch 256 | 0.66M | 5.56e-5 | 0.755 | 0.779 |
| **patch 1024 (original)** | **1.89M** | 9.26e-5 | 0.802 | **0.888** |
| patch 2048 | 5.42M | 1.020e-4 | 0.818 | 0.840 |
| patch 1024 d256 L6 | 5.17M | **3.95e-5** | 0.814 | 0.826 |
| *classical LogReg* | — | — | — | *0.881* |

Even letting each backbone pick its own best pooling G **on the selection subset only**
(MTBLS563 + BrC-T2D diabetes), the held-out means are: original 0.849, ps2048 0.824,
d256L6 0.817. **The original small model wins under every pooling.**

Two conclusions:

1. **Patch size 1024 is near-optimal and this is not a capacity limit.** ps2048 carries
   2.9× the baseline's parameters and still loses. Four attempts (128, 256, 2048, and
   2.7× capacity) all failed. Stop scaling the backbone.
2. **Capacity compensates for bad pooling but does not beat fixing the pooling.** Under
   mean-pool, accuracy rises monotonically with parameters (Spearman = +1.00, p<0.01:
   0.745 → 0.755 → 0.802 → 0.814 → 0.818). But the 1.89M model with *flatten* (0.888)
   beats every 5M model. The bottleneck was information destroyed by pooling, not model
   capacity — extra capacity partially papers over the loss instead of removing it.

### 5e. Reconstruction loss is not a proxy for downstream utility

Across the five backbones, Spearman(validation reconstruction loss, held-out accuracy) =
**+0.60** for flatten and +0.40 for mean-pool — i.e. if anything, *worse* reconstruction
goes with *better* transfer (n=5, not significant, but the sign is consistent).

The starkest case: the d256 L6 model reconstructs **2.3× better** than the baseline
(3.95e-5 vs 9.26e-5) and transfers **worse** (0.826 vs 0.888). The same disconnect showed
up within a single run in §5b, where ps128's final epoch-1813 checkpoint had 2.4% better
reconstruction and scored lower downstream on 4 of 5 targets.

**Operational rule: never select checkpoints, architectures, or epochs on reconstruction
loss.** Selection must use a downstream signal — but on a *pre-committed* subset, never on
the datasets used for reporting (see the note below).

> **On selection bias.** Choosing configurations by comparing downstream CV scores and
> then quoting the winner inflates the reported number, even though no label information
> crosses folds. From §5d onward, configuration choices (pooling G, backbone) are made on
> a designated selection subset — MTBLS563 + BrC-T2D diabetes — and reported on the
> held-out three. The *comparative signs* in this document are more trustworthy than the
> absolute values, because each holds consistently across five independent datasets.

---

### 5f. ~~The largest effect measured so far is the baseline itself~~ — ❌ FULLY RETRACTED

> **RETRACTED 2026-08-09 by experiment #15 (§15). There is no corpus effect.**
>
> Everything below rests on comparing a v3 reference of **n=1** against a v4 mean of n=3.
> Running both corpora at n=5 seeds dissolves it — and the point estimate ends up slightly favouring **v4**, the opposite of the claim:
>
> | | held-out mean |
> |---|---|
> | v3, n=5 | **0.8165 ± 0.0449** (0.8884, 0.8190, 0.8067, 0.8033, 0.7653) |
> | v4, n=5 | **0.8305 ± 0.0206** (0.8667, 0.8272, 0.8232, 0.8199, 0.8158) |
> | gap **as reported** (v3 n=1) | **+0.0687** |
> | gap **with error bars on both sides** | **−0.0140, se 0.0221 → 0.6 se** |
>
> The 0.8884 reference ranks **1 of 5** and sits **+1.60 sd** above its own distribution. It
> was a lucky draw. The gap does not shrink — it vanishes and marginally reverses.
>
> Consequences, all recorded in §15: §8 and §13 were hunting the mechanism of an effect that
> does not exist; the single-run sd is **0.045**, not the 0.020 derived below; and the
> instruction "never compare a v3-pretrained checkpoint to a v4-pretrained one" is void —
> the two corpora are interchangeable.
>
> The text below is left in place as the record of what was believed, and of how a single
> unreplicated reference produced two follow-up experiments and a headline slide.

Experiment #7 required a v4-pretrained ps1024 baseline (arm D). Comparing it to the
v3-pretrained ps1024 checkpoint that every earlier number was measured against — **same
architecture, same objective, same hyperparameters, config verified byte-identical against
the CONFIG block in force on 2026-07-25** — gives:

| target | v3 baseline | v4 baseline (arm D) | Δ |
|---|---|---|---|
| Barth | 0.8059 | 0.7484 | −0.0575 |
| MTBLS326 | 1.0000 | 0.9296 | −0.0704 |
| BrC-T2D cancer | 0.8592 | 0.7816 | −0.0776 |
| MTBLS563 | 0.6176 | 0.6283 | +0.0108 |
| BrC-T2D diabetes | 0.7654 | 0.7052 | −0.0602 |
| **held-out mean** | **0.8884** | **0.8199** | **−0.0685** |

Down on 4/5 targets, all by a similar 0.06–0.08, so this is not one outlier dataset. Only
two things differ between the runs: the pretraining corpus version (v3 → v4, the EDTA
cutoff fix) and the unseeded model initialization / shuffling order.

**This −0.069 is larger than every effect any experiment here has reported** (§5b patch
size ±0.08 is comparable; §5c pooling +0.03..+0.13 overlaps; §7's factorial effects are
−0.030 and +0.011). Two mutually exclusive readings, and the data cannot currently
distinguish them:

1. **The v4 corpus is worse for pretraining than v3.** Plausible in principle — v4 caps the
   EDTA cutoff at the baseline-to-peak midpoint, so it removes less of the artifact and
   leaves more residual structure. Note arm D also *reconstructs better* (7.10e-5 vs
   9.26e-5) while transferring worse, which is another instance of §5e.
2. **Run-to-run variance at n=37..113 is ≈0.07.** If so, most conclusions in this document
   that rest on a single run per arm are underpowered, and only §5c (pooling, which is
   measured *within* a fixed checkpoint and therefore paired) is safe.

**Neither reading is optimistic, and (2) is the one to rule out first.** The check is cheap:
rerun arm D's exact command on v4 a second time (~4.5 h on one L40S; init is unseeded, so a
rerun is a genuine replicate) and see whether it lands at 0.82 or 0.89. Until that is done,
**treat any single-run difference below ~0.07 in this document as not established**, and do
not compare a v3-pretrained checkpoint against a v4-pretrained one.

`trainer_revised.py` has no `--seed` flag, which is why this could not be diagnosed from the
existing artifacts. Adding one is a prerequisite for any further single-run comparison.

> **✅ RESOLVED — reading (1). The corpus is the cause.** Two further v4 baseline runs
> (`--seed 101`, `--seed 202`) put three independent draws of this configuration at held-out
> means **0.8199 / 0.8232 / 0.8158**. The v3 reference at **0.8884** sits above the entire v4
> cluster, and at or above every individual v4 draw on all three held-out targets (strictly
> above on Barth and MTBLS326; tied on cancer). The +0.069 gap is real and **the v4 corpus is
> worse for pretraining than v3** — the EDTA-cutoff "fix" improved artifact removal on paper
> and cost downstream transfer. This is §5e again in a new place: v4 reconstructs better
> (7.10e-5 vs 9.26e-5) and transfers worse. See §7b for the noise floor this establishes,
> which is smaller than 0.069 but still large enough to invalidate several claims here.

## 6. Controls — what this is NOT

### Not CV overfitting
200-permutation nulls under the identical protocol, 1024 bins:

| dataset | observed | null mean | null p95 | null max | p |
|---|---|---|---|---|---|
| BrC-T2D cancer | 0.937 | 0.494 | 0.616 | 0.717 | 0.005 |
| BrC-T2D diabetes | 0.829 | 0.498 | 0.619 | 0.674 | 0.005 |
| MTBLS563 | 0.721 | 0.332 | 0.417 | 0.526 | 0.005 |
| MTBLS326 | 1.000 | 0.485 | 0.634 | 0.737 | 0.005 |
| Barth | 0.705 | 0.498 | 0.655 | **0.792** | 0.020 |

p=0.005 is the floor at 200 permutations. **Barth is marginal** — its null reaches 0.792,
*above* the observed 0.705. Treat Barth's classical result as weak evidence.

### Not a global dilution/intensity confound
A single scalar feature:

| dataset | total abs. area | row std | LogReg @1024 bins |
|---|---|---|---|
| BrC-T2D cancer | 0.616 | 0.705 | 0.937 |
| BrC-T2D diabetes | 0.593 | 0.392 | 0.829 |
| MTBLS563 | 0.337 | 0.426 | 0.721 |

There is *some* global component (notably cancer's row_std at 0.705) but it is far below
the full model. The signal is genuinely local spectral structure.

### CORRECTION: the Xavier ablation does not test what it appears to

`--reinit-unfrozen-xavier` calls `reinit_xavier_(layer)` **only inside the unfrozen-layer
loop**. Therefore:
- In `frozen` mode nothing is unfrozen → **nothing is reinitialized** → the "xavier" arm
  is byte-identical to the pretrained arm. Verified: Barth masking 0.526398 in both,
  MTBLS326 0.762963 in both.
- `patch_embedding`, `pos_encoding`, and all still-frozen layers **remain pretrained** in
  every mode.

So it tests "does discarding pretrained weights *in the layers being fine-tuned* hurt?",
not "pretrained vs random backbone." Read mode-by-mode (not max-over-modes),
**pretraining helps**, increasingly with depth — MTBLS326 masked:

| mode | pretrained | reinit |
|---|---|---|
| frozen | 0.763 | 0.763 (identical by construction) |
| +1 layer | 0.896 | 0.830 |
| +2 layers | 0.930 | 0.733 |
| +3 layers | **0.948** | 0.648 |

MTBLS563 at +3 layers: pretrained wins all three families (0.566/0.540/0.463 vs
0.509/0.476/0.414). Barth is noisy and mixed (n=37).

An earlier verbal claim in this project that "pretraining contributes nothing" came from
taking `max()` over four noisy modes per arm and is **retracted**. A true random-backbone
control has now been run — see §6b, which supersedes this section.

### 6b. Experiment #3 — the true random-backbone control (DONE)

`code/evaluation/ssl_linear_probe_eval.py --random-backbone` keeps each architecture but
loads **no pretrained weights anywhere** (patch embedding and positional encoding
included), and the classifier is held fixed at a converged LogReg (C=1) on frozen
features in both arms. Holding the head fixed removes the head-underfitting confound of
§4b, which is what made the original ablation unreadable.
Verified as a genuine control: embeddings differ from pretrained for all three families,
and are reproducible under a fixed seed.

Δ balanced accuracy (pretrained − random init):
`results/linear_probe/exp3_pretrained_vs_random.csv`,
figure `fig8_pretraining_gain.png`

| dataset | masking | jigsaw | joint |
|---|---|---|---|
| Barth | **+0.252** | +0.087 | +0.202 |
| MTBLS326 | **+0.052** | +0.015 | −0.100 |
| BrC-T2D cancer | **+0.063** | −0.028 | −0.077 |
| BrC-T2D diabetes | **+0.171** | −0.067 | −0.023 |
| MTBLS563 | **+0.047** | −0.065 | −0.126 |
| **mean** | **+0.117** | −0.011 | −0.025 |

**This is the single most important result in this document:**

- **Masked pretraining works.** Positive on 5/5 targets, mean +0.117. It is the only
  objective that earns its keep.
- **Jigsaw pretraining is worthless** (mean −0.011, loses to random on 3/5).
- **Joint pretraining is actively harmful** (mean −0.025, loses on 3/5, worst −0.126).
  A *random* joint backbone scores 0.846 on BrC-T2D cancer and 0.911 on MTBLS326 versus
  the pretrained 0.769 and 0.811 — the joint objective is destroying useful structure.

Interpretation caveat: a random transformer is a legitimately strong random-projection
baseline (cf. random-feature kernel methods), so "random wins" means **the objective adds
nothing over a random projection**, not that the architecture is useless.

Consequence for the roadmap: concentrate on the masking objective. Jigsaw and especially
joint pretraining need rethinking or dropping — and experiment #7 (peak-weighted
objective) is now *more* attractive for them, not less.

---

## 7. Open caveats to resolve

- **MTBLS326's perfect 1.000** on 42 samples clears its permutation null (p=0.005), so it
  is not a CV artifact — but permutation does not test batch confounding. Check whether
  the Yes/No label correlates with acquisition date / run order / instrument before
  publishing this number.
- **Barth and MTBLS326 have no error bars** (LOOCV). Their SSL-vs-classical gaps are
  within single-sample noise.
- **BrC-T2D diabetes' gap is weaker than the point estimate suggests** — per-fold IQRs
  overlap substantially (`fig5_brc_t2d_fold_variability.png`).
- **Stale defaults are a live hazard.** `fewshot_benchmark.py` still defaults to
  20260708 checkpoints, and its Barth entry points at
  `aligned_128K_Workbench_Barth_Syndrome.npy` — the **completely unsuppressed** raw file.
  Always pass paths explicitly.

---

## 8. Experiment queue

Ranked by expected payoff per unit effort. Run one at a time.

### ✅ #1 — Gap decomposition (DONE, §4/§5)
```bash
python -u code/analysis/probe_logreg_advantage.py \
  --datasets brc_t2d_cancer brc_t2d_diabetes mtbls563 mtbls326 barth \
  --n-permutations 200 2>&1 | tee results/logreg_advantage_probe.log
python code/plotting/plot_logreg_advantage_probe.py
```

### ✅ #2 — Linear probe on frozen embeddings (DONE, §4b)
```bash
python -u code/analysis/linear_probe_frozen_embeddings.py 2>&1 | tee results/linear_probe_frozen.log
python code/plotting/plot_linear_probe_vs_head.py
```
**Outcome:** masking head underfit by +0.120 mean on 5/5 targets. jigsaw/joint unaffected.

### ✅ #3 — True random-backbone control (DONE, §6b)
```bash
python -u code/evaluation/ssl_linear_probe_eval.py --no-tune-C \
  --output-root results/linear_probe/exp3_pretrained_C1
python -u code/evaluation/ssl_linear_probe_eval.py --no-tune-C --random-backbone \
  --output-root results/linear_probe/exp3_randominit_C1
python code/plotting/plot_pretraining_gain.py
```
**Outcome:** masked pretraining +0.117 on 5/5; jigsaw −0.011; joint −0.025 (harmful).
Still outstanding if wanted: the same three-arm comparison *through the fine-tuning
path* (pretrained / unfrozen-reinit / fully-random), mode by mode. The frozen-feature
version above is cleaner and was sufficient to answer the question.

### ✅ #4 — Patch size (DONE, REFUTED — see §5b)
```bash
# pretraining (ps128 ~6.2h, ps256 ~4.0h on an L40S)
python -u code/training/trainer_revised.py --patch-size 128 --nhead 4 \
  --data-path data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v4.npy
# evaluation
python -u code/analysis/compare_patch_sizes.py
python code/plotting/plot_patch_size_experiment.py
```
**Outcome: shrinking patch_size hurt.** Do not pursue further. The win was pooling.

### ✅ #5 — Linear-probe head as a first-class evaluator (DONE)
```bash
python -u code/evaluation/ssl_linear_probe_eval.py            # nested-CV tuned C
python -u code/evaluation/ssl_linear_probe_eval.py --no-tune-C # fixed C=1 (recommended)
```
`code/evaluation/ssl_linear_probe_eval.py` emits standard `summary.csv` /
`fold_metrics.csv` / `oof_predictions.csv` / `run_config.json` per dataset, reporting the
probe as an **additional** model per family rather than silently replacing the fine-tuned
heads (on MTBLS326 the fine-tuned masking head 0.981 genuinely beats the probe 0.963).

**Outcome:** masking probe beats its best fine-tuned head by **+0.054** (tuned C) /
**+0.057** (fixed C=1).

**Negative finding worth recording: nested-CV tuning of C did not help.** Mean change
across the 15 dataset×family cells was negative — it gained in 4 cells (+0.013..+0.031)
but lost in 7 (down to −0.077). With n=78 at 10-fold the inner CV selects C from ~14
samples per inner fold, so tuning adds variance without reducing bias. **Use fixed
`C=1`** at these sample sizes; the theoretically-cleaner option is empirically worse.

### ❌ #6 — RESULT: the few-shot advantage does not exist

The experiment the project's premise rests on. Every number in §3–§5 is measured with the
full dataset, where n=37..113 is already near the ceiling of what is learnable — the worst
regime in which to look for a transfer advantage. Transfer is supposed to pay off when
labels are scarce, so this sweeps the label budget from 2 per class upwards on **all five
targets** and asks where, if anywhere, the SSL curve sits above the classical one.

Scripts: `code/evaluation/fewshot_benchmark.py`,
`code/analysis/summarize_fewshot_masking_vs_classical.py`,
`code/plotting/plot_fewshot_masking_vs_classical.py`. Data
`results/analysis/fewshot_masking_vs_classical/`, figures `fig17`, `fig18`.

**Three prerequisites had to be fixed before the run meant anything.** The fine-tuning
classifiers still hardcoded `encoded.mean(dim=1)` — the exact pooling defect §5c fixed
everywhere else — so left alone this would have handicapped SSL by 0.03..0.13 in the one
experiment designed to favour it. The dataset defaults still pointed at pre-v4 files
(including Barth's *unsuppressed* raw array). And BrC-T2D had no loader here at all, so
only 3 of 5 targets were reachable. All three are now fixed; `--pooling` defaults to
`regional:16`.

**The comparison is PAIRED.** `build_shared_splits` generates the (support, query) episodes
once per (seed, support size), independent of model, and every family consumes identical
draws. So at a fixed (dataset, support, repeat) the classical and SSL rows saw the same
support set and the same query set, and their difference removes the episode-draw variance —
which is large here, sd 0.07..0.15 on a single episode, versus a paired se near 0.016. This
is the §15 distinction applied prospectively rather than retrospectively.

| target | Δ at smallest budget | Δ at largest | pooled Δ ± se | reading |
|---|---|---|---|---|
| Barth (n=37) | +0.060 @2 | −0.123 @12 | **−0.029 ± 0.012** | crossover |
| MTBLS326 (n=42) | −0.039 @2 | +0.018 @13 | **−0.019 ± 0.013** | crossover |
| MTBLS563 (n=113) | −0.061 @2 | −0.099 @24 | **−0.088 ± 0.009** | classical ahead everywhere |
| BrC-T2D cancer (n=78) | +0.024 @2 | −0.050 @36 | **−0.059 ± 0.011** | crossover |
| BrC-T2D diabetes (n=78) | +0.022 @2 | −0.087 @32 | **−0.047 ± 0.014** | crossover |

**Masked SSL does not beat classical LogReg at any label budget on any target.** Three
targets look positive at 2 labels/class, none individually significant, and **pooled across
all five: +0.0012 ± 0.0162, p = 0.74** (Wilcoxon, n=50) — dead even. What looks like a
low-shot edge is both methods sitting near chance where nothing is learnable yet.

**The deficit widens as labels accumulate** — Spearman(support, Δ) is negative on 4 of 5
(diabetes p=0.036). That is the opposite direction from the transfer premise, which predicts
the SSL edge shrinking as labels arrive, not growing.

**Fine-tuning the backbone adds nothing.** All four modes land within ~0.01 of each other on
every target; `frozen` matches `unfreeze_last_3`. Consistent with §5e — the representation
is being read out, not usefully adapted.

Two checks that make this a result rather than a plumbing artifact:

- **Validation gate.** The top of each curve lands just below that target's known full-data
  value (Barth 0.782 vs 0.705, cancer 0.900 vs 0.937, MTBLS326 0.957 vs 1.000, MTBLS563
  0.659 vs 0.721, diabetes 0.762 vs 0.829). The harness measures what §3 measures.
- **Pooling sensitivity.** The obvious objection is that `regional:16` gives 2,048 features
  against 4 samples at support=2, handicapping SSL precisely where the claim is made. It
  does not: `regional:16` **won 6 of 6** low-support points against `mean_pool` (128 feat)
  and `regional:4` (512 feat) on Barth / BrC cancer / MTBLS563. Even letting SSL pick its
  best pooling post-hoc per point — which inflates SSL — the pooled estimate is
  **−0.033 ± 0.014, p = 0.006**. Still negative.

**What this closes.** §6 was the strongest remaining case for the backbone. Combined with
§6b (masked pretraining does learn something real, +0.117 over random init) the picture is
coherent rather than contradictory: the objective works, but it encodes less discriminative
signal than 1,024-bin integrated areas already carry, and no axis we pushed (patch size §5b,
capacity §5d, objective §7, corpus §15) moved it.

**The most likely cause still untested is corpus scale**: 9,670 spectra is very small for a
foundation model. §13 only probed corpus size *downward* (dropping 1–10%), which cannot
detect an under-training regime. See the queue below.

```bash
# pass 1 -- classical + masking, coarse support grid, 2 jobs per GPU
python -u code/evaluation/fewshot_benchmark.py --dataset brc_t2d_cancer \
  --families classical masking --support-sizes 2 5 10 15 20 25 30 36 --device cuda \
  --output-dir results/fewshot/brc_t2d_cancer_v2_coarse
# pass 2 -- jigsaw + joint. NOTE: a DEDICATED --output-dir per pass. The script
# rebuilds all_rows from scratch and checkpoints after every family, so pointing
# pass 2 at pass 1's directory overwrites it (this destroyed MTBLS326's pass-1
# episodes once; load_episodes() now refuses a directory missing a family).
python -u code/evaluation/fewshot_benchmark.py --dataset brc_t2d_cancer \
  --families jigsaw joint_ssl --support-sizes 2 5 10 15 20 25 30 36 --device cuda \
  --output-dir results/fewshot/brc_t2d_cancer_pass2
python code/analysis/summarize_fewshot_masking_vs_classical.py
python code/plotting/plot_fewshot_masking_vs_classical.py
```

### ❌ #7 — RESULT: block masking fails, peak weighting is a wash

All four arms early-stopped cleanly (D ep1254, A ep1703, B ep1628, C ep1640).
Evaluated through the frozen linear probe, `results/analysis/exp7_objective_comparison/`,
summarized by `code/analysis/summarize_exp7_factorial.py`, figure `fig12_exp7_factorial.png`.

**The pretext-difficulty mechanism fired.** D and A optimize the same uniform loss on the
same data, so their val losses are directly comparable: block masking raised val loss
7.10e-5 → 1.00e-4 (**+41% harder**) and pushed the best epoch out by 450. The task did get
harder as designed. It just did not help.

Balanced accuracy, flatten pooling:

| target | classical | D (ref) | A block | B peak | C both |
|---|---|---|---|---|---|
| Barth | 0.705 | 0.748 | 0.562 | 0.655 | 0.634 |
| MTBLS326 | 1.000 | 0.930 | 0.944 | **1.000** | 0.911 |
| BrC-T2D cancer | 0.937 | 0.782 | 0.872 | 0.845 | 0.858 |
| MTBLS563 *(sel)* | 0.721 | 0.628 | 0.556 | 0.596 | 0.594 |
| BrC-T2D diabetes *(sel)* | 0.829 | 0.705 | 0.710 | 0.754 | 0.686 |
| **held-out mean** | **0.881** | 0.820 | 0.793 | **0.834** | 0.801 |
| selection mean | 0.775 | 0.667 | 0.633 | 0.675 | 0.640 |

Factorial main effects (each averaged over both levels of the other factor):

| effect | held-out | selection | all 5 |
|---|---|---|---|
| block masking | **−0.030** | **−0.034** | **−0.032** |
| peak weighting | +0.011 | +0.007 | +0.009 |
| interaction | −0.006 | −0.001 | −0.004 |

1. **Block masking hurts, consistently.** Negative on both splits and both poolings
   (−0.014..−0.034), and it does not merely fail to help — Barth collapses 0.748 → 0.562.
   Making the pretext task harder is not sufficient; §5b's diagnosis ("the task is too
   easy") was correct as a description but wrong as a prescription. A plausible reading: an
   8-patch span is ~0.75 ppm, wide enough that the true content is genuinely
   unrecoverable rather than merely non-trivial, so the model learns to predict a
   conditional mean and loses the sharp local detail the probe reads.
2. **Peak weighting is a wash.** +0.011 held-out flatten, +0.007 selection — same sign
   everywhere and B is the best of the four arms (and reaches 1.000 on MTBLS326 under both
   poolings), but the magnitude is ~6× smaller than the baseline uncertainty established in
   §5f. **Not established.** Retesting at `--peak-top-fraction 0.50` is defensible; treating
   +0.011 as a win is not.
3. **No arm approaches classical.** Best held-out mean 0.834 vs 0.881 classical.
4. **The factorial's real payoff was arm D**, which exposed the v3/v4 baseline confound
   (§5f) — an effect 2× larger than anything the experiment was designed to measure.

**Do not pursue block masking further.** Resolve §5f before running any new arm.

### ❌ #7b — RESULT: peak weighting also fails once the corpus is matched (noise floor here is ~~0.020~~ → **0.045**, see §15)

Three follow-up runs, `results/analysis/exp7_replicates/`, summarized by
`code/analysis/summarize_exp7_replicates.py`, figure `fig13_exp7_replicates.png`.

**(1) ~~`--seed` does not make GPU training reproducible~~ — ⚠️ THIS CLAIM WAS WRONG, RETRACTED
2026-08-09.**

> The original claim: the v3 peak arm was launched twice with `--seed 101`, identical flags
> and corpus, and the two runs appeared not to match — best epoch 724 vs 776, val loss
> 2.386e-4 vs 2.190e-4, max|ΔW| = 5.3e-2 — which was attributed to `cudnn.benchmark = True`
> autotuning kernels by timing race plus AMP's dynamic rescaling.
>
> **That comparison was made against a checkpoint that was still being written.** `r1` was at
> epoch 724 *at the time of measurement* and went on to reach epoch 776. Its file mtime
> (13:29) is later than `r2`'s (09:28) — which should have been checked and was not. With
> both runs finished, the two checkpoints are **byte-identical**: same epoch (776), same val
> loss (2.189681e-04), `torch.equal` True on every tensor, **max|ΔW| = 0.000e+00**.
>
> **`--seed` does make this training run reproducible.** Consequences:
> - Queue item #10 (`--deterministic` mode) is **unnecessary**; dropped.
> - The noise floor in (2) below is **unaffected** — it comes from three v4 runs with
>   *different* seeds, which are genuine independent draws. But it is now attributable
>   entirely to **seed/initialization**, not to hardware nondeterminism.
> - The peak-weighting result in (3) below used the mid-training `r1` value (0.8525). With
>   the final checkpoint it is 0.8461, moving the matched delta from −0.039 to **−0.042**.
>   The conclusion (peak weighting loses) is unchanged and slightly strengthened.
>
> This is the third time a stale or mid-training checkpoint has produced a wrong number in
> this project (see also the ps128 correction in §5b). **Guard needed:** write a
> `finished: true` flag into the checkpoint at end-of-training and have the evaluators refuse
> to score a checkpoint that lacks it.

**(2) The noise floor.** Three independent v4 baseline runs (unseeded, seed 101, seed 202):

| target | run 1 | run 2 | run 3 | sd | range | v3 ref |
|---|---|---|---|---|---|---|
| Barth | 0.7484 | 0.6988 | 0.6770 | 0.037 | 0.071 | 0.8059 |
| MTBLS326 | 0.9296 | 0.9630 | 0.9111 | 0.026 | 0.052 | 1.0000 |
| BrC-T2D cancer | 0.7816 | 0.8079 | 0.8592 | 0.040 | 0.078 | 0.8592 |
| MTBLS563 | 0.6283 | 0.5505 | 0.6331 | 0.046 | 0.083 | 0.6176 |
| BrC-T2D diabetes | 0.7052 | 0.7701 | 0.7052 | 0.037 | 0.065 | 0.7654 |
| **held-out mean** | **0.8199** | **0.8232** | **0.8158** | **0.0037** | 0.0074 | **0.8884** |

**Do not read that 0.0037 as precision.** Per-target sd averages 0.035, so a 3-target mean
should scatter by ≈0.035/√3 = **0.020** if the targets were independent. The observed 0.0037
is 5.4× tighter because Barth *falls* while cancer *rises* across the three draws
(r = −0.92) and the errors cancel inside the mean. With three draws that cancellation is
luck, not a property to rely on. **Use 0.020 as the floor for a held-out-mean claim and
≈0.035 for any single-target claim.**

> **⚠️ CORRECTED by §15.** The 0.020 above was still too optimistic, and for the reason this
> passage itself identifies: it was derived from a three-run cancellation that was flagged as
> a fluke and then used as a floor anyway. Measured directly from five v3 seeds, the
> single-run sd on the held-out mean is **0.045**, and per-target it reaches **0.076** (Barth).
> Every "vs 0.020 floor" figure below understates the noise by more than 2×; the corrected
> re-scoring is in §15.**

**(3) Peak weighting, matched corpus — it loses.** Peak-weighted runs on v3, against the v3
baseline (no corpus confound):

| target | classical | v3 baseline | v3 + top-25% r1 | r2 |
|---|---|---|---|---|
| Barth | 0.705 | 0.8059 | 0.6910 | 0.6910 |
| MTBLS326 | 1.000 | 1.0000 | 0.9630 | 0.9630 |
| BrC-T2D cancer | 0.937 | 0.8592 | 0.8842 | 0.8842 |
| MTBLS563 | 0.721 | 0.6176 | 0.5949 | 0.5949 |
| BrC-T2D diabetes | 0.829 | 0.7654 | **0.6237** | **0.6237** |
| **held-out mean** | 0.8807 | **0.8884** | 0.8461 | 0.8461 |
| selection mean | 0.7750 | 0.6915 | 0.6093 | 0.6093 |

**−0.042 held-out** (2.1× the floor) and **−0.082 on the selection subset**, driven by a
−0.142 collapse on BrC-T2D diabetes. (r1 and r2 are the same seed and, as corrected in (1)
above, the same final model — the columns are identical; both are shown only because the
original write-up treated them as a pair.) Arm B's +0.011 was an artifact of being measured
against a v4 baseline that the corpus had already depressed by 0.069. **Experiment #7 is now
negative on both factors.**

**(4) Recalibration — half of this document's claims do not clear the floor.**

> ⚠️ **This table uses the 0.020 floor and is superseded.** The measured single-run sd is
> 0.045 (§15), against which only §6b survives. The corrected table is in §15.

| claim | Δ | vs 0.020 floor *(superseded)* | verdict *(superseded)* |
|---|---|---|---|
| §5f v3 vs v4 corpus | +0.069 | 3.4× | **survives** |
| §5b patch 128 vs 1024 | −0.042 | 2.1× | **survives** |
| §7b peak weighting (v3, matched) | −0.042 | 2.1× | survives (marginal) |
| §5b patch 256 vs 1024 | −0.034 | 1.7× | marginal |
| §7 block masking | −0.030 | 1.5× | marginal |
| §5d ps2048 vs ps1024 | +0.020 | 1.0× | **within noise** |
| §7 peak weighting (v4, unmatched) | +0.011 | 0.5× | **within noise** |
| §5d d256L6 vs ps1024 | +0.006 | 0.3× | **within noise** |

Consequences: **§5d's partial retraction is itself now unsupported** — the "+0.020 ps2048"
that motivated it sits exactly at the floor, so the honest statement is that ps2048, d256L6
and ps1024 are *indistinguishable* on one run each, not that scaling helps. §5c (pooling)
is unaffected because it is measured *within* a fixed checkpoint and is therefore paired —
that remains the only robust positive result in this document.

**Standing rule from here on: no single-run comparison below 0.04 gets reported as an
effect.** Either run ≥3 replicates per arm, or restrict claims to paired within-checkpoint
comparisons like §5c.

### 🟡 #8 — RESULT: the 164 differing rows do not explain the corpus gap
### ⚠️ SUPERSEDED by §15 — this experiment was looking for the mechanism of a non-effect

> **§15 showed there is no corpus gap to explain.** With n=5 seeds per corpus the v3-vs-v4
> difference is −0.0140 (0.6 se), not +0.069. This experiment's finding — that
> dropping the 164 differing rows is indistinguishable from dropping 164 arbitrary rows —
> was *correct*, and now has an obvious reading it could not reach at the time: **both arms
> are just draws from the same distribution, because the corpora are equivalent.**
>
> Note also that its central comparison (common vs control, |Δ| = 0.001) was judged against a
> 0.020 floor. The real single-run sd is 0.045, so that comparison was even less able to
> resolve a difference than stated. Kept for the record.

§5f/§7b established the v3→v4 corpus swap costs 0.069 held-out, the largest effect in this
document. A direct diff of the two corpora narrowed the search: only **164 of 9,670 rows
(1.7%)** differ at all, but each differs almost entirely — 99.998% of its 131,072 points
change, median max|Δ| = 0.34 — because rows are min-max normalized, so changing a row's
maximum rescales the whole row. Row indices: `results/analysis/corpus_v3_v4_diff/differing_rows.csv`.

The suspected mechanism (v4 leaves a residual EDTA artefact that compresses those 164
spectra) was tested first and **refuted**: only 7/164 rows have their row-max inside the
EDTA window in either version, and v4's rows are slightly *brighter* outside it (99.9th
pct 0.969 vs 0.917) — less compressed, not more.

**Ablation, built by `code/preprocessing/build_corpus_subset.py`:**

| arm | construction | held-out mean |
|---|---|---|
| `common9506` | the 9,506 rows identical in v3 and v4 (164 dropped) — verified `max\|Δ\|=0` against both corpora | 0.837 |
| `v3rand9506_control` | v3 with 164 **different**, always-unchanged rows dropped at random — same size, keeps all 164 special rows | 0.836 |
| v3 reference | (unchanged) | 0.888 |
| v4 (3-replicate mean) | (unchanged) | 0.820 |

**The decisive comparison is common vs control, not common vs {v3, v4}.** If dropping an
*arbitrary* 164 rows lands in the same place as dropping the *specific* 164 that differ
between corpora, the content of those rows isn't shown to matter. It doesn't:

| target | common | control | \|Δ\| |
|---|---|---|---|
| Barth | 0.691 | 0.713 | 0.022 |
| MTBLS326 | 0.963 | 0.963 | 0.000 |
| BrC-T2D cancer | 0.858 | 0.834 | 0.024 |
| MTBLS563 | 0.591 | 0.618 | 0.028 |
| BrC-T2D diabetes | 0.751 | 0.751 | 0.000 |
| held-out mean | 0.837 | 0.836 | **0.001** |

Every per-target gap is inside the 0.035 floor; the held-out-mean gap (0.001) is far inside
the 0.020 floor. **Row content is refuted as the (sole) explanation.**

Both arms instead sit close to the **v4** mean (Δ 0.018, inside the floor) and clearly
below **v3** (Δ 0.051, outside it) — which looks like it points at corpus *size* (9,506 vs
9,670, a 1.7% cut) as the driver. **That reading is not confirmed either.** A 1.7% size cut
producing a ~0.05 held-out swing would be wildly disproportionate next to every capacity
result in this document: scaling the backbone by up to 2.9× parameters moved accuracy by
≤0.02 (§5d, and even that is now known to be within the floor). Treat "it's corpus size"
as an open hypothesis, not a second established fact — n=1 per arm here, same as every
number in §7/§7b before replication.

```bash
python code/preprocessing/build_corpus_subset.py --mode common
python code/preprocessing/build_corpus_subset.py --mode random-control --seed 7
V=data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax
python -u code/training/trainer_revised.py --patch-size 1024 --nhead 4 --seed 101 --data-path ${V}_common9506.npy
python -u code/training/trainer_revised.py --patch-size 1024 --nhead 4 --seed 101 --data-path ${V}_v3rand9506_seed7.npy
python -u code/analysis/compare_patch_sizes.py --out-dir results/analysis/exp8_corpus_subset
python code/analysis/summarize_exp8_corpus_subset.py
python code/plotting/plot_exp8_corpus_subset.py
```

**Next**: ≥2 more replicates each of `common9506` and the control before either the
row-content or corpus-size hypothesis is reported as established. If size is confirmed,
the natural follow-up is a size sweep (drop 1%, 5%, 10% at random from v3) to see whether
0.05 is really what a 1.7% cut costs, or whether these two single runs simply landed
unluckily.

<details><summary>Original design notes (kept for the record)</summary>
This is now the top-ranked experiment, because §5b identified the *reason* the previous
attempt failed and this attacks it directly. Two orthogonal changes, run as a 2×2
factorial on identical v4 data at the winning geometry (ps1024, d128, L3, nhead 4):

**(a) Block masking** — `--mask-strategy block --mask-block-patches 8`.
§5b showed that shrinking the patch size made reconstruction *easier* (loss fell
9.26e-5 → 4.36e-5) while downstream transfer got *worse*: a lone masked patch bracketed
by intact neighbours is largely interpolable, so the model can win by local smoothing
without learning spectral structure. Masking a contiguous 8-patch span (8192 points,
~0.75 ppm of a 12 ppm window) removes the neighbours too, so filling it in requires
long-range information. This is the standard fix in vision MAE (block/grid masking) for
exactly the same failure.

**(b) Peak-weighted reconstruction** — `--peak-top-fraction 0.25`.
Restricts the loss to the highest-magnitude 25% of patches per spectrum, ranked by
`0.5·mean|x| + 0.5·max|x|`. Ported verbatim from `top_peak_bin_weights()` in
`train_joint_ssl.py`, which the joint family already uses, so the two families now weight
reconstruction identically. Verified by `code/tests/verify_top_peak_loss.py` (elementwise
agreement with the joint implementation on synthetic and real spectra; exact-k selection;
per-spectrum thresholding; bit-identical to the old uniform loss at fraction 1.0).

**Measured caveat on (b), from that verification.** At ps1024 each patch spans 1024
points, so nearly every patch contains *some* signal and the "mostly flat baseline"
framing is weaker than it sounds. Share of total |intensity| held by the kept patches, on
64 real corpus spectra:

| fraction kept | 0.05 | 0.10 | 0.25 | 0.50 | 0.75 |
|---|---|---|---|---|---|
| intensity share | 0.234 | 0.394 | 0.594 | 0.761 | 0.897 |
| enrichment vs. uniform | 4.67× | 3.94× | 2.38× | 1.52× | 1.20× |

So `0.25` concentrates supervision 2.4× but **discards ~40% of the signal**. 0.25 is used
anyway for comparability with the joint family; if the peak-weighted arms lose, rerun at
`--peak-top-fraction 0.50` (1.5× enrichment, only 24% of signal dropped) before
concluding the idea is wrong.

**Arms.** D is not optional: every existing masking baseline was pretrained on **v3**,
so without a v4 ps1024 default run the factorial confounds objective with data version.

| arm | flags | checkpoint tag |
|---|---|---|
| D — baseline | *(defaults)* | *(none)* |
| A — block only | `--mask-strategy block --mask-block-patches 8` | `_blk8` |
| B — peak only | `--peak-top-fraction 0.25` | `_pk0.25` |
| C — both | both of the above | `_blk8_pk0.25` |

**Reading the result.** Validation loss is **not** comparable across arms — the arms
optimize different quantities, and per §5e reconstruction loss does not predict downstream
utility anyway. Judge only on the frozen linear probe with flatten pooling, using the
pre-committed split: select on MTBLS563 + BrC-T2D diabetes, report on Barth + MTBLS326 +
BrC-T2D cancer.

Evaluation requires adding the four new checkpoint paths to `ARMS` in
`code/analysis/compare_patch_sizes.py` once the timestamps exist.

</details>

```bash
# the four arms as run (each ~4.5 h on an L40S)
V4=data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v4.npy
python -u code/training/trainer_revised.py --patch-size 1024 --nhead 4 --data-path $V4                      # D
python -u code/training/trainer_revised.py --patch-size 1024 --nhead 4 --data-path $V4 \
  --mask-strategy block --mask-block-patches 8                                                              # A
python -u code/training/trainer_revised.py --patch-size 1024 --nhead 4 --data-path $V4 \
  --peak-top-fraction 0.25                                                                                  # B
python -u code/training/trainer_revised.py --patch-size 1024 --nhead 4 --data-path $V4 \
  --mask-strategy block --mask-block-patches 8 --peak-top-fraction 0.25                                     # C
# verification of the loss, before any GPU time
python code/tests/verify_top_peak_loss.py
# evaluation
python -u code/analysis/compare_patch_sizes.py --out-dir results/analysis/exp7_objective_comparison
python code/analysis/summarize_exp7_factorial.py
python code/plotting/plot_exp7_factorial.py

# #7b follow-up: 3 v4 replicates + a matched-corpus peak arm (~4.5 h each)
V3=data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v3.npy
python -u code/training/trainer_revised.py --patch-size 1024 --nhead 4 --data-path $V4 --seed 101
python -u code/training/trainer_revised.py --patch-size 1024 --nhead 4 --data-path $V4 --seed 202
python -u code/training/trainer_revised.py --patch-size 1024 --nhead 4 --data-path $V3 \
  --peak-top-fraction 0.25 --seed 101
python -u code/analysis/compare_patch_sizes.py --out-dir results/analysis/exp7_replicates
python code/analysis/summarize_exp7_replicates.py
python code/plotting/plot_exp7_replicates.py
```

### ~~#9 — Revert the pretraining corpus to v3~~ — ❌ VOID (§15): the corpora are equivalent

> §15 measured both corpora at n=5 seeds: the difference is −0.014, inside one standard
> error. There is nothing to revert to and nothing to diagnose. Text below kept as the record.

§5f established with three replicates that v3-pretrained backbones transfer **+0.069 better**
than v4-pretrained ones at identical configuration — the largest and best-supported effect in
this document. §8 tried to localize *why* to the 164 rows that differ and could not: the
row-content hypothesis is refuted (common ≈ control), and the leading alternative (corpus
size) is unconfirmed and looks disproportionate for a 1.7% cut. **This does not weaken the
recommendation** — the +0.069 v3-vs-v4 gap itself is still 3.4× the noise floor and unexplained
either way. Nothing needs training to start: **every v4-pretrained arm should be re-read
against v3**, and all future pretraining should default to v3 until the mechanism is
understood. §8's next step (a size sweep: drop 1%/5%/10% of v3 at random) is now the
concrete way to make progress on *why*.

### ~~#10 — Determinism~~ — DROPPED, not needed (2026-08-09)
This item existed because §7b claimed `--seed` was insufficient on GPU. **That claim was
wrong** (see the retraction in §7b): two same-seed runs are byte-identical once both finish,
so `cudnn.benchmark = False` / `use_deterministic_algorithms(True)` / `CUBLAS_WORKSPACE_CONFIG`
would buy nothing and cost throughput. Replaced by the cheaper, actually-needed fix:

### ⭐ #10b — Stale-checkpoint guard (TRIVIAL, do this before any further evaluation)
Three wrong numbers in this project have come from scoring a checkpoint whose training run had
not finished (ps128 in §5b; the r1/r2 pair in §7b). Write `finished: true` into the checkpoint
at end-of-training and have `compare_patch_sizes.py` / `ssl_linear_probe_eval.py` refuse to
score a checkpoint that lacks it. Until then, manually confirm "Training completed after" in
the run's log before scoring.

### ✅ #15 — RESULT: the corpus effect was a sampling artifact, and the noise floor is 0.045

Ten pretraining runs, five per corpus, identical configuration differing only by seed.
`results/analysis/exp15_seed_study/`, `code/analysis/summarize_exp15_seed_study.py`,
figure `fig16_exp15_seed_study.png`.

| v3 run | held-out mean | | v4 run | held-out mean |
|---|---|---|---|---|
| **original (unseeded)** | **0.8884** ← rank 1 of 5 | | original (unseeded) | 0.8199 |
| seed 505 | 0.8190 | | seed 101 | 0.8232 |
| seed 404 | 0.8067 | | seed 202 | 0.8158 |
| seed 202 | 0.8033 | | seed 303 | 0.8667 |
| seed 303 | 0.7653 | | seed 404 | 0.8272 |
| **mean ± sd** | **0.8165 ± 0.0449** | | **mean ± sd** | **0.8305 ± 0.0206** |

| the §5f comparison | value |
|---|---|
| as reported — v3 (n=1) vs v4 (n=3) | **+0.0687** |
| with error bars on both sides — v3 (n=5) vs v4 (n=5) | **−0.0140, se 0.0221 → 0.6 se** |

**The gap does not shrink; it vanishes and marginally reverses.** The 0.8884 reference sits
**+1.60 sd** above its own distribution and is the highest of its five draws. §5f is retracted.

**The measured single-run sd is 0.045** on the held-out mean — more than double the 0.020
assumed since §7b — and per-target it is worse: **Barth 0.076**, MTBLS326 0.045, diabetes
0.029, MTBLS563 0.030, cancer 0.019. The 0.020 came from three v4 runs whose per-target
errors happened to cancel inside the mean; §7b explicitly called that cancellation a fluke
and then adopted it as a floor anyway. That was the error which let §5f, §8 and §13 proceed.

**Re-scoring every single-run claim against 0.045:**

| claim | Δ | vs sd | verdict |
|---|---|---|---|
| §6b masked pretraining vs random-init | +0.117 | 2.6× | **survives** |
| §5f v3 vs v4 corpus | +0.069 | 1.5× | **retracted — n=5 says ≈0** |
| §7b peak weighting (v3, matched) | −0.042 | 0.9× | within noise |
| §5b patch 128 vs 1024 | −0.042 | 0.9× | within noise |
| §5b patch 256 vs 1024 | −0.034 | 0.7× | within noise |
| §7 block masking | −0.030 | 0.7× | within noise |
| §5d ps2048 vs ps1024 | +0.020 | 0.5× | within noise |
| §7 peak weighting (v4, unmatched) | +0.011 | 0.2× | within noise |
| §5d d256L6 vs ps1024 | +0.006 | 0.1× | within noise |

**What survives the whole document:** §6b (pretraining beats a random-init control, the only
single-run claim clearing 2 sd) plus the **paired within-checkpoint** results, which carry no
training variance by construction because they compare transforms on a *fixed* checkpoint —
§4b (head fix, +0.120), §5c (pooling, +0.03…+0.13) and §14 (jigsaw/joint pooling, +0.079 /
+0.049). Everything else that was reported as an effect is within noise.

**Standing rule, replacing the one in §7b:** a single-run difference below **0.09** (2 sd) is
not an effect. Report either ≥5 replicates per arm with a standard error, or a paired
within-checkpoint comparison. At ~4.5 GPU-hours per run, a 5-replicate arm costs ~23 GPU-h —
that is now the price of a reportable pretraining result, and it should be budgeted before an
experiment is proposed, not discovered afterwards.

```bash
V3=data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v3.npy
for s in 202 303 404 505; do
  python -u code/training/trainer_revised.py --patch-size 1024 --nhead 4 --seed $s --data-path $V3
done   # (and likewise for v4; see code/training/run_seed_queue.sh for the memory-gated launcher)
python -u code/analysis/compare_patch_sizes.py --out-dir results/analysis/exp15_seed_study
python code/analysis/summarize_exp15_seed_study.py
python code/plotting/plot_exp15_seed_study.py
```

### ⭐ #16 — Cross-cohort transfer (the strongest untested case for the backbone)
Every evaluation so far is *within* a cohort. A foundation model's actual selling point is
generalising across instruments, sites and batches — exactly where absolute binned
intensities should degrade and a learned representation might not. Train the probe on one
cohort, test on another; 4 cohorts are available and no retraining is needed, so this is
cheap. If SSL wins anywhere it is here; if it loses here too, the backbone approach is closed
at this data scale.

### ⭐ #17 — Corpus scaling curve, upward
Pretrain at 25 / 50 / 100% of available spectra and check whether downstream utility is still
climbing at 9,670. Flat ⇒ the objective is the limit. Rising ⇒ the answer is "needs 10–100×
more data". Either is actionable. Budget ≥5 seeds per point per §15's rule (~23 GPU-h each),
so ~70 GPU-h for three points — decide the point count before starting.

### #11 — Batch-confound audit of MTBLS326 (PREREQUISITE, still outstanding)
Classical LR scores a perfect 1.000 and several SSL arms reach 0.963–1.000. A perfect score on
n=42 is more likely a batch/run-order artifact than real biology. Until this is checked,
MTBLS326 should not be counted as evidence for anything.

### #12 — Hybrid features (CHEAP, worth a shot)
Concatenate the SSL embedding with binned areas. They are partly complementary — on
diabetes and Barth the embedding beats same-resolution binning — so the union may exceed
both.

### 🟡 #13 — RESULT: corpus size is not supported either
### ⚠️ SUPERSEDED by §15 — same reason as §8: there was no effect to explain

> **§15 showed the v3-vs-v4 gap does not exist**, so the "cause" this sweep went looking for
> was never there. Its own reading — non-monotonic, 5%→10% *recovering* accuracy, "the
> signature of n=1-per-point sampling noise, not a real effect" — turns out to have been
> exactly right, and to apply to the thing that motivated it as well as to the sweep itself.
> That the sweep's own points scattered by 0.081 across a 1–10% cut was, in hindsight, the
> noise floor announcing itself. Kept for the record.

Three arms — `build_corpus_subset.py --mode size-sweep --frac {0.01, 0.05, 0.10} --seed 301`
— drop that fraction of **all** v3 rows uniformly at random, decoupled entirely from the 164
rows that differ from v4. Pretrained identically to every other arm in this line
(ps1024, nhead4, seed 101). Read through the frozen probe against the existing v3 reference
(0.888 held-out) and experiment #8's common9506/control pair (mean 0.837 at a 1.7% cut):

| corpus | held-out mean |
|---|---|
| v3 (0% dropped) | 0.888 |
| exp8 common+control mean (1.7% dropped, the 164 rows) | 0.837 |
| **1% dropped (random)** | **0.814** |
| **5% dropped (random)** | **0.807** |
| **10% dropped (random)** | **0.859** |

**Non-monotonic, and in the direction that argues against a size effect.** A genuine size
effect predicts degradation that is monotone-or-flat as more rows are removed. Instead the
5%→10% step *recovers* 0.052 — more than double the 0.020 mean floor — rather than getting
worse. That is the signature of n=1-per-point sampling noise, not a real effect: each point
here, like every arm in §8, is a single pretraining run.

**Combined with §8 (row content refuted), neither live hypothesis for the v3-vs-v4 gap
survives.** The +0.069 gap itself (§5f) remains real and well-supported — three replicates,
3.4× the noise floor — but its *mechanism* is still open. Single-run size/subset ablations
are not the right tool to find it; each one only adds another n=1 data point to a question
that needs either many replicates per condition or a different kind of evidence entirely
(e.g., characterizing what actually changed in the 164 rows beyond "the row maximum moved,"
or a systematic look at what preprocessing step differs between v3 and v4 outside the EDTA
window specifically).

```bash
python code/preprocessing/build_corpus_subset.py --mode size-sweep --frac 0.01 --seed 301
python code/preprocessing/build_corpus_subset.py --mode size-sweep --frac 0.05 --seed 301
python code/preprocessing/build_corpus_subset.py --mode size-sweep --frac 0.10 --seed 301
V=data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax
for f in 1 5 10; do
  python -u code/training/trainer_revised.py --patch-size 1024 --nhead 4 --seed 101 \
    --data-path ${V}_v3drop${f}pct_seed301.npy
done
python -u code/analysis/compare_patch_sizes.py --out-dir results/analysis/exp13_size_sweep
python code/analysis/summarize_exp13_size_sweep.py
python code/plotting/plot_exp13_size_sweep.py
```

**Next, if this is worth resolving further**: replicate at least one size point (e.g. 5%,
multiple seeds) to get an actual noise estimate for a *dropped-corpus* arm rather than
assuming it matches §7b's floor (measured on the full corpus, not a subset — subset
training could plausibly be noisier). Absent that, do not spend more single-run ablations on
this question.

### #14 — RESULT: pooling helps jigsaw/joint too (modest); hybrid features refuted

Both used **existing v3 checkpoints only** — no retraining.

**Pooling.** `pool_tokens()` (§5c's frozen, position-preserving transform) extended to
jigsaw and joint: each family already exposes per-token encodings before its internal
mean-pool (jigsaw's per-bin-size transformer output; joint's `encode_bins`), so the same
transform applies per bin size before concatenating across bin sizes, exactly as native
pooling does at G=1. Regression-tested first: `pooling="native"` reproduces the committed
BrC-T2D-cancer numbers exactly (jigsaw 0.7684210526315789, joint 0.7690789473684211) —
confirms the refactor changed nothing about the existing path.

| family | native (G=1) mean, all 5 targets | best-other-G mean | Δ |
|---|---|---|---|
| jigsaw | 0.678 | 0.758 | **+0.079** |
| joint | 0.673 | 0.721 | **+0.049** |

Same direction as masking's pooling win, smaller magnitude. Like §5c, this is a **paired
within-checkpoint comparison** — same frozen features, only the pooling transform changes —
so it does not carry the GPU-training noise from §7b/§13; it is not an n=1 claim in the same
sense as everything above. `flatten`/`G=64` tend to top the sweep but at 49k–208k features
against n=37–113, which is a real overfitting risk despite the good CV numbers; a moderate
G (8–16) is the safer default pending a proper nested-CV pick, matching the reasoning that
settled masking's default at G=16 over flatten in §5c. Script:
`code/analysis/sweep_pooling_jigsaw_joint.py`.

**Hybrid features — refuted.** Concatenating each family's best frozen embedding with the
standard 1024-bin classical feature:

| family | wins vs. max(solo embedding, classical alone) | mean Δ |
|---|---|---|
| masking | 0/5 | **−0.043** |
| jigsaw | 1/5 | **−0.107** |
| joint | 0/5 | **−0.115** |

Consistently worse, not better, on nearly every target. The two feature sets are not
complementary enough to outweigh the cost of concatenating 1k–17k extra features on
n=37–113 samples — the earlier "partly complementary on Barth/diabetes" observation (§4b)
does not survive contact with the full high-dimensional concatenation. Close this out;
don't revisit without a dimensionality-reduction step first. Script:
`code/analysis/hybrid_features.py`.

---

## 9. Provenance

- Data: v4 (`*_v4.npy`), built by `code/preprocessing/build_clean_datasets.py` after the
  EDTA cutoff fix (commit `96f6a73`). 7 rows repo-wide retain a dominant EDTA-window peak
  (6 corpus, 1 MTBLS563) — documented residual.
- Checkpoints (all pretrained on the v3 cleaned corpus, 2026-07-25):
  masking `..._085527_..._best.pth`, jigsaw `multibin_20260725_085608_best.pth`,
  joint `joint_ssl_20260725_085627_best.pth`.
- Scripts: `code/analysis/probe_logreg_advantage.py`,
  `code/analysis/linear_probe_frozen_embeddings.py`,
  `code/plotting/plot_all_datasets_summary.py`,
  `code/plotting/plot_brc_t2d_fold_variability.py`,
  `code/plotting/plot_logreg_advantage_probe.py`,
  `code/plotting/plot_linear_probe_vs_head.py`,
  `code/evaluation/ssl_linear_probe_eval.py`,
  `code/plotting/plot_pretraining_gain.py`,
  `code/analysis/compare_patch_sizes.py`,
  `code/plotting/plot_patch_size_experiment.py`,
  `code/analysis/summarize_exp7_factorial.py`,
  `code/plotting/plot_exp7_factorial.py`,
  `code/analysis/summarize_exp7_replicates.py`,
  `code/plotting/plot_exp7_replicates.py`,
  `code/tests/verify_top_peak_loss.py`,
  `code/preprocessing/build_corpus_subset.py`,
  `code/analysis/summarize_exp8_corpus_subset.py`,
  `code/plotting/plot_exp8_corpus_subset.py`,
  `code/analysis/summarize_exp13_size_sweep.py`,
  `code/plotting/plot_exp13_size_sweep.py`,
  `code/analysis/sweep_pooling_jigsaw_joint.py`,
  `code/analysis/hybrid_features.py`,
  `code/evaluation/fewshot_benchmark.py`,
  `code/analysis/summarize_fewshot_masking_vs_classical.py`,
  `code/plotting/plot_fewshot_masking_vs_classical.py`,
  `code/plotting/plot_groupmeeting_figures.py`,
  `code/training/run_seed_queue.sh`,
  `code/analysis/summarize_exp15_seed_study.py`,
  `code/plotting/plot_exp15_seed_study.py`.
- Figures: `results/plots/all_datasets_summary_v4/fig1..fig18`; deck figures in `docs/gm_figures/`.
- **Reproducibility (§7b, corrected 2026-08-09):** `--seed` DOES make training reproducible —
  two same-seed runs of the same arm are byte-identical (max|ΔW| = 0.000e+00) once both have
  finished. The earlier claim to the contrary compared against a mid-training checkpoint and is
  retracted. **Measured noise floor (§15, from 5 seeds): single-run sd = 0.045 on the held-out
  mean, up to 0.076 per target (Barth).** The earlier 0.020/0.035 figures came from three runs
  whose per-target errors cancelled and understate the noise by more than 2×. A single-run
  difference below 0.09 (2 sd) is not an effect.
- **Stale-checkpoint hazard:** three wrong numbers in this project have come from scoring a
  checkpoint whose training run had not finished (ps128 in §5b; the r1/r2 pair in §7b). Always
  confirm "Training completed after" in the run's log before scoring its checkpoint.
- **~~Corpus caveat (§5f)~~ — VOID (§15):** the claimed +0.069 v3-vs-v4 advantage was a sampling
  artifact of a single lucky v3 draw. At n=5 per corpus the difference is −0.0140, inside one
  standard error (0.6 se). The two corpora are interchangeable; the "never mix v3 and v4 checkpoints"
  rule is withdrawn.
- **~~Open question (§8, §13)~~ — CLOSED (§15):** §8 and §13 could not find a mechanism for the
  +0.069 gap because there was no gap. Both experiments were sound and both are kept for the
  record; their inconclusive readings were correct. Row indices from the corpus diff remain at
  `results/analysis/corpus_v3_v4_diff/differing_rows.csv` if the 164 rows are ever of interest
  for another reason.
- **Pooling caveat (§14):** the position-preserving pooling win (§5c, masking-only) extends to
  jigsaw (+0.079 mean) and joint (+0.049 mean), on existing checkpoints with no retraining.
  Hybrid features (embedding + binned areas) are refuted — worse than the better solo feature
  on nearly every target for every family.
