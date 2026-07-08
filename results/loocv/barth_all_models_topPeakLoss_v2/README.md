# Barth LOOCV — Joint SSL, Top-Peak-Loss Architecture (v2)

Leave-one-out cross-validation results for the joint masked-reconstruction +
multibin-jigsaw SSL model on the Barth Syndrome dataset (37 samples, 14
Control / 23 Case), evaluated with a checkpoint trained on a revised
architecture and pretraining loss.

## What's different about this checkpoint

Checkpoint: `models/joint_ssl/joint_ssl_20260707_085510/joint_ssl_20260707_085510_best.pth`
(see `results/joint_ssl/joint_ssl_20260707_085510/config.json` for the full
training config.)

Relative to the original joint SSL model, this run combines three changes in
[`code/training/train_joint_ssl.py`](../../../code/training/train_joint_ssl.py):

1. **Bottlenecked task embedding.** The masked-vs-jigsaw task signal is
   projected through an 8-dim bottleneck (`task_embed_dim=8`) before being
   added to the encoder, instead of a full-width (192-dim) embedding. A
   full-width task embedding gave the encoder an easy way to fork into two
   nearly-disjoint per-task sub-networks rather than sharing representation
   across both objectives.
2. **Top-peak-fraction reconstruction loss.** Masked-reconstruction loss is
   now computed only over the top `peak_top_fraction=0.175` (~17.5%) of bins
   by magnitude per spectrum, rather than a continuous magnitude-based weight
   over all bins. The previous continuous weighting was dominated by a single
   recurring high-magnitude bin per spectrum (likely a residual solvent/EDTA
   artifact) and left the reconstruction loss trivially satisfiable by
   predicting near-zero baseline everywhere; restricting supervision to each
   spectrum's own top bins forces the model to actually learn peak structure.
3. **Batch-split jigsaw bin sizes.** Each training step now assigns a
   randomly-chosen subset of the batch to each jigsaw bin size (256/512/1024/
   2048), rather than running the full batch through all four bin sizes every
   step. This balances the jigsaw objective's gradient volume against the
   masked-reconstruction objective's single forward pass per step.

At evaluation time (`code/evaluation/barth_all_models_loocv.py`), the
joint-SSL feature extractor also pools the masked-reconstruction task's
embedding alongside the jigsaw embeddings (`joint_include_masked_task=true`
in `run_config.json`) — previously, downstream classifiers only ever saw
embeddings computed under the jigsaw task id, so the masked-reconstruction
objective never influenced the features actually used for classification.

## How this was generated

```bash
python code/evaluation/barth_all_models_loocv.py \
  --families joint_ssl \
  --joint-checkpoint joint_ssl=models/joint_ssl/joint_ssl_20260707_085510/joint_ssl_20260707_085510_best.pth \
  --output-dir results/loocv/barth_all_models_topPeakLoss_v2
```

Full CLI args and resolved paths are recorded in `run_config.json`.

Only the `joint_ssl` family was run here (frozen backbone + unfreezing the
last 1/2/3 transformer layers). For the classical/masking/jigsaw baselines
on the same dataset, see `results/loocv/barth_all_models/summary.csv`.

## Files

| File | Contents |
|---|---|
| `summary.csv` | Per-model LOOCV metrics (accuracy, balanced accuracy, macro/weighted F1, ROC-AUC, PR-AUC, precision/recall, confusion counts) for each fine-tuning mode |
| `oof_predictions.csv` | Out-of-fold prediction for every sample and every fine-tuning mode, with class probabilities |
| `joint_ssl_<mode>_oof_pred.npy` | Out-of-fold predicted class indices, one array per fine-tuning mode |
| `joint_ssl_<mode>_oof_prob.npy` | Out-of-fold predicted class probabilities, one array per fine-tuning mode |
| `run_config.json` | Full CLI args and resolved dataset/checkpoint metadata for this eval run |

## Results

| Model | Balanced Accuracy | Macro F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| joint_ssl_frozen | 0.613 | 0.612 | 0.755 | 0.837 |
| joint_ssl_unfreeze_last_1 | 0.685 | 0.692 | 0.764 | 0.838 |
| joint_ssl_unfreeze_last_2 | 0.649 | 0.653 | 0.823 | 0.906 |
| **joint_ssl_unfreeze_last_3** | **0.707** | **0.716** | **0.879** | **0.928** |

### Comparison with the original (pre-modification) joint SSL checkpoint

| Model | Balanced Accuracy (orig → v2) | ROC-AUC (orig → v2) |
|---|---|---|
| joint_ssl_frozen | 0.685 → 0.613 | 0.696 → 0.755 |
| joint_ssl_unfreeze_last_1 | 0.663 → 0.685 | 0.708 → 0.764 |
| joint_ssl_unfreeze_last_2 | 0.613 → 0.649 | 0.696 → 0.823 |
| joint_ssl_unfreeze_last_3 | 0.671 → **0.707** | 0.680 → **0.879** |

("orig" = `results/loocv/barth_all_models/summary.csv`, the checkpoint trained
before all three modifications above.)

Best balanced accuracy across fine-tuning modes improves from 0.685 (orig,
frozen) to 0.707 (v2, unfreeze_last_3), and ROC-AUC improves substantially
across every fine-tuning mode — most notably from 0.680 to 0.879 at
`unfreeze_last_3`. `joint_ssl_frozen` is the one mode that regresses on
balanced accuracy (0.685 → 0.613), suggesting the new architecture benefits
more from fine-tuning the backbone than the original did.
