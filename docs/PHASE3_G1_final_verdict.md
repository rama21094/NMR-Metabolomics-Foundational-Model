# Gate G1, completed: adaptation does not rescue the pretrained model

## What was missing

G1 was decided in Phase 3 on frozen embeddings alone. The research plan (§3.1)
specified four transfer modes in increasing adaptation; only the linear probe had
been run, so the verdict was a claim about **representations**, not about what
the architecture could do if adapted. That was flagged as untested on slide 6 of
the 2026-09-18 deck. It is now tested.

## Design

Every arm is read out by the **same** Phase 3 probe -- StandardScaler + balanced
logistic regression on pooled embeddings -- on the **same** episodes, with the
same stable crc32 seeding. The arms differ only in whether, and how deeply, the
encoder was adapted on the support set first:

| arm | trained on the support set |
|---|---|
| `head` | nothing; reproduces the Phase 3 probe by construction |
| `last_block` | final transformer block |
| `full` | entire encoder |

Fine-tuning was given its best honest chance: 75/25 support split with the best
validation epoch restored, balanced class weights, gradient clipping, and a
random-init control at every mode.

6 targets x 4 label budgets x 3 modes x 3 seeds x 12 episodes, plus random-init
controls. 288 records.

## Result 1 — adaptation buys nothing

Against the frozen probe, across all 24 target x budget cells:

| mode | mean Δ | sd | better in | best | worst |
|---|---|---|---|---|---|
| `last_block` | **−0.0087** | 0.0131 | 3 / 24 | +0.019 | −0.041 |
| `full` | **−0.0104** | 0.0205 | 7 / 24 | +0.019 | −0.076 |

Both are negative on average and lose more often than they win. The largest gain
anywhere is +0.019, below the project's 0.045 noise floor; the largest loss is
−0.076. Deeper adaptation is worse than shallower, which is the signature of
overfitting, and expected: the cohorts hold 40–142 samples against ~1.7 M encoder
parameters.

## Result 2 — G1 stands, and is now a statement about the architecture

Classical ML leads in **24 of 24** cells, mean gap **+0.055**. There is no cell,
at any label budget including k=5, where any transfer mode of any seed beats it.

The earlier verdict cannot now be explained away as an artefact of freezing the
backbone.

## Result 3 — pretraining still works

Pretrained minus random-init, same architecture, same episodes, across 72 cells:
**+0.057 mean, positive in 64 of 72**. This reproduces the Phase 3 finding
through an independent code path.

So the pretext task teaches the encoder something real and transferable. It is
simply worth less than what a shrinkage LDA extracts from 500 spectral bins, and
no amount of adaptation on a small cohort recovers the difference.

## A design error worth recording

The first version of this experiment trained a linear head by gradient descent as
the `head` control. It scored 0.42–0.63 against the Phase 3 probe's 0.67 on the
same cohort, because 60 full-batch steps barely move a linear head. Neither a
longer schedule, a higher learning rate, nor a LayerNorm closed the gap, and at 3
episodes every setting sat within the ~0.04 standard error of the others.

Had that version been run to completion, `last_block` and `full` would have
"beaten" the head arm by 0.06–0.14 and the conclusion would have been the
opposite of the correct one -- not because fine-tuning helped, but because the
control was crippled by an optimiser difference. The shared-readout design
removes that confound: `head` reproduces the probe exactly, by construction.

## Consequence for the tree

G1 is now answered completely: **classical ML is the better method for these
tasks at this data scale**, across all four transfer modes the plan specified.
Combined with G2 (volume saturates below 2,000 spectra; diversity helps but this
corpus cannot exceed an effective ~7 studies), the remaining live question is G3,
whose prior the G2 result already lowers.
