# SynLess

Do SCSS synthetic negatives get selected by influence-based methods, and do those
selections improve held-out retrieval learning?

This repository provides a **Colab A100 pilot**, adapting
[LESS](https://github.com/princeton-nlp/LESS) and
[BIDS](https://aclanthology.org/2025.findings-emnlp.373/) to
[SCSS](https://github.com/akashm776/SCSS)'s contrastive negative constructions.
It is not a reproduction of either instruction-tuning paper.

**Status:** implementation and CPU smoke tests are available. Real CLIP/CUB
results require running the A100 notebook. Toy results are not research evidence.

[Open the A100 notebook](https://colab.research.google.com/github/akashm776/SynLess/blob/main/colabs/SynLess_A100.ipynb)

## Run on Colab

Select an **A100** runtime, open the notebook, and run the cells in order. The
notebook installs the experiment environment, mounts Drive, verifies the runtime,
and launches the pilot. Use a fresh runtime if Torch was already imported before
installation. Outputs live under `MyDrive/SynLess/runs/pilot_v1`.

Rerun the launch cell after a disconnect. Checkpoints, completed scoring contexts,
and completed continuation arms are reused. An interrupted arm restarts from its
parent checkpoint. Configuration/source/package changes require a new output
folder; they cannot silently mix with an existing run.

The default is three training seeds, native warmup states at updates 100 and 500,
four fixed 32-pair training contexts, 384 candidates per seed, 256 selection
validation images, and a 512-image official-test reporting subset. Each candidate
context contains OT, uniform-top-8, and hardest-real choices for every query.
Each continuation uses 20 updates and eight selected candidates per context.
This is a feasibility pilot, not a full training benchmark.

The dataset and pretrained model download automatically. No SCSS clone or private
checkpoint is needed. This is a fresh experiment with the SCSS construction rules;
it does not load historical SCSS optimizer states or replay historical schedules.
Constant learning rates and deterministic fp32 forwards are intentional here.

## Local verification

```bash
python -m pip install -e '.[test]'
python -m pytest -q
synless run --config configs/smoke.json --output runs/smoke
synless summarize --output runs/smoke
```

The smoke run exercises all selectors, multiple model states, matched training,
reporting, and resume. A tiny random CLIP adapter test runs when Transformers is
installed, without downloading pretrained weights.

## What is selected?

A candidate is `(fixed context, query, source images, mixing coefficients)`, not a
detached embedding. At the first scoring checkpoint we save support and weights,
then hold that recipe fixed across subsequent states. Images and text are
re-encoded live, and gradients flow through every source image. Thus this tests
the same candidate recipes at different model states, not a freshly recomputed
top-k/OT plan at each state. The hardest-real identity is also frozen.

`ot` and `uniform` use identical top-k support and exclude the matched image. OT
reproduces SCSS's `historical_sparse_ot` solver, including its dense numerical
floor and final support mask. Uniform means equal weights on the same support,
not a random negative. `hardest_real` adds another denominator contribution for
an existing real negative, matching the SCSS comparator.

Each candidate uses the relative-denominator auxiliary loss:

```
softplus(synthetic_logit - logsumexp(native_row_logits))
```

The auxiliary uses a detached CLIP scale; the native symmetric loss still trains
the scale. Multiple chosen candidates for one query contribute separate auxiliary
losses, averaged over the selected budget. They do not form one jointly expanded
denominator. Scoring a single candidate uses `alpha / selected_per_batch`, its
coefficient in that average. Candidate interactions can therefore invalidate
individual rankings; actual continuation experiments are essential.

To replay an exported bank, add `"candidate_banks": {"789": "/path/candidates.json"}`
to a copied configuration. Its contexts must match the run exactly. The bank
format includes query IDs, source IDs, support indices, weights, and a context
hash; see a smoke run's `seed_1/candidates.json`. Importing old detached embedding
tensors cannot recover their original training gradients.

## Scores and selectors

| Name | Meaning |
|---|---|
| `raw_aux_cosine` | Exact auxiliary gradient vs mean selection-validation gradient |
| `less_aux_cosine` | Released LESS Adam feature for the weighted auxiliary vs validation gradient |
| `less_total_cosine` | The same feature for native + candidate loss |
| `marginal_predicted_gain` | `-g_val · (delta_native+candidate - delta_native)`, with actual AdamW rules |
| `head_influence` | Candidate-by-validation-instance cosine matrix in the two projection heads |

`less_aux_cosine` matches the released feature convention
`(.9*m + .1*g) / sqrt(.999*v + .001*g² + 1e-8)`. It omits bias correction,
clipping, learning rates, and decay, as in that feature extractor. The marginal
score includes group-specific learning rates, bias correction, clipping, decay,
and scale clamping. Stored moments and the model are never modified by scoring.
All parameters share a fixed layout; absent auxiliary derivatives are zero-filled.

Exact full-trainable-space scores use both projections, final vision/text blocks,
and logit scale. The BIDS matrix uses **both projection heads only** to avoid
storing hundreds of full-model gradients. All `less_head_*` and BIDS selectors
share that identical matrix, without random projection. This is an explicit
parameter-subspace approximation to full-model influence, not LoRA LESS.

Continuation arms include native-only, top/bottom full-space LESS, top marginal
gain, random selections, construction-only controls, and these matrix selectors:

- `less_head_mean`: mean influence across validation instances.
- `less_head_task_max`: mean within each retrieval direction, then maximum across directions.
- `bids`: column normalization followed by greedy selection against the selected mean.
- `bids_no_normalization`: greedy selection without normalization.
- `normalized_instance_max`: normalization without iterative balancing.

All auxiliary arms have the same batch order, number of updates, coefficient,
candidate count, and exposure count. Each training context gets the same budget.
For BIDS this introduces a **context-capacity constraint**, an adaptation beyond
the paper. Unconstrained BIDS and LESS task-max selections are also saved, so
the original selection comparison can be inspected separately from the matched
training experiment. No quota forces equal OT/uniform/real selection.

## BIDS interpretation

BIDS normalizes each validation-instance column across training candidates and
iteratively chooses the candidate maximizing its largest improvement over the
selected set's average influence profile. It addresses imbalance that raw
cross-task influence comparisons can create. See [paper §4 and Algorithm 1](https://aclanthology.org/2025.findings-emnlp.373.pdf).

We use a zero reference for the initially empty set (unspecified in Algorithm 1),
sample standard deviation, and discard constant columns. Ties use stable input
order. Validation columns are individual image-query losses in both retrieval
directions. These are two objectives within CUB, not the paper's five diverse
language capabilities. BIDS does not guarantee positive absolute influence:
normalization is relative, and a fixed budget can select harmful candidates.
This is why we retain the native-only and actual-utility controls.

## Reading results

`REPORT.md` and `summary.json` contain seed-level matched reporting-loss and R@1
deltas. Negative loss delta is helpful. R@1 deltas are fractions, not percentage
points. Reported loss averages fixed within-batch native losses; R@1 uses the full
recorded reporting pool and one canonical caption per image.

For each seed/state:

- `selection_summary.json`: generator selection counts, pool-adjusted enrichment, score distributions.
- `scores_batch_*.json`: individual scores and attribution matrix rows.
- `validation_columns.json`: exact attribution column identities and retrieval direction.
- `bids_diagnostics.json`: influence distributions, task-dominance counts, unconstrained selections.
- `trajectory_scores.json`: LR-weighted accumulation across checkpoints **up to this state**.
- `selections.json`: chosen candidate IDs, fixed before reporting outcomes are read.
- `continuation_*.json`: matched branch results and exposure counts.

Trajectory scores are diagnostic; the first pilot's training selectors use the
current state. Gradient features change with training, and combining early and
later scores can obscure sign reversals. Scoring checkpoints use a constant
schedule; the scalar trajectory weight is the projection learning rate, not a
claim that all parameter groups have the same learning rate.

Selection validation is held out from the new training split at the image level.
The reporting subset is drawn from the official test split and never used for
selection. Prior SCSS work has evaluated CUB, so we do not call this dataset
historically untouched. Random repeats are not additional training seeds. A
selection preference, a short-horizon loss improvement, and a retrieval gain are
different outcomes and must be reported separately.

## Provenance

- SCSS construction reference: commit `40153cff55d050139f97d75713822631e1343d1d`,
  `model/clip_training.py` and `src/clip_geometry_v2_metrics.py`.
- [LESS paper](https://arxiv.org/abs/2402.04333) and
  [released Adam feature](https://github.com/princeton-nlp/LESS/blob/main/less/data_selection/collect_grad_reps.py).
- Dai et al. (2025), [Improving Influence-based Instruction Tuning Data Selection for Balanced Learning of Diverse Capabilities](https://aclanthology.org/2025.findings-emnlp.373/).

Run manifests record source hashes, dependency versions, dataset fingerprints,
actual image/caption splits, and resolved pretrained-model revision. Do not run
two processes against the same output directory.
