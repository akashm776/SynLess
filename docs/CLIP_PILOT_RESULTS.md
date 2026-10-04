# CLIP/CUB pilot: findings and boundaries

[Project overview](../README.md) · [Method and reproduction](CLIP_PILOT.md) ·
[Full report](../results/clip_cub_pilot_v1/REPORT.md) ·
[Per-seed metrics](../results/clip_cub_pilot_v1/summary.json)

## What ran

CLIP ViT-B/32 on CUB, three seeds (789, 2026, 31415), native starting states at
updates 100 and 500, and twenty matched continuation updates. Fifteen arms per
state/seed give ninety branches. Auxiliary arms have 160 candidate exposures;
native has none. The reporting pool contains 512 images with one canonical
caption each, separate from the new run's 256-image selection-validation pool.

These are subset retrieval results, not full-CUB evaluation or LLM instruction
tuning. The actual run used an A100-SXM4-40GB, fp32, with TF32 disabled; the
[manifest](../results/clip_cub_pilot_v1/run.json) records the environment/config.

## Main observations

Mean loss delta is treatment minus a native continuation restored from the same
model and optimizer state. Negative is helpful. See the full report for SDs,
all fifteen arms, and both retrieval directions.

| Arm | State 100 | State 500 |
|---|---:|---:|
| BIDS | -0.00141933 | -0.000282767 |
| BIDS without normalization | -0.00128212 | -0.000404092 |
| Normalized instance max | -0.00119861 | -0.000557592 |
| OT | -0.000860157 | -0.000497895 |
| Uniform | -0.000469419 | -0.000178359 |
| Top LESS-style | -0.0000048702 | +0.00116703 |
| Bottom LESS-style | +0.000391019 | -0.00158757 |
| Hardest real | +0.000437712 | +0.000433209 |

1. **Synthetic candidates were selected.** BIDS OT/uniform enrichment was
   1.500/1.438 at state 100 and 1.438/1.469 at state 500. Hardest-real enrichment
   was 0.062/0.094. The candidate pool has equal construction prevalence;
   enrichment 1 means no preference. These are selection preferences, not gains.
2. **Early BIDS gains did not persist at the same magnitude.** Its loss delta
   relative to the within-seed mean of three random arms was approximately
   -0.001222 at state 100, but only -0.0000886 at state 500. At 500 its mean
   text-to-image loss worsened (+0.00103637) while image-to-text improved
   (-0.00160193). An aggregate loss can conceal a directional tradeoff.
3. **LESS-style rank was not reliable evidence of utility.** At state 500,
   top_less had worse paired loss than bottom_less in all three seeds. This is a
   result of this contextual CLIP adaptation, not a refutation of the LLM paper.
4. **OT outperformed uniform in these matched comparisons.** Paired OT-minus-
   uniform loss was negative in all six seed/state pairs. Both arms use identical
   query/source supports in this pilot, isolating their weight construction.
   This is encouraging but too narrow to establish general OT superiority.
5. **Retrieval changes were small.** Early BIDS mean text-to-image R@1 rose
   0.001953125, one net additional correct query per 512-image seed evaluation.
   Its mean image-to-text R@1 change was zero; later BIDS mean changes were zero
   in both directions. Zero mean can hide opposing seed-level changes.

## Important comparison limits

- BIDS and less_head_* share the exact projection-head attribution matrix;
  top_less uses all trainable parameters. Comparing those headline names alone
  does not isolate selection algorithm quality.
- Training selectors have equal per-context capacities; this is an adaptation
  beyond unconstrained paper selection. Unconstrained selections are diagnostic.
- Random repeats share a training seed and are not independent replications.
  Three seeds and many arms do not support a definitive significance claim.
- Prior SCSS work used CUB. The reporting split is disjoint from new-run
  selection, but it is not historically untouched.
- Hardest-real adds an auxiliary denominator contribution for an existing
  negative. Its harm does not mean ordinary real-data training is harmful.
- BIDS may benefit partly by rejecting that harmful construction. A useful next
  control is random selection matched to its per-context construction composition.

## Consequences for the LLM design

The new [LLM protocol](LLM_GENERATOR_PROTOCOL.md) therefore requires real matched
continuations, multiple learner states, and a comparison against meta-learned
weighting of existing losses. It does not assume that a high influence score,
an OT label, or a synthetic mixture guarantees useful learning.

The archived metrics support this CLIP pilot only. No learned generator or LLM
was trained in this run, and no CLIP-to-LLM transfer result has been established.
