# Implementation validation

Validated locally on CPU with Python 3.12, PyTorch 2.9.1, NumPy 2.3.5,
Transformers 4.57.3 and Torchvision 0.24.1. The subsequent three-seed Colab A100
CLIP/CUB run is complete; see [pilot results](CLIP_PILOT_RESULTS.md) and the
[recorded runtime manifest](../results/clip_cub_pilot_v1/run.json). This does not
validate the new LLM generator's real-model performance or GPU memory fit.
The local CUB loader test uses a mocked dataset; it does not download
or validate the live remote dataset. Colab pins Datasets 2.21.0 for the SCSS
dataset loader; the local environment has Datasets 4.8.4.

Checks cover:

- Synthetic auxiliary equals the additional log-partition term; constituent-image
  gradients remain live and the auxiliary cannot directly update CLIP scale.
- OT/uniform share support; positive leakage and changed context IDs are rejected.
- Analytic AdamW delta matches actual PyTorch steps with warm moments, clipping,
  distinct learning rates, weight decay, and scale clamping.
- LESS feature matches the released moment/epsilon convention.
- Influence scoring does not mutate model weights or optimizer state.
- Marginal prediction agrees with actual paired updates to first order.
- Restoring saved optimizer state reproduces matched branches.
- BIDS follows the greedy selected-mean rule, is invariant to positive column
  affine transformations after normalization, excludes constant columns, and
  respects explicit context capacities without repeated selections.
- Directional instance losses average to the symmetric CLIP objective.
- A tiny offline CLIP model verifies the live-encoder adapter and trainable policy.
- CUB grouping/splitting is tested with duplicate image rows and canonical captions.
- End-to-end resume reuses checkpoints and reproduces the saved summary; changed
  configurations are rejected.

An additional direct comparison against SCSS commit
`40153cff55d050139f97d75713822631e1343d1d` passed numerical parity checks for the
OT support mask, normalized barycentric weights, and relative-denominator loss.
The A100 notebook's JSON and all code-cell syntax were also validated.

The three-seed, two-state toy run exercises every selection and continuation arm.
Its numbers are plumbing evidence only, not evidence that synthetic negatives
help CLIP or that BIDS outperforms LESS.

The local Python installation's native `readline` extension crashes during pytest
startup. Tests were run with plugin autoload disabled and `readline` marked
unavailable in the test process; Torch and the tests themselves were unchanged.
The standard pytest command is retained for Linux CI and Colab.

## LLM implementation checks

Local verification after the LLM addition: **30 tests passed**, including 11
LLM tests. A separate complete offline smoke run exercised all seven arms, two
starting states, and two reporting horizons (28 paired reporting records), plus
development locking and report generation. Both Colab notebooks pass JSON and
Python-cell syntax checks. These are not real-model experimental results.

`tests/test_llm.py` independently checks numeric parsing, distractors, disjoint
partitions, boundary-safe token masks, deterministic data order, generator
permutation behavior, live constituent gradients, and matched control identities.
Functional AdamW is compared to real PyTorch steps with warm moments, distinct
learning rates, clipping, decay, and unused parameters. An fp64 mixed derivative
is checked with central finite differences, including a zero-moment edge case.
The tiny-transformer barycenter meta-gradient also matches a true-fp64 reference
finite-difference check. Qwen's RMSNorm and attention softmax explicitly downcast
to fp32 even after `.double()`; the test substitutes mathematically equivalent
fp64 operations only inside that fixture to avoid rounding-noise comparisons.
The production model is unchanged. A frozen-bf16-base/fp32-adapter CPU test verifies finite
meta-gradient connectivity in mixed precision; it is not an A100 resource test.

A tiny randomly initialized Qwen2 decoder verifies raw-layer capture, agreement
with full-logit masked CE, generator meta-gradient connectivity, non-mutating
virtual steps, exact learner restore, interrupted pipeline resume, and the
development-before-test gate. This fixture downloads no weights or dataset.
Resume coverage includes interruptions inside meta-training and continuation
training. Shared parent optimizer tensors must remain unchanged across restores;
loading optimizer states takes a deep copy to prevent same-device tensor aliasing.
It cannot demonstrate Qwen2.5-1.5B/GSM8K gains or A100 feasibility.

The [LLM notebook](../colabs/SynLess_LLM_A100.ipynb) invokes the separate
real-runtime preflight before permitting substantive training. See
[run instructions](LLM_RUN.md) for resource checks and limitations.
