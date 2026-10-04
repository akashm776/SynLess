# Implementation validation

Validated locally on CPU with Python 3.12, PyTorch 2.9.1, NumPy 2.3.5,
Transformers 4.57.3 and Torchvision 0.24.1. GPU execution is pending the Colab
A100 run. The local CUB loader test uses a mocked dataset; it does not download
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
