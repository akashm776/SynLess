# Recorded CLIP/CUB pilot v1

This lightweight archive contains the completed user-provided A100 run's report,
per-seed/aggregate metrics, runtime manifest, and per-state selection summaries
and BIDS diagnostics. These are observed results, not regenerated toy output.

- Source implementation: [`7d64c22c2f81e9ce203e6bd8227a6805ac7c735c`](https://github.com/akashm776/SynLess/commit/7d64c22c2f81e9ce203e6bd8227a6805ac7c735c).
- Backend: CLIP ViT-B/32 on CUB, not an LLM.
- Seeds: 789, 2026, 31415; starting updates 100/500; continuation horizon 20.
- Hardware: NVIDIA A100-SXM4-40GB; fp32 and TF32 disabled.
- [Original generated report](REPORT.md), [full numerical summary](summary.json),
  [runtime/configuration manifest](run.json).
- Each `seed_*/step_*` directory contains `selection_summary.json` and
  `bids_diagnostics.json`. Large checkpoints, images, raw gradient stores, and
  the complete resumable run directory are intentionally not included.

The reporting pool has 512 images. Selection validation is disjoint within this
run, but CUB was used in earlier SCSS work. Random repeats are not extra training
seeds. Negative loss deltas mean helpful relative to matched native continuation;
R@1 deltas are fractions. The JSON flag `research_evidence` distinguishes the
real-data run from toy checks; it is not a claim of statistical significance.

Use the original source revision and configuration for replay. Do not resume
from these report-only files: a resume also requires the original checkpoints
and run artifacts. See [methodology](../../docs/CLIP_PILOT.md) and
[interpretation](../../docs/CLIP_PILOT_RESULTS.md).
