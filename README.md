# SynLess

**Influence-based selection of synthetic negatives for CLIP/CUB.**

[![CPU science checks](https://github.com/akashm776/SynLess/actions/workflows/tests.yml/badge.svg)](https://github.com/akashm776/SynLess/actions/workflows/tests.yml)

## Language-model work has moved to SYNAPSE

The learned hidden-state generator experiment now lives in
**[SYNAPSE — State-Dependent Synthetic Supervision for Language Models](https://github.com/akashm776/SYNAPSE)**.
That repository contains the LLM code, tests, A100 notebook, protocol, and completed
Qwen/GSM8K results. It is the active home for future language-model development.

- [SYNAPSE results and interpretation](https://github.com/akashm776/SYNAPSE/blob/main/docs/QWEN_PILOT_RESULTS.md)
- [New-run A100 notebook](https://colab.research.google.com/github/akashm776/SYNAPSE/blob/main/colabs/SYNAPSE_A100.ipynb)
- [Historical run resume and migration](https://github.com/akashm776/SYNAPSE/blob/main/docs/MIGRATION.md)

The Qwen pilot is complete, not pending. It did not establish an overall
fine-tuning gain and did not pass its development gate to Llama. Uniform mixing
slightly improved mean test loss but reduced accuracy; learned barycenters
worsened primary loss. See SYNAPSE for the complete results and limitations.

The `synless/llm` code and old LLM documentation/notebook remain here as a
**historical snapshot**, not a second active implementation. Existing Drive runs
must retain their original source/environment. The completed `llm_qwen_v2` used
commit `34f8060c40f5842910ef0dda3d5281ceab0a1f4a`; do not change that run's manifest
or move its Drive directory just because the project has a new name.

## CLIP/CUB research question

Do LESS-style and BIDS influence selectors identify useful synthetic negatives
constructed using the OT/uniform rules explored in
[SCSS](https://github.com/akashm776/SCSS)? This is an adaptation to contrastive
learning, not a reproduction of the original LLM instruction-selection benchmarks.

[CLIP method and setup](docs/CLIP_PILOT.md) · [Results and caveats](docs/CLIP_PILOT_RESULTS.md)

## Recorded pilot

Three training seeds; starting states 100/500; twenty matched continuation updates;
canonical-caption retrieval on a fixed 512-image reporting subset.

| Arm | Mean loss delta at 100 | Mean loss delta at 500 |
|---|---:|---:|
| BIDS adaptation | −0.00141933 | −0.000282767 |
| OT-only | −0.000860157 | −0.000497895 |
| Uniform-only | −0.000469419 | −0.000178359 |
| Top LESS-style | −0.0000048702 | +0.00116703 |
| Bottom LESS-style | +0.000391019 | −0.00158757 |

Treatment minus matched native loss; negative is helpful. These are selected
comparisons: [all arms](results/clip_cub_pilot_v1/REPORT.md),
[per-seed metrics](results/clip_cub_pilot_v1/summary.json).

Selectors favored synthetic candidates, but preference did not reliably imply
downstream benefit. Early BIDS improvement weakened later; top LESS-style
selection became harmful on average. OT beat uniform on paired loss in all six
seed/state comparisons in this setup.

Limits include three seeds, tiny retrieval changes, prior CUB use in SCSS, and
different attribution subspaces for full-space LESS and head-space BIDS. There is
no established full-dataset retrieval gain, general BIDS-versus-LESS ranking, or
LLM transfer claim.

## Run CLIP

[Open CLIP/CUB Colab A100 notebook](https://colab.research.google.com/github/akashm776/SynLess/blob/main/colabs/SynLess_A100.ipynb)

```bash
python -m pip install -e '.[test]'
python -m pytest -q tests/test_science.py tests/test_bids.py
python scripts/check_research_docs.py
synless run --config configs/smoke.json --output runs/smoke
```

See the [CLIP guide](docs/CLIP_PILOT.md) for GPU dependencies, data, and resume.
Use the [recorded source revision](results/clip_cub_pilot_v1/README.md) for pilot
reproduction. Toy runs check implementation, not scientific effectiveness.

## Research lineage

- [SCSS](https://github.com/akashm776/SCSS): state-conditioned synthetic supervision.
- [LESS](https://arxiv.org/html/2402.04333v2): influential instruction-data selection.
- [BIDS](https://aclanthology.org/2025.findings-emnlp.373/): balanced influence selection.
- [SYNAPSE](https://github.com/akashm776/SYNAPSE): the language-model investigation.
