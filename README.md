# SynLess

**From influence-selected synthetic negatives to learned generators of internal supervision.**

[![CPU science checks](https://github.com/akashm776/SynLess/actions/workflows/tests.yml/badge.svg)](https://github.com/akashm776/SynLess/actions/workflows/tests.yml)

[Run the LLM pilot](docs/LLM_RUN.md) · [LLM protocol](docs/LLM_GENERATOR_PROTOCOL.md) · [Research roadmap](docs/ROADMAP.md) · [CLIP pilot results](docs/CLIP_PILOT_RESULTS.md)

## Research question

Can we learn a reusable function that constructs helpful hidden-state negatives
for instruction tuning—and demonstrate that it does more than select or reweight
existing negatives?

SynLess began by testing whether [LESS](https://github.com/princeton-nlp/LESS)-style
influence and [BIDS](https://aclanthology.org/2025.findings-emnlp.373/) select the
OT/uniform negatives explored in [SCSS](https://github.com/akashm776/SCSS). That
CLIP/CUB pilot is complete. The next experiment moves to **LLM supervised
fine-tuning with a learned, layer-specific auxiliary loss**.

## Two tracks, explicit status

| Track | Status | Entry point |
|---|---|---|
| CLIP/CUB synthetic-negative selection | Implemented; three-seed A100 pilot recorded | [Method and runnable notebook](docs/CLIP_PILOT.md) |
| Qwen2.5-1.5B learned-generator pilot | Implemented; offline tiny-model checks pass; real A100 run pending | [LLM run guide](docs/LLM_RUN.md) |
| Llama-3-8B method replication | Planned after Qwen development and resource gates | [Stage definitions](docs/LLM_GENERATOR_PROTOCOL.md#2-stages-and-model-choices) |
| Genuine learned OT / multiple layers / diverse capabilities | Follow-up research, not an existing feature | [Roadmap](docs/ROADMAP.md) |

The new `synless-llm` runner and `SynLess_LLM_A100.ipynb` are separate from the
original CLIP workflow. All seven generator/control arms are implemented.
**No real-model LLM gain or A100 memory fit has yet been established.**

## LLM experiment

1. **Start small:** Qwen2.5-1.5B **base**, LoRA SFT on answer-only GSM8K, one middle
   decoder block. Qwen is our feasibility choice, not a model used by LESS/BIDS.
2. **Learn a function:** a small shared scorer chooses barycentric weights over
   four wrong-answer hidden states. Train its parameters through a differentiable
   learner update using a disjoint meta-training set.
3. **Test reuse:** freeze the generator, then train fresh learners on disjoint
   examples. Compare matched continuations at early/late states and 1/5/20 steps.
4. **Isolate generation:** compare against native SFT, uniform/fixed mixtures,
   hardest existing negatives, a global mixture, and equally meta-trained
   **weighting of existing negative losses**.
5. **Replicate on Llama:** only after development-set and A100 resource checks.
   Independently fitting a Llama generator tests the method; reusing the frozen
   Qwen function is a different, stronger transfer experiment.

The generated object is a **hidden-state mixture**, not new text. The first
generator is a barycentric-weight function, **not yet a learned OT map**. A
single math task does not test BIDS's balanced multi-capability claim.
The [full protocol](docs/LLM_GENERATOR_PROTOCOL.md) specifies splits, losses,
optimizer state, seed pairing, controls, preflight checks, and failure criteria.

[Open the LLM A100 notebook](https://colab.research.google.com/github/akashm776/SynLess/blob/main/colabs/SynLess_LLM_A100.ipynb)

Run **preflight → development training → official-test report** in separate
cells. The preflight verifies live meta-gradients and measures resources on your
A100; training refuses to start without a matching successful check. Official
test reporting requires a locked development comparator and unchanged checkpoint
hashes. See the [run guide](docs/LLM_RUN.md) for resume rules and compute limits.

## What the CLIP pilot actually found

Three training seeds; starting states at updates 100/500; twenty matched
continuation updates; canonical-caption retrieval on a fixed 512-image subset.

| Arm | Mean loss delta at 100 | Mean loss delta at 500 |
|---|---:|---:|
| BIDS adaptation | -0.00141933 | -0.000282767 |
| OT-only | -0.000860157 | -0.000497895 |
| Uniform-only | -0.000469419 | -0.000178359 |
| Top LESS-style | -0.0000048702 | +0.00116703 |
| Bottom LESS-style | +0.000391019 | -0.00158757 |

Deltas are treatment minus matched native continuation; negative is helpful.
These are selected comparisons, not the complete arm table:
[all arms and seed variability](results/clip_cub_pilot_v1/REPORT.md),
[per-seed metrics](results/clip_cub_pilot_v1/summary.json).

Selectors favored synthetic candidates, but selection preference did not
reliably imply downstream benefit. Early BIDS improvement weakened at the later
state; top LESS-style selection became harmful on average. OT beat uniform on
paired loss in all six seed/state comparisons within this setup.

**Limits:** three seeds, tiny retrieval changes, CUB's prior use in the broader
SCSS research context, and different attribution subspaces
for full-space LESS versus head-space BIDS. These results do not establish
full-dataset retrieval gains, a general BIDS-vs-LESS ranking, or transfer to LLMs.
Read the [interpretation and caveats](docs/CLIP_PILOT_RESULTS.md).

## Run the existing CLIP experiment

[Open CLIP/CUB Colab A100 notebook](https://colab.research.google.com/github/akashm776/SynLess/blob/main/colabs/SynLess_A100.ipynb)

Choose an A100 runtime and follow the notebook. It saves resumable outputs to
Drive. To replay the published pilot, use the recorded source revision in the
[result provenance](results/clip_cub_pilot_v1/README.md); code/config changes
must not be mixed into an existing run directory.

For CPU implementation checks, from the repository root:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python scripts/check_research_docs.py
synless run --config configs/smoke.json --output runs/smoke
```

Toy output is not scientific evidence. Full setup, selector definitions,
artifacts, and resume rules are in the [CLIP guide](docs/CLIP_PILOT.md).
See [validation notes](docs/VALIDATION.md) for the local test environment.

## Repository map

```text
docs/
  LLM_GENERATOR_PROTOCOL.md   Scientific specification and boundaries
  LLM_RUN.md                  LLM setup, commands, resume, and resource checks
  ROADMAP.md                 Implementation gates and deferred hypotheses
  CLIP_PILOT.md              Existing CLIP workflow and methodology
  CLIP_PILOT_RESULTS.md      Interpretation of the completed pilot
  VALIDATION.md              Implementation-check coverage
results/clip_cub_pilot_v1/    Lightweight recorded results; no model weights
synless/                    CLIP construction, scoring, training, reporting
synless/llm/                Separate LLM data, LoRA, generators, meta-learning, reporting
colabs/SynLess_A100.ipynb    Existing CLIP-only A100 entry point
colabs/SynLess_LLM_A100.ipynb New LLM A100 entry point
configs/                    Separate CLIP, Qwen, and offline smoke configurations
tests/                      CLIP and LLM implementation checks
```

## Research lineage

- [SCSS](https://github.com/akashm776/SCSS): OT barycentric and uniform contrastive negatives.
- [LESS](https://arxiv.org/html/2402.04333v2): influential instruction-data selection;
  Llama-2-7B/13B and Mistral-7B.
- [BIDS](https://aclanthology.org/2025.findings-emnlp.373/): normalized, iterative
  influence selection for balanced capabilities; Llama-3-8B and Mistral-7B-v0.3.

SynLess's CLIP implementation adapts these selection ideas. The learned
LLM generator is a new hypothesis; neither paper establishes its effectiveness.
