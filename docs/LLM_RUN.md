# Run the learned-generator pilot

[Overview](../README.md) · [Scientific protocol](LLM_GENERATOR_PROTOCOL.md)

## Status and scope

The separate LLM implementation is available. It is tested locally with a tiny,
randomly initialized Qwen2 decoder and an offline arithmetic fixture. This is
implementation evidence, not Qwen2.5-1.5B/GSM8K utility evidence. The real-model
A100 preflight and pilot have not been run by the local implementation tests.

Start with the [LLM A100 notebook](https://colab.research.google.com/github/akashm776/SynLess/blob/main/colabs/SynLess_LLM_A100.ipynb).
The older `SynLess_A100.ipynb` still runs CLIP/CUB only.

## Commands

From the repository root:

```bash
python -m pip install -e '.[llm,test]'
python -m pytest -q tests/test_llm.py
python -m synless.llm.cli preflight --config configs/llm_qwen_a100.json --output runs/llm_qwen_v1
python -m synless.llm.cli run --config configs/llm_qwen_a100.json --output runs/llm_qwen_v1
python -m synless.llm.cli report --config configs/llm_qwen_a100.json --output runs/llm_qwen_v1
```

The installed `synless-llm` entry point is equivalent. The `synless` entry point
is unchanged and still serves CLIP. Never pass an LLM config to the CLIP runner.

For offline smoke checks, replace the config with `configs/llm_smoke.json` and
use `runs/llm_smoke`. This downloads no pretrained weights or dataset, but requires
Transformers to instantiate the tiny random Qwen2 architecture. Its output says
`research_evidence: false` and must not be presented as benchmark results.

## Stage separation

1. `preflight`: resolve immutable HF model/data revisions; build and audit fixed
   partitions; validate token boundaries; warm a learner; test all learned arms'
   meta-gradients; measure native, auxiliary, and virtual inner/outer steps on
   long training examples. Check four GB reserved-memory headroom by default.
2. `run`: requires a matching successful preflight. Warm teachers on A, fit
   generators using A/M, warm fresh learners on B, fork all arms, evaluate D.
   No official-test loss or accuracy is computed. Save a development lock with
   the chosen non-generating comparator and hashes of reporting checkpoints.
3. `report`: requires unchanged development artifacts and checkpoint hashes.
   Evaluate the complete eligible official test split, create `summary.json`
   and `REPORT.md`. No generator fitting or automatic Llama launch happens here.

The tokenized official-test records are prepared for split/coverage audits, but
only `report` uses them to measure performance. Training and meta-feedback draw
only from their assigned partitions.

## What is implemented

- Frozen base model with custom fp32 LoRA adapters on q/v projections; rank 8,
  scaling 16, no dropout, no quantization, eager attention, bf16 base on CUDA.
- A raw decoder-block hook, live correct/wrong-answer representations, and
  completion-only vocabulary projection. The masked CE matches a full-logit
  computation; skipping prompt logits saves memory without changing the loss.
- All seven protocol arms, including learned barycenters and a capacity-matched
  meta-learned weighting of existing losses. Their feature inputs are detached;
  the constituent representations remain differentiable.
- Non-mutating functional AdamW with warm moments, clipping, bias correction,
  correct absent-gradient handling, and a zero-safe sqrt derivative. Real
  optimizer parity and fp64 finite differences are tested.
- Independent teacher/meta/learner seed triples, saved optimizer state, native
  baselines, one-step meta-objectives, and trajectories at all recorded horizons.
- Strict numeric exact match: no explanation extraction; malformed or
  token-cap-truncated completions count as incorrect. Numeric-token NLL excludes
  EOS and standalone leading-space tokens.

The four wrong answers are arithmetic offsets, not LLM-generated rationales.
The model does not treat them as correct next-token targets. The auxiliary
mixture is a hidden vector, not decoded text or an OT solution.

## Resume and integrity

Rerun the same command in the same output directory. Completed native states,
meta episodes at the last checkpoint, branches, and evaluations are reused.
An interrupted unsaved interval is replayed; saved metric histories are restored
with their corresponding optimizer checkpoint. `--max-units N` deliberately
pauses after N updates/evaluations and is useful for testing recovery.

The manifest locks the effective config (including defaults), LLM source hash,
package versions, Python version, hardware name, resolved model/data revisions,
and prepared-data hash. Changes require a new output folder. Documentation-only
changes do not affect the LLM source hash. Retain the exact original Git commit
when starting a new Colab runtime to resume an older run.

Only load checkpoints from your trusted run directories. An advisory process
lock prevents concurrent local writers; do not rely on it to coordinate separate
Colab machines/Drive mounts. Atomic local file replacement is used, but cloud
Drive synchronization is not a transactional checkpoint store: keep backups.

## Resource accounting and limitations

Each step records synchronized wall time, peak allocated/reserved CUDA memory,
forward token counts, and gradient/clipping diagnostics. The preflight records
per-layer native/auxiliary gradient diagnostics. Meta histories include matched
native virtual loss, outer loss, and generator gradient norm.
`command_history.json` records end-to-end command time, including setup/I/O,
unless the process is killed before it can write the final record. Per-step
times alone exclude checkpoint I/O, model loading, and preparatory cleanup.

The default has 1,000 native warmup updates and 600 meta episodes per seed,
plus 280 continuation updates per seed (two states, seven arms, twenty steps).
Development and test decoding at every horizon are additional cost. Use the
measured preflight to assess a run; no fixed hours-to-completion promise is made.

bf16 casts can round small fp32 adapter updates in the virtual forward. Finite
nonzero meta-gradients demonstrate connectivity, not optimizer usefulness or
high-precision equivalence. The real Qwen A100 run is required to test numerical
behavior, memory fit, and empirical utility. No FlashAttention, first-order
approximation, quantization, gradient checkpointing, or optimizer substitution
is silently enabled on OOM.

No LESS/BIDS recipe selector, genuine learned OT solver, multi-layer combination,
or multi-capability benchmark is added by this implementation. Llama replication
remains gated follow-up work with its own access and resource checks.

## Output contract

- `protocol.json`, `manifest.json`: effective settings and provenance.
- `data.json`, `splits.json`, `candidate_audit.json`: prepared examples, identities,
  duplicate/exclusion decisions, candidate strings, and token-length audits.
- `preflight.json`: pass/fail, measured resources, and gradient diagnostics.
- `seed_*/teacher`, `seed_*/learner`: native states and resumable histories.
- `seed_*/generators`: fitted controls, outer optimizer, episode and RNG state.
- `seed_*/branches`: matched continuation checkpoints and D/test metrics.
- `development_summary.json`, `development_lock.json`: all development outcomes,
  comparator choice, gate decision, and reporting-checkpoint hashes.
- `summary.json`, `REPORT.md`: all reporting pairs and primary seed-level contrasts.
- `command_history.json`: completed/interrupted command timings.

Training artifacts are ignored by Git. Do not commit access tokens or cached
model weights. Real-model results should be archived separately after review.
