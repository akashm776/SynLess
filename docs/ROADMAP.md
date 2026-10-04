# Research roadmap

[Project overview](../README.md) · [LLM protocol](LLM_GENERATOR_PROTOCOL.md)

This roadmap separates completed work from proposed experiments. Unchecked items
are not available features. All existing executable configurations remain CLIP/toy.

## Completed foundation: CLIP/CUB

- [x] Port SCSS OT/uniform/hardest-real negative constructions with live gradients.
- [x] Implement LESS-style, marginal-update, and BIDS selection adaptations.
- [x] Test optimizer-state restoration, attribution, construction, and resume.
- [x] Run the three-seed, two-state A100 pilot and record ninety branches.
- [x] Publish lightweight metrics, provenance, and interpretation.

The pilot shows that enrichment and utility are different outcomes and that
effects depend on the learner state. It does not validate the new LLM method.

## Stage 0: implement and validate the LLM path

- [x] Define the research question, model progression, controls, and reporting rules.
- [ ] Add a separate LLM package/CLI and notebook; preserve CLIP entry points.
- [ ] Implement GSM8K partition manifests, numeric distractor audits, token masks,
      and strict answer-only evaluation.
- [ ] Implement layer hooks, permutation-equivariant scoring, and all seven arms.
- [ ] Implement functional AdamW and one-step outer differentiation; test against
      real optimizer updates and finite differences on a tiny fp64 model.
- [ ] Verify fresh-learner forks and exact resume of generator/learner state.
- [ ] Measure actual Qwen A100 memory, throughput, and gradient behavior.
- [ ] Freeze executable protocol, revisions, and resource-approved settings.

Gate: no substantive GPU pilot until correctness and resource checks pass.
Ordinary LoRA training fitting in memory is not sufficient evidence that the
differentiable inner/outer loop fits.

## Stage 1: Qwen feasibility pilot

- [ ] Train separate generators and learned controls for three pipeline seeds.
- [ ] Evaluate frozen generators on fresh B-partition learners at states 100/500.
- [ ] Compare paired continuations at 1/5/20 updates on development data.
- [ ] Lock all choices and comparator selection before official-test reporting.
- [ ] Publish every arm, seed, decision rule, memory measurement, and compute cost.

Gate to Llama: development evidence beyond native SFT and equally meta-trained
existing-loss reweighting, without consistent exact-match harm. A failed gate
should produce a negative-result report, not a claim based on influence rankings.

## Stage 2: Llama-3-8B method replication

- [ ] Verify model access and terms using the user's authorized account.
- [ ] Independently validate the full differentiable step on the available A100.
- [ ] Fit and evaluate a Llama generator using the same experimental logic.
- [ ] Optionally test frozen Qwen-function transfer as a separately labeled arm.

Same model as BIDS does not mean same benchmark as BIDS. Replicating a learning
procedure on another model is not proof that its fitted function transfers.

## Stage 3: explicitly deferred research

- [ ] Compare quarter/middle/three-quarter layers individually, then interactions.
- [ ] Define a genuine OT problem before proposing a learned-cost transport map.
- [ ] Add LESS/BIDS selection over a frozen recipe bank with a new selection split,
      shared attribution space, composition-matched random controls, and real updates.
- [ ] Specify multiple capabilities and per-task outcomes before claiming balance.
- [ ] Test longer training and compute-matched controls before efficiency claims.

These require a protocol amendment; they are not quietly bundled into the first
Qwen pilot. No dependency on published LLM success is implied by this list.
