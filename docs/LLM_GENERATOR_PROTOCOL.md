# Learned layer-wise negative generators: experiment protocol v0.2

Status: a separate LLM runner implements the Qwen pilot and all seven arms;
offline tiny-model correctness and resume tests are available. Real Qwen/GSM8K
training, GPU-memory validation, and utility results remain pending the A100 run.
See the [run guide](LLM_RUN.md). The original CLIP runner remains unchanged.
Run manifests lock revisions and numerical settings; changes after preflight
require a new output directory, before examining reporting outcomes.

Implementation clarifications in v0.2: SFT completion loss includes the leading
space and EOS; numeric-token metrics/pooling exclude standalone whitespace and
EOS. Token-cap-truncated decoding counts as incorrect. Correct-answer eligibility
precedes partition allocation; additional wrong-answer eligibility applies only
to A/B. The development accuracy gate requires nonnegative pooled exact-match
differences and no majority-seed harm versus both native and learned-loss controls.
These refinements were made before any real-model pilot outcome was observed.

## 1. Question and scope

Can a reusable, example-conditioned generator of hidden-state negatives improve
LLM supervised fine-tuning (SFT) on fresh learners and disjoint training examples,
beyond fixed mixtures and learning to reweight existing negative losses?

The intervention is **SFT plus an auxiliary representation loss**. It does not
generate textual instructions, replace correct labels, or insert mixed vectors
into the autoregressive token stream. A mixture of wrong-answer representations
is not guaranteed to represent an actual wrong answer. Its usefulness is an
empirical hypothesis, established by downstream native loss and answer accuracy.

The CLIP/CUB pilot is motivation, not evidence of LLM transfer. Its state-dependent
results motivate matched optimizer states, early/late checkpoints, and multiple
continuation horizons here. Do not pool its outcomes with the new experiment.

## 2. Stages and model choices

| Stage | Learner | Purpose | Advancement condition |
|---|---|---|---|
| 0 | Tiny random causal LM, then Qwen on A100 | Correctness, derivatives, resource measurement | All preflight checks pass |
| 1 | `Qwen/Qwen2.5-1.5B` base | One-layer feasibility and utility pilot | Useful development-set evidence beyond non-generating controls |
| 2 | `meta-llama/Meta-Llama-3-8B` base | Independently repeat the method on BIDS's main model family/version | Separate access, memory, and compute preflight |
| 3 | Deferred | Genuine OT, multiple layers, and multi-capability selection | Separate protocol amendment, not automatic expansion |

Qwen is our practical pilot choice. LESS used Llama-2-7B, Llama-2-13B, and
Mistral-7B; BIDS used Llama-3-8B and Mistral-7B-v0.3. Matching BIDS's model does
**not** reproduce its UltraInteract training mixture or seven evaluation tasks.
This one-task pilot cannot establish balanced learning of diverse capabilities.

Stage 2 fits a new generator on Llama using the same protocol: this tests
replication of the method, not transfer of the learned Qwen function. A frozen
Qwen generator evaluated on Llama, with no Llama meta-training, is a separately
labeled exploratory arm. Do not tune it on Llama reporting outcomes.

Start after decoder block 14 (one-based) of Qwen's 28 blocks. Capture the block's
residual-stream output, not a final-normalized substitute. Assert the hook's
identity in tests. For Llama choose the corresponding middle block after reading
the resolved model config; record its exact index. Only later compare quarter,
middle, and three-quarter depths, one at a time before joint interventions.

## 3. Dataset, targets, and separation

Use `openai/gsm8k`, `main`. Extract the numeric final answer after the dataset's
answer delimiter; do not train on the worked solution in this pilot. This is a
restricted answer-only math instruction-tuning test, not chain-of-thought tuning.

Freeze this plain-text prompt for both base models:

```text
Question: {question}
Give only the final numeric answer.
Answer:
```

The completion is a space, the canonical numeric answer, then EOS. Mask prompt
and padding labels. Average completion-token cross-entropies within each example,
then average examples. Track numeric-token-only NLL separately so EOS improvements
cannot conceal worse answer-token prediction. Check token boundaries using the
joint prompt/completion encoding; do not assume concatenated tokenizations agree.

Deduplicate normalized questions before a deterministic group-level split using
split seed 1701. Allocate the eligible official training questions approximately
40/15/10/35 percent (whole groups, with realized counts recorded):

| Partition | Role | May influence generator parameters? |
|---|---|---|
| A, 40% | Native teacher warmup and meta-training inner steps | Yes, through inner updates |
| M, 15% | Outer/meta-training feedback | Yes, directly |
| D, 10% | Development, feasibility decisions, comparator choice | Indirectly; explicitly not reporting |
| B, 35% | Fresh learner warmup and continuation training | No fitting of generator parameters |
| Official test | Locked reporting | Never |

Persist IDs, content hashes, duplicate groups, exclusions, candidate strings,
tokenizer revision, and split hash. Audit train/test duplicate questions and report
exclusion counts. Never silently truncate answers; for overlength prompts freeze
an eligibility policy before training and report test coverage. Do not use test
performance to choose a sequence limit, layer, loss coefficient, or generator.
This split prevents new-run leakage, not possible pretraining contamination.

For every eligible A/B question construct four distinct, numerically unequal
wrong answers using deterministic nearby numeric perturbations. Parse with exact
decimal/rational arithmetic, not float-string equality. Start with offsets
[-2, -1, +1, +2]; exclude negative values for nonnegative gold answers and fill
from [+3, -3, +4, -4, ...] until four valid unequal values exist. Keep the same
strings across arms/models; report token-length differences. These are simple
numeric distractors, not verified plausible reasoning errors.

Candidate ordering is randomized reproducibly per example and shared across arms.
No wrong completion receives a positive language-model target. Meta/reporting
losses use correct answers only. Correct answers are available during supervised
training, but no generator or gold-answer representation is needed at inference.

## 4. Representation construction

For a prompt x and layer l, re-encode current learner states on every update:

- q_l: the hidden state at the last prompt token, before any answer token.
- p_l: mean hidden state over correct answer tokens, excluding EOS and padding.
- n_lj: mean hidden state over wrong answer j's tokens, excluding EOS and padding.

All answer states are obtained by teacher-forced forward passes on their own
prompt + answer sequence. The causal prompt state must agree across these
sequences with dropout disabled. Normalize q, p, and each n separately in fp32
with a fixed epsilon before taking similarities or mixing.

Use a shared candidate scorer, not an order-sensitive vector of four logits:

```text
f_j = [cos(q,n_j), cos(p,n_j),
       mean_{k != j} cos(n_j,n_k), cos(q,p)]
w_j = softmax_j(MLP_phi(stop_gradient(f_j)))
z   = normalize(sum_j w_j * n_j)
aux = softplus((cos(q,z) - cos(q,p)) / tau)
L_inner = L_SFT + alpha * mean_examples(aux)
```

Initial scorer: 4 -> 32 -> 1 with tanh, final layer initialized to zero so the
initial generator is uniform. Candidate permutation must permute weights and
leave z/loss unchanged. Dimension-independent geometry permits a frozen-function
cross-model test, but does not imply that geometry or usefulness transfers.

Initial design defaults: alpha=0.01, tau=0.1, normalization epsilon=1e-8. These
are starting choices, not validated hyperparameters. Freeze identical values for
all auxiliary arms. Any preflight revision gets a new protocol version and is
made without viewing reporting outcomes. Log auxiliary/native gradient norm ratios
and clipping frequency; an apparent gain caused solely by loss scaling needs the
non-generating controls below.

Generator inputs are stop-gradient learner features. Constituent q, p, n remain
live in the auxiliary loss, so the learner gets representation gradients. During
meta-training retain dependence on phi through the weights and differentiable
learner update. At evaluation freeze phi; weights still adapt to each new
example/state, but their feature path does not differentiate into the learner.
No learned loss-strength gate or separate trainable projection head in v0.2.

This is a learned **barycentric-weight function**, not yet an OT map. Calling it
OT would require an explicit transport problem with source/target measures,
marginal constraints, cost, regularization, and solver checks. With one source
and a fixed target marginal, a balanced coupling has no free allocation to learn;
renaming a softmax or a similarity weighting "OT" does not solve that problem.
A genuine learned-cost OT experiment belongs in stage 3.

## 5. Meta-training and fresh learners

Freeze base weights; use LoRA in q_proj/v_proj across decoder blocks. Initial
design: rank 8, LoRA scaling 16, dropout 0; AdamW with LR 1e-4, betas (0.9,0.999),
epsilon 1e-8, weight decay 0, global gradient clip 1.0, constant LR. Save full
trainable weights, moments, step counters, RNG state, and data order.

For each independent pipeline replication:

1. Train a native-only teacher on A and save states at 100 and 500 updates.
2. Meta-train phi for 200 episodes, alternating the two states. Each episode
   restores the teacher/optimizer, draws an A inner batch and M outer batch,
   makes one differentiable AdamW update with L_inner, and minimizes the new
   learner's native M loss. Only phi persists across episodes. The matched
   native-only virtual update is recorded as a diagnostic.
3. Freeze phi. Initialize a fresh LoRA learner from the same pretrained base,
   train natively on B, and save its own states at 100 and 500 updates.
4. From each fresh state fork all arms, restoring identical model, optimizer,
   RNG, and B batch order. Evaluate at continuation updates 1, 5, and 20 along
   each branch, not as independently restarted horizons.

Initial batch plan: 4 examples per native/continuation optimizer update and
1 example for each meta inner/outer batch. If resource checks require changes,
freeze the revised counts before the pilot and apply them to every arm; record
both steps and processed tokens. Training order is shuffled without replacement
within each partition pass, with deterministic reshuffles across passes.

Outer optimizer: AdamW, LR 1e-3, betas (0.9,0.999), epsilon 1e-8, weight decay 0,
generator gradient clip 1.0. Final episode 200 is the generator checkpoint;
do not select the best reporting episode. This is a one-step meta-objective,
not an assumption that improvements persist for twenty steps.

Differentiate the real specified AdamW update, including current moments, step
number, bias correction, epsilon convention, and gradient clipping. A nominal
"first-order" shortcut that discards the mixed derivative may remove the
generator learning signal; it is not an interchangeable memory optimization.
Do not silently replace AdamW with SGD or omit optimizer history.

Use three complete pipeline replications, pairing teacher/meta/learner seeds:
(101,11,789), (102,22,2026), (103,33,31415). Each replication fits its own phi and
learned controls. This measures combined generator/learner variability. States,
horizons, test examples, and extra random draws are not independent seed trials.

## 6. Matched intervention arms

| Arm | Auxiliary construction | Purpose |
|---|---|---|
| native | None | Matched SFT continuation |
| uniform_barycenter | w_j=1/4 | Fixed synthetic mixture |
| similarity_barycenter | w_j proportional to exp(cos(q,n_j)/0.1), detached weights | Fixed geometry heuristic; not OT |
| hardest_existing | One n_j with largest cos(q,n_j) | Non-generating negative control |
| global_rank_mixture | Four meta-learned logits over candidates ranked by detached cos(q,n_j) | Learned but non-example-conditioned mixing rule |
| learned_loss_reweighting | Same conditional scorer; sum_j w_j softplus((cos(q,n_j)-cos(q,p))/tau) | Meta-learning without mixing hidden states |
| learned_barycenter | Conditional mixture z defined above | Proposed method |

The global rank arm sorts candidates by hardness with deterministic tie handling;
fixed logits over randomly permuted candidate identities would be meaningless.
All learned arms have the same A/M access, teacher states, episode count, outer
optimizer, and seed pairing. The learned-loss arm uses the same scorer capacity
and feature detachment as learned barycenters.

Hold native examples, candidate support, per-step auxiliary example count,
optimizer, clipping, and alpha fixed across auxiliary arms. Native-only omits
wrong-answer forwards: equality of update counts is not equality of compute.
Log measured time, tokens, memory, and meta-training cost separately. No claim of
compute efficiency without an additional compute-matched baseline.

The sharpest comparison is learned_barycenter versus learned_loss_reweighting:
mixing activations before a nonlinear loss differs from mixing existing losses.
Track weight entropy, max weight, effective support 1/sum(w_j^2), pre-normalization
mixture norm, and distance to the nearest constituent. Near-one-hot collapse is
evidence of selection behavior, not evidence that novel mixtures caused the gain.

## 7. Where LESS and BIDS fit

They select training data using influence estimates; they do not train the
generator defined here. Keep two questions separate:

1. Does the frozen generator improve actual learning? The matched arms above
   answer this and are the primary experiment.
2. Do LESS/BIDS favor its candidate constructions, and do their scores predict
   actual gains? A subsequent, explicitly labeled selection experiment answers
   this. A high influence score alone is not evidence of improvement.

For that follow-up, freeze a bank of (context, example, layer, construction)
recipes at each learner state; compare uniform, fixed, learned, and existing
negative choices with the same support. Score all recipes and validation
instances in the identical LoRA parameter space, using a shared projection if
one is necessary. Document whether scores use auxiliary-only or total-update
gradients; do not compare selectors using different attribution spaces.

Use a separately reserved selection-validation split, construction-matched random
controls, bottom-score controls, and actual native-matched continuations. Test
both frozen-mixture recipes and a dynamic frozen-generator policy separately:
rescoring one object and training another obscures what a ranking predicts.

This follow-up is not activated by v0.2. Before activating it, amend the split
allocation and selection budgets. To study BIDS's balanced-capability claim,
also specify multiple genuinely different tasks, their validation weights and
per-task reporting metrics. Generator categories or layers are not capabilities.

## 8. Reporting and decision rules

Primary outcome: completion NLL at 20 continuation updates, treatment minus its
matched native arm, averaged equally over the two starting states within each
pipeline seed. Negative is helpful. Also report each state separately so the
pooled result cannot hide a sign reversal. Secondary outcomes: numeric-token NLL,
strict numeric exact-match accuracy, and trajectories at updates 1 and 5.

Accuracy uses greedy decoding, max_new_tokens=32, and an exact canonical numeric
comparison of the entire stripped completion; malformed/truncated answers count
as incorrect. Any looser extraction metric is secondary and labeled. Report
correct-count differences as well as fractions. Evaluate the complete eligible
official test set only after all arms and choices are frozen. Do not tune on
intermediate official-test horizons; produce them together in the reporting pass.

Report paired differences against (a) native, (b) learned_loss_reweighting, and
(c) the best non-generating arm chosen using D only, before opening test results.
The candidate set for (c) is native, hardest_existing, learned_loss_reweighting.
Publish all arms, not only winners, per-seed values, mean, seed SD, and clearly
descriptive intervals. Three replications support a pilot, not a definitive
significance or generalization claim. Evaluation-question uncertainty and training
seed uncertainty are different; question bootstrap samples cannot replace seeds.

Advance to Llama based on D, not a favorable official-test result: pooled
20-step NLL improves over native and learned loss reweighting, has the same sign
in at least two of three replications, and shows no consistent exact-match harm.
If this fails, report the failed hypothesis; any redesign gets a new protocol
version, and old test outcomes cannot subsequently be treated as untouched.

Interpretation limits:

- Beating native only: useful auxiliary supervision, not yet useful generation.
- Beating fixed mixtures but not learned loss weighting: no distinct mixing gain.
- Winning only on teacher/A states: no evidence of reuse on fresh learners.
- Winning at one update but losing at twenty: short-horizon meta-objective mismatch.
- Better NLL without accuracy: likelihood improvement, not demonstrated task gain.
- Independently fitting on Llama and winning: method replication, not frozen-map transfer.
- A one-layer gain: does not establish all-layer or jointly composed generators.

## 9. A100 preflight and implementation acceptance

Do not assume an 8B model's ordinary LoRA fit guarantees a differentiable
meta-update fits. First measure Qwen on the actual Colab A100, with frozen base
weights in bf16 and LoRA/generator parameters and moments in fp32. Start with
sequence length 512; verify dataset coverage before freezing eligibility.
Avoid quantization and fused/flash attention for the first derivative checks;
each alternative needs its own higher-order-gradient validation.

Required checks before any substantive run:

- Partition and candidate audits; correct prompt/answer masks; no test IDs in
  optimizers, selection, generator fitting, or hyperparameter decisions.
- Native and auxiliary forwards finite; live gradients reach intended LoRA
  modules below the tapped layer; native loss reaches modules above it.
- Uniform initialization and permutation invariance; weights sum to one;
  alpha=0 reproduces native; one-hot barycenter equals its existing-negative loss.
- Functional AdamW equals a real optimizer step on a tiny model with nonzero
  stored moments and step counters, including clipping and unused parameters.
- Finite-difference meta-gradient agrees on a tiny fp64 problem; phi gets a
  finite, nonzero signal in the real model. Do not claim equivalence merely
  because the outer loss has a grad_fn.
- Teacher weights/moments do not mutate across virtual episodes; branch restore
  reproduces native; resume restores generator optimizer/RNG and data position.
- Peak allocated/reserved memory and synchronized wall time for native,
  auxiliary, and full differentiable inner+outer steps, with headroom for
  checkpointing/reporting. Extrapolate runtime from measurements, not guesses.
- Record source commit/hash, exact package versions, model/tokenizer/dataset
  revisions, hardware, precision, attention backend, and all seeds/configuration.

Stage 2 additionally requires user-provided gated-model access and agreement to
the model's terms. Use a secret store, never a notebook literal or committed
token. If the real differentiable 8B update does not fit, report that limitation
and revise the protocol explicitly; do not call a changed estimator the same run.

Implemented output contract: `protocol.json`, `manifest.json`, `splits.json`,
`candidate_audit.json`, `preflight.json`, per-replication teacher/generator/learner
checkpoints, matched branch metrics, `summary.json`, and `REPORT.md`. Separate
output roots from CLIP; no CLIP checkpoint or metric file is overwritten.

## Primary references

- [LESS paper, especially sections 4 and 5](https://arxiv.org/html/2402.04333v2).
- [BIDS paper](https://aclanthology.org/2025.findings-emnlp.373/).
- [Qwen2.5-1.5B model card](https://huggingface.co/Qwen/Qwen2.5-1.5B)
  and [architecture config](https://huggingface.co/Qwen/Qwen2.5-1.5B/raw/main/config.json).
- [Llama-3-8B model card and access terms](https://huggingface.co/meta-llama/Meta-Llama-3-8B).
- [GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k).
