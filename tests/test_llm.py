from pathlib import Path
import json
from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from synless.llm.core import Generator, auxiliary, functional_adamw
from synless.llm.data import ByteTokenizer, numeric, canonical, distractors, encode, build_records, scheduled_batch, collate
from synless.llm.experiment import validate_config, snapshot, restore, digest_tensor_tree, execute, Paused, read_json


@pytest.fixture
def config():
    torch.set_num_threads(2)
    return validate_config(json.loads((Path(__file__).parents[1]/"configs/llm_smoke.json").read_text()))


def test_numeric_exactness_and_distractors():
    for text in ["1,200", "-2", "0", "0.10", "1/3", "-0.000000000000000001"]:
        assert numeric(canonical(numeric(text))) == numeric(text)
        wrong = distractors(text, "example")
        assert len({numeric(x) for x in wrong}) == 4
        assert numeric(text) not in {numeric(x) for x in wrong}
        assert distractors(text, "example") == wrong
    for text in ["answer is 4", "4 cats", "1,2", "NaN", "inf", "", "1e3"]:
        with pytest.raises(ValueError):
            numeric(text)
    assert all(numeric(x)>=0 for x in distractors("0", "zero"))


def test_masks_partitions_deduplication_and_schedule():
    tokenizer = ByteTokenizer()
    row = encode(tokenizer, "What is 1+1?", "2", 128)
    assert row["input_ids"][:row["prompt_end"]+1] == row["prompt_ids"]
    assert all(x == -100 for x in row["labels"][:row["prompt_end"]+1])
    assert row["labels"][-1] == tokenizer.eos_token_id
    assert not row["numeric_mask"][-1] and sum(row["numeric_mask"]) == 1
    with pytest.raises(ValueError, match="overlength"):
        encode(tokenizer, "long "*100, "2", 128)
    train = [{"question": f"Q {i}", "answer": f"#### {i}"} for i in range(40)]
    test = [{"question": "Q 2", "answer": "#### 2"}, {"question": "New question", "answer": "#### 8"}]
    splits, audit = build_records(train+[train[0]], test, tokenizer, 128, 7)
    groups = [r["question_hash"] for rows in splits.values() for r in rows]
    assert len(groups) == len(set(groups)) and len(audit["duplicates"]) == 2
    assert len(splits["test"]) == 1
    for name in ("A", "B"):
        for item in splits[name]:
            assert all(n["prompt_ids"] == item["correct"]["prompt_ids"] for n in item["negatives"])
    rows = list(range(8))
    batches = [scheduled_batch(rows, step, 2, 99) for step in range(4)]
    assert sorted(sum(batches, [])) == rows
    assert batches[2] == scheduled_batch(rows, 2, 2, 99)


def test_barycenter_live_gradients_permutation_and_controls():
    torch.manual_seed(8)
    q, p, n = [torch.randn(*shape, requires_grad=True) for shape in ((3, 8), (3, 8), (3, 4, 8))]
    generator = Generator("learned_barycenter")
    uniform, _, _ = auxiliary(q, p, n, "uniform_barycenter")
    learned, stats, w = auxiliary(q, p, n, "learned_barycenter", generator)
    torch.testing.assert_close(uniform, learned)
    torch.testing.assert_close(w, torch.full_like(w, .25))
    assert stats["effective_support"] == pytest.approx(4.)
    with torch.no_grad():
        generator.scorer[-1].weight.normal_()
    learned, _, w = auxiliary(q, p, n, "learned_barycenter", generator)
    permutation = [2, 0, 3, 1]
    other, _, pw = auxiliary(q, p, n[:, permutation], "learned_barycenter", generator)
    torch.testing.assert_close(other, learned)
    torch.testing.assert_close(pw, w[:, permutation])
    gradients = torch.autograd.grad(learned, (q,p,n,*generator.parameters()), allow_unused=True)
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert all(g.norm() > 0 for g in gradients[:3])
    hard, _, weights = auxiliary(q, p, n, "hardest_existing")
    qn = (F.normalize(q, dim=-1)[:,None]*F.normalize(n, dim=-1)).sum(-1)
    qp = (F.normalize(q,dim=-1)*F.normalize(p,dim=-1)).sum(-1)
    torch.testing.assert_close(hard, (weights*F.softplus((qn-qp[:,None])/.1)).sum(-1).mean())
    reweighter = Generator("learned_loss_reweighting")
    rw, _, _ = auxiliary(q,p,n,"learned_loss_reweighting",reweighter)
    assert not torch.isclose(rw, uniform)


@pytest.mark.parametrize("warmup", [0, 3])
def test_functional_adamw_exact_clipping_groups_decay_unused(warmup):
    torch.manual_seed(42)
    p = torch.nn.Parameter(torch.randn(3, dtype=torch.float64))
    q = torch.nn.Parameter(torch.randn(2, dtype=torch.float64))
    unused = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    opt = torch.optim.AdamW([{"params": [p,unused], "lr": .01}, {"params": [q], "lr": .002}],
                           betas=(.8,.97), weight_decay=.03, foreach=False)
    for _ in range(warmup):
        p.grad, q.grad, unused.grad = torch.randn_like(p), torch.randn_like(q), torch.ones_like(unused)
        opt.step()
    opt.zero_grad(set_to_none=True)
    state_before = digest_tensor_tree(opt.state_dict())
    gs = (torch.tensor([0., 5., -3.], dtype=torch.float64), torch.tensor([2., -8.], dtype=torch.float64), None)
    predicted = functional_adamw({"p":p,"q":q,"unused":unused}, gs, opt, .3)
    assert digest_tensor_tree(opt.state_dict()) == state_before
    # Clone predicted values before the real in-place update, particularly skipped parameters.
    predicted = {name: value.detach().clone() for name,value in predicted.items()}
    for param, grad in zip((p,q,unused), gs):
        param.grad = grad
    torch.nn.utils.clip_grad_norm_([p,q,unused], .3)
    opt.step()
    for name, actual in [("p",p),("q",q),("unused",unused)]:
        torch.testing.assert_close(predicted[name], actual, atol=1e-12, rtol=1e-10)


def test_meta_gradient_finite_difference_and_zero_moment():
    torch.manual_seed(19)
    theta = torch.nn.Parameter(torch.randn(5, dtype=torch.float64))
    phi = torch.nn.Parameter(torch.tensor(.3, dtype=torch.float64))
    opt = torch.optim.AdamW([theta], lr=.01, weight_decay=.03, foreach=False)
    opt.state[theta] = {"step": torch.tensor(5.), "exp_avg": torch.full_like(theta,.03),
                        "exp_avg_sq": torch.full_like(theta,.2)}
    def outer(value):
        inner = theta.square().sum() + value * theta.sin().sum()
        grads = torch.autograd.grad(inner, (theta,), create_graph=True)
        after = functional_adamw({"theta":theta}, grads, opt, .2)["theta"]
        return (after-1).square().sum()
    analytic = torch.autograd.grad(outer(phi), phi)[0]
    h = 1e-5
    numerical = (outer(phi.detach()+h)-outer(phi.detach()-h))/(2*h)
    torch.testing.assert_close(analytic, numerical, rtol=2e-6, atol=1e-9)
    zero = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    zero_opt = torch.optim.AdamW([zero], lr=.01)
    after = functional_adamw({"zero":zero}, (phi*zero,), zero_opt)["zero"]
    assert torch.isfinite(torch.autograd.grad(after.sum(), phi)[0])


def tiny_setup(config):
    pytest.importorskip("transformers")
    from synless.llm.model import load_learner, make_optimizer
    learner = load_learner(config, {}, 101)
    opt = make_optimizer(learner, config)
    train = [{"question": f"Q {i}", "answer": f"#### {i}"} for i in range(40)]
    parts, _ = build_records(train, [{"question":"test", "answer":"#### 2"}], ByteTokenizer(),128,7)
    return learner,opt,parts


def test_tiny_hook_masked_ce_alpha_zero_and_virtual_nonmutation(config):
    pytest.importorskip("transformers")
    from synless.llm.model import objective, update, meta_objective
    learner,opt,parts = tiny_setup(config)
    tok = ByteTokenizer()
    rows = parts["A"][:2]
    batch = collate([r["correct"] for r in rows],tok.pad_token_id,"cpu")
    out = learner(batch)
    full = learner.backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                            output_hidden_states=True, use_cache=False)
    ce = F.cross_entropy(full.logits[:,:-1].reshape(-1,259), batch["labels"][:,1:].reshape(-1),
                         reduction="none", ignore_index=-100).reshape(2,-1)
    expected = (ce.sum(1)/(batch["labels"][:,1:]!=-100).sum(1)).mean()
    torch.testing.assert_close(out["native"], expected)
    torch.testing.assert_close(out["q"], full.hidden_states[1][torch.arange(2),batch["prompt_end"]])
    native,_ = objective(learner,rows,tok,config)
    zero,_ = objective(learner,rows,tok,{**config,"alpha":0},"uniform_barycenter")
    torch.testing.assert_close(native,zero,rtol=0,atol=0)
    for _ in range(2):
        update(learner,opt,rows,tok,config)
    generator = Generator("learned_barycenter")
    before = snapshot(learner,opt)
    meta,_ = meta_objective(learner,opt,rows,parts["M"][:1],tok,config,"learned_barycenter",generator)
    grads = torch.autograd.grad(meta,tuple(generator.parameters()))
    assert all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.norm()) for g in grads) > 0
    assert digest_tensor_tree(before) == digest_tensor_tree(snapshot(learner,opt))
    update(learner,opt,rows,tok,config,"uniform_barycenter")
    expected_state = snapshot(learner,opt)
    restore(learner,opt,before)
    parent_hash = digest_tensor_tree(before)
    update(learner,opt,rows,tok,config,"uniform_barycenter")
    assert digest_tensor_tree(before) == parent_hash
    assert digest_tensor_tree(expected_state) == digest_tensor_tree(snapshot(learner,opt))


@pytest.mark.parametrize("pause_at", [3, 14])
def test_full_smoke_resume_and_test_gate(config,tmp_path,pause_at):
    pytest.importorskip("transformers")
    config = {**config, "states":[2], "horizons":[1], "meta_episodes":3}
    uninterrupted = tmp_path/"uninterrupted"
    resumed = tmp_path/"resumed"
    for path in (uninterrupted,resumed):
        execute("preflight",config,path)
    with pytest.raises(ValueError, match="lock development"):
        execute("report",config,resumed)
    with pytest.raises(Paused):
        execute("run",config,resumed,max_units=pause_at)
    assert not list(resumed.glob("seed_*/branches/step_*/*/test_*.json"))
    execute("run",config,uninterrupted)
    execute("run",config,resumed)
    assert read_json(uninterrupted/"development_summary.json")["primary"] == read_json(resumed/"development_summary.json")["primary"]
    for arm in ("global_rank_mixture","learned_loss_reweighting","learned_barycenter"):
        a = torch.load(uninterrupted/f"seed_789/generators/{arm}.pt",weights_only=True)
        b = torch.load(resumed/f"seed_789/generators/{arm}.pt",weights_only=True)
        assert digest_tensor_tree(a["generator"]) == digest_tensor_tree(b["generator"])
    execute("report",config,resumed)
    assert read_json(resumed/"manifest.json")["status"] == "complete"
    assert len(read_json(resumed/"summary.json")["paired_results"]) == 7
    assert read_json(resumed/"summary.json")["research_evidence"] is False
    with pytest.raises(ValueError, match="changed"):
        execute("run",{**config,"alpha":.9},resumed)
    checkpoint = next(resumed.glob("seed_*/branches/step_*/*/horizon_*.pt"))
    with checkpoint.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checkpoint changed"):
        execute("report",config,resumed)


def test_actual_barycenter_meta_gradient_fp64(config, monkeypatch):
    pytest.importorskip("transformers")
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    from synless.llm.model import update, meta_objective
    # Qwen explicitly downcasts RMSNorm and attention softmax to fp32 even
    # after model.double(). Use their mathematically identical fp64 reference
    # here; otherwise finite differences measure float32 rounding noise. This
    # fixture-local patch never changes production or mixed-precision tests.
    def reference_norm(self, hidden):
        return self.weight * hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + self.variance_epsilon)
    original_softmax = F.softmax
    def reference_softmax(input, dim=None, _stacklevel=3, dtype=None):
        if input.dtype == torch.float64:
            dtype = torch.float64
        return original_softmax(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
    monkeypatch.setattr(Qwen2RMSNorm, "forward", reference_norm)
    monkeypatch.setattr(F, "softmax", reference_softmax)
    learner,opt,parts = tiny_setup(config)
    learner.double()
    for _ in range(2):
        update(learner,opt,parts["A"][:1],ByteTokenizer(),config)
    generator = Generator("learned_barycenter").double()
    weight = generator.scorer[-1].weight
    def loss():
        return meta_objective(learner,opt,parts["A"][:1],parts["M"][:1],ByteTokenizer(),
                              config,"learned_barycenter",generator)[0]
    analytic = torch.autograd.grad(loss(),weight)[0][0,0].item()
    h = 1e-3
    with torch.no_grad():
        weight[0,0] += h
    plus = loss().item()
    with torch.no_grad():
        weight[0,0] -= 2*h
    minus = loss().item()
    with torch.no_grad():
        weight[0,0] += h
    assert abs(analytic) > 1e-12
    assert analytic == pytest.approx((plus-minus)/(2*h), rel=.01, abs=2e-10)


def test_bf16_base_fp32_adapters_meta_connectivity(config):
    pytest.importorskip("transformers")
    from synless.llm.model import update, meta_objective
    learner,opt,parts = tiny_setup(config)
    for param in learner.parameters():
        if not param.requires_grad:
            param.data = param.data.to(torch.bfloat16)
    assert all(p.dtype == torch.float32 for p in learner.trainable().values())
    for _ in range(2):
        update(learner,opt,parts["A"][:1],ByteTokenizer(),config)
    generator = Generator("learned_barycenter")
    loss,_ = meta_objective(learner,opt,parts["A"][:1],parts["M"][:1],ByteTokenizer(),
                            config,"learned_barycenter",generator)
    gradients = torch.autograd.grad(loss,tuple(generator.parameters()))
    assert all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.norm()) for g in gradients)>0


@pytest.mark.parametrize("bf16", [False, True])
def test_same_shape_causality_and_future_leak_detection(config, monkeypatch, bf16):
    from synless.llm.model import check_prompt_causality
    learner, _, parts = tiny_setup(config)
    if bf16:
        for p in learner.parameters():
            if not p.requires_grad:
                p.data = p.data.to(torch.bfloat16)
    rows = parts["A"][:2]
    diagnostics = check_prompt_causality(learner, rows, ByteTokenizer())
    assert all(v["max_abs_difference"] == 0 for v in diagnostics.values())
    forward = learner.forward
    def leaking_forward(batch, need_loss=True):
        result = forward(batch, need_loss)
        # Deliberately make the prompt depend on the first future token.
        future = batch["input_ids"][torch.arange(len(batch["input_ids"])), batch["prompt_end"]+1]
        result["q"] = result["q"] + future[:,None].float()
        return result
    monkeypatch.setattr(learner, "forward", leaking_forward)
    with pytest.raises(RuntimeError, match="depends on future answer tokens"):
        check_prompt_causality(learner, rows, ByteTokenizer())


def test_cross_batch_prompt_drift_is_diagnostic_and_bad_prefix_fails(config, monkeypatch):
    from synless.llm.model import objective
    learner, _, parts = tiny_setup(config)
    rows = parts["A"][:2]
    original_loss, _ = objective(learner, rows, ByteTokenizer(), config, "uniform_barycenter")
    forward = learner.forward
    def rounded_wrong_query(batch, need_loss=True):
        result = forward(batch, need_loss)
        if not need_loss:
            # Simulate cross-layout rounding in the unused duplicate query.
            result["q"] = result["q"] + .01
        return result
    monkeypatch.setattr(learner, "forward", rounded_wrong_query)
    loss, metrics = objective(learner, rows, ByteTokenizer(), config, "uniform_barycenter")
    torch.testing.assert_close(loss, original_loss, rtol=0, atol=0)
    assert metrics["cross_batch_prompt_max_abs"] > .009
    assert metrics["cross_batch_prompt_max_relative_l2"] > 0
    corrupted = deepcopy(rows)
    corrupted[0]["negatives"][0]["input_ids"][0] += 1
    with pytest.raises(ValueError, match="identical prompt tokens"):
        objective(learner, corrupted, ByteTokenizer(), config, "uniform_barycenter")
