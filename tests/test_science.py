from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from synless.core import historical_plan, build_candidates, candidate_losses, native_loss, validate_bank
from synless.data import ToyEncoder, ToyData, make_optimizer, move
from synless.experiment import (snapshot, restore, update, score_batch, validation_gradient,
                               select, run, validate_config)
from synless.optim import adamw_delta, less_feature, gradients, optimizer_parameters, flatten


def config():
    return json.loads((Path(__file__).parents[1] / "configs/smoke.json").read_text())


@pytest.fixture
def setup():
    torch.set_num_threads(2)
    torch.manual_seed(23)
    c = config()
    model = ToyEncoder()
    optimizer = make_optimizer(model, c)
    data = ToyData(c)
    batch = data.batch("train", list(range(8)))
    for _ in range(3):
        update(model, optimizer, batch, c)
    return model, optimizer, batch, c, data


def test_ot_uniform_same_support_no_positives(setup):
    model, _, batch, c, _ = setup
    bank = build_candidates(model(batch), batch["ids"], 0, top_k=c["top_k"])
    validate_bank([bank], [batch])
    for row in range(8):
        ot, uniform, real = bank[3*row:3*row+3]
        assert ot["indices"] == uniform["indices"]
        assert row not in ot["indices"]
        assert uniform["weights"] == pytest.approx([1/3]*3)
        assert len(real["indices"]) == 1
        assert sum(ot["weights"]) == pytest.approx(1)


def test_auxiliary_is_added_denominator_and_live_source_gradient():
    torch.manual_seed(8)
    images = torch.randn(5, 6, requires_grad=True)
    texts = torch.randn(5, 6, requires_grad=True)
    scale = torch.tensor(3., requires_grad=True)
    output = F.normalize(images, dim=-1), F.normalize(texts, dim=-1), scale
    candidates = build_candidates(output, [str(i) for i in range(5)], 0, top_k=2)
    candidate = candidates[1]
    loss = candidate_losses(output, [candidate])[0]
    logits = output[1] @ output[0].T * scale.detach()
    synthetic = F.normalize(sum(w * output[0][i] for i, w in zip(candidate["indices"], candidate["weights"])), dim=0)
    extra = (output[1][0] * synthetic).sum() * scale.detach()
    expected = torch.logsumexp(torch.cat([logits[0], extra[None]]), dim=0) - torch.logsumexp(logits[0], dim=0)
    torch.testing.assert_close(loss, expected)
    gi, gt, gs = torch.autograd.grad(loss, (images, texts, scale), allow_unused=True)
    assert gi[candidate["indices"]].norm() > 0
    assert gt[0].norm() > 0
    assert gs is None


@pytest.mark.parametrize("warmup", [0, 3])
def test_adamw_delta_matches_torch_with_clipping_groups_and_decay(warmup):
    torch.manual_seed(4)
    p = torch.nn.Parameter(torch.randn(3, dtype=torch.float64))
    q = torch.nn.Parameter(torch.tensor(4.61, dtype=torch.float64))
    opt = torch.optim.AdamW([{"params": [p], "lr": .007}, {"params": [q], "lr": .0002}], weight_decay=.03, foreach=False)
    for _ in range(warmup):
        p.grad, q.grad = torch.randn_like(p), torch.randn_like(q)
        opt.step()
    values = {p: torch.tensor([5., -3., 7.], dtype=torch.float64), q: torch.tensor(-3., dtype=torch.float64)}
    before = flatten([p.detach().clone(), q.detach().clone()])
    predicted = adamw_delta(opt, values, .1, q)
    assert torch.equal(before, flatten([p, q]))
    for param, grad in values.items():
        param.grad = grad.clone()
    torch.nn.utils.clip_grad_norm_([p, q], .1)
    opt.step()
    with torch.no_grad():
        q.clamp_(max=__import__("math").log(100))
    actual = flatten([p, q]) - before
    torch.testing.assert_close(predicted, actual, rtol=1e-9, atol=1e-12)


def test_less_feature_matches_released_epsilon_and_moments():
    p = torch.nn.Parameter(torch.tensor([1., 2.]))
    opt = torch.optim.AdamW([p])
    opt.state[p] = {"step": torch.tensor(5.), "exp_avg": torch.tensor([.1, -.2]), "exp_avg_sq": torch.tensor([.001, .004])}
    g = torch.tensor([.02, -.1])
    state = deepcopy(opt.state_dict())
    expected = (.9 * state["state"][0]["exp_avg"] + .1 * g) / (.999 * state["state"][0]["exp_avg_sq"] + .001 * g.square() + 1e-8).sqrt()
    torch.testing.assert_close(less_feature(opt, {p: g}), expected)
    torch.testing.assert_close(opt.state[p]["exp_avg"], state["state"][0]["exp_avg"])


def test_scoring_leaves_parameters_and_moments_unchanged(setup):
    model, optimizer, batch, c, data = setup
    state = snapshot(model, optimizer, 3)
    val = [data.batch("validation", list(range(8)))]
    vg = validation_gradient(model, optimizer, val, "cpu")
    candidates = build_candidates(model(batch), batch["ids"], 0, top_k=3)
    scores = score_batch(model, optimizer, batch, candidates, vg, c)
    after = snapshot(model, optimizer, 3)
    for name in state["model"]:
        assert torch.equal(state["model"][name], after["model"][name])
    for index, values in state["optimizer"]["state"].items():
        for key, value in values.items():
            assert torch.equal(value, after["optimizer"]["state"][index][key])
    assert len(scores) == 24
    assert all(abs(r["less_aux_cosine"]) <= 1.00001 for r in scores)


def test_marginal_prediction_matches_real_paired_step_to_first_order(setup):
    model, optimizer, batch, c, data = setup
    for group in optimizer.param_groups:
        group["lr"] = 1e-5
    state = snapshot(model, optimizer, 3)
    val = data.batch("validation", list(range(8)))
    vg = validation_gradient(model, optimizer, [val], "cpu")
    candidate = build_candidates(model(batch), batch["ids"], 0, top_k=3)[1]
    score = score_batch(model, optimizer, batch, [candidate], vg, c)[0]
    update(model, optimizer, batch, c)
    base_loss = float(native_loss(model(val)).detach())
    restore(model, optimizer, state)
    single_config = {**c, "alpha": c["alpha"] / c["selected_per_batch"]}
    update(model, optimizer, batch, single_config, [candidate])
    augmented_loss = float(native_loss(model(val)).detach())
    assert base_loss - augmented_loss == pytest.approx(score["marginal_predicted_gain"], abs=5e-7)


def test_restore_preserves_saved_optimizer_state(setup):
    model, optimizer, batch, c, _ = setup
    state = snapshot(model, optimizer, 3)
    restore(model, optimizer, state)
    update(model, optimizer, batch, c)
    first = snapshot(model, optimizer, 4)
    restore(model, optimizer, state)
    update(model, optimizer, batch, c)
    second = snapshot(model, optimizer, 4)
    for name in first["model"]:
        assert torch.equal(first["model"][name], second["model"][name])


def test_bank_rejects_changed_context_and_positive_leakage(setup):
    model, _, batch, _, _ = setup
    bank = build_candidates(model(batch), batch["ids"], 0, top_k=3)
    bad_batch = {**batch, "ids": list(reversed(batch["ids"]))}
    with pytest.raises(ValueError):
        validate_bank([bank], [bad_batch])
    bank[0]["indices"][0] = bank[0]["row"]
    with pytest.raises(ValueError):
        validate_bank([bank], [batch])


def test_split_ids_disjoint_and_selection_excludes_undefined():
    data = ToyData(config())
    splits = [set(v) for v in data.manifest["splits"].values()]
    assert not any(a & b for i, a in enumerate(splits) for b in splits[i+1:])
    assert select([{"id": "a", "x": None}, {"id": "b", "x": 2}], 1, "x")[0]["id"] == "b"
    with pytest.raises(ValueError):
        select([{"id": "a", "x": None}], 1, "x")


def test_end_to_end_resume_and_identity_guard(tmp_path):
    c = {**config(), "seeds": [1], "checkpoints": [1, 2], "candidate_batches": 1, "continuation_steps": 1}
    run(c, tmp_path)
    original = (tmp_path / "summary.json").read_bytes()
    mtime = (tmp_path / "seed_1/checkpoint_2.pt").stat().st_mtime_ns
    run(c, tmp_path)
    assert (tmp_path / "summary.json").read_bytes() == original
    assert (tmp_path / "seed_1/checkpoint_2.pt").stat().st_mtime_ns == mtime
    assert json.loads((tmp_path / "run.json").read_text())["status"] == "complete"
    with pytest.raises(ValueError, match="changed"):
        run({**c, "alpha": .1}, tmp_path)


def test_live_clip_adapter_offline(monkeypatch):
    transformers = pytest.importorskip("transformers")
    from synless.data import LiveCLIP
    cfg = transformers.CLIPConfig(
        text_config={"vocab_size": 32, "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2,
                     "num_attention_heads": 2, "max_position_embeddings": 8, "eos_token_id": 2},
        vision_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2,
                       "num_attention_heads": 2, "image_size": 16, "patch_size": 8}, projection_dim=8)
    monkeypatch.setattr(transformers.CLIPModel, "from_pretrained", lambda *a, **k: transformers.CLIPModel(cfg))
    model = LiveCLIP("offline-test")
    batch = {"pixel_values": torch.randn(4, 3, 16, 16), "input_ids": torch.tensor([[1, 3, 2]]*4), "attention_mask": torch.ones(4, 3, dtype=torch.long)}
    output = model(batch)
    assert output[0].shape == (4, 8)
    assert not model.clip.vision_model.encoder.layers[0].self_attn.q_proj.weight.requires_grad
    loss = native_loss(output)
    loss.backward()
    assert model.clip.vision_model.encoder.layers[-1].self_attn.q_proj.weight.grad is not None


def test_cub_loader_splits_images_and_uses_canonical_caption(monkeypatch):
    datasets = pytest.importorskip("datasets")
    transformers = pytest.importorskip("transformers")
    from PIL import Image
    from synless.data import CUBData

    class Split:
        _fingerprint = "offline_fixture"
        def __init__(self, prefix, count):
            self.rows = [{"file_name": f"{prefix}/{i}", "description": f"first {i}\nsecond {i}",
                          "image": Image.new("RGB", (16, 16))} for i in range(count)]
            self.rows.append(self.rows[0])  # duplicate rows must not duplicate image identities
        def __getitem__(self, index):
            return [r[index] for r in self.rows] if isinstance(index, str) else self.rows[index]

    class Processor:
        def __call__(self, images, text, **kwargs):
            assert all(t.startswith("first") for t in text)
            return {"pixel_values": torch.zeros(len(images), 3, 16, 16), "input_ids": torch.ones(len(images), 4, dtype=torch.long),
                    "attention_mask": torch.ones(len(images), 4, dtype=torch.long)}

    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: {"train": Split("train", 20), "test": Split("test", 8)})
    monkeypatch.setattr(transformers.CLIPProcessor, "from_pretrained", lambda *a, **k: Processor())
    c = {**config(), "dataset": "fixture", "model": "fixture", "validation_size": 4, "test_size": 4}
    data = CUBData(c)
    sets = [set(r[0] for r in records) for records in data.records.values()]
    assert [data.count(s) for s in ("train", "validation", "test")] == [16, 4, 4]
    assert not any(a & b for i, a in enumerate(sets) for b in sets[i+1:])
    assert data.batch("validation", [0, 1])["pixel_values"].shape == (2, 3, 16, 16)
