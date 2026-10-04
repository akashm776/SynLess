"""Small auditable LoRA adapter and completion-only causal-LM objectives."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call

from .core import auxiliary, functional_adamw, finite_gradients
from .data import ByteTokenizer, collate, numeric


class LoRALinear(nn.Module):
    """Frozen dense projection plus fp32 adapters; no dropout or quantization."""
    def __init__(self, base, rank, scaling):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = scaling / rank

    def forward(self, x):
        original = self.base(x)
        extra = F.linear(F.linear(x.to(self.a.dtype), self.a), self.b)
        return original + (self.scale * extra).to(original.dtype)


class Learner(nn.Module):
    def __init__(self, backbone, layer):
        super().__init__()
        self.backbone = backbone
        self.layer = layer
        if not 1 <= layer <= len(backbone.model.layers):
            raise ValueError("Layer is a one-based decoder-block index")

    def trainable(self):
        return {name: p for name, p in self.named_parameters() if p.requires_grad}

    def forward(self, batch, need_loss=True):
        captured = []
        handle = self.backbone.model.layers[self.layer-1].register_forward_hook(
            lambda module, args, output: captured.append(output[0] if isinstance(output, tuple) else output))
        try:
            out = self.backbone.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                      use_cache=False, return_dict=True)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("Expected one raw decoder-block output")
        tapped = captured[0]
        dtype = torch.float64 if tapped.dtype == torch.float64 else torch.float32
        hidden = tapped.to(dtype)
        mask = batch["numeric_mask"].to(dtype)
        q = hidden[torch.arange(len(hidden), device=hidden.device), batch["prompt_end"]]
        p = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        result = {"q": q, "p": p}
        if need_loss:
            targets = batch["labels"][:, 1:]
            selected = targets != -100
            # Project only completion predictor positions to the large vocabulary.
            # This is exactly the masked full-logit CE, without prompt logits.
            logits = self.backbone.lm_head(out.last_hidden_state[:, :-1][selected]).to(dtype)
            ce = F.cross_entropy(logits, targets[selected], reduction="none")
            sample = torch.arange(len(targets), device=targets.device)[:, None].expand_as(targets)[selected]
            totals = torch.zeros(len(targets), device=ce.device, dtype=ce.dtype).scatter_add(0, sample, ce)
            counts = selected.sum(1)
            numeric_selected = batch["numeric_mask"][:, 1:][selected]
            numeric_totals = torch.zeros_like(totals).scatter_add(0, sample, ce * numeric_selected)
            result["native"] = (totals / counts).mean()
            result["numeric"] = (numeric_totals / batch["numeric_mask"][:, 1:].sum(1)).mean()
        return result


def load_tokenizer(config, resolved):
    if config["backend"] == "tiny":
        return ByteTokenizer()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=resolved["model_revision"],
                                               use_fast=True, trust_remote_code=False)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with offset mappings is required")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_learner(config, resolved, seed):
    from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM
    device = torch.device(config["device"])
    if config["backend"] == "tiny":
        # Every pipeline seed shares identical tiny pretrained/base weights.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1234)
            base_config = Qwen2Config(vocab_size=259, hidden_size=16, intermediate_size=32,
                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                max_position_embeddings=512, attention_dropout=0., pad_token_id=0, eos_token_id=1)
            base_config._attn_implementation = "eager"
            backbone = Qwen2ForCausalLM(base_config).to(device)
    else:
        backbone = AutoModelForCausalLM.from_pretrained(config["model"], revision=resolved["model_revision"],
            torch_dtype=torch.bfloat16, attn_implementation="eager", trust_remote_code=False).to(device)
    backbone.requires_grad_(False)
    backbone.eval()  # Dropout disabled; autograd remains enabled.
    backbone.config.use_cache = False
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    replaced = []
    for name, module in list(backbone.named_modules()):
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in ("q_proj", "v_proj"):
            parent, leaf = name.rsplit(".", 1)
            setattr(backbone.get_submodule(parent), leaf, LoRALinear(module, config["rank"], config["lora_alpha"]))
            replaced.append(name)
    if len(replaced) != 2 * backbone.config.num_hidden_layers:
        raise ValueError("Unsupported architecture: expected q/v projections in every decoder block")
    learner = Learner(backbone, config["layer"]).eval()
    return learner


def make_optimizer(learner, config):
    return torch.optim.AdamW(list(learner.trainable().values()), lr=config["lr"],
        betas=(.9, .999), eps=1e-8, weight_decay=config["weight_decay"], foreach=False)


def objective(learner, rows, tokenizer, config, arm="native", generator=None, parameters=None):
    device = next(learner.parameters()).device
    def forward(batch, need_loss=True):
        if parameters is None:
            return learner(batch, need_loss=need_loss)
        return functional_call(learner, parameters, (batch,), {"need_loss": need_loss}, strict=False)
    batch = collate([r["correct"] for r in rows], tokenizer.pad_token_id, device)
    output = forward(batch)
    native = output["native"]
    diagnostics = {"native_loss": float(native.detach()), "numeric_loss": float(output["numeric"].detach()),
                   "forward_tokens": int(batch["attention_mask"].sum())}
    if arm == "native" or config["alpha"] == 0:
        return native, diagnostics
    wrong_batch = collate([n for row in rows for n in row["negatives"]], tokenizer.pad_token_id, device)
    wrong = forward(wrong_batch, need_loss=False)
    # Compare the actual causal states, not just token prefixes.
    expected = output["q"][:, None].expand(-1, 4, -1).reshape_as(wrong["q"])
    if not torch.allclose(expected.detach(), wrong["q"].detach(), rtol=2e-2, atol=2e-3):
        raise RuntimeError("Prompt states changed across teacher-forced answers")
    aux, stats, _ = auxiliary(output["q"], output["p"], wrong["p"].reshape(len(rows), 4, -1),
                               arm, generator, config["tau"])
    diagnostics.update(stats)
    diagnostics.update(auxiliary_loss=float(aux.detach()),
                       forward_tokens=diagnostics["forward_tokens"] + int(wrong_batch["attention_mask"].sum()))
    return native + config["alpha"] * aux, diagnostics


def update(learner, optimizer, rows, tokenizer, config, arm="native", generator=None):
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = objective(learner, rows, tokenizer, config, arm, generator)
    loss.backward()
    parameters = list(learner.trainable().values())
    if not torch.isfinite(loss) or not finite_gradients([p.grad for p in parameters]):
        raise FloatingPointError("Nonfinite learner loss/gradient")
    norm = torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"], error_if_nonfinite=True)
    optimizer.step()
    metrics["gradient_norm"] = float(norm)
    metrics["clipped"] = float(norm > config["max_grad_norm"])
    return metrics


def meta_objective(learner, optimizer, inner, outer, tokenizer, config, arm, generator):
    parameters = learner.trainable()
    loss, metrics = objective(learner, inner, tokenizer, config, arm, generator)
    gradients = torch.autograd.grad(loss, tuple(parameters.values()), create_graph=True, allow_unused=True)
    virtual = functional_adamw(parameters, gradients, optimizer, config["max_grad_norm"])
    outer_loss, outer_metrics = objective(learner, outer, tokenizer, config, parameters=virtual)
    metrics.update(meta_loss=float(outer_loss.detach()), outer_forward_tokens=outer_metrics["forward_tokens"])
    return outer_loss, metrics


@torch.no_grad()
def evaluate(learner, rows, tokenizer, config):
    native, number, correct, truncated = 0., 0., 0, 0
    device = next(learner.parameters()).device
    for row in rows:
        _, metrics = objective(learner, [row], tokenizer, config)
        native += metrics["native_loss"]
        number += metrics["numeric_loss"]
        ids = torch.tensor([row["correct"]["prompt_ids"]], device=device)
        generated = learner.backbone.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
            do_sample=False, max_new_tokens=config["max_new_tokens"], use_cache=True,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        tail = generated[0, ids.shape[1]:].tolist()
        ended = tokenizer.eos_token_id in tail
        # At the token cap even a numeric prefix may be an incomplete answer.
        truncated += int(not ended)
        text = tokenizer.decode(tail[:tail.index(tokenizer.eos_token_id)] if ended else tail,
                                skip_special_tokens=False).strip()
        try:
            correct += int(ended and numeric(text) == numeric(row["answer"]))
        except (ValueError, ZeroDivisionError):
            pass
    return {"native_loss": native/len(rows), "numeric_loss": number/len(rows),
            "exact_match": correct/len(rows), "correct": correct, "truncated": truncated, "count": len(rows)}
