"""Live barycenters, matched controls, and non-mutating differentiable AdamW."""

import torch
from torch import nn
from torch.nn import functional as F

ARMS = ("native", "uniform_barycenter", "similarity_barycenter", "hardest_existing",
        "global_rank_mixture", "learned_loss_reweighting", "learned_barycenter")
LEARNED = ("global_rank_mixture", "learned_loss_reweighting", "learned_barycenter")


class Generator(nn.Module):
    def __init__(self, arm, hidden=32):
        super().__init__()
        if arm not in LEARNED:
            raise ValueError(arm)
        self.arm = arm
        if arm == "global_rank_mixture":
            self.logits = nn.Parameter(torch.zeros(4))
        else:
            self.scorer = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            nn.init.zeros_(self.scorer[-1].weight)
            nn.init.zeros_(self.scorer[-1].bias)

    def forward(self, features):
        features = features.detach()
        if self.arm == "global_rank_mixture":
            order = features[..., 0].argsort(dim=-1, descending=True, stable=True)
            weights = self.logits.softmax(-1).expand_as(order)
            return torch.zeros_like(weights).scatter(-1, order, weights)
        return self.scorer(features).squeeze(-1).softmax(-1)


def auxiliary(q, p, negatives, arm, generator=None, tau=.1):
    # Preserve float64 for derivative tests; fp16/bf16 representations use fp32.
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, p, n = [F.normalize(x.to(dtype), dim=-1, eps=1e-8) for x in (q, p, negatives)]
    if n.shape[-2] != 4 or tau <= 0:
        raise ValueError("Exactly four negatives and a positive temperature are required")
    qn, pn = (q[:, None] * n).sum(-1), (p[:, None] * n).sum(-1)
    qp = (q * p).sum(-1)
    gram = n @ n.transpose(-1, -2)
    others = (gram.sum(-1) - gram.diagonal(dim1=-2, dim2=-1)) / 3
    features = torch.stack((qn, pn, others, qp[:, None].expand_as(qn)), -1).detach()
    if arm in LEARNED:
        if generator is None or generator.arm != arm:
            raise ValueError("Missing or mismatched learned generator")
        w = generator(features)
    elif arm == "uniform_barycenter":
        w = torch.full_like(qn, .25)
    elif arm == "similarity_barycenter":
        w = (qn.detach() / .1).softmax(-1)
    elif arm == "hardest_existing":
        w = F.one_hot(qn.detach().argmax(-1), 4).to(qn)
    else:
        raise ValueError(arm)
    mixture = (w[..., None] * n).sum(-2)
    z = F.normalize(mixture, dim=-1, eps=1e-8)
    if arm == "learned_loss_reweighting":
        per_example = (w * F.softplus((qn - qp[:, None]) / tau)).sum(-1)
    else:
        per_example = F.softplus(((q * z).sum(-1) - qp) / tau)
    with torch.no_grad():
        diagnostics = {
            "entropy": float(-(w * w.clamp_min(1e-30).log()).sum(-1).mean()),
            "max_weight": float(w.max(-1).values.mean()),
            "effective_support": float(w.square().sum(-1).reciprocal().mean()),
            "mixture_norm": float(mixture.norm(dim=-1).mean()),
            "nearest_distance": float((z[:, None] - n).norm(dim=-1).min(-1).values.mean()),
        }
    return per_example.mean(), diagnostics, w


def finite_gradients(gradients):
    return all(g is None or bool(torch.isfinite(g).all()) for g in gradients)


def functional_adamw(named_parameters, gradients, optimizer, max_norm=1.0):
    """Exact next dense AdamW parameters, without changing model or moments.

    None gradients are skipped (including step/decay), matching PyTorch. The
    safe sqrt branch preserves the forward value at zero and prevents 0*inf
    NaNs in mixed derivatives at zero Adam moments. No SGD/first-order shortcut.
    """
    params = dict(named_parameters)
    grads = dict(zip(params, gradients))
    if len(gradients) != len(params) or not finite_gradients(gradients):
        raise ValueError("Invalid or nonfinite gradients")
    present = [g for g in gradients if g is not None]
    if not present:
        return params
    norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in present]))
    factor = (max_norm / (norm + 1e-6)).clamp(max=1.)
    groups = {id(p): group for group in optimizer.param_groups for p in group["params"]}
    result = {}
    for name, p in params.items():
        g = grads[name]
        if g is None:
            result[name] = p
            continue
        group = groups[id(p)]
        if group.get("amsgrad") or group.get("maximize"):
            raise ValueError("Unsupported AdamW variant")
        state = optimizer.state.get(p, {})
        b1, b2 = group["betas"]
        g = g * factor
        m = b1 * state.get("exp_avg", torch.zeros_like(p)).detach() + (1-b1) * g
        v = b2 * state.get("exp_avg_sq", torch.zeros_like(p)).detach() + (1-b2) * g.square()
        step = float(state.get("step", 0)) + 1
        corrected_v = v / (1-b2**step)
        positive = corrected_v > 0
        root = torch.where(positive, torch.where(positive, corrected_v,
                                               torch.ones_like(v)).sqrt(), torch.zeros_like(v))
        result[name] = p * (1-group["lr"] * group["weight_decay"]) - group["lr"] * (
            m / (1-b1**step)) / (root + group["eps"])
    return result
