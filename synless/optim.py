"""Separate the released LESS feature from the actual AdamW intervention."""

import math

import torch


def gradients(loss, parameters):
    values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return tuple(torch.zeros_like(p) if g is None else g.detach() for p, g in zip(parameters, values))


def flatten(values):
    return torch.cat([g.flatten() for g in values])


def cosine(a, b):
    norm = a.norm() * b.norm()
    if not torch.isfinite(norm):
        raise ValueError("Nonfinite gradient feature")
    return None if norm <= 1e-20 else float(torch.dot(a, b) / norm)


def less_feature(optimizer, gradients_by_parameter):
    """Released LESS convention: updated m / sqrt(updated v + 1e-8).

    No bias correction, clipping, learning rate, or weight decay. This deliberately
    differs from torch AdamW, whose exact step is used for the marginal score.
    Model parameters with a structurally absent auxiliary gradient receive zero
    gradient in a fixed common parameter layout (including the scale parameter).
    """
    values = []
    for group in optimizer.param_groups:
        if group["betas"] != (0.9, 0.999):
            raise ValueError("Released LESS feature requires betas=(0.9, 0.999)")
        for p in group["params"]:
            state = optimizer.state[p]
            g = gradients_by_parameter[p]
            m = state.get("exp_avg", torch.zeros_like(p))
            v = state.get("exp_avg_sq", torch.zeros_like(p))
            values.append((0.9 * m + 0.1 * g) / (0.999 * v + 0.001 * g.square() + 1e-8).sqrt())
    return flatten(values)


def adamw_delta(optimizer, gradients_by_parameter, max_norm=1.0, clamp_parameter=None):
    """Predict torch AdamW's complete next delta without mutating any state.

    Parameters are returned in optimizer group order. Includes global clipping,
    bias correction, group learning rates, decoupled decay and CLIP scale clamp.
    Requires dense gradients and the standard (non-AMSGrad/non-maximize) optimizer.
    """
    norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in gradients_by_parameter.values()]))
    factor = (max_norm / (norm + 1e-6)).clamp(max=1.0)
    values = []
    for group in optimizer.param_groups:
        if group.get("amsgrad") or group.get("maximize"):
            raise ValueError("Unsupported AdamW variant")
        b1, b2 = group["betas"]
        for p in group["params"]:
            state = optimizer.state[p]
            g = gradients_by_parameter[p] * factor
            m = b1 * state.get("exp_avg", torch.zeros_like(p)) + (1 - b1) * g
            v = b2 * state.get("exp_avg_sq", torch.zeros_like(p)) + (1 - b2) * g.square()
            step = float(state.get("step", 0)) + 1
            update = (m / (1 - b1**step)) / ((v / (1 - b2**step)).sqrt() + group["eps"])
            after = p.detach() * (1 - group["lr"] * group["weight_decay"]) - group["lr"] * update
            if p is clamp_parameter:
                after = after.clamp(max=math.log(100))
            values.append(after - p.detach())
    return flatten(values)


def optimizer_parameters(optimizer):
    return tuple(p for group in optimizer.param_groups for p in group["params"])
