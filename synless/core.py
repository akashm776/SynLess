"""SCSS-compatible constructions; candidate identities include the batch context."""

import hashlib
import json

import torch
import torch.nn.functional as F

KINDS = ("ot", "uniform", "hardest_real")


def native_loss(output):
    images, texts, scale = output
    logits = scale * (texts @ images.T)
    labels = torch.arange(len(images), device=images.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def historical_plan(scores, top_k=8, epsilon=0.049, iterations=30):
    """Match SCSS historical_sparse_ot, including its dense floor and final mask.

    Provenance: akashm776/SCSS src/clip_geometry_v2_metrics.py.
    No gradients pass through support selection or transport coefficients.
    """
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1] or len(scores) < 2:
        raise ValueError("Need a square similarity matrix with at least two pairs")
    if top_k < 1 or epsilon <= 0 or iterations < 1 or not torch.isfinite(scores).all():
        raise ValueError("Invalid transport inputs")
    scores = scores.detach().float()
    diagonal = torch.eye(len(scores), device=scores.device, dtype=torch.bool)
    masked = scores.masked_fill(diagonal, -torch.inf)
    support = torch.zeros_like(diagonal)
    support.scatter_(1, masked.topk(min(top_k, len(scores) - 1), dim=1).indices, True)
    affinity = torch.exp(-(masked.max() - scores).clamp_min(0) / epsilon)
    kernel = (affinity * support).clamp_min(1e-12)
    target = torch.full((len(scores),), 1 / len(scores), device=scores.device)
    u, v = torch.ones_like(target), torch.ones_like(target)
    for _ in range(iterations):
        u = target / (kernel @ v + 1e-8)
        v = target / (kernel.T @ u + 1e-8)
    plan = u[:, None] * kernel * v[None, :] * support
    if (plan.sum(1) <= 0).any():
        raise ValueError("Degenerate transport row")
    return plan / plan.sum(1, keepdim=True).clamp_min(1e-8), support


def build_candidates(output, ids, batch_id, **transport):
    images, texts, _ = output
    if len(ids) != len(images) or len(set(ids)) != len(ids):
        raise ValueError("Every pair in a context must have a unique stable image ID")
    raw = texts.detach() @ images.detach().T
    weights, support = historical_plan(raw, **transport)
    uniform = support.float() / support.sum(1, keepdim=True)
    hardest = raw.masked_fill(torch.eye(len(raw), dtype=torch.bool, device=raw.device), -torch.inf).argmax(1)
    context = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    candidates = []
    for row in range(len(ids)):
        for kind in KINDS:
            if kind == "hardest_real":
                indices, coefficients = [int(hardest[row])], [1.0]
            else:
                indices = support[row].nonzero().flatten().tolist()
                coefficients = (weights if kind == "ot" else uniform)[row, indices].tolist()
            candidates.append({"id": f"{batch_id}:{row}:{kind}", "batch": batch_id,
                               "row": row, "kind": kind, "query_id": ids[row],
                               "context_sha256": context, "indices": indices,
                               "source_ids": [ids[i] for i in indices], "weights": coefficients})
    return candidates


def candidate_losses(output, candidates):
    """One relative-denominator loss per candidate, without detaching source images.

    Several selected candidates on one row are separate auxiliary losses; they
    are NOT treated as a jointly expanded denominator.
    """
    images, texts, scale = output
    scale = scale.detach()
    logits = (texts @ images.T) * scale
    rows = torch.tensor([c["row"] for c in candidates], device=images.device)
    mixing = images.new_zeros((len(candidates), len(images)))
    for i, c in enumerate(candidates):
        mixing[i, c["indices"]] = images.new_tensor(c["weights"])
    synthetic = F.normalize(mixing @ images, dim=-1)
    extra = (texts[rows] * synthetic).sum(-1) * scale
    return F.softplus(extra - torch.logsumexp(logits[rows], dim=1))


def validate_bank(bank, batches):
    seen = set()
    for batch_id, (candidates, batch) in enumerate(zip(bank, batches)):
        ids = batch["ids"]
        expected_context = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        expected = {f"{batch_id}:{r}:{k}" for r in range(len(ids)) for k in KINDS}
        if {c["id"] for c in candidates} != expected or len(candidates) != len(expected):
            raise ValueError("Candidate bank must contain every row and construction exactly once")
        for c in candidates:
            idx, weights = c["indices"], c["weights"]
            if (c["id"] in seen or c["context_sha256"] != expected_context
                    or c["batch"] != batch_id or c["kind"] not in KINDS
                    or c["id"] != f'{batch_id}:{c["row"]}:{c["kind"]}'
                    or not 0 <= c["row"] < len(ids) or c["query_id"] != ids[c["row"]]
                    or not idx or len(idx) != len(set(idx)) or len(idx) != len(weights)
                    or any(i < 0 or i >= len(ids) or i == c["row"] for i in idx)
                    or c["source_ids"] != [ids[i] for i in idx]
                    or not all(0 <= w <= 1 for w in weights)
                    or abs(sum(weights) - 1) > 1e-5):
                raise ValueError("Invalid or mismatched candidate bank")
            seen.add(c["id"])
    if len(bank) != len(batches):
        raise ValueError("Candidate batch count mismatch")
