"""Exact per-instance gradient features in the two live CLIP projection heads."""

import torch
import torch.nn.functional as F

from .data import move
from .optim import gradients, flatten


def directional_losses(output):
    images, texts, scale = output
    logits = scale * (texts @ images.T)
    targets = torch.arange(len(images), device=images.device)
    return {"text_to_image": F.cross_entropy(logits, targets, reduction="none"),
            "image_to_text": F.cross_entropy(logits.T, targets, reduction="none")}


def validation_head_features(model, optimizer, batches, device):
    if optimizer.param_groups[0].get("name") != "projection":
        raise ValueError("Projection heads must be the first optimizer group")
    parameters = tuple(optimizer.param_groups[0]["params"])
    size = sum(p.numel() for p in parameters)
    count = 2 * sum(len(b["ids"]) for b in batches)
    # Preallocate to avoid a second complete copy of the ~1.3 GB CLIP matrix.
    features = torch.empty((count, size), device=device)
    columns, offset = [], 0
    for batch_index, batch in enumerate(batches):
        output = model(move(batch, device))
        for direction, losses in directional_losses(output).items():
            for row, loss in enumerate(losses):
                vector = flatten(gradients(loss, parameters))
                norm = vector.norm()
                if not torch.isfinite(norm) or norm <= 1e-20:
                    raise ValueError("Undefined validation gradient cosine")
                features[offset].copy_(vector / norm)
                columns.append({"image_id": batch["ids"][row], "context": batch_index, "row": row,
                                "task": direction, "parameter_space": "visual_and_text_projection_heads"})
                offset += 1
        print(f"validation instance features {offset}/{count}", flush=True)
    return features, columns
