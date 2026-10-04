"""Disjoint image-level splits, deterministic contexts, and live encoders."""

import hashlib
import json
import random
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class ToyEncoder(nn.Module):
    """Small differentiable dual encoder used only for plumbing validation."""
    def __init__(self):
        super().__init__()
        self.visual_projection = nn.Linear(12, 8, bias=False)
        self.text_projection = nn.Linear(12, 8, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        return (F.normalize(self.visual_projection(batch["images"]), dim=-1),
                F.normalize(self.text_projection(batch["texts"]), dim=-1), self.logit_scale.exp())


class LiveCLIP(nn.Module):
    def __init__(self, checkpoint, revision=None):
        super().__init__()
        from transformers import CLIPModel
        self.clip = CLIPModel.from_pretrained(checkpoint, revision=revision)
        self.clip.requires_grad_(False)
        for module in (self.clip.visual_projection, self.clip.text_projection,
                       self.clip.vision_model.encoder.layers[-1], self.clip.text_model.encoder.layers[-1]):
            module.requires_grad_(True)
        self.clip.logit_scale.requires_grad_(True)

    @property
    def logit_scale(self):
        return self.clip.logit_scale

    def forward(self, batch):
        # Direct towers avoid version-dependent get_*_features return types.
        image = self.clip.vision_model(pixel_values=batch["pixel_values"]).pooler_output
        text = self.clip.text_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).pooler_output
        return (F.normalize(self.clip.visual_projection(image), dim=-1),
                F.normalize(self.clip.text_projection(text), dim=-1), self.logit_scale.exp())


def make_optimizer(model, config):
    groups = {"projection": [], "encoder": [], "scale": []}
    for name, p in model.named_parameters():
        if p.requires_grad:
            key = "projection" if "projection" in name else ("scale" if name.endswith("logit_scale") else "encoder")
            groups[key].append(p)
    return torch.optim.AdamW([
        {"params": values, "lr": config[f"{key}_lr"], "name": key}
        for key, values in groups.items() if values
    ], weight_decay=config["weight_decay"], foreach=False)


def move(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


class ToyData:
    def __init__(self, config):
        gen = torch.Generator().manual_seed(config["data_seed"])
        self.values = {}
        n = max(config["candidate_batches"] * config["batch_size"], 32)
        for split, count in (("train", n), ("validation", 32), ("test", 32)):
            latent = torch.randn(count, 12, generator=gen)
            self.values[split] = {"images": latent + 0.2 * torch.randn(count, 12, generator=gen),
                                  "texts": latent + 0.2 * torch.randn(count, 12, generator=gen),
                                  "ids": [f"{split}:{i}" for i in range(count)]}
        self.manifest = {"backend": "toy", "warning": "Synthetic plumbing check, not research evidence",
                         "splits": {s: v["ids"] for s, v in self.values.items()}}

    def count(self, split):
        return len(self.values[split]["ids"])

    def batch(self, split, indices):
        data = self.values[split]
        return {k: [v[i] for i in indices] if k == "ids" else v[indices] for k, v in data.items()}


class CUBData:
    """One canonical caption per image; selection validation comes from train.

    The official test split is used only for final reporting. It is not claimed
    to be historically untouched, because prior SCSS experiments used CUB.
    """
    def __init__(self, config):
        from datasets import load_dataset
        from transformers import CLIPProcessor
        self.processor = CLIPProcessor.from_pretrained(config["model"], revision=config.get("model_revision"))
        dataset = load_dataset(config["dataset"], revision=config.get("dataset_revision"), trust_remote_code=True)
        train, test = dataset["train"], dataset["test"]

        def records(raw):
            found = {}
            for i, (key, description) in enumerate(zip(raw["file_name"], raw["description"])):
                captions = [x.strip() for x in description.split("\n") if x.strip()]
                if captions:
                    found.setdefault(str(key), (i, captions[0]))
            return [(key, *value) for key, value in sorted(found.items())]

        training, reporting = records(train), records(test)
        if set(x[0] for x in training) & set(x[0] for x in reporting):
            raise ValueError("Train/test image IDs overlap")
        random.Random(config["data_seed"]).shuffle(training)
        random.Random(config["data_seed"] + 1).shuffle(reporting)
        nval, ntest = config["validation_size"], config["test_size"]
        if nval >= len(training) or ntest > len(reporting):
            raise ValueError("Requested split sizes exceed dataset")
        self.records = {"train": training[nval:], "validation": training[:nval], "test": reporting[:ntest]}
        self.raw = {"train": train, "validation": train, "test": test}
        self.manifest = {"backend": "cub", "dataset": config["dataset"],
                         "fingerprints": {k: v._fingerprint for k, v in self.raw.items()},
                         "splits": {k: [{"id": r[0], "row": r[1], "caption": r[2]} for r in v]
                                    for k, v in self.records.items()}}

    def count(self, split):
        return len(self.records[split])

    def batch(self, split, indices):
        from PIL import Image
        import io
        records = [self.records[split][i] for i in indices]
        images = []
        for _, index, _ in records:
            value = self.raw[split][index]["image"]
            if isinstance(value, dict):
                value = Image.open(io.BytesIO(value["bytes"])) if value.get("bytes") else Image.open(value["path"])
            images.append(value.convert("RGB"))
        result = dict(self.processor(images=images, text=[r[2] for r in records],
                                     return_tensors="pt", padding=True, truncation=True))
        result["ids"] = [r[0] for r in records]
        return result


def partitions(count, batch_size):
    if count < 2 or batch_size < 2:
        raise ValueError("Contrastive batches need at least two examples")
    chunks = [list(range(i, min(i + batch_size, count))) for i in range(0, count, batch_size)]
    if len(chunks[-1]) == 1:
        chunks[-2].extend(chunks.pop())
    return chunks


def training_indices(count, batch_size, step, seed):
    # Stateless deterministic sampler; supports restarting from saved updates.
    if count < batch_size:
        raise ValueError("Training split smaller than a batch")
    return random.Random(seed * 1_000_003 + step).sample(range(count), batch_size)
