"""Resumable warmup -> score -> matched continuation experiment."""

from collections import Counter
from copy import deepcopy
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import statistics

import torch

from .core import build_candidates, candidate_losses, native_loss, validate_bank, KINDS
from .data import (ToyData, CUBData, ToyEncoder, LiveCLIP, digest, make_optimizer,
                   move, partitions, training_indices)
from .optim import gradients, flatten, cosine, less_feature, adamw_delta, optimizer_parameters
from .attribution import validation_head_features, directional_losses
from .bids import matrix_selections


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def snapshot(model, optimizer, step):
    return {"step": step, "model": {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad},
            "optimizer": cpu_tree(optimizer.state_dict())}


def restore(model, optimizer, state):
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    if expected != set(state["model"]):
        raise ValueError("Checkpoint trainable parameter layout mismatch")
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(state["model"][name])
    optimizer.load_state_dict(deepcopy(state["optimizer"]))
    optimizer.zero_grad(set_to_none=True)


def save_state(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def update(model, optimizer, batch, config, candidates=None):
    output = model(batch)
    loss = native_loss(output)
    if candidates:
        loss = loss + config["alpha"] * candidate_losses(output, candidates).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(optimizer_parameters(optimizer), config["max_grad_norm"], error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        model.logit_scale.clamp_(max=math.log(100))
    return float(loss.detach())


def validation_gradient(model, optimizer, batches, device):
    params = optimizer_parameters(optimizer)
    total = sum(len(b["ids"]) for b in batches)
    result = torch.zeros(sum(p.numel() for p in params), device=device)
    for batch in batches:
        g = flatten(gradients(native_loss(model(move(batch, device))), params))
        result.add_(g, alpha=len(batch["ids"]) / total)
    return result


@torch.no_grad()
def evaluate(model, batches, device):
    all_images, all_texts, losses = [], [], []
    direction_sums = {"text_to_image_loss": 0., "image_to_text_loss": 0.}
    total = sum(len(b["ids"]) for b in batches)
    for batch in batches:
        output = model(move(batch, device))
        losses.append(float(native_loss(output)) * len(batch["ids"]) / total)
        for direction, values in directional_losses(output).items():
            direction_sums[direction + "_loss"] += float(values.sum()) / total
        all_images.append(output[0].cpu())
        all_texts.append(output[1].cpu())
    scores = torch.cat(all_texts) @ torch.cat(all_images).T
    labels = torch.arange(len(scores))
    return {"native_loss": sum(losses), **direction_sums,
            "text_to_image_r1": float((scores.argmax(1) == labels).float().mean()),
            "image_to_text_r1": float((scores.argmax(0) == labels).float().mean()),
            "pool_size": total}


def score_batch(model, optimizer, batch, candidates, val_gradient, config, head_validation=None):
    params = optimizer_parameters(optimizer)
    output = model(batch)
    base = gradients(native_loss(output), params)
    base_map = dict(zip(params, base))
    base_delta = adamw_delta(optimizer, base_map, config["max_grad_norm"], model.logit_scale)
    aux_losses = candidate_losses(output, candidates)
    result = []
    # Match each candidate's coefficient in the selected-K average.
    coefficient = config["alpha"] / config["selected_per_batch"]
    for candidate, loss in zip(candidates, aux_losses):
        aux = gradients(coefficient * loss, params)
        total = tuple(g + a for g, a in zip(base, aux))
        aux_map, total_map = dict(zip(params, aux)), dict(zip(params, total))
        feature = less_feature(optimizer, aux_map)
        total_feature = less_feature(optimizer, total_map)
        delta = adamw_delta(optimizer, total_map, config["max_grad_norm"], model.logit_scale)
        increment = delta - base_delta
        record = {**candidate, "aux_loss": float(loss.detach()),
                       "aux_gradient_norm": float(flatten(aux).norm()),
                       "raw_aux_cosine": cosine(val_gradient, flatten(aux)),
                       "less_aux_cosine": cosine(val_gradient, feature),
                       "less_total_cosine": cosine(val_gradient, total_feature),
                       "marginal_cosine": cosine(val_gradient, -increment),
                       "marginal_predicted_gain": float(-torch.dot(val_gradient, increment)),
                       "candidate_coefficient": coefficient}
        if head_validation is not None:
            head = feature[:head_validation.shape[1]]
            if head.norm() <= 1e-20:
                raise ValueError("Undefined candidate projection-head cosine")
            record["head_influence"] = (head_validation @ (head / head.norm())).cpu().tolist()
        result.append(record)
    return result


def select(rows, count, key, largest=True):
    valid = [r for r in rows if r.get(key) is not None and math.isfinite(r[key])]
    if len(valid) < count:
        raise ValueError(f"Only {len(valid)} valid {key} scores for {count} selections")
    return sorted(valid, key=lambda r: ((-1 if largest else 1) * r[key], r["id"]))[:count]


def selections(scores, config, seed):
    k = config["selected_per_batch"]
    arms = {name: [] for name in ("native", "top_less", "bottom_less", "top_marginal", *KINDS)}
    arms.update({f"random_{i}": [] for i in range(config["random_repeats"])})
    for batch_id, rows in enumerate(scores):
        arms["native"].append([])
        arms["top_less"].append(select(rows, k, "less_aux_cosine"))
        arms["bottom_less"].append(select(rows, k, "less_aux_cosine", largest=False))
        arms["top_marginal"].append(select(rows, k, "marginal_predicted_gain"))
        for kind in KINDS:
            arms[kind].append(random.Random(seed + batch_id).sample([r for r in rows if r["kind"] == kind], k))
        for i in range(config["random_repeats"]):
            arms[f"random_{i}"].append(random.Random(seed + 1009 * i + batch_id).sample(rows, k))
    return arms


def selection_summary(scores, arms):
    pool = [r for batch in scores for r in batch]
    summary = {"pool_counts": dict(Counter(r["kind"] for r in pool)), "arms": {}, "score_distributions": {}}
    for name, batches in arms.items():
        chosen = [r for batch in batches for r in batch]
        counts = Counter(r["kind"] for r in chosen)
        summary["arms"][name] = {"count": len(chosen), "counts": dict(counts),
            "enrichment_over_pool": {k: (counts[k] / len(chosen)) / (summary["pool_counts"][k] / len(pool))
                                     for k in KINDS} if chosen else {}}
    for kind in KINDS:
        summary["score_distributions"][kind] = {}
        for key in ("less_aux_cosine", "raw_aux_cosine", "marginal_predicted_gain"):
            values = [r[key] for r in pool if r["kind"] == kind and r[key] is not None]
            summary["score_distributions"][kind][key] = {"n": len(values), "mean": statistics.mean(values) if values else None,
                "median": statistics.median(values) if values else None,
                "positive_fraction": sum(x > 0 for x in values) / len(values) if values else None}
    return summary


def validate_config(config):
    if config["backend"] not in {"toy", "cub"}:
        raise ValueError("Unsupported backend")
    for key in ("batch_size", "candidate_batches", "selected_per_batch", "continuation_steps", "random_repeats", "top_k", "iterations"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["batch_size"] < 2 or config["selected_per_batch"] > config["batch_size"]:
        raise ValueError("Invalid contrastive batch or selection budget")
    if config["continuation_steps"] % config["candidate_batches"]:
        raise ValueError("Continuation steps must cover whole cycles for equal candidate exposure")
    for key in ("alpha", "epsilon", "projection_lr", "encoder_lr", "scale_lr", "max_grad_norm"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if not math.isfinite(config["weight_decay"]) or config["weight_decay"] < 0:
        raise ValueError("Invalid weight decay")
    if not config["checkpoints"] or config["checkpoints"] != sorted(set(config["checkpoints"])) or any(type(s) is not int or s < 1 for s in config["checkpoints"]):
        raise ValueError("Checkpoints must be increasing positive update counts")
    if not config["seeds"] or len(config["seeds"]) != len(set(config["seeds"])):
        raise ValueError("Need unique training seeds")
    if config["backend"] == "cub" and any(config[k] < 2 for k in ("validation_size", "test_size")):
        raise ValueError("Validation and test pools need at least two images")


def run(config, output):
    validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if config["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: choose an A100 Colab runtime")
        if config.get("require_a100") and "A100" not in torch.cuda.get_device_name():
            raise RuntimeError("This configuration requires an A100")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    package_root = Path(__file__).parent
    code = {p.name: p.read_text() for p in sorted(package_root.glob("*.py"))}
    versions = {p: importlib.metadata.version(p) for p in ("torch", "numpy")}
    if config["backend"] == "cub":
        versions.update({p: importlib.metadata.version(p) for p in ("transformers", "datasets")})
    identity = {"config": config, "code_sha256": digest(code), "versions": versions}
    imported_banks = {str(seed): read_json(path) for seed, path in config.get("candidate_banks", {}).items()}
    identity["imported_bank_sha256"] = {seed: digest(bank) for seed, bank in imported_banks.items()}
    manifest_path = output / "run.json"
    if manifest_path.exists() and read_json(manifest_path)["identity"] != identity:
        raise ValueError("Run config, source, or package versions changed; use a new output directory")
    data = ToyData(config) if config["backend"] == "toy" else CUBData(config)
    if (output / "data.json").exists() and read_json(output / "data.json") != data.manifest:
        raise ValueError("Dataset changed since this run was started")
    write_json(output / "data.json", data.manifest)
    write_json(manifest_path, {"identity": identity, "python": platform.python_version(),
                              "device": torch.cuda.get_device_name() if config["device"] == "cuda" else "cpu",
                              "precision": "float32; TF32 disabled", "candidate_policy": "frozen support and coefficients, live embeddings",
                              "status": "running", "research_evidence": config["backend"] != "toy"})
    candidate_count = config["candidate_batches"] * config["batch_size"]
    if candidate_count > data.count("train"):
        raise ValueError("Candidate pool exceeds training split")
    candidate_batches = [data.batch("train", p) for p in partitions(candidate_count, config["batch_size"])]
    val_batches = [data.batch("validation", p) for p in partitions(data.count("validation"), config["batch_size"])]
    test_batches = [data.batch("test", p) for p in partitions(data.count("test"), config["batch_size"])]
    device = config["device"]
    for seed in config["seeds"]:
        seed_dir = output / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        torch.manual_seed(seed)
        random.seed(seed)
        model = (ToyEncoder() if config["backend"] == "toy" else LiveCLIP(config["model"], config.get("model_revision"))).to(device)
        model.eval()  # Deterministic forwards for scoring AND every training arm.
        optimizer = make_optimizer(model, config)
        model_info = {"base_revision": getattr(getattr(getattr(model, "clip", None), "config", None), "_commit_hash", None),
                      "parameters": [(n, list(p.shape)) for n, p in model.named_parameters() if p.requires_grad]}
        # JSON round-trip keeps tuple/list identity stable on resume.
        model_info = json.loads(json.dumps(model_info))
        if (seed_dir / "model.json").exists() and read_json(seed_dir / "model.json") != model_info:
            raise ValueError("Pretrained model revision or parameter layout changed")
        write_json(seed_dir / "model.json", model_info)
        last_step = 0
        previous_scores = []
        for step in config["checkpoints"]:
            checkpoint = seed_dir / f"checkpoint_{step}.pt"
            if checkpoint.exists():
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if state["step"] != step:
                    raise ValueError("Checkpoint update count mismatch")
                restore(model, optimizer, state)
            else:
                for training_step in range(last_step, step):
                    indices = training_indices(data.count("train"), config["batch_size"], training_step, seed)
                    loss = update(model, optimizer, move(data.batch("train", indices), device), config)
                    if (training_step + 1) % 25 == 0:
                        print(f"seed={seed} native warmup {training_step+1}/{step}, loss={loss:.6f}", flush=True)
                state = snapshot(model, optimizer, step)
                save_state(checkpoint, state)
            last_step = step
            bank_path = seed_dir / "candidates.json"
            if bank_path.exists():
                bank = read_json(bank_path)
            elif str(seed) in imported_banks:
                bank = imported_banks[str(seed)]
                validate_bank(bank, candidate_batches)
                write_json(bank_path, bank)
            else:
                with torch.no_grad():
                    bank = [build_candidates(model(move(b, device)), b["ids"], i,
                                             top_k=config["top_k"], epsilon=config["epsilon"], iterations=config["iterations"])
                            for i, b in enumerate(candidate_batches)]
                write_json(bank_path, bank)
            validate_bank(bank, candidate_batches)
            stage = seed_dir / f"step_{step}"
            stage.mkdir(exist_ok=True)
            missing_scores = any(not (stage / f"scores_batch_{i}.json").exists() for i in range(len(bank)))
            if missing_scores:
                val_gradient = validation_gradient(model, optimizer, val_batches, device)
                head_validation, columns = validation_head_features(model, optimizer, val_batches, device)
                write_json(stage / "validation_columns.json", columns)
            else:
                val_gradient, head_validation = None, None
                columns = read_json(stage / "validation_columns.json")
            scores = []
            for i, (batch, candidates) in enumerate(zip(candidate_batches, bank)):
                path = stage / f"scores_batch_{i}.json"
                if path.exists():
                    rows = read_json(path)
                    if len(rows) != len(candidates) or [{k: r[k] for k in c} for r, c in zip(rows, candidates)] != candidates:
                        raise ValueError("Saved scores do not match candidate bank")
                else:
                    rows = score_batch(model, optimizer, move(batch, device), candidates, val_gradient, config, head_validation)
                    write_json(path, rows)
                scores.append(rows)
                print(f"seed={seed} state={step} scored context {i+1}/{len(bank)}", flush=True)
            del val_gradient, head_validation
            previous_scores.append(scores)
            # Constant learning rates; same coefficient and same frozen candidate at every checkpoint.
            trajectory = []
            for i, rows in enumerate(scores):
                for j, row in enumerate(rows):
                    vals = [s[i][j]["less_aux_cosine"] for s in previous_scores]
                    trajectory.append({"id": row["id"], "checkpoints": config["checkpoints"][:len(previous_scores)],
                        "lr_weighted_less_score": sum(v * config["projection_lr"] for v in vals) if all(v is not None for v in vals) else None})
            write_json(stage / "trajectory_scores.json", trajectory)
            arms = selections(scores, config, seed)
            pooled_rows = [row for batch in scores for row in batch]
            matrix_arms, matrix_diagnostics = matrix_selections(pooled_rows, [c["task"] for c in columns], config["selected_per_batch"])
            for name, indices in matrix_arms.items():
                arms[name] = [[pooled_rows[i] for i in indices if pooled_rows[i]["batch"] == b] for b in range(len(scores))]
            write_json(stage / "bids_diagnostics.json", matrix_diagnostics)
            write_json(stage / "selection_summary.json", selection_summary(scores, arms))
            write_json(stage / "selections.json", {a: [[r["id"] for r in b] for b in rows] for a, rows in arms.items()})
            # Freeze selections before reading reporting outcomes.
            for arm, chosen in arms.items():
                path = stage / f"continuation_{arm}.json"
                if path.exists():
                    continue
                restore(model, optimizer, state)
                initial = evaluate(model, test_batches, device)
                for t in range(config["continuation_steps"]):
                    batch_id = t % len(candidate_batches)
                    update(model, optimizer, move(candidate_batches[batch_id], device), config, chosen[batch_id])
                final = evaluate(model, test_batches, device)
                result = {"seed": seed, "starting_update": step, "arm": arm,
                          "updates": config["continuation_steps"], "initial": initial, "final": final,
                          "auxiliary_exposures": 0 if arm == "native" else config["selected_per_batch"] * config["continuation_steps"]}
                write_json(path, result)
                print(f"seed={seed} state={step} arm={arm} test_loss={final['native_loss']:.6f}", flush=True)
            restore(model, optimizer, state)
        del model, optimizer
        if device == "cuda":
            torch.cuda.empty_cache()
    summarize(output)
    manifest = read_json(manifest_path)
    manifest["status"] = "complete"
    write_json(manifest_path, manifest)


def summarize(output):
    output = Path(output)
    manifest = read_json(output / "run.json")
    config = manifest["identity"]["config"]
    rows = []
    for seed in config["seeds"]:
        for step in config["checkpoints"]:
            stage = output / f"seed_{seed}" / f"step_{step}"
            base_path = stage / "continuation_native.json"
            if not base_path.exists():
                continue
            baseline = read_json(base_path)
            for path in sorted(stage.glob("continuation_*.json")):
                result = read_json(path)
                result["paired_delta"] = {k: result["final"][k] - baseline["final"][k]
                                          for k in ("native_loss", "text_to_image_loss", "image_to_text_loss", "text_to_image_r1", "image_to_text_r1")}
                rows.append(result)
    grouped = {}
    for row in rows:
        key = f'{row["starting_update"]}:{row["arm"]}'
        grouped.setdefault(key, []).append(row)
    aggregate = []
    for key, group in sorted(grouped.items()):
        values = [r["paired_delta"]["native_loss"] for r in group]
        aggregate.append({"state_arm": key, "training_seeds": len(group),
                          "mean_loss_delta": statistics.mean(values),
                          "seed_std_loss_delta": statistics.stdev(values) if len(values) > 1 else None,
                          "per_seed_loss_delta": {str(r["seed"]): r["paired_delta"]["native_loss"] for r in group},
                          "mean_t2i_r1_delta": statistics.mean(r["paired_delta"]["text_to_image_r1"] for r in group),
                          "mean_i2t_r1_delta": statistics.mean(r["paired_delta"]["image_to_text_r1"] for r in group),
                          "mean_t2i_loss_delta": statistics.mean(r["paired_delta"]["text_to_image_loss"] for r in group),
                          "mean_i2t_loss_delta": statistics.mean(r["paired_delta"]["image_to_text_loss"] for r in group)})
    for entry in aggregate:
        group = grouped[entry["state_arm"]]
        paired = []
        for row in group:
            random_rows = [r for r in rows if r["seed"] == row["seed"] and r["starting_update"] == row["starting_update"] and r["arm"].startswith("random_")]
            if len(random_rows) == config["random_repeats"]:
                paired.append(row["final"]["native_loss"] - statistics.mean(r["final"]["native_loss"] for r in random_rows))
        entry["mean_loss_delta_vs_seed_mean_random"] = statistics.mean(paired) if paired else None
    write_json(output / "summary.json", {"research_evidence": config["backend"] != "toy", "paired_results": rows, "aggregate": aggregate})
    lines = ["# SynLess experiment report", "",
             "TOY SMOKE TEST ONLY — these numbers are not evidence about SCSS negatives." if config["backend"] == "toy" else
             "CLIP/CUB pilot; canonical-caption retrieval on the recorded reporting subset.", "",
             "Loss deltas are treatment minus a matched native continuation; negative is helpful. R@1 deltas are fractions.", "",
             "| Starting state / arm | Seeds | Mean loss delta | Seed SD | T→I R@1 delta | I→T R@1 delta |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in aggregate:
        sd = f'{r["seed_std_loss_delta"]:.6g}' if r["seed_std_loss_delta"] is not None else "n/a"
        lines.append(f'| {r["state_arm"]} | {r["training_seeds"]} | {r["mean_loss_delta"]:+.6g} | {sd} | {r["mean_t2i_r1_delta"]:+.6g} | {r["mean_i2t_r1_delta"]:+.6g} |')
    lines += ["", "## Which constructions were selected?", "",
              "Mean per-seed enrichment relative to candidate-pool prevalence; 1 means no preference.", "",
              "| State / selector | OT | Uniform | Hardest real |", "|---|---:|---:|---:|"]
    for step in config["checkpoints"]:
        for arm in ("top_less", "less_head_task_max", "bids", "top_marginal"):
            available = []
            for seed in config["seeds"]:
                path = output / f"seed_{seed}" / f"step_{step}" / "selection_summary.json"
                if path.exists():
                    available.append(read_json(path)["arms"][arm]["enrichment_over_pool"])
            if available:
                means = [statistics.mean(a[k] for a in available) for k in KINDS]
                lines.append(f"| {step}:{arm} | {means[0]:.3f} | {means[1]:.3f} | {means[2]:.3f} |")
    lines += ["", "Inspect each stage's selection_summary.json for construction enrichment and score distributions.",
              "BIDS and less_head_* use the same exact projection-head attribution matrix; top_less uses all trainable parameters.",
              "Training selectors have equal per-context capacities. Unconstrained BIDS/LESS selections are retained as diagnostics.",
              "Random repeats share a training seed and are not independent seed replications.",
              "Selection validation is disjoint from reporting. Prior SCSS use of CUB means the reporting set is not historically untouched.",
              "A high rank is a prediction, not proof of improvement. This pilot does not establish full-dataset retrieval gains."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
