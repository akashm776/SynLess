"""Resumable, protocol-locked LLM pilot. Official reporting is a separate command."""

from contextlib import contextmanager
from copy import deepcopy
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import torch

from .core import ARMS, LEARNED, Generator, finite_gradients, functional_adamw
from .data import build_records, digest, scheduled_batch
from .model import load_tokenizer, load_learner, make_optimizer, objective, update, meta_objective, evaluate


DEFAULTS = {
    "backend": "hf", "model": "Qwen/Qwen2.5-1.5B", "model_revision": "main",
    "dataset": "openai/gsm8k", "dataset_revision": "main", "device": "cuda",
    "require_a100": True, "layer": 14, "rank": 8, "lora_alpha": 16,
    "lr": 1e-4, "weight_decay": 0., "max_grad_norm": 1., "alpha": .01, "tau": .1,
    "outer_lr": 1e-3, "max_length": 512, "max_new_tokens": 32,
    "batch_size": 4, "meta_batch_size": 1, "outer_batch_size": 1,
    "states": [100, 500], "horizons": [1, 5, 20], "meta_episodes": 200,
    "seed_triples": [[101, 11, 789], [102, 22, 2026], [103, 33, 31415]],
    "split_seed": 1701, "checkpoint_interval": 10, "min_memory_headroom_gb": 4.,
}


def validate_config(supplied):
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown LLM settings: {sorted(unknown)}")
    config = {**deepcopy(DEFAULTS), **supplied}
    if config["backend"] not in ("hf", "tiny"):
        raise ValueError("backend must be hf or tiny")
    if config["backend"] == "hf" and config["device"] != "cuda":
        raise ValueError("Real-model pilot requires CUDA; use tiny for offline checks")
    for key in ("states", "horizons"):
        if not config[key] or config[key] != sorted(set(config[key])) or min(config[key]) < 1:
            raise ValueError(f"{key} must be sorted distinct positive update numbers")
    for key in ("layer", "rank", "max_length", "max_new_tokens", "batch_size", "meta_batch_size",
                "outer_batch_size", "meta_episodes", "checkpoint_interval"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"Invalid {key}")
    if config["alpha"] < 0 or config["tau"] <= 0 or config["max_grad_norm"] <= 0:
        raise ValueError("Invalid loss/gradient scaling")
    triples = config["seed_triples"]
    if not triples or any(len(x) != 3 for x in triples) or len({tuple(x) for x in triples}) != len(triples):
        raise ValueError("Expected distinct teacher/meta/learner seed triples")
    if len({x[2] for x in triples}) != len(triples):
        raise ValueError("Learner seed IDs must be unique")
    return config


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu(v) for v in value)
    return deepcopy(value)


def snapshot(learner, optimizer):
    return {"parameters": cpu(learner.trainable()), "optimizer": cpu(optimizer.state_dict()),
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore(learner, optimizer, state):
    params = learner.trainable()
    if set(params) != set(state["parameters"]):
        raise ValueError("Checkpoint parameter layout mismatch")
    with torch.no_grad():
        for name, p in params.items():
            p.copy_(state["parameters"][name])
    # PyTorch may reuse same-device moment tensors from the supplied state dict.
    # Copy before loading so one continuation cannot mutate its shared parent.
    optimizer.load_state_dict(deepcopy(state["optimizer"]))
    optimizer.zero_grad(set_to_none=True)
    torch.set_rng_state(state["rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def source_hash():
    directory = Path(__file__).parent
    return digest({p.name: p.read_text() for p in sorted(directory.glob("*.py"))})


def hardware(config):
    if config["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: select a Colab A100 runtime")
        name = torch.cuda.get_device_name()
        if config["require_a100"] and "A100" not in name:
            raise RuntimeError(f"A100 required, got {name}")
        return name
    return "CPU"


@contextmanager
def run_lock(output):
    """Kernel-released advisory lock: Colab disconnects cannot leave stale PID locks."""
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is using this run directory") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def prepare(config, output):
    device_name = hardware(config)
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")}
    if config["backend"] == "hf":
        versions["datasets"] = importlib.metadata.version("datasets")
        versions["huggingface-hub"] = importlib.metadata.version("huggingface-hub")
        versions["tokenizers"] = importlib.metadata.version("tokenizers")
        versions["accelerate"] = importlib.metadata.version("accelerate")
    identity = {"config": config, "source_sha256": source_hash(), "versions": versions,
                "python": platform.python_version(), "device": device_name}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if identity != manifest["identity"]:
            raise ValueError("Config, source, hardware, or environment changed: use a new output directory")
        resolved = manifest["resolved"]
    else:
        if config["backend"] == "tiny":
            resolved = {"model_revision": "offline-tiny-qwen2", "dataset_revision": "synthetic-arithmetic-v1"}
        else:
            from huggingface_hub import HfApi
            api = HfApi()
            resolved = {"model_revision": api.model_info(config["model"], revision=config["model_revision"]).sha,
                        "dataset_revision": api.dataset_info(config["dataset"], revision=config["dataset_revision"]).sha}
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[2], text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            commit = None
        manifest = {"identity": identity, "resolved": resolved, "git_commit": commit,
                    "research_evidence": config["backend"] == "hf", "status": "prepared",
                    "precision": "fp32" if config["backend"] == "tiny" else "frozen bf16 base; fp32 LoRA/generator/moments",
                    "attention": "eager", "protocol_version": "0.2"}
        atomic_json(manifest_path, manifest)
        atomic_json(output / "protocol.json", config)
    tokenizer = load_tokenizer(config, resolved)
    data_path = output / "data.json"
    if data_path.exists():
        data = read_json(data_path)
        if manifest.get("data_sha256") and digest(data) != manifest["data_sha256"]:
            raise ValueError("Prepared data was modified")
    else:
        if config["backend"] == "tiny":
            train = [{"question": f"What is {i} plus 1?", "answer": f"#### {i+1}"} for i in range(40)]
            test = [{"question": f"What is {i} plus 1?", "answer": f"#### {i+1}"} for i in range(60, 63)]
            fingerprint = "offline-arithmetic"
        else:
            from datasets import load_dataset
            dataset = load_dataset(config["dataset"], "main", revision=resolved["dataset_revision"])
            train, test = list(dataset["train"]), list(dataset["test"])
            fingerprint = {name: dataset[name]._fingerprint for name in ("train", "test")}
        partitions, audits = build_records(train, test, tokenizer, config["max_length"], config["split_seed"])
        data = {"partitions": partitions, "audit": audits, "dataset_fingerprint": fingerprint}
        atomic_json(data_path, data)
    manifest["data_sha256"] = digest(data)
    atomic_json(manifest_path, manifest)
    atomic_json(output / "splits.json", {name: [{"id": r["id"], "question_hash": r["question_hash"]} for r in rows]
                                        for name, rows in data["partitions"].items()})
    atomic_json(output / "candidate_audit.json", data["audit"])
    return tokenizer, data["partitions"], manifest


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(function):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    sync()
    start = time.perf_counter()
    result = function()
    sync()
    stats = {"seconds": time.perf_counter()-start,
             "peak_allocated_gb": torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else 0.,
             "peak_reserved_gb": torch.cuda.max_memory_reserved()/2**30 if torch.cuda.is_available() else 0.}
    return result, stats


def preflight(config, output, tokenizer, partitions, manifest):
    path = output / "preflight.json"
    # A failed/OOM attempt must invalidate an older success.
    atomic_json(path, {"passed": False, "status": "running"})
    learner = load_learner(config, manifest["resolved"], config["seed_triples"][0][0])
    optimizer = make_optimizer(learner, config)
    longest = sorted(partitions["A"], key=lambda r: max(len(e["input_ids"]) for e in r["negatives"]), reverse=True)
    outer = sorted(partitions["M"], key=lambda r: len(r["correct"]["input_ids"]), reverse=True)[:config["outer_batch_size"]]
    inner = longest[:config["meta_batch_size"]]
    batch = longest[:config["batch_size"]]
    for _ in range(2):
        update(learner, optimizer, inner, tokenizer, config)
    state = snapshot(learner, optimizer)
    stats = {}
    for arm in ("native", "uniform_barycenter"):
        restore(learner, optimizer, state)
        metrics, timing = timed(lambda: update(learner, optimizer, batch, tokenizer, config, arm))
        stats[arm] = {**metrics, **timing}
    meta_norms = {}
    for arm in LEARNED:
        restore(learner, optimizer, state)
        generator = Generator(arm).to(config["device"])
        def check_meta():
            loss, metrics = meta_objective(learner, optimizer, inner, outer, tokenizer, config, arm, generator)
            gradients = torch.autograd.grad(loss, tuple(generator.parameters()), allow_unused=True)
            if not finite_gradients(gradients):
                raise FloatingPointError("Nonfinite meta-gradient")
            norm = sum(float(g.detach().square().sum()) for g in gradients if g is not None)**.5
            if norm <= 0:
                raise RuntimeError("Generator received no learning signal")
            meta_norms[arm] = norm
            return metrics
        metrics, timing = timed(check_meta)
        stats[arm] = {**metrics, **timing}
        after = snapshot(learner, optimizer)
        if any(digest_tensor_tree(state[key]) != digest_tensor_tree(after[key]) for key in ("parameters", "optimizer")):
            raise RuntimeError("Virtual step mutated teacher state")
        del generator
    # Diagnostic auxiliary/native gradients on the same live state.
    restore(learner, optimizer, state)
    params = tuple(learner.trainable().values())
    native, _ = objective(learner, inner, tokenizer, config)
    ng = torch.autograd.grad(native, params, allow_unused=True)
    total, _ = objective(learner, inner, tokenizer, config, "uniform_barycenter")
    tg = torch.autograd.grad(total, params, allow_unused=True)
    native_norm = sum(float(g.square().sum()) for g in ng if g is not None)**.5
    aux_norm = sum(float(((t if t is not None else torch.zeros_like(p)) -
                          (n if n is not None else torch.zeros_like(p))).square().sum())
                   for p, n, t in zip(params, ng, tg))**.5
    layer_norms = {}
    for (name, p), n, t in zip(learner.trainable().items(), ng, tg):
        layer = int(name.split(".layers.")[1].split(".")[0]) + 1
        entry = layer_norms.setdefault(str(layer), {"native_squared": 0., "weighted_aux_squared": 0.})
        entry["native_squared"] += 0. if n is None else float(n.square().sum())
        diff = (t if t is not None else torch.zeros_like(p)) - (n if n is not None else torch.zeros_like(p))
        entry["weighted_aux_squared"] += float(diff.square().sum())
    if not any(v["weighted_aux_squared"] > 0 for l, v in layer_norms.items() if int(l) <= config["layer"]):
        raise RuntimeError("Auxiliary gradient does not reach the tapped/lower layers")
    if any(v["weighted_aux_squared"] > 1e-12 for l, v in layer_norms.items() if int(l) > config["layer"]):
        raise RuntimeError("Unexpected auxiliary gradient above the tapped layer")
    total_memory = torch.cuda.get_device_properties(0).total_memory/2**30 if config["device"] == "cuda" else None
    peak = max(v["peak_reserved_gb"] for v in stats.values())
    passed = total_memory is None or total_memory-peak >= config["min_memory_headroom_gb"]
    result = {"passed": passed, "status": "complete", "identity_hash": digest(manifest["identity"]),
              "data_sha256": manifest["data_sha256"], "measurements": stats, "meta_gradient_norms": meta_norms,
              "weighted_aux_to_native_gradient_ratio": aux_norm/max(native_norm, 1e-30),
              "per_layer_gradient_diagnostics": layer_norms,
              "total_memory_gb": total_memory, "remaining_headroom_gb": None if total_memory is None else total_memory-peak,
              "test_evaluated": False, "longest_training_tokens": max(len(r["correct"]["input_ids"]) for r in longest),
              "limitations": "Resource check only; utility untested. bf16 virtual forward can round small fp32 adapter updates."}
    atomic_json(path, result)
    print(json.dumps(result, indent=2), flush=True)
    del learner, optimizer
    gc.collect()
    if not passed:
        raise RuntimeError("Insufficient memory headroom: revise protocol/config in a new output directory")
    return result


def digest_tensor_tree(value):
    if torch.is_tensor(value):
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return digest([str(value.dtype), list(value.shape), hashlib.sha256(raw).hexdigest()])
    if isinstance(value, dict):
        return digest({str(k): digest_tensor_tree(v) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        return digest([digest_tensor_tree(v) for v in value])
    return digest(value)


class Paused(Exception):
    pass


class Budget:
    def __init__(self, maximum=None):
        self.maximum, self.count = maximum, 0

    def tick(self):
        self.count += 1
        if self.maximum is not None and self.count >= self.maximum:
            raise Paused("Stopped at requested work-unit limit; rerun the same command to resume")


def save_progress(path, learner, optimizer, step, history):
    atomic_torch(path, {"state": snapshot(learner, optimizer), "step": step, "history": history})


def warmup(learner, optimizer, rows, tokenizer, config, directory, seed, budget):
    directory.mkdir(parents=True, exist_ok=True)
    latest = directory / "latest.pt"
    start, history = 0, []
    if latest.exists():
        saved = load_checkpoint(latest)
        restore(learner, optimizer, saved["state"])
        start, history = saved["step"], saved["history"]
    for index in range(start, max(config["states"])):
        batch = scheduled_batch(rows, index, config["batch_size"], seed)
        metrics, resources = timed(lambda: update(learner, optimizer, batch, tokenizer, config))
        history.append({"step": index+1, **metrics, **resources})
        step = index+1
        if step in config["states"]:
            save_progress(directory / f"step_{step}.pt", learner, optimizer, step, history)
        if step % config["checkpoint_interval"] == 0 or step in config["states"] or budget.maximum is not None:
            save_progress(latest, learner, optimizer, step, history)
        if step % config["checkpoint_interval"] == 0:
            print(f"{directory.name}: native update {step}/{max(config['states'])}", flush=True)
        budget.tick()
    for step in config["states"]:
        if not (directory / f"step_{step}.pt").exists():
            raise RuntimeError("Missing warmup checkpoint; do not resume an incomplete/corrupt artifact set")


def meta_train(learner, optimizer, partitions, tokenizer, config, directory, teacher_dir, seed, arm, budget):
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    generator = Generator(arm).to(config["device"])
    outer_optimizer = torch.optim.AdamW(generator.parameters(), lr=config["outer_lr"], betas=(.9, .999),
                                        eps=1e-8, weight_decay=0., foreach=False)
    path = directory / f"{arm}.pt"
    start, history = 0, []
    if path.exists():
        saved = load_checkpoint(path)
        generator.load_state_dict(saved["generator"])
        outer_optimizer.load_state_dict(saved["optimizer"])
        start, history = saved["episode"], saved["history"]
        torch.set_rng_state(saved["rng"])
    for episode in range(start, config["meta_episodes"]):
        state = config["states"][episode % len(config["states"])]
        teacher = load_checkpoint(teacher_dir / f"step_{state}.pt")
        restore(learner, optimizer, teacher["state"])
        inner = scheduled_batch(partitions["A"], episode, config["meta_batch_size"], seed)
        outer = scheduled_batch(partitions["M"], episode, config["outer_batch_size"], seed+10000)
        def episode_update():
            outer_optimizer.zero_grad(set_to_none=True)
            # Native-only virtual update on identical data/moments; no meta feedback.
            native, _ = objective(learner, inner, tokenizer, config)
            parameters = learner.trainable()
            ng = torch.autograd.grad(native, tuple(parameters.values()), allow_unused=True)
            virtual = functional_adamw(parameters, ng, optimizer, config["max_grad_norm"])
            with torch.no_grad():
                baseline, _ = objective(learner, outer, tokenizer, config, parameters=virtual)
            baseline_value = float(baseline)
            del native, ng, virtual, baseline
            loss, metrics = meta_objective(learner, optimizer, inner, outer, tokenizer, config, arm, generator)
            gp = tuple(generator.parameters())
            grads = torch.autograd.grad(loss, gp, allow_unused=True)
            if not finite_gradients(grads):
                raise FloatingPointError("Nonfinite generator gradient")
            for p, g in zip(gp, grads):
                p.grad = g
            norm = torch.nn.utils.clip_grad_norm_(gp, 1., error_if_nonfinite=True)
            outer_optimizer.step()
            metrics.update(meta_native_loss=baseline_value, meta_delta=float(loss.detach())-baseline_value,
                           generator_gradient_norm=float(norm))
            return metrics
        metrics, resources = timed(episode_update)
        history.append({"episode": episode+1, "teacher_state": state, **metrics, **resources})
        if (episode+1) % config["checkpoint_interval"] == 0 or episode+1 == config["meta_episodes"] or budget.maximum is not None:
            atomic_torch(path, {"episode": episode+1, "generator": cpu(generator.state_dict()),
                               "optimizer": cpu(outer_optimizer.state_dict()), "history": history,
                               "rng": torch.get_rng_state()})
            print(f"{arm}: meta episode {episode+1}/{config['meta_episodes']}", flush=True)
        budget.tick()
    generator.requires_grad_(False)
    return generator


def continuations(learner, optimizer, partitions, tokenizer, config, directory, learner_dir, seed, generators, budget):
    for state in config["states"]:
        parent = load_checkpoint(learner_dir / f"step_{state}.pt")
        for arm in ARMS:
            arm_dir = directory / f"step_{state}" / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            latest = arm_dir / "latest.pt"
            restore(learner, optimizer, parent["state"])
            start, history = 0, []
            if latest.exists():
                saved = load_checkpoint(latest)
                restore(learner, optimizer, saved["state"])
                start, history = saved["step"], saved["history"]
            for index in range(start, max(config["horizons"])):
                rows = scheduled_batch(partitions["B"], state+index, config["batch_size"], seed)
                metrics, resources = timed(lambda: update(learner, optimizer, rows, tokenizer, config, arm, generators.get(arm)))
                history.append({"step": index+1, **metrics, **resources})
                step = index+1
                if step in config["horizons"]:
                    save_progress(arm_dir / f"horizon_{step}.pt", learner, optimizer, step, history)
                save_progress(latest, learner, optimizer, step, history)
                budget.tick()
            # Development evaluation is separate from checkpoints and repeat-safe.
            for horizon in config["horizons"]:
                metrics_path = arm_dir / f"dev_{horizon}.json"
                if metrics_path.exists():
                    continue
                saved = load_checkpoint(arm_dir / f"horizon_{horizon}.pt")
                restore(learner, optimizer, saved["state"])
                metrics, resources = timed(lambda: evaluate(learner, partitions["D"], tokenizer, config))
                atomic_json(metrics_path, {"seed": seed, "state": state, "arm": arm, "horizon": horizon,
                    "partition": "D", "metrics": metrics, "resources": resources,
                    "training_history": saved["history"], "auxiliary_exposures": 0 if arm == "native" else horizon*config["batch_size"]})
                print(f"Dev seed={seed} state={state} arm={arm} horizon={horizon}: {metrics}", flush=True)
                budget.tick()


def metric_rows(output, prefix):
    return [read_json(path) for path in sorted(output.glob(f"seed_*/branches/step_*/*/{prefix}_*.json"))]


def summarize_rows(rows, config):
    keyed = {(r["seed"], r["state"], r["arm"], r["horizon"]): r for r in rows}
    if len(keyed) != len(rows):
        raise ValueError("Duplicate metric rows")
    paired = []
    for r in rows:
        baseline = keyed[(r["seed"], r["state"], "native", r["horizon"])]["metrics"]
        paired.append({**r, "paired_delta": {m: r["metrics"][m]-baseline[m]
                                            for m in ("native_loss", "numeric_loss", "exact_match", "correct")}})
    primary = []
    for arm in ARMS:
        per_seed = {}
        for _, _, seed in config["seed_triples"]:
            matches = [r for r in paired if r["seed"] == seed and r["arm"] == arm and r["horizon"] == max(config["horizons"])]
            if len(matches) != len(config["states"]):
                raise ValueError("Incomplete primary evaluation")
            per_seed[str(seed)] = statistics.mean(r["paired_delta"]["native_loss"] for r in matches)
        values = list(per_seed.values())
        primary.append({"arm": arm, "per_seed_mean_state_delta": per_seed, "mean_loss_delta": statistics.mean(values),
                        "seed_sd": statistics.stdev(values) if len(values)>1 else None})
    return {"paired_results": paired, "primary": primary, "horizon": max(config["horizons"]),
            "research_evidence": config["backend"] == "hf"}


def lock_development(output, config):
    rows = metric_rows(output, "dev")
    expected = len(config["seed_triples"])*len(config["states"])*len(ARMS)*len(config["horizons"])
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} completed development evaluations, found {len(rows)}")
    summary = summarize_rows(rows, config)
    primary = {r["arm"]: r for r in summary["primary"]}
    candidates = ("native", "hardest_existing", "learned_loss_reweighting")
    chosen = min(candidates, key=lambda arm: primary[arm]["mean_loss_delta"])
    generated = primary["learned_barycenter"]
    comparisons = {}
    for other in ("native", "learned_loss_reweighting"):
        diffs = {seed: value-primary[other]["per_seed_mean_state_delta"][seed]
                 for seed, value in generated["per_seed_mean_state_delta"].items()}
        comparisons[other] = {"mean": statistics.mean(diffs.values()), "per_seed": diffs,
                              "helpful_seeds": sum(v<0 for v in diffs.values())}
    # Operationalize "no consistent exact-match harm" conservatively: no negative
    # pooled EM versus either primary control, and not worse in a majority of seeds.
    exact_match = {}
    for other in ("native", "learned_loss_reweighting"):
        diffs = []
        for _, _, seed in config["seed_triples"]:
            def mean_arm(arm):
                return statistics.mean(r["metrics"]["exact_match"] for r in rows if r["seed"] == seed and
                    r["arm"] == arm and r["horizon"] == max(config["horizons"]))
            diffs.append(mean_arm("learned_barycenter")-mean_arm(other))
        exact_match[other] = {"mean": statistics.mean(diffs), "harmful_seeds": sum(v<0 for v in diffs)}
    nseeds = len(config["seed_triples"])
    gate = nseeds >= 3 and all(v["mean"]<0 and v["helpful_seeds"]>=2 for v in comparisons.values()) and all(
        v["mean"]>=0 and v["harmful_seeds"] <= nseeds//2 for v in exact_match.values())
    atomic_json(output / "development_summary.json", summary)
    lock = {"development_sha256": digest(rows), "best_non_generating": chosen,
            "nll_comparisons": comparisons, "exact_match_comparisons": exact_match,
            "advance_to_llama": gate, "test_evaluated_when_locked": False,
            "protocol_sha256": digest(config), "llama_auto_launch": False}
    lock["checkpoint_sha256"] = {str(p.relative_to(output)): file_hash(p) for p in
                                sorted(output.glob("seed_*/branches/step_*/*/horizon_*.pt"))}
    atomic_json(output / "development_lock.json", lock)
    return lock


def run(config, output, tokenizer, partitions, manifest, budget):
    check_preflight(config, output, manifest)
    if (output / "development_lock.json").exists():
        if read_json(output / "development_lock.json")["development_sha256"] != digest(metric_rows(output, "dev")):
            raise ValueError("Development artifacts changed after locking")
        print("Development already complete and locked. Use report to evaluate the official test split.")
        return
    for teacher_seed, meta_seed, learner_seed in config["seed_triples"]:
        directory = output / f"seed_{learner_seed}"
        learner = load_learner(config, manifest["resolved"], teacher_seed)
        optimizer = make_optimizer(learner, config)
        warmup(learner, optimizer, partitions["A"], tokenizer, config, directory / "teacher", teacher_seed, budget)
        generators = {arm: meta_train(learner, optimizer, partitions, tokenizer, config, directory / "generators",
                                     directory / "teacher", meta_seed, arm, budget) for arm in LEARNED}
        del learner, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        learner = load_learner(config, manifest["resolved"], learner_seed)
        optimizer = make_optimizer(learner, config)
        warmup(learner, optimizer, partitions["B"], tokenizer, config, directory / "learner", learner_seed, budget)
        continuations(learner, optimizer, partitions, tokenizer, config, directory / "branches",
                      directory / "learner", learner_seed, generators, budget)
        del learner, optimizer, generators
        gc.collect()
    lock = lock_development(output, config)
    manifest["status"] = "development_complete"
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps(lock, indent=2), flush=True)


def check_preflight(config, output, manifest):
    path = output / "preflight.json"
    if not path.exists():
        raise ValueError("Run preflight before training")
    check = read_json(path)
    if not check.get("passed") or check.get("identity_hash") != digest(manifest["identity"]) or check.get("data_sha256") != manifest["data_sha256"]:
        raise ValueError("A successful matching preflight is required")


def report(config, output, tokenizer, partitions, manifest, budget):
    path = output / "development_lock.json"
    if not path.exists():
        raise ValueError("Complete and lock development before accessing official test metrics")
    lock = read_json(path)
    if lock["development_sha256"] != digest(metric_rows(output, "dev")) or lock["protocol_sha256"] != digest(config):
        raise ValueError("Development/protocol changed after locking")
    for name, expected_hash in lock["checkpoint_sha256"].items():
        if file_hash(output / name) != expected_hash:
            raise ValueError("Learner checkpoint changed after development locking")
    for _, _, seed in config["seed_triples"]:
        learner = load_learner(config, manifest["resolved"], seed)
        optimizer = make_optimizer(learner, config)
        for state in config["states"]:
            for arm in ARMS:
                directory = output / f"seed_{seed}" / "branches" / f"step_{state}" / arm
                for horizon in config["horizons"]:
                    path = directory / f"test_{horizon}.json"
                    if path.exists():
                        continue
                    checkpoint = load_checkpoint(directory / f"horizon_{horizon}.pt")
                    restore(learner, optimizer, checkpoint["state"])
                    metrics, resources = timed(lambda: evaluate(learner, partitions["test"], tokenizer, config))
                    atomic_json(path, {"seed": seed, "state": state, "arm": arm, "horizon": horizon,
                        "partition": "test", "metrics": metrics, "resources": resources,
                        "auxiliary_exposures": 0 if arm == "native" else horizon*config["batch_size"]})
                    print(f"Test seed={seed} state={state} arm={arm} horizon={horizon}: {metrics}", flush=True)
                    budget.tick()
        del learner, optimizer
        gc.collect()
    summary = summarize_rows(metric_rows(output, "test"), config)
    summary["best_non_generating_chosen_on_D"] = lock["best_non_generating"]
    primary = {r["arm"]: r for r in summary["primary"]}
    for row in summary["primary"]:
        row["contrasts"] = {}
        for other in dict.fromkeys(("native", "learned_loss_reweighting", lock["best_non_generating"])):
            diffs = {seed: v-primary[other]["per_seed_mean_state_delta"][seed]
                     for seed, v in row["per_seed_mean_state_delta"].items()}
            row["contrasts"][other] = {"per_seed": diffs, "mean": statistics.mean(diffs.values()),
                "seed_sd": statistics.stdev(diffs.values()) if len(diffs)>1 else None}
    atomic_json(output / "summary.json", summary)
    lines = ["# LLM generator pilot report", "", "Real-model pilot." if config["backend"] == "hf" else
             "OFFLINE TINY SMOKE: plumbing evidence only; not a Qwen/GSM8K utility result.", "",
             "Loss deltas are treatment minus matched native; negative is helpful.",
             f"Primary horizon: {max(config['horizons'])}; equal mean over states {config['states']} within each seed.", "",
             "| Arm | Mean loss delta | Seed SD |", "|---|---:|---:|"]
    for row in summary["primary"]:
        sd = "n/a" if row["seed_sd"] is None else f"{row['seed_sd']:.8g}"
        lines.append(f"| {row['arm']} | {row['mean_loss_delta']:+.8g} | {sd} |")
    lines += ["", "See summary.json for every seed/state/horizon and exact-match/numeric-NLL outcomes.",
              "Development chooses the non-generating comparator before this report.",
              f"Comparator: {lock['best_non_generating']}. Development Llama advancement gate: {lock['advance_to_llama']}.",
              "No automatic Llama launch. No OT-map, multi-capability balance, or compute-efficiency claim.",
              "Meta-training and repeated evaluation costs are recorded separately in checkpoints and metric artifacts.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    manifest["status"] = "complete"
    atomic_json(output / "manifest.json", manifest)


def execute(command, supplied, output, max_units=None):
    config = validate_config(supplied)
    output = Path(output)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    with run_lock(output):
        started = time.perf_counter()
        completed = False
        try:
            tokenizer, partitions, manifest = prepare(config, output)
            budget = Budget(max_units)
            if command == "preflight":
                result = preflight(config, output, tokenizer, partitions, manifest)
            elif command == "run":
                result = run(config, output, tokenizer, partitions, manifest, budget)
            elif command == "report":
                result = report(config, output, tokenizer, partitions, manifest, budget)
            else:
                raise ValueError(command)
            completed = True
            return result
        finally:
            path = output / "command_history.json"
            history = read_json(path) if path.exists() else []
            history.append({"command": command, "completed": completed,
                            "elapsed_wall_seconds": time.perf_counter()-started})
            atomic_json(path, history)
