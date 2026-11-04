"""Historical-nanochat training to a fixed token target, with a wall-clock safety cap and explicit configuration."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time

from experiments.config import BASE_COMMIT, MODEL_TYPE, NODE_HOUR_USD, read_config, schedule
from experiments.data import make_loader, shard_index, split_paths


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def data_manifest(base_dir, paths):
    """Seed-independent description of the data; compared across runs. Per-run order lives in data_order.json."""
    # Never permit the historical loader's silent FineWeb fallback in this suite.
    if len(paths) < 2 or any(Path(p).parent.name != "base_data_climbmix" for p in paths):
        raise ValueError("prepare ClimbMix shards; legacy FineWeb fallback is disallowed")
    train, val, test, _ = split_paths(paths, 1)
    members = sorted([*train, val, test], key=shard_index)
    return {
        "dataset": "karpathy/climbmix-400b-shuffle",
        # No mtimes: they aren't guaranteed stable across containers and would fail every resume.
        "shards": [{"path": str(Path(p).resolve()), "bytes": Path(p).stat().st_size} for p in members],
        "train_shards": len(train),
        "validation_shard": Path(val).name, "validation_sha256": sha256(val),
        "test_shard": Path(test).name, "test_sha256": sha256(test),
        "tokenizer": {name: sha256(Path(base_dir) / "tokenizer" / name)
                      for name in ("tokenizer.pkl", "token_bytes.pt")},
        "loader": "historical BOS best-fit; tokenizer_threads=4, tokenizer_batch_size=128, buffer_size=1000",
    }


def nvidia_info(arguments):
    try:
        return subprocess.check_output(["nvidia-smi", *arguments], text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def gpu_utilization():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=5, check=True)
        return result.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--resume", type=int)
    p.add_argument("--device-type", choices=("cuda", "cpu"), default="cuda", help="CPU is for tiny smoke tests only")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != BASE_COMMIT:
        raise RuntimeError("training must run in the pinned worktree; launch through experiments/run.py")
    config = read_config(args.config)

    import torch
    import torch.distributed as dist
    from nanochat.modified_gpt import GPT, GPTConfig
    from nanochat.common import compute_init, compute_cleanup, get_base_dir
    import nanochat.dataloader as dataloader
    from nanochat.dataset import list_parquet_files
    from nanochat.tokenizer import get_tokenizer, get_token_bytes
    from nanochat.loss_eval import evaluate_bpb
    from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
    from nanochat.flash_attention import HAS_FA3

    ddp, rank, _, world, device = compute_init(args.device_type)
    if world != config["workers"]:
        raise ValueError("launched worker count differs from configuration")
    if args.device_type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("suite requires BF16-capable GPUs; FP8 is disabled")
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    if args.device_type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    sync = (lambda: torch.cuda.synchronize(device)) if args.device_type == "cuda" else lambda: None
    def barrier():
        if ddp:
            dist.barrier()
    def maximum(value):
        tensor = torch.tensor(value, dtype=torch.float64, device=device)
        if ddp:
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return tensor.item()
    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16) if args.device_type == "cuda" else nullcontext()
    def memory():
        return maximum(torch.cuda.max_memory_allocated(device) if args.device_type == "cuda" else 0)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    tokenizer = get_tokenizer()
    model_config = GPTConfig(**config["model"])
    if tokenizer.get_vocab_size() != model_config.vocab_size:
        raise ValueError("prepared tokenizer vocabulary differs from config")
    paths = list_parquet_files()
    manifest = data_manifest(get_base_dir(), paths) if rank == 0 else None
    if ddp:
        values = [manifest]
        dist.broadcast_object_list(values, src=0)
        manifest = values[0]
    manifest_path = output / "data_manifest.json"
    if args.resume is not None and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("data/tokenizer manifest changed since original run")
    train_order, _, _, rotation = split_paths(paths, config["seed"])
    if rank == 0:
        write_json(manifest_path, manifest)
        write_json(output / "data_order.json", {"seed": config["seed"], "rotation": rotation,
                                                "train_order": [Path(p).name for p in train_order]})
    token_bytes = get_token_bytes(device=device)
    with torch.device("meta"):
        model = GPT(model_config)
        # Freeze historical scaling against MHA B0, including for both GQA variants.
        baseline = GPT(replace(model_config, n_kv_head=model_config.n_head, rotary_fraction=1.0, mlp_ratio=4.0))
        reference = GPT(replace(model_config, n_layer=12, n_embd=768, n_head=6, n_kv_head=6, rotary_fraction=1.0, mlp_ratio=4.0))
    def scaling_params(m):
        counts = m.num_scaling_params()
        return counts["transformer_matrices"] + counts["lm_head"]
    o, t = config["optimizer"], config["training"]
    batch_scale = math.sqrt(t["total_batch_size"] / o["reference_batch_size"])
    weight_decay = o["weight_decay"] * batch_scale * scaling_params(reference) / scaling_params(baseline)
    scaling_parameter_count = scaling_params(model)
    del reference, baseline
    model.to_empty(device=device)
    model.init_weights()
    optimizer = model.setup_optimizer(
        embedding_lr=o["embedding_lr"] * batch_scale, unembedding_lr=o["unembedding_lr"] * batch_scale,
        matrix_lr=o["matrix_lr"] * batch_scale, scalar_lr=o["scalar_lr"] * batch_scale,
        weight_decay=weight_decay, adam_betas=tuple(o["adam_betas"]))
    effective_groups = [{**{k: v for k, v in g.items() if k != "params"},
                         "parameter_shapes": sorted({tuple(p.shape) for p in g["params"]}),
                         "parameter_count": sum(p.numel() for p in g["params"])} for g in optimizer.param_groups]
    parameter_count = sum(p.numel() for p in model.parameters())
    flops_per_token = model.estimate_flops()
    step, elapsed_before, val_bpb = 0, 0.0, None
    restored_rng = None
    if args.resume is not None:
        weights, opt_state, meta = load_checkpoint(str(checkpoints), args.resume, device, load_optimizer=True, rank=rank)
        if meta.get("experiment_config") != config or meta.get("model_type") != MODEL_TYPE:
            raise ValueError("checkpoint does not match experiment configuration")
        model.load_state_dict(weights, strict=True)
        optimizer.load_state_dict(opt_state)
        del weights, opt_state
        state = torch.load(checkpoints / f"runtime_{args.resume:06d}_rank{rank}.pt", map_location="cpu", weights_only=True)
        step, restored_rng = state["step"], state
        elapsed_before = json.loads((checkpoints / f"timing_{step:06d}.json").read_text())["training_seconds"]
        val_bpb = meta["val_bpb"]
    accum = t["total_batch_size"] // (world * t["device_batch_size"] * model_config.sequence_len)
    def new_loader(split):
        return make_loader(dataloader, tokenizer, t["device_batch_size"], model_config.sequence_len, split, device, paths, config["seed"])
    train_loader = new_loader("train")
    data_epoch = 1
    total_steps = t["target_tokens"] // t["total_batch_size"]
    # The historical loader exposes only approximate row-group resume. Replay keeps exact
    # document packing/data order without changing the historical loader implementation.
    replay_started = time.monotonic()
    for _ in range(step * accum):
        data_epoch = max(data_epoch, next(train_loader)[2]["epoch"])
    replay_seconds = time.monotonic() - replay_started
    if restored_rng:
        torch.set_rng_state(restored_rng["torch_rng"])
        random.setstate(restored_rng["python_rng"])
        if args.device_type == "cuda":
            torch.cuda.set_rng_state(restored_rng["cuda_rng"], device)
    train_model = torch.compile(model, dynamic=False) if t["compile"] else model
    hardware = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                "world_size": world, "device": str(device), "attention_backend": "FA3" if HAS_FA3 else "SDPA",
                "precision": "BF16 autocast; FP32 linear weights, BF16 embeddings" if args.device_type == "cuda" else "FP32 CPU smoke test",
                "gpu": torch.cuda.get_device_name(device) if args.device_type == "cuda" else platform.processor(),
                "effective_optimizer_groups": effective_groups, "batch_lr_scale": batch_scale,
                "effective_initial_weight_decay": weight_decay, "parameters": parameter_count,
                "flops_per_token": flops_per_token,
                "flops_formula": "6 * matmul_parameters + sum_layers(12 * query_heads * head_dim * effective_window); excludes lookup, norm, softmax, rotary elementwise ops",
                "unique_data_estimate": None, "unique_data_note": "packing/cropping prevents inference of unique tokens from total tokens",
                "gradient_accumulation": accum, "resume_replay_batches": step * accum, "resume_replay_seconds": replay_seconds,
                "nvidia_smi": gpu_utilization(),
                "gpu_topology": nvidia_info(["topo", "-m"]),
                "gpu_driver": nvidia_info(["--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
                "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "pyarrow", "kernels", "rustbpe", "tiktoken")}}
    if rank == 0:
        write_json(output / ("hardware.json" if args.resume is None else f"hardware_resume_{step:06d}.json"), hardware)
    eval_micro_tokens = world * t["device_batch_size"] * model_config.sequence_len
    def evaluate(split, tokens):
        if tokens % eval_micro_tokens:
            raise ValueError("evaluation tokens must divide exactly into whole microbatches")
        model.eval()
        eval_loader = new_loader(split)
        def batches():
            for x, y, _ in eval_loader:
                yield x, y
        with autocast():
            value = evaluate_bpb(model, batches(), tokens // eval_micro_tokens, token_bytes)
        model.train()
        return value
    e = config["evaluation"]
    sync()
    barrier()
    started = time.monotonic()
    def elapsed():
        sync()
        return maximum(elapsed_before + time.monotonic() - started)
    metrics_path = output / "metrics.jsonl"
    def log(record):
        if rank == 0:
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record), flush=True)
    def checkpoint():
        seconds = elapsed()
        save_checkpoint(str(checkpoints), step, model.state_dict(), optimizer.state_dict(), {
            "model_type": MODEL_TYPE, "model_config": asdict(model_config), "experiment_config": config,
            "step": step, "val_bpb": val_bpb, "training_seconds_before_save": seconds,
            "tokens": step * t["total_batch_size"], "estimated_flops": step * t["total_batch_size"] * flops_per_token,
        }, rank=rank)
        torch.save({"step": step, "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                    "cuda_rng": torch.cuda.get_rng_state(device) if args.device_type == "cuda" else None},
                   checkpoints / f"runtime_{step:06d}_rank{rank}.pt")
        barrier()
        seconds = elapsed()
        if rank == 0:
            write_json(checkpoints / f"timing_{step:06d}.json", {"training_seconds": seconds})
        barrier()
        return elapsed()
    last_saved = None
    while True:
        if step >= total_steps:
            stop_reason = "token_target"
            break
        if t["max_steps"] is not None and step >= t["max_steps"]:
            stop_reason = "pilot_step_cap"
            break
        if elapsed() >= t["wallclock_cap_seconds"]:
            stop_reason = "wallclock_cap"  # run is no longer comparable; flagged in result.json
            break
        lr_scale, wd_scale = schedule(step / total_steps, o)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lr_scale
            if group["kind"] == "muon":
                group["momentum"] = 0.85 + 0.10 * min(step / 300, 1.0)
                group["weight_decay"] = weight_decay * wd_scale
        step_started = time.monotonic()
        loss_sum = 0.0
        for _ in range(accum):
            x, y, state = next(train_loader)
            data_epoch = max(data_epoch, state["epoch"])
            with autocast():
                loss = train_model(x, targets=y)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("nonfinite training loss")
            loss_sum += loss.detach().item() / accum
            (loss / accum).backward()
        optimizer.step()
        model.zero_grad(set_to_none=True)
        step += 1
        sync()
        dt = maximum(time.monotonic() - step_started)
        seconds = elapsed()
        log({"kind": "train", "step": step, "training_seconds": seconds, "tokens": step * t["total_batch_size"],
             "estimated_flops": step * t["total_batch_size"] * flops_per_token, "loss": loss_sum,
             "step_seconds": dt, "tokens_per_second": t["total_batch_size"] / dt,
             "whole_phase_tokens_per_second": (step * t["total_batch_size"]) / seconds,
             "lr_multiplier": lr_scale, "peak_vram_bytes": memory(),
             "gpu_samples": gpu_utilization() if rank == 0 and step % 10 == 0 else None})
        if step >= total_steps:
            continue  # the final checkpoint and evaluations follow the loop
        if t["eval_every"] > 0 and step % t["eval_every"] == 0:
            val_bpb = evaluate("val", e["interim_tokens"])
            log({"kind": "validation", "step": step, "training_seconds": elapsed(), "val_bpb": val_bpb,
                 "tokens": step * t["total_batch_size"], "estimated_flops": step * t["total_batch_size"] * flops_per_token})
        if t["save_every"] > 0 and step % t["save_every"] == 0:
            checkpoint()
            last_saved = step
    # A resumed completed checkpoint needs no duplicate write. The final evaluation is outside the training clock.
    training_seconds = checkpoint() if last_saved != step else elapsed()
    peak_vram = memory()
    evaluation_started = time.monotonic()
    final_bpb = evaluate("val", e["final_tokens"])
    test_bpb = evaluate("test", e["test_tokens"])
    evaluation_seconds = maximum(time.monotonic() - evaluation_started)
    final_record = {"step": step, "training_seconds": training_seconds, "tokens": step * t["total_batch_size"],
                    "estimated_flops": step * t["total_batch_size"] * flops_per_token}
    log({"kind": "validation", "final": True, "val_bpb": final_bpb, **final_record})
    log({"kind": "test", "final": True, "test_bpb": test_bpb, **final_record})
    result = {"run_id": config["run_id"], "step": step, "stop_reason": stop_reason,
              "parameters": parameter_count, "scaling_parameters": scaling_parameter_count,
              "target_tokens": t["target_tokens"], "data_epoch": data_epoch,
              "model_config": asdict(model_config), "tokens": step * t["total_batch_size"],
              "estimated_flops": step * t["total_batch_size"] * flops_per_token,
              "training_seconds": training_seconds, "overrun_seconds": max(0, training_seconds - t["wallclock_cap_seconds"]),
              "val_bpb": final_bpb, "test_bpb": test_bpb, "training_tokens_per_second": step * t["total_batch_size"] / training_seconds,
              "peak_vram_bytes": peak_vram, "evaluation_seconds": evaluation_seconds,
              "training_gpu_hours": training_seconds * world / 3600,
              "evaluation_gpu_hours": evaluation_seconds * world / 3600,
              "training_cost_usd": training_seconds * NODE_HOUR_USD / 3600,
              "checkpoint_dir": str(checkpoints), "checkpoint_step": step}
    if rank == 0:
        write_json(output / "result.json", result)
        meta_path = checkpoints / f"meta_{step:06d}.json"
        meta = json.loads(meta_path.read_text())
        meta["val_bpb"] = final_bpb
        meta["training_seconds"] = training_seconds
        write_json(meta_path, meta)
    barrier()
    compute_cleanup()


if __name__ == "__main__":
    main()
