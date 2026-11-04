"""Single-GPU, fixed-workload inference benchmark for experiment checkpoints."""
import argparse
import json
from pathlib import Path
import time


def kv_cache_bytes_analytic(n_layer, batch, capacity, n_kv_head, head_dim, element_bytes=2):
    """K and V tensors for every layer: 2 x layers x batch x positions x KV heads x head_dim x bytes."""
    return 2 * n_layer * batch * capacity * n_kv_head * head_dim * element_bytes


def kv_cache_bytes_allocated(cache):
    return (cache.k_cache.numel() + cache.v_cache.numel()) * cache.k_cache.element_size()


def benchmark(model, config, device):
    import torch
    from experiments.backend import kernel_provenance, require_fa3
    from nanochat.engine import KVCache
    from nanochat.flash_attention import HAS_FA3
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("inference timing requires one CUDA GPU")
    require_fa3()
    b = config["benchmark"]
    cfg = model.config
    dtype = torch.bfloat16
    # Use identical deterministic token IDs across models; never stop early on EOS.
    generator = torch.Generator(device="cpu").manual_seed(config["seed"])
    prompts = {n: torch.randint(0, cfg.vocab_size, (1, n), generator=generator).to(device) for n in b["prompt_lengths"]}
    rows = []
    for batch in b["batch_sizes"]:
        for prompt_len in b["prompt_lengths"]:
            capacity = prompt_len + b["decode_tokens"]
            if capacity > model.rotary_seq_len:
                raise ValueError("benchmark exceeds the model rotary cache; prompt length will not be clamped")
            try:
                ids = prompts[prompt_len].expand(batch, -1).contiguous()
                cache = KVCache(batch, cfg.n_kv_head, capacity, cfg.n_embd // cfg.n_head, cfg.n_layer, device, dtype)
                measurements = []
                for repeat in range(b["warmup"] + b["repeats"]):
                    cache.reset()
                    torch.cuda.synchronize(device)
                    if repeat == b["warmup"]:
                        torch.cuda.reset_peak_memory_stats(device)
                    started = time.perf_counter()
                    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                        logits = model(ids, kv_cache=cache, last_token_only=True)
                        torch.cuda.synchronize(device)
                        prefill_end = time.perf_counter()
                        next_id = logits[:, -1].argmax(-1, keepdim=True)
                        del logits
                        torch.cuda.synchronize(device)
                        first_token = time.perf_counter()
                        for _ in range(b["decode_tokens"] - 1):
                            logits = model(next_id, kv_cache=cache, last_token_only=True)
                            next_id = logits[:, -1].argmax(-1, keepdim=True)
                        torch.cuda.synchronize(device)
                        finished = time.perf_counter()
                    if repeat >= b["warmup"]:
                        measurements.append((prefill_end - started, first_token - started, finished - first_token, finished - started))
                prefill, ttft, decode, total = [sum(v[i] for v in measurements) / len(measurements) for i in range(4)]
                decode_count = b["decode_tokens"] - 1
                rows.append({"batch_size": batch, "prompt_tokens": prompt_len, "generated_tokens_per_request": b["decode_tokens"],
                             "decode_timed_tokens_per_request": decode_count, "prefill_tokens_per_second": batch * prompt_len / prefill,
                             "time_to_first_token_seconds": ttft, "decode_tokens_per_second_per_request": decode_count / decode,
                             "decode_tokens_per_second_aggregate": batch * decode_count / decode,
                             "total_workload_seconds": total, "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
                             "beyond_training_context": capacity > cfg.sequence_len, "kv_heads": cfg.n_kv_head,
                             "kv_cache_capacity_tokens": capacity,
                             "kv_cache_bytes_analytic": kv_cache_bytes_analytic(cfg.n_layer, batch, capacity, cfg.n_kv_head, cfg.n_embd // cfg.n_head),
                             "kv_cache_bytes_allocated": kv_cache_bytes_allocated(cache), "status": "ok"})
                del cache, ids, logits, next_id
            except torch.OutOfMemoryError:
                rows.append({"batch_size": batch, "prompt_tokens": prompt_len, "status": "oom"})
                # Drop references before freeing allocator blocks for the next fixed workload.
                cache = ids = logits = next_id = None
                torch.cuda.empty_cache()
            print(json.dumps({"benchmark_row": rows[-1]["batch_size"], "prompt_tokens": prompt_len, "status": rows[-1]["status"]}), flush=True)
    return {"device": torch.cuda.get_device_name(device), "attention_backend": "FA3" if HAS_FA3 else "SDPA",
            "attention_kernel": kernel_provenance(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "sampling": "greedy, fixed length, no EOS early exit",
            "precision": "BF16 autocast", "warmup": b["warmup"], "repeats": b["repeats"],
            "decode_definition": "first token included in TTFT; remaining decode_tokens-1 tokens timed as decode",
            "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    import os
    import torch
    from experiments.config import read_config
    from experiments.train import write_json
    from nanochat.checkpoint_manager import build_model
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("benchmark must run on exactly one worker without torchrun")
    run = args.run_dir.resolve()
    config = read_config(run / "config.json")
    result = json.loads((run / "result.json").read_text())
    device = torch.device("cuda:0")
    model, _, _ = build_model(str(run / "checkpoints"), result["checkpoint_step"], device, "eval")
    started = time.monotonic()
    payload = benchmark(model, config, device)
    payload["elapsed_seconds"] = time.monotonic() - started
    payload["gpu_hours"] = payload["elapsed_seconds"] / 3600
    write_json(run / "inference.json", payload)


if __name__ == "__main__":
    main()
