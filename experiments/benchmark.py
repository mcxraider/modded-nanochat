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


def kv_bytes_read_per_token(window_sizes, batch, position, n_kv_head, head_dim, element_bytes=2):
    """K and V bytes one decode step reads at `position`: each layer attends to at most window + 1 keys, so
    the traffic stops growing once the position passes the (training-context) long window."""
    keys = sum(min(position, left + 1) for left, _ in window_sizes)
    return 2 * batch * keys * n_kv_head * head_dim * element_bytes


def host_position_cache():
    """KVCache whose position lives on the host. The pinned KVCache.get_pos() is a `.item()` device read,
    i.e. a GPU sync on every forward; the device `cache_seqlens` that FA3 reads still advances as before."""
    from nanochat.engine import KVCache

    class HostPositionCache(KVCache):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.pos = 0

        def reset(self):
            super().reset()
            self.pos = 0

        def get_pos(self):
            return self.pos

        def advance(self, num_tokens):
            super().advance(num_tokens)
            self.pos += num_tokens

    return HostPositionCache


def cast_linears(model, dtype):
    """Store matmul weights in `dtype` once. modified_gpt.Linear casts its FP32 weight on every call (an
    uncached copy per matmul per token); the rounded values are the same, so outputs are unchanged."""
    import torch
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.to(dtype)
    return model


def benchmark(model, config, device):
    import torch
    from experiments.backend import kernel_provenance, require_fa3
    from nanochat.flash_attention import HAS_FA3
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("inference timing requires one CUDA GPU")
    require_fa3()
    b = config["benchmark"]
    cfg = model.config
    dtype = torch.bfloat16
    KVCache = host_position_cache()
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
                measurements, prefill_peak, decode_peak, working_set = [], 0, 0, 0
                for repeat in range(b["warmup"] + b["repeats"]):
                    cache.reset()
                    torch.cuda.synchronize(device)
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
                        # Allocator statistics are host-side; read them outside both timed spans.
                        repeat_prefill_peak = torch.cuda.max_memory_allocated(device)
                        decode_baseline = torch.cuda.memory_allocated(device)  # weights + full cache + next_id
                        torch.cuda.reset_peak_memory_stats(device)
                        decode_started = time.perf_counter()
                        for _ in range(b["decode_tokens"] - 1):
                            logits = model(next_id, kv_cache=cache, last_token_only=True)
                            next_id = logits[:, -1].argmax(-1, keepdim=True)
                        torch.cuda.synchronize(device)
                        finished = time.perf_counter()
                    if repeat >= b["warmup"]:
                        measurements.append((prefill_end - started, first_token - started, finished - decode_started,
                                             first_token - started + finished - decode_started))
                        prefill_peak = max(prefill_peak, repeat_prefill_peak)
                        decode_peak = max(decode_peak, torch.cuda.max_memory_allocated(device))
                        working_set = max(working_set, torch.cuda.max_memory_allocated(device) - decode_baseline)
                prefill, ttft, decode, total = [sum(v[i] for v in measurements) / len(measurements) for i in range(4)]
                decode_count = b["decode_tokens"] - 1
                head_dim = cfg.n_embd // cfg.n_head
                rows.append({"batch_size": batch, "prompt_tokens": prompt_len, "generated_tokens_per_request": b["decode_tokens"],
                             "decode_timed_tokens_per_request": decode_count, "prefill_tokens_per_second": batch * prompt_len / prefill,
                             "time_to_first_token_seconds": ttft, "decode_tokens_per_second_per_request": decode_count / decode,
                             "decode_tokens_per_second_aggregate": batch * decode_count / decode,
                             "total_workload_seconds": total, "peak_vram_bytes": max(prefill_peak, decode_peak),
                             "prefill_peak_vram_bytes": prefill_peak, "decode_working_set_bytes": working_set,
                             "beyond_training_context": capacity > cfg.sequence_len, "kv_heads": cfg.n_kv_head,
                             "kv_cache_capacity_tokens": capacity,
                             "kv_cache_bytes_analytic": kv_cache_bytes_analytic(cfg.n_layer, batch, capacity, cfg.n_kv_head, head_dim),
                             "kv_cache_bytes_allocated": kv_cache_bytes_allocated(cache),
                             "kv_bytes_read_per_token_analytic": kv_bytes_read_per_token(model.window_sizes, batch, capacity, cfg.n_kv_head, head_dim),
                             "status": "ok"})
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
            "precision": "BF16 autocast; BF16 matmul weights cast once", "warmup": b["warmup"], "repeats": b["repeats"],
            "decode_definition": "first token included in TTFT; remaining decode_tokens-1 tokens timed as decode",
            "execution": "eager, host-side cache position (no per-forward GPU sync); no CUDA graphs",
            "decode_working_set_definition": "peak allocation during decode minus allocation at decode start (weights + cache)",
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
    cast_linears(model, torch.bfloat16)
    started = time.monotonic()
    payload = benchmark(model, config, device)
    payload["elapsed_seconds"] = time.monotonic() - started
    payload["gpu_hours"] = payload["elapsed_seconds"] / 3600
    write_json(run / "inference.json", payload)


if __name__ == "__main__":
    main()
