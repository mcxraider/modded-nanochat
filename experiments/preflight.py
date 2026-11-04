"""GPU correctness gate for every distinct suite architecture, before training pilots."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time

from experiments.config import RUNS, model_for

# BF16 has ~0.8% relative spacing. Different fused reductions can accumulate
# several rounding steps across eight blocks; 3% relative/absolute is a
# conservative smoke-test tolerance, not a claim of bitwise equivalence.
RTOL = 0.03
ATOL = 0.03


def unique_variants():
    """Group repeated seeds without dropping any architectural difference."""
    variants = {}
    for run_id in RUNS:
        config = model_for(run_id)
        key = json.dumps(config, sort_keys=True)
        variants.setdefault(key, {"model": config, "run_ids": []})["run_ids"].append(run_id)
    return list(variants.values())


def compare_logits(actual, expected):
    """Fail closed on invalid outputs or numerical disagreement; return evidence."""
    import torch
    if actual.shape != expected.shape:
        raise AssertionError(f"logit shapes differ: {actual.shape} vs {expected.shape}")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError("nonfinite preflight logits")
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    error = actual.float() - expected.float()
    return {"max_absolute_error": error.abs().max().item(), "rms_error": error.square().mean().sqrt().item()}


def run_preflight():
    import torch
    import nanochat.flash_attention as attention
    from experiments.backend import require_fa3, kernel_provenance
    from nanochat.modified_gpt import GPT, GPTConfig
    from nanochat.engine import KVCache

    require_fa3()
    if not hasattr(attention, "_use_fa3"):
        raise RuntimeError("preflight must run in the pinned historical worktree")
    device = torch.device("cuda:0")
    started = time.monotonic()
    results = []
    original_override = attention._override_impl
    try:
        for variant in unique_variants():
            torch.manual_seed(1729)
            torch.cuda.manual_seed_all(1729)
            config = GPTConfig(**variant["model"])
            model = GPT(config).to(device).eval()
            model.init_weights()
            # Default zero projections conceal attention mistakes at initialization.
            # Raise the head scale too: its training initialization produces logits
            # smaller than the BF16 smoke-test absolute tolerance.
            torch.nn.init.normal_(model.lm_head.weight, std=0.02)
            for block in model.transformer.h:
                torch.nn.init.normal_(block.attn.c_proj.weight, std=0.02)
                torch.nn.init.normal_(block.mlp.c_proj.weight, std=0.02)
            boundary = config.sequence_len // 2
            length = boundary + 6
            ids = torch.randint(0, config.vocab_size, (1, length), device=device)
            cache = KVCache(1, config.n_kv_head, length, config.n_embd // config.n_head,
                            config.n_layer, device, torch.bfloat16)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                attention._override_impl = "fa3"
                full = model(ids)
                # Prefix followed by a chunk spanning the first eviction, then
                # single-token decodes: both cached kernel paths are exercised.
                prefix = boundary - 1
                model(ids[:, :prefix], kv_cache=cache, last_token_only=True)
                chunk = model(ids[:, prefix:boundary + 3], kv_cache=cache)
                tail = [model(ids[:, i:i + 1], kv_cache=cache, last_token_only=True)
                        for i in range(boundary + 3, length)]
                cached = torch.cat([chunk, *tail], dim=1)
                cache_check = compare_logits(cached, full[:, prefix:])
                if cache.get_pos() != length:
                    raise AssertionError("preflight cache did not consume the whole prompt")
                attention._override_impl = "sdpa"
                reference = model(ids)
                backend_check = compare_logits(full, reference)
            torch.cuda.synchronize(device)
            results.append({**variant, "tokens": length, "cached_vs_full": cache_check,
                            "fa3_vs_sdpa": backend_check, "status": "passed"})
            del model, cache, ids, full, chunk, tail, cached, reference
            torch.cuda.empty_cache()
    finally:
        attention._override_impl = original_override
    require_fa3()
    return {"status": "passed", "device": torch.cuda.get_device_name(device),
            "attention_kernel": kernel_provenance(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "rtol": RTOL, "atol": ATOL,
            "elapsed_seconds": time.monotonic() - started, "variants": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run_preflight()
    encoded = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as stream:
            temporary = stream.name
            stream.write(encoded)
        os.replace(temporary, args.output)
    print(encoded, end="")


if __name__ == "__main__":
    main()
