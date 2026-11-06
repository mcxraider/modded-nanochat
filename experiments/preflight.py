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
# Per-tensor relative L2 error. Norms average out per-element BF16 rounding (~0.4%
# spacing), but FA3 and SDPA backward reduce in different orders and the noise
# compounds through eight blocks of backprop; 5% is ~6x the BF16 epsilon while a
# wrong mask, window, or GQA head mapping gives O(1) error.
GRAD_RTOL = 0.05
# Scalar gains (resid_lambdas, x0_lambdas) have one gradient per layer, each a single
# ~1M-term BF16 reduction that inductor reorders vs eager; with ~8 elements nothing
# averages out (observed 5.03% compiled-vs-eager). Real bugs still show as O(1) error
# in the matrix gradients, which keep GRAD_RTOL.
SCALAR_GRAD_RTOL = 0.15
SCALAR_NUMEL = 64


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


def compare_backward(actual, expected):
    """Compare (loss, {name: grad}) pairs; fail closed and return evidence."""
    import torch
    (loss, grads), (reference_loss, reference_grads) = actual, expected
    if grads.keys() != reference_grads.keys():
        raise AssertionError(f"gradient parameter sets differ: {sorted(grads.keys() ^ reference_grads.keys())}")
    if not torch.isfinite(loss) or not torch.isfinite(reference_loss):
        raise AssertionError("nonfinite preflight loss")
    torch.testing.assert_close(loss, reference_loss, rtol=RTOL, atol=ATOL)
    ratio, worst, worst_name, worst_limit = 0.0, 0.0, None, GRAD_RTOL
    for name, reference in reference_grads.items():
        if not torch.isfinite(grads[name]).all() or not torch.isfinite(reference).all():
            raise AssertionError(f"nonfinite gradient for {name}")
        # A zero reference makes any nonzero actual gradient fail.
        error = ((grads[name] - reference).norm() / reference.norm().clamp_min(1e-12)).item()
        limit = SCALAR_GRAD_RTOL if reference.numel() <= SCALAR_NUMEL else GRAD_RTOL
        if error / limit > ratio:
            ratio, worst, worst_name, worst_limit = error / limit, error, name, limit
    if ratio > 1:
        raise AssertionError(f"gradient relative error {worst:.4f} for {worst_name} exceeds {worst_limit}")
    return {"loss_difference": (loss - reference_loss).abs().item(),
            "max_grad_relative_error": worst, "worst_parameter": worst_name}


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

    def backward(model, module, ids, targets):
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = module(ids, targets=targets)
        loss.backward()
        return loss.detach().float(), {name: p.grad.float().clone()
                                       for name, p in model.named_parameters() if p.grad is not None}

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
            # Training path: FA3 backward (incl. GQA) against SDPA, then the compiled
            # model train.py actually runs against eager. Compiling forward+backward
            # costs roughly 30-90 s per variant on an H100 (8 variants).
            # Full training length: at `length` only ~6 queries pass the short window, so an
            # ignored window moves gradients by ~0.3% (under GRAD_RTOL); at sequence_len, ~24%.
            model.train()
            train_ids = torch.randint(0, config.vocab_size, (1, config.sequence_len), device=device)
            targets = torch.randint(0, config.vocab_size, (1, config.sequence_len), device=device)
            attention._override_impl = "fa3"
            eager = backward(model, model, train_ids, targets)
            attention._override_impl = "sdpa"
            grad_check = compare_backward(eager, backward(model, model, train_ids, targets))
            attention._override_impl = "fa3"  # before tracing: dynamo specializes on it
            compiled = torch.compile(model, dynamic=False)
            compile_check = compare_backward(backward(model, compiled, train_ids, targets), eager)
            torch.cuda.synchronize(device)
            results.append({**variant, "tokens": length, "training_tokens": config.sequence_len, "cached_vs_full": cache_check,
                            "fa3_vs_sdpa": backend_check, "fa3_vs_sdpa_grad": grad_check,
                            "compiled_vs_eager": compile_check, "status": "passed"})
            del model, cache, ids, train_ids, full, chunk, tail, cached, reference, targets, eager, compiled
            torch._dynamo.reset()
            torch.cuda.empty_cache()
    finally:
        attention._override_impl = original_override
    require_fa3()
    return {"status": "passed", "device": torch.cuda.get_device_name(device),
            "attention_kernel": kernel_provenance(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "rtol": RTOL, "atol": ATOL, "grad_rtol": GRAD_RTOL,
            "scalar_grad_rtol": SCALAR_GRAD_RTOL,
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
