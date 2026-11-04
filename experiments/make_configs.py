"""Generate experiments/configs/<run_id>.json from the RUNS table in config.py."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.config import BASE_COMMIT, RUNS, SUITE_BATCH, model_for, read_config, validate_suite

# Per-depth policy shared by every run at that depth. PROVISIONAL until the GPU pilot measures
# tokens/s, peak VRAM, and evaluation cost; edit here and regenerate. Micro-batches must tile the global batch.
DEFAULTS = {
    8: {"device_batch_size": 32, "eval_every": 200, "save_every": 400},
}
OPTIMIZER = {"embedding_lr": 0.3, "unembedding_lr": 0.004, "matrix_lr": 0.02, "scalar_lr": 0.5, "weight_decay": 0.2,
             "adam_betas": [0.8, 0.95], "reference_batch_size": 524288, "warmup_ratio": 0.0, "warmdown_ratio": 0.5,
             "final_lr_frac": 0.0}
EVALUATION = {"interim_tokens": 2**21, "final_tokens": 2**24, "test_tokens": 2**24}
BATCH = SUITE_BATCH
# Upstream base_train.py default --target-param-data-ratio: B0 trains int(10.5 x scaling params) tokens,
# cut to whole updates exactly like upstream's `target_tokens // total_batch_size`.
TOKENS_PER_PARAM = 10.5


def _meta_model(run_id):
    import torch
    from nanochat.modified_gpt import GPT, GPTConfig
    with torch.device("meta"):
        return GPT(GPTConfig(**model_for(run_id)))


def flops_per_token(run_id):
    """The model's own estimate (GPT.estimate_flops), built on the meta device: no weights are allocated."""
    return _meta_model(run_id).estimate_flops()


def scaling_params(run_id):
    """Upstream's scaling-parameter count: transformer matrices + lm_head."""
    counts = _meta_model(run_id).num_scaling_params()
    return counts["transformer_matrices"] + counts["lm_head"]


def b0_run(depth):
    """The depth's reference run: MHA, full RoPE, historical MLP and value embeddings, seed 1, token budget."""
    return next(r for r, spec in RUNS.items() if spec["depth"] == depth and spec["gqa"] == 1 and spec["ve"]
                and spec["mlp"] == 4.0 and spec["rotary"] == 1.0 and spec["seed"] == 1 and spec["budget"] == "tokens")


def token_target(depth):
    """B0's training tokens: upstream's int(ratio x scaling params), in whole updates."""
    return int(TOKENS_PER_PARAM * scaling_params(b0_run(depth))) // BATCH * BATCH


def flop_budget(depth):
    """Declared per-run budget: B0 trained on the depth's token target."""
    return token_target(depth) * flops_per_token(b0_run(depth))


def target_tokens(run_id):
    r = RUNS[run_id]
    if r["budget"] == "tokens":
        return token_target(r["depth"])
    return flop_budget(r["depth"]) // (flops_per_token(run_id) * BATCH) * BATCH  # most whole updates within budget


BENCHMARK = {"batch_sizes": [1, 4, 16, 64], "prompt_lengths": [128, 1792, 8192], "decode_tokens": 256, "warmup": 2, "repeats": 5}


def build(run_id):
    r, d = RUNS[run_id], DEFAULTS[RUNS[run_id]["depth"]]
    return {
        "run_id": run_id, "base_commit": BASE_COMMIT, "seed": r["seed"], "workers": 1, "model": model_for(run_id),
        "training": {"target_tokens": target_tokens(run_id), "tokens_per_scaling_param": TOKENS_PER_PARAM,
                     "flop_budget": flop_budget(r["depth"]),
                     "wallclock_cap_seconds": 7200,
                     "device_batch_size": d["device_batch_size"], "total_batch_size": BATCH,
                     "save_every": d["save_every"], "eval_every": d["eval_every"], "max_steps": None, "compile": True},
        "optimizer": dict(OPTIMIZER), "evaluation": dict(EVALUATION), "benchmark": dict(BENCHMARK),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-dir", type=Path, default=ROOT / "experiments/configs")
    args = p.parse_args()
    args.config_dir.mkdir(parents=True, exist_ok=True)
    for run_id in RUNS:
        path = args.config_dir / f"{run_id}.json"
        path.write_text(json.dumps(build(run_id), indent=2) + "\n")
        read_config(path)
    validate_suite([read_config(args.config_dir / f"{run_id}.json") for run_id in RUNS])
    print(f"wrote {len(RUNS)} configs to {args.config_dir}")


if __name__ == "__main__":
    main()
