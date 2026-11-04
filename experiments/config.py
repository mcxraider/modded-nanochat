"""Small, dependency-free experiment configuration, run table and token-budget schedule."""
import json
import math
from pathlib import Path

BASE_COMMIT = "324e69c45d3606095adb6b409078647145165454"
MODEL_TYPE = "modified_gpt_324e69c"
NODE_HOUR_USD = 4.43  # Modal: H100 + 6 cores + 24 GiB; see modal_app.py

# Depth -> (n_layer, n_embd, n_head). nanochat uses n_embd = 64 * depth rounded up to a multiple of head_dim=128.
DEPTHS = {8: (8, 512, 4), 12: (12, 768, 6)}
SEQUENCE_LEN, VOCAB_SIZE, WINDOW_PATTERN = 2048, 32768, "SSSL"
# Upstream base_train.py auto batch rule at d8: 2^round(log2(2^19 * (D / D_REF)^0.383)) = 2^18 (tests recompute it).
SUITE_BATCH = 262144


def _run(depth, gqa, rotary, mlp, seed, experiment, compare_to, ve=True, budget="tokens"):
    return {"depth": depth, "gqa": gqa, "rotary": rotary, "mlp": mlp, "seed": seed, "ve": ve, "budget": budget,
            "experiment": experiment, "compare_to": tuple(compare_to)}


# Single source of truth for the suite. `gqa` is query heads / KV heads. `compare_to` names a run or a
# seed group (see GROUPS); the first entry is the primary reference. `ve` toggles the value embeddings.
# `budget`: "tokens" trains on the depth's token target (B0's); "flops" trains the most whole updates that fit
# the depth's declared FLOP budget (see make_configs.py), so a cheaper architecture sees more tokens.
RUNS = {
    "d8-b0-s1": _run(8, 1, 1.0, 4.0, 1, "E1", ()),
    "d8-b0-s2": _run(8, 1, 1.0, 4.0, 2, "E1", ()),
    "d8-b0-s3": _run(8, 1, 1.0, 4.0, 3, "E1", ()),
    "d8-gqa2-s1": _run(8, 2, 1.0, 4.0, 1, "E2", ("d8-b0",)),
    "d8-gqa2-s2": _run(8, 2, 1.0, 4.0, 2, "E2", ("d8-b0",)),
    "d8-gqa2-s3": _run(8, 2, 1.0, 4.0, 3, "E2", ("d8-b0",)),
    "d8-gqa2-r75": _run(8, 2, 0.75, 4.0, 1, "E2", ("d8-b0", "d8-gqa2")),
    "d8-gqa2-pm": _run(8, 2, 1.0, 4.5, 1, "E3", ("d8-b0", "d8-gqa2")),
    "d8-gqa4": _run(8, 4, 1.0, 4.0, 1, "E4", ("d8-b0", "d8-gqa2")),
    "d8-gqa2-r25": _run(8, 2, 0.25, 4.0, 1, "E4", ("d8-gqa2", "d8-gqa2-r75")),
    "d8-b0-nove": _run(8, 1, 1.0, 4.0, 1, "E5", ("d8-b0",), ve=False),
    "d8-gqa2-nove": _run(8, 2, 1.0, 4.0, 1, "E5", ("d8-b0-nove", "d8-gqa2"), ve=False),
    "d8-gqa2-iso": _run(8, 2, 1.0, 4.0, 1, "E6", ("d8-gqa2", "d8-b0"), budget="flops"),
    "d8-gqa4-iso": _run(8, 4, 1.0, 4.0, 1, "E6", ("d8-gqa4", "d8-b0"), budget="flops"),
}
GROUPS = {"d8-b0": ("d8-b0-s1", "d8-b0-s2", "d8-b0-s3"),
          "d8-gqa2": ("d8-gqa2-s1", "d8-gqa2-s2", "d8-gqa2-s3")}
# 2x2 interactions (a - b) - (c - d) on test BPB. E5: GQA-2's effect with value embeddings minus without.
INTERACTIONS = {"E5": ("d8-gqa2", "d8-b0", "d8-gqa2-nove", "d8-b0-nove")}


def members(name):
    """Run IDs behind a run ID or seed-group name."""
    if name in RUNS:
        return (name,)
    if name in GROUPS:
        return GROUPS[name]
    raise KeyError(name)


def model_for(run_id):
    """Fully resolved model section for a suite run."""
    r = RUNS[run_id]
    n_layer, n_embd, n_head = DEPTHS[r["depth"]]
    return {"sequence_len": SEQUENCE_LEN, "vocab_size": VOCAB_SIZE, "n_layer": n_layer, "n_head": n_head,
            "n_kv_head": n_head // r["gqa"], "n_embd": n_embd, "window_pattern": WINDOW_PATTERN,
            "rotary_fraction": r["rotary"], "mlp_ratio": r["mlp"], "value_embeds": r["ve"]}


def read_config(path):
    config = json.loads(Path(path).read_text())
    required = {"run_id", "base_commit", "seed", "workers", "model", "training", "optimizer", "evaluation", "benchmark"}
    if set(config) != required:
        raise ValueError(f"config fields must be {sorted(required)}")
    if config["base_commit"] != BASE_COMMIT:
        raise ValueError("experiment must use the pinned historical commit")
    if not isinstance(config["run_id"], str) or not config["run_id"] or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in config["run_id"]):
        raise ValueError("run_id must be a nonempty directory-safe identifier")
    schemas = {
        "model": {"sequence_len", "vocab_size", "n_layer", "n_head", "n_kv_head", "n_embd", "window_pattern", "rotary_fraction", "mlp_ratio", "value_embeds"},
        "training": {"target_tokens", "tokens_per_scaling_param", "flop_budget", "wallclock_cap_seconds", "device_batch_size", "total_batch_size", "save_every", "eval_every", "max_steps", "compile"},
        "optimizer": {"embedding_lr", "unembedding_lr", "matrix_lr", "scalar_lr", "weight_decay", "adam_betas", "reference_batch_size", "warmup_ratio", "warmdown_ratio", "final_lr_frac"},
        "evaluation": {"interim_tokens", "final_tokens", "test_tokens"},
        "benchmark": {"batch_sizes", "prompt_lengths", "decode_tokens", "warmup", "repeats"},
    }
    for section, keys in schemas.items():
        if not isinstance(config[section], dict) or set(config[section]) != keys:
            raise ValueError(f"{section} fields must be {sorted(keys)}")
    def positive_int(value):
        return type(value) is int and value > 0
    if type(config["seed"]) is not int or not positive_int(config["workers"]):
        raise ValueError("seed must be an integer and workers must be positive")
    m, t, o, b = (config[k] for k in ("model", "training", "optimizer", "benchmark"))
    for name in ("sequence_len", "vocab_size", "n_layer", "n_head", "n_kv_head", "n_embd"):
        if not positive_int(m[name]):
            raise ValueError(f"model.{name} must be a positive integer")
    if m["n_embd"] % m["n_head"] or m["n_head"] % m["n_kv_head"] or m["n_embd"] < 32:
        raise ValueError("invalid embedding/query/KV head dimensions")
    f = m["rotary_fraction"]
    if type(f) not in (int, float) or not math.isfinite(f) or not 0 < f <= 1:
        raise ValueError("rotary_fraction must be in (0, 1]")
    dim = float(m["n_embd"] // m["n_head"] * f)
    if not dim.is_integer() or dim < 2 or int(dim) % 2:
        raise ValueError("rotary dimension must be an exact positive even integer")
    ratio = m["mlp_ratio"]
    if type(ratio) not in (int, float) or not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("mlp_ratio must be a positive finite number")
    hidden = float(ratio * m["n_embd"])
    if not hidden.is_integer() or int(hidden) % 128:
        raise ValueError("MLP hidden width must be an exact multiple of 128")
    if type(m["value_embeds"]) is not bool:
        raise ValueError("value_embeds must be a boolean")
    if not m["window_pattern"] or any(c not in "SL" for c in m["window_pattern"]):
        raise ValueError("invalid historical window pattern")
    for key in ("device_batch_size", "total_batch_size"):
        if not positive_int(t[key]):
            raise ValueError(f"training.{key} must be positive")
    if type(t["compile"]) is not bool:
        raise ValueError("compile must be boolean")
    for key in ("save_every", "eval_every"):
        if type(t[key]) is not int or (t[key] != -1 and t[key] <= 0):
            raise ValueError(f"{key} must be -1 or a positive integer")
    if t["max_steps"] is not None and not positive_int(t["max_steps"]):
        raise ValueError("max_steps must be null or positive (pilot only)")
    if type(t["wallclock_cap_seconds"]) not in (int, float) or not math.isfinite(t["wallclock_cap_seconds"]) or t["wallclock_cap_seconds"] <= 0:
        raise ValueError("wallclock_cap_seconds must be finite and positive")
    if type(t["tokens_per_scaling_param"]) not in (int, float) or not math.isfinite(t["tokens_per_scaling_param"]) or t["tokens_per_scaling_param"] <= 0:
        raise ValueError("tokens_per_scaling_param must be a positive finite number")
    if not positive_int(t["flop_budget"]):
        raise ValueError("flop_budget must be a positive integer (estimated training FLOPs per run)")
    if not positive_int(t["target_tokens"]) or t["target_tokens"] % t["total_batch_size"]:
        raise ValueError("target_tokens must be a positive multiple of total_batch_size")
    micro = config["workers"] * t["device_batch_size"] * m["sequence_len"]
    if t["total_batch_size"] % micro:
        raise ValueError("global token batch must be divisible by tokens per microstep")
    for key in ("embedding_lr", "unembedding_lr", "matrix_lr", "scalar_lr", "weight_decay"):
        if type(o[key]) not in (int, float) or not math.isfinite(o[key]) or o[key] < 0:
            raise ValueError(f"invalid optimizer.{key}")
    for key in ("warmup_ratio", "warmdown_ratio", "final_lr_frac"):
        if type(o[key]) not in (int, float) or not math.isfinite(o[key]) or not 0 <= o[key] <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    if o["warmup_ratio"] + o["warmdown_ratio"] > 1 or not positive_int(o["reference_batch_size"]):
        raise ValueError("invalid schedule or reference batch")
    if not isinstance(o["adam_betas"], list) or len(o["adam_betas"]) != 2 or any(type(v) not in (int, float) or not 0 <= v < 1 for v in o["adam_betas"]):
        raise ValueError("adam_betas must contain two values in [0, 1)")
    for key in ("interim_tokens", "final_tokens", "test_tokens"):
        if not positive_int(config["evaluation"][key]):
            raise ValueError(f"evaluation.{key} must be positive")
        if config["evaluation"][key] % micro:
            raise ValueError("evaluation tokens must divide exactly into complete validation microbatches")
    for key in ("batch_sizes", "prompt_lengths"):
        if not isinstance(b[key], list) or not b[key] or any(not positive_int(v) for v in b[key]):
            raise ValueError(f"invalid benchmark.{key}")
    for key in ("decode_tokens", "warmup", "repeats"):
        if not positive_int(b[key]):
            raise ValueError(f"benchmark.{key} must be positive")
    if b["decode_tokens"] < 2:
        raise ValueError("benchmark decode_tokens must be at least two to time decoding after TTFT")
    if max(b["prompt_lengths"]) + b["decode_tokens"] > 10 * m["sequence_len"]:
        raise ValueError("benchmark workload exceeds rotary cache capacity")
    return config


def validate_suite(configs):
    """Check a set of suite configs against RUNS and against each other, grouped by depth."""
    ids = [c["run_id"] for c in configs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate run IDs")
    signatures = [(json.dumps(c["model"], sort_keys=True), c["seed"], c["training"]["target_tokens"]) for c in configs]
    if len(set(signatures)) != len(signatures):
        raise ValueError("duplicate model configuration, seed and token target")
    by_depth, targets = {}, {}
    for config in configs:
        run_id = config["run_id"]
        if run_id not in RUNS:
            raise ValueError(f"unknown suite run {run_id}")
        if config["model"] != model_for(run_id) or config["seed"] != RUNS[run_id]["seed"]:
            raise ValueError(f"{run_id} model or seed differs from the RUNS table")
        t = config["training"]
        if config["workers"] != 1 or t["total_batch_size"] != SUITE_BATCH or t["max_steps"] is not None:
            raise ValueError(f"suite requires one worker, batch {SUITE_BATCH}, and no step cap")
        # Everything but the token target must match at a depth. Token-budget runs must also share one target;
        # a FLOP-budget run's target is derived from the shared flop_budget instead.
        rest = {k: v for k, v in config.items() if k not in ("run_id", "seed", "model")}
        rest["training"] = {k: v for k, v in t.items() if k != "target_tokens"}
        depth = RUNS[run_id]["depth"]
        target = targets.setdefault(depth, t["target_tokens"]) if RUNS[run_id]["budget"] == "tokens" else t["target_tokens"]
        if rest != by_depth.setdefault(depth, rest) or target != t["target_tokens"]:
            raise ValueError("suite configs at one depth may differ only in run_id, seed, KV heads, rotary fraction, "
                             "MLP ratio, value embeddings, and (FLOP-budget runs) token target")


def schedule(progress, optimizer):
    """LR multiplier and Muon weight-decay scale as functions of training progress in [0, 1] (steps / total steps)."""
    progress = min(max(progress, 0.0), 1.0)
    warmup, warmdown = optimizer["warmup_ratio"], optimizer["warmdown_ratio"]
    if warmup and progress < warmup:
        lr = progress / warmup
    elif warmdown and progress > 1 - warmdown:
        lr = optimizer["final_lr_frac"] + (1 - optimizer["final_lr_frac"]) * (1 - progress) / warmdown
    else:
        lr = 1.0
    return lr, 1 - progress
