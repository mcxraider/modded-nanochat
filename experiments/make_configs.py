"""Generate experiments/configs/<run_id>.json from the RUNS table in config.py."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.config import BASE_COMMIT, RUNS, model_for, read_config, validate_suite

# Per-depth policy shared by every run at that depth. PROVISIONAL until the GPU pilot measures
# tokens/s, peak VRAM, and evaluation cost; edit here and regenerate.
# target_tokens ~ 10 x scaling params (d8 ~42M, d12 ~110M); micro-batches must tile total_batch_size.
DEFAULTS = {
    8: {"target_tokens": 3200 * 131072, "device_batch_size": 32, "eval_every": 400, "save_every": 1600},
    12: {"target_tokens": 8400 * 131072, "device_batch_size": 16, "eval_every": 700, "save_every": 2100},
}
OPTIMIZER = {"embedding_lr": 0.3, "unembedding_lr": 0.004, "matrix_lr": 0.02, "scalar_lr": 0.5, "weight_decay": 0.2,
             "adam_betas": [0.8, 0.95], "reference_batch_size": 524288, "warmup_ratio": 0.0, "warmdown_ratio": 0.5,
             "final_lr_frac": 0.0}
EVALUATION = {"interim_tokens": 2**21, "final_tokens": 2**24, "test_tokens": 2**24}
BENCHMARK = {"batch_sizes": [1, 4, 16, 64], "prompt_lengths": [128, 2048, 8192], "decode_tokens": 256, "warmup": 2, "repeats": 5}


def build(run_id):
    r, d = RUNS[run_id], DEFAULTS[RUNS[run_id]["depth"]]
    return {
        "run_id": run_id, "base_commit": BASE_COMMIT, "seed": r["seed"], "workers": 1, "model": model_for(run_id),
        "training": {"target_tokens": d["target_tokens"], "wallclock_cap_seconds": 7200,
                     "device_batch_size": d["device_batch_size"], "total_batch_size": 131072,
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
