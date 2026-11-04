"""Bookkeeping that lets an interrupted suite pick up where it stopped: what to do with a run directory, and what it cost."""
import json
from pathlib import Path
import re
import statistics

# Compilation, checkpoints and the benchmark grid. A guess until a real run measures it.
OVERHEAD_SECONDS = 420


def resume_action(run_dir, workers):
    """Return ("skip", None), ("resume", step) from the newest complete checkpoint, or ("fresh", None)."""
    run_dir = Path(run_dir)
    if (run_dir / "inference.json").exists():  # written last by the launcher
        return "skip", None
    checkpoints = run_dir / "checkpoints"
    steps = [int(m.group(1)) for m in (re.fullmatch(r"timing_(\d+)\.json", p.name) for p in checkpoints.glob("timing_*.json")) if m]
    for step in sorted(steps, reverse=True):
        # train.py writes timing after every rank has saved, but a lost commit could still leave gaps.
        names = [f"model_{step:06d}.pt", f"meta_{step:06d}.json"]
        for rank in range(workers):
            names += [f"optim_{step:06d}_rank{rank}.pt", f"runtime_{step:06d}_rank{rank}.pt"]
        if all((checkpoints / name).is_file() for name in names):
            return "resume", step
    return "fresh", None


def truncate_metrics(path, step):
    """Drop rows the resumed run will log again: anything past `step`, final evaluations, and a half-written last line."""
    path = Path(path)
    if not path.exists():
        return
    kept = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row["step"] <= step and not row.get("final"):
            kept.append(line)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(line + "\n" for line in kept))
    temporary.replace(path)


def spent_seconds(roots):
    """Node seconds recorded across run directories. Force-killed attempts leave no record; budget margin covers those."""
    total = 0.0
    for root in roots:
        for path in Path(root).glob("*/rental_accounting.json"):
            total += json.loads(path.read_text()).get("total_pipeline_node_seconds", 0)
    return total


def pilot_tokens_per_second(metrics_path):
    """Median training speed from a pilot, skipping the steps that include compilation."""
    rows = [json.loads(line) for line in Path(metrics_path).read_text().splitlines()]
    return statistics.median(r["tokens_per_second"] for r in rows if r["kind"] == "train" and r["step"] > 2)


def project_seconds(config, tokens_per_second):
    """Projected time for a full run. `training` is what counts toward the wall-clock cap (it includes interim validation)."""
    t, e = config["training"], config["evaluation"]
    updates = t["target_tokens"] // t["total_batch_size"]
    # The loop skips the interim evaluation on the final update. Evaluation is charged at the training rate, which overestimates it.
    interims = (updates - 1) // t["eval_every"] if t["eval_every"] > 0 else 0
    training = (t["target_tokens"] + interims * e["interim_tokens"]) / tokens_per_second
    total = training + (e["final_tokens"] + e["test_tokens"]) / tokens_per_second + OVERHEAD_SECONDS
    return {"training": training, "total": total}
