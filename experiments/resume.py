"""Bookkeeping that lets an interrupted suite pick up where it stopped: what to do with a run directory, and what it cost."""
from contextlib import contextmanager
import json
import math
import time
import uuid
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
    """Legacy accounting, used once to seed the durable Modal budget ledger."""
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


def validate_saved_run(run_dir, config, overlay):
    """Reject stale outputs before skipping, resuming, or deleting an incomplete run."""
    run_dir = Path(run_dir)
    for name, expected in (("config.json", config), ("launch.json", overlay)):
        path = run_dir / name
        if not path.exists():
            if (run_dir / "inference.json").exists() or any((run_dir / "checkpoints").glob("timing_*.json")):
                raise ValueError(f"missing provenance: {path}")
            continue
        saved = json.loads(path.read_text())
        if name == "launch.json":
            saved = saved["overlay_sha256"]
        if saved != expected:
            raise ValueError(f"saved {name} differs from current run: {run_dir}; use a new Volume")


@contextmanager
def exclusive_writer(locks):
    """One writer across Apps; a hard-killed owner requires manual recovery."""
    owner = uuid.uuid4().hex
    if not locks.put("writer", owner, skip_if_exists=True):
        raise RuntimeError("another pipeline owns the Volume writer lock; stop all old Apps before clearing a stale lock")
    try:
        yield
    finally:
        locks.pop("writer")


class BudgetSession:
    """Reserve remaining funds durably; interrupted reservations count as spent."""

    def __init__(self, path, budget_usd, hourly_usd, legacy_usd, commit):
        if not math.isfinite(budget_usd) or budget_usd <= 0:
            raise ValueError("budget must be a positive finite USD amount")
        self.path = Path(path)
        self.hourly_usd = hourly_usd
        self.commit = commit
        self.ledger = json.loads(self.path.read_text()) if self.path.exists() else {"spent_usd": legacy_usd}
        self.before = self.ledger["spent_usd"]
        self.remaining_usd = max(0.0, budget_usd - self.before)

    def write(self, value):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(self.path)
        self.commit()

    def __enter__(self):
        if self.remaining_usd <= 0:
            raise RuntimeError("budget exhausted or reserved by an interrupted attempt; reconcile accounting before restarting")
        self.started = time.monotonic()
        self.write({"spent_usd": self.before + self.remaining_usd, "reserved_usd": self.remaining_usd,
                    "started_unix_seconds": time.time()})
        return self

    def spent(self):
        return self.before + (time.monotonic() - self.started) * self.hourly_usd / 3600

    def seconds_left(self):
        return max(0.0, self.remaining_usd * 3600 / self.hourly_usd - (time.monotonic() - self.started))

    def __exit__(self, *exc):
        self.write({"spent_usd": self.spent(), "reserved_usd": 0})
