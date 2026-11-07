"""Bookkeeping that lets an interrupted suite pick up where it stopped: what to do with a run directory, and what it cost."""
from contextlib import contextmanager
import json
import math
import time
import uuid
from pathlib import Path
import re
import statistics

from experiments.checkpoints import valid as valid_checkpoint

# Reserve for compilation, checkpoint IO, and export, beyond measured evaluation/inference.
OVERHEAD_SECONDS = 420


def resume_action(run_dir, workers):
    """Return skip, benchmark-only, training/evaluation resume, or fresh from verified evidence."""
    run_dir = Path(run_dir)
    if (run_dir / "inference.json").exists():  # written last by the launcher
        return "skip", None
    checkpoints = run_dir / "checkpoints"
    steps = [int(m.group(1)) for m in (re.fullmatch(r"timing_(\d+)\.json", p.name) for p in checkpoints.glob("timing_*.json")) if m]
    for step in sorted(steps, reverse=True):
        if valid_checkpoint(checkpoints, step, workers):
            result_path = run_dir / "result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if result.get("checkpoint_step") == step:
                    return "benchmark", step
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
        for path in Path(root).rglob("rental_accounting.json"):
            total += json.loads(path.read_text()).get("total_pipeline_node_seconds", 0)
    return total


def pilot_tokens_per_second(metrics_path):
    """Median training speed from a pilot, skipping the steps that include compilation."""
    rows = [json.loads(line) for line in Path(metrics_path).read_text().splitlines()]
    return statistics.median(r["tokens_per_second"] for r in rows if r["kind"] == "train" and r["step"] > 2)


def pilot_data_wait_fraction(metrics_path):
    """Median share of each pilot update spent blocked on the data worker (compile steps skipped)."""
    rows = [json.loads(line) for line in Path(metrics_path).read_text().splitlines()]
    return statistics.median(r["data_wait_seconds"] / r["step_seconds"] for r in rows if r["kind"] == "train" and r["step"] > 2)


def project_seconds(config, tokens_per_second, evaluation_tokens_per_second=None, benchmark_seconds=None):
    """Projected time for a full run. `training` is what counts toward the wall-clock cap (it includes interim validation)."""
    t, e = config["training"], config["evaluation"]
    updates = t["target_tokens"] // t["total_batch_size"]
    # The loop skips interim evaluation on the final update. Prefer measured evaluation speed.
    interims = (updates - 1) // t["eval_every"] if t["eval_every"] > 0 else 0
    eval_speed = evaluation_tokens_per_second or tokens_per_second
    training = t["target_tokens"] / tokens_per_second + interims * e["interim_tokens"] / eval_speed
    total = training + (e["final_tokens"] + e["test_tokens"]) / eval_speed + OVERHEAD_SECONDS
    if benchmark_seconds is not None:
        total += benchmark_seconds * 1.25  # timing variance and longer steady-state runs
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
            raise ValueError(f"saved {name} differs from current run: {run_dir}; use a new experiment namespace")


@contextmanager
def exclusive_writer(locks, owner=None):
    """One writer across Apps. A retry of the same owner (e.g. after preemption) reclaims
    its own stale lock; any other hard-killed owner requires manual recovery."""
    owner = owner or uuid.uuid4().hex
    if not locks.put("writer", owner, skip_if_exists=True) and locks.get("writer") != owner:
        raise RuntimeError("another pipeline owns the Volume writer lock; stop all old Apps before clearing a stale lock")
    try:
        yield
    finally:
        locks.pop("writer")


class BudgetSession:
    """Reserve remaining funds durably; interrupted reservations count as spent, except that a
    retry by the same owner settles its own reservation at wall-clock time since it started
    (over-counts any gap between attempts, never under-counts)."""

    def __init__(self, path, budget_usd, hourly_usd, legacy_usd, commit, owner=None):
        if not math.isfinite(budget_usd) or budget_usd <= 0:
            raise ValueError("budget must be a positive finite USD amount")
        self.path = Path(path)
        self.hourly_usd = hourly_usd
        self.commit = commit
        self.owner = owner
        self.ledger = json.loads(self.path.read_text()) if self.path.exists() else {"spent_usd": legacy_usd}
        self.before = self.ledger["spent_usd"]
        if owner and self.ledger.get("reserved_usd") and self.ledger.get("owner") == owner:
            used = max(0.0, time.time() - self.ledger["started_unix_seconds"]) * hourly_usd / 3600
            self.before = self.ledger["before_usd"] + min(self.ledger["reserved_usd"], used)
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
                    "started_unix_seconds": time.time(), "before_usd": self.before, "owner": self.owner})
        return self

    def spent(self):
        return self.before + (time.monotonic() - self.started) * self.hourly_usd / 3600

    def seconds_left(self):
        return max(0.0, self.remaining_usd * 3600 / self.hourly_usd - (time.monotonic() - self.started))

    def __exit__(self, *exc):
        self.write({"spent_usd": self.spent(), "reserved_usd": 0})
