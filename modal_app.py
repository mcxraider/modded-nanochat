"""Run the whole experiment suite on Modal: data, pilots, budget gates, 8 runs, report.

    modal run modal_app.py::gpu_check         # is the GPU available, does FA3 load?
    modal run modal_app.py::image_check       # CPU only: image builds, configs resolve
    modal run modal_app.py --pilot-only       # data + pilots + cost table, no full runs
    modal run --detach modal_app.py           # everything; re-running skips finished work
"""
import json
import hashlib
import math
import time
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import modal


GPU = "H100!"  # "!" blocks Modal's H200 upgrade; report.py rejects mixed GPUs across runs
CPU_CORES, MEMORY_GIB = 6, 24  # priced into config.NODE_HOUR_USD; change them together
SHARDS = 48  # training pool, shards 0..47; prepare() checks it covers the largest run
BUDGET_USD = 42.0  # application estimate; also configure a workspace usage limit
PREPARE_USD = 0.5  # CPU preparation is not in the rental accounting
VRAM_BYTES = 80e9

REPO = Path("/root/modded-nanochat")
if REPO.exists() and str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from experiments.backend import FA3_REPO, FA3_REVISION
WORKTREE = Path("/root/modded-nanochat-experiments-324e69c")
PYTHON = str(WORKTREE / ".venv/bin/python")
VOLUME_PATH = Path("/vol")
BASE_DIR = VOLUME_PATH / "nanochat"
ARTIFACTS = VOLUME_PATH / "artifacts"
PILOTS = VOLUME_PATH / "pilots"
PILOT_RUNS = ("d8-b0-s1", "d8-gqa2", "d8-gqa2-r75", "d8-gqa2-pm", "d8-gqa4", "d8-gqa2-r25")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    # .git comes along: train.py requires a real checkout at the pinned commit.
    .add_local_dir(Path(__file__).parent, str(REPO), copy=True,
                   ignore=[".env", ".venv", "experiments/artifacts", "dev-ignore", "**/__pycache__"])
    .run_commands(
        "git config --global --add safe.directory '*'",
        f"git -C {REPO} config core.fileMode false",
        f"git -C {REPO} worktree prune",  # the copied .git still lists the local worktree
        f"python {REPO}/experiments/run.py --config {REPO}/experiments/configs/d8-b0-s1.json --prepare-only",
        f"uv sync --frozen --no-dev --extra gpu --project {WORKTREE}",
        # Baked in so containers don't fetch ~800 MB on every start. No GPU here, so nothing is loaded yet.
        f"""{PYTHON} -c "from kernels import install_kernel; install_kernel('{FA3_REPO}', revision='{FA3_REVISION}')" """,
    )
    .env({"NANOCHAT_BASE_DIR": str(BASE_DIR)})
)
volume = modal.Volume.from_name("nanochat-experiments", create_if_missing=True, version=2)
writer_locks = modal.Dict.from_name("nanochat-experiments-locks", create_if_missing=True)
app = modal.App("nanochat-experiments", image=image)

COUNT_TOKENS = """
import sys
import pyarrow.parquet as pq
from nanochat.tokenizer import get_tokenizer
texts = pq.ParquetFile(sys.argv[1]).read().column("text").to_pylist()
print(sum(len(ids) for ids in get_tokenizer().encode(texts, num_threads=8)))
"""


def in_worktree(*args, **kwargs):
    return subprocess.run([PYTHON, *args], cwd=WORKTREE, check=True, **kwargs)


def experiments_modules():
    """Imported late: the suite code only exists inside the image."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from experiments import config, resume, run
    return config, resume, run


def suite_config(run_id):
    return REPO / "experiments/configs" / f"{run_id}.json"


def configure_namespace(name):
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("experiment must be a directory-safe name")
    global ARTIFACTS, PILOTS
    ARTIFACTS = VOLUME_PATH / "experiments" / name / "artifacts"
    PILOTS = VOLUME_PATH / "experiments" / name / "pilots"


def read_run_json(run_dir, name):
    return json.loads((run_dir / name).read_text())


@app.function(cpu=8.0, memory=32 * 1024, volumes={VOLUME_PATH: volume}, timeout=4 * 3600)
def prepare():
    _, resume, _ = experiments_modules()
    with resume.exclusive_writer(writer_locks):
        volume.reload()
        try:
            return prepare_data()
        finally:
            volume.commit()


def prepare_data():
    """Download shards and train the tokenizer, skipping whatever is already on the Volume."""
    config, _, _ = experiments_modules()
    data = BASE_DIR / "base_data_climbmix"
    def missing():
        return [i for i in (*range(SHARDS), 6541, 6542) if not (data / f"shard_{i:05d}.parquet").exists()]
    if missing():
        # dataset.py exits 0 even when downloads fail, hence the second check.
        in_worktree("-m", "nanochat.dataset", "-n", str(SHARDS), "-w", "8")
        in_worktree("-c", "from nanochat.dataset import download_single_file; download_single_file(6541)")
        if missing():
            raise RuntimeError(f"shards failed to download: {missing()}")
        volume.commit()
    if not (BASE_DIR / "tokenizer/token_bytes.pt").exists():  # written last by tok_train
        in_worktree("-m", "scripts.tok_train")
        volume.commit()
    tokens = int(in_worktree("-c", COUNT_TOKENS, str(data / "shard_00000.parquet"),
                             capture_output=True, text=True).stdout.strip().splitlines()[-1])
    largest = max(config.read_config(suite_config(r))["training"]["target_tokens"] for r in config.RUNS)
    needed = 1.1 * largest / 0.65  # best-fit packing crops about 35% of raw tokens
    print(f"{tokens:,} tokens per shard; pool {tokens * SHARDS:,}; needed {needed:,.0f}", flush=True)
    if tokens * SHARDS < needed:
        raise RuntimeError(f"raise SHARDS: {SHARDS} shards hold {tokens * SHARDS:,} tokens, need {needed:,.0f}")


def run_one(config_path, output_root, budget):
    """Launch, resume or skip one run; safe to call again after any interruption."""
    config, resume, run = experiments_modules()
    cfg = config.read_config(config_path)
    run_dir = output_root / cfg["run_id"]
    _, overlay = run.prepare_runtime(WORKTREE)
    resume.validate_saved_run(run_dir, cfg, overlay)
    action, step = resume.resume_action(run_dir, cfg["workers"])
    print(f"{cfg['run_id']}: {action} {step if step is not None else ''}", flush=True)
    if action == "skip":
        return
    if action == "fresh":
        shutil.rmtree(run_dir, ignore_errors=True)
    elif action == "resume":
        resume.truncate_metrics(run_dir / "metrics.jsonl", step)
    try:
        run.launch(config_path, output_root, WORKTREE, resume=step, timeout_seconds=budget.seconds_left())
    except subprocess.CalledProcessError:
        for name in ("train.log", "benchmark.log"):  # these go to files, not the Modal logs
            if (run_dir / name).exists():
                print(f"--- tail of {name}\n" + "\n".join((run_dir / name).read_text().splitlines()[-40:]), flush=True)
        raise
    finally:
        volume.commit()


def pilot_config(run_id, pilot_root=None):
    """The documented 10-update pilot, under its own run ID so it never counts as a suite run."""
    config, _, _ = experiments_modules()
    cfg = config.read_config(suite_config(run_id))
    cfg["run_id"] = f"pilot-{run_id}"
    cfg["training"].update(max_steps=10, eval_every=-1, save_every=-1)
    cfg["evaluation"].update(interim_tokens=131072, final_tokens=131072, test_tokens=131072)
    cfg["benchmark"].update(warmup=1, repeats=1)  # full shapes and decode length, fewer timing repeats
    path = (pilot_root or PILOTS) / "configs" / f"{cfg['run_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    return path


def check_pilot(run_dir, workers):
    """Everything that makes the full runs invalid or pointless; returns a list of problems."""
    hardware = read_run_json(run_dir, "hardware.json")
    result = read_run_json(run_dir, "result.json")
    inference = read_run_json(run_dir, "inference.json")
    problems = []
    if "H100" not in hardware["gpu"] or hardware["world_size"] != workers:
        problems.append(f"unexpected hardware: {hardware['gpu']} x {hardware['world_size']}")
    # flash_attention.py swallows FA3 load errors and silently falls back to SDPA.
    for label, backend in (("training", hardware["attention_backend"]), ("inference", inference["attention_backend"])):
        if backend != "FA3":
            problems.append(f"{label} backend is {backend}, expected FA3")
    # Last-token prefill keeps the largest d8 workload near 12 GiB on an 80 GB H100,
    # so any OOM here is a regression that would leave holes in the full grid.
    for row in inference["rows"]:
        if row["status"] != "ok" or row["kv_cache_bytes_allocated"] != row["kv_cache_bytes_analytic"]:
            problems.append(f"bad inference row: {row}")
    if result["peak_vram_bytes"] > 0.9 * VRAM_BYTES:
        problems.append(f"peak VRAM {result['peak_vram_bytes'] / 1e9:.1f} GB is above 90%")
    return [f"{run_dir.name}: {p}" for p in problems]


@app.function(gpu=GPU, cpu=float(CPU_CORES), memory=MEMORY_GIB * 1024, volumes={VOLUME_PATH: volume},
              retries=modal.Retries(initial_delay=0.0, max_retries=5), single_use_containers=True,
              max_containers=1, timeout=12 * 3600)
def suite(budget_usd=BUDGET_USD, pilot_only=False, experiment="default"):
    configure_namespace(experiment)
    config, resume, _ = experiments_modules()
    with resume.exclusive_writer(writer_locks):
        volume.reload()
        legacy = PREPARE_USD + resume.spent_seconds([VOLUME_PATH / "pilots", VOLUME_PATH / "artifacts", VOLUME_PATH / "experiments"]) * config.NODE_HOUR_USD / 3600
        with resume.BudgetSession(VOLUME_PATH / "accounting/suite.json", budget_usd,
                                  config.NODE_HOUR_USD, legacy, volume.commit) as budget:
            try:
                return run_suite(budget_usd, pilot_only, budget)
            except subprocess.TimeoutExpired:
                return {"status": "stopped_for_budget", "spent_usd": budget.spent()}


def run_suite(budget_usd, pilot_only, budget):
    config, resume, run = experiments_modules()
    os.environ["OMP_NUM_THREADS"] = "1"  # Modal sets this to the CPU count; the launcher's documented default is 1
    configs = {r: config.read_config(suite_config(r)) for r in config.RUNS}
    config.validate_suite(list(configs.values()))
    def spent():
        return budget.spent()
    def usd(seconds):
        return seconds * config.NODE_HOUR_USD / 3600

    _, overlay = run.prepare_runtime(WORKTREE)
    for r, cfg in configs.items():
        resume.validate_saved_run(ARTIFACTS / r, cfg, overlay)
    fingerprint = hashlib.sha256(json.dumps({"configs": configs, "overlay": overlay}, sort_keys=True).encode()).hexdigest()[:16]
    pilot_root = PILOTS / fingerprint
    pilot_root.mkdir(parents=True, exist_ok=True)
    # Recheck the current container even if pilot artifacts were completed earlier.
    try:
        run.run_process([PYTHON, "-m", "experiments.preflight", "--output", str(pilot_root / "preflight.json")],
                        cwd=WORKTREE, timeout=budget.seconds_left())
    except subprocess.CalledProcessError:
        return {"status": "preflight_failed", "problems": ["GPU correctness check failed; see Modal logs"]}
    problems, speeds, pilot_results, pilot_inference = [], {}, {}, {}
    for run_id in PILOT_RUNS:
        try:
            run_one(pilot_config(run_id, pilot_root), pilot_root, budget)
        except subprocess.CalledProcessError:
            return {"status": "pilot_failed", "problems": [f"{run_id}: pilot process failed; see logs"]}
        run_dir = pilot_root / f"pilot-{run_id}"
        problems += check_pilot(run_dir, configs[run_id]["workers"])
        if problems:
            return {"status": "pilot_failed", "problems": problems}
        speeds[run_id] = resume.pilot_tokens_per_second(run_dir / "metrics.jsonl")
        pilot_results[run_id] = read_run_json(run_dir, "result.json")
        pilot_inference[run_id] = read_run_json(run_dir, "inference.json")
    print("pilot tokens/s:", json.dumps(speeds), flush=True)
    print(f"GQA-2 / B0 speed: d8 {speeds['d8-gqa2'] / speeds['d8-b0-s1']:.3f}", flush=True)
    if problems:
        return {"status": "pilot_failed", "problems": problems}

    projected = {}
    for r, cfg in configs.items():
        pilot = r if r in speeds else "d8-b0-s1"  # baseline seed replicates use the B0 pilot
        projected[r] = resume.project_seconds(cfg, speeds[pilot],
            evaluation_tokens_per_second=262144 / pilot_results[pilot]["evaluation_seconds"],
            benchmark_seconds=pilot_inference[pilot]["elapsed_seconds"] *
                (cfg["benchmark"]["warmup"] + cfg["benchmark"]["repeats"]) / 2)
    remaining = [r for r in config.RUNS if resume.resume_action(ARTIFACTS / r, configs[r]["workers"])[0] != "skip"]
    total = spent() + usd(sum(projected[r]["total"] for r in remaining))
    table = {r: {"train_minutes": round(projected[r]["training"] / 60, 1), "usd": round(usd(projected[r]["total"]), 2)}
             for r in remaining}
    print(json.dumps({"spent_usd": round(spent(), 2), "projected_total_usd": round(total, 2), "runs": table}, indent=2), flush=True)
    over_cap = [r for r in remaining if projected[r]["training"] > 0.9 * configs[r]["training"]["wallclock_cap_seconds"]]
    if total > budget_usd or over_cap:
        return {"status": "stopped_at_gate", "projected_total_usd": total, "budget_usd": budget_usd, "over_wallclock_cap": over_cap}
    if pilot_only:
        return {"status": "pilot_ok", "projected_total_usd": total, "runs": table}

    for run_id in config.RUNS:
        # Gate estimates; subprocess deadlines enforce the remaining application budget.
        if run_id in remaining and spent() + usd(projected[run_id]["total"]) > budget_usd:
            return {"status": "stopped_for_budget", "spent_usd": spent(), "next_run": run_id,
                    "not_started": remaining[remaining.index(run_id):]}
        run_one(suite_config(run_id), ARTIFACTS, budget)

    run.run_process([PYTHON, "-m", "experiments.report", "--output-root", str(ARTIFACTS)],
                    cwd=WORKTREE, timeout=budget.seconds_left())
    try:
        run.run_process([PYTHON, "experiments/visualise.py", "--output-root", str(ARTIFACTS)],
                        cwd=WORKTREE, timeout=budget.seconds_left())
    except subprocess.CalledProcessError:
        print("visualise failed; the results themselves are intact. Rerun it locally on the tarball.", flush=True)
    with tarfile.open(ARTIFACTS.parent / "results.tar.gz", "w:gz") as tar:
        tar.add(ARTIFACTS, arcname="artifacts", filter=lambda m: None if "/checkpoints" in m.name else m)
    with tarfile.open(ARTIFACTS.parent / "checkpoints.tar", "w") as tar:
        for r in config.RUNS:
            result = read_run_json(ARTIFACTS / r, "result.json")
            step = result["checkpoint_step"]
            from experiments.checkpoints import filenames
            for name in [*filenames(step, configs[r]["workers"]), f"timing_{step:06d}.json"]:
                tar.add(ARTIFACTS / r / "checkpoints" / name, arcname=f"artifacts/{r}/checkpoints/{name}")
    volume.commit()
    return {"status": "complete", "spent_usd": spent(), "results": str((ARTIFACTS.parent / "results.tar.gz").relative_to(VOLUME_PATH)),
            "checkpoints": str((ARTIFACTS.parent / "checkpoints.tar").relative_to(VOLUME_PATH))}


@app.function(volumes={VOLUME_PATH: volume}, timeout=600)
def recover(spent_usd: float, confirm_all_stopped: bool = False):
    """Manual hard-crash recovery; stop all old Apps and reconcile actual spend first."""
    if not confirm_all_stopped:
        raise ValueError("stop all old Apps, then pass --confirm-all-stopped")
    if not math.isfinite(spent_usd) or spent_usd < 0:
        raise ValueError("spent_usd must be the reconciled nonnegative total, including earlier attempts")
    volume.reload()
    ledger = VOLUME_PATH / "accounting/suite.json"
    previous = json.loads(ledger.read_text()) if ledger.exists() else None
    lock = writer_locks.get("writer", None)
    from experiments.train import write_json
    ledger.parent.mkdir(parents=True, exist_ok=True)
    write_json(ledger.parent / f"recovery_{time.time_ns()}.json",
               {"previous": previous, "previous_writer": lock, "reconciled_spent_usd": spent_usd})
    write_json(ledger, {"spent_usd": spent_usd, "reserved_usd": 0})
    volume.commit()
    if lock is not None:
        writer_locks.pop("writer")
    return {"status": "reconciled", "spent_usd": spent_usd}


@app.function(timeout=24 * 3600)
def pipeline(budget_usd=BUDGET_USD, pilot_only=False, experiment="default"):
    """Remote, so the next step still starts after the laptop that launched it goes away."""
    prepare.remote()
    return suite.remote(budget_usd, pilot_only, experiment)


@app.function(timeout=600)
def image_check():
    """CPU only, no spend beyond the image build."""
    subprocess.run([sys.executable, "experiments/run_all.py", "--dry-run"], cwd=REPO, stdout=subprocess.DEVNULL, check=True)
    print("image OK: worktree valid, all suite configs resolve")


@app.function(gpu=GPU, timeout=300)
def gpu_check():
    subprocess.run(["nvidia-smi"], check=True)
    in_worktree("-c", "from experiments.backend import require_fa3, kernel_provenance; require_fa3(); print(kernel_provenance())")


@app.local_entrypoint()
def main(pilot_only: bool = False, budget_usd: float = BUDGET_USD, experiment: str = "default"):
    call = pipeline.spawn(budget_usd, pilot_only, experiment)
    print(f"Started {call.object_id}; safe to close this terminal if launched with --detach.")
    print(json.dumps(call.get(), indent=2))
