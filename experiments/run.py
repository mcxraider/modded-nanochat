"""Launch one config in a managed historical worktree; no current training defaults leak in."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.config import BASE_COMMIT, NODE_HOUR_USD, read_config
from experiments.resume import validate_saved_run, resume_action
from experiments.backend import patch_attention_loader, kernel_provenance


def patch_checkpoint_loader(source):
    needle = "    model_config = GPTConfig(**model_config_kwargs)"
    replacement = '''    model_class, config_class = GPT, GPTConfig
    if meta_data.get("model_type") == "modified_gpt_324e69c":
        from nanochat.modified_gpt import GPT as model_class, GPTConfig as config_class
    elif meta_data.get("model_type") is not None:
        raise ValueError(f"Unknown checkpoint model_type: {meta_data['model_type']}")
    model_config = config_class(**model_config_kwargs)'''
    if source.count(needle) != 1 or source.count("        model = GPT(model_config)") != 1:
        raise RuntimeError("historical checkpoint loader no longer matches the expected source")
    return source.replace(needle, replacement).replace("        model = GPT(model_config)", "        model = model_class(model_config)")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_runtime(directory):
    directory = Path(directory).resolve()
    marker = directory / ".experiment-overlay.json"
    if directory.exists():
        if not marker.is_file():
            raise RuntimeError(f"refusing to modify unmanaged directory {directory}")
        previous = json.loads(marker.read_text())
        for name, sha in previous.items():
            target = directory / name
            if not target.is_file() or digest(target) != sha:
                raise RuntimeError(f"runtime was edited: {target}; use another --worktree")
        head = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
        if head != BASE_COMMIT:
            raise RuntimeError("runtime HEAD must remain at the pinned commit")
        changed = subprocess.check_output(["git", "-C", str(directory), "diff", "--name-only"], text=True).splitlines()
        untracked = subprocess.check_output(["git", "-C", str(directory), "ls-files", "--others", "--exclude-standard"], text=True).splitlines()
        if any(p not in previous and p != marker.name for p in changed + untracked):
            raise RuntimeError("runtime contains unmanaged changes; use another --worktree")
    else:
        directory.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach", str(directory), BASE_COMMIT], check=True)
    files = [ROOT / "nanochat/modified_gpt.py", *sorted((ROOT / "experiments").glob("*.py"))]
    hashes = {}
    for path in files:
        relative = path.relative_to(ROOT)
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.resolve() != target.resolve():
            shutil.copy2(path, target)
        hashes[str(relative)] = digest(target)
    historical = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{BASE_COMMIT}:nanochat/checkpoint_manager.py"], text=True)
    target = directory / "nanochat/checkpoint_manager.py"
    target.write_text(patch_checkpoint_loader(historical))
    hashes["nanochat/checkpoint_manager.py"] = digest(target)
    historical_attention = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{BASE_COMMIT}:nanochat/flash_attention.py"], text=True)
    target = directory / "nanochat/flash_attention.py"
    target.write_text(patch_attention_loader(historical_attention))
    hashes["nanochat/flash_attention.py"] = digest(target)
    marker.write_text(json.dumps(hashes, indent=2) + "\n")
    return directory, hashes


def command(config, config_path, output, resume=None, interpreter=None):
    cmd = [interpreter or sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={config['workers']}",
           "-m", "experiments.train", "--config", str(config_path), "--output", str(output)]
    if resume is not None:
        cmd += ["--resume", str(resume)]
    return cmd


def run_process(cmd, *, timeout, **kwargs):
    """Stop torchrun and its workers together when a budget deadline expires."""
    if timeout is None:
        return subprocess.run(cmd, check=True, **kwargs)
    if timeout <= 0:
        raise subprocess.TimeoutExpired(cmd, timeout)
    with subprocess.Popen(cmd, start_new_session=True, **kwargs) as process:
        try:
            returncode = process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
        if returncode:
            raise subprocess.CalledProcessError(returncode, cmd)


def _launch(config_path, output_root, worktree, dry_run=False, resume=None, timeout_seconds=None):
    config_path = Path(config_path).resolve()
    config = read_config(config_path)
    output = Path(output_root).resolve() / config["run_id"]
    cmd = command(config, config_path, output, resume)
    print(json.dumps({"config": config, "command": cmd, "worktree": str(Path(worktree).resolve()), "output": str(output)}, indent=2), flush=True)
    if dry_run:
        return
    if output.exists() and resume is None:
        raise FileExistsError(f"run directory exists: {output}; choose --output-root or explicitly --resume STEP")
    if resume is not None and not output.is_dir():
        raise FileNotFoundError(f"no run to resume: {output}")
    launch_started = time.monotonic()
    runtime, overlay = prepare_runtime(worktree)
    runtime_python = runtime / ".venv/bin/python"
    interpreter = str(runtime_python) if runtime_python.is_file() else sys.executable
    cmd = command(config, output / "config.json", output, resume, interpreter)
    if resume is not None:
        validate_saved_run(output, config, overlay)
    if resume is None:
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        provenance = {"attention_kernel": kernel_provenance(), "base_commit": BASE_COMMIT, "overlay_sha256": overlay, "command": cmd,
                      "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                      "source_diff": subprocess.check_output(["git", "-C", str(ROOT), "diff", "HEAD"], text=True)}
        (output / "launch.json").write_text(json.dumps(provenance, indent=2) + "\n")
    # Freeze the actual source used, including new files absent from git diff.
    source = output / "source"
    if not source.exists():
        for relative in overlay:
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(runtime / relative, target)
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "1"))
    accounting_path = output / "rental_accounting.json"
    accounting = json.loads(accounting_path.read_text()) if accounting_path.exists() else {"launches": []}
    attempt = {"resume_step": resume, "started_unix_seconds": time.time(), "status": "running", "command": cmd}
    accounting["launches"].append(attempt)
    accounting_path.write_text(json.dumps(accounting, indent=2) + "\n")
    def remaining():
        return None if timeout_seconds is None else timeout_seconds - (time.monotonic() - launch_started)
    try:
        benchmark_only = resume is not None and resume_action(output, config["workers"]) == ("benchmark", resume)
        attempt["stage"] = "benchmark" if benchmark_only else "training"
        if not benchmark_only:
            if resume is not None:
                # Results from a later, damaged generation cannot describe the replayed model.
                for path in output.glob("evaluation_*.json"):
                    if int(path.stem.split("_")[-1]) > resume:
                        path.unlink()
                for name in ("result.json", "training_complete.json"):
                    path = output / name
                    if path.exists() and json.loads(path.read_text()).get("step", -1) > resume:
                        path.unlink()
            with (output / "train.log").open("a") as log:
                run_process(cmd, cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=remaining())
        benchmark_env = dict(env)
        benchmark_env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        with (output / "benchmark.log").open("a") as log:
            run_process([interpreter, "-m", "experiments.benchmark", "--run-dir", str(output)],
                           cwd=runtime, env=benchmark_env, stdout=log, stderr=subprocess.STDOUT, timeout=remaining())
        attempt["status"] = "complete"
    finally:
        if attempt["status"] == "running":
            attempt["status"] = "failed_or_interrupted"
        attempt["node_seconds"] = time.monotonic() - launch_started
        accounting["total_pipeline_node_seconds"] = sum(item.get("node_seconds", 0) for item in accounting["launches"])
        accounting["total_pipeline_gpu_hours"] = accounting["total_pipeline_node_seconds"] * config["workers"] / 3600
        accounting["cost_usd"] = accounting["total_pipeline_node_seconds"] * NODE_HOUR_USD / 3600
        accounting["note"] = "Excludes earlier dependency/data preparation, pilots, idle gaps, and unrecorded force-killed launches."
        accounting_path.write_text(json.dumps(accounting, indent=2) + "\n")
    print(f"Completed {config['run_id']}: {output}", flush=True)


@contextmanager
def runtime_lock(worktree):
    path = Path(worktree).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / (path.name + ".lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another experiment is using this runtime") from exc
        yield


def launch(config_path, output_root, worktree, dry_run=False, resume=None, timeout_seconds=None):
    if dry_run:
        return _launch(config_path, output_root, worktree, True, resume)
    with runtime_lock(worktree):
        return _launch(config_path, output_root, worktree, False, resume, timeout_seconds)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output-root", default=ROOT / "experiments/artifacts", type=Path)
    p.add_argument("--worktree", default=ROOT.parent / f"{ROOT.name}-experiments-324e69c", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", type=int)
    p.add_argument("--prepare-only", action="store_true", help="create/update the pinned worktree without launching training")
    args = p.parse_args()
    if args.prepare_only:
        if args.dry_run:
            raise ValueError("choose --prepare-only or --dry-run")
        read_config(args.config)
        with runtime_lock(args.worktree):
            runtime, _ = prepare_runtime(args.worktree)
        print(f"Prepared {runtime}. Install the pinned dependencies here before training.")
        return
    launch(args.config, args.output_root, args.worktree, args.dry_run, args.resume)


if __name__ == "__main__":
    main()
