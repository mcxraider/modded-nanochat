"""Run suite configurations sequentially (default: every run in RUNS order), stopping on failure."""
import argparse
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.config import RUNS, read_config, validate_suite
from experiments.run import launch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-dir", type=Path, default=ROOT / "experiments/configs")
    p.add_argument("--output-root", type=Path, default=ROOT / "experiments/artifacts")
    p.add_argument("--worktree", type=Path, default=ROOT.parent / f"{ROOT.name}-experiments-324e69c")
    p.add_argument("--runs", nargs="+", default=list(RUNS), metavar="RUN_ID", help="run IDs to launch, in order")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    unknown = [r for r in args.runs if r not in RUNS]
    if unknown:
        raise SystemExit(f"unknown run IDs: {unknown}; choose from {list(RUNS)}")
    paths = [args.config_dir / f"{run_id}.json" for run_id in args.runs]
    configs = [read_config(path) for path in paths]
    validate_suite(configs)
    if not args.dry_run:
        for c in configs:
            if (args.output_root / c["run_id"]).exists():
                raise FileExistsError(f"{c['run_id']} output already exists; use a new --output-root or resume the individual run")
    for path in paths:
        launch(path, args.output_root, args.worktree, args.dry_run)
    if not args.dry_run:
        subprocess.run([sys.executable, "-m", "experiments.report", "--output-root", str(args.output_root.resolve())], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
