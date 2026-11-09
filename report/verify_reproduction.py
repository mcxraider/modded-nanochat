"""Check that the report, the code and the saved d8-v5 results agree. No GPU, data or LaTeX needed.

Usage (from the repository root):  uv run --frozen python report/verify_reproduction.py
Read-only: everything is regenerated in a temporary directory. Prints one PASS/FAIL line per
check and exits non-zero if any check fails.

Checks
  1. checksums    results/d8-v5/export/SHA256SUMS (only the checkpoint binaries kept out of git may be absent)
  2. provenance   each run's launch.json -> launch commit, pinned base commit, and file hashes of the code
                  that ran, against the saved source snapshot and the current working tree
  3. configs      experiments/make_configs.py regenerates experiments/configs/*.json and each run's config.json
  4. analysis     experiments.report rebuilds tables.md byte for byte and comparison.json within 1e-9
  5. numbers      headline numbers quoted in report/report.tex equal values computed from the saved results
  6. figures      report/make_figures.py regenerates every PDF the report includes, byte for byte
  7. source       every figure, citation key and repository path named in report.tex exists
"""
import fnmatch
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.config import BASE_COMMIT, RUNS  # noqa: E402

RESULTS = ROOT / "results/d8-v5"
ARTIFACTS = RESULTS / "artifacts"
REPORT = ROOT / "report"
TEX = (REPORT / "report.tex").read_text()
LAUNCH_COMMIT = "c07aa7e"
# Large binaries recorded in SHA256SUMS but deliberately not committed (see results/README.md).
MAY_BE_ABSENT = ("export/checkpoints.tar", "pilots/*/pilot-*/checkpoints/model_*.pt",
                 "pilots/*/pilot-*/checkpoints/optim_*_rank*.pt")
# Overlay files copied verbatim into the pinned runtime; the two other overlay files are patched from
# upstream nanochat 324e69c at launch, so only their saved snapshot is comparable.
VERBATIM = re.compile(r"^(experiments/[^/]+\.py|nanochat/modified_gpt\.py)$")

failures = []


def report(name, problems, detail=""):
    status = "FAIL" if problems else "PASS"
    print(f"[{status}] {name}" + (f": {detail}" if detail else ""))
    for p in problems[:20]:
        print(f"         - {p}")
    if len(problems) > 20:
        print(f"         - ... and {len(problems) - 20} more")
    if problems:
        failures.append(name)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def run_py(*args):
    return subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True)


def check_checksums():
    problems, missing, ok = [], 0, 0
    for line in (RESULTS / "export/SHA256SUMS").read_text().splitlines():
        expected, rel = line.split(None, 1)
        rel = rel.lstrip("*")
        path = RESULTS / rel
        if not path.exists():
            if any(fnmatch.fnmatch(rel, pat) for pat in MAY_BE_ABSENT):
                missing += 1
            else:
                problems.append(f"missing {rel}")
        elif sha256(path) != expected:
            problems.append(f"hash mismatch {rel}")
        else:
            ok += 1
    report("1 checksums", problems, f"{ok} files match; {missing} checkpoint binaries not in git (allowed)")


def check_provenance():
    problems = []
    recovered = RESULTS / "export/recovered-source"
    for run in RUNS:
        launch = load(ARTIFACTS / run / "launch.json")
        if not launch["source_commit"].startswith(LAUNCH_COMMIT):
            problems.append(f"{run}: source_commit {launch['source_commit'][:7]} != {LAUNCH_COMMIT}")
        if launch["source_diff"]:
            problems.append(f"{run}: launched from a dirty tree")
        if launch["base_commit"] != BASE_COMMIT:
            problems.append(f"{run}: base_commit {launch['base_commit'][:7]} != {BASE_COMMIT[:7]}")
        for rel, expected in launch["overlay_sha256"].items():
            snap = ARTIFACTS / run / "source" / rel
            if not snap.exists():
                snap = recovered / rel
            if not snap.exists() or sha256(snap) != expected:
                problems.append(f"{run}: saved source {rel} does not match launch hash")
            if VERBATIM.match(rel) and sha256(ROOT / rel) != expected:
                problems.append(f"{run}: working-tree {rel} differs from the code that ran")
    report("2 provenance", problems, f"{len(RUNS)} runs launched from {LAUNCH_COMMIT} on nanochat {BASE_COMMIT[:7]}; "
                                     "repository code matches the code that ran")


def check_configs(tmp):
    out = tmp / "configs"
    proc = run_py("experiments/make_configs.py", "--config-dir", out)
    if proc.returncode:
        return report("3 configs", [proc.stderr.strip()[-500:]])
    problems = []
    for run in RUNS:
        fresh = load(out / f"{run}.json")
        if load(ROOT / "experiments/configs" / f"{run}.json") != fresh:
            problems.append(f"experiments/configs/{run}.json is stale")
        if load(ARTIFACTS / run / "config.json") != fresh:
            problems.append(f"{run}: saved config.json differs from experiments/config.py")
    report("3 configs", problems, f"{len(RUNS)} configs regenerated from experiments/config.py")


def close(a, b, path, problems):
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            problems.append(f"{path}: keys differ")
        for k in a.keys() & b.keys():
            close(a[k], b[k], f"{path}.{k}", problems)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            problems.append(f"{path}: length differs")
        for i, (x, y) in enumerate(zip(a, b)):
            close(x, y, f"{path}[{i}]", problems)
    elif isinstance(a, float) or isinstance(b, float):
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))
                and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)):
            problems.append(f"{path}: {a!r} != {b!r}")
    elif a != b:
        problems.append(f"{path}: {a!r} != {b!r}")


def check_analysis(tmp):
    out = tmp / "artifacts"
    shutil.copytree(ARTIFACTS, out)
    for name in ("comparison.json", "tables.md"):
        (out / name).unlink()
    proc = run_py("-m", "experiments.report", "--output-root", out)
    if proc.returncode:
        return report("4 analysis", [proc.stderr.strip()[-500:]])
    problems = []
    if (out / "tables.md").read_bytes() != (ARTIFACTS / "tables.md").read_bytes():
        problems.append("tables.md differs")
    close(load(out / "comparison.json"), load(ARTIFACTS / "comparison.json"), "comparison", problems)
    report("4 analysis", problems, "tables.md identical; comparison.json equal to 1e-9 relative")


def check_numbers():
    c = load(ARTIFACTS / "comparison.json")
    res = {r: load(ARTIFACTS / r / "result.json") for r in RUNS}
    comp = {(x["run_id"], x["reference"]): x for x in c["comparisons"]}
    thousands = lambda n: f"{n:,}".replace(",", "{,}")
    pct = lambda x: f"{(x - 1) * 100:.2f}\\%"
    claims = [
        ("B0 seed 1 test BPB", f"{res['d8-b0-s1']['test_bpb']:.6f}"),
        ("B0 seed 2 test BPB", f"{res['d8-b0-s2']['test_bpb']:.6f}"),
        ("B0 seed 3 test BPB", f"{res['d8-b0-s3']['test_bpb']:.6f}"),
        ("B0 mean test BPB", f"{c['noise']['mean']:.6f}"),
        ("B0 seed s.d. s", f"{c['noise']['std']:.6f}"),
        ("resolution threshold 2*sqrt(2)*s", f"{c['noise']['threshold']:.6f}"),
        ("GQA-2 minus B0 test BPB", f"{comp[('d8-gqa2', 'd8-b0')]['test_bpb_delta']:.6f}"),
        ("MQA minus B0 test BPB", f"{comp[('d8-gqa4', 'd8-b0')]['test_bpb_delta']:.6f}"),
        ("GQA-2 training throughput gain", pct(comp[("d8-gqa2", "d8-b0")]["training_speedup"])),
        ("MQA training throughput gain", pct(comp[("d8-gqa4", "d8-b0")]["training_speedup"])),
        ("total updates", thousands(sum(r["step"] for r in res.values()))),
        ("total training tokens", thousands(sum(r["tokens"] for r in res.values()))),
        ("per-run FLOP budget C", f"{res['d8-b0-s1']['flop_budget'] / 1e17:.3f}\\times10^{{17}}"),
        ("B0 updates", thousands(res["d8-b0-s1"]["step"])),
        ("GQA-2 updates", thousands(res["d8-gqa2-s1"]["step"])),
        ("MQA updates", thousands(res["d8-gqa4"]["step"])),
    ]
    problems = [f"{label}: computed {text!r} not found in report.tex" for label, text in claims if text not in TEX]
    if any(r["flop_budget"] != res["d8-b0-s1"]["flop_budget"] for r in res.values()):
        problems.append("runs do not share one FLOP budget")
    if any(r["estimated_flops"] > r["flop_budget"] for r in res.values()):
        problems.append("a run exceeded the FLOP budget")
    report("5 numbers", problems, f"{len(claims)} headline numbers recomputed and found in report.tex")


def included_figures():
    return re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", TEX)


def check_figures(tmp):
    out = tmp / "figures"
    proc = run_py("report/make_figures.py", "--out", out)
    if proc.returncode:
        return report("6 figures", [proc.stderr.strip()[-500:]])
    problems = []
    for rel in included_figures():
        fresh, committed = out / Path(rel).name, REPORT / rel
        if not fresh.exists():
            problems.append(f"{rel} is not produced by make_figures.py")
        elif not committed.exists() or fresh.read_bytes() != committed.read_bytes():
            problems.append(f"{rel} differs from a fresh render (rerun report/make_figures.py)")
    report("6 figures", problems, f"{len(included_figures())} included figures regenerate byte for byte")


def check_source():
    problems = [f"missing figure {f}" for f in included_figures() if not (REPORT / f).exists()]
    bib = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", (REPORT / "references.bib").read_text()))
    cited = {k.strip() for group in re.findall(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]+)\}", TEX) for k in group.split(",")}
    problems += [f"citation {k} missing from references.bib" for k in sorted(cited - bib)]
    paths = {p.replace("\\_", "_") for p in re.findall(r"\\(?:texttt|path)\{([^}]+)\}", TEX)}
    paths = {p for p in paths if re.match(r"^((results|experiments|report|nanochat)/|readme\.md$)", p)}
    problems += [f"report names missing path {p}" for p in sorted(paths) if not (ROOT / p).exists()]
    report("7 source", problems, f"{len(included_figures())} figures, {len(cited)} citations, {len(paths)} repository paths")


def main():
    with tempfile.TemporaryDirectory(prefix="verify-repro-") as tmp:
        tmp = Path(tmp)
        check_checksums()
        check_provenance()
        check_configs(tmp)
        check_analysis(tmp)
        check_numbers()
        check_figures(tmp)
        check_source()
    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'All checks passed.'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
