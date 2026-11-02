"""Render experiment comparisons from recorded artifacts, during or after the sweep.

Usage: python experiments/visualise.py --output-root experiments/artifacts
Requires matplotlib (already in the project's dev dependencies). No GPU required.
"""
import argparse
from collections import deque
import json
import math
from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import re

from experiments.config import RUNS, members, validate_suite
from experiments.report import pooled, seed_noise

_VARIANTS = ("b0", "gqa2", "gqa2-r75", "gqa2-pm", "gqa4", "gqa2-r25")
_COLORS = ("#315A8C", "#B47B15", "#A13D6F", "#2E7D5B", "#7A4EA3", "#C0504D")
_MARKERS = ("o", "s", "D", "^", "v", "P")


def variant(run_id):
    """'d8-b0-s2' -> 'b0'; 'd12-gqa2' -> 'gqa2'."""
    return re.sub(r"-s\d$", "", run_id.split("-", 1)[1])


# Colour/marker identify the variant; line style identifies depth. Seeds of one variant share a style.
STYLES = {name: dict(color=_COLORS[_VARIANTS.index(variant(name))], marker=_MARKERS[_VARIANTS.index(variant(name))],
                     linestyle="-" if RUNS[name]["depth"] == 8 else "--") for name in RUNS}
AXES = (("training_seconds", 60, "Training-phase elapsed time (minutes)"),
        ("tokens", 1e6, "Training tokens (millions)"),
        ("estimated_flops", 1e15, "Estimated training FLOPs (PFLOP)"))
NOTE = "Seed noise is measured on d8 baseline runs only; other configurations have one seed. Lines connect observations, without extrapolation."


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def read_metrics(path):
    """Use the latest resume branch; ignore only an unfinished trailing JSONL write."""
    if not path.is_file():
        return []
    text = path.read_text()
    rows = []
    lines = text.splitlines()
    last_train_step = -1
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1 and not text.endswith("\n"):
                warnings.warn(f"{path}: unfinished final record skipped")
                break
            raise ValueError(f"{path}:{index + 1}: invalid JSON") from exc
        if row.get("kind") not in ("train", "validation", "test"):
            continue
        required = ["step", "training_seconds", "tokens", "estimated_flops"]
        required += (["loss", "step_seconds", "tokens_per_second", "whole_phase_tokens_per_second", "peak_vram_bytes"]
                     if row["kind"] == "train" else ["test_bpb" if row["kind"] == "test" else "val_bpb"])
        for key in required:
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{path}:{index + 1}: {key} must be finite and nonnegative")
        if type(row["step"]) is not int:
            raise ValueError(f"{path}:{index + 1}: step must be an integer")
        step = row["step"]
        if row["kind"] == "train" and step <= last_train_step:
            rows = [r for r in rows if r["step"] < step]
        elif rows and step < rows[-1]["step"]:
            rows = [r for r in rows if r["step"] <= step]
            last_train_step = max((r["step"] for r in rows if r["kind"] == "train"), default=-1)
        if row["kind"] == "train":
            last_train_step = step
        rows = [r for r in rows if (r["kind"], r["step"]) != (row["kind"], step)]
        rows.append(row)
    return rows


def rolling_mean(values, window):
    recent = deque()
    total = 0.0
    output = []
    for value in values:
        recent.append(value)
        total += value
        if len(recent) > window:
            total -= recent.popleft()
        output.append(total / len(recent))
    return output


def load_runs(root):
    runs = {}
    for name in RUNS:
        directory = root / name
        if not directory.is_dir():
            warnings.warn(f"{name}: run directory not available yet")
            continue
        rows = read_metrics(directory / "metrics.jsonl")
        result, inference = (read_json(directory / file) for file in ("result.json", "inference.json"))
        accounting = read_json(directory / "rental_accounting.json") or {}
        attempts = accounting.get("launches", [])
        latest_start = attempts[-1].get("started_unix_seconds", 0) if attempts else 0
        # A resumed run can still contain its previous branch's final artifacts.
        if result is not None and (directory / "result.json").stat().st_mtime < latest_start:
            warnings.warn(f"{name}: previous-attempt final result skipped while awaiting a new result")
            result = None
        if inference is not None and (directory / "inference.json").stat().st_mtime < latest_start:
            warnings.warn(f"{name}: previous-attempt inference result skipped")
            inference = None
        if not rows and result is None and inference is None:
            warnings.warn(f"{name}: no recorded measurements yet")
            continue
        if result is not None:
            for key in ("val_bpb", "training_tokens_per_second", "peak_vram_bytes", "training_seconds", *(("test_bpb",) if "test_bpb" in result else ())):
                value = result.get(key)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"{directory / 'result.json'}: invalid {key}")
        runs[name] = dict(directory=directory, rows=rows, result=result, inference=inference)
    if not runs:
        raise ValueError(f"No experiment measurements found under {root}. Run training first or select --output-root.")
    configs = [read_json(runs[name]["directory"] / "config.json") for name in RUNS if name in runs]
    if all(config is not None for config in configs):
        validate_suite(configs)
    # Reject known confounds; sparse/live artifacts remain usable with an explicit notice.
    checks = {
        "data_manifest.json": None,
        "launch.json": ("base_commit", "overlay_sha256"),
        "hardware.json": ("gpu", "world_size", "torch", "cuda", "precision", "attention_backend", "packages"),
    }
    verified = len(runs) == len(RUNS) and all(c is not None for c in configs)
    for filename, fields in checks.items():
        evidence = {name: read_json(run["directory"] / filename) for name, run in runs.items()}
        verified = verified and all(item is not None for item in evidence.values())
        present = {name: item for name, item in evidence.items() if item is not None}
        if fields is not None and any(any(field not in item for field in fields) for item in present.values()):
            verified = False
        # Data manifests are compared within a depth (token budgets differ); code and hardware across the whole suite.
        groups = {}
        for name, item in present.items():
            key = RUNS[name]["depth"] if fields is None else 0
            groups.setdefault(key, []).append(item if fields is None else {k: item.get(k) for k in fields})
        if any(any(item != group[0] for item in group[1:]) for group in groups.values()):
            raise ValueError(f"Runs differ in {filename}; resolve the confound before comparing them.")
    if not verified:
        warnings.warn("Partial run set or missing provenance: comparability is not fully verified.")
    return runs, verified


def canvas(title, subtitle, nrows=1, ncols=1, sharey=False):
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.8 if ncols == 1 else 5.1 * ncols + 0.8, 5.8 if nrows == 1 else 3.3 * nrows + 1.5),
                             squeeze=False, sharey=sharey)
    fig.suptitle(title, x=0.075, y=0.98, ha="left", fontsize=19, fontweight="bold", color="#192C42")
    fig.text(0.075, 0.91 if nrows == 1 else 0.925, subtitle, fontsize=10, color="#526173")
    fig.text(0.075, 0.025, NOTE, fontsize=9, color="#526173")
    fig.subplots_adjust(left=0.12 if ncols == 1 else 0.085, right=0.96,
                        top=0.88 if nrows >= 3 else 0.83, bottom=0.19 if nrows == 1 else 0.14, hspace=0.48, wspace=0.30)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E2E7ED", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(colors="#526173", labelsize=9)
        ax.yaxis.set_major_locator(MaxNLocator(5))
    return fig, axes


def series(ax, rows, xkey, ykey, name, xscale=1, yscale=1, window=1):
    if not rows:
        return
    x = [r[xkey] / xscale for r in rows]
    y = [r[ykey] / yscale for r in rows]
    style = STYLES[name]
    if window > 1:
        ax.plot(x, y, color=style["color"], alpha=0.18, linewidth=0.8)
    ax.plot(x, rolling_mean(y, window), label=name, color=style["color"], linestyle=style["linestyle"],
            marker=style["marker"], markersize=4, markevery=max(1, len(x) // 9), linewidth=1.8)


def learning_curves(runs, context, window):
    fig, axes = canvas("Learning curves", f"{context} · loss: worker 0; validation BPB: all workers", 2, 3, sharey="row")
    for col, (xkey, scale, xlabel) in enumerate(AXES):
        for name, run in runs.items():
            series(axes[0, col], [r for r in run["rows"] if r["kind"] == "train"], xkey, "loss", name, scale, window=window)
            series(axes[1, col], [r for r in run["rows"] if r["kind"] == "validation"], xkey, "val_bpb", name, scale)
        for row in range(2):
            axes[row, col].set_xlabel(xlabel)
        axes[0, col].set_title(f"Training loss · trailing {window}-step mean", fontsize=11)
        axes[1, col].set_title("Validation BPB · observed evaluations", fontsize=11)
    axes[0, 0].set_ylabel("Cross-entropy (nats/token)")
    axes[1, 0].set_ylabel("Bits per byte · lower is better")
    for row in range(2):
        if axes[row, 0].lines:
            axes[row, 0].legend(frameon=False, fontsize=9)
    return fig


def training_performance(runs, context, window):
    fig, axes = canvas("Training performance", f"{context} · faint lines: raw steps; bold lines: trailing means", 2, 2)
    panels = (("tokens_per_second", 1, "Optimizer-step throughput", "Combined tokens/second", window),
              ("whole_phase_tokens_per_second", 1, "Whole-phase throughput, including overhead", "Combined tokens/second", 1),
              ("step_seconds", 1, "Optimizer-step duration, including data loading", "Seconds per step", window),
              ("peak_vram_bytes", 2**30, "Peak allocated VRAM · maximum across GPUs", "GiB per GPU · cumulative peak", 1))
    for ax, (key, scale, title, ylabel, smooth) in zip(axes.flat, panels):
        for name, run in runs.items():
            series(ax, [r for r in run["rows"] if r["kind"] == "train"], "training_seconds", key, name, 60, scale, smooth)
        ax.set(title=title, xlabel="Training-phase elapsed time (minutes)", ylabel=ylabel)
        ax.set_ylim(0, max(ax.get_ylim()[1] * 1.08, 1))
    if axes[0, 0].lines:
        axes[0, 0].legend(frameon=False, fontsize=9)
    return fig


def quality_of(result, metric):
    return result[metric]


def baseline_noise(runs):
    """Seed noise from the d8 baseline group, if enough seeds finished (test BPB)."""
    values = [runs[r]["result"]["test_bpb"] for r in members("d8-b0")
              if r in runs and runs[r]["result"] is not None and runs[r]["result"].get("test_bpb") is not None]
    return seed_noise(values)


def pair_deltas(runs, metric):
    results = {name: run["result"] for name, run in runs.items() if run["result"] is not None}
    out = []
    for name, result in results.items():
        for ref in RUNS[name]["compare_to"]:
            reference = pooled(results, ref)
            if reference is not None and reference.get(metric) is not None and result.get(metric) is not None:
                out.append((name, ref, result[metric] - reference[metric]))
    return out


def final_comparison(runs, context, noise=None):
    completed = [(name, run["result"]) for name, run in runs.items() if run["result"] is not None]
    if not completed:
        return None
    metric = "test_bpb" if all(r.get("test_bpb") is not None for _, r in completed) else "val_bpb"
    label = "test" if metric == "test_bpb" else "validation"
    fig, axes = canvas("Final quality and training efficiency", f"{context} · exact final {label} BPB; completed runs only", 2, 2)
    names, results = zip(*completed)
    positions = list(range(len(names)))
    colors = [STYLES[name]["color"] for name in names]
    for i, (name, result) in enumerate(completed):
        axes[0, 0].scatter(result[metric], i, color=colors[i], marker=STYLES[name]["marker"], s=65, zorder=3)
        axes[0, 0].annotate(f" {result[metric]:.5f}", (result[metric], i), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9)
    axes[0, 0].set(yticks=positions, yticklabels=names, xlabel=f"Final {label} BPB · lower is better", title="Quality after the fixed token budget")
    axes[0, 0].margins(x=0.45, y=0.35)
    pairs = pair_deltas(runs, metric)
    if pairs:
        deltas = [d for _, _, d in pairs]
        bars = axes[0, 1].barh(range(len(pairs)), deltas, color=[STYLES[name]["color"] for name, _, _ in pairs], height=0.5)
        axes[0, 1].axvline(0, color="#526173", linewidth=0.8)
        if noise is not None and metric == "test_bpb":
            for sign in (-1, 1):
                axes[0, 1].axvline(sign * noise["threshold"], color="#526173", linewidth=0.8, linestyle=":")
        axes[0, 1].bar_label(bars, labels=[f"{v:+.5f}" for v in deltas], padding=5, fontsize=9)
        axes[0, 1].set(yticks=list(range(len(pairs))), yticklabels=[f"{name}\n− {ref}" for name, ref, _ in pairs],
                       xlabel="BPB difference · negative is better" + (" · dotted: ±2√2·s seed-noise threshold" if noise is not None and metric == "test_bpb" else ""),
                       title="Pairwise hypothesis comparisons")
        axes[0, 1].margins(x=0.35)
    else:
        axes[0, 1].text(0.5, 0.5, "Waiting for comparable final results", transform=axes[0, 1].transAxes, ha="center", fontsize=9)
        axes[0, 1].set_axis_off()
    for ax, key, scale, title, label in (
        (axes[1, 0], "training_tokens_per_second", 1, "Whole-phase training throughput", "Combined tokens/second · higher is better"),
        (axes[1, 1], "peak_vram_bytes", 2**30, "Peak allocated VRAM", "GiB per GPU · lower is better"),
    ):
        values = [r[key] / scale for r in results]
        bars = ax.barh(positions, values, color=colors, height=0.5)
        ax.bar_label(bars, labels=[f"{v:,.1f}" for v in values], padding=5, fontsize=9)
        ax.set(yticks=positions, yticklabels=names, title=title, xlabel=label, xlim=(0, max(values, default=0) * 1.28 or 1))
    return fig


def final_value(runs, name, metric):
    run = runs.get(name)
    result = run["result"] if run else None
    return None if result is None else result.get(metric)


def dose_response(runs, context):
    """Final BPB against KV heads (full RoPE) and against rotary fraction (2 KV heads), d8 only."""
    metric = "test_bpb" if all(final_value(runs, r, "test_bpb") is not None for r in RUNS if final_value(runs, r, "val_bpb") is not None) else "val_bpb"
    results = {n: r["result"] for n, r in runs.items() if r["result"] is not None}
    base = pooled(results, "d8-b0")
    kv = [(4, base[metric])] if base and base.get(metric) is not None else []
    kv += [(4 // RUNS[n]["gqa"], results[n][metric]) for n in ("d8-gqa2", "d8-gqa4") if n in results]
    rope = [(RUNS[n]["rotary"], results[n][metric]) for n in ("d8-gqa2", "d8-gqa2-r75", "d8-gqa2-r25") if n in results]
    if len(kv) < 2 and len(rope) < 2:
        return None
    fig, axes = canvas("Dose-response", f"{context} · d8 · {metric.replace('_bpb', '')} BPB; single seeds except the baseline mean", 1, 2, sharey=True)
    noise = baseline_noise(runs)
    for ax, points, xlabel, title in ((axes[0, 0], kv, "KV heads (4 = MHA, 1 = MQA)", "KV sharing · full RoPE"),
                                      (axes[0, 1], rope, "Rotary fraction of each head", "Partial RoPE · 2 KV heads")):
        if points:
            xs, ys = zip(*sorted(points))
            ax.plot(xs, ys, color="#315A8C", marker="o", linewidth=1.8)
            if noise is not None and metric == "test_bpb":
                ax.fill_between(xs, [y - noise["threshold"] / 2 for y in ys], [y + noise["threshold"] / 2 for y in ys], color="#315A8C", alpha=0.12)
        ax.set(xlabel=xlabel, title=title, xticks=sorted({x for x, _ in points}))
    axes[0, 0].set_ylabel("Final BPB · lower is better")
    return fig


def scale_transfer(runs, context):
    """GQA-2 minus MHA final BPB at d8 and d12."""
    results = {n: r["result"] for n, r in runs.items() if r["result"] is not None}
    gaps = []
    for depth, (run, ref) in ((8, ("d8-gqa2", "d8-b0")), (12, ("d12-gqa2", "d12-b0"))):
        reference = pooled(results, ref)
        for metric in ("test_bpb", "val_bpb"):
            if run in results and reference and results[run].get(metric) is not None and reference.get(metric) is not None:
                gaps.append((depth, metric, results[run][metric] - reference[metric]))
                break
    if len(gaps) < 2 or len({m for _, m, _ in gaps}) != 1:
        return None
    fig, axes = canvas("Scale transfer", f"{context} · GQA-2 minus MHA, {gaps[0][1].replace('_bpb', '')} BPB; d12 has one seed", 1, 1)
    ax = axes[0, 0]
    bars = ax.bar([f"d{d}" for d, _, _ in gaps], [g for _, _, g in gaps], color=["#315A8C", "#B47B15"], width=0.5)
    ax.bar_label(bars, labels=[f"{g:+.5f}" for _, _, g in gaps], padding=4, fontsize=9)
    noise = baseline_noise(runs)
    if noise is not None and gaps[0][1] == "test_bpb":
        for sign in (-1, 1):
            ax.axhline(sign * noise["threshold"], color="#526173", linewidth=0.8, linestyle=":")
    ax.axhline(0, color="#526173", linewidth=0.8)
    ax.set(ylabel="BPB difference · negative is better", title="Gap persists with size?")
    return fig


def quality_speed(runs, context):
    completed = [(name, run["result"]) for name, run in runs.items() if run["result"] is not None]
    if not completed:
        return None
    metric = "test_bpb" if all(r.get("test_bpb") is not None for _, r in completed) else "val_bpb"
    fig, axes = canvas("Quality / speed tradeoff", context)
    ax = axes[0, 0]
    for name, result in completed:
        x, y = result["training_tokens_per_second"], result[metric]
        ax.scatter(x, y, color=STYLES[name]["color"], marker=STYLES[name]["marker"], s=80, label=name, zorder=3)
    ax.set(xlabel="Whole-phase combined training tokens/second → faster", ylabel=f"Final {metric.replace('_bpb', '')} BPB · lower is better")
    ax.margins(x=0.2, y=0.25)
    ax.legend(frameon=False, fontsize=9)
    return fig


def inference_comparison(runs, context):
    available = [(name, run["inference"]) for name, run in runs.items() if run["inference"] is not None]
    if not available:
        return None
    for field in ("device", "precision", "sampling", "attention_backend", "torch", "cuda"):
        if any(item.get(field) != available[0][1].get(field) for _, item in available[1:]):
            raise ValueError(f"Inference {field} differs across runs")
    prompts = sorted({r["prompt_tokens"] for _, item in available for r in item["rows"]})
    if not prompts:
        return None
    panels = [("prefill_tokens_per_second", 1, "Prefill tokens/second ↑"),
              ("decode_tokens_per_second_aggregate", 1, "Aggregate decode tokens/second ↑"),
              ("time_to_first_token_seconds", 1000, "Time to first token (ms) ↓"),
              ("peak_vram_bytes", 1 / 2**30, "Peak allocated VRAM (GiB) ↓")]
    if any("kv_cache_bytes_allocated" in r for _, item in available for r in item["rows"]):
        panels.append(("kv_cache_bytes_allocated", 1 / 2**30, "Allocated KV cache (GiB) ↓"))
    fig, axes = canvas("Single-GPU inference", f"{context} · fixed workloads; gaps indicate OOM or missing measurements", len(panels), len(prompts), sharey="row")
    for col, prompt in enumerate(prompts):
        for row_index, (metric, scale, ylabel) in enumerate(panels):
            ax = axes[row_index, col]
            batches = sorted({r["batch_size"] for _, item in available for r in item["rows"] if r["prompt_tokens"] == prompt})
            for name, item in available:
                records = {r["batch_size"]: r for r in item["rows"] if r["prompt_tokens"] == prompt}
                values = []
                for batch in batches:
                    record = records.get(batch, {})
                    value = record.get(metric)
                    if record.get("status") == "ok" and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                        raise ValueError(f"{name}: invalid inference {metric}")
                    values.append(value * scale if record.get("status") == "ok" else math.nan)
                ax.plot(batches, values, label=name, **STYLES[name], linewidth=1.8, markersize=5)
            ax.set(xlabel="Batch size", ylabel=ylabel, xticks=batches)
            ax.set_ylim(bottom=0)
            if row_index == 0:
                ax.set_title(f"{prompt:,}-token prompt", fontsize=12)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.text(0.075, 0.05, "TTFT includes the first token; decode times the remaining tokens. Context overruns are speed-only tests.", fontsize=9, color="#526173")
    oom = [f"{name}: batch {r['batch_size']}, prompt {r['prompt_tokens']}"
           for name, item in available for r in item["rows"] if r["status"] == "oom"]
    if oom:
        warnings.warn("Inference OOM workloads: " + "; ".join(oom))
    return fig


def render(output_root, output_dir=None, smooth_window=10):
    if smooth_window < 1:
        raise ValueError("smooth_window must be positive")
    root = Path(output_root).resolve()
    runs, verified = load_runs(root)
    output = Path(output_dir).resolve() if output_dir is not None else root / "figures"
    context = "Full suite" if verified else "PARTIAL / provenance not fully verified"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                         "axes.labelcolor": "#25364A", "axes.edgecolor": "#A7B3C0", "savefig.facecolor": "white"})
    noise = baseline_noise(runs)
    builders = {}
    for depth in sorted({RUNS[name]["depth"] for name in runs}):
        group = {name: run for name, run in runs.items() if RUNS[name]["depth"] == depth}
        tag, ctx = f"_d{depth}", f"{context} · d{depth}"
        builders.update({
            "learning_curves" + tag: lambda g=group, c=ctx: learning_curves(g, c, smooth_window),
            "training_performance" + tag: lambda g=group, c=ctx: training_performance(g, c, smooth_window),
            "final_comparison" + tag: lambda g=group, c=ctx: final_comparison(g, c, noise),
            "quality_speed" + tag: lambda g=group, c=ctx: quality_speed(g, c),
            "inference_comparison" + tag: lambda g=group, c=ctx: inference_comparison(g, c),
        })
    builders["dose_response"] = lambda: dose_response(runs, context)
    builders["scale_transfer"] = lambda: scale_transfer(runs, context)
    written = []
    for name, build in builders.items():
        if name.startswith(("learning_curves", "training_performance")):
            depth = int(name.rsplit("_d", 1)[1])
            if not any(run["rows"] for n, run in runs.items() if RUNS[n]["depth"] == depth):
                continue
        fig = build()
        if fig is None:
            continue
        try:
            output.mkdir(parents=True, exist_ok=True)
            for extension in ("png", "svg"):
                target = output / f"{name}.{extension}"
                fig.savefig(target, dpi=170)
                written.append(target)
        finally:
            plt.close(fig)
    if not written:
        raise ValueError("No plottable measurements yet")
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=ROOT / "experiments/artifacts")
    p.add_argument("--output-dir", type=Path, help="defaults to <output-root>/figures")
    p.add_argument("--smooth-window", type=int, default=10, help="trailing training steps; raw lines remain visible")
    args = p.parse_args()
    try:
        for path in render(args.output_root, args.output_dir, args.smooth_window):
            print(path)
    except (ValueError, OSError) as exc:
        p.exit(1, f"visualise: {exc}\n")


if __name__ == "__main__":
    main()
