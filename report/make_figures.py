"""Regenerate report figures from the exported d8-v5 evidence; no GPU needed."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "results/d8-v5/artifacts"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
COLORS = ["#245c8c", "#d17627", "#398362", "#8d5898", "#b94c57"]


def result(run):
    return json.loads((ROOT / run / "result.json").read_text())


def save(fig, name):
    fig.savefig(OUT / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


# Keep legends outside the axes, and compare schedules within the same horizon.
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
groups = [
    [("d8-b0-s1", "B0 s1"), ("d8-b0-s2", "B0 s2"),
     ("d8-b0-s3", "B0 s3"), ("d8-gqa2-pm", "GQA-2-PM"),
     ("d8-b0-nove", "B0-noVE")],
    [("d8-gqa2-s1", "GQA-2 s1"), ("d8-gqa2-s2", "GQA-2 s2"),
     ("d8-gqa2-s3", "GQA-2 s3"), ("d8-gqa2-r75", "R75"),
     ("d8-gqa2-r25", "R25")]]
for ax, group, horizon in zip(axes, groups, [1680, 1750]):
    for (run, label), color in zip(group, COLORS):
        rows = [json.loads(line) for line in (ROOT / run / "metrics.jsonl").read_text().splitlines()]
        rows = [r for r in rows if r["kind"] == "validation"]
        end = result(run)
        points = {r["step"]: r["val_bpb"] for r in rows}
        points[end["step"]] = end["val_bpb"]
        x = sorted(points)
        ax.plot(x, [points[i] for i in x], ".-", label=label, color=color, lw=1.3)
    ax.set(title=f"{horizon:,}-update schedule", xlabel="Update", ylabel="Validation BPB (lower is better)")
    ax.set_ylim(.95, 1.24)
    ax.grid(alpha=.18)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.19), ncol=3, frameon=False, fontsize=8)
save(fig, "learning_curves_d8")

# Matched-seed E6 arrows avoid confusing a single-seed gain with a group mean.
fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), layout="constrained")
c = json.loads((ROOT / "comparison.json").read_text())
base, gqa = c["noise"]["mean"], c["gqa2_noise"]["mean"]
axes[0].plot([4, 2, 1], [base, gqa, result("d8-gqa4")["test_bpb"]], "o-", color=COLORS[0])
axes[0].set(xlabel="K/V heads", xticks=[1, 2, 4], ylabel="Test BPB", title="K/V sharing at fixed compute")
axes[1].plot([.25, .75, 1], [result("d8-gqa2-r25")["test_bpb"], result("d8-gqa2-r75")["test_bpb"], gqa], "o-", color=COLORS[1])
t = c["noise"]["threshold"]
axes[1].axhspan(gqa-t, gqa+t, color=COLORS[1], alpha=.12)
axes[1].set(xlabel="Rotary fraction", xticks=[.25, .75, 1], ylabel="Test BPB", title="Partial RoPE (GQA-2)")
for run, tok, label, color in [("d8-gqa2-s1", "d8-gqa2-tok", "GQA-2 s1", COLORS[1]), ("d8-gqa4", "d8-gqa4-tok", "MQA s1", COLORS[2])]:
    a, b = result(tok), result(run)
    x = [a["estimated_flops"]/1e17, b["estimated_flops"]/1e17]
    y = [a["test_bpb"], b["test_bpb"]]
    axes[2].plot(x, y, "o-", label=label, color=color)
    axes[2].annotate("", xy=(x[1], y[1]), xytext=(x[0], y[0]), arrowprops={"arrowstyle": "->", "color": color})
axes[2].axvline(result("d8-b0-s1")["flop_budget"]/1e17, color="gray", ls=":")
axes[2].set(xlabel="Estimated training FLOPs ($10^{17}$)", ylabel="Test BPB", title="Extra tokens and longer schedule")
axes[2].legend(frameon=False, fontsize=8)
for ax in axes:
    ax.grid(alpha=.18)
    ax.ticklabel_format(axis="y", useOffset=False)
save(fig, "dose_budget")

# Each prompt column gets independent y limits, preventing the original clipping.
fig, axes = plt.subplots(4, 3, figsize=(10.5, 9), layout="constrained")
metrics = [("decode_tokens_per_second_aggregate", "Aggregate decode (tok/s)", 1),
           ("time_to_first_token_seconds", "TTFT (ms)", 1000),
           ("peak_vram_bytes", "Peak allocation (GiB)", 1/2**30),
           ("kv_cache_bytes_allocated", "KV cache (GiB)", 1/2**30)]
for run, label, color in [("d8-b0-s1", "B0 s1", COLORS[0]), ("d8-gqa2-s1", "GQA-2 s1", COLORS[1]), ("d8-gqa4", "MQA s1", COLORS[2])]:
    rows = json.loads((ROOT / run / "inference.json").read_text())["rows"]
    for j, prompt in enumerate([128, 1792, 8192]):
        cells = sorted([r for r in rows if r["prompt_tokens"] == prompt], key=lambda r:r["batch_size"])
        for i, (metric, ylabel, scale) in enumerate(metrics):
            ax = axes[i, j]
            ax.plot([r["batch_size"] for r in cells], [r[metric]*scale for r in cells], "o-", label=label, color=color, ms=3)
            ax.set_xscale("log", base=4)
            ax.set_xticks([1, 4, 16, 64], ["1", "4", "16", "64"])
            ax.grid(alpha=.18)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 0:
                ax.set_title(f"Prompt {prompt:,}" + (" (beyond training)" if prompt == 8192 else ""), fontsize=10)
            if i == 3:
                ax.set_xlabel("Batch size")
axes[0, 1].legend(loc="upper left", frameon=False, fontsize=8)
save(fig, "inference_comparison_d8")

# Reuse the audited quality and cost plots without modifying original evidence.
import shutil
for name in ["quality_summary", "cost_reconciliation"]:
    shutil.copyfile(HERE.parent / "results/d8-v5/audit" / (name + ".png"), OUT / (name + ".png"))
