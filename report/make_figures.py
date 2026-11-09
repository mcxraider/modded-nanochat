"""Regenerate report figures from the exported d8-v5 evidence; no GPU needed.

Usage (from the repository root): uv run --frozen python report/make_figures.py [--out DIR]
Output is byte-reproducible (PDF creation date stripped), so report/verify_reproduction.py
can compare a fresh render against report/figures/.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "results/d8-v5/artifacts"
_parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_parser.add_argument("--out", type=Path, default=HERE / "figures", help="output directory (default: report/figures)")
OUT = _parser.parse_args().out
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
COLORS = ["#245c8c", "#d17627", "#398362", "#8d5898", "#b94c57"]


def result(run):
    return json.loads((ROOT / run / "result.json").read_text())


def save(fig, name):
    fig.savefig(OUT / (name + ".pdf"), bbox_inches="tight", metadata={"CreationDate": None})
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

NAMES = {"d8-b0": "B0", "d8-gqa2": "GQA-2", "d8-gqa2-r75": "GQA-2-R75", "d8-gqa2-r25": "GQA-2-R25",
         "d8-gqa2-pm": "GQA-2-PM", "d8-gqa4": "MQA", "d8-b0-nove": "B0-noVE", "d8-gqa2-nove": "GQA-2-noVE",
         "d8-gqa2-tok": "GQA-2-tok", "d8-gqa4-tok": "MQA-tok"}
GROUPS = [("d8-b0", ["d8-b0-s1", "d8-b0-s2", "d8-b0-s3"]), ("d8-gqa2", ["d8-gqa2-s1", "d8-gqa2-s2", "d8-gqa2-s3"]),
          ("d8-gqa2-r75", ["d8-gqa2-r75"]), ("d8-gqa2-r25", ["d8-gqa2-r25"]), ("d8-gqa2-pm", ["d8-gqa2-pm"]),
          ("d8-gqa4", ["d8-gqa4"]), ("d8-b0-nove", ["d8-b0-nove"]), ("d8-gqa2-nove", ["d8-gqa2-nove"]),
          ("d8-gqa2-tok", ["d8-gqa2-tok"]), ("d8-gqa4-tok", ["d8-gqa4-tok"])]


def mean(xs):
    return sum(xs) / len(xs)


# Quality summary: per-run scores (left) and every registered contrast with its pooled-t interval (right).
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained", gridspec_kw={"width_ratios": [1, 1.25]})
ax = axes[0]
for i, (group, runs) in enumerate(GROUPS):
    vals = [result(r)["test_bpb"] for r in runs]
    color = COLORS[0] if len(runs) > 1 else COLORS[1]
    ax.scatter(vals, [i] * len(vals), color="#9db4ce", s=22, zorder=3)
    ax.scatter([mean(vals)], [i], marker="D", color=color, s=40, zorder=4)
    ax.text(.9835, i, f"{mean(vals):.4f}", va="center", fontsize=8)
ax.axvline(base, color="gray", ls=":", lw=1)
ax.axvspan(base - t, base + t, color="gray", alpha=.10, lw=0)
ax.set(yticks=range(len(GROUPS)), yticklabels=[NAMES[g] for g, _ in GROUPS], xlim=(.957, .9885),
       xlabel="Final test BPB (lower is better)", title="Test BPB per run (diamond = mean)")
ax.invert_yaxis()
ax.grid(axis="x", alpha=.18)
ax = axes[1]
comps = c["comparisons"]
for i, comp in enumerate(comps):
    lo, hi = comp["sensitivity"]["ci95"]
    d = comp["test_bpb_delta"]
    color = COLORS[4] if comp["sensitivity"]["resolved"] else ("#d6a35c" if comp["resolved"] else "#888888")
    ax.plot([lo, hi], [i, i], color=color, lw=2, solid_capstyle="butt")
    ax.plot([d], [i], "o", color=color, ms=5)
ax.axvline(0, color="black", lw=.8)
ax.axvspan(-t, t, color="gray", alpha=.12, lw=0)
ax.set(yticks=range(len(comps)),
       yticklabels=[f"{NAMES[x['run_id']]} $-$ {NAMES[x['reference']]}" for x in comps],
       xlabel=r"$\Delta$ test BPB (positive = worse than reference)", title="Contrasts with pooled-$t$ 95% intervals")
ax.tick_params(axis="y", labelsize=8)
ax.invert_yaxis()
ax.grid(axis="x", alpha=.18)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color=COLORS[4], marker="o", label="resolved (primary + pooled)"),
                   Line2D([], [], color="#d6a35c", marker="o", label="primary rule only"),
                   Line2D([], [], color="#888888", marker="o", label="unresolved")],
          loc="upper center", bbox_to_anchor=(.4, -.13), ncol=3, frameon=False, fontsize=7.5)
save(fig, "quality_summary")

# Paired E2 seeds (left) and the E5 value-embedding 2x2 interaction (right).
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
ax = axes[0]
for k, color in zip([1, 2, 3], COLORS):
    y = [result(f"d8-b0-s{k}")["test_bpb"], result(f"d8-gqa2-s{k}")["test_bpb"]]
    ax.plot([0, 1], y, "o-", color=color, label=f"seed {k} ($\\Delta$={y[1]-y[0]:+.4f})")
ax.set(xticks=[0, 1], xticklabels=["B0 (MHA)", "GQA-2"], xlim=(-.3, 1.3), ylabel="Test BPB",
       title="E2: same-seed pairs share data order")
ax.legend(frameon=False, fontsize=8)
ax = axes[1]
ve_on = [base, gqa]
ve_off = [result("d8-b0-nove")["test_bpb"], result("d8-gqa2-nove")["test_bpb"]]
for y, label, color in [(ve_on, "value embeddings on", COLORS[0]), (ve_off, "value embeddings off", COLORS[3])]:
    ax.plot([4, 2], y, "o-", color=color, label=f"{label} (gap {y[1]-y[0]:+.4f})")
inter = c["interactions"]["E5"]
ax.set(xticks=[4, 2], xticklabels=["4 K/V heads", "2 K/V heads"], xlim=(4.4, 1.6), ylabel="Test BPB",
       title=f"E5: interaction {inter['interaction']:+.4f} (unresolved)")
ax.legend(frameon=False, fontsize=8, loc="center left")
for ax in axes:
    ax.grid(alpha=.18)
    ax.ticklabel_format(axis="y", useOffset=False)
save(fig, "paired_interaction")

# Deployment trade-off: quality against training throughput and KV-cache size (fixed-compute runs only).
OFFSETS = {"d8-gqa2": (5, -8), "d8-gqa2-r75": (-5, -8), "d8-gqa2-r25": (-5, 3), "d8-gqa2-pm": (5, 3)}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
for group, runs in GROUPS[:8]:
    rs = [result(r) for r in runs]
    bpb = mean([r["test_bpb"] for r in rs])
    tps = mean([r["steady_tokens_per_second"] for r in rs]) / 1e6
    cfg = rs[0]["model_config"]
    kv = 2 * cfg["n_layer"] * cfg["n_kv_head"] * (cfg["n_embd"] // cfg["n_head"]) * 2 / 1024
    color = {4: COLORS[0], 2: COLORS[1], 1: COLORS[2]}[cfg["n_kv_head"]]
    marker = "s" if not cfg["value_embeds"] else "o"
    for ax, x in zip(axes, [tps, kv]):
        ax.scatter([x], [bpb], color=color, marker=marker, s=36, zorder=3)
        kv_panel = ax is axes[1]
        if kv_panel and group in ("d8-gqa2-r75", "d8-gqa2-r25", "d8-gqa2-pm"):
            continue  # same cache size as GQA-2; labelled jointly below
        label = "GQA-2 (and -R75, -R25, -PM)" if kv_panel and group == "d8-gqa2" else NAMES[group]
        dx, dy = (6, -3) if kv_panel else OFFSETS.get(group, (4, 3))
        ax.annotate(label, (x, bpb), xytext=(dx, dy), textcoords="offset points", fontsize=7.5,
                    ha="right" if dx < 0 else "left")
axes[0].set(xlabel="Steady training throughput (M tokens/s)", ylabel="Test BPB (lower is better)",
            title="Quality vs training speed")
axes[1].set(xlabel="KV cache (KiB per token, BF16, all layers)", ylabel="Test BPB (lower is better)",
            title="Quality vs inference cache size", xticks=[4, 8, 16])
axes[1].set_xlim(2, 19.5)
axes[0].set_xlim(1.165, 1.275)
fig.legend(handles=[Line2D([], [], color=COLORS[0], marker="o", ls="", label="4 K/V heads"),
                        Line2D([], [], color=COLORS[1], marker="o", ls="", label="2 K/V heads"),
                        Line2D([], [], color=COLORS[2], marker="o", ls="", label="1 K/V head"),
                        Line2D([], [], color="gray", marker="s", ls="", label="no value embeddings")],
               frameon=False, fontsize=8, loc="outside lower center", ncol=4)
for ax in axes:
    ax.grid(alpha=.18)
    ax.ticklabel_format(axis="y", useOffset=False)
save(fig, "tradeoff")

# Inference speed relative to B0 seed 1 (prefill and decode) across the full benchmark grid.
fig, axes = plt.subplots(2, 3, figsize=(10.5, 5), layout="constrained", sharey="row")
for comp in comps:
    if comp["reference"] != "d8-b0" or comp["run_id"] not in ("d8-gqa2", "d8-gqa4"):
        continue
    color = COLORS[1] if comp["run_id"] == "d8-gqa2" else COLORS[2]
    for j, prompt in enumerate([128, 1792, 8192]):
        cells = sorted([s for s in comp["inference_speedups"] if s["prompt_tokens"] == prompt],
                       key=lambda s: s["batch_size"])
        for i, metric in enumerate(["prefill", "decode"]):
            axes[i, j].plot([s["batch_size"] for s in cells], [s[metric] for s in cells], "o-", ms=3,
                            color=color, label=f"{NAMES[comp['run_id']]} s1 / B0 s1")
for i, metric in enumerate(["Prefill", "Decode"]):
    for j, prompt in enumerate([128, 1792, 8192]):
        ax = axes[i, j]
        ax.axhline(1, color="gray", ls=":")
        ax.set_xscale("log", base=4)
        ax.set_xticks([1, 4, 16, 64], ["1", "4", "16", "64"])
        ax.grid(alpha=.18)
        if i == 0:
            ax.set_title(f"Prompt {prompt:,}" + (" (beyond training)" if prompt == 8192 else ""), fontsize=10)
        if i == 1:
            ax.set_xlabel("Batch size")
    axes[i, 0].set_ylabel(f"{metric} speed vs B0")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=8)
save(fig, "speed_ratios")

# Cost reconciliation: pilot projection against the measured run pipelines.
audit = json.loads((HERE.parent / "results/d8-v5/audit/audit.json").read_text())["totals"]
rate = 4.43 / 3600
labels = ["Training incl. interim eval", "Final validation + test", "Inference benchmark", "Other overhead"]
rows = [[audit["projected_training_seconds"], audit["projected_final_evaluation_seconds"],
         audit["projected_benchmark_seconds"], audit["projected_overhead_seconds"]],
        [audit["training_seconds"], audit["final_evaluation_seconds"], audit["benchmark_seconds"],
         audit["actual_other_seconds"]]]
fig, ax = plt.subplots(figsize=(9, 2.6), layout="constrained")
for y, vals in enumerate(rows):
    left = 0
    for label, value, color in zip(labels, vals, [COLORS[0], "#739dce", "#bc8c36", "#cccccc"]):
        cost = value * rate
        ax.barh(y, cost, left=left, color=color, label=label if y == 0 else None)
        if cost > .6:
            ax.text(left + cost / 2, y, f"${cost:.2f}", ha="center", va="center", fontsize=8,
                    color="#222" if color == "#cccccc" else "white")
        left += cost
    ax.text(left + .15, y, f"${left:.2f}", va="center", weight="bold", fontsize=9)
ax.set(yticks=[0, 1], yticklabels=["Pilot projection", "Measured (14 runs)"], xlim=(0, 21.5),
       xlabel="USD at \\$4.43 per H100-hour")
ax.invert_yaxis()
ax.legend(loc="upper center", bbox_to_anchor=(.5, -.35), ncol=4, frameon=False, fontsize=8)
save(fig, "cost_reconciliation")
