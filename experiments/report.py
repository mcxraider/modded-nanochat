"""Collect source-backed comparisons, seed noise and resolution thresholds; no winner is declared."""
import argparse
import json
import math
from pathlib import Path

from experiments.config import RUNS, members

SIGNIFICANCE_FACTOR = 2 * math.sqrt(2)  # two-sigma on a difference of two single runs


def compare(result, reference, metric="val_bpb"):
    delta = result[metric] - reference[metric]
    return {"reference": reference["run_id"], "bpb_delta": delta,
            "bpb_relative_delta": delta / reference[metric],
            "training_speedup": result["training_tokens_per_second"] / reference["training_tokens_per_second"]}


def matched_observations(rows, ref_rows, axis):
    # Exact shared observations only; never interpolate measured quality into invented results.
    reference = {r[axis]: r["val_bpb"] for r in ref_rows if r.get("kind") == "validation"}
    return [{axis: r[axis], "bpb_delta": r["val_bpb"] - reference[r[axis]]}
            for r in rows if r.get("kind") == "validation" and r[axis] in reference]


def seed_noise(values):
    """Sample standard deviation s (ddof=1) and the resolution threshold 2*sqrt(2)*s for a difference of two runs."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    s = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    return {"n": len(values), "mean": mean, "std": s, "threshold": SIGNIFICANCE_FACTOR * s}


def pooled(results, name):
    """Run, or mean over a baseline group, as a reference record. None if any member has no result yet."""
    runs = [results.get(r) for r in members(name)]
    if any(r is None for r in runs):
        return None
    keys = ("val_bpb", "test_bpb", "training_tokens_per_second", "parameters", "scaling_parameters")
    record = {"run_id": name, "members": [r["run_id"] for r in runs]}
    for key in keys:
        values = [r.get(key) for r in runs]
        record[key] = None if any(v is None for v in values) else sum(values) / len(values)
    return record


def pair_entry(run, reference, threshold):
    entry = {"run_id": run["run_id"], **compare(run, reference), "reference_members": reference["members"]}
    if run.get("test_bpb") is not None and reference.get("test_bpb") is not None:
        delta = run["test_bpb"] - reference["test_bpb"]
        entry.update({"test_bpb_delta": delta, "test_bpb_relative_delta": delta / reference["test_bpb"],
                      "resolved": None if threshold is None else abs(delta) > threshold})
    entry["parameters"] = {"run": run.get("parameters"), "reference": reference.get("parameters")}
    entry["scaling_parameters"] = {"run": run.get("scaling_parameters"), "reference": reference.get("scaling_parameters")}
    return entry


def inference_speedups(rows, ref_rows):
    ref = {(r["batch_size"], r["prompt_tokens"]): r for r in ref_rows if r["status"] == "ok"}
    out = []
    for row in rows:
        key = (row["batch_size"], row["prompt_tokens"])
        if row["status"] == "ok" and key in ref:
            out.append({"batch_size": key[0], "prompt_tokens": key[1],
                        "prefill": row["prefill_tokens_per_second"] / ref[key]["prefill_tokens_per_second"],
                        "decode": row["decode_tokens_per_second_aggregate"] / ref[key]["decode_tokens_per_second_aggregate"],
                        "kv_cache_bytes_ratio": row["kv_cache_bytes_allocated"] / ref[key]["kv_cache_bytes_allocated"]
                        if "kv_cache_bytes_allocated" in row and "kv_cache_bytes_allocated" in ref[key] else None})
    return out


def read(directory, name):
    path = directory / name
    return json.loads(path.read_text()) if path.exists() else None


def check_consistency(loaded):
    """Runs at one depth share data; all runs share code and hardware/backend."""
    first = next(iter(loaded.values()))
    by_depth = {}
    for run_id, item in loaded.items():
        manifest = by_depth.setdefault(RUNS[run_id]["depth"], item["data_manifest"])
        if item["data_manifest"] != manifest:
            raise ValueError("data/tokenizer manifests differ across runs at one depth")
        launch, ref_launch = item["launch"], first["launch"]
        if launch["base_commit"] != ref_launch["base_commit"] or launch["overlay_sha256"] != ref_launch["overlay_sha256"]:
            raise ValueError("runtime code differs across runs")
        for key in ("gpu", "world_size", "torch", "cuda", "precision", "attention_backend", "packages"):
            if item["hardware"][key] != first["hardware"][key]:
                raise ValueError(f"incomparable run hardware/backend: {key}")
        if item["result"].get("stop_reason") not in (None, "token_target"):
            print(f"WARNING: {run_id} ended with stop_reason={item['result']['stop_reason']}; not comparable")
        if item["result"].get("data_epoch", 1) != 1:
            print(f"WARNING: {run_id} repeated training data (epoch {item['result']['data_epoch']})")


def tables(payload):
    lines = ["| Run | Exp | Params (total / scaling) | Tokens | Test BPB | Val BPB | Train tok/s |", "|---|---|---|---|---|---|---|"]
    fmt = lambda v, f: "-" if v is None else format(v, f)
    for r in payload["runs"]:
        lines.append(f"| {r['run_id']} | {RUNS[r['run_id']]['experiment']} | {fmt(r.get('parameters'), ',')} / {fmt(r.get('scaling_parameters'), ',')} | "
                     f"{fmt(r.get('tokens'), ',')} | {fmt(r.get('test_bpb'), '.4f')} | {fmt(r.get('val_bpb'), '.4f')} | "
                     f"{fmt(r.get('training_tokens_per_second'), ',.0f')} |")
    noise = payload["noise"]
    lines += ["", "Seed noise (d8-b0, test BPB): " + ("not yet available (needs >= 2 seeds)" if noise is None else
              f"n={noise['n']}, mean={noise['mean']:.4f}, s={noise['std']:.4f}, resolution threshold 2*sqrt(2)*s = {noise['threshold']:.4f}"),
              "", "| Run | vs | dTest BPB | resolved? | dVal BPB | train speedup |", "|---|---|---|---|---|---|"]
    for c in payload["comparisons"]:
        resolved = {True: "yes", False: "no", None: "-"}[c.get("resolved")]
        lines.append(f"| {c['run_id']} | {c['reference']} | {fmt(c.get('test_bpb_delta'), '+.4f')} | {resolved} | "
                     f"{fmt(c['bpb_delta'], '+.4f')} | {c['training_speedup']:.3f} |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    args = p.parse_args()
    loaded = {}
    for run_id in RUNS:
        directory = args.output_root / run_id
        item = {k: read(directory, f) for k, f in (("result", "result.json"), ("config", "config.json"),
                ("data_manifest", "data_manifest.json"), ("launch", "launch.json"), ("hardware", "hardware.json"),
                ("inference", "inference.json"))}
        if item["result"] is not None:
            if any(item[k] is None for k in ("config", "data_manifest", "launch", "hardware")):
                raise ValueError(f"{run_id} is missing provenance files")
            lines = (directory / "metrics.jsonl").read_text().splitlines()
            item["curves"] = [json.loads(line) for line in lines]
            loaded[run_id] = item
    if not loaded:
        raise SystemExit("no completed runs under " + str(args.output_root))
    from experiments.config import validate_suite
    validate_suite([item["config"] for item in loaded.values()])
    check_consistency(loaded)
    results = {run_id: item["result"] for run_id, item in loaded.items()}
    noise = seed_noise([results[r]["test_bpb"] for r in members("d8-b0") if r in results and results[r].get("test_bpb") is not None])
    threshold = None if noise is None else noise["threshold"]
    comparisons = []
    for run_id, item in loaded.items():
        for ref_name in RUNS[run_id]["compare_to"]:
            reference = pooled(results, ref_name)
            if reference is None or reference["val_bpb"] is None:
                continue
            entry = pair_entry(results[run_id], reference, threshold)
            ref_id = reference["members"][0]
            entry["matched_tokens"] = matched_observations(item["curves"], loaded[ref_id]["curves"], "tokens")
            entry["matched_estimated_flops"] = matched_observations(item["curves"], loaded[ref_id]["curves"], "estimated_flops")
            entry["elapsed_seconds_difference"] = results[run_id]["training_seconds"] - results[ref_id]["training_seconds"]
            entry["inference_speedups"] = []
            if item["inference"] and loaded[ref_id]["inference"]:
                a, b = item["inference"], loaded[ref_id]["inference"]
                for field in ("device", "precision", "sampling", "attention_backend", "torch", "cuda"):
                    if a[field] != b[field]:
                        raise ValueError(f"inference {field} differs across compared runs")
                entry["inference_speedups"] = inference_speedups(a["rows"], b["rows"])
            comparisons.append(entry)
    scale = None
    gap = {d: next((c["test_bpb_delta"] for c in comparisons if c["run_id"] == r and c["reference"] == ref and "test_bpb_delta" in c), None)
           for d, (r, ref) in {8: ("d8-gqa2", "d8-b0"), 12: ("d12-gqa2", "d12-b0")}.items()}
    if None not in gap.values():
        scale = {"gap_d8": gap[8], "gap_d12": gap[12], "difference": gap[12] - gap[8],
                 "caveat": "threshold comes from d8 seeds only; d12 has one seed per configuration"}
    payload = {"runs": list(results.values()), "noise": noise, "comparisons": comparisons, "scale_transfer": scale,
               "missing_runs": [r for r in RUNS if r not in results],
               "caveat": "seed noise from three d8 baseline runs only; unresolved differences are not ties"}
    from experiments.train import write_json
    write_json(args.output_root / "comparison.json", payload)
    (args.output_root / "tables.md").write_text(tables(payload))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
