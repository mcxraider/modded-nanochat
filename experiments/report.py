"""Collect source-backed comparisons, seed noise and resolution thresholds; no winner is declared."""
import argparse
import json
import math
from pathlib import Path

from experiments.config import GROUPS, INTERACTIONS, RUNS, members, model_for

SIGNIFICANCE_FACTOR = 2 * math.sqrt(2)  # two-sigma on a difference of two single runs
T95 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228)  # two-sided 95% Student t, df 1..10
POOLED_GROUPS = ("d8-b0", "d8-gqa2")


def comparison_eligibility(result, config=None):
    """Final comparisons require the configured token target, within the declared FLOP budget, without data repetition."""
    if result is None:
        return {"eligible": False, "reasons": ["no final result"]}
    reasons = []
    if result.get("stop_reason") != "token_target":
        reasons.append(f"stop_reason={result.get('stop_reason', 'missing')}")
    target = config["training"]["target_tokens"] if config else result.get("target_tokens")
    if type(target) is not int or target <= 0:
        reasons.append("missing or invalid target_tokens")
    elif result.get("target_tokens") != target or result.get("tokens") != target:
        reasons.append("tokens/target_tokens do not match the configured token target")
    budget = config["training"]["flop_budget"] if config else result.get("flop_budget")
    if type(budget) is not int or type(result.get("estimated_flops")) not in (int, float):
        reasons.append("missing flop_budget or estimated_flops")
    elif result["estimated_flops"] > budget:
        reasons.append("exceeds declared FLOP budget")
    if result.get("data_epoch") != 1:
        reasons.append("training data was repeated or data_epoch is missing")
    return {"eligible": not reasons, "reasons": reasons}


def ratio(a, b):
    return None if a is None or not b else a / b


def compare(result, reference, metric="val_bpb"):
    """`training_speedup` uses steady-state step throughput (compile warm-up excluded); the whole-phase ratio,
    which also counts compile, evaluation and checkpoint I/O, is kept as `whole_phase_training_speedup`."""
    delta = result[metric] - reference[metric]
    return {"reference": reference["run_id"], "bpb_delta": delta,
            "bpb_relative_delta": delta / reference[metric],
            "training_speedup": ratio(result.get("steady_tokens_per_second"), reference.get("steady_tokens_per_second")),
            "whole_phase_training_speedup": ratio(result.get("training_tokens_per_second"), reference.get("training_tokens_per_second"))}


def matched_observations(rows, ref_rows, axis):
    # Exact shared interim observations only; never interpolate measured quality into invented results.
    # Final evaluations use a larger token sample than interim ones, so they are never paired.
    interim = lambda rs: (r for r in rs if r.get("kind") == "validation" and not r.get("final"))
    reference = {r[axis]: r["val_bpb"] for r in interim(ref_rows)}
    return [{axis: r[axis], "bpb_delta": r["val_bpb"] - reference[r[axis]]} for r in interim(rows) if r[axis] in reference]


def seed_noise(values):
    """Sample standard deviation s (ddof=1) and the resolution threshold 2*sqrt(2)*s for a difference of two runs."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    s = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    return {"n": len(values), "mean": mean, "std": s, "threshold": SIGNIFICANCE_FACTOR * s}


def pooled(results, name):
    """Run, or mean over a seed group, as a record. None if any member has no eligible result yet."""
    runs = [results.get(r) for r in members(name)]
    if any(r is None for r in runs):
        return None
    keys = ("val_bpb", "test_bpb", "steady_tokens_per_second", "training_tokens_per_second", "parameters", "scaling_parameters")
    record = {"run_id": name, "members": [r["run_id"] for r in runs]}
    for key in keys:
        values = [r.get(key) for r in runs]
        record[key] = None if any(v is None for v in values) else sum(values) / len(values)
    return record


def means_threshold(std, n_run, n_ref):
    """Secondary rule: two-sigma bound on a difference of an n_run-seed mean and an n_ref-seed mean."""
    return 2 * std * math.sqrt(1 / n_run + 1 / n_ref)


def pooled_noise(results, groups=POOLED_GROUPS):
    """Sensitivity analysis only: SD pooled over the seed groups, s_p = sqrt(sum_g sum (x - mean_g)^2 / (N - G)),
    with the two-sided 95% Student-t quantile for df = N - G. None if df < 1."""
    samples = {g: {r: results[r]["test_bpb"] for r in members(g) if r in results and results[r].get("test_bpb") is not None}
               for g in groups}
    samples = {g: v for g, v in samples.items() if v}
    n = sum(len(v) for v in samples.values())
    df = n - len(samples)
    if df < 1:
        return None
    ss = sum((x - sum(v.values()) / len(v)) ** 2 for v in samples.values() for x in v.values())
    return {"members": {g: list(v) for g, v in samples.items()}, "n": n, "df": df,
            "std": math.sqrt(ss / df), "t_critical": T95[df - 1]}


def sensitivity(delta, noise, ns):
    """Pooled-t 95% interval for a contrast of independent means with sizes `ns`; None without pooled noise."""
    if noise is None:
        return None
    h = noise["t_critical"] * noise["std"] * math.sqrt(sum(1 / n for n in ns))
    return {"ci95": [delta - h, delta + h], "resolved": abs(delta) > h}


def paired_pairs():
    """Same-seed (subject, reference) pairs, derived from RUNS: `seeds` pairs each GQA-2 seed with the baseline seed
    that shares its data order; `iso` pairs each FLOP-budget run with its matched-token twin (same architecture and
    seed, hence identical initial weights and an identical batch sequence until the twin stops)."""
    base = {RUNS[r]["seed"]: r for r in GROUPS["d8-b0"]}
    seeds = [(r, base[RUNS[r]["seed"]]) for r in GROUPS["d8-gqa2"]]
    iso = [(r, next(t for t in RUNS if RUNS[t]["budget"] == "tokens" and RUNS[t]["seed"] == RUNS[r]["seed"]
                    and model_for(t) == model_for(r))) for r in RUNS if RUNS[r]["budget"] == "flops"]
    return seeds, iso


def paired_analysis(results, noise=None):
    """Secondary, pre-registered before E1: uses the pairing the seeds create instead of treating runs as independent.

    E2: d_k = gqa2-sk - b0-sk on test BPB for every seed (all pairs must be eligible); mean, SD s_d (ddof=1), the
    Student-t 95% interval mean +- t * s_d / sqrt(n), and s_d / (sqrt(2) * s), the share of seed noise that the
    shared data order does not cancel. E6: each iso run minus its twin, resolved if |delta| > 2 * s_d. The iso pairs
    also share initial weights, so s_d from the E2 pairs is assumed to be an upper bound on their noise."""
    seeds, iso = paired_pairs()
    if any(a not in results or b not in results for a, b in seeds):
        return None
    diffs = {a: results[a]["test_bpb"] - results[b]["test_bpb"] for a, b in seeds}
    n = len(diffs)
    mean = sum(diffs.values()) / n
    s_d = math.sqrt(sum((d - mean) ** 2 for d in diffs.values()) / (n - 1))
    h = T95[n - 2] * s_d / math.sqrt(n)
    e2 = {"pairs": [list(pair) for pair in seeds], "differences": diffs, "n": n, "df": n - 1, "mean": mean, "std": s_d,
          "t_critical": T95[n - 2], "ci95": [mean - h, mean + h], "resolved": abs(mean) > h,
          "unpaired_difference_std": None if noise is None else math.sqrt(2) * noise["std"],
          "noise_ratio": None if noise is None or not noise["std"] else s_d / (math.sqrt(2) * noise["std"])}
    e6 = []
    for a, b in iso:
        if a in results and b in results:
            delta = results[a]["test_bpb"] - results[b]["test_bpb"]
            e6.append({"run_id": a, "twin": b, "test_bpb_delta": delta, "threshold": 2 * s_d, "resolved": abs(delta) > 2 * s_d})
    return {"e2_seed_pairs": e2, "e6_iso_vs_twin": e6, "n_contrasts": 1 + len(e6)}


def pair_entry(run, reference, threshold, std=None, pn=None):
    """`run` and `reference` are `pooled` records. `resolved` is the pre-registered single-run rule
    (|delta| > 2*sqrt(2)*s); `resolved_by_means` accounts for the number of seeds on each side;
    `sensitivity` is the pooled-t interval from `pooled_noise`."""
    n_run, n_ref = len(run.get("members", [run["run_id"]])), len(reference["members"])
    entry = {"run_id": run["run_id"], **compare(run, reference), "members": run.get("members", [run["run_id"]]),
             "reference_members": reference["members"], "n_run": n_run, "n_ref": n_ref}
    if run.get("test_bpb") is not None and reference.get("test_bpb") is not None:
        delta = run["test_bpb"] - reference["test_bpb"]
        entry.update({"test_bpb_delta": delta, "test_bpb_relative_delta": delta / reference["test_bpb"],
                      "resolved": None if threshold is None else abs(delta) > threshold,
                      "resolved_by_means": None if std is None else abs(delta) > means_threshold(std, n_run, n_ref),
                      "sensitivity": sensitivity(delta, pn, (n_run, n_ref))})
    entry["parameters"] = {"run": run.get("parameters"), "reference": reference.get("parameters")}
    entry["scaling_parameters"] = {"run": run.get("scaling_parameters"), "reference": reference.get("scaling_parameters")}
    return entry


def interaction(results, a, b, c, d, std, pn=None):
    """2x2 interaction (a - b) - (c - d) on pooled test BPB. Its two-sigma bound is 2*s*sqrt(sum 1/n_i):
    the four (group-mean) terms are independent, each with variance s^2/n_i."""
    records = [pooled(results, name) for name in (a, b, c, d)]
    if any(r is None or r.get("test_bpb") is None for r in records):
        return None
    ya, yb, yc, yd = (r["test_bpb"] for r in records)
    ns = [len(r["members"]) for r in records]
    threshold = None if std is None else 2 * std * math.sqrt(sum(1 / n for n in ns))
    value = (ya - yb) - (yc - yd)
    return {"terms": [a, b, c, d], "members": [r["members"] for r in records], "effect_with": ya - yb,
            "effect_without": yc - yd, "interaction": value, "threshold": threshold,
            "resolved": None if threshold is None else abs(value) > threshold, "sensitivity": sensitivity(value, pn, ns)}


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
    lines = ["| Run | Exp | Params (total / scaling) | Tokens | Est. FLOPs / budget | Test BPB | Val BPB | Steady tok/s | Whole-phase tok/s (incl. compile/eval/ckpt) |",
             "|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v, f: "-" if v is None else format(v, f)
    for r in payload["runs"]:
        share = r["estimated_flops"] / r["flop_budget"] if r.get("estimated_flops") and r.get("flop_budget") else None
        lines.append(f"| {r['run_id']} | {RUNS[r['run_id']]['experiment']} | {fmt(r.get('parameters'), ',')} / {fmt(r.get('scaling_parameters'), ',')} | "
                     f"{fmt(r.get('tokens'), ',')} | {fmt(share, '.2%')} | {fmt(r.get('test_bpb'), '.4f')} | {fmt(r.get('val_bpb'), '.4f')} | "
                     f"{fmt(r.get('steady_tokens_per_second'), ',.0f')} | {fmt(r.get('training_tokens_per_second'), ',.0f')} |")
    lines += ["", "Comparison exclusions:"]
    exclusions = [f"- {name}: {'; '.join(status['reasons'])}" for name, status in payload["eligibility"].items() if not status["eligible"]]
    lines += exclusions or ["None."]
    noise, gqa2, pn = payload["noise"], payload["gqa2_noise"], payload["pooled_noise"]
    lines += ["", "Seed noise (d8-b0, test BPB): " + ("not yet available (needs >= 2 seeds)" if noise is None else
              f"n={noise['n']}, mean={noise['mean']:.4f}, s={noise['std']:.4f}, resolution threshold 2*sqrt(2)*s = {noise['threshold']:.4f}"),
              "Seed noise (d8-gqa2, test BPB; checks the equal-noise assumption): " + ("not yet available (needs >= 2 eligible GQA-2 seeds)" if gqa2 is None else
              f"n={gqa2['n']}, mean={gqa2['mean']:.4f}, s={gqa2['std']:.4f}"),
              "Sensitivity analysis (pooled d8-b0 + d8-gqa2 seed SD, Student-t 95%): " + ("not yet available" if pn is None else
              f"s_p={pn['std']:.4f}, df={pn['df']}, t={pn['t_critical']:.3f}") + f"; contrasts reported: {payload['n_contrasts']}",
              "", "| Run (seeds) | vs (seeds) | dTest BPB | resolved (primary) | by means (secondary) | pooled-t 95% CI (sensitivity) | "
              "resolved (sensitivity) | dVal BPB | steady speedup | whole-phase speedup |", "|---|---|---|---|---|---|---|---|---|---|"]
    flag = {True: "yes", False: "no", None: "-"}
    ci = lambda x: "-" if x is None else f"[{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]"
    sens = lambda x: flag[None if x is None else x["resolved"]]
    for c in payload["comparisons"]:
        lines.append(f"| {c['run_id']} ({c['n_run']}) | {c['reference']} ({c['n_ref']}) | {fmt(c.get('test_bpb_delta'), '+.4f')} | "
                     f"{flag[c.get('resolved')]} | {flag[c.get('resolved_by_means')]} | {ci(c.get('sensitivity'))} | {sens(c.get('sensitivity'))} | "
                     f"{fmt(c['bpb_delta'], '+.4f')} | {fmt(c['training_speedup'], '.3f')} | {fmt(c['whole_phase_training_speedup'], '.3f')} |")
    lines += ["", "Paired analysis (secondary; same-seed pairs share data order, iso runs also share initial weights):"]
    pa = payload["paired"]
    if pa is None:
        lines.append("not yet available (needs every d8-b0 and d8-gqa2 seed)")
    else:
        e2 = pa["e2_seed_pairs"]
        ratio = "-" if e2["noise_ratio"] is None else f"{e2['noise_ratio']:.2f}"
        lines += [f"- E2 gqa2-sk - b0-sk: mean {e2['mean']:+.4f}, s_d={e2['std']:.4f} (df={e2['df']}), "
                  f"95% CI [{e2['ci95'][0]:+.4f}, {e2['ci95'][1]:+.4f}], resolved: {flag[e2['resolved']]}; "
                  f"s_d / (sqrt(2)*s) = {ratio}"]
        lines += [f"- E6 {i['run_id']} - {i['twin']}: {i['test_bpb_delta']:+.4f} vs 2*s_d = {i['threshold']:.4f}, "
                  f"resolved: {flag[i['resolved']]}" for i in pa["e6_iso_vs_twin"]] or ["- E6: iso runs or twins not yet available"]
    lines += ["", "| Interaction (a-b)-(c-d) | with | without | interaction | threshold | resolved (primary) | "
              "pooled-t 95% CI (sensitivity) | resolved (sensitivity) |", "|---|---|---|---|---|---|---|---|"]
    for name, i in payload["interactions"].items():
        if i is None:
            lines.append(f"| {name} | - | - | - | - | - | - | - |")
        else:
            a, b, c, d = i["terms"]
            lines.append(f"| {name}: ({a}-{b})-({c}-{d}) | {i['effect_with']:+.4f} | {i['effect_without']:+.4f} | "
                         f"{i['interaction']:+.4f} | {fmt(i['threshold'], '.4f')} | {flag[i['resolved']]} | "
                         f"{ci(i['sensitivity'])} | {sens(i['sensitivity'])} |")
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
    raw_results = {run_id: item["result"] for run_id, item in loaded.items()}
    eligibility = {run_id: comparison_eligibility(item["result"], item["config"]) for run_id, item in loaded.items()}
    results = {run_id: result for run_id, result in raw_results.items() if eligibility[run_id]["eligible"]}
    def group_noise(name):
        return seed_noise([results[r]["test_bpb"] for r in members(name) if r in results and results[r].get("test_bpb") is not None])
    noise, gqa2_noise = group_noise("d8-b0"), group_noise("d8-gqa2")
    threshold = None if noise is None else noise["threshold"]
    std = None if noise is None else noise["std"]
    pn = pooled_noise(results)
    # Subjects: each seed group once (as its mean), plus every ungrouped run.
    grouped = {r: g for g, rs in GROUPS.items() for r in rs}
    subjects = list(dict.fromkeys(grouped.get(r, r) for r in RUNS))
    comparisons = []
    for subject_name in subjects:
        subject = pooled(results, subject_name)
        if subject is None:
            continue
        run_id = subject["members"][0]  # the actual run behind ancillary measurements
        item = loaded[run_id]
        for ref_name in RUNS[run_id]["compare_to"]:
            reference = pooled(results, ref_name)
            if reference is None or reference["val_bpb"] is None:
                continue
            entry = pair_entry(subject, reference, threshold, std, pn)
            ref_id = reference["members"][0]
            # These ancillary measurements use one actual run on each side, whereas final
            # quality and throughput use the explicitly named group means.
            entry["field_subject"] = run_id
            entry["field_references"] = {field: ref_id for field in (
                "matched_tokens", "matched_estimated_flops", "elapsed_seconds_difference", "inference_speedups")}
            if RUNS[run_id]["budget"] == RUNS[ref_id]["budget"]:
                entry["matched_tokens"] = matched_observations(item["curves"], loaded[ref_id]["curves"], "tokens")
                entry["matched_estimated_flops"] = matched_observations(item["curves"], loaded[ref_id]["curves"], "estimated_flops")
            else:  # the FLOP-budget twin's learning rate decays later; only final BPB is compared across regimes
                entry.update(matched_tokens=[], matched_estimated_flops=[],
                             matched_observations_note="different budget regimes; curves are not schedule-matched")
            entry["elapsed_seconds_difference"] = results[run_id]["training_seconds"] - results[ref_id]["training_seconds"]
            entry["inference_speedups"] = []
            if item["inference"] and loaded[ref_id]["inference"]:
                a, b = item["inference"], loaded[ref_id]["inference"]
                for field in ("device", "precision", "sampling", "attention_backend", "torch", "cuda"):
                    if a[field] != b[field]:
                        raise ValueError(f"inference {field} differs across compared runs")
                entry["inference_speedups"] = inference_speedups(a["rows"], b["rows"])
            comparisons.append(entry)
    interactions = {name: interaction(results, *terms, std, pn) for name, terms in INTERACTIONS.items()}
    payload = {"runs": list(raw_results.values()), "eligibility": eligibility, "noise": noise, "gqa2_noise": gqa2_noise,
               "pooled_noise": pn, "comparisons": comparisons, "interactions": interactions,
               "paired": paired_analysis(results, noise),
               "n_contrasts": sum(c.get("test_bpb_delta") is not None for c in comparisons)
                              + sum(i is not None for i in interactions.values()),
               "missing_runs": [r for r in RUNS if r not in raw_results],
               "caveat": ("primary resolution uses seed noise from the three d8 baseline runs only (s on 2 df); the "
                          "pre-registered 2*sqrt(2)*s rule then has a false-positive rate well above 5% (about 13-18% per "
                          "contrast by simulation); resolved_by_means is a secondary rule for three-seed group means; "
                          "`sensitivity` is a separately labelled sensitivity analysis using the SD pooled over the d8-b0 and "
                          "d8-gqa2 seeds with Student-t 95% intervals; n_contrasts counts the contrasts reported, with no "
                          "multiplicity correction; `paired` is a secondary paired analysis of same-seed pairs (E2 seeds, E6 iso vs "
                          "twin) on 2 df, with its own contrast count; unresolved differences are not ties")}
    from experiments.train import write_json
    write_json(args.output_root / "comparison.json", payload)
    (args.output_root / "tables.md").write_text(tables(payload))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
