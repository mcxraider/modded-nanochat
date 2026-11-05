"""Plot checks use explicit synthetic fixtures, never claimed experiment results."""
import copy
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pytest

from experiments import make_configs, visualise as v
from experiments.config import RUNS

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture(scope="module")
def config_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("configs")
    saved, sys.argv = sys.argv, ["make_configs", "--config-dir", str(directory)]
    try:
        make_configs.main()
    finally:
        sys.argv = saved
    return directory


@pytest.fixture
def artifacts(tmp_path, monkeypatch, config_dir):
    monkeypatch.setattr(v, "NOTE", "SYNTHETIC TEST FIXTURE — not experiment results. No confidence intervals.")
    for index, name in enumerate(RUNS):
        depth = RUNS[name]["depth"]
        directory = tmp_path / name
        directory.mkdir()
        config = json.loads((config_dir / f"{name}.json").read_text())
        write_json(directory / "config.json", config)
        rows = []
        for step in (1, 2, 3):
            common = dict(step=step, training_seconds=step * (120 + 5 * index), tokens=step * 131072,
                          estimated_flops=step * 1e15)
            rows.append(dict(kind="train", **common, loss=6.0 - step * 0.3 + index * 0.02,
                             step_seconds=110 + index, tokens_per_second=1200 + index * 200,
                             whole_phase_tokens_per_second=1000 + index * 150, peak_vram_bytes=(30 + index) * 2**30))
            rows.append(dict(kind="validation", **common, val_bpb=1 - step * 0.07 + index * 0.01))
        rows.append(dict(kind="test", final=True, test_bpb=0.72 + index * 0.001, step=3, training_seconds=3 * (120 + 5 * index),
                         tokens=3 * 131072, estimated_flops=3e15))
        (directory / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        write_json(directory / "result.json", dict(run_id=name, val_bpb=0.718 + index * 0.001, test_bpb=0.72 + index * 0.001,
                   training_tokens_per_second=1000 + index * 150, steady_tokens_per_second=1200 + index * 200,
                   peak_vram_bytes=(30 + index) * 2**30, training_seconds=7200 + index, stop_reason="token_target", data_epoch=1,
                   tokens=config["training"]["target_tokens"], target_tokens=config["training"]["target_tokens"],
                   estimated_flops=3e15, flop_budget=config["training"]["flop_budget"]))
        write_json(directory / "data_manifest.json", {"dataset": f"SYNTHETIC TEST FIXTURE d{depth}"})
        write_json(directory / "launch.json", dict(base_commit=config["base_commit"], overlay_sha256={"fixture": "test"}))
        write_json(directory / "hardware.json", dict(gpu="SYNTHETIC", world_size=2, torch="test", cuda="test",
                   precision="test", attention_backend="test", packages={"fixture": "test"}))
        inference = dict(device="SYNTHETIC", precision="test", sampling="greedy", attention_backend="test", torch="test", cuda="test", rows=[])
        for prompt in (128, 2048):
            for batch in (1, 4, 16):
                if name == "d8-gqa2-s1" and batch == 16:
                    row = dict(batch_size=batch, prompt_tokens=prompt, status="oom")
                else:
                    row = dict(batch_size=batch, prompt_tokens=prompt, status="ok", generated_tokens_per_request=256,
                               decode_timed_tokens_per_request=255, prefill_tokens_per_second=batch * (1000 + index * 100),
                               decode_tokens_per_second_aggregate=batch * (80 + index * 10),
                               time_to_first_token_seconds=prompt / 1000, peak_vram_bytes=(20 + batch) * 2**30,
                               kv_cache_bytes_allocated=batch * 2**20)
                inference["rows"].append(row)
        write_json(directory / "inference.json", inference)
    return tmp_path


def d8(runs):
    return {name: run for name, run in runs.items() if RUNS[name]["depth"] == 8}


def test_all_plots_export_png_and_svg(artifacts):
    with pytest.warns(UserWarning, match="OOM workloads"):
        paths = v.render(artifacts)
    per_depth = {"learning_curves", "training_performance", "final_comparison", "quality_speed", "inference_comparison"}
    assert {p.stem for p in paths} == {f"{n}_d{d}" for n in per_depth for d in (8,)} | {"dose_response", "budget_comparison"}
    assert len(paths) == 14  # 7 figures x (png, svg)
    for path in paths:
        assert path.stat().st_size > 5000
    assert plt.get_fignums() == []


def test_partial_training_does_not_require_final_results(artifacts):
    for directory in artifacts.iterdir():
        (directory / "result.json").unlink()
        (directory / "inference.json").unlink()
    paths = v.render(artifacts)
    assert {p.stem for p in paths} == {f"{n}_d{d}" for n in ("learning_curves", "training_performance") for d in (8,)}


def test_smoothing_and_units_preserve_evidence(artifacts):
    assert v.rolling_mean([1, 3, 8], 2) == [1, 2, 5.5]
    runs, _ = v.load_runs(artifacts)
    fig = v.learning_curves(d8(runs), "SYNTHETIC TEST", 2)
    first = fig.axes[0]
    assert list(first.lines[0].get_xdata()) == [2, 4, 6]
    assert list(first.lines[0].get_ydata()) == pytest.approx([5.7, 5.4, 5.1])
    assert list(first.lines[1].get_ydata()) == pytest.approx([5.7, 5.55, 5.25])
    plt.close(fig)
    with pytest.warns(UserWarning, match="OOM workloads"):
        fig = v.inference_comparison(d8(runs), "SYNTHETIC TEST")
    # Fourth series (d8-gqa2-s1) has no fabricated zero or extrapolated OOM value.
    assert list(fig.axes[0].lines[3].get_ydata())[:2] == [1300, 5200]
    assert str(fig.axes[0].lines[3].get_ydata()[2]) == "nan"
    # TTFT seconds are converted to milliseconds.
    assert list(fig.axes[4].lines[0].get_ydata()) == [128, 128, 128]
    # The KV-cache row exists when the benchmark recorded allocated cache bytes.
    assert len(fig.axes) == 5 * 2
    plt.close(fig)


def test_resume_discards_abandoned_future_and_incomplete_tail(artifacts):
    path = artifacts / "d8-b0-s1/metrics.jsonl"
    rows = v.read_metrics(path)
    resumed = copy.deepcopy(next(r for r in rows if r["kind"] == "train" and r["step"] == 2))
    resumed["loss"] = 4.0
    with path.open("a") as stream:
        stream.write(json.dumps(resumed) + '\n{"kind":')
    with pytest.warns(UserWarning, match="unfinished final record"):
        restored = v.read_metrics(path)
    assert max(r["step"] for r in restored) == 2
    assert next(r for r in restored if r["kind"] == "train" and r["step"] == 2)["loss"] == 4.0
    assert not any(r["kind"] == "validation" and r["step"] == 2 for r in restored)


def test_malformed_completed_record_is_not_silently_skipped(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"kind":\n')
    with pytest.raises(ValueError, match="invalid JSON"):
        v.read_metrics(path)


def test_known_data_confounds_are_rejected(artifacts):
    write_json(artifacts / "d8-gqa2-r75/data_manifest.json", {"dataset": "different"})
    with pytest.raises(ValueError, match="data_manifest.json"):
        v.load_runs(artifacts)


def test_consistent_d8_data_manifests_are_verified(artifacts):
    runs, verified = v.load_runs(artifacts)
    assert verified and len(runs) == len(RUNS)


def test_no_results_gives_actionable_error(tmp_path):
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match="Run training first"):
            v.render(tmp_path)


def test_previous_attempt_summaries_are_hidden_during_resume(artifacts):
    directory = artifacts / "d8-b0-s1"
    latest = max((directory / name).stat().st_mtime for name in ("result.json", "inference.json")) + 1
    write_json(directory / "rental_accounting.json", {"launches": [{"started_unix_seconds": latest, "status": "running"}]})
    with pytest.warns(UserWarning, match="previous-attempt"):
        runs, _ = v.load_runs(artifacts)
    assert runs["d8-b0-s1"]["result"] is None
    assert runs["d8-b0-s1"]["inference"] is None
    assert runs["d8-b0-s1"]["rows"]


def test_final_pairwise_differences_follow_the_runs_table_and_use_test_bpb(artifacts):
    runs, _ = v.load_runs(artifacts)
    noise = v.baseline_noise(runs)
    assert noise["std"] == pytest.approx(0.001)
    fig = v.final_comparison(d8(runs), "SYNTHETIC TEST", noise)
    # Fixture test BPB = 0.72 + 0.001 * index in RUNS; b0 mean 0.721, gqa2 mean 0.724 (seeds at 0.723-0.725).
    # (gqa2, b0) (r75, b0) (r75, gqa2) (pm, b0) (pm, gqa2) (gqa4, b0) (gqa4, gqa2) (r25, gqa2) (r25, r75)
    # (b0-nove, b0) (gqa2-nove, b0-nove) (gqa2-nove, gqa2) (gqa2-iso, gqa2) (gqa2-iso, b0) (gqa4-iso, gqa4) (gqa4-iso, b0)
    assert [bar.get_width() for bar in fig.axes[1].patches] == pytest.approx(
        [0.003, 0.005, 0.002, 0.006, 0.003, 0.007, 0.004, 0.005, 0.003, 0.009, 0.001, 0.007, 0.008, 0.011, 0.005, 0.012])
    assert "d8-gqa2-r75" in fig.axes[1].get_yticklabels()[2].get_text() and "− d8-gqa2" in fig.axes[1].get_yticklabels()[2].get_text()
    assert "seed-noise" in fig.axes[1].get_xlabel()
    plt.close(fig)


def test_dose_response_uses_final_results(artifacts):
    runs, _ = v.load_runs(artifacts)
    fig = v.dose_response(runs, "SYNTHETIC TEST")
    assert list(fig.axes[0].lines[0].get_xdata()) == [1, 2, 4]
    assert list(fig.axes[0].lines[0].get_ydata()) == pytest.approx([0.728, 0.724, 0.721])  # gqa4, gqa2 mean, b0 mean
    assert list(fig.axes[1].lines[0].get_xdata()) == [0.25, 0.75, 1.0]
    plt.close(fig)


def test_test_records_are_read_and_validated(tmp_path):
    path = tmp_path / "metrics.jsonl"
    record = dict(kind="test", final=True, test_bpb=0.7, step=3, training_seconds=1, tokens=1, estimated_flops=1)
    path.write_text(json.dumps(record) + "\n")
    assert v.read_metrics(path)[0]["test_bpb"] == 0.7
    path.write_text(json.dumps({**record, "test_bpb": float("nan")}).replace("NaN", "NaN") + "\n")
    with pytest.raises(ValueError, match="test_bpb"):
        v.read_metrics(path)


@pytest.mark.parametrize("missing", [None, "d8-gqa2-r25"])
def test_d8_report_omits_scale_transfer_and_tracks_missing_runs(artifacts, monkeypatch, missing):
    from experiments import report
    if missing:
        (artifacts / missing / "result.json").unlink()
    monkeypatch.setattr(sys, "argv", ["report", "--output-root", str(artifacts)])
    report.main()
    payload = json.loads((artifacts / "comparison.json").read_text())
    assert "scale_transfer" not in payload
    assert payload["missing_runs"] == ([] if missing is None else [missing])
    assert len(payload["runs"]) == len(RUNS) - (missing is not None)
    assert payload["noise"]["n"] == 3
    assert payload["comparisons"]
    assert (artifacts / "tables.md").is_file()
    paired = payload["paired"]  # every B0 / GQA-2 seed is present in both cases
    assert paired["e2_seed_pairs"]["n"] == 3
    assert {i["run_id"] for i in paired["e6_iso_vs_twin"]} == {"d8-gqa2-iso", "d8-gqa4-iso"}
    assert "Paired analysis" in (artifacts / "tables.md").read_text()


@pytest.mark.parametrize("bad_run,change", [
    ("d8-b0-s1", {"stop_reason": "wallclock_cap"}),
    ("d8-gqa2-s1", {"stop_reason": "wallclock_cap"}),
    ("d8-gqa2-s1", {"tokens": 1}),
    ("d8-gqa2-s1", {"data_epoch": 2}),
])
def test_ineligible_runs_remain_raw_but_leave_final_comparisons(artifacts, monkeypatch, bad_run, change):
    from experiments import report
    path = artifacts / bad_run / "result.json"
    result = json.loads(path.read_text())
    result.update(change)
    write_json(path, result)
    monkeypatch.setattr(sys, "argv", ["report", "--output-root", str(artifacts)])
    report.main()
    payload = json.loads((artifacts / "comparison.json").read_text())
    assert any(r["run_id"] == bad_run for r in payload["runs"])
    assert not payload["eligibility"][bad_run]["eligible"]
    assert bad_run not in payload["missing_runs"]
    assert all(bad_run not in c["members"] and bad_run not in c["reference_members"] for c in payload["comparisons"])
    assert payload["noise"]["n"] == (2 if bad_run.startswith("d8-b0") else 3)
    assert bad_run in (artifacts / "tables.md").read_text().split("Comparison exclusions:")[1]
    with pytest.warns(UserWarning, match="excluded from final comparisons"):
        runs, _ = v.load_runs(artifacts)
    assert runs[bad_run]["rows"] and runs[bad_run]["result"]
    assert v.baseline_noise(runs) == payload["noise"]
    assert bad_run not in v.eligible_results(runs)
    fig = v.final_comparison(runs, "SYNTHETIC")
    assert bad_run not in [t.get_text() for t in fig.axes[0].get_yticklabels()]
    plt.close(fig)
    fig = v.quality_speed(runs, "SYNTHETIC")
    assert bad_run not in fig.axes[0].get_legend_handles_labels()[1]
    plt.close(fig)


def test_ancillary_measurements_explicitly_name_actual_reference(artifacts, monkeypatch):
    from experiments import report
    monkeypatch.setattr(sys, "argv", ["report", "--output-root", str(artifacts)])
    report.main()
    payload = json.loads((artifacts / "comparison.json").read_text())
    comparison = next(c for c in payload["comparisons"] if c["reference"] == "d8-b0")
    assert comparison["reference_members"] == ["d8-b0-s1", "d8-b0-s2", "d8-b0-s3"]
    assert set(comparison["field_references"].values()) == {"d8-b0-s1"}


def test_report_excludes_capped_baseline_and_labels_ancillary_references(artifacts, monkeypatch):
    from experiments import report
    monkeypatch.setattr(sys, 'argv', ['report', '--output-root', str(artifacts)])
    report.main()
    payload = json.loads((artifacts / 'comparison.json').read_text())
    comparison = next(c for c in payload['comparisons'] if c['run_id'] == 'd8-gqa2')
    assert comparison['members'] == ['d8-gqa2-s1', 'd8-gqa2-s2', 'd8-gqa2-s3'] and comparison['field_subject'] == 'd8-gqa2-s1'
    assert comparison['reference'] == 'd8-b0'
    assert comparison['field_references']['inference_speedups'] == 'd8-b0-s1'
    result_path = artifacts / 'd8-b0-s1/result.json'
    result = json.loads(result_path.read_text())
    result.update(stop_reason='wallclock_cap', tokens=131072)
    write_json(result_path, result)
    report.main()
    payload = json.loads((artifacts / 'comparison.json').read_text())
    assert payload['noise']['n'] == 2
    assert not payload['eligibility']['d8-b0-s1']['eligible']
    assert len(payload["runs"]) == len(RUNS)
    assert not any(c['reference'] == 'd8-b0' for c in payload['comparisons'])
    assert 'wallclock_cap' in (artifacts / 'tables.md').read_text()


def test_gqa2_seeds_are_compared_as_one_mean_everywhere(artifacts, monkeypatch):
    from experiments import report
    monkeypatch.setattr(sys, "argv", ["report", "--output-root", str(artifacts)])
    report.main()
    payload = json.loads((artifacts / "comparison.json").read_text())
    assert payload["missing_runs"] == []
    # Fixture test BPB is 0.72 + 0.001 * (index in RUNS): gqa2 seeds sit at indices 3, 4, 5.
    gqa2 = next(c for c in payload["comparisons"] if c["run_id"] == "d8-gqa2" and c["reference"] == "d8-b0")
    assert gqa2["members"] == ["d8-gqa2-s1", "d8-gqa2-s2", "d8-gqa2-s3"] and gqa2["n_run"] == 3 and gqa2["n_ref"] == 3
    assert gqa2["test_bpb_delta"] == pytest.approx(0.724 - 0.721)
    assert payload["gqa2_noise"]["n"] == 3
    # Seeds are never compared individually; downstream runs reference the GQA-2 mean.
    assert not any(c["run_id"].startswith("d8-gqa2-s") for c in payload["comparisons"])
    r25 = next(c for c in payload["comparisons"] if c["run_id"] == "d8-gqa2-r25" and c["reference"] == "d8-gqa2")
    assert r25["reference_members"] == gqa2["members"]
    runs, verified = v.load_runs(artifacts)
    assert verified
    fig = v.dose_response(runs, "SYNTHETIC TEST")
    assert dict(zip(fig.axes[0].lines[0].get_xdata(), fig.axes[0].lines[0].get_ydata()))[2] == pytest.approx(0.724)
    plt.close(fig)


def test_report_splits_regimes_labels_rules_and_tolerates_missing_steady_throughput(artifacts, monkeypatch):
    from experiments import report
    path = artifacts / "d8-gqa4/result.json"
    write_json(path, {**json.loads(path.read_text()), "steady_tokens_per_second": None})
    monkeypatch.setattr(sys, "argv", ["report", "--output-root", str(artifacts)])
    report.main()
    payload = json.loads((artifacts / "comparison.json").read_text())
    iso = next(c for c in payload["comparisons"] if c["run_id"] == "d8-gqa2-iso" and c["reference"] == "d8-gqa2")
    assert iso["matched_tokens"] == iso["matched_estimated_flops"] == [] and "not schedule-matched" in iso["matched_observations_note"]
    same = next(c for c in payload["comparisons"] if c["run_id"] == "d8-gqa2" and c["reference"] == "d8-b0")
    assert same["matched_tokens"] and "matched_observations_note" not in same
    # Fixture steady tok/s = 1200 + 200 * index (gqa2 mean 2000, b0 1400); whole-phase 1000 + 150 * index (1600, 1150).
    assert same["training_speedup"] == pytest.approx(2000 / 1400) and same["whole_phase_training_speedup"] == pytest.approx(1600 / 1150)
    assert next(c for c in payload["comparisons"] if c["run_id"] == "d8-gqa4")["training_speedup"] is None
    assert payload["pooled_noise"]["df"] == 4 and payload["n_contrasts"] == len(payload["comparisons"]) + 1
    table = (artifacts / "tables.md").read_text()
    assert all(h in table for h in ("resolved (primary)", "by means (secondary)", "pooled-t 95% CI (sensitivity)", "df=4", "t=2.776"))
    runs, _ = v.load_runs(artifacts)
    plt.close(v.final_comparison(d8(runs), "SYNTHETIC"))
    with pytest.warns(UserWarning, match="No steady-state throughput"):
        fig = v.quality_speed(d8(runs), "SYNTHETIC")
    assert "d8-gqa4" not in fig.axes[0].get_legend_handles_labels()[1] and "Steady-state" in fig.axes[0].get_xlabel()
    plt.close(fig)
