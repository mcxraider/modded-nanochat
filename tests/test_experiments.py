import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.config import GROUPS, RUNS, members, model_for, read_config, schedule, validate_suite
from experiments import make_configs, run, run_all
from experiments.report import compare, matched_observations

ROOT = Path(__file__).resolve().parents[1]
FIRST = "d8-b0-s1"


@pytest.fixture(scope="module")
def config_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("configs")
    sys.argv, saved = ["make_configs", "--config-dir", str(directory)], sys.argv
    try:
        make_configs.main()
    finally:
        sys.argv = saved
    return directory


def configs(config_dir):
    return [read_config(config_dir / f"{run_id}.json") for run_id in RUNS]


def test_generated_suite_matches_runs_table_and_batch_policy(config_dir):
    suite = configs(config_dir)
    assert [c["run_id"] for c in suite] == list(RUNS)
    validate_suite(suite)
    for config in suite:
        t, m = config["training"], config["model"]
        micro = config["workers"] * t["device_batch_size"] * m["sequence_len"]
        assert t["total_batch_size"] % micro == 0 and t["target_tokens"] % t["total_batch_size"] == 0
        assert config["model"]["n_embd"] // config["model"]["n_head"] == 128
    assert {RUNS[r]["depth"] for r in RUNS} == {8, 12}


def test_runs_table_references_resolve_and_encode_the_experiment_design():
    for run_id, spec in RUNS.items():
        for ref in spec["compare_to"]:
            assert members(ref)
    assert members("d8-b0") == GROUPS["d8-b0"] and len(GROUPS["d8-b0"]) == 3
    assert [model_for(r)["n_kv_head"] for r in ("d8-b0-s1", "d8-gqa2", "d8-gqa4")] == [4, 2, 1]
    assert [model_for(r)["n_kv_head"] for r in ("d12-b0", "d12-gqa2")] == [6, 3]
    assert model_for("d8-gqa2-pm")["mlp_ratio"] == 4.5 and model_for("d8-gqa2-r25")["rotary_fraction"] == 0.25
    assert sorted(RUNS[r]["seed"] for r in GROUPS["d8-b0"]) == [1, 2, 3]


def test_suite_rejects_policy_drift_duplicates_and_table_mismatch(config_dir):
    suite = configs(config_dir)
    drifted = copy.deepcopy(suite)
    drifted[3]["optimizer"]["matrix_lr"] *= 2
    with pytest.raises(ValueError, match="may differ only"):
        validate_suite(drifted)
    duplicate = copy.deepcopy(suite)
    duplicate[4]["model"], duplicate[4]["seed"] = duplicate[3]["model"], duplicate[3]["seed"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_suite(duplicate)
    wrong = copy.deepcopy(suite)
    wrong[0]["model"]["rotary_fraction"] = 0.5
    with pytest.raises(ValueError, match="RUNS table"):
        validate_suite(wrong)
    capped = copy.deepcopy(suite)
    capped[0]["training"]["max_steps"] = 10
    with pytest.raises(ValueError, match="no step cap"):
        validate_suite(capped)


@pytest.mark.parametrize("section,key,value,match", [
    ("model", "mlp_ratio", 4.3, "multiple of 128"),
    ("training", "target_tokens", 131073, "multiple of total_batch_size"),
    ("evaluation", "test_tokens", 1000, "divide exactly"),
    ("training", "wallclock_cap_seconds", 0, "wallclock"),
])
def test_invalid_budget_and_model_values_rejected(config_dir, tmp_path, section, key, value, match):
    config = configs(config_dir)[0]
    config[section][key] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=match):
        read_config(path)


def test_unknown_config_fields_rejected(config_dir, tmp_path):
    config = configs(config_dir)[0]
    config["training"]["fp8"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="training fields"):
        read_config(path)


def test_step_based_schedule(config_dir):
    optimizer = configs(config_dir)[0]["optimizer"]
    assert schedule(0.0, optimizer) == (1, 1)
    assert schedule(0.5, optimizer) == (1, 0.5)
    assert schedule(0.75, optimizer) == (0.5, 0.25)
    assert schedule(1.0, optimizer) == (0, 0)
    assert schedule(1.2, optimizer) == (0, 0)  # clamped
    # progress is a pure function of step / total_steps, so a resumed run continues the same schedule
    assert schedule(2400 / 3200, optimizer) == schedule(0.75, optimizer)


def test_dry_run_does_not_create_runtime_or_artifacts(config_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(run, "prepare_runtime", lambda _: pytest.fail("dry run prepared a runtime"))
    run.launch(config_dir / f"{FIRST}.json", tmp_path / "outputs", tmp_path / "runtime", dry_run=True)
    assert list(tmp_path.iterdir()) == []


def test_existing_output_rejected(config_dir, tmp_path):
    (tmp_path / FIRST).mkdir()
    with pytest.raises(FileExistsError):
        run.launch(config_dir / f"{FIRST}.json", tmp_path, tmp_path / "runtime")


def test_sequential_order_and_failure_propagation(config_dir, tmp_path, monkeypatch):
    called = []
    def launch(path, *_):
        called.append(path.name)
        if len(called) == 2:
            raise subprocess.CalledProcessError(1, ["torchrun"])
    monkeypatch.setattr(run_all, "launch", launch)
    monkeypatch.setattr(sys, "argv", ["run_all", "--config-dir", str(config_dir), "--output-root", str(tmp_path)])
    with pytest.raises(subprocess.CalledProcessError):
        run_all.main()
    assert called == [f"{r}.json" for r in list(RUNS)[:2]]


def test_run_all_rejects_unknown_runs(config_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_all", "--config-dir", str(config_dir), "--output-root", str(tmp_path), "--runs", "B0"])
    with pytest.raises(SystemExit, match="unknown run IDs"):
        run_all.main()


def test_runtime_loader_patch():
    source = subprocess.check_output(["git", "show", "324e69c:nanochat/checkpoint_manager.py"], cwd=ROOT, text=True)
    patched = run.patch_checkpoint_loader(source)
    compile(patched, "checkpoint_manager.py", "exec")
    assert "model = model_class(model_config)" in patched
    assert "modified_gpt_324e69c" in patched


def test_report_arithmetic_and_no_interpolated_observations():
    reference = {"run_id": "B0", "val_bpb": 2, "training_tokens_per_second": 100}
    result = {"val_bpb": 1.5, "training_tokens_per_second": 150}
    assert compare(result, reference) == {"reference": "B0", "bpb_delta": -0.5, "bpb_relative_delta": -0.25, "training_speedup": 1.5}
    assert matched_observations([{"kind": "validation", "tokens": 10, "val_bpb": 1}], [], "tokens") == []


def test_launch_trains_then_benchmarks_on_one_gpu(config_dir, tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr(run, "prepare_runtime", lambda _: (runtime, {}))
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "test-source")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-first,GPU-second")
    calls = []
    def execute(cmd, **kwargs):
        calls.append((cmd, kwargs["env"]))
    monkeypatch.setattr(subprocess, "run", execute)
    run.launch(config_dir / f"{FIRST}.json", tmp_path / "artifacts", runtime)
    assert "--nproc_per_node=2" in calls[0][0]
    assert "experiments.train" in calls[0][0]
    assert "experiments.benchmark" in calls[1][0]
    assert calls[1][1]["CUDA_VISIBLE_DEVICES"] == "GPU-first"
    accounting = json.loads((tmp_path / f"artifacts/{FIRST}/rental_accounting.json").read_text())
    assert accounting["launches"][0]["status"] == "complete"


def test_failed_launch_records_attempt_and_skips_benchmark(config_dir, tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr(run, "prepare_runtime", lambda _: (runtime, {}))
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "test-source")
    calls = []
    def execute(cmd, **kwargs):
        calls.append(cmd)
        raise subprocess.CalledProcessError(1, cmd)
    monkeypatch.setattr(subprocess, "run", execute)
    with pytest.raises(subprocess.CalledProcessError):
        run.launch(config_dir / f"{FIRST}.json", tmp_path / "artifacts", runtime)
    assert len(calls) == 1
    accounting = json.loads((tmp_path / f"artifacts/{FIRST}/rental_accounting.json").read_text())
    assert accounting["launches"][0]["status"] == "failed_or_interrupted"


def test_seed_noise_threshold_and_resolution():
    from experiments.report import pair_entry, pooled, seed_noise
    noise = seed_noise([1.00, 1.02, 0.98])
    assert noise["n"] == 3 and noise["mean"] == pytest.approx(1.0) and noise["std"] == pytest.approx(0.02)
    assert noise["threshold"] == pytest.approx(2 * 2 ** 0.5 * 0.02)
    assert seed_noise([1.0]) is None
    result = lambda i, bpb, test: {"run_id": i, "val_bpb": bpb, "test_bpb": test, "training_tokens_per_second": 100.0,
                                   "parameters": 10, "scaling_parameters": 5}
    results = {f"d8-b0-s{i}": result(f"d8-b0-s{i}", 1.0, t) for i, t in zip((1, 2, 3), (1.00, 1.02, 0.98))}
    reference = pooled(results, "d8-b0")
    assert reference["test_bpb"] == pytest.approx(1.0) and len(reference["members"]) == 3
    assert pooled({"d8-b0-s1": results["d8-b0-s1"]}, "d8-b0") is None  # incomplete group is not a baseline
    small = pair_entry(result("d8-gqa2", 1.0, 1.04), reference, noise["threshold"])
    big = pair_entry(result("d8-gqa2", 1.0, 1.07), reference, noise["threshold"])
    assert small["resolved"] is False and big["resolved"] is True
    assert big["test_bpb_delta"] == pytest.approx(0.07)
    assert pair_entry(result("d8-gqa2", 1.0, 1.07), reference, None)["resolved"] is None
