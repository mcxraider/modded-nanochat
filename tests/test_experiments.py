import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.config import GROUPS, RUNS, members, model_for, read_config, schedule, validate_suite
from experiments import make_configs, run, run_all
from experiments.report import compare, matched_observations
from experiments.resume import pilot_tokens_per_second, project_seconds, resume_action, spent_seconds, truncate_metrics

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
    assert len(suite) == 14
    assert {p.stem for p in config_dir.glob("*.json")} == set(RUNS)
    assert [c["run_id"] for c in suite] == list(RUNS)
    validate_suite(suite)
    for config in suite:
        t, m = config["training"], config["model"]
        micro = config["workers"] * t["device_batch_size"] * m["sequence_len"]
        assert t["total_batch_size"] % micro == 0 and t["target_tokens"] % t["total_batch_size"] == 0
        assert config["model"]["n_embd"] // config["model"]["n_head"] == 128
    assert {RUNS[r]["depth"] for r in RUNS} == {8}


def test_flop_budget_is_b0s_usage_and_iso_runs_fill_it(config_dir):
    suite = {c["run_id"]: c for c in configs(config_dir)}
    budget = suite[FIRST]["training"]["flop_budget"]
    assert {c["training"]["flop_budget"] for c in suite.values()} == {budget}
    assert budget == suite[FIRST]["training"]["target_tokens"] * make_configs.flops_per_token(FIRST)
    for run_id, c in suite.items():
        fpt, target, batch = make_configs.flops_per_token(run_id), c["training"]["target_tokens"], c["training"]["total_batch_size"]
        assert target * fpt <= budget  # no run exceeds the declared budget
        if RUNS[run_id]["budget"] == "flops":
            assert (target + batch) * fpt > budget  # the most whole updates that fit
            assert target * fpt > 0.999 * budget
        else:
            assert target == suite[FIRST]["training"]["target_tokens"]
    assert {r: suite[r]["training"]["target_tokens"] // 131072 for r in ("d8-gqa2-iso", "d8-gqa4-iso")} == {
        "d8-gqa2-iso": 3333, "d8-gqa4-iso": 3404}


def test_suite_allows_token_target_drift_only_on_flop_budget_runs(config_dir):
    suite = configs(config_dir)
    index = {c["run_id"]: i for i, c in enumerate(suite)}
    drifted = copy.deepcopy(suite)
    drifted[index["d8-gqa2-r75"]]["training"]["target_tokens"] += 131072
    with pytest.raises(ValueError, match="may differ only"):
        validate_suite(drifted)
    budget = copy.deepcopy(suite)
    budget[index["d8-gqa2-iso"]]["training"]["flop_budget"] += 1
    with pytest.raises(ValueError, match="may differ only"):
        validate_suite(budget)
    iso = copy.deepcopy(suite)
    iso[index["d8-gqa2-iso"]]["training"]["target_tokens"] = suite[index[FIRST]]["training"]["target_tokens"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_suite(iso)  # identical to d8-gqa2-s1 once its target matches


def test_value_embedding_runs_form_a_2x2_with_b0_and_gqa2():
    from experiments.config import INTERACTIONS
    a, b, c, d = INTERACTIONS["E5"]
    assert {model_for(members(x)[0])["value_embeds"] for x in (a, b)} == {True}
    assert {model_for(x)["value_embeds"] for x in (c, d)} == {False}
    for on, off in ((members(a)[0], c), (members(b)[0], d)):
        assert {**model_for(on), "value_embeds": False} == model_for(off)


def test_runs_table_references_resolve_and_encode_the_experiment_design():
    for run_id, spec in RUNS.items():
        for ref in spec["compare_to"]:
            assert members(ref)
    assert members("d8-b0") == GROUPS["d8-b0"] and len(GROUPS["d8-b0"]) == 3
    assert [model_for(r)["n_kv_head"] for r in ("d8-b0-s1", "d8-gqa2-s1", "d8-gqa4")] == [4, 2, 1]
    assert model_for("d8-gqa2-pm")["mlp_ratio"] == 4.5 and model_for("d8-gqa2-r25")["rotary_fraction"] == 0.25
    assert sorted(RUNS[r]["seed"] for r in GROUPS["d8-b0"]) == [1, 2, 3]
    # GQA-2 seeds: one architecture, seeds 1-3; seeds share B0's data rotations.
    assert sorted(RUNS[r]["seed"] for r in GROUPS["d8-gqa2"]) == [1, 2, 3]
    assert {json.dumps(model_for(r), sort_keys=True) for r in GROUPS["d8-gqa2"]} == {json.dumps(model_for("d8-gqa2-s1"), sort_keys=True)}


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


def write_checkpoint(directory, step, workers=1, skip=()):
    directory.mkdir(parents=True, exist_ok=True)
    from experiments.checkpoints import filenames, digest
    names = filenames(step, workers)
    for name in names:
        if name not in skip:
            (directory / name).write_text("{}")
    if f"timing_{step:06d}.json" not in skip:
        checksums = {name: digest(directory / name) for name in names if (directory / name).exists()}
        (directory / f"timing_{step:06d}.json").write_text(json.dumps({"sha256": checksums, "training_seconds": 1}))


def test_resume_action_picks_newest_complete_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    assert resume_action(run_dir, 1) == ("fresh", None)
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_text("")
    assert resume_action(run_dir, 1) == ("fresh", None)  # partial directory, no checkpoint
    write_checkpoint(run_dir / "checkpoints", 1600)
    assert resume_action(run_dir, 1) == ("resume", 1600)
    write_checkpoint(run_dir / "checkpoints", 3200, skip=("optim_003200_rank0.pt",))
    assert resume_action(run_dir, 1) == ("resume", 1600)  # newest one lost a file
    (run_dir / "checkpoints/timing_003200.json").unlink()
    write_checkpoint(run_dir / "checkpoints", 3200, skip=("timing_003200.json",))
    assert resume_action(run_dir, 1) == ("resume", 1600)  # no timing means the save never finished
    assert resume_action(run_dir, 2) == ("fresh", None)  # rank 1 files are required with two workers
    (run_dir / "inference.json").write_text("{}")
    assert resume_action(run_dir, 1) == ("skip", None)


def test_truncate_metrics_keeps_only_rows_the_resume_will_not_repeat(tmp_path):
    path = tmp_path / "metrics.jsonl"
    rows = [{"kind": "train", "step": 1}, {"kind": "validation", "step": 2}, {"kind": "train", "step": 3},
            {"kind": "validation", "step": 2, "final": True}, {"kind": "test", "step": 2, "final": True}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"kind": "tra')
    truncate_metrics(path, 2)
    assert [json.loads(line) for line in path.read_text().splitlines()] == rows[:2]
    truncate_metrics(tmp_path / "absent.jsonl", 2)


def test_budget_helpers(tmp_path, config_dir):
    for name, seconds in (("a", 100), ("b", 50)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "rental_accounting.json").write_text(json.dumps({"total_pipeline_node_seconds": seconds}))
    (tmp_path / "c").mkdir()  # started, nothing recorded yet
    assert spent_seconds([tmp_path, tmp_path / "absent"]) == 150

    metrics = tmp_path / "metrics.jsonl"
    rows = [{"kind": "train", "step": s, "tokens_per_second": v} for s, v in ((1, 1), (2, 2), (3, 100), (4, 200), (5, 300))]
    metrics.write_text("".join(json.dumps(r) + "\n" for r in [*rows, {"kind": "test", "step": 5}]))
    assert pilot_tokens_per_second(metrics) == 200  # compilation steps excluded

    config = read_config(config_dir / f"{FIRST}.json")
    t, e = config["training"], config["evaluation"]
    projected = project_seconds(config, 1000)
    assert projected["training"] == pytest.approx((t["target_tokens"] + 7 * e["interim_tokens"]) / 1000)  # 3200 updates, every 400, none on the last
    assert projected["total"] > projected["training"] + (e["final_tokens"] + e["test_tokens"]) / 1000


def test_runtime_loader_patch():
    source = subprocess.check_output(["git", "show", "324e69c:nanochat/checkpoint_manager.py"], cwd=ROOT, text=True)
    patched = run.patch_checkpoint_loader(source)
    compile(patched, "checkpoint_manager.py", "exec")
    assert "model = model_class(model_config)" in patched
    assert "modified_gpt_324e69c" in patched


def test_report_arithmetic_and_no_interpolated_observations():
    reference = {"run_id": "B0", "val_bpb": 2, "training_tokens_per_second": 100, "steady_tokens_per_second": 200}
    result = {"val_bpb": 1.5, "training_tokens_per_second": 150, "steady_tokens_per_second": 220}
    assert compare(result, reference) == {"reference": "B0", "bpb_delta": -0.5, "bpb_relative_delta": -0.25,
                                          "training_speedup": pytest.approx(1.1), "whole_phase_training_speedup": 1.5}
    assert compare({**result, "steady_tokens_per_second": None}, reference)["training_speedup"] is None
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
        assert kwargs["watch"].name in ("train.log", "benchmark.log") and kwargs["stall_seconds"] == run.STALL_SECONDS
    monkeypatch.setattr(run, "run_process", execute)
    run.launch(config_dir / f"{FIRST}.json", tmp_path / "artifacts", runtime)
    assert "--nproc_per_node=1" in calls[0][0]
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
    monkeypatch.setattr(run, "run_process", execute)
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
    small = pair_entry(result("d8-gqa2-s1", 1.0, 1.04), reference, noise["threshold"])
    big = pair_entry(result("d8-gqa2-s1", 1.0, 1.07), reference, noise["threshold"])
    assert small["resolved"] is False and big["resolved"] is True
    assert big["test_bpb_delta"] == pytest.approx(0.07)
    assert pair_entry(result("d8-gqa2-s1", 1.0, 1.07), reference, None)["resolved"] is None


def test_saved_outputs_reject_changed_config_source_and_missing_provenance(tmp_path):
    from experiments.resume import validate_saved_run
    cfg, overlay = {"workers": 1}, {"experiments/train.py": "original"}
    validate_saved_run(tmp_path, cfg, overlay)  # no previous run
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "launch.json").write_text(json.dumps({"overlay_sha256": overlay}))
    (tmp_path / "inference.json").write_text("{}")
    validate_saved_run(tmp_path, cfg, overlay)
    with pytest.raises(ValueError, match="config.json"):
        validate_saved_run(tmp_path, {"workers": 2}, overlay)
    with pytest.raises(ValueError, match="launch.json"):
        validate_saved_run(tmp_path, cfg, {"experiments/train.py": "changed"})
    (tmp_path / "launch.json").unlink()
    with pytest.raises(ValueError, match="missing provenance"):
        validate_saved_run(tmp_path, cfg, overlay)


def test_shared_writer_rejects_another_app_and_releases_after_failure():
    from experiments.resume import exclusive_writer
    class SharedDict(dict):
        def put(self, key, value, *, skip_if_exists):
            assert skip_if_exists
            if key in self:
                return False
            self[key] = value
            return True
    locks = SharedDict()
    with pytest.raises(ValueError, match="worker failed"):
        with exclusive_writer(locks):
            with pytest.raises(RuntimeError, match="another pipeline"):
                with exclusive_writer(locks):
                    pytest.fail("second writer acquired lock")
            assert "writer" in locks
            raise ValueError("worker failed")
    assert not locks
    with exclusive_writer(locks):
        pass


def test_budget_reservation_survives_a_kill_and_deleted_outputs(tmp_path, monkeypatch):
    from experiments import resume
    clock = [100.0]
    monkeypatch.setattr(resume.time, "monotonic", lambda: clock[0])
    ledger = tmp_path / "accounting/suite.json"
    commits = []
    attempt = resume.BudgetSession(ledger, 40, 4, 2, lambda: commits.append(True))
    attempt.__enter__()  # simulate a hard kill: __exit__ is never reached
    assert len(commits) == 1
    assert json.loads(ledger.read_text())["reserved_usd"] == 38
    clock[0] += 3600
    assert attempt.spent() == 6
    assert attempt.seconds_left() == 8.5 * 3600
    # A fresh output tree / lower legacy count cannot erase the reservation.
    with pytest.raises(RuntimeError, match="budget exhausted or reserved"):
        with resume.BudgetSession(ledger, 40, 4, 0, lambda: None):
            pass


def test_budget_settles_elapsed_time_and_counts_retries(tmp_path, monkeypatch):
    from experiments import resume
    clock = [0.0]
    monkeypatch.setattr(resume.time, "monotonic", lambda: clock[0])
    ledger = tmp_path / "accounting/suite.json"
    with pytest.raises(ValueError):
        with resume.BudgetSession(ledger, 40, 4, 2, lambda: None):
            clock[0] += 1800
            raise ValueError("training failed")
    assert json.loads(ledger.read_text()) == {"spent_usd": 4, "reserved_usd": 0}
    with resume.BudgetSession(ledger, 40, 4, 0, lambda: None) as budget:
        assert budget.spent() == 4
        clock[0] += 900
    assert json.loads(ledger.read_text())["spent_usd"] == 5
    for invalid in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            resume.BudgetSession(ledger, invalid, 4, 0, lambda: None)


def test_budget_timeout_kills_the_entire_training_process_group(monkeypatch):
    import signal
    from unittest.mock import MagicMock
    process = MagicMock()
    process.pid = 12345
    process.wait.side_effect = [subprocess.TimeoutExpired(["torchrun"], 1), -9]
    process.__enter__.return_value = process
    popen = MagicMock(return_value=process)
    monkeypatch.setattr(subprocess, "Popen", popen)
    killed = []
    monkeypatch.setattr(run.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    with pytest.raises(subprocess.TimeoutExpired):
        run.run_process(["torchrun"], timeout=1)
    assert killed == [(12345, signal.SIGKILL)]
    popen.assert_called_once_with(["torchrun"], start_new_session=True)
    popen.reset_mock()
    with pytest.raises(subprocess.TimeoutExpired):
        run.run_process(["torchrun"], timeout=0)
    popen.assert_not_called()


def test_stall_watchdog_kills_silent_process_as_a_failure(tmp_path):
    import sys, time
    log = tmp_path / "train.log"
    log.write_text("")
    started = time.monotonic()
    with pytest.raises(subprocess.CalledProcessError):
        run.run_process([sys.executable, "-c", "import time; time.sleep(60)"], timeout=None, watch=log, stall_seconds=1)
    assert time.monotonic() - started < 10
    # A process that keeps writing is not a stall.
    with log.open("a") as stream:
        run.run_process([sys.executable, "-c", "import time\nfor _ in range(4): print(1, flush=True); time.sleep(0.4)"],
                        timeout=30, watch=log, stall_seconds=1, stdout=stream)


def test_writer_lock_is_reclaimed_by_same_owner_only():
    from experiments.resume import exclusive_writer
    class SharedDict(dict):
        def put(self, key, value, *, skip_if_exists):
            if key in self:
                return False
            self[key] = value
            return True
    locks = SharedDict(writer="call-1")  # stale lock left by a preempted attempt
    with pytest.raises(RuntimeError, match="another pipeline"):
        with exclusive_writer(locks, owner="call-2"):
            pass
    with exclusive_writer(locks, owner="call-1"):
        assert locks["writer"] == "call-1"
    assert not locks


def test_budget_retry_by_same_owner_settles_its_own_reservation(tmp_path, monkeypatch):
    from experiments import resume
    wall = [1000.0]
    monkeypatch.setattr(resume.time, "time", lambda: wall[0])
    ledger = tmp_path / "accounting/suite.json"
    resume.BudgetSession(ledger, 40, 4, 2, lambda: None, owner="call-1").__enter__()  # hard kill
    wall[0] += 3600
    with pytest.raises(RuntimeError, match="budget exhausted or reserved"):
        resume.BudgetSession(ledger, 40, 4, 0, lambda: None, owner="call-2").__enter__()
    retry = resume.BudgetSession(ledger, 40, 4, 0, lambda: None, owner="call-1")
    assert retry.before == 6  # 2 earlier + one charged hour
    assert retry.remaining_usd == 34
    wall[0] += 100 * 3600  # never charges more than was reserved
    assert resume.BudgetSession(ledger, 40, 4, 0, lambda: None, owner="call-1").before == 40
