"""Exercise Modal orchestration locally without provisioning remote resources."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from experiments import config, resume, run


@pytest.fixture
def modal_app(monkeypatch):
    decorator = lambda **kwargs: lambda function: function
    sdk = MagicMock()
    sdk.App.return_value = SimpleNamespace(function=decorator, local_entrypoint=decorator)
    monkeypatch.setitem(sys.modules, "modal", sdk)
    spec = importlib.util.spec_from_file_location("modal_app_under_test", Path(__file__).resolve().parents[1] / "modal_app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "experiments_modules", lambda: (config, resume, run))
    return module


def test_run_one_rejects_stale_completed_run_before_skip_or_delete(modal_app, tmp_path, monkeypatch):
    cfg = config.read_config(Path(__file__).resolve().parents[1] / "experiments/configs/d8-b0-s1.json")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))
    directory = tmp_path / cfg["run_id"]
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(cfg))
    (directory / "launch.json").write_text(json.dumps({"overlay_sha256": {"source": "old"}}))
    (directory / "inference.json").write_text("{}")
    monkeypatch.setattr(run, "prepare_runtime", lambda _: (tmp_path, {"source": "new"}))
    launch = MagicMock()
    monkeypatch.setattr(run, "launch", launch)
    with pytest.raises(ValueError, match="launch.json"):
        modal_app.run_one(config_path, tmp_path, MagicMock())
    assert (directory / "inference.json").exists()
    launch.assert_not_called()


def test_duplicate_suite_is_rejected_before_volume_access(modal_app):
    modal_app.writer_locks.put.return_value = False
    result = modal_app.suite()  # returned, not raised: Modal must not retry a held lock
    assert result["status"] == "error" and "another pipeline" in result["error"]
    modal_app.volume.reload.assert_not_called()
    modal_app.volume.commit.assert_not_called()


def test_prepare_commits_before_releasing_shared_lock(modal_app, monkeypatch):
    events = []
    modal_app.writer_locks.put.return_value = True
    modal_app.writer_locks.pop.side_effect = lambda _: events.append("unlock")
    modal_app.volume.reload.side_effect = lambda: events.append("reload")
    modal_app.volume.commit.side_effect = lambda: events.append("commit")
    def fail():
        events.append("prepare")
        raise ValueError("download failed")
    monkeypatch.setattr(modal_app, "prepare_data", fail)
    with pytest.raises(ValueError, match="download failed"):
        modal_app.prepare()
    assert events == ["reload", "prepare", "commit", "unlock"]


def test_d8_pilots_project_every_run(modal_app, tmp_path, monkeypatch):
    config_dir = Path(__file__).resolve().parents[1] / "experiments/configs"
    assert len(modal_app.PILOT_RUNS) == 8
    assert modal_app.BUDGET_USD > 0
    monkeypatch.setattr(modal_app, "suite_config", lambda r: config_dir / f"{r}.json")
    monkeypatch.setattr(modal_app, "pilot_config", lambda r, root: r)
    monkeypatch.setattr(modal_app, "PILOTS", tmp_path / "pilots")
    monkeypatch.setattr(modal_app, "ARTIFACTS", tmp_path / "artifacts")
    launches = MagicMock()
    monkeypatch.setattr(modal_app, "run_one", launches)
    monkeypatch.setattr(modal_app, "check_pilot", lambda *_: [])
    monkeypatch.setattr(resume, "pilot_tokens_per_second", lambda _: 1_000_000)
    monkeypatch.setattr(run, "prepare_runtime", lambda _: (tmp_path, {}))
    monkeypatch.setattr(run, "run_process", MagicMock())
    monkeypatch.setattr(modal_app, "read_run_json", lambda directory, name: {
        "evaluation_seconds": 1.0, "elapsed_seconds": 10.0})
    budget = SimpleNamespace(spent=lambda: 0.5, seconds_left=lambda: 1000)
    result = modal_app.run_suite(40.0, True, budget)
    assert result["status"] == "pilot_ok"
    assert set(result["runs"]) == set(config.RUNS) and len(result["runs"]) == 14
    assert [call.args[0] for call in launches.call_args_list] == list(modal_app.PILOT_RUNS)


def test_benchmark_only_resume_preserves_final_metrics(modal_app, tmp_path, monkeypatch):
    cfg_path = Path(__file__).resolve().parents[1] / 'experiments/configs/d8-b0-s1.json'
    directory = tmp_path / 'd8-b0-s1'
    directory.mkdir()
    metrics = directory / 'metrics.jsonl'
    metrics.write_text('{"kind":"test","final":true,"step":3200}\n')
    monkeypatch.setattr(run, 'prepare_runtime', lambda _: (tmp_path, {}))
    monkeypatch.setattr(resume, 'resume_action', lambda *_: ('benchmark', 3200))
    launch = MagicMock()
    monkeypatch.setattr(run, 'launch', launch)
    modal_app.run_one(cfg_path, tmp_path, SimpleNamespace(seconds_left=lambda: 60))
    assert '"final":true' in metrics.read_text()
    assert launch.call_args.kwargs['resume'] == 3200


def test_pilot_preserves_all_full_run_inference_shapes(modal_app, tmp_path, monkeypatch):
    config_dir = Path(__file__).resolve().parents[1] / 'experiments/configs'
    monkeypatch.setattr(modal_app, 'suite_config', lambda r: config_dir / f'{r}.json')
    full = config.read_config(config_dir / 'd8-gqa2-pm.json')
    pilot = config.read_config(modal_app.pilot_config('d8-gqa2-pm', tmp_path))
    for key in ('batch_sizes', 'prompt_lengths', 'decode_tokens'):
        assert pilot['benchmark'][key] == full['benchmark'][key]
    assert pilot['training']['device_batch_size'] == full['training']['device_batch_size']


def test_recovery_requires_stopped_workers_and_preserves_audit(modal_app, tmp_path, monkeypatch):
    monkeypatch.setattr(modal_app, 'VOLUME_PATH', tmp_path)
    with pytest.raises(ValueError, match='stop all'):
        modal_app.recover(3)
    modal_app.volume.reload.assert_not_called()
    for value in (-1, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='nonnegative'):
            modal_app.recover(value, True)
    directory = tmp_path / 'accounting'
    directory.mkdir()
    previous = {'spent_usd': 40, 'reserved_usd': 38}
    (directory / 'suite.json').write_text(json.dumps(previous))
    modal_app.writer_locks.get.return_value = 'dead-owner'
    result = modal_app.recover(5, True)
    assert result['spent_usd'] == 5
    assert json.loads((directory / 'suite.json').read_text()) == {'spent_usd': 5, 'reserved_usd': 0}
    audit = json.loads(next(directory.glob('recovery_*.json')).read_text())
    assert audit['previous'] == previous
    assert audit['previous_writer'] == 'dead-owner'
    modal_app.writer_locks.pop.assert_called_once_with('writer')


def test_pilot_rejects_any_inference_oom(modal_app, tmp_path):
    (tmp_path / 'hardware.json').write_text(json.dumps({'gpu': 'NVIDIA H100 80GB HBM3', 'world_size': 1, 'attention_backend': 'FA3'}))
    (tmp_path / 'result.json').write_text(json.dumps({'peak_vram_bytes': 1}))
    ok = dict(batch_size=1, prompt_tokens=128, status='ok', kv_cache_bytes_allocated=8, kv_cache_bytes_analytic=8)
    inference = {'attention_backend': 'FA3', 'rows': [ok]}
    (tmp_path / 'inference.json').write_text(json.dumps(inference))
    assert modal_app.check_pilot(tmp_path, 1) == []
    inference['rows'].append(dict(batch_size=64, prompt_tokens=8192, status='oom'))
    (tmp_path / 'inference.json').write_text(json.dumps(inference))
    assert any('bad inference row' in p for p in modal_app.check_pilot(tmp_path, 1))


def test_full_run_process_failure_stops_suite_without_retry(modal_app, tmp_path, monkeypatch):
    import subprocess
    config_dir = Path(__file__).resolve().parents[1] / "experiments/configs"
    monkeypatch.setattr(modal_app, "suite_config", lambda r: config_dir / f"{r}.json")
    monkeypatch.setattr(modal_app, "pilot_config", lambda r, root: r)
    monkeypatch.setattr(modal_app, "PILOTS", tmp_path / "pilots")
    monkeypatch.setattr(modal_app, "ARTIFACTS", tmp_path / "artifacts")
    def launch(target, *_):
        if isinstance(target, Path):  # full run, not a pilot
            raise subprocess.CalledProcessError(1, ["torchrun"])
    launches = MagicMock(side_effect=launch)
    monkeypatch.setattr(modal_app, "run_one", launches)
    monkeypatch.setattr(modal_app, "check_pilot", lambda *_: [])
    monkeypatch.setattr(resume, "pilot_tokens_per_second", lambda _: 1_000_000)
    monkeypatch.setattr(run, "prepare_runtime", lambda _: (tmp_path, {}))
    monkeypatch.setattr(run, "run_process", MagicMock())
    monkeypatch.setattr(modal_app, "read_run_json", lambda directory, name: {
        "evaluation_seconds": 1.0, "elapsed_seconds": 10.0})
    budget = SimpleNamespace(spent=lambda: 0.5, seconds_left=lambda: 1000)
    result = modal_app.run_suite(40.0, False, budget)
    assert result["status"] == "run_failed"
    assert result["run_id"] == list(config.RUNS)[0]
    assert launches.call_count == len(modal_app.PILOT_RUNS) + 1


def suite_harness(modal_app, tmp_path, monkeypatch, done=()):
    """run_suite with pilots stubbed; `done` runs already have inference.json (skipped)."""
    config_dir = Path(__file__).resolve().parents[1] / "experiments/configs"
    monkeypatch.setattr(modal_app, "suite_config", lambda r: config_dir / f"{r}.json")
    monkeypatch.setattr(modal_app, "pilot_config", lambda r, root: r)
    monkeypatch.setattr(modal_app, "PILOTS", tmp_path / "pilots")
    monkeypatch.setattr(modal_app, "ARTIFACTS", tmp_path / "artifacts")
    for r in done:
        (tmp_path / "artifacts" / r).mkdir(parents=True)
        (tmp_path / "artifacts" / r / "inference.json").write_text("{}")
    monkeypatch.setattr(modal_app, "VOLUME_PATH", tmp_path)
    monkeypatch.setattr(resume, "validate_saved_run", lambda *_: None)
    monkeypatch.setattr(run, "run_process", MagicMock())  # preflight
    launches = MagicMock()
    monkeypatch.setattr(modal_app, "run_one", launches)
    finishes = MagicMock()
    monkeypatch.setattr(modal_app, "finish", finishes)
    monkeypatch.setattr(modal_app, "check_pilot", lambda *_: [])
    speeds = {p: 1_000_000 + i for i, p in enumerate(modal_app.PILOT_RUNS)}
    monkeypatch.setattr(resume, "pilot_tokens_per_second", lambda path: speeds[Path(path).parent.name.removeprefix("pilot-")])
    monkeypatch.setattr(run, "prepare_runtime", lambda _: (tmp_path, {}))
    monkeypatch.setattr(modal_app, "read_run_json", lambda directory, name: {
        "evaluation_seconds": 1.0, "elapsed_seconds": 10.0, "stop_reason": "token_target"})
    return launches, finishes, speeds


def full_runs(launches, done=()):
    """Full-run launches of unfinished runs (run_one itself skips finished runs)."""
    return [Path(c.args[0]).stem for c in launches.call_args_list
            if isinstance(c.args[0], Path) and Path(c.args[0]).stem not in done]


def test_suite_runs_every_run_then_reports_once(modal_app, tmp_path, monkeypatch):
    launches, finishes, _ = suite_harness(modal_app, tmp_path, monkeypatch)
    result = modal_app.run_suite(40.0, False, SimpleNamespace(spent=lambda: 0.5, seconds_left=lambda: 1e6))
    assert result["status"] == "complete"
    assert full_runs(launches) == list(config.RUNS) and len(config.RUNS) == 14
    assert finishes.call_count == 1


def test_gqa2_seeds_count_toward_the_budget_gate(modal_app, tmp_path, monkeypatch):
    launches, finishes, _ = suite_harness(modal_app, tmp_path, monkeypatch)
    # Each run projects to ~$1.3 here: $17 covers 12 runs but the gate projects all 14 ($18.5), seeds included.
    result = modal_app.run_suite(17.0, False, SimpleNamespace(spent=lambda: 0.0, seconds_left=lambda: 1e6))
    assert result["status"] == "stopped_at_gate"
    assert result["projected_total_usd"] == pytest.approx(18.46, abs=0.05)
    assert full_runs(launches) == [] and finishes.call_count == 0


def test_gqa2_seeds_are_projected_from_the_gqa2_pilot(modal_app):
    assert modal_app.pilot_for("d8-gqa2-s3") == "d8-gqa2-s1"
    assert modal_app.pilot_for("d8-b0-s2") == "d8-b0-s1"
    assert modal_app.pilot_for("d8-gqa2-pm") == "d8-gqa2-pm"


def test_suite_returns_deterministic_errors_after_settling_budget_and_lock(modal_app, tmp_path, monkeypatch):
    monkeypatch.setattr(modal_app, "VOLUME_PATH", tmp_path)
    modal_app.writer_locks.put.return_value = True
    def crash(*_):
        raise KeyError("evaluation_seconds")
    monkeypatch.setattr(modal_app, "run_suite", crash)
    result = modal_app.suite(10.0)
    assert result["status"] == "error" and "evaluation_seconds" in result["error"]
    assert json.loads((tmp_path / "accounting/suite.json").read_text())["reserved_usd"] == 0
    modal_app.writer_locks.pop.assert_called_once_with("writer")


def test_suite_stops_on_non_comparable_run_even_if_already_complete(modal_app, tmp_path, monkeypatch):
    first = list(config.RUNS)[0]
    launches, finishes, _ = suite_harness(modal_app, tmp_path, monkeypatch, done=(first,))
    monkeypatch.setattr(modal_app, "read_run_json", lambda directory, name: {
        "evaluation_seconds": 1.0, "elapsed_seconds": 10.0,
        "stop_reason": "wallclock_cap" if Path(directory).name == first else "token_target"})
    result = modal_app.run_suite(40.0, False, SimpleNamespace(spent=lambda: 0.5, seconds_left=lambda: 1e6))
    assert result["status"] == "run_not_comparable" and result["run_id"] == first
    assert result["stop_reason"] == "wallclock_cap"
    assert full_runs(launches, (first,)) == [] and finishes.call_count == 0


def test_slow_run_hits_its_own_deadline_not_the_budget(modal_app, tmp_path, monkeypatch):
    import subprocess
    launches, finishes, _ = suite_harness(modal_app, tmp_path, monkeypatch)
    def launch(target, *args):
        if isinstance(target, Path):
            raise subprocess.TimeoutExpired(["torchrun"], args[-1])
    launches.side_effect = launch
    result = modal_app.run_suite(40.0, False, SimpleNamespace(spent=lambda: 0.5, seconds_left=lambda: 1e6))
    assert result["status"] == "run_overtime" and result["run_id"] == list(config.RUNS)[0]
    assert result["limit_seconds"] == launches.call_args.args[3] < 1e6
    # When the budget is the binding deadline, suite() reports stopped_for_budget instead.
    with pytest.raises(subprocess.TimeoutExpired):
        modal_app.run_suite(40.0, False, SimpleNamespace(spent=lambda: 0.5, seconds_left=lambda: 1.0))


def test_run_one_deadline_is_the_tighter_of_run_limit_and_budget(modal_app, tmp_path, monkeypatch):
    cfg_path = Path(__file__).resolve().parents[1] / "experiments/configs/d8-b0-s1.json"
    monkeypatch.setattr(run, "prepare_runtime", lambda _: (tmp_path, {}))
    launch = MagicMock()
    monkeypatch.setattr(run, "launch", launch)
    for limit, left, expected in ((10, 60, 10), (100, 60, 60), (None, 60, 60)):
        modal_app.run_one(cfg_path, tmp_path, SimpleNamespace(seconds_left=lambda: left), limit)
        assert launch.call_args.kwargs["timeout_seconds"] == expected


def test_provenance_covers_every_experiments_module_training_imports():
    import ast
    root = Path(__file__).resolve().parents[1]
    seen, todo = set(), ["train", "benchmark"]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        for node in ast.walk(ast.parse((root / f"experiments/{name}.py").read_text())):
            if isinstance(node, ast.ImportFrom) and node.module == "experiments":
                todo += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("experiments."):
                todo.append(node.module.split(".")[1])
            elif isinstance(node, ast.Import):
                todo += [a.name.split(".")[1] for a in node.names if a.name.startswith("experiments.")]
    assert {f"experiments/{name}.py" for name in seen} <= set(run.RESULT_FILES)
    assert seen >= {"train", "benchmark", "data", "config", "checkpoints", "backend"}


def test_suite_reraises_transient_modal_errors_for_retry(modal_app, tmp_path, monkeypatch):
    monkeypatch.setattr(modal_app, "VOLUME_PATH", tmp_path)
    modal_app.writer_locks.put.return_value = True
    Transient = type("ConnectionError", (Exception,), {"__module__": "modal.exception"})
    def commit():
        raise Transient("volume commit failed")
    modal_app.volume.commit.side_effect = commit  # fails inside BudgetSession.__enter__
    with pytest.raises(Transient):
        modal_app.suite(10.0)
    modal_app.writer_locks.pop.assert_called_once_with("writer")
    assert modal_app.transient(Transient()) and not modal_app.transient(KeyError("x"))
