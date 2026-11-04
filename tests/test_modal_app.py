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
    with pytest.raises(RuntimeError, match="another pipeline"):
        modal_app.suite()
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


def test_d8_pilots_project_all_eight_runs_without_d12(modal_app, tmp_path, monkeypatch):
    config_dir = Path(__file__).resolve().parents[1] / "experiments/configs"
    assert modal_app.PILOT_RUNS == ("d8-b0-s1", "d8-gqa2")
    assert modal_app.BUDGET_USD == 40.0
    monkeypatch.setattr(modal_app, "suite_config", lambda r: config_dir / f"{r}.json")
    monkeypatch.setattr(modal_app, "pilot_config", lambda r: r)
    monkeypatch.setattr(modal_app, "PILOTS", tmp_path / "pilots")
    monkeypatch.setattr(modal_app, "ARTIFACTS", tmp_path / "artifacts")
    launches = MagicMock()
    monkeypatch.setattr(modal_app, "run_one", launches)
    monkeypatch.setattr(modal_app, "check_pilot", lambda *_: [])
    monkeypatch.setattr(resume, "pilot_tokens_per_second", lambda _: 1_000_000)
    monkeypatch.setattr(run, "prepare_runtime", lambda _: (tmp_path, {}))
    budget = SimpleNamespace(spent=lambda: 0.5)
    result = modal_app.run_suite(40.0, True, budget)
    assert result["status"] == "pilot_ok"
    assert set(result["runs"]) == set(config.RUNS)
    assert len(result["runs"]) == 8
    assert [call.args[0] for call in launches.call_args_list] == list(modal_app.PILOT_RUNS)
