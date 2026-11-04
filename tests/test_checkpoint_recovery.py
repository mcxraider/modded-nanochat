"""Failure injection for persisted training and benchmark recovery."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from experiments import checkpoints, resume, run
from experiments.make_configs import build


def stage(root, step):
    staging = root / f'.pending_{step}'
    staging.mkdir(parents=True)
    for name in checkpoints.filenames(step, 1):
        (staging / name).write_bytes(b'complete checkpoint payload')
    return staging


def save(root, step):
    checkpoints.publish(stage(root, step), root, step, 1, {'training_seconds': 123})


def test_interrupted_publication_falls_back_to_previous_generation(tmp_path, monkeypatch):
    root = tmp_path / 'checkpoints'
    save(root, 1600)
    staging = stage(root, 3200)
    replace = Path.replace
    def fail_optimizer(path, target):
        if path.name == 'optim_003200_rank0.pt':
            raise OSError('simulated interruption')
        return replace(path, target)
    monkeypatch.setattr(Path, 'replace', fail_optimizer)
    with pytest.raises(OSError):
        checkpoints.publish(staging, root, 3200, 1, {'training_seconds': 456})
    assert resume.resume_action(tmp_path, 1) == ('resume', 1600)
    assert not (root / 'timing_003200.json').exists()


def test_corruption_is_rejected_and_can_be_replaced(tmp_path):
    root = tmp_path / 'checkpoints'
    save(root, 1600)
    save(root, 3200)
    (root / 'model_003200.pt').write_bytes(b'truncated')
    assert resume.resume_action(tmp_path, 1) == ('resume', 1600)
    save(root, 3200)
    assert resume.resume_action(tmp_path, 1) == ('resume', 3200)


def test_completed_checkpoint_cannot_be_overwritten(tmp_path):
    root = tmp_path / 'checkpoints'
    save(root, 3200)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    with pytest.raises(FileExistsError, match='already published'):
        save(root, 3200)
    assert all((root / name).read_bytes() == data for name, data in before.items())


def test_benchmark_recovery_does_not_launch_training_or_touch_metrics(tmp_path, monkeypatch):
    cfg = build('d8-b0-s1')
    output = tmp_path / cfg['run_id']
    save(output / 'checkpoints', 3200)
    (output / 'config.json').write_text(json.dumps(cfg))
    (output / 'launch.json').write_text(json.dumps({'overlay_sha256': {}}))
    (output / 'result.json').write_text(json.dumps({'checkpoint_step': 3200}))
    metrics = output / 'metrics.jsonl'
    metrics.write_text('final evaluation evidence\n')
    (output / 'source').mkdir()
    monkeypatch.setattr(run, 'prepare_runtime', lambda _: (tmp_path, {}))
    process = MagicMock()
    monkeypatch.setattr(run, 'run_process', process)
    assert resume.resume_action(output, 1) == ('benchmark', 3200)
    run.launch(output / 'config.json', tmp_path, tmp_path / 'runtime', resume=3200)
    assert process.call_count == 1
    assert process.call_args.args[0][1:3] == ['-m', 'experiments.benchmark']
    assert metrics.read_text() == 'final evaluation evidence\n'


def test_prune_keeps_two_newest_and_deletes_marker_first(tmp_path, monkeypatch):
    root = tmp_path / 'checkpoints'
    for step in (400, 800, 1200):
        save(root, step)
    checkpoints.prune(root, {800, 1200}, 1)
    assert sorted(p.name for p in root.glob('timing_*.json')) == ['timing_000800.json', 'timing_001200.json']
    assert not any(root.glob('*_000400*'))
    assert resume.resume_action(tmp_path, 1) == ('resume', 1200)
    # Interrupted right after the marker goes: the generation is invalid, never half-valid.
    unlink = Path.unlink
    def fail_payload(path, missing_ok=False):
        if path.name.startswith('model_'):
            raise OSError('simulated interruption')
        return unlink(path, missing_ok=missing_ok)
    monkeypatch.setattr(Path, 'unlink', fail_payload)
    with pytest.raises(OSError):
        checkpoints.prune(root, {1200}, 1)
    assert not checkpoints.valid(root, 800, 1)
    assert resume.resume_action(tmp_path, 1) == ('resume', 1200)
