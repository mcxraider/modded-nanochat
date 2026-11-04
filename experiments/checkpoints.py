"""Publish checkpoint files before a checksum-verified completion marker."""
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def filenames(step, workers):
    return [f'model_{step:06d}.pt', f'meta_{step:06d}.json',
            *[name for rank in range(workers) for name in
              (f'optim_{step:06d}_rank{rank}.pt', f'runtime_{step:06d}_rank{rank}.pt')]]


def publish(staging, directory, step, workers, timing):
    """Never replace a committed generation. An interrupted new save has no marker."""
    staging, directory = Path(staging), Path(directory)
    marker = directory / f'timing_{step:06d}.json'
    if marker.exists():
        if valid(directory, step, workers):
            raise FileExistsError(f'checkpoint already published: {marker}')
        marker.unlink()  # invalidate a damaged generation before replacing any file
    names = filenames(step, workers)
    checksums = {name: digest(staging / name) for name in names}
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (staging / name).replace(directory / name)
    temporary = marker.with_suffix('.json.tmp')
    temporary.write_text(json.dumps({**timing, 'sha256': checksums}, allow_nan=False) + '\n')
    temporary.replace(marker)
    staging.rmdir()


def valid(directory, step, workers):
    directory = Path(directory)
    try:
        record = json.loads((directory / f'timing_{step:06d}.json').read_text())
        checksums = record['sha256']
        names = filenames(step, workers)
        return set(checksums) == set(names) and all(
            (directory / name).stat().st_size > 0 and digest(directory / name) == checksums[name]
            for name in names)
    except (OSError, ValueError, KeyError, TypeError):
        return False
