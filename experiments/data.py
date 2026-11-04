"""Train/validation/test shard selection on top of the unmodified historical dataloader.

The pinned `nanochat.dataloader` takes train = all shards but the last and val = the last shard. This module
adds a held-out TEST shard and a seed-dependent training order without touching that code: it temporarily
substitutes the loader's `list_parquet_files` so its own slicing picks the intended files.
"""
import itertools
import re

import torch

# One batch takes well under a second to pack; a stuck worker fails fast instead of waiting for the stall watchdog.
WORKER_TIMEOUT_SECONDS = 300
TEST_INDEX = 6541  # shard_06541.parquet; validation is the highest shard (06542, always fetched by nanochat.dataset)
_SHARD = re.compile(r"shard_(\d+)\.parquet$")


def shard_index(path):
    match = _SHARD.search(str(path))
    if not match:
        raise ValueError(f"not a ClimbMix shard file: {path}")
    return int(match.group(1))


def split_paths(paths, seed):
    """Return `(train, val, test, rotation)`.

    Train is every downloaded shard below the test shard, rotated by `(seed - 1) * (pool // 3)` so seed 1 keeps the
    historical order and other seeds see a different order and starting subset.
    """
    ordered = sorted(paths, key=shard_index)
    val = ordered[-1]
    tests = [p for p in ordered if shard_index(p) == TEST_INDEX]
    if len(tests) != 1:
        raise ValueError(f"test shard shard_{TEST_INDEX:05d}.parquet must be downloaded")
    if shard_index(val) <= TEST_INDEX:
        raise ValueError("validation must be the highest-numbered shard, above the test shard")
    pool = [p for p in ordered[:-1] if shard_index(p) < TEST_INDEX]
    if len(pool) < 3:
        raise ValueError("at least three training shards are required")
    rotation = ((seed - 1) * (len(pool) // 3)) % len(pool)
    train = pool[rotation:] + pool[:rotation]
    if len({*train, val, tests[0]}) != len(train) + 2:
        raise ValueError("train, validation, and test shards must be disjoint")
    return train, val, tests[0], rotation


def make_loader(dataloader, tokenizer, B, T, split, device, paths, seed, **kwargs):
    """Historical BOS best-fit loader over the `split` ('train', 'val' or 'test') shards.

    `dataloader` is the imported `nanochat.dataloader` module. The generator reads the shard list lazily on its
    first `next()`, so the first batch is fetched while the substitution is active and chained back in.
    """
    train, val, test, _ = split_paths(paths, seed)
    selected = {"train": train + [val], "val": [val], "test": [test]}[split]  # historical code drops the last entry of train
    original = dataloader.list_parquet_files
    dataloader.list_parquet_files = lambda *args, **kw: list(selected)
    try:
        generator = dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(
            tokenizer, B, T, "train" if split == "train" else "val", device=device, **kwargs)
        first = next(generator)
    finally:
        dataloader.list_parquet_files = original
    return itertools.chain([first], generator)


class TrainStream(torch.utils.data.IterableDataset):
    """The historical train loader, run inside one DataLoader worker process so that tokenizing and best-fit
    packing overlap GPU work instead of sitting between microbatches."""

    def __init__(self, B, T, paths, seed):
        super().__init__()
        self.B, self.T, self.paths, self.seed = B, T, list(paths), seed

    def __iter__(self):
        import nanochat.dataloader as dataloader
        from nanochat.tokenizer import get_tokenizer
        for x, y, state in make_loader(dataloader, get_tokenizer(), self.B, self.T, "train", torch.device("cpu"),
                                       self.paths, self.seed):
            # The historical loader refills one buffer in place: hand off copies, never views of it.
            yield x.clone(), y.clone(), state


def train_stream(B, T, paths, seed, device, pin):
    """Yield `(x, y, state)` on `device`, in exactly the historical order (one worker, one stream).

    spawn: the parent has already initialized CUDA. The private generator keeps the DataLoader from drawing
    its worker seed from the global torch RNG, so model-side randomness is unchanged."""
    loader = torch.utils.data.DataLoader(
        TrainStream(B, T, paths, seed), batch_size=None, num_workers=1, multiprocessing_context="spawn",
        prefetch_factor=4, pin_memory=pin, generator=torch.Generator().manual_seed(seed),
        timeout=WORKER_TIMEOUT_SECONDS)
    for x, y, state in loader:
        yield x.to(device, non_blocking=True), y.to(device, non_blocking=True), state
    # The historical stream is infinite (multi-epoch); an end means the worker failed to deliver.
    raise RuntimeError("training data worker stopped producing batches")
