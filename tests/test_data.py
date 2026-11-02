"""Shard selection for the experiment suite (no tokenizer or GPU needed)."""
from pathlib import Path
import types

import pytest

from experiments.data import TEST_INDEX, make_loader, shard_index, split_paths


def shards(count, extra=(TEST_INDEX, TEST_INDEX + 1)):
    base = Path("/data/base_data_climbmix")
    return [base / f"shard_{i:05d}.parquet" for i in [*range(count), *extra]]


def test_split_is_disjoint_and_seed_one_is_historical_order():
    train, val, test, rotation = split_paths(shards(9), seed=1)
    assert [shard_index(p) for p in train] == list(range(9))
    assert shard_index(val) == TEST_INDEX + 1 and shard_index(test) == TEST_INDEX
    assert rotation == 0 and not {*train} & {val, test}


def test_seed_rotates_training_order_without_changing_membership():
    pool = shards(9)
    base, *_ = split_paths(pool, 1)
    for seed in (2, 3):
        train, _, _, rotation = split_paths(pool, seed)
        assert rotation == (seed - 1) * 3 and train[0] != base[0]
        assert sorted(train) == sorted(base)


@pytest.mark.parametrize("paths", [shards(9, extra=(TEST_INDEX + 1,)), shards(2), shards(9, extra=(TEST_INDEX,))])
def test_missing_test_shard_or_tiny_pool_rejected(paths):
    with pytest.raises(ValueError):
        split_paths(paths, 1)


def fake_dataloader(record):
    """Mimic the historical loader: it reads list_parquet_files lazily and slices [:-1] / [-1:]."""
    module = types.SimpleNamespace(list_parquet_files=lambda *a, **k: ["unpatched"])

    def loader(tokenizer, B, T, split, device="cpu", **kwargs):
        files = module.list_parquet_files()
        used = files[:-1] if split == "train" else files[-1:]
        record.append((split, [shard_index(f) for f in used]))
        for i in range(3):
            yield i, i, {"epoch": 1}
    module.tokenizing_distributed_data_loader_with_state_bos_bestfit = loader
    return module


@pytest.mark.parametrize("split,expected", [("train", list(range(9))), ("val", [TEST_INDEX + 1]), ("test", [TEST_INDEX])])
def test_loader_reads_only_the_requested_shards_and_restores_the_module(split, expected):
    record = []
    module = fake_dataloader(record)
    original = module.list_parquet_files
    batches = list(make_loader(module, None, 1, 1, split, "cpu", shards(9), 1))
    assert len(batches) == 3  # the primed first batch is chained back in
    assert record == [("train" if split == "train" else "val", expected)]
    assert module.list_parquet_files is original
