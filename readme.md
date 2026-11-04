# GQA and partial RoPE at small scale (nanochat d8)

**Research question:** at depth 8 under a fixed token budget, what do grouped-query attention (GQA) and partial
RoPE cost or buy in validation quality, training speed, and inference memory and speed?

The full pre-registered plan (hypotheses, budget, comparison rules) is in
[`experiments/experiments.md`](experiments/experiments.md). The report is in [`report/`](report/), and group members are listed in
[`authors.md`](authors.md). This repository is a fork of Karpathy's nanochat, whose original README is now
[`nanochat_readme.md`](nanochat_readme.md).

## Repository layout

```
├── readme.md                 # this file
├── authors.md                # group members and student IDs
├── report/                   # report source (LaTeX), bibliography, figures
├── nanochat/modified_gpt.py  # configurable GPT: n_kv_head, rotary_fraction, mlp_ratio
├── experiments/
│   ├── config.py             # RUNS table: single source of truth for every run
│   ├── make_configs.py       # generates experiments/configs/*.json from RUNS
│   ├── configs/              # resolved per-run JSON configs
│   ├── run.py / run_all.py   # launch one run / the suite (pinned historical worktree)
│   ├── train.py, data.py     # fixed-token training, held-out test shard, seeded data order
│   ├── benchmark.py          # inference benchmark (prefill/decode, VRAM, KV cache)
│   ├── report.py             # seed noise, resolution threshold, comparison tables
│   ├── visualise.py          # figures
│   └── experiments.md        # plan and pipeline reference
├── modal_app.py              # runs the whole suite on a Modal H100
├── tests/                    # CPU tests
├── code/                     # (to be added)
└── results/                  # saved numerical results (to be added)
```

## Environment and dependencies

- Python 3.10+ and [uv](https://docs.astral.sh/uv/). Dependencies are pinned in `pyproject.toml` / `uv.lock`
  (PyTorch 2.9.1).
- The experiments ran on **one NVIDIA H100 SXM 80 GB in BF16** on [Modal](https://modal.com). FlashAttention-3 is
  required and is pinned through `kernels` (see `experiments/backend.py`). If the run silently falls back to SDPA, it fails.
- Local setup (CPU tests, report and figures):

```bash
uv sync --extra cpu --group dev
.venv/bin/python -m pytest tests -q
```

- For Modal: run `pip install modal`, then `modal setup`. The Modal image builds its own environment from `uv.lock`.

The training code runs inside a detached worktree pinned to the upstream nanochat commit
`324e69c45d3606095adb6b409078647145165454`. Our `modified_gpt.py` and `experiments/*.py` are overlaid on top of it, so the
data loader, optimizer, and attention backend are the historical ones. `experiments/run.py --prepare-only` creates
this worktree.

## Data

- **Corpus:** NVIDIA ClimbMix shards, downloaded with nanochat's `nanochat.dataset`. The training pool uses shards
  0–47. Shard 06542 is used for validation and shard 06541 is the held-out test shard, which is evaluated once per run.
- **Tokenizer:** the nanochat BPE tokenizer (vocabulary 32,768), trained with `scripts.tok_train`.
- On Modal, the `prepare()` stage downloads the shards and trains the tokenizer automatically. Instructions for doing
  this by hand are in `experiments/experiments.md`, under "Prepare once on the GPU node".

## Reproducing the main results

**1. Run the suite on Modal (recommended).** Commit your changes first, because the commit hash is recorded in each run's provenance.

```bash
modal run modal_app.py::image_check                       # CPU: image builds, configs resolve
modal run modal_app.py::gpu_check                         # H100 available, FA3 loads
modal run modal_app.py --pilot-only --experiment d8-v2    # data, numerical preflight, 6 pilots, cost table
modal run --detach modal_app.py --experiment d8-v2        # full suite: 8 core runs (+ optional GQA-2 seeds)
```

**2. Download the results.**

```bash
modal volume get nanochat-experiments experiments/d8-v2/results.tar.gz .
mkdir -p experiments/artifacts && tar xzf results.tar.gz -C experiments/artifacts
```

**3. Regenerate the tables and figures from the saved results.** This step needs no GPU.

```bash
.venv/bin/python -m experiments.report --output-root experiments/artifacts
.venv/bin/python experiments/visualise.py --output-root experiments/artifacts
```

The figures are written to `experiments/artifacts/figures/`.

**Alternative: run by hand on a GPU node.** Follow these steps:

```bash
python experiments/make_configs.py
python experiments/run_all.py --dry-run   # validates all configs against RUNS
python experiments/run_all.py             # all core runs; --runs <ids> for a subset
```

### Run list

All runs use d8 (8 layers, width 512, 4 heads of dimension 128), context 2,048, and a global batch of 131,072 tokens.
They are trained from random initialisation on the same number of tokens. The runs are defined in `experiments/config.py`.

| Run ID | Exp. | KV heads | Rotary fraction | MLP ratio | Seed |
|---|---|---|---|---|---|
| `d8-b0-s1/s2/s3` | E1 noise floor | 4 | 1.0 | 4.0 | 1/2/3 |
| `d8-gqa2-s1` | E2 | 2 | 1.0 | 4.0 | 1 |
| `d8-gqa2-r75` | E2 | 2 | 0.75 | 4.0 | 1 |
| `d8-gqa2-pm` | E3 param-matched | 2 | 1.0 | 4.5 | 1 |
| `d8-gqa4` | E4 | 1 | 1.0 | 4.0 | 1 |
| `d8-gqa2-r25` | E4 | 2 | 0.25 | 4.0 | 1 |
| `d8-gqa2-s2/s3` | E2 (optional) | 2 | 1.0 | 4.0 | 2/3 |

The primary metric is test bits per byte (BPB). A difference counts as resolved only if it exceeds `2√2·s`, where
`s` is the seed standard deviation of the three B0 runs.

## Mapping report to repository

| Report element | Produced by | Saved in |
|---|---|---|
| Per-run test/val BPB, params, tokens, FLOPs | `experiments/train.py` | `<run>/result.json`, `metrics.jsonl` |
| Seed noise, threshold, pairwise comparisons | `experiments/report.py` | `experiments/artifacts/comparison.json`, `tables.md` |
| Inference throughput, VRAM, KV-cache size | `experiments/benchmark.py` | `<run>/inference.json` |
| Figures | `experiments/visualise.py` | `experiments/artifacts/figures/` |
| Run definitions | `experiments/config.py` | `experiments/configs/*.json` |
