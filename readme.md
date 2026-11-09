# GQA and partial RoPE at small scale (nanochat d8)

**Research question:** at depth 8 under a fixed compute budget (B0's training FLOPs, $1.385\times10^{17}$ estimated
FLOPs per run on one H100), what do grouped-query attention (GQA) and partial RoPE cost or buy in held-out quality,
training speed, and inference memory and speed? And how much of GQA's result comes from the extra tokens its FLOP
saving buys?

- Report: [`report/report.tex`](report/report.tex) (complete LaTeX source), built as [`report/report.pdf`](report/report.pdf).
- Pre-registered plan (hypotheses, budget, analysis rules, written before any results): [`experiments/experiments.md`](experiments/experiments.md).
- Group members and student IDs: [`authors.md`](authors.md).
- Saved numerical results: [`results/`](results/README.md).
- This repository is a fork of Karpathy's nanochat. Its original README is [`nanochat_readme.md`](nanochat_readme.md).

## Quick verification (no GPU, about a minute)

```bash
uv sync --frozen --extra cpu --group dev                  # Python 3.10 env from uv.lock (CPU torch 2.9.1, matplotlib)
uv run --frozen python report/verify_reproduction.py     # report <-> code <-> saved results
uv pip install tokenizers==0.22.0                         # optional: needed only by the pinned-runtime smoke test
uv run --frozen python -m pytest tests -q                 # CPU unit tests (includes the check above)
```

Keep `--frozen`. Newer uv versions otherwise re-resolve and rewrite `uv.lock`, the dependency pin the results were
produced with. The `tokenizers` package is imported by the pinned upstream runtime that
`tests/test_pipeline_smoke.py` trains in. That test is skipped if the package is not installed.

[`report/verify_reproduction.py`](report/verify_reproduction.py) is read-only: it works in a temporary directory and
prints one PASS/FAIL line per check.

1. **Checksums:** every file in `results/d8-v5/export/SHA256SUMS` matches. Only the checkpoint binaries kept
   out of git may be missing.
2. **Provenance:** all 14 runs were launched from commit `c07aa7e` with a clean tree, on upstream nanochat
   `324e69c`. The code in this repository hashes identically to the code that ran.
3. **Configs:** `experiments/make_configs.py` regenerates every `experiments/configs/*.json` and every run's saved
   `config.json`.
4. **Analysis:** `experiments.report` rebuilds `tables.md` byte for byte, and `comparison.json` to within 1e-9
   relative. The last floating-point digit can differ.
5. **Numbers:** the report's headline numbers are recomputed from `result.json` / `comparison.json` and must
   appear verbatim in `report.tex`: B0 seed scores, mean, seed s.d., resolution threshold, the GQA-2 and MQA BPB
   deltas, throughput gains, update counts, total tokens and the FLOP budget.
6. **Figures:** `report/make_figures.py` regenerates every PDF that `report.tex` includes, byte for byte.
7. **Report source:** every included figure, citation key and repository path named in `report.tex` exists.

## Repository layout

```
├── readme.md                  # this file: setup, data, experiments, reproduction
├── authors.md                 # group members and student IDs
├── report/
│   ├── report.tex             # complete report source (LaTeX)
│   ├── references.bib         # bibliography
│   ├── figures/               # every figure the report includes (PDF, generated)
│   ├── make_figures.py        # regenerates figures/ from results/d8-v5
│   ├── verify_reproduction.py # checks report <-> code <-> results (see above)
│   └── report.pdf             # built report
├── experiments/               # experiment code ("code/" in the rubric)
│   ├── config.py              # RUNS table: single source of truth for all 14 runs
│   ├── make_configs.py        # writes experiments/configs/*.json from RUNS
│   ├── configs/               # resolved per-run configs ("configs/" in the rubric)
│   ├── run.py / run_all.py    # launch one run / the suite inside the pinned historical worktree
│   ├── train.py, data.py      # FLOP-budgeted training, seeded shard order, held-out val/test evaluation
│   ├── checkpoints.py, resume.py, backend.py, preflight.py   # checkpointing, resume, FA3 pinning, GPU checks
│   ├── benchmark.py           # inference benchmark (prefill/decode, VRAM, KV cache)
│   ├── report.py              # seed noise, resolution threshold, contrasts -> comparison.json, tables.md
│   ├── visualise.py           # exploratory figures
│   └── experiments.md         # pre-registered plan and pipeline reference
├── nanochat/modified_gpt.py   # the model: n_kv_head (GQA), rotary_fraction, mlp_ratio, value_embeds
├── modal_app.py               # runs data prep, preflight, pilots and the full suite on a Modal H100
├── tests/                     # CPU tests for model, training, resume, report, Modal pipeline
├── results/                   # saved numerical results ("results/" in the rubric); see results/README.md
│   ├── d8-v5/                 # the reported suite: artifacts/, audit/, pilots/, export/
│   └── pilot_d8-v5.md/.json   # pilot throughput and cost-gate summary
└── pyproject.toml, uv.lock    # pinned dependencies
```

The code stays at `experiments/` and `nanochat/` rather than moving to `code/`. Every run's `launch.json` records
SHA-256 hashes under these exact paths, and the pinned runtime overlays files by path. Renaming them would break the
link between the saved results and the code that produced them.

Folders inherited from upstream nanochat and **not** used in the report: `dev/`, `runs/`, `tasks/`, most of
`scripts/`, and `nanochat/gpt.py` / `modded_gpt.py`. The study uses only `nanochat/dataset.py` and
`scripts/tok_train.py`, for data download and tokenizer training, plus the historical modules inside the pinned
worktree.

## Environment and dependencies

- **Python 3.10** and [uv](https://docs.astral.sh/uv/). Dependencies are pinned in `pyproject.toml` / `uv.lock`
  (PyTorch 2.9.1). Use the `cpu` extra locally and the `gpu` extra (CUDA 12.8) on GPU nodes. matplotlib and pytest
  are in the `dev` group. Run scripts with `uv run --frozen python …` or `.venv/bin/python …`; a system `python3` lacks them.
- **Report build:** [tectonic](https://tectonic-typesetting.github.io/) (`cd report && tectonic report.tex`) or a
  TeX Live installation with `latexmk -pdf report.tex`.
- **Training hardware:** one **NVIDIA H100 SXM 80 GB**, BF16 autocast, on [Modal](https://modal.com)
  (`gpu="H100!"`). FlashAttention-3 is required. It is pinned to `varunneal/flash-attention-3@de87b9b` through
  `kernels` (see `experiments/backend.py`), and a silent fallback to SDPA counts as a failure.
- **Pinned runtime:** training runs inside a detached git worktree at upstream nanochat
  `324e69c45d3606095adb6b409078647145165454`, created next to this repository as
  `../modded-nanochat-experiments-324e69c`. `nanochat/modified_gpt.py` and `experiments/*.py` are overlaid on top
  of it, so the data loader, optimizer and attention backend are the historical ones.
  `experiments/run.py --prepare-only` creates the worktree; the Modal image does this automatically.
- **Modal:** `pip install modal && modal setup`. The image builds its own environment from `uv.lock`.

## Data

- **Corpus:** NVIDIA ClimbMix parquet shards, downloaded with nanochat's `nanochat.dataset`. The corpus is not
  stored in this repository.
  - Training pool: shards 0–47 (`SHARDS = 48` in `modal_app.py`). Seeds 2 and 3 rotate the pool by 16 and 32 shards.
  - **Validation:** shard 06542, $2^{24}$ tokens, used for curves and any decision.
  - **Test:** shard 06541, $2^{24}$ tokens, evaluated once per run after training. This is the primary metric.
- **Tokenizer:** the nanochat BPE recipe (`scripts.tok_train`, vocabulary 32,768), trained by us on the training
  shards. The trained tokenizer is saved in `results/d8-v5/export/tokenizer/`.
- **On Modal:** `prepare()` downloads the shards and the test shard and trains the tokenizer, then checks that the
  pool holds enough tokens for the largest run.
- **By hand on a GPU node:**

```bash
python experiments/make_configs.py
python experiments/run.py --config experiments/configs/d8-b0-s1.json --prepare-only   # creates the pinned worktree
uv sync --frozen --project ../modded-nanochat-experiments-324e69c --extra gpu --group dev
cd ../modded-nanochat-experiments-324e69c
export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat      # use the same value for every run
.venv/bin/python -m nanochat.dataset -n 48           # training shards 0-47 and validation shard 06542
.venv/bin/python -c "from nanochat.dataset import download_single_file; download_single_file(6541)"   # test shard
.venv/bin/python -m scripts.tok_train
cd ../modded-nanochat
```

## Reproducing the main results

### A. From the saved results (no GPU)

Run [Quick verification](#quick-verification-no-gpu-about-a-minute) first. Then:

```bash
uv run --frozen python report/make_figures.py                 # rewrites report/figures/*.pdf (byte-identical)
cd report && tectonic report.tex                      # rebuilds report/report.pdf
```

To recompute the analysis or the exploratory figures yourself, work on a **copy**. These commands write their
outputs into `--output-root`, and pointing them at `results/d8-v5/artifacts` would overwrite the committed evidence.

```bash
mkdir -p experiments/artifacts && cp -R results/d8-v5/artifacts experiments/artifacts/d8-v5   # gitignored
uv run --frozen python -m experiments.report --output-root experiments/artifacts/d8-v5
uv run --frozen python experiments/visualise.py --output-root experiments/artifacts/d8-v5
```

### B. Rerunning the experiments on a Modal H100

Commit your changes first: each run records the commit hash and any uncommitted diff in its `launch.json`, and
the verifier expects an empty diff. The reported suite used the namespace `d8-v5`. Use a new namespace for a fresh
rerun.

```bash
modal run modal_app.py::image_check                      # CPU: image builds, configs resolve
modal run modal_app.py::gpu_check                        # H100 available, FA3 loads
modal run modal_app.py --pilot-only --experiment <name>  # data prep, numerical preflight, 8 pilots, cost gate
modal run --detach modal_app.py --experiment <name>      # full suite: 14 runs, then report.py + visualise.py
```

Download the results and analyse them:

```bash
modal volume get nanochat-experiments experiments/<name>/results.tar.gz .
mkdir -p experiments/artifacts/<name> && tar xzf results.tar.gz -C experiments/artifacts/<name>   # -> .../<name>/artifacts/
uv run --frozen python -m experiments.report --output-root experiments/artifacts/<name>/artifacts
```

`report/make_figures.py` and `report/verify_reproduction.py` read `results/d8-v5/` directly. To use them on a
rerun, copy the new `artifacts/` there.

**Cost of the reported suite:** the 14 run pipelines took 8,326 s on one H100 (2.31 GPU-hours), which is **$10.25**
at the configured $4.43/h. Training itself took 6,160 s, about 6.8–7.6 min per run. Data preparation, preflight and
pilots cost about $2.08 more. Seed noise means a rerun will not reproduce test BPB bit for bit; seed-to-seed s.d. is
0.00066 BPB.

**Alternative: by hand on a GPU node.** After the [Data](#data) steps:

```bash
python experiments/run_all.py --dry-run   # validates all configs against RUNS
python experiments/run_all.py             # all 14 runs (or --runs <ids>), then experiments.report
```

### Run list

All runs use d8 (8 layers, width 512, 4 query heads of dimension 128), context 2,048, and a global batch of 262,144
tokens (upstream nanochat's batch rule at d8). They are trained from random initialisation.

- E1–E5 runs train for the most updates that fit B0's FLOP budget. B0 gets 1,680 updates (10.5 tokens per scaling
  parameter). GQA runs get more, because they cost fewer FLOPs per token.
- E6 runs are a matched-token ablation at B0's 1,680 updates.
- Runs are defined in `experiments/config.py`.

| Run ID | Exp. | KV heads | Rotary fraction | MLP ratio | Value embeds | Budget (updates) | Seed |
|---|---|---|---|---|---|---|---|
| `d8-b0-s1/s2/s3` | E1 noise floor | 4 | 1.0 | 4.0 | on | FLOPs (1,680) | 1/2/3 |
| `d8-gqa2-s1/s2/s3` | E2 | 2 | 1.0 | 4.0 | on | FLOPs (1,750) | 1/2/3 |
| `d8-gqa2-r75` | E2 | 2 | 0.75 | 4.0 | on | FLOPs (1,750) | 1 |
| `d8-gqa2-pm` | E3 param-matched | 2 | 1.0 | 4.5 | on | FLOPs (1,680) | 1 |
| `d8-gqa4` | E4 | 1 | 1.0 | 4.0 | on | FLOPs (1,787) | 1 |
| `d8-gqa2-r25` | E4 | 2 | 0.25 | 4.0 | on | FLOPs (1,750) | 1 |
| `d8-b0-nove` | E5 value-embed 2×2 | 4 | 1.0 | 4.0 | off | FLOPs (1,680) | 1 |
| `d8-gqa2-nove` | E5 value-embed 2×2 | 2 | 1.0 | 4.0 | off | FLOPs (1,750) | 1 |
| `d8-gqa2-tok` | E6 matched-token ablation | 2 | 1.0 | 4.0 | on | tokens (1,680) | 1 |
| `d8-gqa4-tok` | E6 matched-token ablation | 1 | 1.0 | 4.0 | on | tokens (1,680) | 1 |

The primary metric is test bits per byte (BPB). A difference counts as resolved only if it exceeds `2√2·s`, where
`s` is the seed standard deviation of the three B0 runs.

## Mapping report to repository

Paths are relative to `results/d8-v5/artifacts/` unless stated otherwise. `<run>` is a run ID from the table above.

| Report element | Produced by | Saved in |
|---|---|---|
| Declared budget, architectures, updates (Table `tab:arch`) | `experiments/config.py`, `experiments/make_configs.py` | `experiments/configs/*.json`, `<run>/config.json`, `<run>/result.json` (`flop_budget`, `flops_per_token`, `step`, `parameters`) |
| Validity checks (preflight, pilots) | `experiments/preflight.py`, `modal_app.py` | `results/d8-v5/pilots/9a34115f44463cc7/preflight.json`, `results/pilot_d8-v5.md` |
| Per-run val/test BPB, tokens, FLOPs | `experiments/train.py` | `<run>/result.json`, `<run>/metrics.jsonl`, `<run>/evaluation_*.json` |
| Seed noise, threshold, contrasts, sensitivity, paired and E5 interaction (Table `tab:results`, Table `tab:hypotheses`) | `experiments/report.py` | `comparison.json`, `tables.md` |
| Training throughput and memory (Table `tab:speed`) | `experiments/train.py` | `<run>/result.json` (`steady_tokens_per_second`, `peak_vram_bytes`, `eval_peak_vram_bytes`), `results/d8-v5/audit/run_summary.csv` |
| Inference throughput, VRAM, KV cache (Table `tab:inference`) | `experiments/benchmark.py` | `<run>/inference.json`, `comparison.json` (`inference_speedups`) |
| Fig. `fig:curves` (learning curves) | `report/make_figures.py` | `<run>/metrics.jsonl` → `report/figures/learning_curves_d8.pdf` |
| Quality summary, paired/interaction, dose–budget, trade-off, speed-ratio figures | `report/make_figures.py` | `comparison.json`, `<run>/result.json`, `<run>/inference.json` → `report/figures/*.pdf` |
| Fig. `fig:inference`, Fig. `fig:cost` (appendix) | `report/make_figures.py` | `<run>/inference.json`, `results/d8-v5/audit/audit.json` |
| Cost and run-time accounting | `modal_app.py`, `results/d8-v5/audit/evaluate.py` | `<run>/rental_accounting.json`, `results/d8-v5/audit/` |
| Code provenance | `experiments/run.py` | `<run>/launch.json`, `<run>/source/` |

## Known issues

- `tests/test_execution.py::test_memory_limit` is skipped on macOS, because the OS does not enforce `RLIMIT_AS`.
- A fresh `comparison.json` can differ from the saved one in the last floating-point digit, because summation order
  differs across platforms. The verifier compares it to within 1e-9.
- `results/d8-v5/README.md` and `results/d8-v5/audit/report.md` are kept exactly as exported, because checksums
  cover them. Their remark that the report has TBD fields is out of date. See `results/README.md`.
