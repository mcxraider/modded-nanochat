# nanochat experiments

## Research question

**At d8, under a small, fixed training budget, what do grouped-query attention (GQA) and partial RoPE cost or buy us in validation quality, training speed, and inference memory/speed?**

GQA (Ainslie et al., 2023; building on multi-query attention, Shazeer, 2019) shares each key/value head across several query heads, shrinking the KV cache. Partial RoPE (RoFormer, Su et al., 2021; GPT-NeoX's `rotary_pct`, Black et al., 2022) rotates only a fraction of each head's dimensions. Both are widely adopted on the strength of small or indirect evidence at small scale, so we test them in a controlled setting, with a seed-noise estimate and a parameter-matched control.

This is a **small-scale, single-GPU-node study**. The results describe short, compute-limited training; they are not claimed to transfer to other model sizes. Model-size dependence remains unanswered; the former E5 d12 pair is deferred to future work, outside this eight-run budget.

## Budget, initialisation and evaluation (declared before running)

- **Hardware:** one NVIDIA H100 SXM (80 GB), BF16, on Modal (`gpu="H100!"`, 6 cores, 24 GiB) at about $4.43/hour. Chosen over 2× L40S before any results existed: same price, faster, and FA3 runs GQA with sliding windows natively.
- **Budget per run:** a **fixed token count**, identical for every run at the same depth, set to about 10 tokens per scaling parameter (currently provisional: d8 = 3,200 updates ≈ 0.42B tokens, at a global batch of 131,072 tokens; frozen after the pilot). A 2-hour wall-clock safety cap (`wallclock_cap_seconds = 7200`) applies to every run; a run that ends on the cap is flagged `wallclock_cap` and is not comparable. Wall-clock time, tokens and estimated FLOPs are all recorded; a fixed token count means variants differ in time, not in data seen.
- **Initialisation:** every model trains from random initialisation. No pretrained weights, embeddings or components. The nanochat architecture, ClimbMix data and nanochat tokenizer are reused, pinned to upstream commit `324e69c45d3606095adb6b409078647145165454`.
- **Metric:** bits per byte (BPB) on held-out ClimbMix text. BPB is tokenizer-independent, and the tokenizer is fixed anyway. It measures next-token prediction on web-style text only; it does not measure reasoning, instruction following or long-context ability.
- **Data splits** (ClimbMix shards): training = the downloaded pool of shards below index 6541; **validation** = shard 06542 (highest shard, the historical validation shard), used for interim curves and any choices; **test** = shard 06541, held out and evaluated once per run after training. Selection decisions never use the test shard. Evaluation sizes are fixed bytes from the start of each shard: interim validation 2²¹ tokens every `eval_every` updates (d8: 400), final validation and test 2²⁴ tokens each; all runs see identical evaluation text. Interim validation consumes wall-clock time inside training (it counts toward the 2-hour cap, not toward the token target), so its cost is measured in the pilot; the final validation and test passes run after training.
- **Preliminary and tuning experiments** (pilots, microbatch fit) are excluded from the budget and reported separately.

## Experiments and hypotheses

Notation: B0 is the pre-autoresearch nanochat baseline (MHA, full RoPE). `GQA-k` means `n_kv_head = n_head / k`. `R75` means 75% of each head's dimensions are rotated and 25% left unchanged.

At d8 (4 heads of dimension 128), the KV head counts are 4 (MHA), 2 (GQA-2), 1 (GQA-4, i.e. MQA).

Each prediction is written down **before** any results exist. The primary quality metric is the final **test BPB**; validation BPB is used for curves. A difference counts as **resolved** only if it exceeds the Experiment 1 noise threshold `2√2 · s`, where `s` is the sample standard deviation (ddof = 1) of test BPB over the three d8 baseline seeds, so the threshold is a two-sigma bound on the difference of two single runs.

### E1: Noise floor

*Question:* how much does test BPB change from seed to seed for the same configuration?
*Runs (d8):* `d8-b0-s1`, `d8-b0-s2`, `d8-b0-s3`. The seed sets the initialisation **and** the training-data order: seed 1 keeps the historical shard order, seeds 2 and 3 rotate the shard pool by `(seed − 1) × ⌊pool/3⌋`, so they also start on a different subset of shards.
*Prediction:* s is on the order of 0.001–0.003 BPB.
*Use:* defines the resolution threshold for all other comparisons. With three seeds the estimate of s is itself crude, and it includes data-order noise; both are stated in the report. We assume s is similar across variants (checked only if budget allows a second GQA seed).

### E2: Main comparison — does GQA/partial RoPE cost quality?

*Question:* at matched tokens, how much BPB do GQA-2 and GQA-2 + R75 lose relative to B0, and how much training throughput do they gain?
*Runs (d8):* `d8-gqa2` (2 KV heads, full RoPE) and `d8-gqa2-r75` (2 KV heads, 75% RoPE: 96 rotated and 32 untouched dimensions per head). The reference B0 is the mean of the three E1 runs; `d8-gqa2-r75` is also compared with `d8-gqa2`, which isolates the rotary fraction.
*Predictions:*
- H2a: GQA-2 is slightly worse than B0 in BPB (below 1.5× the threshold), consistent with GQA's reported near-parity with MHA.
- H2b: training-throughput gain from GQA-2 is small (under 5%), because KV projections are a minor share of training compute at this size.
- H2c: R75 vs full RoPE is not resolved in BPB, and shows no measurable throughput difference (under 2%).
*What would falsify it:* a resolved BPB loss larger than 1.5× the threshold for GQA-2, or a resolved R75 loss.
*Note:* the historical value embeddings have width `n_kv_head × head_dim`, so GQA also shrinks them; GQA-2 therefore removes parameters beyond the K/V projections. Report both total and scaling-parameter counts.

### E3: Parameter-matched control — does restoring scaling parameters close the gap?

*Question:* does restoring the removed K/V scaling parameters through a wider MLP reduce the GQA-2 loss?
*Runs (d8):* `d8-gqa2-pm`: GQA-2 with `mlp_ratio = 4.5`, i.e. MLP hidden width 2304 instead of 2048, which returns the K/V projection parameters to the MLP. **Matching is on scaling parameters** (transformer matrices + unembedding, the quantity nanochat uses for its token budget): the 2,097,152 parameters removed from K/V equal the 256 added hidden units, leaving only a 256-parameter remainder from the value gates (a test checks the match to within 0.01% at d8). Total parameters cannot be matched, because the value embeddings shrink by about 33.5M at d8 and cannot be recovered through the MLP; both counts are reported.
*Prediction (H3):* GQA-2-PM closes most of the E2 gap (BPB within the threshold of B0).
*Falsifier:* GQA-2-PM stays resolved-worse than B0, which would show that restoring scaling parameters through this MLP allocation does not eliminate the loss. This does not isolate attention capacity: total parameters still differ and the MLP architecture changes.
*Caveats:* GQA-2-PM has more FLOPs per token than GQA-2, which is reported; the comparison is at matched scaling parameters, not matched FLOPs. Muon rescales the learning rate of a matrix by `max(1, rows/cols)^0.5`, so `c_fc` gets about 6% more learning rate in the wider MLP. The weight-decay scale is computed from the MHA baseline of the same depth (with `mlp_ratio` reset to 4.0), so PM uses the same weight decay as B0.

### E4: Dose-response — how far can each knob be pushed?

*Question:* how does quality change along the KV-sharing and rotary-fraction axes?
*Runs (d8):* `d8-gqa4` (1 KV head, MQA) and `d8-gqa2-r25` (2 KV heads, 25% RoPE: 32 rotated dimensions). Combined with E1–E2: KV heads 4, 2, 1 (full RoPE) and rotary fraction 100%, 75%, 25% (on GQA-2).
*Predictions:*
- H4a: BPB degrades monotonically with KV sharing, and MQA is the worst, with a resolved loss.
- H4b: R25 is not resolved from R75/full, because with head dimension 128 even 25% leaves 32 rotated dimensions (consistent with GPT-NeoX's 25%).
*Falsifier:* a resolved R25 loss, or a non-monotone GQA trend.

### Inference study (no extra training)

*Question:* where does the KV-cache saving actually show up?
*Setup:* the benchmark runs automatically after each training run, on the H100, with fixed random prompts, greedy decoding and exactly 256 generated tokens (EOS ignored). Grid: batch size 1, 4, 16, 64 × prompt length 128, 2,048, 8,192; OOM workloads are recorded, not shrunk. The 2,048- and 8,192-token workloads need a cache larger than the 2,048-token training context, so they are speed-only tests with no quality claim. Reported per workload: prefill tokens/s, time to first token, decode tokens/s (the remaining 255 tokens), peak VRAM, the allocated KV-cache bytes, and the analytic KV-cache bytes `2 × n_layer × batch × positions × n_kv_head × head_dim × 2` (BF16).
*Predictions:*
- H6a: measured cache size scales exactly with `n_kv_head` (verifies the implementation does not expand K/V).
- H6b: at batch 1 and short context, decode throughput differs by under 5% between variants.
- H6c: at large batch and long context, GQA-2 and GQA-4 have a resolved decode-throughput advantage that grows with the KV-cache share.
*Risk:* the runtime must select FA3 on the H100. Its loader silently falls back to SDPA, which would repeat K/V to match query heads on masked layers and remove the expected savings; the pilot stops if the backend is not FA3.

## Run list

Defined once in `RUNS` (`experiments/config.py`); every config is generated from it.

| Run ID | Exp. | Depth (layers / width / heads) | KV heads | Rotary | MLP ratio | Seed | Compared with |
|---|---|---|---|---|---|---|---|
| `d8-b0-s1` | E1, E2 | d8 (8 / 512 / 4) | 4 | 1.0 | 4.0 | 1 | — |
| `d8-b0-s2` | E1 | d8 | 4 | 1.0 | 4.0 | 2 | — |
| `d8-b0-s3` | E1 | d8 | 4 | 1.0 | 4.0 | 3 | — |
| `d8-gqa2` | E2 | d8 | 2 | 1.0 | 4.0 | 1 | `d8-b0` (mean of the 3 seeds) |
| `d8-gqa2-r75` | E2 | d8 | 2 | 0.75 | 4.0 | 1 | `d8-b0`, `d8-gqa2` |
| `d8-gqa2-pm` | E3 | d8 | 2 | 1.0 | 4.5 | 1 | `d8-b0`, `d8-gqa2` |
| `d8-gqa4` | E4 | d8 | 1 | 1.0 | 4.0 | 1 | `d8-b0`, `d8-gqa2` |
| `d8-gqa2-r25` | E4 | d8 | 2 | 0.25 | 4.0 | 1 | `d8-gqa2`, `d8-gqa2-r75` |

**8 training runs.** Everything not listed above (data, tokenizer, context 2,048, global batch of 131,072 tokens, optimizer, schedule, evaluation sizes) is held fixed across runs at the same depth. Head dimension is 128 at d8, `SSSL` window pattern, vocabulary 32,768. Seed 1 is used everywhere except the E1 replicates.

## Cost and time estimate

These are estimates from FLOPs, not measurements. Replace them with pilot numbers.

- **Active parameters (matmul, incl. unembedding):** d8 ≈ 42M. At 10 tokens/param that is ≈ 0.42B tokens.
- **FLOPs:** about 6 × params × tokens, plus about 10% for attention. d8 ≈ 1.2×10¹⁷ per run.
- **Throughput assumption:** one H100 has 989 TFLOPs peak BF16 dense and needs no gradient all-reduce. Published benchmarks and peak FLOPs suggest about 1.4–2× the previously assumed 2× L40S throughput.
- **Per-run time:** d8 ≈ 5–10 min of training, or ≈ 40–80 min across eight runs. Evaluation and inference are additional.

| Block | Time | Cost at $4.43/h |
|---|---|---|
| Data and tokenizer (CPU only) | ≈ 0.5 h | under $0.50 |
| Pilots, 6 architectures (full inference grid, reduced repeats) | measure on GPU | measure on GPU |
| Training, 8 runs (optimizer updates only) | ≈ 0.67–1.33 h | ≈ $3–6 |
| Interim + final validation and test, 8 runs | ≈ 0.15–0.3 h | ≈ $0.7–1.3 |
| Inference benchmarks, 8 | ≈ 0.24–0.64 h | ≈ $1.1–2.8 |

Provisional planning allowance: **the earlier 1.2–2.5 GPU-hour / $6–12 estimate is no longer a launch budget**: the expanded six-architecture pilots and numerical preflight add work; use their measured cost table, plus about 0.5 h of CPU preparation. Evaluation and pilot allowances are placeholders, not measured predictions; replace these and training/inference estimates with the d8 pilot's measured costs before relying on the total. The application budget defaults to $40: projections gate the full suite, and training/benchmark subprocesses have a deadline based on the remaining budget. Configure the intended $42.50 workspace usage limit separately in Modal; the script does not set or verify it. Accounting uses the configured hourly estimate and a $0.50 CPU-preparation allowance, so it is not an exact billing cap. Image builds, repeated preparation, GPU checks, other Apps, and storage can add charges.

If the d8 pilot projects excessive cost or cannot fit the declared settings, stop and revise the plan before launching the full suite. No replacement runs or reduced-token d12 runs are included.

## Comparison rules

- **Matched tokens** at the same depth is the primary comparison: every run at a depth trains on the same number of tokens, and the primary number is the final test BPB. BPB against wall-clock time and estimated FLOPs (read from the validation curves at exactly shared observations, never interpolated) are secondary, since GQA changes both per-step time and FLOPs.
- **Resolution threshold:** two runs differ only if the test-BPB gap exceeds `2√2 · s` from E1. Smaller gaps are reported as unresolved, not as ties or wins.
- **Reporting:** absolute and relative test BPB differences (and validation, for curves) against each run's `compare_to` references in the run list, each with a resolved/unresolved flag; parameter counts (total and scaling); tokens; estimated FLOPs; peak VRAM; training tokens/s.
- **No hyperparameter tuning** is done per variant. The inherited H100-tuned optimizer settings are used for all, so a GQA loss could partly reflect that the settings were tuned for MHA. This is listed as a limitation. A future extension could test B0 and GQA-2 at 2× and 0.5× matrix LR at d8 (four additional runs); these runs are outside this suite and its budget.
- **Negative and null results** are reported in full, with the pre-registered prediction next to the outcome.

## Baseline definition

B0 is the nanochat ClimbMix leaderboard configuration immediately before the autoresearch rounds (`324e69c`; round 1 at `6ed7d1d`, round 2 at `a825e63`), re-scaled to d8. The historical launch used H100 FP8; this study uses BF16 throughout. It retains `SSSL` windowed attention, half-context short windows, base-10,000 RoPE, value gates and the historical optimizer grouping, and excludes later architectural improvements. The learning-rate multiplier keeps the historical batch scaling against 524,288 tokens (a 0.5 multiplier at our global batch of 131,072), and the schedule is mapped onto the token target: no warmup, constant for the first 50% of updates, then linear decay to zero; Muon weight decay decays linearly to zero and Muon momentum warmup stays step-based (0.85 → 0.95 over 300 updates). Source: [local leaderboard history](../dev/LEADERBOARD.md), especially Runs 4–6.

## Model implementation and configuration

One configurable `nanochat/modified_gpt.py` derived from the pinned commit exposes `GPTConfig.n_head`, `GPTConfig.n_kv_head` and `rotary_fraction`, plus `mlp_ratio` (MLP hidden width = `mlp_ratio × n_embd`, default 4.0 = historical; the hidden width must be an exact multiple of 128). JSON configs under `experiments/configs/` carry all settings and are generated by `experiments/make_configs.py` from the `RUNS` table, so none are edited by hand. For partial RoPE, frequencies are built for the rotated prefix only, that prefix of Q/K is rotated, and the remaining suffix is concatenated unchanged. The rotary dimension must be an exact even integer (no silent rounding), and fraction 1.0 must reproduce the historical behaviour.

Required checks before the full runs: full-RoPE equivalence to the historical code; untouched partial-RoPE suffixes; GQA projection and cache shapes; cached versus full-sequence decoding; checkpoint reload and generation; the selected attention backend (FA3 on H100) handling GQA without expanding K/V.

## Implementation status

Implemented and covered by CPU tests (`tests/test_experiments.py`, `test_data.py`, `test_modified_gpt.py`, `test_visualise.py`):

- `config.py`: `RUNS` table (single source of truth), token-target and wall-clock-cap schema, three evaluation sizes, `mlp_ratio`, step-based `schedule`, `validate_suite`.
- `make_configs.py`: generates the 8 JSON configs; per-depth policy (token targets, microbatch, cadence) lives in its `DEFAULTS` and is **provisional until the pilot**.
- `data.py`: held-out test shard (`shard_06541`), seed-dependent training order, epoch guard, built on the unmodified historical dataloader.
- `train.py`: fixed-token stop, 2 h safety cap, interim/final validation and a separate test pass, `data_order.json`, `mlp_ratio` reset in the weight-decay reference models.
- `modified_gpt.py`: `GPTConfig.mlp_ratio` (default 4.0 = historical).
- `benchmark.py`: analytic and allocated KV-cache bytes per workload; wider batch/prompt sweep in the generated configs.
- `report.py` / `visualise.py`: seed noise `s`, threshold `2√2·s`, resolved flags, per-depth figures, `dose_response`.

Still to do on the GPU node (not verifiable locally): download the pool and test shard, run the six architecture pilots, freeze `DEFAULTS` and regenerate configs, check that FA3 is selected, so GQA runs without expanding K/V, and confirm `kv_cache_bytes_allocated` equals the analytic value on GPU.

Known confounds to report: the historical value embeddings shrink with KV heads (total parameters cannot be matched in E3); Muon rescales the `c_fc` learning rate with its aspect ratio (about +6% in the 4.5× MLP); the seed varies both initialisation and training-data order, so `s` includes data-order noise.

## Execution order

1. On the GPU node: prepare the pinned worktree, download the shard pool and shard 06541, train the tokenizer, and run the preflight checks above (commands under "Pipeline reference").
2. Pilot all six unique d8 architectures (10 updates each; the automated path uses the declared microbatch, while manual fit checks may sweep device batches): measure tokens/s, peak VRAM and evaluation cost; freeze `DEFAULTS` in `make_configs.py` (token targets, microbatch, cadence) and regenerate the configs.
3. Run E1 (`d8-b0-s1/s2/s3`). Compute `s` and the threshold with `experiments.report`.
4. Run E2, E3 and E4 (the remaining d8 runs).
5. Collect the inference results, which each run produces automatically after training, and rerun `experiments.report` and `visualise.py`.
6. Write up predictions versus outcomes.

## Run record

For every run, save: run ID, experiment, hypothesis, parent config, code commit/diff, launch command, seed, resolved config; GPU model/count, software versions, precision, attention backend; total and scaling parameter counts, KV heads, rotary fraction; dataset/tokenizer manifests and split manifests; tokens, estimated FLOPs, step time, throughput, peak VRAM; validation curve, final test BPB; checkpoint path; GPU-hours and cost.

| Run ID | Seed | Params (total / scaling) | Tokens | Train seconds | Test BPB | Train tok/s | Status |
|---|---|---|---|---|---|---|---|
| `d8-b0-s1` | 1 | — | — | — | — | — | Planned |
| `d8-b0-s2` | 2 | — | — | — | — | — | Planned |
| `d8-b0-s3` | 3 | — | — | — | — | — | Planned |
| `d8-gqa2` | 1 | — | — | — | — | — | Planned |
| `d8-gqa2-r75` | 1 | — | — | — | — | — | Planned |
| `d8-gqa2-pm` | 1 | — | — | — | — | — | Planned |
| `d8-gqa4` | 1 | — | — | — | — | — | Planned |
| `d8-gqa2-r25` | 1 | — | — | — | — | — | Planned |

No results have been produced. Every number in the plan above is a prediction or an estimate.

## GPT-2 reference (optional)

Do not label 0.71854 (nanochat Run 4's validation BPB, on its own distribution) as GPT-2's BPB. The repo's GPT-2 threshold is CORE = 0.256525, a separate metric. Our d8 models are far too small and short-trained to approach it, so GPT-2 is not a target; if a directly comparable BPB number is not available, omit it.

---

# Pipeline reference

## Run on Modal

`modal_app.py` runs everything below in one job on an H100, with stage-aware recovery for handled failures. Hard container crashes require the reconciliation procedure below. Commit your changes first: the commit and any diff go into each run's provenance, and changing `experiments/*.py` midway makes resumes refuse the changed runtime.

```bash
modal run modal_app.py::image_check          # CPU only: image builds, configs resolve
modal run modal_app.py::gpu_check            # H100 available, FA3 loads
modal run modal_app.py --pilot-only --experiment d8-v2  # data, GPU checks, six pilots, cost table
modal run --detach modal_app.py --experiment d8-v2      # reuse pilots, run the full suite
```

Re-running validates the saved config and runtime source hashes before skipping a completed run, resuming a checkpoint, or replacing incomplete outputs. Changed full-run configs or source stop the job; use a new `--experiment` name for a new revision. Artifacts live under `/vol/experiments/<name>/artifacts`. Prepared data/tokenizer remain shared at `/vol/nanochat`, and all namespaces share the durable budget ledger. Pilots live under `/vol/experiments/<name>/pilots/<config-and-source-fingerprint>`, so editing `DEFAULTS` and regenerating configs automatically starts fresh pilots without redownloading data. Old pilot evidence is retained. Only one preparation or suite writer can run across Apps: a named Modal Dict (`nanochat-experiments-locks`) atomically claims its `writer` key, and competing writers fail before touching the Volume.

The durable budget ledger is `/accounting/suite.json` on the `nanochat-experiments` Volume, outside deletable run directories. It imports legacy per-run totals once. Each suite attempt reserves the remaining budget and commits that reservation before work starts; on normal exit, including handled failures, it settles to elapsed suite time. Retries therefore retain earlier costs. Training deadlines kill the whole process group, including torchrun workers. Lower the ceiling with `--budget-usd` if other Modal usage has consumed your allowance.

After a hard kill, automatic recovery intentionally stops: the writer lock may remain and the ledger may retain its full reservation. Stop **all** Apps using this Volume and confirm no worker remains. Inspect Modal usage and the ledger, then explicitly reconcile the cumulative cost of all prior preparation/pilot/suite attempts. Never release a lock while an old worker is active.

```bash
modal volume get nanochat-experiments /accounting/suite.json suite-accounting.json
# Example only: replace 5.25 with the reconciled cumulative spend.
modal run modal_app.py::recover --spent-usd 5.25 --confirm-all-stopped
```

Recovery saves the old ledger and writer identity in `/accounting/recovery_<timestamp>.json`, commits the corrected ledger, then releases the stale lock. It does not reset experiment artifacts or authorize extra budget.

Every suite invocation performs GPU numerical preflight on the current container: full versus cached FA3 across the sliding-window boundary and FA3 versus SDPA, for all six architectures. Every actual training/benchmark process requires FA3 pinned at `de87b9b5af06dd9984df595bef90b2eba44b181a`; silent SDPA fallback is a failure. The kernel identity is recorded in provenance.

The six pilots retain the full inference batch/prompt/decode grid, with one warmup and one measured repeat. Every workload in the grid must succeed: last-token prefill keeps the largest d8 workload near 12 GiB, so a pilot OOM is a regression and blocks the suite. The cost gate uses each architecture's measured training, evaluation, and inference durations, scales inference repeats, and adds a safety margin. It does not sweep microbatch sizes or automatically change the declared token target. Freeze any calibration changes and regenerate configs before the full run.

Everything below `/vol` lives on the `nanochat-experiments` Volume; it does not automatically sync locally. The smaller results archive excludes checkpoint tensors. A separate uncompressed archive contains the final checkpoint and resume state for every run. Earlier checkpoints remain available in the Volume.

```bash
modal volume get nanochat-experiments experiments/d8-v2/artifacts/d8-b0-s1/metrics.jsonl -
modal volume get nanochat-experiments experiments/d8-v2/results.tar.gz .
modal volume get nanochat-experiments experiments/d8-v2/checkpoints.tar .
mkdir -p experiments/downloaded-d8-v2
tar xzf results.tar.gz -C experiments/downloaded-d8-v2
tar xf checkpoints.tar -C experiments/downloaded-d8-v2
```

The sections below describe the same steps by hand on a GPU node.

## Prepare once on the GPU node

The launcher prepares a managed, detached worktree at the pinned commit, by default next to this repository as `modded-nanochat-experiments-324e69c`. It overlays `modified_gpt.py`, every `experiments/*.py` file (including `data.py`), and the patched checkpoint dispatch. Historical data loading, attention backend, optimizer and other dependencies remain pinned. It refuses unmanaged edits and concurrent use of the same runtime.

```bash
python experiments/make_configs.py
python experiments/run.py --config experiments/configs/d8-b0-s1.json --prepare-only
uv sync --project ../modded-nanochat-experiments-324e69c --extra gpu --group dev
```

Prepare the ClimbMix shards and tokenizer from the historical worktree, using its environment and the same `NANOCHAT_BASE_DIR` for every run. The pool must cover the largest run (about `target_tokens / 0.65` raw tokens, because best-fit packing crops roughly 35%) and the d8 seed rotations; measure tokens per shard first. The test shard is fetched separately (`dataset.py` always fetches validation shard 06542):

```bash
cd ../modded-nanochat-experiments-324e69c
.venv/bin/python -m nanochat.dataset -n <POOL_SHARDS>
.venv/bin/python -c "from nanochat.dataset import download_single_file; download_single_file(6541)"
.venv/bin/python -m scripts.tok_train
cd ../modded-nanochat
```

The suite rejects the legacy FineWeb fallback, a missing test shard, and tokenizer vocabulary mismatches. Dependency installation, data preparation, pilots and idle time are outside the measured training phase and remain additional rental cost.

## Pilot, validate, run

```bash
python - <<'PILOT'
import json
from pathlib import Path
config = json.loads(Path("experiments/configs/d8-b0-s1.json").read_text())
config["run_id"] = "d8-b0-s1"
config["training"].update(max_steps=10, eval_every=-1, save_every=-1)
config["evaluation"].update(interim_tokens=131072, final_tokens=131072, test_tokens=131072)
config["benchmark"].update(batch_sizes=[1], prompt_lengths=[128], decode_tokens=16, warmup=1, repeats=1)
Path("/tmp/nanochat-pilot.json").write_text(json.dumps(config, indent=2))
PILOT
python experiments/run.py --config /tmp/nanochat-pilot.json --output-root experiments/artifacts/pilots
```

Repeat for d8 GQA-2 and, if wanted, other `device_batch_size` values (global batch fixed at 131,072); pilot run IDs must be unique and pilots never count as suite runs. Measure tokens/s, peak VRAM and evaluation cost, then edit `DEFAULTS` in `make_configs.py` and regenerate.

```bash
python experiments/run_all.py --dry-run                      # resolves all 8 runs, creates nothing
python experiments/run.py --config experiments/configs/d8-b0-s1.json
python experiments/run_all.py --runs d8-b0-s1 d8-b0-s2 d8-b0-s3   # E1 only; omit --runs for the whole suite
python experiments/run.py --config experiments/configs/d8-b0-s1.json --resume 1600
```

The suite validator checks every config against `RUNS`, that configs at one depth differ only in run ID, seed, KV heads, rotary fraction and MLP ratio, that no (model, seed) pair repeats, and that the suite uses one worker, global batch 131,072 and no step cap. Existing outputs are rejected unless an individual resume is requested. Each launch trains, evaluates (validation then test) after training, then benchmarks the final checkpoint in a fresh process. A failed process stops the sweep. `run_all.py` writes `comparison.json` and `tables.md` at the end.

## Stopping, schedule and optimizer policy

Training runs to `target_tokens / total_batch_size` optimizer updates. The learning-rate schedule is a function of step progress: no warmup, constant for the first 50%, then linear decay to zero; Muon weight decay decays linearly to zero with progress; Muon momentum warmup stays step-based (0.85 → 0.95 over 300 updates). A wall-clock cap (`wallclock_cap_seconds`, 7,200 s) is a safety stop only: a run that ends on it is flagged `stop_reason = wallclock_cap` and is not comparable. Learning rates keep the historical batch scaling against 524,288 tokens (a 0.5 multiplier at the global batch of 131,072 before model-width scaling). Initial weight-decay scaling is resolved from the MHA baseline of the same depth and the historical d12 reference, then frozen across variants (with `mlp_ratio` reset to 4.0 in those references). BF16 replaces the leaderboard's H100 FP8; FP8 is never enabled.

Evaluation uses fixed bytes: interim validation (`interim_tokens`) on the validation shard every `eval_every` updates, then after training a final validation pass (`final_tokens`) and a test pass (`test_tokens`) on the held-out test shard, outside the training clock. Tuning decisions use validation only; the test number is reported once.

Resume verifies checkpoint SHA-256 manifests, restores per-rank optimizer and RNG state, and replays consumed microbatches only when training remains unfinished. Checkpoints are staged and published with their completion marker last; completed generations are immutable. `training_complete.json` skips data replay after training, each final evaluation split is saved independently, and a completed `result.json` permits benchmark-only recovery. Final BPB is authoritative in `result.json`; checkpoint metadata stays immutable. Only the wall-clock cap depends on cumulative seconds; lost time after an interruption is not charged back automatically.

## Observe a running experiment

| Observation | Cadence | File in the run directory |
|---|---|---|
| Step, elapsed time, worker-0 loss, tokens, estimated FLOPs, step duration, throughput, LR multiplier, peak allocated VRAM | Every update | `metrics.jsonl`, `train.log` |
| GPU utilisation and memory (`nvidia-smi`) | Every 10 updates | `metrics.jsonl` |
| Distributed validation BPB | Every `eval_every` updates, and after training (`final: true`) | `metrics.jsonl` |
| Test BPB (`kind: "test"`) | Once, after training | `metrics.jsonl`, `result.json` |
| Hardware/software, effective optimizer groups, parameter counts, manifests, data order | Before training | `hardware.json`, `data_manifest.json`, `data_order.json`, `launch.json`, `config.json` |

Follow `train.log` with `tail -f`, or rerun `visualise.py` on a local copy. Training loss is worker 0's; validation and test BPB are averaged across workers. Peak VRAM is the per-GPU maximum.

## Outputs and comparisons

Each run directory under `experiments/artifacts/` holds the frozen config and source, provenance, logs, curves, checkpoints, `result.json` (including `val_bpb`, `test_bpb`, `scaling_parameters`, `target_tokens`, `data_epoch`), `inference.json`, and rental accounting. `data_manifest.json` is seed-independent (pool shards, validation and test hashes, tokenizer hashes) and is compared across runs at one depth; the per-run training order is in `data_order.json`. `data_epoch > 1` means a run saw repeated data and is flagged.

`report.py` retains raw capped/repeated-data results with explicit exclusion reasons, but excludes them and token-count mismatches from quality comparisons, seed noise and final-quality charts. Ancillary curve, elapsed-time and inference comparisons explicitly identify the individual baseline seed in `field_references`; final BPB and training-speed comparisons use the named group mean. `report.py` computes the d8 baseline seed standard deviation `s` (ddof = 1) from test BPB, the threshold `2√2·s`, and for every `compare_to` pair the test and validation BPB differences, relative differences, a `resolved` flag (`|Δ|` above the threshold), training-speed ratio, total and scaling parameter counts, exact matched-token and matched-FLOP observations (never interpolated), and inference speedups including KV-cache byte ratios. Newly generated `comparison.json` files omit the former `scale_transfer` field.

FLOPs use the historical estimate (six per matmul parameter per token plus attention by query-head count and window); they exclude lookup, norm, softmax and rotary elementwise work. Inference uses fixed greedy prompts and exactly 256 generated tokens ignoring EOS, measures prefill and TTFT separately, times the remaining 255 tokens as decode, records OOM workloads without shrinking the batch, and labels any workload whose cache exceeds the 2,048-token training context as speed-only.

Rental accounting covers the launcher duration at `NODE_HOUR_USD` (`config.py`) and excludes earlier preparation, pilots and idle gaps.

```bash
python -m experiments.report --output-root experiments/artifacts
# Inside the prepared historical worktree, with its Python environment:
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m experiments.benchmark --run-dir /absolute/path/to/artifacts/d8-b0-s1
python -m pytest tests/test_experiments.py tests/test_data.py tests/test_modified_gpt.py tests/test_visualise.py tests/test_engine.py -q
```

Local CPU tests do not establish GPU throughput, memory fit or the selected attention backend; the Modal pilot checks backend and memory fit before the full runs.

## Visualise

```bash
python experiments/visualise.py --output-root experiments/artifacts
python experiments/visualise.py --output-root experiments/artifacts --output-dir /tmp/figures --smooth-window 10
```

Figures are written as PNG and SVG under `<output-root>/figures/`. For d8 (`_d8`): `learning_curves`, `training_performance`, `final_comparison` (final test BPB, pairwise differences against each run's `compare_to` references with the ±2√2·s noise band, throughput, peak VRAM), `quality_speed`, and `inference_comparison` (prefill/decode throughput, TTFT, peak VRAM and allocated KV cache by prompt length and batch size). Also: `dose_response` (BPB against KV heads and rotary fraction). The six figures are each exported as PNG and SVG; no scale-transfer chart is generated. Learning and performance figures work while runs are in progress; missing runs and provenance are flagged; data, code or hardware differences reject the comparison. Charts show observations and descriptive smoothing, with no automatic causal attribution; unresolved differences are not ties.
