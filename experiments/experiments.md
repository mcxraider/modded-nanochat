# nanochat experiments

## Goal

Compare validation bits per byte (BPB), training throughput, and inference throughput for three d24 configurations on **2× L40S 48GB GPUs on one node, with up to two hours of training per run**.

These are planned experiments, not completed results. The two-hour limit applies to each run, giving up to six training hours for all three runs.

## Current scope: three runs

| Run | Configuration | Purpose |
|---|---|---|
| B0 | Pre-autoresearch d24, preserving historical attention and RoPE settings | Establish the baseline |
| E1-GQA-full | d24 with 2:1 GQA and full RoPE (100% of each Q/K head) | Compare GQA/full RoPE against B0 |
| E1-GQA-75 | Same d24 and 2:1 GQA configuration as E1-GQA-full, with 75% RoPE | Isolate the effect of reducing the rotary fraction from 100% to 75% |

Use **2:1 GQA in both E1 runs**: two query heads per KV head (`n_kv_head = n_head / 2`), requiring an even query-head count. Record B0's actual query/KV head counts and rotary fraction; do not assume the historical baseline uses MHA or full RoPE. If B0 already matches E1-GQA-full, resolve the duplicate configuration before training. If B0 differs in both attention and rotary fraction, label the B0 comparison as a combined change rather than attributing its entire effect to GQA.

All models retain d24 depth, query-head count, head dimension, and context length. Keep data, tokenizer, global batch, optimizer, seed, and time-budget learning-rate policy fixed. Record unavoidable parameter-count changes from GQA. Smaller-model sweeps, other GQA ratios, other rotary fractions, data-mix changes, and loader experiments are outside this phase.

Each run receives the same two-hour cap. At the supplied combined rate of $2.18/hour, the training allocation is **up to $4.36 per run and $13.08 for three runs**, or **12 L40S GPU-hours** total. Preparation, pilots, evaluation, inference benchmarks, and overruns cost extra. No additional training runs or repeats are included.

Report absolute and relative BPB differences against B0, and E1-GQA-75 against E1-GQA-full, alongside actual tokens, estimated FLOPs, parameter counts, peak VRAM, and training/inference speedups. With one seed per configuration, treat small differences as preliminary.

## Baseline: before autoresearch rounds 1 and 2

- Pin upstream nanochat commit **`324e69c45d3606095adb6b409078647145165454`** (`324e69c`), the ClimbMix leaderboard entry before either autoresearch round.
- Do not use the current checkout's architecture or tuned defaults as the baseline. Round 1 is recorded at `6ed7d1d`; round 2 at `a825e63`.
- Start from that revision's d24 architecture, ClimbMix data, tokenizer, optimizer, precision, sequence length, batch configuration, and learning-rate settings. Record the resolved configuration, including the actual query/KV head counts; do not assume the baseline uses ordinary multi-head attention.
- Adapt execution to two GPUs using a global batch of 131,072 tokens through gradient accumulation; record this override and any associated learning-rate changes from the historical configuration. Replace the historical token-based stopping horizon with the two-hour budget, and document how the learning-rate schedule maps to that horizon.
- Freeze this adapted configuration as **B0**. All architecture experiments branch from B0, without importing later autoresearch improvements.
- Use separate checkouts or worktrees and artifact directories for historical baseline and variants. Record any compatibility patches.

Source: [local leaderboard history](../dev/LEADERBOARD.md), especially Runs 4–6. The historical Run 4 reported validation BPB **0.71854**, but this is a reference result, not the expected result of our two-GPU, two-hour run.

## Model implementation and experiment configuration

Use **one configurable `nanochat/gpt.py` in a dedicated experiment checkout based on `324e69c45d3606095adb6b409078647145165454`**, preserving existing work in the current checkout. Do not create a model-file copy per experiment. Preserve `nanochat/modded_gpt.py` as exploratory work outside this suite: its current rotary function still rotates the full head, and its `SSL` attention-pattern default differs from the current shared model's `SSSL` default. Freeze the historical attention pattern for all three runs.

GQA is already supported through `GPTConfig.n_head` and `GPTConfig.n_kv_head`; expose the KV-head setting through the runner. Add a serialized `rotary_fraction: float = 1.0` field to `GPTConfig`, preserving full-RoPE behavior by default. CLI flags, if provided, populate that same resolved configuration.

With the historical d24 defaults of 12 query heads and 128 dimensions per head, use:

| Run | Query heads | KV heads | Rotary fraction | Rotated / untouched dimensions per head |
|---|---:|---:|---:|---:|
| B0 | 12 | 12 | 1.0 | 128 / 0 |
| E1-GQA-full | 12 | 6 | 1.0 | 128 / 0 |
| E1-GQA-75 | 12 | 6 | 0.75 | 96 / 32 |

The pinned historical training script sets `n_kv_head=num_heads`, confirming that its default baseline uses multi-head attention. Verify the fully resolved settings during preflight and reject duplicate configurations.

For partial RoPE, generate frequencies for the selected rotary dimension, rotate only that prefix of Q/K, and concatenate the untouched suffix. Validate that the fraction is in `(0, 1]` and produces an even integer rotary dimension without silent rounding. Keep the frequency convention, RoPE base, position handling, and QK normalization fixed. Verify that fraction 1.0 reproduces historical behavior.

Store independent, fully resolved settings in `experiments/configs/b0.json`, `experiments/configs/e1_gqa_full.json`, and `experiments/configs/e1_gqa_75.json`. Use `experiments/run.py` for independent execution and `experiments/run_all.py` for sequential B0 → E1-GQA-full → E1-GQA-75 execution. Launch a fresh two-worker `torchrun` process for each run and wait for all workers to exit before starting the next.

Save query/KV head counts and rotary fraction with each final checkpoint. Loading must reconstruct the shared model from its saved configuration without manual model-file selection. Before full training, verify full-RoPE equivalence, unchanged partial-RoPE suffixes, GQA projection/cache shapes, cached versus full-sequence decoding, and checkpoint reload/generation.

## RunPod hardware and rental plan

Use **2× NVIDIA L40S, 48GB VRAM each, on the same multi-GPU node**, with working NCCL communication. Two separately rented instances cannot use the standalone launch below. Verify availability and interconnect performance before the sweep.

| Item | Planned value |
|---|---|
| Provider | RunPod |
| Combined hourly rate | $2.18/hour (user-supplied quote; confirm at booking) |
| Two-hour training allocation | $4.36 per run; four GPU-hours |
| Three-run matrix | $13.08; 12 GPU-hours, excluding preparation/evaluation/repeats |
| Starting model | Historical d24, approximately 1.38B parameters; record actual count |
| Context length | 2,048 tokens |
| Starting sequences per GPU | 4 |
| Gradient accumulation | 8 microsteps per optimizer update |
| Global batch | 131,072 tokens per optimizer update |

**4 × 2,048 × 2 GPUs × 8 microsteps = 131,072 tokens/update.** Batch 32 per GPU with accumulation 1 has the same effective batch, but activations plus weights, gradients, and optimizer state may exceed 48GB. Batch 4 is a starting point to test, not a guarantee of fit. Pilot per-GPU batches 4, 8, and 16 with accumulation 8, 4, and 2 respectively. Measure peak VRAM and combined tokens/sec while holding the global batch fixed; larger microbatches are not automatically faster. If batch 4 fails, reduce it and increase accumulation accordingly.

After preparing data and tokenizer, the proposed d24 pilot command is:

```bash
torchrun --standalone --nproc_per_node=2 \
  -m scripts.base_train \
  --depth=24 \
  --max-seq-len=2048 \
  --device-batch-size=4 \
  --total-batch-size=131072 \
  --num-iterations=1000
```

The [current training script](../scripts/base_train.py) supports these flags and calculates accumulation automatically. Verify the CLI against the pinned historical revision and supply the full frozen B0 configuration: this command alone does not recreate historical architecture or optimizer defaults. **1,000 updates is a pilot target, not a two-hour stopping mechanism or completion prediction.** Implement and verify time-budget stopping, final checkpointing, and the corresponding learning-rate schedule before full trials. Do not enable the current script's H100+ FP8 option for this plan.

Record the attention backend actually selected on the pinned revision. L40S cannot use the Hopper-specific kernel path, but SDPA is not an unconditional requirement: the [current attention loader](../nanochat/flash_attention.py) attempts a community kernel for non-Hopper GPUs, with an SDPA fallback. Verify backend availability and record compatibility patches; kernel choice can materially affect throughput.

### Illustrative two-hour progress

At an **assumed combined throughput of 5,000 tokens/sec**, 7,200 seconds corresponds to **36 million tokens**, about **275 optimizer updates** (274 complete updates at 131,072 tokens/update), and **0.026 tokens per parameter** for a 1.38B model. This is an illustration, not a measured L40S benchmark. Compilation, evaluation, data stalls, and checkpoint overhead reduce useful training time; revise the estimate using measured end-to-end throughput.

This is a short pretraining experiment for testing loss reduction, throughput, and training correctness, without assuming useful pretrained quality in two hours. Combined tokens/sec measures the training work purchased at the quoted hourly rate; validation BPB remains the quality measure. The $4.36 allocation covers two hours only: preparation, pilots, evaluation, and any overrun increase rental cost.

## Budget and comparison rules

- Each training trial gets two L40S GPUs for **7,200 seconds of training-phase wall time**: four GPU-hours. Include compilation, data stalls, logging, and checkpoint overhead in this clock. Stop at an optimizer-step boundary, save the final checkpoint, and record any overrun.
- Prepare dependencies, data, and tokenizer before the clock starts. Run final evaluation and inference benchmarks afterward and account for their GPU time separately. Thus total rental time exceeds two hours; a strict two-hour rental cap would require reserving setup/evaluation time inside that cap.
- Fixed hardware and time are **not fixed FLOPs**. Record estimated cumulative training FLOPs and tokens processed for every run. Use architecture-aware FLOP estimates, including attention, rather than parameter count alone.
- Primary comparison: final validation BPB at the time limit. Secondary comparisons: BPB at matched token counts and matched estimated FLOPs, plus training tokens/sec, step time, peak VRAM, and GPU utilization.
- Keep validation bytes, tokenizer, context length, evaluation procedure, seed, and training-data order fixed unless explicitly varied. Keep global token batch and optimizer settings fixed for architecture-only comparisons. Record changes in parameter count caused by GQA.
- Use a consistent time-budget learning-rate schedule across variants; record the schedule and any pilot throughput estimates. Do not stop a configuration early at an automatically calculated token/parameter ratio. Record any run ending before the cap and its reason; use matched-time checkpoints if elapsed times differ.
- Run one seed per configuration in this three-run plan. Additional seeds require a separately budgeted extension; small differences are preliminary.

## GQA and RoPE comparison protocol

- Define the GQA ratio as **query heads / KV heads = 2** in both E1 variants. Keep the query-head count fixed and use half as many KV heads; verify the resolved d24 query-head count is divisible by 2.
- Full RoPE rotates 100% of each Q/K head. **75% RoPE means rotating 75% of the head dimensions and leaving 25% unchanged**, not reducing rotation by 75%. Require an even rotary dimension and record the actual dimension/fraction; choose compatible dimensions rather than silently rounding. Keep RoPE base and position handling fixed between E1 variants.
- Train each configuration from scratch. Confirm the kernel supports the selected GQA layout and the inference cache stores fewer KV heads. Explicitly repeating K/V tensors can erase expected speed and memory benefits.
- Verify cached decoding agrees with full-sequence inference for both GQA/RoPE variants. Check quality at the 2,048-token training context; any longer-context claims require separate evaluation.
- Benchmark all three checkpoints on exactly one L40S, using batch sizes 1, 4, and 16 where they fit, prompt lengths 128 and 2,048 tokens, and a fixed 256-token decode workload. Hold precision, kernels, prompts, sampling settings, and cache policy fixed; warm up and synchronize timings.
- Report prefill tokens/sec, time to first token, per-request and aggregate decode tokens/sec, and peak VRAM. Keep fixed-batch comparisons separate from maximum-throughput-at-capacity tests.
- Measure whole-step training throughput: less rotary work does not guarantee a meaningful speedup. Define acceptable BPB degradation before choosing a speed/quality winner.

## GPT-2 reference

The primary question is whether architecture changes improve **validation BPB under our fixed budget**. GPT-2 is an additional reference if it can be evaluated on the exact same held-out bytes with an appropriate scoring procedure.

Do not label **0.71854** as GPT-2's validation BPB: it is nanochat Run 4's result on its validation distribution. The repo's historical GPT-2 capability threshold is **CORE = 0.256525**, a separate metric. If a directly comparable GPT-2 BPB result is unavailable, report the gap to B0 and use CORE only as a secondary capability check.

## Suggested execution order

1. Pin the historical baseline and save its full configuration and data/tokenizer manifests. Resolve B0's head counts and rotary fraction; verify both E1 variants use the specified 2:1 GQA ratio.
2. Verify GQA, full/75% RoPE, time-budget stopping, measurement, checkpointing, and resume before the full runs. Pilot memory fit and throughput on the target node.
3. Train B0 from scratch for up to two hours.
4. Train E1-GQA-full from scratch under the same two-hour cap.
5. Train E1-GQA-75 from scratch under the same cap, changing only the rotary fraction from E1-GQA-full.
6. Evaluate all three checkpoints on the same validation bytes and run the same one-L40S inference benchmark. Report quality and throughput comparisons with actual elapsed time and rental cost.

The plan contains exactly three training runs, with up to six node-hours / 12 GPU-hours of training. Any repeats or new configurations are outside this budget.

## Run record

For every trial, save:

- Run ID, hypothesis, parent configuration, exact code commit/diff, launch command, seed, and full resolved configuration.
- GPU model/count/interconnect, driver and software versions, precision, attention kernel, model parameters, query/KV heads, rotary fraction, and context length.
- Dataset versions and sampling weights, tokenizer hash, validation manifest, loader settings, global token batch, learning-rate schedule, and checkpoint cadence.
- Training-phase elapsed time, overruns/interruption history, total tokens, unique-data estimate, estimated FLOPs and formula, throughput, peak VRAM, and final checkpoint path.
- Final BPB, BPB curves against time/tokens/FLOPs, optional CORE, inference results, GPU-hours including preparation/evaluation, and rental cost.

| Run ID | Change | Seed | Params | Tokens | Est. FLOPs | Train seconds | Val BPB | Train tok/s | Decode tok/s (batch 1) | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 | Pre-autoresearch d24 | TBD | — | — | — | — | — | — | — | Planned |
| E1-GQA-full | d24, 2:1 GQA, 100% RoPE | TBD | — | — | — | — | — | — | — | Planned |
| E1-GQA-75 | Same d24/2:1 GQA, 75% RoPE | TBD | — | — | — | — | — | — | — | Planned |

## Decisions still to make

- Acceptable BPB degradation for a training/inference speed improvement.

Implementation of the experiment runner, model variants, and measurements is separate from this plan. No training results are claimed here.
