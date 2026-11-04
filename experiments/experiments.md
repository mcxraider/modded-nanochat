# nanochat experiments

## Research question

**At d8, under a small, fixed training budget, what do grouped-query attention (GQA) and partial RoPE cost or buy us in validation quality, training speed, and inference memory/speed? And under a fixed *compute* budget, does GQA's per-token FLOP saving buy any quality back?**

GQA (Ainslie et al., 2023; building on multi-query attention, Shazeer, 2019) shares each key/value head across several query heads, shrinking the KV cache. Partial RoPE (RoFormer, Su et al., 2021; GPT-NeoX's `rotary_pct`, Black et al., 2022) rotates only a fraction of each head's dimensions. Both are widely adopted on the strength of small or indirect evidence at small scale, so we test them in a controlled setting, with a seed-noise estimate and a parameter-matched control.

In this architecture GQA has a side effect: the ResFormer-style value embeddings (value residual learning, Zhou et al., 2024) have width `n_kv_head × head_dim`, so halving the KV heads also halves them. At d8 that is 33.5M of B0's 125.8M total parameters for GQA-2, far more than the 2.1M removed from the K/V projections. Any GQA effect may therefore be a value-embedding effect; E5 separates the two with a 2×2 design. E6 asks the project's budget question directly: GQA lowers FLOPs per token, so at a fixed FLOP budget it can train on more tokens.

This is a **small-scale, single-GPU-node study**. The results describe short, compute-limited training; they are not claimed to transfer to other model sizes. Model-size dependence remains unanswered; a d12 replication is deferred to future work, outside this suite.

## Budget, initialisation and evaluation (declared before running)

- **Hardware:** one NVIDIA H100 SXM (80 GB), BF16, on Modal (`gpu="H100!"`, 6 cores, 24 GiB) at about $4.43/hour. Chosen over 2× L40S before any results existed: same price, faster, and FA3 runs GQA with sliding windows natively.
- **Budget per run: at most C estimated training FLOPs**, where C is B0's usage: 3,200 updates × 131,072 tokens × 314,575,872 FLOPs/token = **1.319 × 10¹⁷ FLOPs** at d8 (provisional until the pilot; `make_configs.py` derives C from B0 and the token target, so it is recomputed if `DEFAULTS` change, and frozen before E1). The token target, about 10 tokens per scaling parameter, is 0.42B tokens at a global batch of 131,072. Every config carries `flop_budget`; training refuses to start a run whose `target_tokens × FLOPs/token` exceeds it, and the report excludes any run whose `estimated_flops` does. FLOPs use nanochat's estimate (`GPT.estimate_flops`: 6 × matmul parameters per token plus windowed attention, 12 × query heads × head_dim × window per layer). It excludes embedding and value-embedding lookups, norms, softmax and rotary elementwise work, so by this measure value embeddings cost no FLOPs; this is a limitation of the measure, stated in the report. A 2-hour wall-clock safety cap (`wallclock_cap_seconds = 7200`) applies to every run; a run that ends on the cap is flagged `wallclock_cap` and is not comparable.
- **Two regimes inside the budget.** *Matched tokens* (E1–E5): every run trains on B0's 0.42B tokens, so variants with fewer FLOPs per token use less than C. *Matched compute* (E6): the run trains the most whole updates that fit C, `⌊C / (FLOPs/token × 131,072)⌋`, using at least 99.9% of it. Wall-clock time, tokens and estimated FLOPs are recorded for every run.
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
*Use:* defines the resolution threshold for all other comparisons. With three seeds the estimate of s is itself crude, and it includes data-order noise; both are stated in the report. We assume s is similar across variants. The three GQA-2 seeds (E2) check this: the report also gives `gqa2_noise`, the GQA-2 seed standard deviation, next to the baseline's.

### E2: Main comparison — does GQA/partial RoPE cost quality?

*Question:* at matched tokens, how much BPB do GQA-2 and GQA-2 + R75 lose relative to B0, and how much training throughput do they gain?
*Runs (d8):* `d8-gqa2-s1/s2/s3` (2 KV heads, full RoPE; three seeds, like the baseline) and `d8-gqa2-r75` (2 KV heads, 75% RoPE: 96 rotated and 32 untouched dimensions per head). The reference B0 is the mean of the three E1 runs; `d8-gqa2-r75` is also compared with `d8-gqa2`, which isolates the rotary fraction. `d8-gqa2` names the GQA-2 seed group, the mean of its three seeds, exactly as `d8-b0` names the baseline mean. Because GQA-2 is the reference for four other runs, its three-seed mean steadies those comparisons too.
*Predictions:*
- H2a: GQA-2 is slightly worse than B0 in BPB (below 1.5× the threshold), consistent with GQA's reported near-parity with MHA.
- H2b: training-throughput gain from GQA-2 is small (under 5%), because KV projections are a minor share of training compute at this size.
- H2c: R75 vs full RoPE is not resolved in BPB, and shows no measurable throughput difference (under 2%).
- *Throughput measure for H2b/H2c (declared before any results):* steady-state training throughput, `steady_tokens_per_second` in `result.json`: the median of per-update tokens/s after the first 20 updates (compile warm-up), where each update's time excludes interim evaluation, checkpoint saves and GPU sampling. Whole-phase throughput (`training_tokens_per_second`, including compile, evaluation and checkpoint I/O, which scale with total parameters) is reported as secondary. Both are single-run measurements with no timing-noise estimate, so the 5% / 2% bounds are descriptive.
*What would falsify it:* a resolved BPB loss larger than 1.5× the threshold for GQA-2, or a resolved R75 loss.
*Secondary rule (declared before any results):* when a comparison involves seed means, the report also gives `resolved_by_means`: `|Δ| > 2·s·√(1/n_run + 1/n_ref)`, the two-sigma bound for a difference of an `n_run`-seed mean and an `n_ref`-seed mean (it equals `2√2·s` for one seed each). The pre-registered `resolved` flag above stays the primary verdict; `resolved_by_means` shows what the extra GQA-2 seeds add (for three seeds on each side the bound is `2·s·√(2/3)`, about 0.58 of `2√2·s`).
*Note:* the historical value embeddings have width `n_kv_head × head_dim`, so GQA also shrinks them; GQA-2 therefore removes parameters beyond the K/V projections. Report both total and scaling-parameter counts.

### E3: Parameter-matched control — does restoring scaling parameters close the gap?

*Question:* does restoring the removed K/V scaling parameters through a wider MLP reduce the GQA-2 loss?
*Runs (d8):* `d8-gqa2-pm`: GQA-2 with `mlp_ratio = 4.5`, i.e. MLP hidden width 2304 instead of 2048, which returns the K/V projection parameters to the MLP. **Matching is on scaling parameters** (transformer matrices + unembedding, the quantity nanochat uses for its token budget): the 2,097,152 parameters removed from K/V equal the 256 added hidden units, leaving only a 256-parameter remainder from the value gates (a test checks the match to within 0.01% at d8). Total parameters cannot be matched, because the value embeddings shrink by about 33.5M at d8 and cannot be recovered through the MLP; both counts are reported.
*Prediction (H3):* GQA-2-PM closes most of the E2 gap (BPB within the threshold of B0).
*Falsifier:* GQA-2-PM stays resolved-worse than B0, which would show that restoring scaling parameters through this MLP allocation does not eliminate the loss. This does not isolate attention capacity: total parameters still differ and the MLP architecture changes.
*Caveats:* GQA-2-PM is FLOP-matched to B0 to within 0.0005% (314,574,336 vs 314,575,872 FLOPs/token), so it is simultaneously a scaling-parameter-matched and a FLOP-matched control, and a second compute-matched comparison alongside E6; total parameters still differ by the value-embedding shrinkage. Muon rescales the learning rate of a matrix by `max(1, rows/cols)^0.5`, so `c_fc` gets about 6% more learning rate in the wider MLP. The weight-decay scale is computed from the MHA baseline of the same depth (with `mlp_ratio` reset to 4.0), so PM uses the same weight decay as B0.

### E4: Dose-response — how far can each knob be pushed?

*Question:* how does quality change along the KV-sharing and rotary-fraction axes?
*Runs (d8):* `d8-gqa4` (1 KV head, MQA) and `d8-gqa2-r25` (2 KV heads, 25% RoPE: 32 rotated dimensions). Combined with E1–E2: KV heads 4, 2, 1 (full RoPE) and rotary fraction 100%, 75%, 25% (on GQA-2).
*Predictions:*
- H4a: BPB degrades monotonically with KV sharing, and MQA is the worst, with a resolved loss.
- H4b: R25 is not resolved from R75/full, because with head dimension 128 even 25% leaves 32 rotated dimensions (consistent with GPT-NeoX's 25%).
*Falsifier:* a resolved R25 loss, or a non-monotone GQA trend.

### E5: Value-embedding control — is the GQA penalty a value-embedding effect?

*Question:* how much of GQA-2's BPB change comes from its smaller value embeddings rather than from sharing K/V heads?
*Runs (d8):* `d8-b0-nove` and `d8-gqa2-nove`: B0 and GQA-2 with the value embeddings and their gates removed (`value_embeds = false`), everything else unchanged. With `d8-b0` and `d8-gqa2` they form a 2×2 design, {4, 2 KV heads} × {value embeddings on, off}.
*Predictions:*
- H5a: removing value embeddings costs B0 a resolved BPB loss, as they were a positive change on the nanochat leaderboard.
- H5b: GQA-2's gap without value embeddings (`gqa2-nove − b0-nove`) is smaller than with them (`gqa2 − b0`).
- H5c: the interaction `(gqa2 − b0) − (gqa2-nove − b0-nove)` is positive but probably unresolved: its threshold is `2·s·√(1/3 + 1/3 + 1 + 1)` ≈ 3.3 s, larger than the single-pair threshold.
*Falsifier:* `gqa2-nove − b0-nove` is resolved and about equal to `gqa2 − b0`, which would mean the penalty comes from K/V sharing, not the value embeddings.
*Caveats:* without value embeddings the two arms differ by only the 2.1M K/V parameters (versus 35.7M with them); weight decay stays frozen to B0's (the reference models keep `value_embeds = true`); no per-variant learning-rate tuning. The value embeddings add no counted FLOPs, so E5 changes parameters and memory, not the FLOP budget.

### E6: Fixed compute — does GQA's FLOP saving buy quality back?

*Question:* at the declared FLOP budget, does spending GQA's per-token saving on more training tokens close its matched-token gap?
*Runs (d8):* `d8-gqa2-iso` (3,333 updates, +4.2% tokens; GQA-2 uses 0.960× B0's FLOPs per token) and `d8-gqa4-iso` (3,404 updates, +6.4%; 0.940×). The schedule has the same shape (constant for 50%, then linear decay to zero), stretched over the longer run.
*Prediction (H6):* by the Chinchilla fit `L(N, D) = E + A/N^α + B/D^β` (Hoffmann et al., 2022; B = 410.7, β = 0.28), 4.2% / 6.4% more data reduces the data term `B/D^β` (≈ 1.58 nats at D = 0.42B) by 1.1% / 1.7%, i.e. about 0.018 / 0.027 nats, or roughly **0.005–0.006 BPB for GQA-2-iso and 0.008–0.009 BPB for GQA-4-iso** at 4.2–4.8 bytes per token. The fit was made on a different tokenizer, data and scale, so treat these as order-of-magnitude. H6a: both iso runs are better than their matched-token twins, by about these amounts. H6b: whether that is resolved depends on `s`: the primary threshold `2√2·s` is 0.003–0.008 BPB over the predicted s = 0.001–0.003, so the GQA-2 gain is resolved only if s ≲ 0.002 and the GQA-4 gain only if s ≲ 0.003. Whether an iso run closes its matched-token gap to B0 is not predicted: H2a only bounds that gap from above.
*Falsifier:* either iso run is resolved-worse than its matched-token twin, or resolved-better by more than twice the predicted amount (more than 0.012 / 0.018 BPB).
*Note (revised before any results):* an earlier draft predicted both iso differences to be unresolved; that did not follow from the Chinchilla numbers above once compared with the expected threshold.
*Use:* the direct answer to the project's fixed-budget question. With E3, which is FLOP-matched by construction, it gives two compute-matched comparisons: one spends the saving on tokens (E6), the other on MLP parameters (E3).

### Inference study (no extra training)

*Question:* where does the KV-cache saving actually show up?
*Setup:* the benchmark runs automatically after each training run, on the H100, with fixed random prompts, greedy decoding and exactly 256 generated tokens (EOS ignored). Grid: batch size 1, 4, 16, 64 × prompt length 128, 2,048, 8,192; OOM workloads are recorded, not shrunk. The 2,048- and 8,192-token workloads need a cache larger than the 2,048-token training context, so they are speed-only tests with no quality claim. Reported per workload: prefill tokens/s, time to first token, decode tokens/s (the remaining 255 tokens), peak VRAM, the allocated KV-cache bytes, and the analytic KV-cache bytes `2 × n_layer × batch × positions × n_kv_head × head_dim × 2` (BF16).
*Predictions:*
- HIa: measured cache size scales exactly with `n_kv_head` (verifies the implementation does not expand K/V).
- HIb: at batch 1 and short context, decode throughput differs by under 5% between variants.
- HIc: at large batch and long context, GQA-2 and GQA-4 have a resolved decode-throughput advantage that grows with the KV-cache share.
*Risk:* the runtime must select FA3 on the H100. Its loader silently falls back to SDPA, which would repeat K/V to match query heads on masked layers and remove the expected savings; the pilot stops if the backend is not FA3.

## Run list

Defined once in `RUNS` (`experiments/config.py`); every config is generated from it.

| Run ID | Exp. | Depth (layers / width / heads) | KV heads | Rotary | MLP ratio | VE | Budget (updates) | Seed | Compared with |
|---|---|---|---|---|---|---|---|---|---|
| `d8-b0-s1` | E1, E2 | d8 (8 / 512 / 4) | 4 | 1.0 | 4.0 | on | tokens (3,200) | 1 | — |
| `d8-b0-s2` | E1 | d8 | 4 | 1.0 | 4.0 | on | tokens (3,200) | 2 | — |
| `d8-b0-s3` | E1 | d8 | 4 | 1.0 | 4.0 | on | tokens (3,200) | 3 | — |
| `d8-gqa2-s1` | E2 | d8 | 2 | 1.0 | 4.0 | on | tokens (3,200) | 1 | `d8-b0` (mean of the 3 seeds), as the `d8-gqa2` mean |
| `d8-gqa2-s2` | E2 | d8 | 2 | 1.0 | 4.0 | on | tokens (3,200) | 2 | as part of the `d8-gqa2` mean |
| `d8-gqa2-s3` | E2 | d8 | 2 | 1.0 | 4.0 | on | tokens (3,200) | 3 | as part of the `d8-gqa2` mean |
| `d8-gqa2-r75` | E2 | d8 | 2 | 0.75 | 4.0 | on | tokens (3,200) | 1 | `d8-b0`, `d8-gqa2` |
| `d8-gqa2-pm` | E3 | d8 | 2 | 1.0 | 4.5 | on | tokens (3,200) | 1 | `d8-b0`, `d8-gqa2` |
| `d8-gqa4` | E4 | d8 | 1 | 1.0 | 4.0 | on | tokens (3,200) | 1 | `d8-b0`, `d8-gqa2` |
| `d8-gqa2-r25` | E4 | d8 | 2 | 0.25 | 4.0 | on | tokens (3,200) | 1 | `d8-gqa2`, `d8-gqa2-r75` |
| `d8-b0-nove` | E5 | d8 | 4 | 1.0 | 4.0 | off | tokens (3,200) | 1 | `d8-b0` |
| `d8-gqa2-nove` | E5 | d8 | 2 | 1.0 | 4.0 | off | tokens (3,200) | 1 | `d8-b0-nove`, `d8-gqa2` |
| `d8-gqa2-iso` | E6 | d8 | 2 | 1.0 | 4.0 | on | FLOPs (3,333) | 1 | `d8-gqa2`, `d8-b0` |
| `d8-gqa4-iso` | E6 | d8 | 1 | 1.0 | 4.0 | on | FLOPs (3,404) | 1 | `d8-gqa4`, `d8-b0` |

E5's 2×2 interaction `(d8-gqa2 − d8-b0) − (d8-gqa2-nove − d8-b0-nove)` is declared in `INTERACTIONS` (`config.py`) and reported separately.

**14 training runs, 8 distinct architectures** (each FLOP-budget run shares its architecture, and its pilot, with its matched-token twin). Seed groups: `d8-b0` = `d8-b0-s1..s3`; `d8-gqa2` = `d8-gqa2-s1..s3`. A group's mean requires every member to have an eligible result; seeds are never compared individually, and every comparison lists the members it averaged. GQA-2 seeds 2 and 3 use the same data rotations as baseline seeds 2 and 3. Everything not listed above (data, tokenizer, context 2,048, global batch of 131,072 tokens, FLOP budget, optimizer, schedule shape, evaluation sizes) is held fixed across runs at the same depth; only the FLOP-budget runs change the token target. Head dimension is 128 at d8, `SSSL` window pattern, vocabulary 32,768. Seed 1 is used everywhere except the B0 and GQA-2 replicates.

## Cost and time estimate

These are estimates from FLOPs, not measurements. Replace them with pilot numbers.

- **Active parameters (matmul, incl. unembedding):** d8 ≈ 42M. At 10 tokens/param that is ≈ 0.42B tokens.
- **FLOPs:** the declared budget is 1.319×10¹⁷ per run (B0); matched-token GQA runs use 94–96% of it, the FLOP-budget runs ≥ 99.9%.
- **Throughput assumption:** one H100 has 989 TFLOPs peak BF16 dense and needs no gradient all-reduce. Published benchmarks and peak FLOPs suggest about 1.4–2× the previously assumed 2× L40S throughput.
- **Per-run time:** d8 ≈ 5–10 min of training, or ≈ 70–140 min across fourteen runs. Evaluation and inference are additional.

| Block | Time | Cost at $4.43/h |
|---|---|---|
| Data and tokenizer (CPU only) | ≈ 0.5 h | under $0.50 |
| Pilots, 8 architectures (full inference grid, reduced repeats) | measure on GPU | measure on GPU |
| Training, 14 runs (optimizer updates only) | ≈ 1.2–2.3 h | ≈ $5.2–10.4 |
| Interim + final validation and test, 14 runs | ≈ 0.27–0.53 h | ≈ $1.2–2.4 |
| Inference benchmarks, 14 | ≈ 0.42–1.1 h | ≈ $1.9–4.9 |

Provisional planning allowance: **the earlier 1.2–2.5 GPU-hour / $6–12 estimate is no longer a launch budget**: the expanded eight-architecture pilots and numerical preflight add work; use their measured cost table, plus about 0.5 h of CPU preparation. Evaluation and pilot allowances are placeholders, not measured predictions; replace these and training/inference estimates with the d8 pilot's measured costs before relying on the total. The application budget defaults to $42 (`BUDGET_USD`): projections gate the full 14-run suite, and training/benchmark subprocesses have a deadline based on the remaining budget. Configure the intended $42.50 workspace usage limit separately in Modal; the script does not set or verify it. Accounting uses the configured hourly estimate and a $0.50 CPU-preparation allowance, so it is not an exact billing cap. Image builds, repeated preparation, GPU checks, other Apps, and storage can add charges.

If the d8 pilot projects excessive cost or cannot fit the declared settings, stop and revise the plan before launching the full suite. No replacement runs or reduced-token d12 runs are included.

## Comparison rules

- **Matched tokens** (E1–E5) measures the effect of an architecture per token seen: those runs train on B0's token count, and the primary number is the final test BPB.
- **Matched compute** (E6, and E3 by construction) answers the fixed-budget question: each run uses the declared FLOP budget. Every run in both regimes stays within that budget.
- Interim validation curves are not schedule-matched between a matched-token run and its FLOP-budget twin (the latter's learning rate decays later), so only final BPB is compared across regimes. BPB against wall-clock time and estimated FLOPs, read from the validation curves at exactly shared observations and never interpolated, is secondary; between the two regimes there are no exactly shared FLOP observations, so that list is empty.
- **Interactions** (E5): `(a − b) − (c − d)` on test BPB of the group means, resolved only if `|Δ| > 2·s·√(Σ 1/n_i)`, the two-sigma bound for four independent means.
- **Resolution threshold:** two runs differ only if the test-BPB gap exceeds `2√2 · s` from E1. Smaller gaps are reported as unresolved, not as ties or wins. Comparisons against or of a seed group use its mean; the secondary `resolved_by_means` flag (see E2) accounts for the number of seeds on each side.
- **Known weakness of the primary rule:** `s` from three seeds has 2 degrees of freedom, so `2√2·s` is not a 95% bound. Simulated with no true effect, it flags about 18% of single-vs-single, 13% of single-vs-3-seed-mean and 7% of mean-vs-mean contrasts as resolved; across the suite's ~17 contrasts, two or three spurious "resolved" flags are expected even if nothing has an effect, and all verdicts move together with the one estimate of `s`. A true effect of 2s against a 3-seed mean is flagged only about 40% of the time.
- **Sensitivity analysis (declared before any results):** the report also pools the seed SD over the `d8-b0` and `d8-gqa2` groups (`s_p`, df = 4 when complete) and gives, for every contrast and interaction, a Student-t 95% interval `Δ ± t_df · s_p · √(Σ 1/n_i)` with its own `resolved` flag (`sensitivity` in `comparison.json`). It also states the number of contrasts (`n_contrasts`); no multiplicity correction is applied. The primary verdict remains `resolved`; where the primary and sensitivity verdicts disagree, the write-up says so.
- **Reporting:** absolute and relative test BPB differences (and validation, for curves) against each run's `compare_to` references in the run list, each with a resolved/unresolved flag; parameter counts (total and scaling); tokens; estimated FLOPs; training-only peak VRAM (evaluation peak separately); steady-state and whole-phase training tokens/s.
- **No hyperparameter tuning** is done per variant. The inherited H100-tuned optimizer settings are used for all, so a GQA loss could partly reflect that the settings were tuned for MHA. This is listed as a limitation. A future extension could test B0 and GQA-2 at 2× and 0.5× matrix LR at d8 (four additional runs); these runs are outside this suite and its budget.
- **Negative and null results** are reported in full, with the pre-registered prediction next to the outcome.

## Baseline definition

B0 is the nanochat ClimbMix leaderboard configuration immediately before the autoresearch rounds (`324e69c`; round 1 at `6ed7d1d`, round 2 at `a825e63`), re-scaled to d8. The historical launch used H100 FP8; this study uses BF16 throughout. It retains `SSSL` windowed attention, half-context short windows, base-10,000 RoPE, value gates and the historical optimizer grouping, and excludes later architectural improvements. The learning-rate multiplier keeps the historical batch scaling against 524,288 tokens (a 0.5 multiplier at our global batch of 131,072), and the schedule is mapped onto the token target: no warmup, constant for the first 50% of updates, then linear decay to zero; Muon weight decay decays linearly to zero and Muon momentum warmup stays step-based (0.85 → 0.95 over 300 updates). Source: [local leaderboard history](../dev/LEADERBOARD.md), especially Runs 4–6.

## Model implementation and configuration

One configurable `nanochat/modified_gpt.py` derived from the pinned commit exposes `GPTConfig.n_head`, `GPTConfig.n_kv_head` and `rotary_fraction`, plus `mlp_ratio` (MLP hidden width = `mlp_ratio × n_embd`, default 4.0 = historical; the hidden width must be an exact multiple of 128). JSON configs under `experiments/configs/` carry all settings and are generated by `experiments/make_configs.py` from the `RUNS` table, so none are edited by hand. For partial RoPE, frequencies are built for the rotated prefix only, that prefix of Q/K is rotated, and the remaining suffix is concatenated unchanged. The rotary dimension must be an exact even integer (no silent rounding), and fraction 1.0 must reproduce the historical behaviour.

Required checks before the full runs: full-RoPE equivalence to the historical code; untouched partial-RoPE suffixes; GQA projection and cache shapes; cached versus full-sequence decoding; checkpoint reload and generation; the selected attention backend (FA3 on H100) handling GQA without expanding K/V.

## Implementation status

Implemented and covered by CPU tests (`tests/test_experiments.py`, `test_data.py`, `test_modified_gpt.py`, `test_report.py`, `test_visualise.py`, `test_preflight.py`, `test_modal_app.py`, `test_pipeline_smoke.py`):

- `config.py`: `RUNS` table (single source of truth, with `ve` and `budget` per run), `INTERACTIONS`, token-target, `flop_budget` and wall-clock-cap schema, three evaluation sizes, `mlp_ratio`, `value_embeds`, step-based `schedule`, `validate_suite`.
- `make_configs.py`: generates the 14 JSON configs; per-depth policy (token targets, microbatch, cadence) lives in its `DEFAULTS` and is **provisional until the pilot**. It derives the FLOP budget from B0 via `GPT.estimate_flops` on the meta device, and the FLOP-budget runs' token targets from it.
- `data.py`: held-out test shard (`shard_06541`), seed-dependent training order, epoch guard, built on the unmodified historical dataloader.
- `train.py`: fixed-token stop, refusal to start a run over the FLOP budget, 2 h safety cap, interim/final validation and a separate test pass, `data_order.json`, `mlp_ratio` and `value_embeds` reset in the weight-decay reference models; next-microbatch prefetch after each backward (data order unchanged, checked by a bit-identical resume test); one loss readback and finiteness check per update; training-only `peak_vram_bytes` and separate `eval_peak_vram_bytes`; `result.json` records `flop_budget`, `flops_per_token` and `steady_tokens_per_second`.
- `modified_gpt.py`: `GPTConfig.mlp_ratio` (default 4.0 = historical) and `GPTConfig.value_embeds` (default true = historical; false removes the value embeddings and their gates, and their now-empty optimizer groups).
- `benchmark.py`: analytic and allocated KV-cache bytes per workload; wider batch/prompt sweep in the generated configs.
- `report.py` / `visualise.py`: seed noise `s`, threshold `2√2·s`, resolved flags, FLOP-budget eligibility, the E5 interaction, the pooled-t sensitivity analysis and `n_contrasts`, steady-state speed comparisons, per-depth figures, `dose_response`, `budget_comparison`.
- `preflight.py`: per-architecture GPU checks of cached decoding, FA3 versus SDPA logits and gradients, and compiled versus eager gradients (CPU tests cover the comparison helpers only).
- `run.py` / `modal_app.py`: provenance limited to `RESULT_FILES`; deterministic errors returned rather than retried; per-run deadline; stop on a non-comparable run.

Still to do on the GPU node (not verifiable locally): download the pool and test shard, run the GPU preflight (confirm the 5% gradient tolerance holds; on CPU the worst case was about 2%), run the eight architecture pilots, measure the separated training and evaluation VRAM peaks and the steady-state throughput, freeze `DEFAULTS` and regenerate configs, check that FA3 is selected, so GQA runs without expanding K/V, and confirm `kv_cache_bytes_allocated` equals the analytic value on GPU.

Known confounds to report: the historical value embeddings shrink with KV heads (total parameters cannot be matched in E3); the value-embedding gate `ve_gate` is a Muon matrix of shape `(n_kv_head, 32)`, which at GQA-4 is a single row, so Muon's orthogonalised update changes character with the KV-head count; the FLOP estimate charges each token the full attention window rather than the causal average, which slightly understates GQA's relative saving and hence the FLOP-budget runs' extra tokens; seed-1 variants share their data order with `d8-b0-s1` and `d8-gqa2-s1`, so their contrasts against seed means are partly paired and the threshold is conservative for them; Muon rescales the `c_fc` learning rate with its aspect ratio (about +6% in the 4.5× MLP); the seed varies both initialisation and training-data order, so `s` includes data-order noise; the FLOP estimate ignores embedding and value-embedding lookups, so value embeddings are "free" by the budget's measure although they cost memory and bandwidth.

## Execution order

1. On the GPU node: prepare the pinned worktree, download the shard pool and shard 06541, train the tokenizer, and run the preflight checks above (commands under "Pipeline reference").
2. Pilot all eight unique d8 architectures (10 updates each; the automated path uses the declared microbatch, while manual fit checks may sweep device batches): measure tokens/s, peak VRAM and evaluation cost; freeze `DEFAULTS` in `make_configs.py` (token targets, microbatch, cadence) and regenerate the configs; this also re-derives the FLOP budget and the FLOP-budget runs' targets. Commit before E1 so the budget is visibly declared before any result.
3. Run E1 (`d8-b0-s1/s2/s3`). Compute `s` and the threshold with `experiments.report`.
4. Run E2 (including the three GQA-2 seeds), E3, E4, E5 and E6 (the remaining d8 runs).
5. Collect the inference results, which each run produces automatically after training, and rerun `experiments.report` and `visualise.py`.
6. Write up predictions versus outcomes.

## Run record

For every run, save: run ID, experiment, hypothesis, parent config, code commit/diff, launch command, seed, resolved config; GPU model/count, software versions, precision, attention backend; total and scaling parameter counts, KV heads, rotary fraction; dataset/tokenizer manifests and split manifests; tokens, estimated FLOPs and share of the FLOP budget, step time, throughput, peak VRAM; validation curve, final test BPB; checkpoint path; GPU-hours and cost.

| Run ID | Seed | Params (total / scaling) | Tokens | Train seconds | Test BPB | Train tok/s | Status |
|---|---|---|---|---|---|---|---|
| `d8-b0-s1` | 1 | — | — | — | — | — | Planned |
| `d8-b0-s2` | 2 | — | — | — | — | — | Planned |
| `d8-b0-s3` | 3 | — | — | — | — | — | Planned |
| `d8-gqa2-s1` | 1 | — | — | — | — | — | Planned |
| `d8-gqa2-s2` | 2 | — | — | — | — | — | Planned |
| `d8-gqa2-s3` | 3 | — | — | — | — | — | Planned |
| `d8-gqa2-r75` | 1 | — | — | — | — | — | Planned |
| `d8-gqa2-pm` | 1 | — | — | — | — | — | Planned |
| `d8-gqa4` | 1 | — | — | — | — | — | Planned |
| `d8-gqa2-r25` | 1 | — | — | — | — | — | Planned |
| `d8-b0-nove` | 1 | — | — | — | — | — | Planned |
| `d8-gqa2-nove` | 1 | — | — | — | — | — | Planned |
| `d8-gqa2-iso` | 1 | — | — | — | — | — | Planned |
| `d8-gqa4-iso` | 1 | — | — | — | — | — | Planned |

No results have been produced. Every number in the plan above is a prediction or an estimate.

## GPT-2 reference (optional)

Do not label 0.71854 (nanochat Run 4's validation BPB, on its own distribution) as GPT-2's BPB. The repo's GPT-2 threshold is CORE = 0.256525, a separate metric. Our d8 models are far too small and short-trained to approach it, so GPT-2 is not a target; if a directly comparable BPB number is not available, omit it.

---

# Pipeline reference

## Run on Modal

`modal_app.py` runs everything below in one job on an H100, with stage-aware recovery for handled failures. Hard container crashes require the reconciliation procedure below. Commit your changes first: the commit and any diff go into each run's provenance, and changing a result-affecting file (`RESULT_FILES` in `experiments/run.py`: the model, `train.py`, `data.py`, `benchmark.py`, `backend.py`, `config.py`, `checkpoints.py` and the two patched historical modules) midway makes resumes refuse the changed runtime. Editing `report.py`, `visualise.py` or other analysis code does not.

```bash
modal run modal_app.py::image_check          # CPU only: image builds, configs resolve
modal run modal_app.py::gpu_check            # H100 available, FA3 loads
modal run modal_app.py --pilot-only --experiment d8-v3  # data, GPU checks, eight pilots, cost table
modal run --detach modal_app.py --experiment d8-v3      # reuse pilots, run the full suite
```

Re-running validates the saved config and runtime source hashes before skipping a completed run, resuming a checkpoint, or replacing incomplete outputs. Changed full-run configs or result-affecting source (`RESULT_FILES`) stop the job; use a new `--experiment` name for a new revision. Artifacts live under `/vol/experiments/<name>/artifacts`. Prepared data/tokenizer remain shared at `/vol/nanochat`, and all namespaces share the durable budget ledger. Pilots live under `/vol/experiments/<name>/pilots/<config-and-source-fingerprint>`, so editing `DEFAULTS` and regenerating configs automatically starts fresh pilots without redownloading data. Old pilot evidence is retained. Only one preparation or suite writer can run across Apps: a named Modal Dict (`nanochat-experiments-locks`) atomically claims its `writer` key, and competing writers fail before touching the Volume.

The durable budget ledger is `/accounting/suite.json` on the `nanochat-experiments` Volume, outside deletable run directories. It imports legacy per-run totals once. Each suite attempt reserves the remaining budget and commits that reservation before work starts; on normal exit, including handled failures, it settles to elapsed suite time. Retries therefore retain earlier costs. Training deadlines kill the whole process group, including torchrun workers. Lower the ceiling with `--budget-usd` if other Modal usage has consumed your allowance.

After a hard kill, automatic recovery intentionally stops: the writer lock may remain and the ledger may retain its full reservation. Stop **all** Apps using this Volume and confirm no worker remains. Inspect Modal usage and the ledger, then explicitly reconcile the cumulative cost of all prior preparation/pilot/suite attempts. Never release a lock while an old worker is active.

```bash
modal volume get nanochat-experiments /accounting/suite.json suite-accounting.json
# Example only: replace 5.25 with the reconciled cumulative spend.
modal run modal_app.py::recover --spent-usd 5.25 --confirm-all-stopped
```

Recovery saves the old ledger and writer identity in `/accounting/recovery_<timestamp>.json`, commits the corrected ledger, then releases the stale lock. It does not reset experiment artifacts or authorize extra budget.

Every suite invocation performs GPU numerical preflight on the current container: full versus cached FA3 across the sliding-window boundary, FA3 versus SDPA logits, FA3 versus SDPA loss and gradients, and the `torch.compile`d model versus eager FA3 loss and gradients (training mode, BF16 autocast, the full 2,048-token training length so the sliding window is exercised; per-tensor relative gradient error at most 5%, with the worst parameter recorded), for all eight architectures. The compile checks add roughly 4–12 minutes (about $0.3–0.9). Every actual training/benchmark process requires FA3 pinned at `de87b9b5af06dd9984df595bef90b2eba44b181a`; silent SDPA fallback is a failure. The kernel identity is recorded in provenance.

The eight pilots retain the full inference batch/prompt/decode grid, with one warmup and one measured repeat. Every workload in the grid must succeed: last-token prefill keeps the largest d8 workload near 12 GiB, so a pilot OOM is a regression and blocks the suite. The cost gate uses each architecture's measured training, evaluation, and inference durations, scales inference repeats, and adds a safety margin. It does not sweep microbatch sizes or automatically change the declared token target. Freeze any calibration changes and regenerate configs before the full run.

Everything below `/vol` lives on the `nanochat-experiments` Volume; it does not automatically sync locally. The smaller results archive excludes checkpoint tensors. A separate uncompressed archive contains the final checkpoint and resume state for every run. Only the newest two checkpoint generations per run are kept in the Volume.

```bash
modal volume get nanochat-experiments experiments/d8-v3/artifacts/d8-b0-s1/metrics.jsonl -
modal volume get nanochat-experiments experiments/d8-v3/results.tar.gz .
modal volume get nanochat-experiments experiments/d8-v3/checkpoints.tar .
mkdir -p experiments/downloaded-d8-v3
tar xzf results.tar.gz -C experiments/downloaded-d8-v3
tar xf checkpoints.tar -C experiments/downloaded-d8-v3
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
python experiments/run_all.py --dry-run                      # resolves all 14 runs, creates nothing
python experiments/run.py --config experiments/configs/d8-b0-s1.json
python experiments/run_all.py --runs d8-b0-s1 d8-b0-s2 d8-b0-s3   # E1 only; omit --runs for the whole suite
python experiments/run.py --config experiments/configs/d8-b0-s1.json --resume 1600
```

The suite validator checks every config against `RUNS`, that configs at one depth differ only in run ID, seed, KV heads, rotary fraction, MLP ratio, value embeddings and (FLOP-budget runs only) token target, that they share one `flop_budget`, that no (model, seed, token target) triple repeats, and that the suite uses one worker, global batch 131,072 and no step cap. Existing outputs are rejected unless an individual resume is requested. Checkpoints are saved every 400 updates; only the newest two generations are kept. The launcher kills a process whose log has been silent for 30 minutes and reports it as a failure; on Modal any non-zero exit stops the suite (`run_failed`) rather than retrying, and so does any Python exception in the orchestrator (`error`, with the traceback in the Modal logs), while container loss (preemption) retries and resumes from the newest checkpoint. A suite run whose `stop_reason` is not `token_target` (for example `wallclock_cap`) stops the suite (`run_not_comparable`), and each suite run has a deadline of 1.5× its pilot-projected time (`run_overtime`) as well as the remaining budget. Modal and gRPC infrastructure errors are still retried, and the retry reclaims its own lock and budget reservation. `modal run` exits non-zero unless the status is `complete` or `pilot_ok`. A `run_not_comparable` run blocks every later invocation of that namespace until its run directory is deleted from the Volume (it then reruns) or a new `--experiment` name is used. Each launch trains, evaluates (validation then test) after training, then benchmarks the final checkpoint in a fresh process. A failed process stops the sweep. `run_all.py` writes `comparison.json` and `tables.md` at the end.

## Stopping, schedule and optimizer policy

Training runs to `target_tokens / total_batch_size` optimizer updates; for FLOP-budget runs that target is the largest whole number of updates within `flop_budget`. The learning-rate schedule is a function of step progress: no warmup, constant for the first 50%, then linear decay to zero; Muon weight decay decays linearly to zero with progress; Muon momentum warmup stays step-based (0.85 → 0.95 over 300 updates). A wall-clock cap (`wallclock_cap_seconds`, 7,200 s) is a safety stop only: a run that ends on it is flagged `stop_reason = wallclock_cap` and is not comparable. Learning rates keep the historical batch scaling against 524,288 tokens (a 0.5 multiplier at the global batch of 131,072 before model-width scaling). Initial weight-decay scaling is resolved from the MHA baseline of the same depth and the historical d12 reference, then frozen across variants (with `mlp_ratio` reset to 4.0 and `value_embeds` to true in those references). BF16 replaces the leaderboard's H100 FP8; FP8 is never enabled.

Evaluation uses fixed bytes: interim validation (`interim_tokens`) on the validation shard every `eval_every` updates, then after training a final validation pass (`final_tokens`) and a test pass (`test_tokens`) on the held-out test shard, outside the training clock. Tuning decisions use validation only; the test number is reported once.

Resume verifies checkpoint SHA-256 manifests, restores per-rank optimizer and RNG state, and replays consumed microbatches only when training remains unfinished. Checkpoints are staged and published with their completion marker last; completed generations are immutable. `training_complete.json` skips data replay after training, each final evaluation split is saved independently, and a completed `result.json` permits benchmark-only recovery. Final BPB is authoritative in `result.json`; checkpoint metadata stays immutable. Only the wall-clock cap depends on cumulative seconds; lost time after an interruption is not charged back automatically.

## Observe a running experiment

| Observation | Cadence | File in the run directory |
|---|---|---|
| Step, elapsed time, worker-0 loss, tokens, estimated FLOPs, step duration, throughput, LR multiplier, training peak allocated VRAM (evaluation excluded) | Every update | `metrics.jsonl`, `train.log` |
| GPU utilisation and memory (`nvidia-smi`) | Every 10 updates | `metrics.jsonl` |
| Distributed validation BPB | Every `eval_every` updates, and after training (`final: true`) | `metrics.jsonl` |
| Test BPB (`kind: "test"`) | Once, after training | `metrics.jsonl`, `result.json` |
| Hardware/software, effective optimizer groups, parameter counts, manifests, data order | Before training | `hardware.json`, `data_manifest.json`, `data_order.json`, `launch.json`, `config.json` |

Follow `train.log` with `tail -f`, or rerun `visualise.py` on a local copy. Training loss is worker 0's; validation and test BPB are averaged across workers. Peak VRAM is the per-GPU maximum; `peak_vram_bytes` covers training only and `eval_peak_vram_bytes` the evaluations, which run uncompiled and materialise full FP32 logits.

## Outputs and comparisons

Each run directory under `experiments/artifacts/` holds the frozen config and source, provenance, logs, curves, checkpoints, `result.json` (including `val_bpb`, `test_bpb`, `scaling_parameters`, `target_tokens`, `data_epoch`), `inference.json`, and rental accounting. `data_manifest.json` is seed-independent (pool shards, validation and test hashes, tokenizer hashes) and is compared across runs at one depth; the per-run training order is in `data_order.json`. `data_epoch > 1` means a run saw repeated data and is flagged.

`report.py` retains raw capped/repeated-data/over-budget results with explicit exclusion reasons, but excludes them and token-count mismatches from quality comparisons, seed noise and final-quality charts. Seed groups are compared once, as their means (never seed by seed). Ancillary curve, elapsed-time and inference comparisons use one actual run per side, named in `field_subject` and `field_references` (seed 1 of a group); final BPB and training-speed comparisons use the group means. Each comparison records `members`, `reference_members`, `n_run`, `n_ref`, `resolved_by_means` and the `sensitivity` interval; the payload adds `gqa2_noise`, `pooled_noise` and `n_contrasts`. `report.py` computes the d8 baseline seed standard deviation `s` (ddof = 1) from test BPB, the threshold `2√2·s`, and for every `compare_to` pair the test and validation BPB differences, relative differences, a `resolved` flag (`|Δ|` above the threshold), steady-state training-speed ratio (`training_speedup`, with `whole_phase_training_speedup` as secondary), total and scaling parameter counts, exact matched-token and matched-FLOP observations between interim evaluations only (never interpolated, never pairing a final evaluation with an interim one, and empty with a `matched_observations_note` when the two runs are in different budget regimes), and inference speedups including KV-cache byte ratios. It also writes `interactions` (the E5 2×2: effect with, effect without, interaction, threshold, resolved) and a FLOPs-as-%-of-budget column in `tables.md`. Newly generated `comparison.json` files omit the former `scale_transfer` field.

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

Figures are written as PNG and SVG under `<output-root>/figures/`. For d8 (`_d8`): `learning_curves`, `training_performance`, `final_comparison` (final test BPB, pairwise differences against each run's `compare_to` references with the ±2√2·s noise band, throughput, peak VRAM), `quality_speed`, and `inference_comparison` (prefill/decode throughput, TTFT, peak VRAM and allocated KV cache by prompt length and batch size). Also: `dose_response` (BPB against KV heads and rotary fraction) and `budget_comparison` (E6: final BPB against estimated FLOPs, with arrows from each matched-token run to its FLOP-budget twin and the declared budget marked). Line style: solid = value embeddings on at the token target, dotted = value embeddings off, dashed = FLOP-budget run. The seven figures are each exported as PNG and SVG; no scale-transfer chart is generated. Learning and performance figures work while runs are in progress; missing runs and provenance are flagged; data, code or hardware differences reject the comparison. Charts show observations and descriptive smoothing, with no automatic causal attribution; unresolved differences are not ties.
