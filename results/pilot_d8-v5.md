# Pilot results: experiment `d8-v5`

These are the 10-update pilots, one per distinct architecture, run before the full suite with
`modal run modal_app.py --pilot-only --experiment d8-v5`. Artifacts are on the Modal volume at
`/vol/experiments/d8-v5/pilots/9a34115f44463cc7/` (the folder name is the config and source fingerprint). The raw
numbers below come from the console output and are also saved in [`pilot_d8-v5.json`](pilot_d8-v5.json).

Pilots are preliminary runs. They are **excluded from the per-run compute budget** and are not used for any quality
comparison. They checked that the setup works (H100, FA3 selected, no inference OOM, decode working set below
half the KV cache, data-loader wait ≤ 2%, VRAM < 90%) and produced the cost projection that gated the full suite.
All eight pilots passed every check, so the full 14-run suite was launched under the same namespace with
**unchanged configs**: no calibration change was made after the pilots.

> TODO (when pulled from the volume): add `preflight.json` (worst FA3-vs-SDPA and compiled-vs-eager gradient
> errors), and from each pilot `result.json` (`peak_vram_bytes`, `eval_peak_vram_bytes`, data-wait fraction) and
> `inference.json` (decode working set vs cache).

## Model IDs and acronyms

| Run ID | Short name | Meaning |
|---|---|---|
| `d8-b0-s{1,2,3}` | **B0** | Baseline: multi-head attention (4 KV heads), full RoPE, MLP ratio 4, value embeddings on. Seeds 1–3 |
| `d8-gqa2-s{1,2,3}` | **GQA-2** | Grouped-query attention, 2 KV heads (2 query heads share each K/V head). Seeds 1–3 |
| `d8-gqa2-r75` | **GQA-2-R75** | GQA-2 with RoPE on 75% of each head's dims (96 of 128) |
| `d8-gqa2-r25` | **GQA-2-R25** | GQA-2 with RoPE on 25% of each head's dims (32 of 128) |
| `d8-gqa2-pm` | **GQA-2-PM** | GQA-2, *parameter-matched*: MLP widened 4.0× → 4.5× to return the removed K/V parameters |
| `d8-gqa4` | **MQA** (GQA-4) | 1 KV head shared by all 4 query heads (multi-query attention) |
| `d8-b0-nove` | **B0-noVE** | B0 without value embeddings (and their gates) |
| `d8-gqa2-nove` | **GQA-2-noVE** | GQA-2 without value embeddings |
| `d8-gqa2-tok` | **GQA-2-tok** | GQA-2 trained on B0's token count (1,680 updates) instead of the FLOP budget |
| `d8-gqa4-tok` | **MQA-tok** | MQA trained on B0's token count (1,680 updates) |

`-sN` is the seed, which sets both the initialisation and the training-shard rotation.

## Training throughput (steady state, pilot)

The tokens/s value is the median over updates 3–10 of one 10-update pilot, from `resume.pilot_tokens_per_second`.
FLOPs per token are from `GPT.estimate_flops`. "Achieved" is tokens/s × FLOPs/token, and MFU is that against the
H100's 989 TFLOP/s dense BF16 peak.

| Pilot | Tokens/s | Speed vs B0 | B0 FLOPs / own FLOPs per token | Achieved TFLOP/s | MFU | Updates in budget | Scaling / total params |
|---|---|---|---|---|---|---|---|
| B0 | 1,204,570 | 1.000 | 1.000 | 379 | 38.3% | 1,680 | 41.9M / 125.8M |
| GQA-2 | 1,223,271 | 1.016 | 1.042 | 369 | 37.4% | 1,750 | 39.8M / 90.2M |
| GQA-2-R75 | 1,230,600 | 1.022 | 1.042 | 372 | 37.6% | 1,750 | 39.8M / 90.2M |
| GQA-2-R25 | 1,230,108 | 1.021 | 1.042 | 371 | 37.6% | 1,750 | 39.8M / 90.2M |
| GQA-2-PM | 1,186,941 | 0.985 | 1.000 | 373 | 37.8% | 1,680 | 41.9M / 92.3M |
| MQA | 1,270,563 | 1.055 | 1.064 | 376 | 38.0% | 1,787 | 38.8M / 72.4M |
| B0-noVE | 1,228,083 | 1.020 | 1.000 | 386 | 39.1% | 1,680 | 41.9M / 58.7M |
| GQA-2-noVE | 1,242,196 | 1.031 | 1.042 | 375 | 37.9% | 1,750 | 39.8M / 56.6M |

## Cost gate

The projection includes training, interim and final evaluation, the inference benchmark and a fixed 420 s
overhead per run, all at $4.43/h. The application budget was $42.

| Run | Projected training (min) | Projected cost (USD) |
|---|---|---|
| `d8-b0-s1` / `s2` / `s3` | 6.9 each | 1.26 each |
| `d8-gqa2-s1` / `s2` / `s3` | 9.2 each | 1.74 each |
| `d8-gqa2-r75` | 7.0 | 1.26 |
| `d8-gqa2-pm` | 6.9 | 1.23 |
| `d8-gqa4` | 6.8 | 1.23 |
| `d8-gqa2-r25` | 6.9 | 1.23 |
| `d8-b0-nove` | 6.7 | 1.21 |
| `d8-gqa2-nove` | 6.9 | 1.23 |
| `d8-gqa2-tok` | 8.9 | 1.72 |
| `d8-gqa4-tok` | 6.5 | 1.20 |
| **14 runs** | **104.9** | **19.31** |

Spent before the suite (data preparation allowance, preflight and pilots): **$2.08**. Projected total:
**$21.38**, about half the $42 ceiling, so the gate passed. Every projected training time is far below the
7,200 s wall-clock safety cap.

## Observations

1. **GQA saves FLOPs but turns only part of the saving into speed.** GQA-2 has 4.2% fewer estimated FLOPs per token
   but is only 1.6% faster. MQA has 6.4% fewer and is 5.5% faster. Achieved TFLOP/s drops from 379 (B0) to 369
   (GQA-2). The K/V projection matmuls that GQA shrinks (512→256 output columns) use the tensor cores less
   efficiently than the square 512×512 ones, and attention itself is not reduced: FA3 still runs 4 query heads.
   H2b ("GQA-2 throughput gain under 5%") already looks likely to hold.
2. **Value embeddings cost real time that the FLOP budget does not count.** B0-noVE has *the same* FLOP estimate as
   B0, yet it is 2.0% faster (GQA-2-noVE vs GQA-2: +1.5%). The embedding lookups, their gates and the extra
   AdamW state are excluded from `estimate_flops`. So the FLOP-matched budget slightly favours architectures with
   value embeddings when measured in wall-clock time. This limitation of the measure should be stated in the
   report.
3. **The parameter-matched control is slower at equal FLOPs.** GQA-2-PM is 1.5% slower than B0 despite being
   FLOP-matched to 0.0005%: the 2,304-wide MLP is less matmul-friendly than 2,048, and there are uneven K/V shapes.
   Under a wall-clock budget, PM would see about 1.5% fewer tokens than B0.
4. **Partial RoPE makes no measurable speed difference.** R75 and R25 are 0.5–0.6% faster than full-RoPE GQA-2, which
   is within what one 8-sample pilot can resolve. H2c ("< 2% throughput difference") is consistent with this.
5. **Anomaly to check in the full runs: the projected time for GQA-2.** `d8-gqa2-s1` (and so `-s2`, `-s3` and `-tok`, which reuse
   its pilot) is projected to take 9.2 min of training, whereas R75 and R25, at the same speed and update count, are projected
   to take 7.0 min. Pure training at the pilot speed is about 6.2 min for both. The difference comes from the
   pilot's *evaluation* timing: the projection implies about 95k tokens/s of evaluation for the GQA-2 pilot
   versus about 350k for the others. Since GQA-2 and R75 differ only in the rotary fraction, this is almost certainly a
   one-off in that pilot's evaluation (a cold cache or first read of the evaluation shards), not an architectural
   effect. It only inflates the cost projection and the per-run deadline (1.5× projection), so it does no
   harm. Check `evaluation_seconds` in the full runs' `result.json` to confirm.
6. **Precision of these numbers.** Each throughput is one 10-update measurement with no repeat, so differences of 1–2%
   are not resolved. The full runs' `steady_tokens_per_second` (median over about 1,660 updates) replaces these
   values in the report.
