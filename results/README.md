# Saved results

All numbers, tables and figures in [`report/report.tex`](../report/report.tex) come from the files in this folder.
To check that they still agree with the report and the code, run this from the repository root:
`uv run --frozen python report/verify_reproduction.py`.

| Path | Contents |
|---|---|
| [`d8-v5/artifacts/<run>/`](d8-v5/artifacts/) | One folder per run (14 in total): `config.json` (resolved config), `result.json` (final val/test BPB, tokens, estimated FLOPs, throughput, VRAM, cost), `metrics.jsonl` (training and validation curve), `evaluation_*.json`, `inference.json` (168-cell inference grid in total), `launch.json` (launch commit, pinned base commit, SHA-256 of every file that can affect results), `data_manifest.json` / `data_order.json` (shard lists and seeded order), `hardware.json`, `rental_accounting.json`, `train.log` / `benchmark.log`, and `source/` (a snapshot of the code that ran) |
| [`d8-v5/artifacts/comparison.json`](d8-v5/artifacts/comparison.json), [`tables.md`](d8-v5/artifacts/tables.md) | Output of `python -m experiments.report`: seed noise, resolution threshold, every registered contrast, pooled-*t* sensitivity, paired analysis and E5 interaction |
| `d8-v5/artifacts/figures/` | Output of `experiments/visualise.py` (exploratory figures; the report's figures are produced by `report/make_figures.py`) |
| [`d8-v5/audit/`](d8-v5/audit/) | Post-run audit of the exported evidence: `evaluate.py` → `audit.json`, `run_summary.csv` (per-run timing table) and `check-output.txt`; `report.md` (cost reconciliation, execution checks) |
| [`d8-v5/pilots/9a34115f44463cc7/`](d8-v5/pilots/9a34115f44463cc7/) | 10-update pilots (one per architecture) and `preflight.json` (H100 numerical checks); summarised in [`pilot_d8-v5.md`](pilot_d8-v5.md) / [`.json`](pilot_d8-v5.json) |
| [`d8-v5/export/`](d8-v5/export/) | Raw Modal export: `results.tar.gz` (original results archive), `results-repaired.tar.gz` (the same archive with `source/experiments/checkpoints.py` restored), `recovered-source/`, `tokenizer/` (the trained 32,768-token BPE tokenizer), `modal-app.log` (timestamped logs), the launch-time copies of `modal_app.py` and the experiment plan, spend ledgers, and `SHA256SUMS` |

## Not committed to git

- `d8-v5/export/checkpoints.tar` (9,700,812,800 bytes, all 14 final checkpoints with optimizer state). The suite
  wrote it to the Modal volume `nanochat-experiments` at `experiments/d8-v5/checkpoints.tar`. It can be fetched with
  `modal volume get nanochat-experiments experiments/d8-v5/checkpoints.tar results/d8-v5/export/`, as long as the
  volume still exists.
- The pilots' 10-update `model_*.pt` / `optim_*.pt` files.
- The ClimbMix training corpus (see the readme's "Data" section to download it).

`d8-v5/export/SHA256SUMS` still records the hashes of these files. To verify the files that are present, run from
`results/d8-v5`: `shasum -a 256 -c --ignore-missing export/SHA256SUMS`. The verification script runs the same check
and accepts only these files as missing.

## Notes

- The files under `d8-v5/` are kept exactly as exported, because `SHA256SUMS` covers them. They include
  [`d8-v5/README.md`](d8-v5/README.md) and [`d8-v5/audit/report.md`](d8-v5/audit/report.md). Both were written
  before the final report, so the audit's note that "`report/report.tex` still contains TBD fields" is out of date.
  Its links to `export/checkpoints.tar` refer to the file that is not in git (see above).
- Rerunning the audit scripts in `d8-v5/audit/` overwrites `audit.json` and `run_summary.csv` in place. Without
  `checkpoints.tar` they also drop the checkpoint checks, so run them only on a copy of the folder.
- Measured cost of the 14 run pipelines: 8,326.5 s on one H100 (2.31 GPU-hours) = **$10.25** at the configured
  $4.43/h; training alone took 6,160 s. Data preparation, preflight and pilots are separate (about $2.08,
  see `pilot_d8-v5.md`).
