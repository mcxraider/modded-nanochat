# d8-v5 local export

The complete 14-run experiment succeeded. All 56 final checkpoint file checksums pass.

- [Detailed evaluation](audit/report.md): cost reconciliation, execution checks, scientific findings and limitations.
- [Per-run CSV](audit/run_summary.csv) and [machine-readable audit](audit/audit.json).
- [Original results tables](artifacts/tables.md).
- [Original results archive](export/results.tar.gz); [repaired results archive](export/results-repaired.tar.gz) also restores the accidentally omitted provenance source.
- [Final checkpoints archive](export/checkpoints.tar), 9,700,812,800 bytes: all 14 models, optimizer/runtime state and checksum markers.
- [Pilot evidence](pilots/9a34115f44463cc7/preflight.json), including all eight pilot checkpoints.
- [Timestamped Modal logs](export/modal-app.log), [cumulative estimated spend ledger](export/suite-ledger.json), and tokenizer in `export/tokenizer/`.

Measured run pipelines cost $10.2463 at the configured $4.43/hour. The approximately $21 figure was a conservative projection; the configured default ceiling was $42. The recorded cumulative estimated ledger is $12.67905, including earlier work; it is not an invoice.

GQA-2 trained 2.8% faster and halved KV-cache memory, but test BPB worsened by 0.00516 against the baseline mean. See the detailed evaluation for uncertainty and inference results.

To restore final checkpoints, run from the repository root:

```sh
tar -xf results/d8-v5/export/checkpoints.tar -C results/d8-v5
```

Checkpoints then live under `results/d8-v5/artifacts/<run>/checkpoints/`. Loading requires the pinned historical runtime plus saved overlays. No GPU experiment was launched for this audit. The raw training dataset and redundant penultimate full-run checkpoint generations were not exported.

`export/SHA256SUMS` records file hashes at delivery, with paths relative to this directory. Verify from this directory with `shasum -a 256 -c export/SHA256SUMS`. The original evidence is preserved separately from the repaired archive and supplemental audit figures.
