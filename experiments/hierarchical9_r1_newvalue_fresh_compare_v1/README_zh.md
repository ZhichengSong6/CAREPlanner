# R1 vs paired-trained NEW_VALUE frozen fresh-holdout comparison

Reuses the exact persisted `evaluation_fresh_holdout_v1/starts.jsonl` and verifies its SHA256 against the original manifest. No resampling.

Models:
- original R1 scratch-50k champion
- paired-training `new_value/final.pt`, required to have parent SHA equal to fixed R1

Both use the unchanged `runtime_probe.make_probe/run_probe` solver parameters and analytic FOV oracle. Outputs local/uniform/all FOV pass, paired wins/losses, per-sensor results, and failure stages.

This is FOV-only solver evidence. LOS/collision/trajectory/actual-seen remain NOT_RUN.
