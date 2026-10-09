# R1 vs full-scale V4 V2-50K frozen fresh comparison

This is the formal paired learned-branch solver comparison. It reuses exactly the original R1 fresh-holdout starts and does not resample.

Models:
- R1: original scratch-50k private-tail H9.
- V2-50K: full-scale Global+V3+V4 random-init 50k checkpoint.

The runtime probe/solver semantics are unchanged. The run reports local/uniform/all FOV pass, per-sensor rates, paired V2-only/R1-only outcomes with exact sign tests, failure stages, and read-only failure diagnostics including analytic candidate margins.

This is intentionally a full run; there is no pilot/smoke stage.
