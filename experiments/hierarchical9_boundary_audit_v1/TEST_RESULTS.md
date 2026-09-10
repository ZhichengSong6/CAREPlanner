# Local validation scope

Environment: Python 3.13.5, PyTorch 2.10.0+cpu, no CUDA.

`python test_audit.py`: 20 tests enumerated, 19 PASS, 1 SKIPPED.
Skipped: actual repository FK/URDF/runtime integration (the local isolated harness
contains the new diagnostic files, not the full repository/dependencies).
The server submit script passes `--require-repo`, so that integration test cannot
silently be skipped there; missing imports/dependencies are failures.

PASS scope:
- Discrete-bank signed-label jump on a known planar boundary, normal misalignment.
- Exact bank floor (1e-4 rad), degenerate normal, first-index ties, invalid bank rejection.
- Inactive joint distance invariance and raw/masked gradient reporting.
- Audit-only zero refinement, inactive joint preservation, degeneracy/out-of-bounds failure.
- Finite-difference normals, plane-tie/joint-limit/nonfinite rejection.
- Tangent off-bank construction and bank-distance gate.
- Field sensor indexing/input gradients; side profiles do not clamp/relabel.
- Missing data stays null/N/A; paired summary/report construction.
- q_zero presence is not a found-root assertion.
- Bounded analytic control and sampling-RNG independence.
- Complete CLI/report/JSONL pipeline with explicitly SYNTHETIC dependency fixtures.

Python compilation: PASS. Bash syntax: PASS. CLI --help: PASS.
These checks do not measure trained model quality or real-geometry success rates.

NOT RUN here: actual V1/old8 checkpoint deserialization, 7.4GB dataset loading,
real URDF FK/oracle parity, imported runtime solver parity, CUDA tests, server
Slurm submission, boundary/solve evaluation on real samples, any self-occlusion,
GCDF/VBC/path safety, Gazebo, tracker, or training.

The scripts perform mandatory real oracle/label/mask/solver preflight on the
server before audit samples are generated. Only a completed server run with
`manifest.status=COMPLETE` and `[done] mainline_b_boundary_audit_complete` is a
completed audit. `PREFLIGHT_ONLY`, `FAILED`, or partial reports are not one.
