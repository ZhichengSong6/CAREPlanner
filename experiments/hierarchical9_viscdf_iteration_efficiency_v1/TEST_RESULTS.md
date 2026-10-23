# Test record — 2026-10-23

Locally executed with Python 3.13.5 / PyTorch 2.10.0 CPU; no CUDA available.

    IE_TEST_REPO=<uploaded archive frozen dependencies> PYTHONDONTWRITEBYTECODE=1 \
      python -m unittest discover -s experiments/hierarchical9_viscdf_iteration_efficiency_v1 -p 'test_*.py' -v

18/18 tests passed, zero skipped, approximately 4.3 seconds on this environment.

The tests compile the five exact original runtime methods from the pinned
per_sensor_visibility_runtime.py Git blob c1de7ef95b4cdab8046882bfaf9ffdf62a0c1141,
without importing ROS/URDF runtime dependencies. They test hard one-query budgets,
original FULL parity, sufficient-budget parity, root refinement accounting,
projection/ascent/fallback, limits/masks, zero gradients, no oracle-based selection,
returned-candidate versus first-hit distinction, the actual R1 architecture with
frozen parameters, complete-case denominators, bootstrap repeatability, report
merge/reuse, corruption rejection, shell syntax and Slurm spool source resolution.

Separately validated all 22,518 actual frozen starts and their SHA256 from the
uploaded Keep-v2 review archive. No test input was changed or regenerated.

NOT executed here: real checkpoint loading (the archive has no .pt), actual URDF
geometry (URDF backing file absent), CUDA/NCCL, Slurm allocation, four-GPU timing,
or the formal R1/B iteration comparison. The server prepare entry performs real
checkpoint/URDF checks before GPU submission. The formal job then requires
per-case FULL reproduction of the previous completed evaluation.
