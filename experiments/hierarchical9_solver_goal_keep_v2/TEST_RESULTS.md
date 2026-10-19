# Executed local CPU validation

Date: 2026-10-19. Environment: PyTorch 2.10.0+cpu, CUDA unavailable.

Final command:

    K2_TEST_REPO=<local reference checkout> python -m unittest discover -s experiments/hierarchical9_solver_goal_keep_v2 -p 'test_*.py' -v

Result: **17/17 passed, zero skips**, final recorded run 6.461 seconds.

The suite executes the exact original R1 model (Git blob 0b9dab1986114d4b95b33bb0fa01fd8e0c68e53b), real two-process Gloo gradient averaging, finite differences through predicted q updates, zero penalty at teacher weights, replay trajectory sampling/RNG isolation, fixed-attempt train/val generation with synthetic geometry, invalid-cache rejection, checkpoint lineage checks, Adam restoration, x-cluster statistics, Bash syntax and a real shell launch from a simulated Slurm spool directory.

These are component and interface tests. The full historical 400k Cartesian trainer, actual HPC datasets/URDF dependencies, full 20k loops, CUDA and NCCL have NOT been run here. The server performs real-input preflight before GPU submission; formal results are still required. Tests are not evidence that B improves the parent.
