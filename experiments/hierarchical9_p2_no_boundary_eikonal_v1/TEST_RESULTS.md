# P2 validation scope

Local harness: Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5. No CUDA, no actual
V1/P0/P1 checkpoints, no 7.4GB dataset/cache, no complete robot dependency tree.

The exact legacy common.py/calibration.py/train_pilot.py were fetched from commit
0c50a7998a32c9484f78681731da4239f5c10f6c and locally reconstructed byte-for-byte;
Git blob SHA checks matched ce3e95e..., 33fb1ad..., ee4d371... respectively.
Existing scratch model.py/objective.py/train.py hashes matched the prior package.
These old files are NOT changed or added in this commit.

`python test_p2.py`: 12 PASS. Tests cover removal of only the additional Eikonal
value and parameter-gradient term; unchanged norm logging; equal-stratum
microbatch normalization; sensor +1 output mapping; full H9 raw-q backward;
exact config/warmup; old-source, control-stream and P2 identity gates;
smoke-not-full-comparison rejection; shift coverage/undefined entries;
actual-g sign confusion; root-source failure tables; and mocked sbatch.
Submission mock verifies one job per call, 4-GPU request, absolute logs, no array,
no dependencies, no mem options, rejected-sbatch retry, duplicate-submit guard.

`torchrun --standalone --nproc_per_node=4 test_p2.py --ddp --backend gloo`:
PASS. Calls the original train_pilot.perform_update with boundary Eikonal=0.
Two synthetic updates at alpha=.002 and 1.0; compares gradients and Adam parameter
updates with full-batch FP32 reference, including uneven microbatches.
Maximum absolute gradient error: 4.76837158203125e-7.
Maximum absolute Adam parameter error: 0.0.
This is a numerical test, not a run on project training data or proof of AMP correctness.

Python compilation, CLI --help, shell syntax: PASS.
NOT RUN: real training CLI/end-to-end geometry evaluator, real cache validation,
CUDA AMP/NCCL, Slurm submission, full 2000-update sample-stream equality or model
quality. Server smoke includes actual 2-update P2 training and a V1/P2_smoke eval;
full sample-stream equality is enforced after the real 2000-update run.
No robot/runtime/URDF/planner/safety modification or execution.
