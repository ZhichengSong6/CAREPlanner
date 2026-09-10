# Validation scope (not model-quality results)

Local harness: Python 3.13.5, PyTorch 2.10.0+cpu, no CUDA, no complete robot repository,
no 7.4GB data and no actual V1 checkpoint. Existing V1 model/objective/train source
from the previously generated package was available for mathematical tests.

`python test_pilot.py`: **19 tests: 18 PASS, 1 SKIPPED**.
Skipped: real repository geometry/dependency import. Server `--require-repo` turns
this into a failure instead of skip. Preparation also runs prior audit tests and
actual oracle/label/chain-mask preflight before creating training anchors.

PASS covers head offset (+1, no union-zero label), full raw-q derivatives,
parameter gradients, absence of P0 correction gradients, inactive components,
microbatch equivalence, complete default H9 backward, warmup, empty-stratum
rejection, canonical split leakage guard, cache SHA tamper guard, unit-normal
checks, deterministic independent boundary sampler and rank partitioning,
P0/P1 stream-identity gate, local zero-shift metric, synthetic checkpoint loading,
and prepare routing with **explicit synthetic planar geometry fixtures**.

`torchrun --standalone --nproc_per_node=4 test_pilot.py --ddp --backend gloo`:
**PASS** for P0 alpha=0 and P1 alpha=.25. Uses the actual new `perform_update`,
the existing global objective, unequal microbatch/group coverage, one final DDP
synchronization, and compares gradients AND Adam parameter updates with the
single-process whole-batch reference. Maximum absolute gradient error:
**1.430511474609375e-6**. FP32 numerical test; not CUDA AMP certification.

Python compilation / all CLI --help / bash syntax: PASS.
`python test_submit.py`: PASS, mocked sbatch in a temporary Git
checkout: absolute logs, 1/4/1-GPU resource requests, max-one concurrent training
array task, dependencies and jobs.env. No actual job was submitted in that test.

NOT RUN here: real cache generation, real trained checkpoint inference, real
FOV parity, CUDA AMP, NCCL, 400k-pair GPU update, Slurm jobs, validation quality,
any LOS/GCDF/VBC/trajectory/Gazebo/tracker/actual-seen or runtime switching.
The user must run the complete server smoke before the 2000-update pilot.
