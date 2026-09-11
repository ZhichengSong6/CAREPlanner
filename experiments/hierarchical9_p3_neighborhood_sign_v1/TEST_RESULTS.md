# P3 validation scope (not model quality)

Local harness: Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5; CUDA unavailable.
No trained V1/P0/P1/P2 checkpoint, full robot dependency tree, or 7.4GB dataset/cache here.

Reconstructed exact original common.py/calibration.py and P2 protocol.py from connected
GitHub reads. Git blob SHA matched ce3e95e...,33fb1ad...,5991020... respectively.
Scratch model.py/objective.py/train.py SHA256 matched the user's uploaded P2 manifest.
These original sources are not modified/committed in P3.

`python test_p3.py`: 11 PASS.
Covers independent radius RNG and rank partitioning, refreshed probes, actual-g labels
including sign reversal on the opposite FOV plane, no clipping/resampling, ambiguous
and out-of-limit exclusion, all-invalid microbatch graph connectivity, selected sensor
(+1, not union), hinge saturation and gradient direction, global denominator/microbatch
parity, full 1,133,705-parameter H9 backward and raw-q derivative, mixed-x pairwise FK
plane formula against independently computed synthetic values, lineage/settings gates,
and proposal/label/exclusion hashing.

`python test_workflow.py`: 3 PASS.
Explicit synthetic evaluator routing fixture; V1/P0/P1/P2/P3 names/output/paired tables.
Mocked sbatch in a temporary Git checkout: ONE submission/call, 4-GPU request, absolute
paths, no array/dependency/mem, quota rejection releases lock, duplicate and missing
smoke gates. Stdlib ZIP: reports only, integrity check, no checkpoint/overwrite, missing
file rejection. No Slurm job is actually submitted by these tests.

`torchrun --standalone --nproc_per_node=4 test_p3.py --ddp --backend gloo`: PASS.
Actual new P3 update driver, old global/boundary loss functions, side weight=0 and 5,
alpha=.002 and 1.0, uneven micros and a rank with NO valid side labels. Compared
combined gradients and Adam parameter updates with a single full-batch FP32 reference.
Max absolute gradient difference: 1.9073486328125e-6.
Max absolute Adam parameter difference: 2.3283064365386963e-10.
This is numerical testing, not an AMP/NCCL or real-data certification.

Compilation, Python CLI help, shell syntax: PASS.
NOT RUN: real 2000-update training/stream hash check, actual robot pairwise-oracle
preflight, real-data CLI end-to-end, CUDA AMP/NCCL, real Slurm, peak3090memory or model
quality. Server smoke forces actual model/cache/split/source/oracle checks and runs
2 full-sized P3 updates plus V1/P3_smoke geometry evaluation before pilot is allowed.
No runtime/URDF/safety modification or robot execution.
