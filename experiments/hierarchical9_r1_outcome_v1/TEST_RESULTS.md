# Executed validation

Python 3.13; PyTorch 2.10.0+cpu; CUDA unavailable.

14 tests executed, 14 passed, zero skipped.

Coverage: actual R1 architecture backward pass; parameter finite differences through candidate geometry; independent scalar/batched solver-value parity on linear/nonlinear fields; actual one-rank DDP combined optimizer update; explicit two-rank Gloo auxiliary-gradient reduction; warmup delegation; masks and joint limits; source/RNG/parameter isolation; trusted checkpoint round trip; synthetic paired-cache schema, hash rejection and original-spatial-split filtering.

Python compilation and bash -n passed. No real HPC dataset, URDF, checkpoint, CUDA or NCCL training was executed here. Before GPU submission, run.sh additionally tests real source data, original geometry and actual runtime candidate parity on the server. Those gates are not counted as local passes.
