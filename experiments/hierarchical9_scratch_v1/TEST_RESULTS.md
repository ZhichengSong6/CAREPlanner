# Local validation

Pre-publish check on 2026-09-09, Python 3.13.5, PyTorch 2.10.0+cpu.
No CUDA, NCCL, robot simulator or actual 7.4GB training dataset was available.

```bash
python -m py_compile experiments/hierarchical9_scratch_v1/*.py
bash -n experiments/hierarchical9_scratch_v1/submit_3090node3.sh
OMP_NUM_THREADS=1 python experiments/hierarchical9_scratch_v1/test_synthetic.py --device cpu --amp off
OMP_NUM_THREADS=1 python -m torch.distributed.run --standalone --nproc_per_node=4 \
  experiments/hierarchical9_scratch_v1/test_synthetic.py --ddp --device cpu
```

All commands passed. Synthetic checks cover:

- Full 1,133,705-parameter architecture and [union,S0,...,S7] layout.
- 7D input gradients, finite differences and state-dict serialization.
- Value/first/second-input-derivative loss backward; trunk and all nine heads receive gradients.
- Independent loss formula, unequal microbatch accumulation, unavailable sensor masks.
- FP16 output with FP32 400,000-count normalization regression.
- Four-rank CPU DDP gradients vs single-process full-batch reference, including absent local heads.

DDP result: `max_abs_error = 1.1920928955078125e-07`.
PyTorch emitted a gradient-stride/bucket-view performance warning; correctness assertions passed.
This is not a CUDA performance or real-data training qualification.

NOT tested here: CUDA AMP, NCCL, 3090 memory/throughput, real-data smoke, 50k training,
checkpoint recovery under Slurm interruption, neural field quality, or Case026 fallback.
The Slurm entrypoint runs CUDA synthetic tests before loading real data. Run MODE=smoke
on the server before a formal job; do not infer GPU success from CPU success.

Only experiment source, documentation and checksums are committed; no dataset,
trained checkpoint, private credentials or generated Python bytecode is included.
