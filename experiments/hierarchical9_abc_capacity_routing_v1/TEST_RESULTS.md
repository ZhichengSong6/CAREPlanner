# ABC validation scope

This branch was assembled as an isolated Mainline-B experiment. The authoritative real-data validation is the server `smoke` stage described in README_zh.md.

## Static/code-level checks encoded in `test_abc.py`

The smoke worker runs these before touching the P0 checkpoint:

- A/B/C are exactly function-equivalent to the source model at initialization for values and raw-q gradients.
- A routes a selected sensor only into that sensor decoder; shared/union parameters remain frozen.
- B additionally routes into the selected sensor residual adapter; zero-initialized residual output starts as exact identity.
- C routes into the selected private 512→256 tail and decoder; early shared and union paths remain frozen.
- C private tails are copied from the original final shared tail but are distinct trainable parameters.
- routed per-sensor loss is additive across microbatches under fixed global denominators.
- trainable parameter counts satisfy A < B and A < C.

## Real smoke gates

The real two-update smoke additionally enforces:

- completed P0/cache lineage checks;
- identical A/B/C source checkpoint;
- initialization equivalence on an actual held-out batch (`max_abs_value_error` and `max_abs_q_gradient_error` <= 2e-6);
- actual repository dataset/oracle target generation;
- one full 400000-pair optimizer batch per update per arm;
- finite higher-order q-gradient/tension backprop on CUDA AMP;
- identical A/B/C training-stream SHA256;
- matched runtime solver evaluation and unchanged P0 oracle/solver preflight.

## Not claimed before server smoke

No claim is made here that the code has already passed CUDA/NCCL/real-data execution, that 2000-update training will fit the 8-hour allocation, or that any A/B/C architecture improves model quality. The smoke stage is required before formal training.
