# Full-scale V4 dataset production v1

This job builds data only; it does not train a network.

Frozen design:
- preserve the original q0-bank train/val split and exclude every x used by the frozen fresh solver holdout;
- missing old-bank sensor support is UNRESOLVED, never proof of infeasibility;
- 128 uniform arbitrary q per retained x with analytic signs for all 8 sensors;
- select 4 q/x by greedy supported-sensor inside/outside state coverage and joint-space diversity;
- label those q with the original V3 6-face/original-seed SLSQP policy, changing only geometry/Jacobian to the validated analytic implementation;
- choose up to 32 active-joint farthest q0 anchors per supported (x,s), refine to |g|<=1e-6, and retain local tubes at 0, +/-0.005, +/-0.01, +/-0.02 rad.

Stages: prepare_full.sh -> submit_global.sh -> submit_v3.sh; submit_v4.sh is independent after prepare.
Every compute stage is sharded, checksum-verified, resumable, and never overwrites a completed shard.


## Two-node production allocation

Default production uses both `3090node1` and `3090node3`.

- global pool: 2 nodes x 4 GPUs, 4 local GPU workers/node, global world=8;
- V3: 2 nodes x 16 CPU workers, global world=32. One GPU/node is requested only to place the job on the GPU partition; the validated analytic V3 solver is CPU-side;
- V4: same 2-node/32-CPU-worker layout.

The two-node runners use `srun` with one launcher task per node. Global ranks are deterministic and shard ownership remains disjoint, so the same checksum/resume rules apply.
