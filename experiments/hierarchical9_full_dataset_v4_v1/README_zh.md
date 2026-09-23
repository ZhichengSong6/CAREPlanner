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
