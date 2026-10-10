# BF scratch v1 vs original R1: full frozen FOV solver comparison

Full **1963-case** paired evaluation on exactly the previously frozen 2026-09 fresh starts (SHA256 `0e4711fc21628fdbb341191ec634415781696800a2386a4ad48e592c34f3198c`). No pilot or smoke.

- Primary: original R1, BF `best_val.pt` and BF `final.pt`, all with **identical untouched** legacy `runtime_probe` parameters (`projection_epsilon_f=0.03`).
- Secondary prespecified scale-sensitivity: only BF best/final with `projection_epsilon_f=0.002`. These rows are **not equal-solver comparisons** and cannot prove model-only improvement.
- R1 must reproduce baseline 618/683 local, 673/1280 uniform, 1291/1963 all, or merge refuses COMPLETE.
- Checkpoint validation requires both BF hashes and independent checksum guards; best step is taken only from the completed training run, never from solver success.
- Report all five arms on all cases, paired successes/losses, exact paired sign p, per-sensor counts, root failure stages and analytic candidate-margin bins. No LOS, GCDF, VBC, execution or actual seen.
- No checkpoint promotion or runtime modification.

One 4×3090 GPU job, excluding 3090node1 by default:

```bash
bash experiments/hierarchical9_boundary_first_solver_compare_v1/submit.sh
```

Read the state created by submit (default):

```bash
bash experiments/hierarchical9_boundary_first_solver_compare_v1/watch.sh \
 /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/bf_r1_fov_compare_v1/latest_compare.env status
```

Replace `status` with `summary`, `diagnosis`, `log` or `ranks`. Reused starts are a **regression set**, not untouched confirmation of new generalization.
