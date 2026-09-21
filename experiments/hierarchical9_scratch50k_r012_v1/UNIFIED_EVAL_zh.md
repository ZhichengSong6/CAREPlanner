# Unified development evaluation

比较固定为：

```text
V1 / P0 / E0 / E1 / R0 / R1 / R2
```

评估内容：
- union / sensor-max field MAE、sign、q-gradient cosine；
- ranking top1/top2/top3/fallback；
- boundary geometry；
- analytic-FOV two-sided sign profile：±0.005/0.01/0.02/0.05；
- fixed radial neighborhood；
- 与 E012 完全一致的 1388 matched local+uniform solves；
- failure stage / root source / paired-vs-V1；
- per-sensor solve；
- projection / ascent1 / ascent10。

这是 development FOV-only benchmark，不包含 LOS/self-occlusion、collision、trajectory、actual-seen 或最终 runtime promotion。

## 运行

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
git fetch origin mainline-b/h9-scratch50k-r012-v1
git merge --ff-only origin/mainline-b/h9-scratch50k-r012-v1

export R012_ROOT=/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_scratch50k_r012_v1
export E012_REFERENCE_ROOT=/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_cal_pilot_20260910_183735_099nu5

bash experiments/hierarchical9_scratch50k_r012_v1/submit_unified_eval.sh eval
```

跟日志：

```bash
source "$R012_ROOT/r012_jobs.env"
tail -n 160 -F "$R012_ROOT/logs/r012_unified_eval_${R012_UNIFIED_EVAL_JOB}.out"
```

完成后：

```bash
sacct -j "$R012_UNIFIED_EVAL_JOB" --format=JobID,State,ExitCode,Elapsed
cat "$R012_ROOT/evaluation_unified_dev/summary.md"
bash experiments/hierarchical9_scratch50k_r012_v1/submit_unified_eval.sh pack
```

ZIP：

```text
$R012_ROOT/r012_unified_dev_reports.zip
```
