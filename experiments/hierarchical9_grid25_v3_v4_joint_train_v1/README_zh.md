# Grid25 Segment 0: H9 R1 V3 + V4 Local Tube Joint Training v1

**这是模型训练而不是标签生产。** 使用已完成并验证的 Segment 0：V3 距离有效监督 23,541 条，V4 正则局部 Tube 943,493 条。训练 1,000 步（单卡 RTX 3090，`3090node3`），和已有 `grid25_v3_first_train_v1/run_000` 的 V3-only 1000步实验对比。

## 训练信号

- 同一个随机初始化的原版 H9 R1 `PrivateTailCDF`；同一源数据 V3-only 已冻结的 845 Train / 179 Val 空间位置，不能随意随机拆分。
- **V3 距离监督**：只使用 `value_valid` 的 sensor `new_value`，各 head masked MSE。八个传感器同时具备有效标签时，union head 可用 `max_s d_s` **代理目标**，默认权重0.2。不是精确 union SDF。
- **V4 局部边界零值约束**：对 regular anchor `q_star`，约束 `f_s(x,q_star)≈0`，默认权重0.2。
- **V4 局部偏移一致性**：同 sensor 的 `q_t=q_star+t*n`，约束 `f_s(x,q_t)-f_s(x,q_star)≈t`，非零 t ∈ {±.005,±.01,±.02} rad，分母用0.02 rad归一化，默认权重0.05。这不是“全局最近点距离标签”，也不是独立 FD-PASS 梯度；只用于原标签已通过 same-face/sign gate 的局部 Tube。
- 不使用 V3 的 `gradient_candidate`、H9 原 gradient/Eikonal/tension 作为被 FD 验证的真实梯度。本实验是 **第一轮 V3+V4 loss ablation**，并非与旧原版 H9 完全相同 loss 的公平 benchmark。
- V3 train 抽样沿用原训练器 GPU RNG；独立 V4 generator 防止扰动 V3 batch 随机序列；初始 seed、optimizer、1,000步、V3 batch和验证位置/指标保持一致。
- 最优模型仅根据固定 `val_sensor_rmse` 选择；另外固定8192条 Val Tube 记录 `val_tube_zero_rmse` 和 `val_tube_delta_rmse_rad`，不能把新的 loss 数值直接与 V3-only 总 loss 比较。
- `training_ready_full_grid25=false`：仅首次 1024 x Segment，正式 Grid25 完整训练还需更广泛的 V3/V4 数据和评价。

## 一次性命令

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
git fetch origin mainline-b/h9-scratch50k-r012-v1
git merge --ff-only origin/mainline-b/h9-scratch50k-r012-v1

export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
export GRID25_ROOT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"

JOINT="experiments/hierarchical9_grid25_v3_v4_joint_train_v1"

# 1. 冻结 V4 tube 配对训练样本（读 Segment 0，输出独立文件）
bash "$JOINT/run.sh" prepare --root "$GRID25_ROOT"

# 2. 单独一次 Slurm 正式 GPU 提交（先 --test-only，后提交真实作业）
bash "$JOINT/submit.sh"
```

`prepare` 已成功就不要重复执行，避免覆盖。Slurm 仅申请 `3090node3` 1×RTX 3090、4 CPU，4小时 limit；只提交一次且不会启动其他作业。若 `prepare` 或提交失败，保留日志，不要删除 source/Segment 0 或覆盖 checkpoint。

## 监控

```bash
JOB=12345  # 替换 submit 打印的实际 job 编号，不是 test-only job
squeue -j "$JOB"
sacct -X -j "$JOB" --format=JobID,State,ExitCode,Elapsed,NodeList

BASE="$GRID25_ROOT/derived/grid25_v3_v4_joint_train_v1"
tail -n 60 "$BASE/logs/slurm_${JOB}.out"
tail -n 5 "$BASE/run_000/metrics.jsonl"
```

训练完成后比较：

```bash
"$VIS_PYTHON" - <<'PY'
import json,os
from pathlib import Path
root=Path(os.environ["GRID25_ROOT"])/"derived"
b=json.loads((root/"grid25_v3_first_train_v1/run_000/run.json").read_text())
j=json.loads((root/"grid25_v3_v4_joint_train_v1/run_000/run.json").read_text())
print("V3 only:",b["best_val_sensor_rmse"])
print("V3+V4:  ",j["best_val_sensor_rmse"])
print("Improvement (positive=better):",j["improvement_over_v3_only_rmse"])
print("Joint training status:",j["status"])
PY
```

## 输出

`$GRID25_ROOT/derived/grid25_v3_v4_joint_train_v1/`:
- `tube_snapshot_000/{tubes.npz,manifest.json}` — 与 V3 数据集独立、有源 hash 的局部 tube 训练快照；
- `run_000/{best.pt,final.pt,run.json,metrics.jsonl}` — 原 H9 R1 的新 checkpoint 和 V3/Tube 验证指标；
- `logs/slurm_JOBID.out` / `logs/slurm_JOBID.err` — 作业运行记录。

**结论判断：** 重点看 `best_val_sensor_rmse` 是否优于 V3-only 的0.494168，以及各 sensor 的误差是否改善；单批次数据小、验证源窄，任何改善仍需更大 Grid25 覆盖复现才能成为正式效果结论。
