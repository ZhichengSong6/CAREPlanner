# Grid25 Segment 0 → 首次真正的 H9 GPU 训练（V3 距离监督）

**这是训练，不是继续 V4 标注或 pilot。** 使用刚完成的 Segment 0 中约 23541 条有效传感器 V3 连续边界有符号距离标签，保留原 Grid25 train/val x 隔离（845 train、179 val）。复用旧 R1 `PrivateTailCDF` 的完整九分支模型结构，随机初始化，单卡 RTX 3090 训练1000步，生成模型权重与 Train/Val metrics。

## 和原来 50k H9 训练的区别（重要）

- 本实验使用**V3 真实连续边界距离标签**，不是旧 Q0 最近点查表近似。
- 8 个传感器 head 用 `new_value` 且仅 `value_valid=true` 的标签做 masked MSE。
- 当且仅当八个传感器标签全部有效时，union head 使用 `max_s new_value_s` 作**代理目标**；它不是已经认证的全局精确 union SDF。默认代理 loss 系数0.2。
- **不**使用 `gradient_candidate` 作为已验证监督，不使用 V4 tube offsets/normals 代替全局距离或梯度。
- 原 H9 的梯度、Eikonal、tension loss 本实验**不启用**。这是一次新 V3 标签支持的训练验证，不能把这里的结果当作相同 Loss 的正式 R0/R1/R2 公平模型比较。
- 所有源分片、模型、标签快照保留原始 SHA 指纹；训练只新增自己的输出，不修改 Q0 Bank / 原 Segment / 原 runner。
- **不**要求剩余123个生产分段完成才能先得到新训练的 checkpoint。正式 full Grid25 模型性能评价仍需扩大训练空间覆盖。

## 立即执行

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
git fetch origin mainline-b/h9-scratch50k-r012-v1
git merge --ff-only origin/mainline-b/h9-scratch50k-r012-v1

export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
export GRID25_ROOT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"

# 1. 只读源 Segment 0，一次性把 V3 距离+mask 冻结成小训练快照。
bash experiments/hierarchical9_grid25_v3_first_train_v1/run.sh prepare --root "$GRID25_ROOT"

# 2. 提交真正的 GPU 训练：3090node1 1×3090 + 4 CPUs，1000 successful optimizer steps。
bash experiments/hierarchical9_grid25_v3_first_train_v1/submit.sh
```

如果 `prepare` 已成功建立数据集，请不要重复执行；直接提交。若此前有自己正在运行的 Slurm job，提交器会拒绝，不会自动排队占用唯一作业名额。脚本先做 Slurm `--test-only`；只有第二条实际 `[GRID25 H9 V3 TRAIN SUBMITTED]` JobID 才是需要监控的真实任务。

## 查看训练

```bash
JOBID=12345  # 替换为 submit 实际打印的 JobID
squeue -j "$JOBID"
sacct -X -j "$JOBID" --format=JobID,State,ExitCode,Elapsed,NodeList

ROOT="$GRID25_ROOT/derived/grid25_v3_first_train_v1"
tail -n 60 "$ROOT/logs/slurm_${JOBID}.out"
cat "$ROOT/run_000/run.json"
tail -n 5 "$ROOT/run_000/metrics.jsonl"
```

完成时应该看到 `[H9 FIRST GRID25 V3 TRAIN COMPLETE]`，且 `$ROOT/run_000/` 生成：

- `final.pt`（第1000步）
- `best.pt`（验证传感器 RMSE 最低的 checkpoint）
- `run.json`（status=COMPLETE、数据/代码SHA、最佳RMSE）
- `metrics.jsonl`（每100步的 train/val、各传感器误差、符号准确率）

单卡 GPU 时间未实测，申请4小时作为安全上限；不会承诺一定在某时刻结束。若内存不足或GPU作业失败，请保留输出和日志，先定位问题，不要删除或重提同一跑次。

## 下一阶段

得到这条 Loss 曲线和 checkpoint 后，再基于实际结果决定增加 V4 Tube 的**局部边界辅助 loss**（不能直接把 offset 当全局 SDF）、或者添加更多已认证 V3 距离样本。随后才进行更完整的新模型评估和公平 ablation。
