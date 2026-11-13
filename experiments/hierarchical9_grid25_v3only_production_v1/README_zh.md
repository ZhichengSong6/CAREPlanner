# Grid25 V3-only production v1 — 仅 Segment 2–7

这条线的目标是尽快增加真实连续边界 V3 训练标签，不再生成目前未证明收益的 V4 Tube。**不会修改**已成功的原 V3+V4 Segment 0/1、原生产 runner、源 Q0 Bank、Train/Val 划分或已训练 checkpoint。

- 仍使用上游原版 `global_pool.worker/merge` 与 `v3_labels.worker/merge`、原版局部 SLSQP、6-face 求解与 reference gate；
- 生产预算与采样继续保持 `candidate_q_per_x=128, v3_q_per_x=4, x_shard_size=128`，按现有冻结 `segments.json` 取位置；
- 每段约1024个新的 Grid25位置，8个原子分片；不产生任何 `v4_anchors`、`v4_tubes`；
- CPU-only，单节点4CPU，Slurm GPU 分区但 **不申请GPU也不显式申请内存**；一次只手动提交一段，已有成功分片通过原校验后保留，失败不隐式覆盖；
- 准备第3段前要求第2段已完成且全部分片经过验证。准备第2段前先校验原生产 Segment1 完整性；
- 独立输出目录 `$GRID25_ROOT/derived/grid25_v3only_production_v1/segment_002` 至 `segment_007`，与原 Segment 0/1 分开；
- V3 best-found distance 通过 `value_valid` mask 训练，**不等于全局最短已认证 SDF**；未经独立 FD 的梯度不能视为已验证监督。

## 第2段（下一次仅执行这一段）

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
git fetch origin mainline-b/h9-scratch50k-r012-v1
git merge --ff-only origin/mainline-b/h9-scratch50k-r012-v1

export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
export GRID25_ROOT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"
V3RUN=experiments/hierarchical9_grid25_v3only_production_v1

bash "$V3RUN/run.sh" prepare --root "$GRID25_ROOT" --segment 2
bash "$V3RUN/submit.sh" 2
```

**提交成功会打印真实 JobID**，不使用 `--test-only` 显示的编号；不要在同一用户队列已有作业时重新提交。预计单个段主要耗时来自约2万多次 V3 约束求解，不能保证精确实际时间。

## 查看一个分段

```bash
SEG=2
JOB=实际JobID
bash "$V3RUN/run.sh" status --root "$GRID25_ROOT" --segment "$SEG"
sacct -X -j "$JOB" --format=JobID,State,ExitCode,Elapsed,NodeList

OUT="$GRID25_ROOT/derived/grid25_v3only_production_v1/segment_002"
tail -n 50 "$OUT/logs/slurm_${JOB}.out"
```

当 `sacct` 为 `COMPLETED 0:0`：

```bash
bash "$V3RUN/run.sh" verify --root "$GRID25_ROOT" --segment 2
cat "$GRID25_ROOT/derived/grid25_v3only_production_v1/segment_002/v3_summary.json"
```

若通过再手动选择下一段：

```bash
bash "$V3RUN/run.sh" prepare --root "$GRID25_ROOT" --segment 3
bash "$V3RUN/submit.sh" 3
```

以此顺序完成到 7。不要对现有 `segment_002` 再次 `prepare`，不要删除标注分片。中断或超时，必须先确认旧Job结束，检查日志和分片，再决定是否手动对原 segment 提交一次恢复作业。

**完成Segment 7后**，使用 `experiments/hierarchical9_grid25_v3_8segment_train_v1` 合并前8段并运行单GPU H9训练；不用再次生成任何 V4。
