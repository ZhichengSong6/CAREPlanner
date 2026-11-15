# Grid25 V3-only 全量生产：Segment 8–123（116段，一次启动）

此模块只做 Grid25 **训练+验证可用的 126201 个 eligible x** 的剩余 V3 连续 FOV 边界距离标签。已完成的 Segment0–1 原版 V3/V4、Segment2–7 V3-only 和全部 Q0、训练模型不会被覆盖。preflight 124段仍使用原冻结的索引和 split；不会添加 V4 Tube，也不会把未通过 FD 的梯度当作真值。

## 安全机制

- 保持原版四核并行 \`global_pool\` + \`v3_labels\` 算法、选q策略和每段 8 个原子 shard（最后一段按实际大小），**不修改原科学代码**
- **严格顺序**：先检查前段成功，再 prepare 当前段 → work → 逐shard和summary校验 → 写 \`segment_complete.json\`，进而进入下一段
- 自动断点续跑：读取各段持久化的完成标记；处理中断后复用完成的shard，绝不删除、覆盖任何现有 Segment 数据
- 一次Slurm allocation最多处理 **4段**，24h时限，CPU-only 4核，GPU分区但零GPU申请；Slurm下一批使用 \`afterok\` 依赖提交（一次只提交一个新作业）
- 若站点限制连一个后继PENDING作业都无法提交，尝试 \`scontrol requeue\` 当前作业（**是否允许取决于集群管理员策略**）
- 若两种续接机制均不可用，**安全停止**；全部已有 Segment 保留。只需查错后重新执行 \`submit.sh\`，脚本会从第一个未完成段继续，而非重跑116段
- 断电或节点超时若Slurm不自动重排队，也需要人工再次执行 \`submit.sh\` 续跑；无法保证所有集群都允许无人值守的无限续接
- 终点全段强校验产生 \`batch_complete.json\`；此后才允许全量训练数据合并，不会在CPU标注作业中盲目提交GPU训练

## 首次启动（**只执行一次**，不能与本人其他Slurm job重叠）

\`\`\`bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
git fetch origin mainline-b/h9-scratch50k-r012-v1
git merge --ff-only origin/mainline-b/h9-scratch50k-r012-v1

export GRID25_ROOT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"
export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
BULK=experiments/hierarchical9_grid25_v3only_fullscale_v1

bash "$BULK/run.sh" preflight --root "$GRID25_ROOT"
bash "$BULK/submit.sh"
\`\`\`

第二条命令会先 \`sbatch --test-only\` 再只提交一个真实作业，打印 \`job=...\`。后续每批自动排队，不需要手动执行116次 \`prepare/submit\`。

## 查看实时进度

\`\`\`bash
BULK=experiments/hierarchical9_grid25_v3only_fullscale_v1
bash "$BULK/run.sh" status --root "$GRID25_ROOT"
squeue -u "$USER"
cat "$GRID25_ROOT/derived/grid25_v3only_fullscale_v1/batch_progress.json"
ls "$GRID25_ROOT/derived/grid25_v3only_fullscale_v1/logs"
\`\`\`

例如在最新实际JobID的日志尾部查看：
\`\`\`bash
tail -n 60 "$GRID25_ROOT/derived/grid25_v3only_fullscale_v1/logs/slurm_实际JobID.out"
\`\`\`

\`[FULLSCALE NEXT JOB SUBMITTED]\` 表示已自动安排下一个串行作业；\`[FULLSCALE REQUEUED]\` 表示同JobID即将重排队；\`[FULLSCALE ALL 116 V3 SEGMENTS VERIFIED COMPLETE]\` 才表示全部标注真正完成。

## 失败恢复

确认 \`squeue -u "$USER"\` 没有自己正在运行/排队的旧任务后，检查尾部日志以及 \`batch_progress.json\`；如果确认没有活跃任务且未完成，**再次运行 \`bash "$BULK/submit.sh"\`**，自动跳过所有SHA核验成功的 Segment。不要清空任何段目录，也不要直接一次提交116个手动作业。

## 合并完整V3并训练H9

只在 \`batch_complete.json\` 存在且 \`status=COMPLETE\` 时：

\`\`\`bash
TRAIN=experiments/hierarchical9_grid25_v3_full_training_v1

bash "$TRAIN/run.sh" prepare --root "$GRID25_ROOT"
bash "$TRAIN/submit.sh"
\`\`\`

训练为完整eligible Grid25的 V3-only H9 R1 **20,000 steps、batch1024、Adam cosine LR、1张3090**，每500步记录验证RMSE、MAE、符号准确率和S0–S7 per-sensor指标，保存 \`best.pt\` 和 \`final.pt\`。未启用未经FD核验的梯度、Eikonal或V4辅助损失，union仍是8传感器全有效时的 \`max\` 代理监督。

训练完成查询：

\`\`\`bash
bash "$TRAIN/run.sh" status --root "$GRID25_ROOT"
\`\`\`

**注意**：这里称“全量”，指冻结计划内 126201 个 eligible Train+Val x，**不是**把原始173225个Grid25 x的guard区域也参与训练；更不是已经获得梯度FD认证的原版完整H9训练。
