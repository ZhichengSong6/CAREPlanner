# Grid25 八段 V3 距离监督 H9 R1 训练

**在 Segment 0–7 全部验证完成后运行。** 段0–1来自已有 V3/V4 生产目录；段2–7来自新 V3-only 生产目录，二者均保留相同原始 V3 筛选逻辑与固定 Grid25 Train/Val 75mm 空间隔离。

- 数据集：共8192个不同 Grid25空间位置，每位置4个 selected q，最多32768个 `(x,q)` 行，每个row的8个传感器仅采用 `value_valid=true` 的 V3 signed-distance监督。
- 模型：冻结原有的 H9 R1 `PrivateTailCDF` 九分支结构；随机初始化，单卡 `3090node3` RTX3090，Adam lr1e-3，batch512，5000 updates，每250步记录验证误差。
- 有效监督：8 sensor 的 masked value MSE；union head 仍使用8个传感器都有效时的 `max_s d_s` **代理值**，损失权重0.2（不声称精确 union SDF）。
- **不启用**没有被独立FD证实的 gradient candidate，不使用 V4 Tube，不启动原版H9的 Eikonal/tension/gradient loss，因此是“扩大数据的 R1 V3-only实验”，不是全量Grid25完整版正式基准。
- 源 `Segment 0/1`、`Segment 2..7` 都只读。把原8段已标注数据合成独立 `labels.npz` 后再训练，不会写回任何源分片。

## 训练命令（须先完成前8段）

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
export GRID25_ROOT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"
TRAIN=experiments/hierarchical9_grid25_v3_8segment_train_v1

bash "$TRAIN/run.sh" prepare --root "$GRID25_ROOT"
bash "$TRAIN/submit.sh"
```

提交器会申请 `3090node3` 上 **1张 RTX3090、4 CPU、4小时限制**，先执行 `sbatch --test-only`，再手动提交一个实际Job。未收到真实JobID前不代表已经训练。

准备成功后 `dataset_000_007` 不能重复prepare（避免数据口径变化）。新训练不会覆盖原 Segment0-only 和 Segment0+1 模型 checkpoint。

## 监控

```bash
JOB=实际JobID
ROOT8="$GRID25_ROOT/derived/grid25_v3_8segment_train_v1"
sacct -X -j "$JOB" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 40 "$ROOT8/logs/slurm_${JOB}.out"
tail -n 5 "$ROOT8/run_r1_v3/metrics.jsonl"
```

如果成功，输出：
- `run_r1_v3/best.pt`：最佳验证传感器RMSE的原架构模型
- `run_r1_v3/final.pt`：第5000步的模型
- `run_r1_v3/metrics.jsonl`：每250步和最终Train/Val传感器RMSE、MAE、符号准确率与每sensor指标
- `run_r1_v3/run.json`：`COMPLETE`、实际updates、最佳RMSE、最佳步数和模型/数据SHA

训练完成后：

```bash
bash "$TRAIN/run.sh" status --root "$GRID25_ROOT"
```

这一轮验证的是更多空间覆盖后 V3 的数据泛化和 S1/S0 的误差是否改善。**不可直接拿只有前2段、验证集合不同的RMSE作为严格因果对比。** 后续应基于相同固定验证集重新评估旧checkpoint，或以相同的训练分层进行可比实验；是否继续全量Grid25生产再由8段结果决定。

如 `prepare` 或 `submit` 报错，保留现有源目录和日志，先定位问题，**不要删除原 Segment或重跑Q0采集**。
