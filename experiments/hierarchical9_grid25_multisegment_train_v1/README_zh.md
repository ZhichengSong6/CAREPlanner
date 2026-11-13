# H9 Grid25 Segment 0+1 V3-only vs V3+V4 — 两段成对训练

这是**实际 GPU 训练**，不是重新标注或新的 Pilot。已完成的两段真实数据不修改。

| 数据 | Segment 0 | Segment 1 | 合计 |
|---|---:|---:|---:|
| x | 1024 | 1024 | 2048 |
| train x | 845 | 856 | 1701 |
| val x | 179 | 168 | 347 |
| 有效 V3 传感器标签 | 23541 | 24914 | 48455 |
| V4 tubes | 943493 | 1014842 | 1958335 |

## 为什么先重训 V3-only？

因为原来单段的 `best_val_sensor_rmse=0.494168` 是 Segment 0 训练/验证划分计算的，不能直接拿来和新的 Segment 0+1 验证集结果比较。因此同样数据、同样新验证集分别训练两个模型，才能判断 V4 辅助损失有没有改善泛化。

两组模型统一使用原版 H9 R1 PrivateTailCDF、随机 seed=0、1000 updates、batch=512、Adam lr=1e-3、每100步验证、原V3 value masked MSE 和原 union 代理权重0.2；joint 模式增加 V4 局部零值约束（权重0.2）和局部偏移差分约束（归一化后权重0.05），V4 batch=512，单独的抽样 RNG 不消耗 V3 RNG。checkpoint 都按同一份 V3 Val sensor RMSE 选择。原来的梯度/Eikonal/tension loss 没启用，V4 局部 offset 不是完整 SDF，也未通过对每个样本独立 FD 认证。

## 直接操作

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
git fetch origin mainline-b/h9-scratch50k-r012-v1
git merge --ff-only origin/mainline-b/h9-scratch50k-r012-v1

export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
export GRID25_ROOT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"
MODULE=experiments/hierarchical9_grid25_multisegment_train_v1

# A. 只读合并两段现有的标签；不提交 Slurm，不求解新标签。
bash "$MODULE/run.sh" prepare --root "$GRID25_ROOT"

# B. 首先提交新数据上的 V3-only 模型，3090node3，单张 RTX 3090。
bash "$MODULE/submit.sh" v3
```

A 成功一次后不要重新 prepare：如果数据集路径存在，程序拒绝覆写。B 会先尝试 `sbatch --test-only`，确认自己队列为空才提交一份真实 GPU Job。Job ID 以 `[SEG01 H9 TRAIN SUBMITTED]` 实际返回的编号为准。

先查看 V3 作业是否真正完成（将 XXX 替换为实际 Job ID）：

```bash
JOB=XXX
BASE="$GRID25_ROOT/derived/grid25_multisegment_train_v1"
sacct -X -j "$JOB" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 40 "$BASE/logs/v3_${JOB}.out"
cat "$BASE/run_v3/run.json"
```

当 V3 作业 `COMPLETED 0:0` 并且 run.json.status=`COMPLETE`，才提交新数据上的联合训练：

```bash
bash "$MODULE/submit.sh" joint
```

确认 joint 作业完成后：

```bash
bash "$MODULE/run.sh" compare --root "$GRID25_ROOT"
```

结果会打印相同数据的 `v3_only_best_val_rmse`、`v3_v4_joint_best_val_rmse`、绝对差值和相对改善百分比；改善百分比为正数表示 joint 更好。

## 生产文件

```text
$GRID25_ROOT/derived/grid25_multisegment_train_v1/
  dataset_segments_000_001/
    v3.npz          # 8192个 (x,q)，valid mask，8 sensor continuous distance
    v4.npz          # 1958335 regular-tube local zero/delta samples
    manifest.json   # Segment 0+1 各原分片 SHA、源阶段摘要、split
  run_v3/
    best.pt / final.pt / metrics.jsonl / run.json
  run_joint/
    best.pt / final.pt / metrics.jsonl / run.json
  logs/
    v3_JOB.out / v3_JOB.err / joint_JOB.out / joint_JOB.err
```

**训练代码尚未经远程 GPU 执行。** 如果 prepare 报错，请保留之前的 Segment 0/1 和数据快照，不要删除重跑源标注。发报错即可定位。

本步骤的价值是检验 V4 在**相同的扩大训练/验证集**上是否稳定优于 V3-only。它仍是两段数据的小规模实验：不是 126201 x 全 Grid25 的最终正式模型，不能宣称全局最短 signed distance 或已认证梯度。
