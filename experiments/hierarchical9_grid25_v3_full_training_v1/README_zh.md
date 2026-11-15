# 正式全量 Grid25 V3 距离监督 H9 R1

当 `derived/grid25_v3only_fullscale_v1/batch_complete.json` 标记 **116个新Segment全部完成且强校验通过**，使用本模块。全量 eligible 空间位置为126201（其中109490 Train，16711 Val），每个位置4个选定配置，理论得到504804个 `(x,q)` 行；每行最多8个sensor距离标签，`value_valid`为唯一允许监督的有效性mask。

此模式是正式的**V3-only距离监督 H9 R1训练**，不是原版未经FD资格确认的梯度监督，也不等于全局精确SDF训练。V4 Tube已在公平对照中表现不稳，故不用V4 Loss；union仍是全8 sensor有效时的 `max_s V3_s` 代理。

## 全量数据合并（只读124个Segment）

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
export GRID25_ROOT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"
export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
TRAIN=experiments/hierarchical9_grid25_v3_full_training_v1

bash "$TRAIN/run.sh" prepare --root "$GRID25_ROOT"
bash "$TRAIN/run.sh" verify --root "$GRID25_ROOT"
```

输出 `$GRID25_ROOT/derived/grid25_v3_full_training_v1/dataset_segments_000_123/{labels.npz,manifest.json}`，包含每个 Segment 的manifest、stage summary、原子shard SHA和固定spatial train/val划分。该目录一旦创建就不会再覆盖。

## 单GPU正式训练

```bash
bash "$TRAIN/submit.sh"
```

- `3090node3` 上申请 **1×3090、4CPU**，4小时作业上限
- 原 R1 九分支 H9 网络随机初始化；**20,000步**（非8段实验中的5000步）、batch 1024、Adam lr 0.001至0.00001余弦退火
- 距离masked MSE加权0.2的union proxy MSE；没有FD梯度、Eikonal、tension、V4 Loss
- 每500步独立验证全部16711 Val x 的合格sensor标签
- 输出 `run_r1_v3_full/{best.pt,final.pt,run.json,metrics.jsonl}`，记录每传感器S0–S7的RMSE和符号准确率

训练结束后执行：

```bash
bash "$TRAIN/run.sh" status --root "$GRID25_ROOT"
```

如Slurm失败，保留原训练目录和训练日志。训练脚本为防止覆盖数据，不会在已有运行目录存在时隐式从头开始；若确实要恢复失败的训练，需单独评估如何加载checkpoint，不允许删除旧结果。

训练不是自动由标签作业提交的：**先确认116段批处理已完成，完成数据合并，再手动提交一个GPU job。** 这样不会触发账号最多一个作业的并行限制，也能阻止无效或不完整标签进入模型。
