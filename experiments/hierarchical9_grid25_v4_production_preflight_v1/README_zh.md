# Grid25 V4 正式生产：只读容量、成本、分段计划预检查 v1

**本入口没有 Slurm 提交命令，没有 V3/V4 生产 worker，没有模型训练。** 目的是先得到可信的精确任务数量、可审阅的 frozen `x_index` 划分，以及基于已完成 32+32 Pilot 的经验计算量预测，然后再单独设计分段生产作业。

已验证来源：
- Grid25 原始 Q0 mmap bank 已完成，173225×500×7×8；
- train=109490，val=16711，fresh_guard=8669，val_guard=38355；
- 旧 Grid25 2+2 Pilot 和 32+32 Extended Pilot 已通过原 V3/V4 标注及 hash 验证；
- Extended Pilot 的 V3: 1925/1952 `value_valid`，V4: 15343/15616 `refined`，有效 V4 tubes 87838；
- 独立 FD：固定32条、31 PASS、1 HOLD；**只有抽中且 FD-PASS 的局部梯度获得数值证据，不等于整个生产数据集的 gradient mask**。

## 用法

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
git fetch origin mainline-b/h9-scratch50k-r012-v1
git merge --ff-only origin/mainline-b/h9-scratch50k-r012-v1

export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
export GRID25_ROOT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"

PROD_SCRIPT=experiments/hierarchical9_grid25_v4_production_preflight_v1/run.sh

# 1. CPU-only synthetic tests
bash "$PROD_SCRIPT" test

# 2. All eligible train and val x. Reads small valid_fov mmap slices;
#    does not load 19GB q.npy nor the 36GB merged NPZ.
bash "$PROD_SCRIPT" prepare --root "$GRID25_ROOT"

# 3. Verify frozen plan files and report
bash "$PROD_SCRIPT" verify --root "$GRID25_ROOT"

# 4. Inspect cost and full production counts
cat "$GRID25_ROOT/derived/grid25_v4_production_preflight_v1/report.json"
```

服务器端新脚本/测试尚未运行，若出现 `[STOP]` 请不要修改源数据或手动提交全量任务，先检查日志。

## 它精确统计什么

针对 train + val 的所有 `126201` 个位置，逐块读取只读 memmap `valid_fov` 的256行一次小切片，计算各个 (x,s) 的 Q0 数量和支持 mask。不把缺失 Q0 的传感器定义为“不可见”。

依据原始 V4 科学协议：
- 每个保留 x：128 Uniform q → 8 sensor FOV sign（同原版）
- 对每个受支持 (x,s)：4 selected V3 query × 6-face original-seed SLSQP（同原版）
- 对每个受支持 (x,s)：最多32个 boundary anchors → 原版严格 refine 和最多7个 0、±.005、±.01、±.02 rad tube 候选（同原版）
- 只建立 total work 上限和规划索引，**不创建训练标签、FD mask，不运行 V3/V4 求解器**

默认 128 x 为一原子 shard，每个手动 Slurm 批次至多8 shards，即最多1024个 x。所有 shards/segments 都采用新的 Grid25 `x_index`，每个 x 只属于一个 shard 和一个 segment。产生的 segments 只是未来生产作业的候选划分；这次没有实际运行 worker。

## 成本估算方法（不要误当成作业时长承诺）

对已有 Extended Pilot 的 `global_pool_summary.json`、`v3_summary.json`、`v4_summary.json`：
- 用实际 `shard_worker_seconds / 任务数` 推得经验 CPU 秒/任务；
- 假设单节点4个 CPU workers（已成功调度），把三阶段 worker 时间相加再除以4，给出**理想基线**；
- 对批次预测额外乘2倍非乐观浮动系数，另加10分钟固定调度/启动开销；最坏批次必须在20小时预算内，才会允许写出 preflight；
- `projected` 不能反映跨位置难度差异、I/O负载、CPU并行利用率、调度等候或实际实验故障；不得把经验均值当作真实 SLA；
- **不计划全量 FD**：每个候选都重新求解多达24次的成本远高于基本 V3。正式FD认证的策略需要独立设计（固定抽样审计，或在预算允许时用更大范围验证），不能把 `gradient_candidate_valid` 直接当成 `FD-PASS`。

如果保守估计显示某段超过20h，程序会主动 STOP，需在新的 `--out` 中用更小的 `--max-shards-per-job` 重建新计划（保留原输出不可覆盖）；不是自动增加节点或改变原 solver。

## 输出

```text
$GRID25_ROOT/derived/grid25_v4_production_preflight_v1/
  x_plan.npz      # 全量候选 x_index、split、valid_count、support
  shards.json     # 每片精确 x / V3 / V4 工作量
  segments.json   # 连续、互不重叠的手动批次区间，含估算耗时
  report.json     # source hashes、训练/验证分布、总成本与限制
```

这个 output 不会覆盖已有 Pilot、Grid25 formal/ 或 bank_cache/。同一路径已存在时直接 STOP，避免不小心更换统计口径。要调整预算假设时使用新的 `--out`。

下一阶段才会根据真实预算决定：实现可恢复的手动一段一段的 CPU Slurm 作业、写 stage hash/CRC、检查梯度 FD mask 的生产策略、训练输入字段与性能验证。**本 preflight 完成后 `training_ready=false` 仍然正确。**
