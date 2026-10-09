# Boundary-first scratch v1：正式全量训练

本目录只新增一个独立训练实验，不修改原 R1/V2、Global/V3/V4 数据生产、cache 或 runtime solver。
模型定义逐字节复用 R1 private-tail H9（Git blob `0b9dab1986114d4b95b33bb0fa01fd8e0c68e53b`，2,184,329 参数）。随机初始化，不加载 R1/V2 checkpoint；`resume` 只能恢复本实验自身中断的 checkpoint。

## 一次提交

```bash
bash experiments/hierarchical9_boundary_first_scratch_v1/run.sh train
```

这不是 smoke/pilot。提交入口自动编译、CPU 单元测试、核验已有 cache，再请求单节点 4×3090。默认排除之前启动异常的 3090node1；显式 `NODE=节点名` 可覆盖。资源申请默认 24 小时，不是耗时预测。未指定 `--mem`，沿用该集群已有调度方式。

CPU+I/O 预检会验证已有约 1.1 GB cache。作业启动后，在**新的 formal/indices** 中一次性构建抽样行号与分层计数，不重算任何 FOV、V3 优化或 V4 标签。不会覆盖旧 cache。全量验证拒绝 NaN 有效标签、损坏文件、train/val 空间重叠或缺失的局部分层。

代码以内容哈希快照到新输出目录，排队期间后续 git 更新不会改变已提交作业。统一提交锁阻止 train/resume 同时写同一输出；不自动取消作业或清理旧结果。

## 冻结目标（protocol.json）

- Global：sensor 正负类别平衡的 sign loss；union 只作辅助符号输出。
- V3：仅有效 per-sensor value；Huber，尺度 0.5 rad，权重 0.25。不使用候选梯度或 union=max(sensor) 距离目标。
- V4：按 8 sensor × 7 offset 分层，Huber 局部尺度 0.005 rad，权重 1。偏移标签只作为局部法向距离近似，非全局最近距离证明。
- Boundary：按 8 sensor 分层；同一批 q* 同时监督零值（尺度 0.005 rad，权重 1）和完整 7D 梯度向量（权重 0.5）。目标中不活跃关节分量为零；不对预测梯度先做 mask 来隐藏非物理依赖。
- 关闭 union 数值一致性、union 距离、独立 Eikonal loss、tension。仍报告梯度模长和 Eikonal 诊断。
- 原始关节弧度度量不变；FP32，TF32 关闭。50,000 updates。Adam，1,000-step warmup 到 3e-4，之后 cosine 到 1e-5；全局梯度范数裁剪 10。
- 四卡合计 batch：Global 8192、V3 512 个查询组、V4 8192、Boundary 2048。各桶重复/子集关系不被当作独立新增样本；累计读取行数单独记录。

这些超参数是预先冻结的新实验设定，不是已证明的最优配置，也不是与旧 R1 等量的数据/计算预算。

## 多卡与验证

每卡独立模型；先完整计算输入梯度监督并 backward，再一次性对参数梯度求跨卡平均，避免在输入高阶导路径使用 DDP reducer hooks。cell loss 分母使用全局计数，局部分子乘 world 后再平均梯度。附带两进程 Gloo 与单进程全批梯度一致性测试。

验证：V3 遍历所有验证行；Global 固定 16384 个随机行；V4 固定 14336 个分层行；Boundary 固定 4096 个分层行。后面三项不是遍历全验证集。按固定验证目标保留 `best_val.pt`，同时保留 `final.pt` 和每 5000 步快照。**不读取任何 solver starts、solver 成败或 fresh 测试结果选模型。**

## 查看与恢复（都不是新测试）

```bash
bash experiments/hierarchical9_boundary_first_scratch_v1/run.sh status
bash experiments/hierarchical9_boundary_first_scratch_v1/run.sh log
bash experiments/hierarchical9_boundary_first_scratch_v1/run.sh follow
bash experiments/hierarchical9_boundary_first_scratch_v1/run.sh summary
bash experiments/hierarchical9_boundary_first_scratch_v1/run.sh verify
```

只有已确认旧作业结束且本实验 `latest.pt` 存在、`final.pt` 不存在时才允许：

```bash
bash experiments/hierarchical9_boundary_first_scratch_v1/run.sh resume
```

默认输出：`outputs/mainline_b/h9_boundary_first_scratch_v1/formal/`。
查看边界零值 MAE、局部 RMSE、与零预测器的 MSE 比值、逐 sensor/offset 指标、法向余弦和模长，不使用新旧加权 loss 的数值直接判断模型优劣。

`COMPLETE` 仅表示本训练完成。新 checkpoint 格式为 `care_h9_boundary_first_scratch_v1`；union 输出不保证是距离场。未自动接入旧 runtime，不宣称 solver/FOV/LOS/collision/trajectory 提升。后续模型对比须使用同一固定 solver，当前反复使用的 starts 只算回归集。

## 已执行检查

本地 Python 3.13 / PyTorch 2.10 CPU：11 项回归测试通过，包括真实 R1 模型反向传播、无效标签屏蔽、损坏拒绝、分层抽样、cache 不修改、Adam 恢复和两进程梯度一致性。Bash 语法检查通过；模拟 Slurm 提交验证单节点四卡、源码快照、重复提交拦截与只读查看命令。
未在用户集群、真实完整 cache 或 CUDA/NCCL 上执行训练。服务器 GPU 运行是否成功以实际日志为准。

可选开发者分布式单测（不是用户正式流程的前置实验）：

```bash
BF_DISTRIBUTED_TEST=1 python -m unittest discover -s experiments/hierarchical9_boundary_first_scratch_v1 -p test_training.py -v
```

官方接口参考：PyTorch DistributedDataParallel 的 autograd.grad 警告与 torch.distributed all_reduce 文档。此实验使用显式参数梯度归约，不把该警告当作旧模型失败的诊断证据。
