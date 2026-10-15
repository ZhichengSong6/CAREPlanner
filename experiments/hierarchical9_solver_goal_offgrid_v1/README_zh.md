# R1 vs Solver-Goal：512 个新空间点正式评估 v1

这是已有权重的独立测试，不训练、不改 solver、不覆盖旧 1963-case 报告。

固定两个 50k checkpoint：

- R1 SHA256：`4f395926fa79c29474be8748cef4733ec400d155cd8fadb76c632c2838864002`
- Solver-Goal SHA256：`5507b1d16c5d34f4a3e8c01526c7bf3f278a51c07df9f4d878aaee476cee48f5`

## 一次正式作业

在原分支、仓库根目录执行：

```bash
export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
env -u NODE bash experiments/hierarchical9_solver_goal_offgrid_v1/run.sh submit
```

CPU 自动单测和真实源数据校验后，提交单节点四张 3090、16 CPU 的一个正式评估任务；默认排除 3090node1。`TIME_LIMIT` 只控制 Slurm 墙钟限制，不改变每条几何检索或 solver 预算。没有额外 GPU pilot。

任务内部顺序是：**独立几何构造 → 所有测试起点和覆盖报告冻结 → 才加载两网络 → 相同完整 solver 配对运行 → 空间点分组统计**。

前置检查读取已存在的训练完成记录和哈希；不会运行 torch.load 或利用网络预测选择测试样本。旧 V4 边界配置只作新的解析几何优化初值。每一个新 x 的可见见证及 Local 的正负两侧都重新用原始 FP32 几何校验；不借用旧 x 的标签。

## 测试集（完整固定协议在 protocol.json）

在实际原始 NPZ 保存的工作空间范围内，连续均匀抽取 512 个不同 x。以 float32 实际输入坐标检查：距所有旧 x 至少 2 mm，新 x 两两至少 2 mm。拒绝仅依据坐标重合条件，不依据任何网络或可见性结果；点集合先冻结，不能用另一个点替换没有见证的点。

每个 `(x,sensor)` 独立执行有限预算的几何见证搜索：先试邻近旧 x 的边界初值、64 个独立随机配置和中心配置，必要时最多 4 次 SLSQP 最大化最小 FOV 平面裕量，每次最多 80 次迭代。只把原始 FOV 检查通过且 g≥0.0001m 的配置称为可见见证。

有见证的组合：
- 最多 4 个 Uniform 外侧起点（独立随机流、最多256次尝试）。
- 对见证与一个随机外侧配置的线段，二分定位真实 FOV 符号跨越；用该新 x 的解析法向构造 0.01/0.02/0.05 rad 的局部外侧起点和内侧见证，各自重新检查关节限位和 FOV 符号。失效的半径不补采。
- 最大规模 512×8×(4+3)=28,672 条，实际数量由模型无关的几何检查决定。

没有见证的组合全部保存为 `UNRESOLVED_WITNESS`。这不是物理不可行证明，也不是网络失败。主成功率条件于规定几何搜索预算找到见证的组合；报告完整覆盖和筛除数量。此条件采样定义在看新模型结果之前固定。

## 评估不变

直接调用原 `runtime_probe.make_probe` 和 `run_probe`：10次投影、damping=.5、epsilon_f=.03、max_step=.25、12次根细化、tol=.002、1次上升、回退上升8次。模型不更新，参数在前后校验。每个 rank 模型运行顺序交错，各模型进行两次不计入结果的启动预热。

新数据上的 R1 成功率重新计算；**不会要求继续复现旧集合的1291/1963**。本程序不读取旧 regression starts 来选新样本，也不选择中间 checkpoint、扫描阈值或拼接sensor分支。

## 查看和恢复

```bash
bash experiments/hierarchical9_solver_goal_offgrid_v1/run.sh status
bash experiments/hierarchical9_solver_goal_offgrid_v1/run.sh log
bash experiments/hierarchical9_solver_goal_offgrid_v1/run.sh ranks
```

几何阶段看 `[geometry rank...]`，模型阶段看 `[eval rank...]`。输入冻结时会打印 `[FROZEN]`、实际起点数与哈希。

中断或失败且原作业已终止，修复环境问题后可执行：

```bash
bash experiments/hierarchical9_solver_goal_offgrid_v1/run.sh resume
```

这里只恢复评估：复用相同 points/job、已校验完整的几何/评估分片。不会重新抽取512个点；损坏分片和源代码变化明确拒绝，不自动删除。未完成的隐藏临时分片不作为完成证据。

完成后：

```bash
bash experiments/hierarchical9_solver_goal_offgrid_v1/run.sh summary
bash experiments/hierarchical9_solver_goal_offgrid_v1/run.sh verify
bash experiments/hierarchical9_solver_goal_offgrid_v1/run.sh pack
```

默认独立输出根：`.../outputs/mainline_b/h9_solver_goal_offgrid_v1/`；实际结果 `formal/`，包含 points.json、coverage.json、starts.jsonl、inputs_frozen.json、逐卡 solves.jsonl、report.json、summary.md、complete.json。原权重、训练集和已有报告只读。

pack 包含测试输入、几何筛选原因、逐案例结果、分组报告、日志与本次源码快照，不包含 .pt/.npz/.npy。包一次写入并完整读回校验，不向压缩 tar 追加文件。

## 统计和适用范围

报告 Local、Uniform、总体、各sensor的成功率、paired gained/lost/net、0/5mm/1cm/2cm/3cm裕量统计及值/梯度调用量、计时。95%区间采用固定种子的2000次配对空间点cluster bootstrap；每次重抽 x，并带上同一 x 全部sensor和起点，保留没有案例的预选 x。它不是多训练随机种子的稳定性证明，也没有做逐sensor多重检验校正。

这是**原工作空间内的非网格插值测试**，不称作未见整块空间区域、新机器人或外推。输入选点独立于网络；但见证和局部边界仍是有限搜索得到的，不代表全局完整覆盖。FOV通过不等于无遮挡、无碰撞、轨迹安全或真实被观测。

本地测试范围记录在 TEST_RESULTS.md；真实生产数据、URDF解析依赖、CUDA/NCCL和全部正式solver案例仍需在用户服务器运行。
