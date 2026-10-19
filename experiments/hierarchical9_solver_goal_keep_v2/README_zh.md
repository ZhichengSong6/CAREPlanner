# Solver-Goal v2：成功行为保持的双臂微调

只新增本目录；不修改原 R1、Solver-Goal v1、既有数据、运行时或安全链。
父模型固定为 Solver-Goal 50k，SHA256：
`5507b1d16c5d34f4a3e8c01526c7bf3f278a51c07df9f4d878aaee476cee48f5`。
这不是从头训练，也不是恢复父实验的旧 Adam：**A/B 都从相同父权重出发，新建 Adam**。

## 固定协议

- A_continue：保留原 R1 全空间距离/梯度/Eikonal/tension + Solver-Goal 几何目标。
- B_keep：与 A 相同，再增加权重 0.01 的成功行为保持项。
- 两组各额外 20,000 次成功更新；每步原输入仍为 4000×100=400,000；几何额外项仍沿用 v1 的 512 个正负起点和权重 0.01。
- 两组相同新采样种子、FP16 主损失 + FP32 几何/保持项、TF32 关闭。固定 cosine 学习率从 1e-4 到 1e-5，无新的 Plateau 调参。**新 Adam 和学习率对两组都一样，所以 A 是匹配的继续训练对照，不是原 v1 调度器不变的 resume。**
- 原有全局查询采样流的 20k 滚动 SHA256 必须 A/B 相同；否则实验不接受。
- 只用最后 20k 权重，不根据测试挑 checkpoint。保持权重、容许偏差为预先声明的研究设置，不声称已最优化。

## 成功行为如何生成、怎样保持

先从原 R1 train/val 分区与已有 V4 正负配对的交集中，预先固定每 sensor/cohort 的尝试：train 各512，val各128；共10,240个计划尝试。Local 用原负侧 q；Uniform 只由解析几何从最多32个独立随机 q 中选择外侧起点，保留正侧见证。不读取旧1963或前一轮512空间点的成败记录来选训练数据。

**先冻结全部起点，再用冻结的父模型及完整原 solver 运行**。每个尝试的成败都保存；无正侧/外侧几何见证、没找到成功的情况不补抽、不伪装成不可行。只有 FOV 成功轨迹进入成功保持库，train/val 严格分开。

每条成功轨迹最多保存8个实际访问配置，包含起点、候选与 q_zero；记录父模型值和完整运行时的单步投影/上升响应。旧 runtime 的两个单步公式与可微保持实现逐点校验。B 每步全局256个保持点，**先等概率选成功轨迹，再选轨迹内位置**，避免长轨迹被过度采样；16个 sensor×cohort 分层平衡。

保持项不是强制 f=0、单位梯度或新距离标签：
- 值偏差小于0.003不惩罚，超出部分按 max(|父值|,0.03) 归一化；
- 投影/上升位移偏差范数小于0.0025 rad 不惩罚，超出部分按0.05 rad归一化；
- 三项取 SmoothL1 平均，再乘0.01。允许有限变化，**不保证完整 solver 绝不退化**。

冻结教师只用于离线建库；最终在线只有一个同架构模型，不增加本地网络数量或 solver 查询预算。

## 一次提交完整双臂实验

仓库根目录执行：

    export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
    env -u NODE bash experiments/hierarchical9_solver_goal_keep_v2/run.sh train

入口执行CPU测试、父权重/源码/缓存/原始数据哈希、真实正负见证、真实URDF-FOV与真实模型参数反传检查，然后提交**一个四卡3090作业**，按“起点冻结→教师建库→A20k→B20k”顺序执行。默认排除3090node1；Slurm时限3天只是提交上限，不是耗时预测。不会再提交独立GPU pilot。

输出：`outputs/mainline_b/h9_solver_goal_keep_v2/formal/`。完整新代码以及依赖都复制为只读内容快照；Slurm spool 不被当成源码位置。原所有权重和数据只读。

    bash experiments/hierarchical9_solver_goal_keep_v2/run.sh status
    bash experiments/hierarchical9_solver_goal_keep_v2/run.sh log
    bash experiments/hierarchical9_solver_goal_keep_v2/run.sh ranks
    bash experiments/hierarchical9_solver_goal_keep_v2/run.sh summary

只有任务已经终止且未完成时，执行：

    bash experiments/hierarchical9_solver_goal_keep_v2/run.sh resume

完成的建库分片、A或B不会重跑；未完成 arm 恢复本实验 checkpoint 的 Adam、GradScaler、四rank RNG、update序号和样本流。崩溃后的多余日志保留备份并回退至 checkpoint 步，不假装已完成未保存的更新。父实验目录不能用本入口 resume。

## 正式验收与权重

两组完成后：

    bash experiments/hierarchical9_solver_goal_keep_v2/run.sh verify
    bash experiments/hierarchical9_solver_goal_keep_v2/run.sh eval

确认集为**另外512个新非网格位置**，seed在协议里固定，排除旧网格与前一次512个点。点集合在微调提交前已固定；详细几何起点在评估前独立生成并冻结。不沿用旧测试集的成功数门槛。相同原完整 solver 同时评价原R1、父Solver-Goal、A、B。

主比较是 B vs 父模型与 B vs A；Local/Uniform、全部sensor、按x分组95%区间、几何覆盖、解析裕量与调用量均报告。区间不代表多训练seed稳定性。未确认可见见证的组合保留为UNRESOLVED，不当作物理不可行。

输出在 `confirmation/report/`。不自动推广新模型；若 B 未优于父模型，保留父模型。每个 arm 的 `runtime.pt` 是同 R1 state_dict/架构的轻量导出，保留真实“父50k + 微调20k”的血缘，不伪称从头训练。`final.pt` 含恢复所需状态，不覆盖父文件。

    bash experiments/hierarchical9_solver_goal_keep_v2/run.sh summary
    bash experiments/hierarchical9_solver_goal_keep_v2/run.sh verify
    bash experiments/hierarchical9_solver_goal_keep_v2/run.sh pack

打包诊断、源码、参考成败和评估，不打包 .pt 权重及 .npz 缓存。每组基础输入约80亿行；B还多保持项计算，因此不是严格等算力。无LOS/碰撞/轨迹/实际观测认证；原安全流程保持不变。

## 已执行测试的边界

见 TEST_RESULTS.md。CPU通过不等于真实HPC数据、Pinocchio/URDF依赖或CUDA/NCCL已运行。真实输入预检在提交前执行；完整分片建库校验在正式作业中执行。
