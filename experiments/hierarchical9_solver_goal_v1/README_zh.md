# R1 Solver-Goal Scratch v1

**这是一套尚未在真实数据/CUDA 上验证的训练试验，不是已经超过 R1 的模型。**
保持原 R1 的结构、输入输出、原始 4000×100 查询流、损失、Adam、FP16/GradScaler/DDP
和学习率调度规则；不加载旧权重。从头训练 50k，前 10k 步仅原目标，10k–15k 将新增项
从 0 线性增加到 0.01，之后固定。改变学习目标，不改变在线 solver。

## 新增项

读取已经构建的 h9_paired_slope_scratch_v1/formal/pairs，不重跑 V3/V4。
限定同一 x、sensor、有正侧可见见证的原始 +/- 配对，再与原 R1 train/val x 求交集。
每 rank 每 sensor 8 对；每对的正侧/负侧都作为初始配置，总共 512 个额外配置/成功更新。
新 NumPy Generator 使附加抽样不消耗原 R1 随机流。

训练连续展开 **3 次投影 + 1 次上升**。单步数值公式沿用原 runtime 的阻尼 .5、
投影 cap .25、上升 .05、关节 mask 和限位，但 **不包含其预测提前停止、符号跨越二分、
8步回退分支**。这是可微训练 surrogate，绝不能称为完整 solver 成功率。

网络 f 和 df/dq 形成更新后的 q_K，现有可微运动学计算 q_K 的真实 FOV margin g。
目标 margin 为 min(0.005 m, 0.5×该配对实际正侧见证的 g)，其正裕量具有可见见证，
但不证明当前 solver 能达到它，不证明最短配置距离或路径/碰撞/LOS 可行。
损失为 SmoothL1(relu((target-g(q_K))/0.01m), 0)，八 sensor 等权。
不增加边界零值/单位梯度/normal cosine/配对斜率损失，不改变旧标签。

反传链：theta -> f、df/dq -> 每个未 detach 的中间 q -> g -> loss -> theta。
仅最初 q 从旧样本 detach；中间 q 不 detach。参数梯度由原 .backward() 和 DDP 累积，
不是使用 autograd.grad 直接替换 DDP 参数更新。额外项只放入一个 microbatch，
AMP 重试时精确重用原样本和额外配对。原 R1 loss 仍决定 LR 调度；规则相同不意味着
学习率随更新步数的实际轨迹必然相同。计算成本额外增加，不能称严格等 GPU 算力。

## 验证与边界

提交前读取真实数据，核对旧代码 Git blob、R1 checkpoint SHA、完整原数据 SHA、
训练缓存及 pair-cache 各文件 SHA、原 R1 train/val 划分、paired FOV 与原 oracle 一致性。
实际 R1 随机初始网络在真实配对和真实 FK 上做一次 CPU 参数梯度检查，不更新模型。
每次抽样也检查真实正负见证，错误即拒绝，不伪造标签。

原训练源码不修改，只通过独立适配器加一项。完成后原 50k x/q stream 必须与 R1 相同。
新 final.pt 保持原 R012 的架构/format，metadata.solver_goal_v1 记录新试验身份与配置；
不代表原始 R1 checkpoint，也不自动晋升。resume 只接受本目录自己的 latest.pt。

训练中 outside_start_fov_pass / inside_start_fov_pass 只是短 surrogate 的几何指标，
不是完整 1963-case solver 指标。固定输出 final_50000，不拿测试集挑 checkpoint。
full solver 测试入口才用原 runtime_probe（含所有原始条件分支），原 R1 必须复现
618/683 local、673/1280 uniform、1291/1963 total。该1963集是反复使用的回归集，
不作为独立泛化证明；训练不会读取它。任何改善仍需要独立验证确认。

## 命令

    export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
    bash experiments/hierarchical9_solver_goal_v1/run.sh train

单节点四张3090，一次正式50k，不单独提交 GPU pilot，默认排除3090node1。
入口已有防重复提交与源码快照，不用下载训练数据、不覆盖原结果。

    bash experiments/hierarchical9_solver_goal_v1/run.sh status
    bash experiments/hierarchical9_solver_goal_v1/run.sh log

只有作业已停止、有 latest.pt 且没有 final.pt 时：

    bash experiments/hierarchical9_solver_goal_v1/run.sh resume

完成训练后一次完整固定 solver 评估：

    bash experiments/hierarchical9_solver_goal_v1/run.sh eval
    bash experiments/hierarchical9_solver_goal_v1/run.sh summary
    bash experiments/hierarchical9_solver_goal_v1/run.sh pack

输出独立存放于 outputs/mainline_b/h9_r1_solver_goal_v1。pack 不包含 .pt 或大型缓存。
必须看全体、local/uniform、逐 sensor、成对新增/丢失和耗时。仅训练完成或
surrogate 变好，不代表得到更好的网络。这里没有系统集成/论文包装任务。
