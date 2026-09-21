# 给服务器侧 Codex 的执行说明

请使用专用分支 `mainline-b/h9-scratch50k-r012-v1` 中的 `experiments/hierarchical9_offline_relabel_v1/` 做离线标注。代码已纳入仓库，不需要再次传 ZIP。

本轮范围：固定查询、配对旧/新标签、检查几何与数值质量。不启动训练，不加载 R1 checkpoint，不跑 ROS/Gazebo，不修改 runtime、URDF、FOV 参数、安全阈值、旧数据或任何权重，不推送远端。

1. 确认当前代码仓库及数据仓库路径。代码通常在：
   `/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b`。
   数据通常在：
   `/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/src/care_visibility_cdf/data/visibility_yiming_style_grid30_q20000_k500_fovonly.npz`。
   路径不存在则报告，不能换成其他数据或伪造测试数据。
2. 确认已有本实验目录，保留本地未提交修改。本执行任务不自动 pull/merge/reset/clean/stash，也不另建分支；需要更新时由使用者按 README 第 3 节先完成 fast-forward。
3. 从仓库根目录校验 `SHA256SUMS`。确认已有 viscdf 环境含 NumPy、SciPy、PyTorch、URDF 依赖，不自动升级全套环境。
4. 执行合成测试，并提交 `submit.sh smoke`。不要直接跑 pilot 或正式大规模标注。
5. 预检失败时保留错误与原数据，修复依赖/路径需要明确记录；不要放松数值检查或改 FOV/URDF 来获得 PASS。
6. 缓存里 low confidence/无解的数据保留 false validity；不能把 NaN 改成“真实零距离”，不能因一个 sensor 标注失败就重定义 union 的 winner。
7. 输出 `preflight.json`、`label_summary.md`、`label_summary.json` 和输出目录。分别报告执行完成率、值标签有效率、梯度标签有效率及 union 覆盖。
8. 如任务中断，使用相同 MODE、OUT、源码和 solver_config，以 `RESUME=true` 续跑；已完成分片校验后复用。不要删除结果目录重来。
9. 只有收到后续明确指令，才扩大 pilot 或进行 R1 配对训练。新数据 reader 返回分开的 value/gradient/union masks，不能直接套旧 objective 的单 mask / 自动 union 重算逻辑。

必须区分：合成测试通过、真实 oracle 预检通过、标注处理完成、标签有效、R1 训练改善。这五件事不等价。当前几何优化输出的是 best-found 近似局部解，不是全局最近距离证书。
