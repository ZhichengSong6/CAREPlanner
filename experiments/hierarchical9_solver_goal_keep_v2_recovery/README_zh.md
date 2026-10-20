# Keep-v2：资源冲突后的原任务恢复

这是资源申请与启动检查入口，不是新的训练算法。针对 job 18150 的日志：A_continue 在 start=0 的第一次 prepare_batch 解码时 CUDA OOM；四张卡均显示 PID 1570652 占约 22.83 GiB，训练 rank 自身的 PyTorch 分配量仅约 134–220 MiB。日志不足以确认该 PID 的用户或所属任务，不能据此杀进程或宣称是某个用户的作业。

## 运行

在原仓库根目录拉取包含本目录的提交，然后运行：

    export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
    export K2_ROOT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_solver_goal_keep_v2"
    env -u NODE bash experiments/hierarchical9_solver_goal_keep_v2_recovery/run.sh resume

它会确认旧作业不再运行，调用原 frozen k2_common.verify 和 complete.json 哈希验证，然后申请单节点四张 3090，并显式请求 Slurm --exclusive。默认仍排除 3090node1；可显式设置 NODE 指定由调度器分配的节点，不把登录节点的 nvidia-smi 当作远端资源状态。

实际分配节点后，独立检查进程打印 nvidia-smi 和 CUDA 逻辑设备的全局可用显存。四张卡每张至少 20 GiB 空闲才启动原任务。20 GiB 是保守的资源准入规则，不是对后续峰值内存的保证。若检查失败，停止并留下 GPU 进程信息，不杀进程、不清空他人显存、不缩小 batch、不偷偷切 CPU。

Slurm --exclusive 是独占申请，实际隔离仍受集群/分区配置影响，也不能控制调度器外的进程。若集群拒绝独占申请，不自动降级成共享重试，应依据报错处理资源问题。

## 原数据与科学协议保持不变

本目录完全不修改 hierarchical9_solver_goal_keep_v2 的原代码、冻结的 sources、formal/job.json、dependencies、源模型、训练标签或原 replay。通过检查之后执行的仍是：

    bash "$K2_CODE/run.sh" _worker

其中 K2_ACTION=resume。原 worker 按 complete.json 校验并复用 reference_inputs、reference_mined 和 replay，继续 A_continue、B_keep。若某组已有 latest.pt，由原训练器校验和恢复；没有保存更新的组仍从固定 Solver-Goal 教师初始化。不会把恢复整个流水线误解为“已有 A20k checkpoint”。

本入口仅新增 recovery_sources、资源日志和恢复任务状态；source.env 不变。旧状态查看命令继续有效：

    bash experiments/hierarchical9_solver_goal_keep_v2/run.sh status
    bash experiments/hierarchical9_solver_goal_keep_v2/run.sh log
    bash experiments/hierarchical9_solver_goal_keep_v2/run.sh ranks

A/B 两组真正完成后，才使用原 verify/eval。缺少 A_continue/complete.json 是训练未完成的结果，不是需要手动创建的文件。

## 本地测试记录

2026-10-20，本地实际运行：

    PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s . -p 'test_*.py' -v

9/9 项通过，零跳过，耗时约 4.1 秒。覆盖上传日志中的低显存数值拒绝、四卡与 20 GiB 门槛、原数据/源码不变、损坏检测、已完成任务拒绝、登录节点禁止 GPU 检查、Slurm spool 路径处理，以及临时 Git 仓库和模拟调度器中的独占恢复提交/重复提交拦截。

这是 CPU/模拟调度器测试。真实集群上的独占分配、nvidia-smi 与 CUDA 查询，以及恢复后的完整训练均未在本地执行。它不证明 CUDA 训练一定不再 OOM。
