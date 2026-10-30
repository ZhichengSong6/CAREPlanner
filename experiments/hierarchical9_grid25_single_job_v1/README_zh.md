# Grid25：一个作业，单节点4卡或两节点8卡

这是调度修正，不是新采样方案。原 `hierarchical9_grid25_data_v1` 数据协议、采样器、173225个空间点、Q0数量、标签、随机种子及分片均保持原样。只调用原入口的 `prepare`（不提交）；随后由本目录提交**恰好一个 Slurm 作业**。

## 运行（先拉取包含本目录的Git提交）

    export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
    NODES=3090node2,3090node3 bash experiments/hierarchical9_grid25_single_job_v1/run.sh submit

对应一个sbatch：`--nodes=2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:3090:4 --exclusive`。GRES为每节点四卡，总共八卡。只有一个任务号；作业内的一次srun在每节点启动一个协调进程，每个协调进程派生四个单卡采集进程，动态领取原分片。不是两次sbatch，也不是两个独立作业，不需要DDP。

单节点四卡（只选择一个方案；不要在八卡作业排队时另提四卡）：

    NODES=3090node2 bash experiments/hierarchical9_grid25_single_job_v1/run.sh submit

默认单节点3090node2；可显式选择3090node3。账户“最多一个作业”与“一个作业能用多少节点/GPU”是不同限制。代码不能保证账户允许两节点；此前的AssocMaxNodePerJobLimit若仍存在，需管理员调整相应权限/队列，不能通过脚本绕过。

入口先确认账户没有其他已排队或运行任务，再跑本目录CPU测试，调用原数据prepare（含原数据检查和测试），冻结调度源码。随后执行 `sbatch --test-only`（不会创建作业）。若失败就停止，不降级、不拆分、不重试；测试通过也不保证立即有资源或获得永久许可。真实提交只有一次；如作业进入PENDING，读取原因，不能再提交第二个任务。

## 进度与恢复

    bash experiments/hierarchical9_grid25_single_job_v1/run.sh status
    bash experiments/hierarchical9_grid25_single_job_v1/run.sh log
    bash experiments/hierarchical9_grid25_single_job_v1/run.sh ranks
    bash experiments/hierarchical9_grid25_single_job_v1/run.sh summary

输出根仍为 `.../outputs/mainline_b/h9_grid25_data_v1/`（可用GRID25_ROOT覆盖）。原latest_submission.json里只写一个job记录。worker日志编号全局0..7，不会跨节点重名；节点日志为node_JOBID_HOST.log。

仅在原作业退出且数据尚未完成后恢复：

    NODES=3090node2,3090node3 bash experiments/hierarchical9_grid25_single_job_v1/run.sh resume

恢复不修改原job.json、source.env、采样源码快照或已发布分片；四卡/八卡可更换。节点内存检查沿用原采集器；双节点先实测共享文件可见性和flock互斥。一个节点失败，srun终止该步骤的其他节点；保留完整分片，不自动合并不完整数据。

所有节点完成后才在批处理主进程中调用原merge、verify，各一次。最终产物仍为 `formal/dataset/visibility_grid25_fovonly.npz`。

    bash experiments/hierarchical9_grid25_single_job_v1/run.sh verify
    bash experiments/hierarchical9_grid25_single_job_v1/run.sh pack

不更改网格/训练/模型，不杀其他用户进程，不自动取消作业、不提交后续作业。不推断账户权限。旧入口的多作业或只允许单节点的submit都不再用于本方案；原入口的prepare/状态/采集功能原样复用。

## 测试边界

本目录本地18项CPU测试：单sbatch请求4/8卡、test-only失败无实际提交、实际提交失败无重试、队列读取失败拒绝、已有作业拒绝、单srun双协调、全局worker编号、节点失败不合并、合并在步骤结束后恰一次、真实双进程共享锁、源码冻结/损坏拒绝、Slurm spool绝对路径和语法。Slurm与CUDA分配为模拟；共享锁在本地文件系统测试，不是已经验证服务器跨主机文件锁。原真实HPC数据采集和调度权限未在本环境运行。
