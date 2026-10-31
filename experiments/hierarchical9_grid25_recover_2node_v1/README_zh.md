# Grid25 重启恢复：一个24小时作业占用两个节点、8张3090

这是已有25mm采集的恢复/调度层，不是新数据方案。默认节点3090node1,3090node2；始终一个sbatch作业。资源为 --nodes=2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:3090:4，每节点4卡，共8卡。每个逻辑段在同一分配内用一次srun启动每节点一个协调器，协调器各派生4个单卡worker。不是两次sbatch，不是任务数组，不是两个独立作业，也不使用DDP。

## 使用

先拉取本次提交，确认重启前的作业已经退出：

    export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
    export GRID25_ROOT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"
    SCRIPT=experiments/hierarchical9_grid25_recover_2node_v1/run.sh
    bash "$SCRIPT" recover
    NODES=3090node1,3090node2 bash "$SCRIPT" draft
    NODES=3090node1,3090node2 bash "$SCRIPT" next

recover不申请任务；draft只打印资源草稿，不调用调度器；next先test-only（不产生实际分配），然后恰好一次真实sbatch。没有自动取消、资源降级、拆成多个作业或后继提交。拿到任务号后不可重复next。以前账户报过AssocMaxNodePerJobLimit；此次代码不能绕过该限制，服务器重启也不证明权限已改变，需以本次实际排队状态为准。

## 清理边界：不是清空重采

recover要求squeue成功且账户队列为空，检查原/新入口互斥锁及所有分片锁。只隔离已知命名的不完整分片临时目录、无完成标记的分片目录和未发布合并临时目录，移到two_node_v1/recovery/时间戳/。不会永久删除这些文件，不释放它们占用的磁盘空间；不会删除完整分片、历史日志、gate报告、模型、参考NPZ、source.env、job.json、manifest或锁文件。

先逐个核验已发布分片的哈希与原引擎来源；已发布但损坏的分片报错停止，不以清理名义删除。发现未知路径/符号链接不盲删。旧锁文件存在不等于仍持锁，不通过unlink锁文件“解锁”。作业历史缺失会记录为ACCOUNTING_UNAVAILABLE，而非伪造COMPLETED；队列读取失败、仍有任务、明确非终态记账或仍持有分片锁均拒绝清理/提交。

用户此前两个4小时作业以及被重启中断的24小时作业，已发布且通过核验的全部分片继续使用。在CPU/GPU内存中尚未保存的点只能重算；不能凭日志8/64把它算成完成片。

## 不改变采样引擎

加载fast_segments_v1/manifest.json指向的同一冻结run.py（Git blob e695e1f357669bd91918ca47546d12bc93b4f46c），保留数值一致性检查、20k初值/500候选/80步Adam、标签、种子、网格和64点原子分片。每节点的局部worker rank仍为0..3，日志使用全局0..7且gate按节点隔离。八进程通过原分片锁领取同一任务池；每段两节点都退出后才验证覆盖并进入下一段，避免提前宣布完成。

跨节点启动后会实际检查共享文件可见性与flock互斥；检查失败停止，不采集。输出仍是同一个formal/shards，最终使用原fast_segments入口合并，所有来源可合并。不修改旧的两个单节点入口或其快照。

申请24小时，实际启动后23h30m停止领取新片，完成当前片后退出；所有段使用同一绝对截止。提前30分钟USR1同样请求安全停。极慢I/O、硬超时、重启或强制取消仍可能丢失最多8个未发布分片，已完成片保留。所有点采齐则提前结束，不自动合并、不训练。

## 查看与继续

    bash "$SCRIPT" status
    bash "$SCRIPT" summary
    bash "$SCRIPT" ranks
    bash "$SCRIPT" log
    bash "$SCRIPT" verify

status对应two_node_v1/latest_submission.json；日志在logs/two_JOBID.out和two_node_v1/runs/JOBID/segment_NNN/。summary累计所有已保存分片，并按持续8卡外推剩余采集时间（不包括排队、校验启动、硬中断及合并；不是保证2倍提速）。旧4卡入口summary中的@4不能当作8卡剩余时间。

正常窗口结束后重复next；再次发生重启则先recover再next。全量2707片齐全、无作业后，按原流程提交一个独立单节点合并作业：

    NODE=3090node2 bash experiments/hierarchical9_grid25_fast_segments_v1/run.sh merge

本目录verify仅核验已发布分片；最终仍需formal/dataset/complete.json以及原入口verify通过。当前为基础Q0采集，不是新网格训练缓存已经完成。

## 测试范围

本地25/25项CPU/模拟调度测试通过，零跳过；覆盖资源写法、单真实提交、失败不重试、清理保留/隔离/损坏拒绝、真实跨进程锁、跨段截止、原worker参数、日志隔离及Slurm spool绝对路径。不在本地访问服务器文件、运行真实Slurm或CUDA；本地文件锁测试不代替服务器跨主机实测。
