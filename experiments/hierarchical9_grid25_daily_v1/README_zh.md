# Grid25：单作业约24小时连续采集

只增加调度层；不修改已验收的fast_segments_v1、冻结快照、manifest、43个逻辑段、64点分片、已完成数据或采样参数。不是把一个4小时段的Slurm上限改大后仍只跑一个段。

## 用法

先拉取包含本目录的Git提交，账户没有运行/排队任务后：

    export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
    export GRID25_ROOT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"
    NODE=3090node2 bash experiments/hierarchical9_grid25_daily_v1/run.sh next

只调用一次真实sbatch，单节点四张3090，允许3090node1或3090node2，默认node2。不提交多个作业、job array、依赖作业或自动后继作业。若旧4小时作业仍在运行或排队，就停止，不取消它、不删除数据；旧任务退出后再运行本命令。新入口不能修改已经提交的旧作业。

申请24小时上限，在作业实际启动后的23小时30分停止领取新分片，保存当前64点分片后退出。提前30分钟的USR1信号也请求停止领取。按此前每片约16分钟的速度，通常会接近24小时结束；不是精确24小时保证。极慢分片、长时间磁盘阻塞或硬时限/强制取消仍可能丢失尚未发布的最多四个分片，已发布文件不动。全部采齐时提前退出。

每完成一个原逻辑段，在同一个分配、同一个job ID内启动下一个原逻辑段；每段继续调用原冻结worker，四张卡都保留原同设备数值一致性门槛，不跳过验证。所有段共享一个绝对停止时间，绝不每段重新开始23.5h计时。软截止到达时可在第六段中途停；下次仍从未完成片继续，不要求一整段结束才能保存。

不复制或重新定义采样器，不改输出格式、点坐标、候选数量、迭代数、随机种子或标签。新分片的producer仍指向原冻结采集引擎；数值gate存于daily_v1/runs/JOBID/segment_NNN/。保留原64点原子发布、原数据完整性校验和每片来源。

## 查看与继续

    SCRIPT=experiments/hierarchical9_grid25_daily_v1/run.sh
    bash "$SCRIPT" status
    bash "$SCRIPT" summary
    bash "$SCRIPT" ranks
    bash "$SCRIPT" log

新任务用这个新入口看日志；旧fast_segments入口的status仍指向旧任务。summary读取同一份全局已完成数据；ranks显示当前/最近处理的逻辑段的四个worker。每日记录独立存于GRID25_ROOT/daily_v1/，不重写旧latest_submission、source.env或job.json。

作业退出后，再执行同一条NODE=3090node2 ... next即可。它自动跳过已有完整分片。如果没有剩余分片，打印READY TO MERGE且不提交任务。

    bash "$SCRIPT" verify
    bash "$SCRIPT" pack

verify校验已有分片及其原引擎来源，不把部分完成称作整个数据集完成。pack包含日运行/原分段的检查记录和源码，不打包大型NPZ/权重。

## 全量采齐后合并

日作业不自动合并；全部采齐且账户无任务时，使用原入口：

    NODE=3090node2 bash experiments/hierarchical9_grid25_fast_segments_v1/run.sh merge

合并格式与验收规则保持原样，仍需最终dataset/complete.json及原verify通过。日作业完成不等于全量数据完成。

## 测试边界

本目录test_daily.py本地19/19项通过、零跳过，为CPU与模拟调度测试：实际执行了单作业参数构造、空队列与终态检查、busy/失败/模糊响应不重试、源码冻结与损坏检测、原数据保留、原worker调用参数、四GPU隔离、跨段继续、绝对截止、部分段恢复、失败不继续，以及shell/spool路径检查。

没有在此环境执行24小时真实HPC/CUDA任务，不宣称新的采样加速或数值结果。原采样算法和已采数据均不变；服务器每段仍执行原数值资格检查。
