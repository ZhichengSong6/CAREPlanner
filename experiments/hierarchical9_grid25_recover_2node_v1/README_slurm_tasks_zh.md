# Grid25：按站点示例改为每节点 4 个 Slurm task

这是新增的资源编排入口，不改原来的数值采集引擎。原 `run.sh` / `launch.py` 保留不变；新入口是 `slurm_tasks.sh`。不要再用旧的固定双节点入口提交。

## 与用户提供的官方示例对应

保留 `#!/bin/bash`、`#SBATCH`、每节点 4 个 task、每 task 16 个 CPU 和一次 `srun` 跨节点启动。新入口生成的 `batch.sbatch` 核心为：

```bash
#!/bin/bash
#SBATCH --job-name=g25_tasks
#SBATCH --nodes=1-2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:3090:4
#SBATCH --partition=GPU
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --no-requeue
```

固定两节点时生成 `--nodes=2`；固定一节点时生成 `--nodes=1`。CPU 请求是每节点 4×16=64，不是全节点总共 16。每个 Slurm task 只运行一个旧的单卡采集 worker，绝不在四个 task 内各启动四个 worker。

相较用户贴的样例，明确增加了 GPU GRES 申请：`--ntasks-per-node=4` 自身是进程数，不等同 GPU 预留。保留此前 GPU 分区、24 小时窗口和 23h30m 软截止。用户贴的例子没有说明其默认分区、GPU 分配插件或账户权限，因此不能声称完全照抄就解除权限限制。这里没有擅自更换账户或 QOS。

本任务是独立分片采集，不是 DDP 训练，不需要 NCCL 通信；没有照抄 `trainer=ddp` 或其他用户的 Conda 路径。使用现有 `VIS_PYTHON` 的绝对路径，与原 viscdf 环境一致。

## 拉取后使用

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
export GRID25_ROOT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"
SCRIPT=experiments/hierarchical9_grid25_recover_2node_v1/slurm_tasks.sh

bash "$SCRIPT" test
bash "$SCRIPT" cancel-blocked
bash "$SCRIPT" recover
bash "$SCRIPT" summary
GRID25_NODES=auto bash "$SCRIPT" draft
GRID25_NODES=auto bash "$SCRIPT" next
```

逐步执行，遇到 `[STOP]` 不继续。`cancel-blocked` 是显式操作，只针对新/旧入口最近记录、当前用户、正确工作目录/日志根目录、处于 PENDING 且原因是节点限制的作业。使用 `scancel --state=PENDING --user=...`，不会取消已经运行的作业。结束的作业跳过；身份不符、正在运行、取消仍未完成或账户还有其他任务就停止。不会取消其他人的作业。

`recover` 与原恢复层一致：核验已发布分片，仅隔离未发布输出；不清空数据，不删除模型、历史日志、gate、manifest、source.env 或锁文件。旧版本和新版本共用同一个分片池和互斥锁。用户给出的日志已确认 190 个分片、12160 个点；服务器最新数字由本次 recover/summary 核实，而非写死。

`next` 先做 test-only，再恰好一次真实 sbatch。成功后立即查询真实队列，不把 test-only 的预计启动时间当成能运行的保证。提交成功但 PENDING 会明确显示原因。不自动取消新作业，不自动重提，不提交两个并行作业。拿到 JobID 后不要重复 next。

## 自动模式与账户节点数限制

`GRID25_NODES=auto` 默认申请节点范围 1-2，不指定固定总任务数或必需的两台主机。若只读 `sacctmgr` 输出中匹配账户/分区的可见关联明确有 `node=1`，保守地直接申请一节点四卡。多个可见关联取保守上限，这不是完整的账户、继承、分区 QOS 和作业 QOS 解析器。

否则提交 `--nodes=1-2`，由调度器在权限和资源允许范围内选择实际节点数。无法读取策略不等于有双节点权限，也不能保证任意版本/站点配置一定可调度。运行时从 `SLURM_JOB_NUM_NODES` 取实际值，`srun` 启动 4 或 8 个任务；启动后不动态增减节点。

`AssocMaxNodePerJobLimit` 不是 CPU 线程数太少或 `ntasks-per-node` 写法不对。此前固定双节点作业确实受这个限制；本示例不能证明当前账户能用双节点。若自动范围仍因策略受阻，先通过 `cancel-blocked` 取消经核验的未运行作业，再使用：

```bash
GRID25_NODES=1 bash "$SCRIPT" draft
GRID25_NODES=1 bash "$SCRIPT" next
```

确认账户获准双节点后可明确请求：

```bash
GRID25_NODES=2 GRID25_NODELIST=3090node2,3090node3 bash "$SCRIPT" draft
GRID25_NODES=2 GRID25_NODELIST=3090node2,3090node3 bash "$SCRIPT" next
```

固定 nodelist 表示两台都必需，不是候选池，所以自动模式拒绝 `GRID25_NODELIST`。自动模式依据 `gpu:3090:4` 选择 3090 节点。`GRID25_ACCOUNT`、`GRID25_QOS` 仅供填写管理员已授权的值；默认为空，不编造任何账户或 QOS。

## GPU、续采与诊断

srun 每节点预留四张 3090；task 根据 Slurm 给的可见 GPU 列表和 `SLURM_LOCALID` 分卡，不猜物理卡号。每个 task 验证只看到一张 3090，并将真实 GPU UUID、主机名和 rank 发布到共享目录。所有 task 的 GPU/主机拓扑核验通过后才采集；双节点还实际测试跨主机共享文件可见性与 flock 排他性。没有 UUID 或 GPU 重复会停止，不冒险共用 GPU。

旧引擎每节点 local rank 仍为 0..3；新全局 task rank 为 0..3 或 0..7，gate/request 目录按全局 rank 分离。以原子完整分片续采，内存里未发布的中间进度不承诺恢复。

```bash
bash "$SCRIPT" status
bash "$SCRIPT" log
bash "$SCRIPT" ranks
bash "$SCRIPT" summary
bash "$SCRIPT" verify
```

状态和输出在 `$GRID25_ROOT/slurm_tasks_v2/`；主日志为 `logs/tasks_JOBID.out` / `.err`；每段每 task 日志在 `slurm_tasks_v2/runs/JOBID/segment_NNN/rank_N.out` / `.err`。旧结果仍在 `formal/shards/`。summary 沿用旧统计的 `remaining@8`，四 worker 的粗略外推是该值两倍，不是完成时间承诺。数据齐全后只提示 READY_TO_MERGE，不自动合并或训练。

## 验证范围

新增 31 项本地 CPU/模拟调度测试通过，含两种规模、单次真实提交、未知权限不伪造保证、失败不重试、旧/新记录检查、GPU 重复拒绝、单 task 单 worker、跨段同一 JobID、待取消作业身份与状态过滤，以及共享目录文件操作的模拟。另有 Python 编译与生成 Bash 语法检查。

本地没有实际访问服务器数据、运行 Slurm、验证真实 CUDA/GPU UUID 或跨主机文件锁。原恢复文件按 Git blob `bd10c9bd07f0b021c9aa99b2c083f44eaa2c0d90` 固定校验；冻结采集引擎仍按原来的 blob/manifest 校验，不在此次改动范围内。

技术依据（站点示例来自用户粘贴；以下为补充通用定义）：
- https://slurm.schedmd.com/sbatch.html
- https://slurm.schedmd.com/srun.html
- https://slurm.schedmd.com/job_reason_codes.html
- https://slurm.schedmd.com/sacctmgr.html

## 最后一轮：不限时、采齐即退出（新增，显式启用）

已有 24 小时作业不受影响：每次提交都会把启动器冻结到独立快照。**必须等上一轮（例如 19094）完全退出**，确认无其他本人排队/运行任务，再启动下述新作业；不要取消旧作业来切换。本改动仅调整提交器/窗口，不改冻结采集引擎、原始数据、每分片数值和归档格式。

当前分区已显示 `MaxTime=UNLIMITED`；Slurm 的 `#SBATCH --time=0` 表示请求不限制运行时间。是否真正获批仍以本机调度器（及账户、QOS、集群策略）为准，不保证作业绝不会被管理员取消、服务器重启或因程序错误退出。无时间上限的申请也可能比短作业更难排队。

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
# 需要先 git fetch/merge 更新本分支到含不限时功能的提交。
export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
export GRID25_ROOT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"
SCRIPT=experiments/hierarchical9_grid25_recover_2node_v1/slurm_tasks.sh
unset GRID25_NODELIST
squeue -u "$USER"                     # 必须先确认前一轮结束
bash "$SCRIPT" test                    # 本地 CPU/模拟调度检查
bash "$SCRIPT" log                     # 前一轮应为 WINDOW_PAUSED
bash "$SCRIPT" verify                  # 所有已发布分片必须校验通过
bash "$SCRIPT" summary                 # 若已 2707/2707，则不要再采集
GRID25_NODES=1 GRID25_UNLIMITED=1 bash "$SCRIPT" draft
GRID25_NODES=1 GRID25_UNLIMITED=1 bash "$SCRIPT" next
```

预览生成的 batch 应包含 `#SBATCH --nodes=1`、`#SBATCH --time=0`，且不再包含 `--signal=B:USR1@1800`；提交日志应看到 `[duration] UNLIMITED`，运行日志应看到 `unlimited=True`。**一个 sbatch / 一个 JobID / 一节点四张 3090**，启动后直到分片全部完成（`READY_TO_MERGE`）才自然退出；不进行自动合并或训练。内部不再以 23h30m 停止分片领取。旧 worker 必须有数字类型 `deadline_unix`，新入口仅在不限时模式传入公元 9999 年末的有限哨兵；它不是现实意义的运行时间限制，实际无限运行由 Slurm `--time=0` 与 batch 循环共同实现。

同一段如因其他 worker 已持锁而留下未完成分片，在确实增加已发布分片的条件下打印 `[UNLIMITED CONTINUE]` 再补缺；如果没有任何新完成分片则立即报错停止，避免死循环。运行中仍保持原有原子分片与文件锁，管理员取消/重启时只保留已完整写入的分片。遇到 `[STOP]` 或 `FAILED`，应先读日志、必要时执行 `recover`；不允许手工删除分片或重复提交。

如想继续使用旧的 24 小时分段窗口，**不要设置 `GRID25_UNLIMITED`**，沿用 `GRID25_NODES=1 bash "$SCRIPT" next`。如果该变量已在终端持久导出，用 `unset GRID25_UNLIMITED` 关闭。

官方说明：https://slurm.schedmd.com/sbatch.html （`--time=0` 请求不设时间限制）。
