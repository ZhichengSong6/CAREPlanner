# Grid25：不降采样的目标sensor加速与手动分段采集

本目录新增入口，不修改原 `grid25_data_v1` / `grid25_single_job_v1`、冻结采样源码、job.json、已有分片或训练代码。
读取现有 GRID25_ROOT 下已经准备好的25mm采集任务；继续使用同一个基础数据集目录。

## 保持什么，改变什么

保持173225个空间点、base_link、原七关节范围和mask、八sensor顺序、FOV-only、随机种子规则及原NPZ继承的采样参数。
真实历史配置为每点20000个随机q、每sensor500候选、K=500、80步Adam、原lr和epsilon；本入口不降低任何数量，不早停Adam，不扩大epsilon，不增加LOS。
保持 canonical `x/q/k/valid_fov/g/sensor_margins/active_planes` 和计数布局、dtype、有效性mask、+inf无效槽位；最终仍用原流式合并和旧训练读取格式。

加速仅两项：
1. 迭代优化S_i时，用原 visibility_g_batch 的单sensor列表，仅构建该链的FK/FOV和反传图；初始筛选与最终诊断仍计算八sensor。
2. 复用相邻Adam步骤相同q的前向：先记录当前q是否最佳，再用同一次前向反传；额外保留第80次更新后的最终检查。Adam步数、损失mean归一化、限位截断与最佳解选择规则不变。

本地原始Torch运动学/优化器的CPU对比在测试输入上完全一致。但结构兼容不等于全量数值逐位一致：不同CUDA/GPU或计算图可能有浮点差异。每次正式作业内，每张卡先在同设备、相同真实输入下对比旧/新实现：两个固定grid点、全部八sensor，使用真实20000/500/80配置。
门槛为接受候选索引、离散诊断完全一致，q差<=1e-5rad，FOV margin差<=2e-6m，梯度差<=5e-6。失败立即报错，不自动放宽、不切回另一算法继续采。
此有限样本门槛不是全网格逐位等价证明；每个保留Q0仍由原FOV独立重验并检查原epsilon。

原始已完成分片原样保留，未被覆盖。新分片complete.json附 producer：加速源码哈希、GPU、Torch/CUDA版本及通过的数值检查文件/哈希。新旧来源明确记录，不伪称全部由旧实现产生。最终provenance独立保存，NPZ训练数组结构不变。

## 分段与资源

2707个原始64点分片按固定随机排列分成43段；前42段每段64分片(最多4096点)，最后一段19分片(1193点)。保留原点ID随机种子，分段和GPU类型不影响抽取的随机q。
**一次只提交一个单节点四卡作业，且只处理一段。**已有完成片自动跳过；部分段优先补完。没有job array、依赖链或自动提交下一段。43是逻辑段数；如已有片、软暂停或超时，实际提交次数可能不同。
每次申请24h；20h后不再领取新片，完成当前片后正常退出。Slurm临近时限USR1同样请求安全停在分片边界；仍不能保证极慢分片不触发硬时限。强制取消只保留已原子发布的片，最多四个未完成片需重算。

NODE支持一个3090node1..3或4090node1..3，推导对应 `--gres=gpu:3090:4` / `gpu:4090:4`。不是由登录机器名称推断实际计算GPU。账户空闲检查及test-only通过后只真实sbatch一次；资源权限仍由Slurm决定。不给4090承诺固定加速倍数，也不要求第二个并行作业。

## 从当前旧任务切换

先拉取本次Git提交、运行本目录test。旧18768仍在运行时，next会拒绝提交第二个作业。需要用户终止旧任务或等它退出后再next；代码绝不自行scancel，也不删除旧分片。
旧任务没有新的分段暂停能力：主动取消可能损失最多四个正在计算但尚未保存的分片。已经完成的分片全部复用。

    export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
    export GRID25_ROOT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"
    bash experiments/hierarchical9_grid25_fast_segments_v1/run.sh test

旧任务退出且队列为空后，选择一个节点：

    NODE=4090node2 bash experiments/hierarchical9_grid25_fast_segments_v1/run.sh next

或 `NODE=3090node3`。默认3090node3，不擅自迁移。每段结束后仍用同一条next命令补完当前段/开始下一段；resume与next语义相同。每段允许换GPU，设备与资格检查都会重新记录；不声称跨3090/4090逐位一致。

    bash experiments/hierarchical9_grid25_fast_segments_v1/run.sh status
    bash experiments/hierarchical9_grid25_fast_segments_v1/run.sh log
    bash experiments/hierarchical9_grid25_fast_segments_v1/run.sh ranks
    bash experiments/hierarchical9_grid25_fast_segments_v1/run.sh summary

开始日志先出现[gate]，随后[PARITY PASS]、[fast shard=...]。[PARITY PASS]中的speedup只计算边界优化，不是全采集倍数。summary根据已完成的新分片计算真实点吞吐/剩余采集外推，不使用旧27.5天除以理论倍数。

每段结束显示SEGMENT_COMPLETE；软停止显示SEGMENT_PAUSED。两者都不等于整批完成。父程序会校验本段已发布数据和数值检查来源，然后释放节点，不合并不完整网格。

## 全部采齐后

    NODE=4090node2 bash experiments/hierarchical9_grid25_fast_segments_v1/run.sh merge

merge是用户单独提交的一个作业，只有2707片齐全才允许；沿用原流式NPZ合并、CRC及哈希检查，不在登录节点全量合并。合并期间也按该入口申请单节点四卡，但计算主要是CPU/磁盘；不自动启动训练。

    bash experiments/hierarchical9_grid25_fast_segments_v1/run.sh verify
    bash experiments/hierarchical9_grid25_fast_segments_v1/run.sh pack

产物仍是 `formal/dataset/visibility_grid25_fovonly.npz`。分段记录在 `fast_segments_v1/`，新源码快照在`fast_sources/`；旧source.env和job.json不改。
pack不包含NPZ/权重。基础Q0库完成后仍需重建新网格V4配对和训练缓存，不是完整训练已经就绪。
