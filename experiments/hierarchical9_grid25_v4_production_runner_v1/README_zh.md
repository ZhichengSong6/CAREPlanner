# Grid25 V4 Production Runner v1 — 手动单段、原版 V3/V4 算法、可恢复

本目录是 `grid25_v4_production_preflight_v1` 之后的**实际标注阶段**。复用原版
`hierarchical9_full_dataset_v4_v1/global_pool.py`、
`v3_labels.py`、`v4_tubes.py` 的 worker / merge，**只适配 Grid25 冻结的
x-plan、train/val split 和只读 mmap bank**。原 SLSQP、几何导数、
V3 screening/reference gate、V4 zero refinement/tubes 不修改。

## 之前已经完成

- 173225 个 Grid25 点的 Q0 bank 采集、哈希、mmap 已完成。
- Train=109490、Val=16711，保留 75mm fresh/val 隔离；旧 `x_index` 不复用。
- 64 点 Extended Pilot：V3 1925/1952 `value_valid`；V4 15343/15616 refined。
- FD 固定32个 split×sensor×符号审计：31 PASS、1 HOLD（S5 Val FOV 外）。
- Frozen production preflight 已生成 `986 shards`、`124 manual segments`。
- 全量预算：3724128 V3 tasks、最多 29784588 V4 anchor、208492116 tube candidates。
- 64 点 Pilot 推算约 852.697 worker-CPU hours；4CPU 理想约213.174小时，2倍保守估计约426.515小时；**仅为不含排队和意外故障的经验预测**。

## 实验约束

- 一个分段最多 8 个原子 shards（每 shard 128 x），最多 1024 x。
- 每次手动只准备和提交一个指定分段；不调用 job arrays、Slurm dependency、自动 next 或自动重试。禁止同时提交其他用户的任务无关；只检查自己队列是否有活动任务。
- 已完成分片文件和 SHA 不覆写，重启仅继续同一分段未完成的 stage/shard。
- 本实验由 4 CPU worker 执行，保留已成功使用的 GPU 分区 CPU-only 节点申请：`--cpus-per-task=4`，**不申请 GPU、不显式指定内存**，Slurm 24h上限。
- 只保存 FOV-only 监督和候选梯度，**V3 的 `gradient_candidate_valid` 不等于独立 FD-PASS**；不生成全数据集的 FD 验证梯度 mask，也不将 `training_ready` 修改为 true。
- 每个 V4 生产 shard 要验证精确的每 (x,sensor) anchor 数量、V3 查询数、global x 来源及哈希；只有 3 个 stage 及分片、合并统计全通过才发布 `segment_complete.json`。
- Segment 0 成功后才能准备 segment 1，后续依次推进。此 Runner **不会自动启动 1..123 段**。

## 在远程服务器执行（现在仅首段）

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
git fetch origin mainline-b/h9-scratch50k-r012-v1
git merge --ff-only origin/mainline-b/h9-scratch50k-r012-v1

export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
export GRID25_ROOT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"
RUNNER=experiments/hierarchical9_grid25_v4_production_runner_v1/run.sh

# 1. CPU synthetic and original-worker reuse tests
bash "$RUNNER" test

# 2. 只准备第0段，不生成监督标签/不申请资源
bash "$RUNNER" prepare --root "$GRID25_ROOT" --segment 0
bash "$RUNNER" status --root "$GRID25_ROOT" --segment 0

# 3. 单次实际提交，内置 --test-only 预检查
SEGMENT_ID=0 bash experiments/hierarchical9_grid25_v4_production_runner_v1/submit.sh
```

`prepare` 若已成功不需要再次执行，否则会拒绝覆盖。**远程测试如不通过，不要提交。**
提交前脚本会检查已经准备的 frozen segment 与自己的 Slurm 队列，并仅提交一个作业。返回实际 `[PRODUCTION SEGMENT SUBMITTED] job=...` 才代表真正已提交。`sbatch --test-only` 的号码不是实际 JobID。

## 查看第0段实际任务进度

```bash
# 换成 sbatch 实际返回的 JobID
JOBID=12345
squeue -j "$JOBID"
sacct -X -j "$JOBID" --format=JobID,State,ExitCode,Elapsed,NodeList

SEGMENT_OUT="$GRID25_ROOT/derived/grid25_v4_production_v1/segment_000"
tail -n 80 "$SEGMENT_OUT/logs/slurm_${JOBID}.out"
tail -n 80 "$SEGMENT_OUT/logs/slurm_${JOBID}.err"
bash "$RUNNER" status --root "$GRID25_ROOT" --segment 0
```

当作业 `COMPLETED 0:0` 后：

```bash
bash "$RUNNER" verify --root "$GRID25_ROOT" --segment 0
cat "$GRID25_ROOT/derived/grid25_v4_production_v1/segment_000/v3_summary.json"
cat "$GRID25_ROOT/derived/grid25_v4_production_v1/segment_000/v4_summary.json"
```

确认以下输出：

```text
[STAGE VERIFIED] global_pool ...
[STAGE VERIFIED] v3_labels ...
[STAGE VERIFIED] v4_tubes ...
[SEGMENT VERIFIED COMPLETE] 0
```

如果失败、超时或只有部分 stage：先查 sacct/logs，确保原job已退出；再以 `SEGMENT_ID=0` **手动**提交同一已准备段，保留成功分片。若发生同一 shard 的 NPZ/JSON 单边发布（损坏或不完整），脚本会拒绝隐式覆盖；不要删原标签或绕过错误，先诊断并决定显式修复方案。

## 首段验收之后

下一段才执行：

```bash
bash "$RUNNER" prepare --root "$GRID25_ROOT" --segment 1
SEGMENT_ID=1 bash experiments/hierarchical9_grid25_v4_production_runner_v1/submit.sh
```

此后分段按 0..123 依次准备、提交和验证，不能跳过先验完成检查。每次一个 Slurm 作业。

**整批124段标签完成之后，还必须单独设计跨段dataset汇总/训练接口、距离有效 mask 和FD独立认证梯度 mask**。本Runner当前仅发布单段完整标记，不训练任何H9网络，也不声称无遮挡/LOS正确。

代码有版本身份和 upstream source SHA 门控。新的代码变更若影响已准备段，不能继续写入该段的生产数据；必须检查原分片与代码指纹，选择新的生产版本或明确的迁移步骤。
