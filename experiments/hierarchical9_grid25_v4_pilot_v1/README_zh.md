# Grid25 V4 Label Pilot v1（独立的训练前小样本实验）

本实验**不训练模型、不生成完整的 Grid25 V4 标签、不修改原 q0 bank/NPZ、旧 R1/R2 checkpoint、URDF 或运行时逻辑**。

## 目的

复用旧 `hierarchical9_full_dataset_v4_v1` 已验证的算法模块，只替换数据入口与小样本索引：

- Grid25 native train/val spatial splits：`$GRID25_ROOT/derived/grid25_trainprep_v1/splits.npz`
- 只读 mmap bank：`$GRID25_ROOT/derived/grid25_trainprep_v1/bank_cache/`，已由 job 19806 构建并核验
- 旧 fresh holdout 118 个坐标已经通过 75mm 邻域排除，不能把旧 x_index 当新网格 x_index
- 上游原版 `global_pool.py`、`v3_labels.py`、`v4_tubes.py` 的 worker 函数保持原样，由本目录 `pilot.py` 适配 `open_run`
- 使用 **完全一致** 的上游 `prod_common.select_v3_indices`、`core.solve_attempts`、`core.screen_candidates`、`v3.reference_gate`、`prod_common.refine_anchor`
- 同上游基础 V4 默认科学预算：128 uniform q/x；最多 4 个选中 q/x；每个有效(x,s)最多 32 个 FPS anchors；V4 normal tube 0、±0.005、±0.01、±0.02 rad
- Pilot 按 `x_shard_size=1` 分片，便于单独检查和中断恢复；**没有改动每个 x 的任何标注策略**
- 缺失的 q0 仍是 UNRESOLVED，不代表传感器几何不可见
- 必须通过 `RepoOracle.verify` 原几何/旧标签/FD parity gate；记录产出 SHA、每个 stage 统计与失败原因
- V3 `gradient_candidate_valid` **不是**独立 FD-PASS 的保证；不可以把 Pilot 原始候选梯度直接当成已认证训练梯度

## 1. CPU 测试

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
git fetch origin mainline-b/h9-scratch50k-r012-v1
git merge --ff-only origin/mainline-b/h9-scratch50k-r012-v1
export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
export GRID25_ROOT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"
SCRIPT=experiments/hierarchical9_grid25_v4_pilot_v1/run.sh
bash "$SCRIPT" test
```

## 2. 准备小规模 Pilot（仅 CPU，不解包任何大文件）

```bash
bash "$SCRIPT" prepare --root "$GRID25_ROOT"
bash "$SCRIPT" status  --root "$GRID25_ROOT"
```

默认固定输出为：

```text
$GRID25_ROOT/derived/grid25_v4_pilot_v1/pilot_train2_val2/
  manifest.json
  x_plan.npz
```

- 默认从 Grid25 native train、val 各选择2个位置；其中先选8传感器均支持的位置，再尽可能选含 UNRESOLVED 的位置以测试缺失分支；
- 使用确定性 seed=260927（训练流无关）；
- 不覆盖已经存在的 output；如需新 pilot，用 `--out` 指定全新目录；
- 这些4个点的结果只能说明流程跑通，不构成全网格统计学性能保证。

## 3. 提交 CPU-only Pilot 单作业

**只有上面的测试和准备都成功后**：

```bash
PILOT_OUT="$GRID25_ROOT/derived/grid25_v4_pilot_v1/pilot_train2_val2" \
PILOT_WORKERS=4 \
bash experiments/hierarchical9_grid25_v4_pilot_v1/submit.sh
```

提交器默认申请 **1个 GPU 分区节点、4 CPU、24 小时时限，0 GPU**，不显式指定 `--mem`，使用 Slurm 分区默认内存配额。该配置与已成功运行的 Grid25 mmap 缓存作业（Job 19806）的核心资源请求一致。保持 `PILOT_WORKERS=4`，即每个 CPU worker 使用一个请求的 CPU；如果未来调整可设置 `PILOT_CPUS`（1–8，必须大于等于 PILOT_WORKERS），但不要绕过集群资源规则。此前 `--cpus-per-task=8 --mem=32G` 在本集群 `--test-only` 提示 Requested node configuration is not available；**这不是普通排队**，可能是显式内存/CPU请求的资源组合不被接受，无法仅凭报错确认是哪一项。现在去掉未证实可用的资源要求，再测试。前提是集群允许 CPU-only GPU 分区作业（之前 g25_cache 的成功提交表明可行）。先执行 `sbatch --test-only`，再实际 `sbatch --parsable` 一次。若自己已有活跃队列作业、squeue 查询失败、或 --test-only 被拒绝，则 fail closed；不会偷偷提交依赖、自动重试或排多个作业。

作业先调用 `RepoOracle.verify` 做 CPU 几何 / Jacobian / old-target parity gate，然后依次：

1. `global_pool`：128 uniform q/x → 8传感器解析 FOV sign；按既有 original V4 greedy 覆盖策略选4个 q/x。
2. `v3_labels`：4选中查询 × 已有q0支持的sensor；沿用原先6-face SLSQP和原始 seeds、screen/reference gate；输出新旧值比较与数值诊断。
3. `v4_tubes`：最多32个每传感器的已有q0 anchors → strict zero refinement → regular same-face normal offsets。同步产出 anchors/tubes 和未通过理由。

四个进程共享只读 mmap bank，每个 stage 只写自己的原子分片。没有调用旧 `prepare.py`，没有旧索引泄漏。

## 4. 查看和验收

```bash
# 替换成实际打印出来的提交作业号
JOBID=12345
squeue -j "$JOBID"
sacct -X -j "$JOBID" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 100 "$GRID25_ROOT/derived/grid25_v4_pilot_v1/pilot_train2_val2/logs/slurm_${JOBID}.out"
tail -n 100 "$GRID25_ROOT/derived/grid25_v4_pilot_v1/pilot_train2_val2/logs/slurm_${JOBID}.err"

bash "$SCRIPT" verify --root "$GRID25_ROOT"
```

Pilot 的输出包含：

```text
geometry_preflight.json             # 原始几何/输入梯度/旧标签 parity
global_pool/shard_000*.npz/json    # FOV sign/query choice
v3_labels/shard_000*.npz/json      # 旧/新value，SLSQP诊断，候选梯度
v4_anchors/shard_000*.npz/json     # refined q*、regular/tube条件
v4_tubes/shard_000*.npz/json       # normal±tubes
global_pool_summary.json
v3_summary.json
v4_summary.json
pilot_report.json
```

`[PILOT COMPLETE]` 仅指小样本算完、分片hash与结构通过；**不等同于V3梯度的FD验证已通过，不等于完整V4数据训练ready，不等于H9的性能提升**。详细看 `v3_summary.json` 的有效标注比例、reason、失败理由与 `v4_summary.json` 的 refined/regular/tube 数量，再决定是否生成对应的 Grid25 V4 正式标签协议。

## 5. 作业失败或超时怎么办

不要删除 `formal/` 或 `bank_cache/`；不要盲目重跑采集。Pilot 按 shard 原子发布，大部分完成分片可手动续跑；如果出现 `v4_anchors` 已发布而 `v4_tubes` 未发布的极端半完成状态，原上游 worker 会拒绝覆盖，必须检查具体情况后针对新输出设计安全恢复处理，不能直接删除原始样本。

先检查 `sacct` / `slurm_job.err`，确认旧作业已退出，避免同时提交两个 pilot。只有在已停止且故障可恢复的前提下才手动再提交同一 `PILOT_OUT`。

没有自动训练或跟踪任务。
