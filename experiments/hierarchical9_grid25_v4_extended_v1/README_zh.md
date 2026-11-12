# Grid25 V4 Extended Pilot + Gradient Audit v1

本目录**不修改**已经通过的 2+2 Pilot、Grid25 25mm 网格、109490/16711 train/val split、19GB mmap q0 bank、旧 V3/V4 SLSQP/analytic geometry 实现，也不修改模型或开始训练。

两个独立单作业阶段：
1. **Extended Pilot 32+32**：使用原 `hierarchical9_grid25_v4_pilot_v1/pilot.py` 和已验证的 original `global_pool.py`、`v3_labels.py`、`v4_tubes.py`。仍然 128 uniform q/x、4 V3 q/x 和最多32 anchor/(x,s)、原有 SLSQP / screening / reference gate / 0±(.005,.01,.02)rad tubes。没有偷改 sampler 或 solver 预算。
2. **独立 FD Gradient Audit**：固定 `split × sensor × 正/负 FOV` 共32个候选层，每层根据**原始 global_pool 源查询**确定性固定最多1条；虽然是在 Pilot 完成后冻结名单，但选择算法不使用 V3 成功/失败、梯度或 V4 结果；缺的 strata 标记为 missing。无效原 V3 标签不换其他样本。有效原标签在 FD 前使用原V3求解器重新算中心并对账，再通过 `verified_core.verify_gradient` 原版两尺度、两侧和单侧斜率检查（各活动关节+混合方向、扰动后重新SLSQP、重用原 SLSQP/fail-close 设定）。保存 PASS/HOLD/SKIPPED、probe数和误差。绝不把未审计或候选梯度自动当成 FD-PASS 训练目标。

## 1. 从原 GitHub 分支拉代码并运行 CPU 测试

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
git fetch origin mainline-b/h9-scratch50k-r012-v1
git merge --ff-only origin/mainline-b/h9-scratch50k-r012-v1
export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
export GRID25_ROOT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"
EXT_SCRIPT=experiments/hierarchical9_grid25_v4_extended_v1/run.sh
bash "$EXT_SCRIPT" test
```

先确认新测试通过（真实集群测试尚未在开发者环境运行）。

## 2. 准备独立 32+32 Pilot

```bash
bash "$EXT_SCRIPT" prepare
bash "$EXT_SCRIPT" status
```

输出路径与已完成的4点 Pilot完全不同：

```text
$GRID25_ROOT/derived/grid25_v4_extended_v1/pilot_train32_val32/
  manifest.json
  x_plan.npz
```

`prepare` 只选新的 Grid25 x-index，用现有 train/val 及75mm fresh/val guard，结果不覆盖。扩展样本数从 `counts.train_x/val_x` 判定（应为32/32）；复用原 Pilot prepare，旧 `scope` 文案可能仍写“2+2”，它不影响 manifest 里实际的 x 数、哈希或边界算法，不能把这段原文字用于汇报本次规模。

## 3. 只提交一次 Extended CPU Pilot

```bash
bash "$EXT_SCRIPT" submit
```

该脚本转发到原已经通过集群调度验证的单作业提交器，配置为 GPU 分区、**不申请GPU**、1节点4CPU、分区默认内存和24小时walltime，内部4个CPU工作进程。会先 sbatch --test-only，自己已有 Slurm 活动任务时停止，只有实际 sbatch 返回JobID才算提交成功。不自动提交其他阶段。先看 `squeue -u "$USER"` 和实际 `sacct`。此任务规模大于4点pilot，不要根据此前45秒墙钟时间推断其用时。

```bash
JOBID=实际提交的JOBID
squeue -j "$JOBID"
sacct -X -j "$JOBID" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 100 "$GRID25_ROOT/derived/grid25_v4_extended_v1/pilot_train32_val32/logs/slurm_${JOBID}.out"
tail -n 100 "$GRID25_ROOT/derived/grid25_v4_extended_v1/pilot_train32_val32/logs/slurm_${JOBID}.err"
```

作业完成后：

```bash
bash "$EXT_SCRIPT" verify
cat "$GRID25_ROOT/derived/grid25_v4_extended_v1/pilot_train32_val32/v3_summary.json"
cat "$GRID25_ROOT/derived/grid25_v4_extended_v1/pilot_train32_val32/v4_summary.json"
```

确认 `COMPLETED 0:0`、`PILOT VERIFIED`、`EXTENDED PILOT COMPLETE` 再进入FD。

## 4. 固定审计清单（只读，不需要GPU）

```bash
bash "$EXT_SCRIPT" fd-plan
bash "$EXT_SCRIPT" fd-status
```

生成 `$EXT_OUT/extended_fd_audit_v1/fd_plan.json` 和 `audit_manifest.json`。
无 `x_index` 复用；审计计划每个 `(split,sensor,FOV sign)` 最多1条。若某个sensor没有正侧已选查询，则记录缺失层，不偷偷从valid的负侧拿来替补。

## 5. 单独提交 FD 审计任务

```bash
bash experiments/hierarchical9_grid25_v4_extended_v1/submit_audit.sh
```

与 Pilot 相同的单作业CPU-only调度策略、4CPU、单节点、24小时。先 `--test-only`，再真实提交一次，不在登录节点做大量重复SLSQP。

结束后：

```bash
bash "$EXT_SCRIPT" fd-verify
cat "$GRID25_ROOT/derived/grid25_v4_extended_v1/pilot_train32_val32/extended_fd_audit_v1/audit_summary.json"
```

`audit_summary.json` 包含：
- 每个审计固定层是否存在、FD PASS/HOLD/SKIPPED 的**固定样本分母**，不选择性丢弃失败
- 每个sensor的FOV内/外查询数量，按train/val拆分，且`selected_v3_signs`说明包含无Q0支持的head，是全局FOV分布统计，**不等于所有head都生成V3标签**
- S2所有V3无效查询的 `reason`、attempt/success 及部分示例；V4 S2 的 `LINE_SEARCH_STALLED`、`MAXITER`、边界误差、迭代/几何调用示例
- 任何 FD PASS 仅适用于**被抽中的独立查询局部数值梯度**，不是对所有V3标签的认证，也不是新的global最近距离证明

## 明确的边界

- 原 `training_ready=false` 仍然保留，实验绝不写回正式dataset/源trainprep/旧2+2 pilot。
- FD审计是离线诊断，不构造全量梯度训练缓存，也不自动申请第二次标签作业。
- S2问题不单纯按失败重跑或换 seeds；先记录数据，分析是否是空间/几何条件导致。
- 仅在形成可审计的 full Grid25 V4 生产设计、完整 FD mask、以及训练接口数值检查后，才能进入真正的H9训练。
