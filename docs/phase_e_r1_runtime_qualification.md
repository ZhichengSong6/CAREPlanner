# Phase-E R1 最终 Runtime Qualification

目标：在不修改 scalar projector、analytic FOV、primitive self-occlusion、Sparse-SCP、VBC、GCDF、tracker、URDF/安全阈值的前提下，只替换 per-sensor H9 branch model，比较：

- V1: `hierarchical9_scratch_seed0/final.pt`
- R1: private-tail scratch-50k champion

固定 checkpoint SHA256：

- V1: `979552db20bc7e20775758b273613532921c5dbf11c480b13597127683c4c199`
- R1: `4f395926fa79c29474be8748cef4733ec400d155cd8fadb76c632c2838864002`
- historical old8 regression: `43f962729adcd17aa114edb9fc410facbbb97ebe7343f0ad3309fe50d273acdb`

## 资格链

### Q0 — Runtime adapter

`scripts/test_phase_e_v1_r1_runtime_adapter.py`

检查：

- V1 H9 full output 的 1:9 与 runtime sensor view 逐值一致；
- R1 训练期 `model.py` 与 runtime private-tail adapter 逐值一致；
- S0/S4/S7 对原始 7D q 的 min-aggregation gradient 一致；
- q 改变后重新计算特征；
- 参数 frozen 仍保留 q gradient；
- historical old8 仍能加载、输出 8 heads、产生有限 q gradient；
- warmup 后 branch value+gradient latency。

### Q1 — Case026 targeted

`scripts/run_phase_e_v1_r1_targeted_qualification.sh`

固定：

- target / measured seed / historical blocked S4 q_vis 使用原 Case026 精确值；
- scalar checkpoint 不变；
- projection=10, damping=.5, epsilon=.03, max step=.25；
- root refine=12, tolerance=.002；
- targeted branch ascent=1；
- force-first-sensor=S4；
- targeted max attempts=8；
- conservative FOV=50/66°, z=.20/.70, delta=.01；
- primitive self-occlusion 不变。

历史 blocked S4 必须保持 conservative-FOV positive + primitive self-occluded。
`NO_CLEAR_SENSOR_BRANCH` 是合法诊断结果；不能为了制造成功放松 FOV/LOS。

### Q2 — Full Case026 runtime pair

`scripts/run_phase_e_v1_r1_runtime_qualification.sh`

顺序运行 V1、R1，同一个 Case026：

- online max branch attempts=4（不把 targeted 的 8 次预算偷渡到 runtime）；
- branch ascent=1；
- `PER_SENSOR_HYBRID_ENABLED=true` 只在该命名 qualification 中启用；
- scalar q_zero、Sparse-SCP、exact VBC、GCDF、tracker、ToF/confidence map 全部保持现有实现；
- 60 s matched window，默认不 early-stop。

评价使用现有 `scripts/evaluate_phase_d_run.py`，Case026 对应 core12 的
`phase_e_case_009 -> source_goal_id=phase_e_goal_026`。

最终报告：

- `runtime_compare.json`
- `runtime_summary.md`
- V1/R1 Phase-D evaluator JSON
- generator logs
- compact safety/acquisition CSV
- targeted raw JSON
- 最终 ZIP

Promotion gate：

1. targeted adapter/geometry valid；
2. R1 `execution_vbc_unsafe_records == 0`；
3. R1 GCDF hard-hold count == 0（本 qualification 使用更严格 gate）；
4. 若有 commit execution，则 GCDF + exact VBC + execution VBC 必须全部 certified；
5. V1 若 task success，则 R1 不得 task-regress；
6. V1 若 acquisition complete，则 R1 不得 acquisition-regress；
7. 若 R1 无任何 commit/execution evidence，结果必须是 `INCONCLUSIVE_NO_R1_EXECUTION_EVIDENCE`，不能宣称 production qualified。

## R1 checkpoint 从训练服务器复制到本机

在本机：

```bash
cd /home/zhicheng/Project/CAREPlanner
mkdir -p src/care_visibility_cdf/checkpoints/hierarchical9_r1_scratch50k

scp \
  zsong142@10.120.17.131:/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_scratch50k_r012_v1/formal/R1/final.pt \
  src/care_visibility_cdf/checkpoints/hierarchical9_r1_scratch50k/final.pt

sha256sum src/care_visibility_cdf/checkpoints/hierarchical9_r1_scratch50k/final.pt
```

必须得到：

```text
4f395926fa79c29474be8748cef4733ec400d155cd8fadb76c632c2838864002
```

## 运行

先保证本机 checkout 包含本 qualification 的代码，并保护本地未提交改动。

只跑 targeted：

```bash
REPO=/home/zhicheng/Project/CAREPlanner \
R1_CHECKPOINT=/home/zhicheng/Project/CAREPlanner/src/care_visibility_cdf/checkpoints/hierarchical9_r1_scratch50k/final.pt \
VIS_PYTHON=$HOME/miniforge3/envs/viscdf/bin/python \
bash scripts/run_phase_e_v1_r1_targeted_qualification.sh
```

完整 final runtime qualification：

```bash
cd /home/zhicheng/Project/CAREPlanner

REPO=$PWD \
R1_CHECKPOINT=$PWD/src/care_visibility_cdf/checkpoints/hierarchical9_r1_scratch50k/final.pt \
VIS_PYTHON=$HOME/miniforge3/envs/viscdf/bin/python \
GAZEBO_GUI=false USE_RVIZ=false RUN_SECONDS=60 \
bash scripts/run_phase_e_v1_r1_runtime_qualification.sh
```

脚本不会启动真实机器人运动；它复用现有 Gazebo/Phase-E runner。

最终看：

```bash
cat outputs/phase_e_r1_runtime_qualification/<STAMP>/runtime_summary.md
cat outputs/phase_e_r1_runtime_qualification/<STAMP>/runtime_compare.json
```

以及根目录生成的：

```text
CAREPlanner_PHASE_E_V1_R1_RUNTIME_QUAL_<STAMP>.zip
```

把 ZIP 发回分析即可。
