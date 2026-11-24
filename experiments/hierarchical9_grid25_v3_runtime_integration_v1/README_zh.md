# Grid25 全量 V3 H9 → CAREPlanner 运行时接入（受控诊断，不替换默认规划）

本阶段**不再训练**。输入是已完成全124段V3-only正式训练的 `best.pt`（step=19500，验证RMSE≈0.097157），不是原50k步R1 checkpoint。本模块只验证新模型可以通过原有的`per_sensor_visibility_runtime.py`调用并执行候选搜索；不会自动打开ROS/Gazebo、不修改默认per-sensor hybrid开关、不会发布运动命令。

## 代码变化

- `src/care_visibility_cdf/scripts/grid25_v3_h9_runtime_loader.py`：识别正式 `care_grid25_h9_r1_full_124seg_v3_distance_training_v1`，核对同目录 `run.json`、冻结数据 `manifest.json`、checkpoint SHA、step、R1架构与训练权重。**先核对SHA再反序列化**，只接受 `best.pt`。
- `src/care_visibility_cdf/scripts/per_sensor_visibility_runtime.py`：增加仅对上述新格式生效的加载分支。原8-head、旧H9 V1、旧50k R1保持原逻辑。保留九分支模型完整权重，只暴露 `S0..S7` 8个sensor分支，不把union当S0。
- `verify.py`：读现有best.pt，严格对照原训练R1模型与 runtime 的 **9/8输出、S0/S7索引、8个传感器分别对7D q的自动微分梯度**，结果保存 JSON。
- `case026.sh`：复用原始 `scripts/test_phase_e_case026_targeted_per_sensor_fallback.py`，使用相同旧scalar、相同measured seed、相同S4优先和同一FOV/primitive self-occlusion准则执行候选搜索。有旧8-head时比较旧8-head与新H9；旧8-head缺失时只运行新H9并记录 **baseline_available=false、NO BASELINE COMPARISON**，不能声称优于基线。**不自动启用模型生产模式**。

## HPC直接运行

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
git -c http.version=HTTP/1.1 fetch origin mainline-b/h9-scratch50k-r012-v1
git merge --ff-only origin/mainline-b/h9-scratch50k-r012-v1

export GRID25_ROOT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"
export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
NEXT=experiments/hierarchical9_grid25_v3_runtime_integration_v1

# 先运行：只检查加载、值和q梯度，不做规划动作
bash "$NEXT/run.sh" verify --device cpu
```

成功时输出：
```
[GRID25 V3 H9 RUNTIME CHECK PASS]
```
以及 `$GRID25_ROOT/derived/grid25_v3_runtime_integration_v1/runtime_check_cpu.json`。八个sensor梯度是对原网络**自动微分的一致性检查**，不意味着它们已获独立离线FD标签认证。

## Case026对照（离线、无ROS运动）

第一步PASS后，在具有URDF parser等依赖、且能找到以下原始模型文件的环境执行：

```bash
bash "$NEXT/run.sh" case026
```

该命令**必须**有原标量 `src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/final.pt`，才能保持原本的粗投影策略。旧8-head `src/care_visibility_cdf/checkpoints/per_sensor_e2e_fullbatch_seed0/final.pt` 为**可选**：如果 HPC 没有它，就跳过旧8-head，只运行新V3 H9的Case026，并在 `comparison_status.json` 标记 `baseline_available=false`，不作性能优劣结论。如果旧基线文件存储在其他路径，执行前设置 `export OLD8_CHECKPOINT="/实际路径/final.pt"`，原scalar可通过 `SCALAR_CHECKPOINT` 指定。诊断使用 CPU，保留原始scalar粗投影、conservative FOV与primitive LOS。

输出路径：

```
$GRID25_ROOT/derived/grid25_v3_runtime_integration_v1/case026/
  grid25_v3_h9.json
  grid25_v3_h9.log
  comparison_status.json
  old8.json   # 仅在找到旧8-head checkpoint时
  old8.log    # 仅在找到旧8-head checkpoint时
```

记录每支候选、失败原因、rank、耗时、几何认证与选中结果。哪怕两组均是 `NO_CLEAR_SENSOR_BRANCH`，也应如实记录，不代表代码失败或理论上无解。

## 科学与安全边界

- V3 only：只监督 signed distance **值**。没有FD验证的梯度监督；自动微分连续不代表真实FOV边界梯度准确。
- Union 头训练的是 `max` 代理，不是精确全局SDF；该runtime只使用8个sensor头进行ranking/梯度搜索。
- Case026只是单个离线候选检验，检测不等于安全执行；真正的规划还依赖现有GCDF碰撞、VBC、tracker和传感器视线等条件。
- 不改变 `PER_SENSOR_HYBRID_ENABLED` 全局默认false。需要后续明确配置受控实验后再进行小规模仿真对照。不要自动连接真实机器人。
- 不重复运行 `verify` 覆盖已有证明；如果需要不同GPU设备可显式用 `--device cuda` 输出单独的报告。不能在缺GPU的节点上要求CUDA。

## Case026 纯几何可行性搜索（新增，无需旧8-head checkpoint）

目前的完整 Grid25 V3 H9 在Case026中8/8分支未找到候选，但**旧8-head模型本次没有运行**：集群上缺少旧checkpoint；旧scalar模型曾得到 conservative FOV margin +0.018168 的S4候选，但被link3遮挡。因此不能声称旧模型优于新模型。

要区分“神经网络搜索策略失败”与“当前几何约束下没有可见配置”，使用严格**不加载任何神经网络**的独立候选采样：

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
export GRID25_ROOT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"
NEXT=experiments/hierarchical9_grid25_v3_runtime_integration_v1
bash "$NEXT/run.sh" geometry026
```

搜索保持原Case026 target、关节区间、保守FOV (50度/66度，z=.20/.70，delta=.01) 和原始self-filter URDF primitive LOS参数。第一步必须重现已知S4自遮挡作为几何回归检查。候选依次覆盖 measured seed、已知被遮挡旧q_vis、Case026中已有的H9候选（只用其q向量，绝不使用其网络输出打分）、2048个姿态邻域高斯采样、4096个覆盖全关节空间的确定性Sobol采样，以及每个传感器最多4个初始点的纯解析FOV局部优化（SciPy L-BFGS-B）。只有FOV margin非负的候选才会进一步执行完全相同的primitive LOS检查。找到合格候选会提前结束，完整JSON保留所有来源与算法预算：

```
$GRID25_ROOT/derived/grid25_v3_runtime_integration_v1/case026_geometry_only.json
```

输出中的 `FOUND_FOV_AND_PRIMITIVE_LOS_CANDIDATE` 只表明这两个静态几何谓词下存在一个符合条件的 `(q,sensor)`，**不是GCDF/VBC/轨迹执行安全或机器人碰撞认证**。`NOT_FOUND_WITHIN_FINITE_SEARCH_BUDGET` 只表示搜索预算内没有发现，不是数学意义无解。两种结果都不应直接归因于“旧网络更好”。

可用参数控制资源，如 `bash "$NEXT/run.sh" geometry026 --local-samples 4096 --global-samples 8192 --refine-per-sensor 6`。如果上一轮已写出结果，脚本会拒绝覆盖；不要在没有查看原结果的情况下删除它。

