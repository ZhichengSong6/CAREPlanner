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

搜索保持原Case026 target、关节区间、保守FOV (50度/66度，z=.20/.70，delta=.01) 和原始self-filter URDF primitive LOS参数。第一步必须重现已知S4自遮挡作为几何回归检查。候选依次覆盖 measured seed、已知被遮挡的旧scalar q_vis、2048个姿态邻域高斯采样、4096个覆盖全关节空间的确定性Sobol采样，以及每个传感器最多4个初始点的纯解析FOV局部优化（SciPy L-BFGS-B）。**不会读取H9生成的候选，也不运行任何网络；检验与新旧网络预测完全独立。**只有FOV margin非负的候选才会进一步执行完全相同的primitive LOS检查。找到合格候选会提前结束，完整JSON保留所有来源与算法预算：

```
$GRID25_ROOT/derived/grid25_v3_runtime_integration_v1/case026_geometry_only.json
```

输出中的 `FOUND_FOV_AND_PRIMITIVE_LOS_CANDIDATE` 只表明这两个静态几何谓词下存在一个符合条件的 `(q,sensor)`，**不是GCDF/VBC/轨迹执行安全或机器人碰撞认证**。`NOT_FOUND_WITHIN_FINITE_SEARCH_BUDGET` 只表示搜索预算内没有发现，不是数学意义无解。两种结果都不应直接归因于“旧网络更好”。

可用参数控制资源，如 `bash "$NEXT/run.sh" geometry026 --local-samples 4096 --global-samples 8192 --refine-per-sensor 6`。如果上一轮已写出结果，脚本会拒绝覆盖；不要在没有查看原结果的情况下删除它。


## 下一步：针对 Case026 S4 的唯一修复实验（`repair026`）

全量V3-only模型的 RMSE=0.097157，不能和旧网络不同数据集的 RMSE 严格比较。Case026 H9 在8个传感器上未找到通过保守 FOV+primitive LOS 的候选，但单独的几何 oracle 在第62个配置找到 S4 有效候选。这**不意味着**细化标注让网络预测变差：原训练只优化 V3 距离值的 masked MSE（没有独立FD梯度监督，没有LOS/self-occlusion目标），而运行时按传感器链掩码沿FOV梯度找边界。已知原S4候选被 `link3` 遮挡，FOV边界与无遮挡模式不一致。

只做一个定位实验，避免重新训练或没有意义的批量测试：

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
git -c http.version=HTTP/1.1 fetch origin mainline-b/h9-scratch50k-r012-v1
git merge --ff-only origin/mainline-b/h9-scratch50k-r012-v1
export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
export GRID25_ROOT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"
NEXT=experiments/hierarchical9_grid25_v3_runtime_integration_v1

bash "$NEXT/run.sh" repair026
```

该命令读取**已存在**的H9 Case026失败JSON、纯几何成功JSON、同SHA的正式`best.pt`；检查H9在原候选和纯几何可见解处的S4输出以及原FOV/LOS结果。几何成功解仅用于**诊断/后验对照**，不会作为搜索起点。修复搜索从原H9 S4 `q_candidate`/`q_zero`、已知被遮挡的旧scalar候选和measured seed开始，在S4运动学祖先关节（由URDF自动识别）上进行256个局部采样和1024个Sobol采样。剩余关节在全局采样中固定为measured q。判定仍然采用同一个原始解析FOV、相同的self-filter URDF、相同primitive LOS，**不改变安全阈值、模型参数或在线默认开关**。

结果位于 `$GRID25_ROOT/derived/grid25_v3_runtime_integration_v1/case026_s4_repair.json`。如果发现候选：仅说明多起点+几何筛选可补救本次候选生成，并非证明“新网络性能提升”或轨迹可执行。如果没有找到：只是该预算下没找到，不是无解证明。最有诊断价值的字段是 `observations.independent_geometry_witness.h9_s4_predicted_value`：对几何真实可见点，模型自己是否打正分。应结合训练FOV定义与本次保守FOV差异解释，不能将它直接当成同一真实标签。

任何报告都不声称旧8-head成功，因为其checkpoint没有在本次HPC上运行。单个静态候选通过FOV/primitive LOS不等于GCDF/VBC/轨迹安全认证。

## FOV-only 严格对照：第一轮 vs 第二轮 H9（新增，下一步唯一实验）

用户实测同一个冻结Val、同一|V3|≤0.10指标：
- 第一轮 global RMSE 0.09715688490905942；near-boundary RMSE 0.058270900400498、sign 0.8119720260351752
- 第二轮 global RMSE 0.09807170521342248；near-boundary RMSE 0.04795601933348324、sign 0.8433042514887135
- 近边界 RMSE 相对降低17.7%，sign提升3.13个百分点，global RMSE相对升0.94%。

这证明在该验证集上近边界监督效果改善，不证明Case026搜索已经改善，更不证明自遮挡由网络处理。本实验**完全不执行LOS、自遮挡判定**，严格沿用先前的Case026目标、旧scalar保存的q_zero、measured branch seed、8分支排序（S4强制首位）、projection/root/ascent参数以及保守FOV几何阈值；只替换两个已完成的最佳 H9 模型。两者都按原代码独立运行八分支，然后用 conservative FOV margin≥0 作为唯一通过条件。这里不增加新训练或多起点搜索。

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
git -c http.version=HTTP/1.1 fetch origin mainline-b/h9-scratch50k-r012-v1
git merge --ff-only origin/mainline-b/h9-scratch50k-r012-v1
export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
export GRID25_ROOT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"
NEXT=experiments/hierarchical9_grid25_v3_runtime_integration_v1
bash "$NEXT/run.sh" fovpair026
```

输出 `$GRID25_ROOT/derived/grid25_v3_runtime_integration_v1/case026_fov_only_pair_v1.json`；控制台两条 `[CASE026 FOV-ONLY]` 可直观查看各自八分支有多少通过解析FOV、S4 margin、各自首次通过的sensor。只按FOV判定，不做安全/LOS结论。任何一方失败只说明本Case、当前预算下的FOV候选生成结果。

## 决定是否启动第三轮训练的唯一下一步：固定64个Val x的FOV-only配对搜索

用户已经实测第一/第二轮 H9 近边界RMSE **0.0582709 → 0.0479560**（改善17.7%），近边界sign **81.20% → 84.33%**，整体RMSE **0.097157 → 0.098072**。Case026 FOV-only第一轮0/8、第二轮1/8（S7成功），但只是单个场景。

下面的评估**不训练、不计算LOS/自遮挡/碰撞、不改变网络**，固定从旧的冻结val split中选64个不同x各一个q槽：seed=260930，且初始q的八个解析 conservative FOV margin 全为负，防止把已经可见的状态算作搜索成功。挑选依据**与两轮网络输出无关**；两模型共享完全相同的64个`(x,q)`、关节限位、每传感器10步projection/12步root/1步ascent/8步fallback、最多8分支。为了直接隔离H9搜索，ranking也在起始q上执行，**不额外运行旧scalar的粗投影**；因此这是H9独立搜索的验证对照，不等于整个CAREPlanner在线成功率。固定Val曾用于checkpoint选择，不是独立最终测试集。

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
git -c http.version=HTTP/1.1 fetch origin mainline-b/h9-scratch50k-r012-v1
git merge --ff-only origin/mainline-b/h9-scratch50k-r012-v1
export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
export GRID25_ROOT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"
NEXT=experiments/hierarchical9_grid25_v3_runtime_integration_v1
bash "$NEXT/run.sh" fovvalpair
```

结果文件：`$GRID25_ROOT/derived/grid25_v3_runtime_integration_v1/fov_val_baseline_vs_boundary_v1.json`。

主要读取`[FOV VAL RESULT]`两行以及`[FOV VAL PAIRED]`，查看是否在更多未参与训练的Val空间位置上提高配置搜索成功率；后续再据此决定是否需要真正的零边界新采样/第三轮训练。没有旧8-head基线对照，本次只有原第一轮 vs 第二轮 H9。
