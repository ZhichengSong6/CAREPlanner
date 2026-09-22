# R1 离线重标注 v2：候选修复与固定案例梯度验证

本目录是 v2 修复/验证入口。v1 源码及其旧缓存保持不变，避免破坏旧结果的源码哈希和恢复条件。
只按 `(query_id, sensor_id)` 组织结果，不新增按 FOV 面划分的数据集或网络输出。
六个面仅是同一 sensor 的完整 FOV 约束；不能忽略其他面来制造成功。

## 本轮改了什么

- 数值等价端点聚类：同时限制端点距离和目标距离，采用固定代表而非链式扩张。
  同一个数值解优先采用已成功、驻点检查合格且几何残差较小的代表。
  不以更远的成功解掩盖一个明显更近的不同低置信度解。
- 竞争候选区分：合格的不同近等距解为 ambiguity；未合格的近等距竞争解单独记录 uncertainty。
  两类都保守屏蔽梯度，不把未收敛结果说成真实多解。
- 非零距离不再因多面交会或投影落在限位上直接丢弃梯度。
  使用原7D有效关节距离的位移方向作为 **candidate**，经完整几何、局部 KKT 和
  两尺度、双侧的重新求距离检查后，才设置 `grad_valid=true`。
- 每个有效关节坐标方向，以及两个确定性混合方向，分别用 h 和 h/2 做 ± 扰动。
  每次扰动都重新进行多起点、全部 FOV 面约束的 SLSQP 优化，不能固定 q_star 直接算距离冒充验证。
  同时比较中心差分、左右单侧斜率和尺度稳定性，避免对称差分掩盖尖点。
- 扰动中发现明显更近的原查询边界候选时，将旧中心值标为待复核，不自动替换成一个未经重新验证的答案。
- `gradient_reasons`、active faces、bound joints、每次优化耗时/几何调用数均保留。
  `ZERO_DISTANCE_NONREGULAR_BOUNDARY` 和查询自身限位导致无法双侧验证的情况仍保持屏蔽。

这不是全局最近点证明，也不证明新标签能改善 R1。
有限差分检查只提供局部数值一致性证据；warm starts 和更多局部优化仍可能漏掉其他更近边界区域。
所有几何定义、关节度量与原 sensor masks 不变；诊断容差不是 runtime 安全阈值。

## 复用哪份已有结果

默认读取已经完成的 job 16513：

```
/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_offline_labels_v1/smoke_20260921_202241
```

核对源 queries、split、分片标签、audit、complete、run_spec 校验值，以及 v1 源码哈希。
源原始 NPZ 的 numeric bank cache 必须仍在 manifest 记录的位置；不会重新解压7GB数据。
验证 URDF/关键几何源码与源 smoke 一致，但允许当前 repo HEAD 因本次纯实验代码新增而不同。

默认只复核 6 个明确案例：

| query:sensor | 目的 |
|---|---|
| 4:0 | 真实数值等价候选回归 |
| 1:7 | 同 sensor 多面交会 |
| 5:6 | 投影落在关节限位 |
| 3:2 | 旧的光滑梯度基线 |
| 2:4 | 近等距竞争候选，不应盲目解除 ambiguity |
| 2:3 | 无可行候选，不应伪造标签或宣称不可行 |

中心查询直接复核旧 attempts，不重跑其完整搜索；只为通过初筛的案例补做扰动求解。
因此某些案例输出 HOLD/SKIPPED 是有效诊断，不要求6项全部变成梯度通过。
现有8条查询、全部旧标签、R1和其他checkpoint、训练器、ROS/Gazebo、VBC/GCDF/tracker都不改。

## 本地纯代码测试（服务器也可先运行）

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
"$VIS_PYTHON" -m unittest discover -s experiments/hierarchical9_offline_relabel_v2 -p test_verified.py -v
```

包含合成几何、mask、真实候选的数值等价fixture、源文件不可覆盖、恢复/损坏检测和 mock Slurm 接线测试。
mock 只验证提交/查看命令，不代表实际调度或机器人预检成功。

## 服务器只提交一次验证

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
export REPO="$PWD"
export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
export SOURCE_OUT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_offline_labels_v1/smoke_20260921_202241"
NODE=3090node1 DEVICE=cuda WORKERS=4 \
  bash experiments/hierarchical9_offline_relabel_v2/submit.sh
```

提交前导入依赖；缺 SciPy 等直接停止，不申请GPU、不自动安装升级。
默认4张GPU/16CPU/3小时walltime，仅为申请上限，不是预计耗时。
SLSQP在CPU调度，CUDA只用于FK/Jacobian；本轮不声称GPU比CPU快。
`DEVICE=cpu` 会省去GRES请求，是否允许这种作业由集群策略决定，不自动切换。

提交器保存独立的 `r1_offline_labels_v2/latest_verify.env`，不覆盖旧的 latest_smoke.env。
输出放在独立v2目录，不覆盖source。submit.sh忽略旧终端遗留的OUT；自定义/恢复路径使用VERIFY_OUT。
原任务不用取消，不要为看日志重复提交。

## 只查看输出

```bash
# 主日志实时跟随：主日志每15秒汇集worker当前阶段；Ctrl+C只退出查看。
bash experiments/hierarchical9_offline_relabel_v2/watch.sh

# 单个任务也可用提交器打印的 [state] 路径替代这个latest文件。
STATE="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_offline_labels_v2/latest_verify.env"
bash experiments/hierarchical9_offline_relabel_v2/watch.sh "$STATE" workers
bash experiments/hierarchical9_offline_relabel_v2/watch.sh "$STATE" status
bash experiments/hierarchical9_offline_relabel_v2/watch.sh "$STATE" summary
```

日志包含 `[case]` 和 `[probe] direction/scale/side`。一次非线性扰动求解尚未结束时，方向行可能不变。
watch.sh只读，不含sbatch/scancel。任务未启动时主日志尚未存在是正常情况，tail会等文件出现。

## 输出和验收

- `verification_spec.json`：源身份、固定case、solver/验证设置、代码与库版本。
- `preflight.rank*.json`：每个设备的原几何/旧标签 parity。
- `case_XXXXXX_Ss/result.json`：候选选择、拒绝原因、每个扰动解及数值差分。
- `case_XXXXXX_Ss/complete.json`：逐case完成标记、校验值。
- `verification_summary.md/json`：旧/新有效性和明确HOLD原因。

完成只表示复核流程结束；`training_ready=false`、`global_nearest_certified=false`。
本目录不输出可被误认为完整训练缓存的dataset_index.json。先审查真实报告，再决定是否推广到批量标注。
若新梯度仍被HOLD，查看具体原因；不能直接删gate或调容差来追求通过率。

中断后保留逐case结果。确认原作业结束后，可用相同 SOURCE_OUT、相同 OUT 和 CASES、代码/库版本，
显式 `VERIFY_OUT=原v2输出路径 RESUME=true` 重新提交；已完成case校验后跳过。不同代码/参数不能混入同一输出。
不要 `RESUME=true` 指向旧v1 smoke，也不要删除旧结果重来。

## 文件与方法依据

`verified_core.py` 可用于后续标注器：`label_verified(..., attempts=existing_attempts)` 或不传 attempts 做新中心求解。
此版本优先提供固定真实case验收，不自行改写旧全量pipeline/训练器。
SLSQP字典约束、显式Jacobian和变量bounds仅使用已存在的SciPy接口：
https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.minimize.html
https://docs.scipy.org/doc/scipy-1.15.3/reference/optimize.minimize-slsqp.html
