# CASE001 exp92：逐 link 启动 prior 对照

日期：2026-09-23。有效运行 ID：`case_001_r1_motion_prior_20260923_exp92`。
固定仿真窗口 `[6.388,66.388] s`；对照 exp90 为 `[7.134,67.134] s`。
首次 exp91 在仿真前的沙箱 CUDA 预检退出，记录为空；exp92 在可访问 GPU 的环境完成 60 s 窗口。
当前主线入口为 `bash scripts/run_phase_e_case001_r1_mainline.sh`；脚本使用独立
ROS/Gazebo 端口和唯一运行 ID，并显式加载本次验证的 prior 配置。运行前需在本机
准备未纳入 Git 的 collision-CDF 与 R1 visibility 权重；对应路径在脚本中给出。
两份权重的 SHA-256 分别为
`41a4228eee797971d56b46849e0f52c663eae79dc35667e8704c1bc3c8d72c70` 和
`4f395926fa79c29474be8748cef4733ec400d155cd8fadb76c632c2838864002`。

## 改动与实际加载

- exp90 的 `confidence_map.yaml` 保持原样；exp92 单独加载
  `confidence_map_case001_motion_prior.yaml`。其余地图、sensor、规划参数按 YAML
  解析检查与 exp90 一致，逐 link 半径以原统一 0.10 m 为下限：
  base/link1/link2/link3 = 0.10 m，link4 = 0.1421 m，
  wrist_link1/2/3 = 0.1806/0.1812/0.2100 m。
- 这些候选半径来自既有 Phase E 50 ms 运动包络表。该表基于旧 body samples、
  15 mm GCDF inflation 和 25 mm selector band；exp92 用 exact primitives 和
  0 mm GCDF inflation，故此次只检验宽 prior 的效果，不构成新的 primitive
  运动包络证明。为保证一次性启动语义，设 `lock_after_complete_refresh=true`。
- exp92 启动日志确认 8 个逐 link 条目、26 个 primitive、1227 个已标记网格中心，
  完整刷新后锁定。exp90 对应是 0 个逐 link 条目、731 个网格中心。主 VBC
  和执行审计的 `continuous_motion_bound_enabled` 均保持 false；逐步位移上界仅用于
  VBC 中点采样密度，不额外膨胀 swept volume。
- exp90 的 O6 七个高点中心按初始 exact primitive 距离均落在 exp92 的 wrist
  逐 link prior 中。因此它们从启动起作为 bootstrap known-free，不能把此结果
  解读成 S6/S7 后来真实观察了这些点。

## 固定窗口结果

| 指标 | exp90 统一 0.10 m | exp92 逐 link prior |
| --- | ---: | ---: |
| CASE001 目标保持 0.1 s 成功 | 否 | 是，窗口起约 8.23 s |
| 最终末端位置误差 | 0.461981 m | 0.008619 m |
| 最终末端姿态误差 | 1.639502 rad | 0.004775 rad |
| 最多待观察 obligation | 4 | 2 |
| 观察任务清空 | 否 | 10.605 s ROS 时间，seen 2/2 |
| 运行阶段 | REPAIR 停滞 | REPAIR → PROBE_NORMAL → NORMAL |
| 窗口内最终安全提交 | 30 | 96 |
| 窗口内 verification unsafe | 4 | 2（候选拒绝） |
| 窗口内 execution VBC unsafe | 4/2429 | 0/2493 |

exp92 的两次候选 VBC 拒绝仍发现未被 prior 覆盖的 unknown 点
`(-0.20,-0.15,0.85)`，对应 wrist_link3 exact primitive 穿入约
2.56/3.15 mm；这两个候选没有提交，随后恢复流程继续。观察任务在
10.605 s 清空，10.651 s 进入 PROBE_NORMAL，12.651 s 返回 NORMAL；
窗口内测得的目标保持从约 14.615 s 开始，最终仍在位置 20 mm、姿态 0.20 rad
阈值内。执行审计记录均 safe，但当前 VBC motion-bound 膨胀关闭，不能把它写成
连续时间扫掠证明。该实验的主要结论是：**扩大启动可信自由区可使这个空世界
CASE001 成功，同时直接消除了原 O6 unknown blocker**；不能据此证明传感器策略
解决了 O6，也不能把这些半径直接作为其他场景的正式安全参数。

## 验证和证据

- `bash -n`、YAML 结构/exp90 差异检查、7/7 O6 初始点覆盖检查、catkin
  5 包构建、60 s 仿真和 ZIP CRC 检查通过。
- 整理主线后，catkin 全部 7 包构建、R1 CPU 权重等价检查、候选替换 37 项、
  occupied 几何 5 项、连续 primitive 4 个保存反例及 GCDF Case011
  私有 ROS master 回归均通过；两个小型几何输入已纳入 `scripts/fixtures/`。
- 固定窗口目标 FK 与门禁对比：
  `outputs/phase_e_owner_fov_followup/exp92_windowed_goal_comparison.json`、
  `outputs/phase_e_owner_fov_followup/exp92_windowed_safety_comparison.json`。
- 完整记录：`outputs/c5_5_vbc_gcdf_regime/case_001_r1_motion_prior_20260923_exp92/run/`
  与 `CAREPlanner_C5_RESULT_case_001_r1_motion_prior_20260923_exp92.zip`。
  ZIP SHA-256：`45f9ff1228fad36b8da27b91ba5855233a74d1f03e2097ae5a6ba185c3e2e81f`。
- exp92 专用的 ROS/Gazebo 残留进程已终止，11497/11498 端口已释放。

下一步若要保留此策略，应针对当前 exact primitives、速度限制和 50 ms
步长重新推导或严格上界验证逐 link prior，并在包含障碍物的场景检查被标为
bootstrap known-free 的区域；不能仅凭本次任务成功替代这些验证。
