# 当前发布运行源码

目录名保留 2026-10-30 的入口兼容性，当前内容已晋升为 **2026-11-06 near-prefix closed-loop candidate**。`source_manifest.json` 固定每个文件的来源与 SHA256；`overlay/` 保存对根工程的冻结覆盖。`prepare.py` 生成独立 catkin 工作区，不读取以前的实验结果。

推荐从仓库根目录运行 `bash scripts/run_demo.sh`。该入口固定 R1、CPU/MKL 2 threads、GPU 碰撞、active observation、关节位置命令包络和 UNKNOWN enforce。`demo/runtime.json` 固定：

| 参数 | 当前值 |
|---|---|
| policy | prefix_objective |
| handoff-retry | disabled |
| repair-goal-stopping | on |
| normal-goal-prefix | off |
| view-selection | current |
| observation-release | matched |
| observation-guidance | unseen |
| run-seconds-wall | 60 |

准备后 `run.py` 仍提供研究开关；直接调用时必须传入这些参数及相同环境。它的通用默认值不等于推荐发布配置。未晋升的历史策略没有被默认为开启。

`demo/task.json` 提供目标 007、初始关节角和场景名。`demo_inputs.py` 根据配置、URDF、碰撞几何和世界文件重新生成任务与启动检查；无须本地历史输入。默认 `detour_box`，另可选择 `baseline`、`nonblocking_box`。

初始已知区域采用 `src/care_confidence_map/config/confidence_map_static_box_ray.yaml` 的离散先验及实际机器人几何规则；它不是包围盒内全部空间的自由认证。物理地面保留，规划中同时使用解析平面。

近端观察划分使用精确的 `(trajectory_stamp_ns, bundle_seq)` 对齐 VBC 证据。在同一个空间区域中分开最早原轨迹区间与后续点，保留每个点和已有所有者。证据不完整时使用原区域。实际完成仍依靠 ToF/confidence 更新，运动仍须原审核通过。

终端修复保留同目标重规划时正在审核/接管的候选身份，避免清空尚未完成的停稳交接；目标变化、VBC/GCDF 停止、跟踪中止等仍撤销。关节命令包络是运动学限速机制，不是未测执行器动力学的严格误差界。

`release_provenance.json` 记录冻结候选源清单及证据索引的哈希、开发验证分母和发布路径适配。封存实验未改写。当前发布不声称新的完整 30+18 成功率。

模型仍为 `cdf_25mm_v2/model.pt` 与 `hierarchical9_r1_scratch50k/final.pt`，整体输出和八个 sensor head 共用一个 R1 实例。25 mm 是碰撞模型空间采样间距，输出仍是关节配置空间距离；VisCDF 的解析 FOV 距离不是环境感知的无碰撞代价，标签也不是连续边界的数学精确距离。

## 维护与检查

```bash
python3 -B scripts/check_release.py
bash scripts/run_demo.sh --build-only
bash scripts/run_demo.sh --env viscdf --visibility-env ncdf_l4c --preflight-only
```

每次使用新输出目录。准备阶段校验所有源码和模型；构建集成已包含在候选 CMake 中，`build_support/` 仅补齐缺失集成，避免重复声明库。

旧研究的 `--inputs-root` 入口与旧碰撞模型开关仍保留；其原始输入保持外置，仅在显式提供时使用。默认 `--demo` 只从版本化配置生成输入。发布不包含历史实验输出，也不需要它们。
