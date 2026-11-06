# CAREPlanner

七自由度机械臂的 egocentric ToF 可见性约束运动规划：link-mounted ToF 建图、碰撞 CDF、统一 R1 VisCDF、局部 SCP/QP 与执行审核组成观察—运动闭环。

当前发布为 **2026-11-06 near-prefix closed-loop candidate**。默认入口保留 UNKNOWN 约束：实际 ToF/confidence 更新确认观察后才允许进入对应区域。到达观察姿态本身不等于已观察。

## 获取与安装

支持 Ubuntu 20.04、ROS Noetic、Gazebo Classic、Linux x86_64、NVIDIA GPU/CUDA 驱动和 Conda。先安装 ROS Noetic 与 Conda，再执行：

```bash
git clone --branch research/phase-e-static-baseline-20261030 --recurse-submodules https://github.com/ZhichengSong6/CAREPlanner.git
cd CAREPlanner
bash scripts/setup_runtime.sh
bash scripts/run_demo.sh
```

已有同一分支的检出可更新：

```bash
git pull --ff-only
git submodule update --init --recursive
git lfs pull
bash scripts/setup_runtime.sh
bash scripts/run_demo.sh
```

安装脚本安装系统构建依赖并创建两个 Conda 环境：

| 默认环境名 | 用途 | 固定关键依赖 |
|---|---|---|
| `careplanner` | GPU 碰撞 CDF | PyTorch 2.11.0 / CUDA 12.8 wheel |
| `careplanner_visibility` | CPU R1 VisCDF | PyTorch 2.11.0+cpu、NumPy 2.2.5 / MKL 2025、2 threads |

NumPy 的 BLAS 实现影响本项目的小矩阵几何计算耗时；安装时保留已验收的 MKL 配置。环境名可以改变，模型始终是同一个 R1。`CARE_RUNTIME_ENV`、`CARE_VISIBILITY_ENV` 可覆盖安装环境名；Conda 路径可由 `CONDA_SH` 指定。不修改显卡驱动。

两份必需权重通过 Git LFS 提供，合计约 28 MiB，安装脚本会下载并由运行入口校验 SHA256。见 [模型资产说明](docs/model_assets.txt)。默认演示不依赖任何历史 `outputs/` 文件。

## 运行与输出

默认运行目标 007 / `detour_box`，任务窗口 60 秒，保留物理地面与解析平面、原有离散初始已知区域和严格停稳检查。每次生成独立工作区并从源码编译。

```bash
bash scripts/run_demo.sh --scene baseline
bash scripts/run_demo.sh --scene nonblocking_box
bash scripts/run_demo.sh --preflight-only
bash scripts/run_demo.sh --build-only
```

本机已有环境的等价入口：

```bash
bash scripts/run_demo.sh --env viscdf --visibility-env ncdf_l4c
```

新输出保存到 `outputs/demo_TIMESTAMP/`；可用 `--output PATH` 指定尚不存在的目录。日志、轨迹与审核结果位于该目录，最终结果为 `runs/demo.json`。仿真异步时序会影响结果，固定 `--seed` 不保证逐次轨迹一致。`--build-only` 不需要模型或 GPU；`--preflight-only` 包含编译及环境/配置检查，不启动任务。

请使用不含空格或引号的较短仓库路径。入口将编译临时文件和 CDF socket 放在仓库 `.runtime_tmp/`，退出时清理。无显示器时使用 Xvfb。`outputs/` 建议维持在 100 GB 内；检查结束后可删除自己生成的完整 demo 目录，不要混删原始证据。

`--unknown-vbc shadow` 仅用于历史约束移除诊断，允许未观察空间运动，不能作为满足本方法核心要求的结果。推荐入口默认为 `enforce`。

## 当前已完成的工作

- 整体可见性与八传感器分支统一共享 R1，每进程只加载一次权重。
- 冻结 CPU/MKL 可见性环境和 GPU 碰撞环境；修复观察目标身份、实际观测释放与目标停稳交接。
- 根据精确 VBC 候选身份及原轨迹区间，将同一区域内的近端阻挡点优先作为观察子任务；其余点保持待处理。划分本身不授予运动许可。
- 启用实测关节位置相关的命令限速包络，保留原 GCDF/VBC、地面和执行审核。

最新预先指定的 **10 个开发窗口（全部 60 秒）为 9/10 严格且原始限位合格**：015/nonblocking 2/3，010/baseline 3/3，002/005/006/baseline 与 007/detour 各 1/1。6,643 条执行审核未记录 UNKNOWN 违规或接触；14,940 个采样机体状态的本体及 10 mm 包络检查无 UNKNOWN。观察子任务实际增益后接任务推进的精确关联有 9 次。

这是六个开发任务上的重复验证，不是完整 30 个空场景或 18 个静态障碍物的总体成功率，也不是连续时间安全证明。剩余失败为 015 的一个 60 秒窗口，超时时已返回 NORMAL 并继续执行，末端尚距目标约 10.8 cm。当前版本尚未重跑完整 30+18，也未做实机验证。

## 发布边界

Git 保存源码、必要配置、安装/构建脚本、URDF/STL、场景和校验清单；Git LFS 保存两份运行权重。编译产物、训练数据、原始采集、实验结果及本地研究工作目录不上传。

推荐入口由 [运行源码说明](experiments/phase_e_static_baseline_20261030/README.md)、`source_manifest.json` 和 `overlay/` 确定；根目录 `src/` 的运行源码已同步。后续修改须同步清单与对应 overlay，单改 `src/` 可能仍加载冻结版本。发布来源摘要见 `release_provenance.json`。
