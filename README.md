# CAREPlanner

七自由度机械臂的可见性约束运动规划，结合 ToF 建图、碰撞 CDF、VisCDF、局部 SCP/QP 规划和执行审核。

默认入口运行一个静态箱体仿真。目标与场景由源码配置提供，启动几何检查在本机计算，不需要此前的实验输出、采集记录或分析报告。

## 准备环境

支持环境：Ubuntu 20.04、ROS Noetic、Gazebo Classic、NVIDIA GPU 和可用的 CUDA 驱动，以及 Conda。请将仓库放在不含空格或引号的路径中。

```bash
git clone --branch research/phase-e-static-baseline-20261030 --recurse-submodules git@github.com:ZhichengSong6/CAREPlanner.git
cd CAREPlanner
bash scripts/setup_runtime.sh
```

安装脚本要求已安装 ROS Noetic 和 Conda；会安装构建依赖，并创建 `careplanner` Python 3.10 环境。PyTorch 固定为 2.11.0，使用 [官方 CUDA 12.8 wheel 索引](https://download.pytorch.org/whl/cu128/torch/)。不修改显卡驱动。

运行前需要三份指定推理权重：新碰撞 CDF、VisCDF R1 和仍参与观察位姿求解的标量 VisCDF。旧碰撞 CDF 仅供可选对照，默认演示不需要。路径与校验值见 [模型资产说明](docs/model_assets.txt)。

## 运行

```bash
bash scripts/run_demo.sh
```

入口依次检查环境与权重、准备独立工作区、从源码编译、预检查并运行仿真。无显示器时使用 Xvfb。输出写入新的 `outputs/demo_时间戳/`，包括本次运行生成的日志和审核结果。

可选命令：

```bash
bash scripts/run_demo.sh --scene detour_box
bash scripts/run_demo.sh --preflight-only
bash scripts/run_demo.sh --build-only
```

`--build-only` 不需要权重或 GPU。重复运行会使用新的输出目录；也可以用 `--output` 指定一个尚不存在的目录。

已有项目 Python 环境可以直接使用：

```bash
bash scripts/run_demo.sh --env viscdf --visibility-env ncdf_l4c
```

Conda 不在常见安装位置时，通过 `CONDA_SH` 指定其 `etc/profile.d/conda.sh` 文件。

## 代码与本地产物

Git 保存源码、构建脚本、运行配置、机器人 URDF/STL 和必要说明。编译产物、训练数据、采集记录、分析报告与实验结果不纳入源码提交。

- `src/`：ROS 包及机器人描述；保留历史集成入口。
- `scripts/run_demo.sh`：推荐的当前静态仿真入口。
- `experiments/phase_e_static_baseline_20261030/`：当前运行源码、演示配置与构建支持。
- `patches/piqp/`：固定上游版本的索引修复补丁。

历史研究重建入口和默认演示的区别见 [运行源码说明](experiments/phase_e_static_baseline_20261030/README.md)。本示例不用于复述历史 30-case 实验成绩。
