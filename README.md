# CAREPlanner

七自由度机械臂的可见性约束运动规划研究，结合 ToF 建图、碰撞 CDF、分传感器 VisCDF、局部 SCP/QP 规划和执行审核，使用 ROS Noetic 与 Gazebo。

本仓库发布源码、构建脚本、运行配置和机器人 URDF/STL。模型权重、训练数据、采集记录、分析报告、实验结果及编译产物均在本地单独管理。

## 构建

依赖 ROS Noetic、Gazebo Classic、catkin_tools、Eigen3、Pinocchio/hpp-fcl 和 OpenSSL 开发库。推理使用 `viscdf`、`ncdf_l4c` Python 环境。

```bash
git submodule update --init --recursive
python3 experiments/phase_e_static_baseline_20261030/prepare.py --output outputs/source_build --without-models
bash experiments/phase_e_static_baseline_20261030/build.sh outputs/source_build
```

该入口在独立工作区恢复静态障碍物版本的源码。准备与构建均不启动仿真。根目录历史集成工程的构建入口为 `scripts/build_phase_e_primitive_release.sh`；部分旧启动脚本仍默认使用旧碰撞模型。

## 仿真与外部资产

仿真另需模型权重和本地实验输入。路径及 SHA 清单仅用于匹配外部文件，本仓库不携带这些文件的内容。

- [模型文件与接口](docs/model_assets.txt)
- [静态障碍物源码入口及运行方法](experiments/phase_e_static_baseline_20261030/README.md)

运行使用显式的新 CDF/ReLU、现有 VisCDF R1 与标量粗投影模型，保留物理地面及解析地面处理。原有目标、标定及验证输入通过 `prepare.py --inputs-root` 从本地提供。

## 目录

- `src/`：ROS 包、机器人描述、规划器、建图和推理接口。
- `scripts/`：构建、测试、训练、实验启动及分析程序；程序产生的结果不入库。
- `experiments/`：训练程序和独立运行入口源码。
- `patches/piqp/`：PIQP 有限边界索引修复及版本校验。

PIQP 保持固定的上游 submodule 提交。根工程构建时，`scripts/prepare_piqp.py` 在构建目录生成打补丁后的头文件副本。
