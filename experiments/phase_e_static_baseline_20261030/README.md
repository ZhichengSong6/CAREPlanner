# 静态障碍物版本源码

`source_manifest.json` 固定运行源码；共享文件复用根工程，差异文件保存在 `overlay/`。本目录只发布源码、构建支持和运行配置，不发布历史成绩、分析报告或记录数据。

## 仅构建源码

```bash
git submodule update --init --recursive
python3 experiments/phase_e_static_baseline_20261030/prepare.py --output outputs/source_build --without-models
bash experiments/phase_e_static_baseline_20261030/build.sh outputs/source_build
```

每次准备一个新的输出目录。构建脚本显式使用其中的独立 catkin 工作区；不会启动仿真。

## 附加本地输入并运行

仿真另需 `model_assets.json` 指定的四份权重，以及 `input_assets.json` 指定的本地输入。后者包括原目标、标定、验证记录和回归夹具，Git 只保存路径及校验值。`--inputs-root` 指向保留这些输入的本地研究目录，文件按清单中的相对路径排列。

在已有本地实验目录的机器上：

```bash
python3 experiments/phase_e_static_baseline_20261030/prepare.py --output outputs/reproduce_20261030 --assets-root . --inputs-root outputs/phase_e_executable_view_selection_r2_20261030
bash experiments/phase_e_static_baseline_20261030/build.sh outputs/reproduce_20261030
python3 -B outputs/reproduce_20261030/run.py --model new --case 007 --configuration detour_box --tag check_007 --handoff-retry disabled --view-selection current --preflight
```

移除 `--preflight` 后运行一个仿真窗口，tag 必须唯一。每个窗口保留原 60 秒任务预算、地图反馈、运动审核和物理接触监测。模型、场景与运行前后源码指纹仍会核对；新结果写入指定输出目录。

缺少外部资产时可构建源码；需提供相应文件后才能运行原实验入口。旧报告和本地模板副本即使仍在工作目录，也不会被准备脚本隐式读取为输入。

`build_support/` 提供解析地面库和显式链接依赖。PIQP 修复头文件固定在差异层中；`preparation.json` 在本地记录重建过程。
