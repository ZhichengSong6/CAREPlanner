# 静态障碍物运行源码

共享源码由 `source_manifest.json` 固定，差异文件位于 `overlay/`。`prepare.py` 建立独立 catkin 工作区；`build_support/` 补齐解析地面库和链接依赖，`runtime_portability.py` 处理启动脚本路径与 Conda 环境名。

## 默认独立演示

在仓库根目录运行：

```bash
bash scripts/run_demo.sh
```

`demo/task.json` 是任务输入配置，只含目标、初始关节角和场景名。世界模型保存在 `template/fixed/scenes/`。`demo_inputs.py` 从这些配置与 URDF 重新生成场景、任务输入和启动几何检查；不读取以前的实验输出。检查覆盖机器人碰撞体、初始地图膨胀、半体素对角线和起始关节偏差。

场景可选 `baseline`、`nonblocking_box`、`detour_box`。默认使用新 CDF/ReLU、统一 R1 VisCDF、`current` 策略。R1 的整体输出负责投影与回退，八个传感器输出负责分支求解和引导，共享一次加载的网络。所有新日志与结果均写入本地输出目录。

## 分步准备和构建

```bash
python3 experiments/phase_e_static_baseline_20261030/prepare.py --demo --output outputs/my_demo --assets-root .
bash experiments/phase_e_static_baseline_20261030/build.sh outputs/my_demo
python3 -B outputs/my_demo/run.py --model new --case 007 --configuration nonblocking_box --tag demo --handoff-retry disabled --view-selection current --preflight
```

完整运行建议使用顶层 `run_demo.sh`，它还会检查推理环境并处理无显示器场景。仅构建可用 `run_demo.sh --build-only`，或在准备命令中添加 `--without-models`。

## 可选历史研究输入

使用旧目标和旧校准输入时，可用 `--inputs-root` 替代 `--demo`。`input_assets.json` 只记录这些外置输入的路径和校验值；内容不在 Git 中。当前源码已统一使用 R1，因此这是在原输入上运行新实现，不是原混合模型的逐字复现。默认演示不依赖这些输入。

`model_assets.json` 将新碰撞 CDF 和 VisCDF R1 标为两份必需资产。默认准备和运行不依赖旧碰撞 CDF，也不加载旧标量 VisCDF；做旧碰撞模型对照时，准备命令添加 `--include-legacy-models`，运行时选择 `--model old`，VisCDF 仍使用 R1。每次准备使用新的输出目录，每次运行使用唯一 tag；源码、选中的权重和运行配置在执行前后核对。原有封存研究不被修改。
