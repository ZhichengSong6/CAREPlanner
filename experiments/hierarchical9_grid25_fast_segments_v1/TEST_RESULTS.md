# 测试范围

在隔离CPU环境运行本目录unittest；最终29/29项通过，零跳过（本地约12秒）。测试使用仓库原始data_generator.py与extract_visibility_zero_level_sets.py（Git blob校验），不替换其Torch FK、FOV或Adam函数。

本地缺少urdf_parser_py，执行器仅给未使用的URDF导入类提供shim，以构建合成运动学链；shim不在生产代码中。真实服务器测试/worker必须安装真实URDF依赖。

检查八sensor目标值与梯度、候选/索引/全部诊断、完整80步单sensor轨迹结果、全空间点generate/store布局一致性、原R1读取器可读、inactive关节、优化函数恢复、43段2707片覆盖、单sbatch/3090/4090资源构造、失败无重试、忙队列拒绝、冻结来源完整性、旧片保留/新片producer校验和Slurm spool绝对路径。

实际HPC的20000/500/80全量配置、真实URDF、CUDA、Slurm和4090吞吐不在本地测试范围内。正式作业内置同设备旧/新资格对比，未通过不生产新片。CPU相等不表示跨GPU逐位相等；未宣称实际总加速倍数。
