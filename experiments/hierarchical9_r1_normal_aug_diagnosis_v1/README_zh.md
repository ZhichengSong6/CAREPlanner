# R1 vs R1+Normal：1963 个现有 Solver 结果逐例复盘（只读）

本目录是**已完成实验的事后诊断**。输入是原 R1 的 50k checkpoint、受控 R1+0.01 法向增广的 50k checkpoint、已经完成的 1963 frozen starts 固定 solver 比较，以及现有 V4 缓存中的解析边界验证点。

**没有 GPU/Slurm 作业、没有重训、没有重跑 solver、没有重新生成标注。**唯一写入新输出根：outputs/mainline_b/r1_normal_aug_diagnosis_v1。原 checkpoint、缓存、训练指标、solver 评估与运行时代码都不修改。

## 一步运行（CPU-only）

拉取 GitHub 原分支以后，在仓库根目录执行：

    export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
    bash experiments/hierarchical9_r1_normal_aug_diagnosis_v1/run.sh diagnose

默认 diagnose 完成全部三部分。先执行源码语法检查与 CPU 单元测试，然后校验原 R1 和新模型 checkpoint SHA、两轮原始训练 x/q 流匹配、固定 starts SHA 和全部 1963 行 solver 数据与历史报告的一致性。

只分析已经记载的 solver/训练指标、不做真实 checkpoint 的 CPU 输入梯度对比：

    bash experiments/hierarchical9_r1_normal_aug_diagnosis_v1/run.sh fast

查看完整摘要、打包成不含 checkpoint 的小型压缩包（便于上传对话）：

    bash experiments/hierarchical9_r1_normal_aug_diagnosis_v1/run.sh summary
    bash experiments/hierarchical9_r1_normal_aug_diagnosis_v1/run.sh pack

只跑 CPU 测试：

    bash experiments/hierarchical9_r1_normal_aug_diagnosis_v1/run.sh test

## 输出结构与意义

**(1) 成败逐例对齐。**读取全部 1963 行原 solver JSONL，与冻结 starts 的 x/q/sensor/group 原样比对，R1 固定必须复现 local 618/683、uniform 673/1280、all 1291/1963。候选的解析真实 FOV margin 与 fov_pass 必须一致，合并报告中的逐组/paired 结果必须与原始明细相同。

- cases_all.csv：全部 1963 行的原/新增模型候选 margin、网络初始值、预测根来源、实际根位置、距离和失败阶段；
- cases_paired.csv：R1-only 退化与 augment-only 新增（按先前报告预期为 136 和 82；实际仍以一致性核验的 solver 原始行为准）；
- case_audit.json：local/uniform、八 sensor、root source/失败阶段转移、预测接近容差、初始符号误差、真实候选 margin 改善或退化、候选距离变化及分布、典型案例 case IDs。

特别提醒：root_source 是**网络预测根的来源**，不是解析几何根。candidate_g_m 是真实 FOV margin（米）；initial_score、f_zero 是网络预测的标量值，不能与 margin 直接相减。

**(2) 训练目标的真实变化。**metrics_comparison.json 对齐两模型所有共同原 R1 验证 step，比较原始目标 loss、union/S0-S7 的旧距离 RMSE、梯度 cosine/norm、Eikonal、符号正确率。当前增强模型的历史 boundary_direction_diagnostic 还记录每个 sensor 的法向 cosine、梯度模长与首层 ReLU 活跃统计。

旧 R1 历史日志并不含对应 ReLU 诊断，不能从日志直接假造比较值；要进行公平的同点对照见下条。

**(3) 两个冻结模型的同点边界 CPU 验证。**从 V4 val boundary 中筛选同时属于原 R1 val x-index 的锚点，每个 sensor 用固定随机种子独立选取 128 条（1024 总计），在 CPU FP32 上评估原 R1 与增强模型，并保存两者：

- 预测边界输出（注意 V4 零值不等于 R1 原离散全局距离）；
- 七关节输入梯度与解析法向 cosine，近零梯度的 cosine 为 null、单独计数；
- 梯度模长、第一层 sensor-head ReLU 激活率、在这批样本上曾激活的神经元数；
- 网络预测在原固定 solver epsilon_f=0.03 容差内的比例。

输出 boundary_same_points.json / boundary_same_points.csv。两个模型使用**精确相同的 q、x、sensor、normal**，且不会调用 optimizer、改变模型或查询新的几何标签。

## 科学结论界限

这次得到的均为事后**现象关联**，不能直接断言法向损失一定造成某类候选失败、梯度失活或 Adam 更新变化。1963 starts 已反复使用，是回归集而非独立未见确认集。这里只评 FOV-only，不提供 LOS、碰撞、轨迹、GCDF/VBC 或实际观察认证，也不进行阈值/权重搜索和模型晋升。

我们要回答的问题是：136 个损失案例为何在 solver 中失败、82 个新增成功有何共同特征、原 R1 距离/梯度/候选是否被干扰，以及增广模型是否在 S0-S7 出现和 Paired-Slope 相似的 ReLU 失活问题。
