# Paired-Slope Scratch v1（正式全量训练）

这个新目录与旧 R1、V2-50K、Boundary-first v1 完全隔离。沿用 R1 private-tail H9 的模型结构和原始缓存；随机初始化，不加载任何旧模型权重。不修改 GCDF、VBC、LOS、tracker 或在线 runtime。

## 为什么引入成对监督

Boundary-first v1 在 job 17438 的 50k 训练完成后表现出 near-zero field：局部两侧的正负符号识别和梯度大小明显不足。原 V4 training cache 只保留单点的 x、q、sensor、offset，缺失 anchor 身份。本版从冻结的原始 v4_tubes/shard_*.npz 恢复 (x_index, sensor, source_slot, offset)，只匹配同一 anchor 的正负偏移点。检查两侧原始 FOV g_m 符号、零偏移锚点残差、同一法向几何一致性与原分片 SHA256；缺少任何一侧则不配对。不会生成新 q 或复用标签给新 q。

新训练作业在独立 formal/pairs 下建立已认证配对 mmap 与 24 分层索引（8 sensor × 3 radius），原 V4 文件及全量 training cache 保持只读。不会重新运行 V3/V4 标签生产。

## 固定方案

与 R1 同样的 2,184,329 参数 private-tail H9，随机初始化，50,000 Adam updates，4×RTX3090；FP32，关闭 TF32。Global 全空间 analytic sign 做 class-balanced loss；V3 只用有效 per-sensor value；V4 单点用分层 local Huber；Boundary 用零值和完整 7D normal。Paired 部分同时拟合 q+≈+t、q−≈−t 对应的场值，直接最小化 (f(q+)−f(q−))/(2t) 与 1 的偏差，并用正负 margin 增强符号分离；t 为 0.005、0.01、0.02 rad。V4 距离只是局部法向近似，非全局最近点证明。

固定 weights 与 batch 详见 protocol.json。每 update 四卡合计：Global 8192、V3 512、V4 4096、Boundary 1024、Paired 2048 对，每对两条前向输入，合计 17,920 前向输入行。优化器 warmup 1000 steps 至 3e-4，cosine 到 1e-5；无 pilot/smoke GPU 作业。模型选择使用固定验证 objective，而不读取任何 solver 回归结果。

## 提交

仓库根目录、mainline-b/h9-scratch50k-r012-v1 分支干净并更新后执行：

    export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
    env -u NODE bash experiments/hierarchical9_paired_slope_scratch_v1/run.sh train

内置静态检查、CPU 单元测试、只读 manifest/cache SHA 预检，然后申请单节点 4×3090、16CPU；默认排除以前启动异常的 3090node1。源码被复制到内容哈希目录，Slurm worker 使用该快照，不会把 spool 当作 train.py 位置。输出根目录：

    /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_paired_slope_scratch_v1/

只读查看：

    bash experiments/hierarchical9_paired_slope_scratch_v1/run.sh status
    bash experiments/hierarchical9_paired_slope_scratch_v1/run.sh log
    bash experiments/hierarchical9_paired_slope_scratch_v1/run.sh follow
    bash experiments/hierarchical9_paired_slope_scratch_v1/run.sh summary
    bash experiments/hierarchical9_paired_slope_scratch_v1/run.sh verify

仅当旧 Slurm 作业已退出、本实验 latest.pt 已存在且 final.pt 不存在时，才可执行 run.sh resume。原 experiment checkpoint 永不覆盖。本版保存 best_val.pt、final.pt、每5000步快照与校验哈希。

## 判断是否真正超过 R1

优先关注 pair_slope_mae（理想 0）、pair_slope_mean（理想 1）、pair_both_sign_accuracy、pair_side_rmse_rad、local sign、boundary zero/normal，以及真正的固定 solver 比较。R1 50k 是 50,000 optimizer updates，不是 50k epochs；旧 R1 每步 400,000 pairs，累计约 200 亿输入行。新方案 50k 累计约 8.96 亿前向输入行，高阶导监督和计算成本也不同。所以 50k vs 50k 只是相同更新次数，不是等数据量或等 GPU 算力。后续必须在相同 solver 配置和启动点集上报告成功率、实际 GPU 时和计算预算。当前已反复使用的 1963 个 frozen starts 是回归集，不再是最终未调参测试集。learned proposer 不能代替 FOV/LOS/GCDF/VBC 安全认证。

本地 13 项 CPU 单测（含实际执行的 2-rank Gloo 梯度一致性）通过；这些检查不代表 CUDA/NCCL 或正式四卡实验已经通过。
