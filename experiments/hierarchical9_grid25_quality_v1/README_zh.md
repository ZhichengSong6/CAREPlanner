# Grid25 完成后：只读质量审计与训练前门控

## 现有生产数据

- Grid25：65×65×41 = 173,225 个位置，2,707 个完整分片
- FOV-only Q0 bank，八个 sensor；不是 LOS/self-occlusion 标注，也不是新 V4 signed-distance labels
- 已有最终合并文件 formal/dataset/visibility_grid25_fovonly.npz（约 36GB）
- 最终完整性请继续使用原来的 fast_segments_v1/run.sh verify，它会核验分片/合并哈希及 CRC
- 本目录只做覆盖统计和少量数值抽查，不改 frozen 采样引擎、不启动训练

## 运行（登录节点，仅 CPU，无需申请 Slurm/GPU）

先确认作业已结束，并有充足输出目录空间。以下操作只写到 analysis/ 子目录：

    cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
    export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
    export GRID25_ROOT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"

    "$VIS_PYTHON" -m unittest discover -s experiments/hierarchical9_grid25_quality_v1 -p 'test_audit.py' -v
    bash experiments/hierarchical9_grid25_fast_segments_v1/run.sh verify
    "$VIS_PYTHON" experiments/hierarchical9_grid25_quality_v1/audit.py --root "$GRID25_ROOT" --sample-shards 8

结果在：

    $GRID25_ROOT/analysis/grid25_quality_v1/report.json

这个脚本循环读取 2,707 片的 num_sensor_q0 小型数组，汇总每传感器的有效 Q0 条目数、支持的 x 数、未解析的 (x,s) 数、每个位置有多少个可支持传感器，并与正式 complete.json 汇总严格比对。另按确定性序列抽样 8 片检查 q/valid/g/关节范围/原始 epsilon。没有读取整个 36GB 的合并 q 数组，不重新计算归档的完整哈希（这一步由原 verify 实现）。

**解释注意**：
- 没找到 Q0 的 (x,s) 必须记为 UNRESOLVED，绝不是几何上“不可能可见”
- FOV-only 数据不含 LOS / self-occlusion，不能用于直接宣称真实遮挡安全
- 完整合并的 complete.json 中 training_ready=false，原因明确要求新网格的派生 V4 pairs/cache/splits 重新构造
- 旧的 V4 x_index / 配对样本索引属于旧网格，不允许直接复用
- 旧 trainer 的 VisibilityQ0Dataset 会从压缩 NPZ 中解包 x/q/valid，而这个 36GB ZIP 可能导致远大于 36GB 的内存占用；不得未经内存评估直接开训练
- PASS_Q0_AUDIT_NOT_TRAINING_READY 只表示基础 Q0 质量统计/抽查通过，不表示已经具备可训练的监督标签

## 下一阶段（另行设计，不随本审计自动启动）

1. 依据新网格和预留 fresh holdout，定义可追溯 train/val/test 划分。
2. 按所需原 V4 协议重建 Grid25 派生训练标签（global signs、V3 pairs、V4 tubes、相应缓存），而不是只替换旧脚本里的 NPZ 路径。
3. 先做小规模 dry-run 和数值校验，再提交正式训练作业。
