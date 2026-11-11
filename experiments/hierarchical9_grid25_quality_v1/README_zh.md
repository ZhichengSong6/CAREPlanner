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


## 第二步：fresh holdout 坐标检查（只读，不生成正式 split）

Grid25 25mm 网格与旧网格不同。旧 starts.jsonl 每行包含老 x_index 和实际 x 坐标，**只能使用坐标**做新网格 holdout 排除方案，不能直接复用整数索引。

    cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
    export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
    export GRID25_ROOT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"
    export R012_ROOT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_scratch50k_r012_v1"
    FRESH="$R012_ROOT/evaluation_fresh_holdout_v1/starts.jsonl"

    "$VIS_PYTHON" -m unittest discover -s experiments/hierarchical9_grid25_quality_v1 -p 'test_split_preflight.py' -v
    "$VIS_PYTHON" experiments/hierarchical9_grid25_quality_v1/split_preflight.py --root "$GRID25_ROOT" --fresh-starts "$FRESH"

脚本按实际 XYZ 距离，列出 25/50/75mm 排除邻域在 Grid25 上影响多少位置，并仅用 ZIP 文件目录读取压缩/未压缩大小预估。输入是旧 fresh holdout 的坐标，不是标签，输出只写 analysis/grid25_quality_v1/split_preflight.json。此步不需要 Slurm/GPU，不复制 36GB 数据，不生成正式划分、不导出 36GB q 张量，也不启动训练。

**不可以直接运行旧 hierarchical9_full_dataset_v4_v1/prepare_full.sh 指向 Grid25 NPZ**：它的 prepare.py 读取的是旧 cache_io.open_job 来源格式，而且使用 starts.jsonl 的旧 x_index。需要后续专门的 Grid25 bank 缓存/坐标 split 适配器与小样本标签 pilot；本脚本仅为此前必要预检查。

## 修复：old fresh x 不全部属于 Grid25 工作空间

此前 split_preflight.py 在 old fresh holdout 任意 x 超出 Grid25 x/y=[-0.8,0.8]m、z=[0,1]m 时直接 STOP。
旧 fresh JSONL 是从旧网格的真实空间坐标生成的（不是新的 25mm 网格），因此这一条件不合理。

新版不会强制 old x 落在新网格内；仍使用全部独立 XYZ 坐标做 25/50/75mm 欧氏距离排除：
- 范围外但在新网格边界附近的 old fresh，仍可排除邻近训练点；
- 范围外很远的 old fresh，不会凭空映射成 Grid25 测试点，也不会排除无关样本；
- 报告新增 fresh_unique_positions_inside_workspace / outside_workspace、按 xyz 方向的越界计数和实例、最大越界距离、各半径内能触及 Grid25 的 fresh 位置数量。

这只是 **坐标邻域预检查**。old fresh 值必须与 Grid25 的 base_link 坐标使用相同坐标系；若来源不同，需要先转换并验证，而不能把错误映射“纠正”为有效分割。即使旧 holdout 位置全部越界，它们也不能直接作为 Grid25 域内的 validation set。正式 Grid25 新划分仍需单独实现、审阅，旧 x_index 永远不复用。

重新执行：

    git fetch origin mainline-b/h9-scratch50k-r012-v1
    git merge --ff-only origin/mainline-b/h9-scratch50k-r012-v1
    "$VIS_PYTHON" -m unittest discover -s experiments/hierarchical9_grid25_quality_v1 -p test_split_preflight.py -v
    "$VIS_PYTHON" experiments/hierarchical9_grid25_quality_v1/split_preflight.py --root "$GRID25_ROOT" --fresh-starts "$FRESH"

如果报告中的 max_distance_outside_workspace_mm 很大、越界比例异常或几乎没有 fresh 坐标靠近 Grid25，需要核实 old holdout 的空间覆盖与 frame，再决定新的独立 fresh cohort；**不能默默忽略越界报告**。
