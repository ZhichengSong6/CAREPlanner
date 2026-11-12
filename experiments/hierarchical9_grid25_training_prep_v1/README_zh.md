# Grid25 Training Preparation v1：空间独立划分 + 按需 mmap 基础 Q0 库

这个阶段只准备基础 Q0 数据 **不生成 V4 标签、不启动 GPU/训练作业**。永远不修改 `formal/`、Grid25 原始 36GB NPZ、2,707 个原始分片或原 fresh holdout。`complete.json` 中 `training_ready=false` 的语义保持不变。

## 已核验的输入

- Grid25 65 × 65 × 41 = 173,225 个 x；所有 2,707 片已通过原合并验证
- 原 NPZ: `$GRID25_ROOT/formal/dataset/visibility_grid25_fovonly.npz`
- 原 fresh: `$R012_ROOT/evaluation_fresh_holdout_v1/starts.jsonl`
- 旧 holdout 的 118 个**独立 XYZ** 中 39 个在 Grid25 工作空间之外；旧 `x_index` 不允许复用
- 2026-11-12 的 preflight 发现 75mm 时排除 8,669 个 Grid25 位置，保留 164,556 个候选位置（尚未划分 train/val）
- 源数据仅包含 FOV-only Q0，缺少完整 V4 global sign / V3 pairs / V4 tubes 标签

## 运行

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
git fetch origin mainline-b/h9-scratch50k-r012-v1
git merge --ff-only origin/mainline-b/h9-scratch50k-r012-v1
export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
export GRID25_ROOT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"
export R012_ROOT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_scratch50k_r012_v1"
SCRIPT=experiments/hierarchical9_grid25_training_prep_v1/run.sh
bash "$SCRIPT" test
```

**先确认两个数据集的 XYZ 都在同一个 `base_link` 坐标系。** 用户明确确认这个前提后运行：

```bash
bash "$SCRIPT" plan \
  --root "$GRID25_ROOT" \
  --fresh-starts "$R012_ROOT/evaluation_fresh_holdout_v1/starts.jsonl" \
  --fresh-mm 75 \
  --val-buffer-mm 75 \
  --block-cells 8 \
  --val-fraction 0.10 \
  --seed 20261112 \
  --confirm-same-base-link

bash "$SCRIPT" verify --root "$GRID25_ROOT"
```

**plan 不会加载完整的 36GB 压缩 q 数据，内存开销主要为 173,225 个网格坐标与布尔 mask。** 它生成：

```text
$GRID25_ROOT/derived/grid25_trainprep_v1/
  splits.npz
    train_x_index
    val_x_index
    fresh_guard_x_index
    val_guard_x_index
    old_fresh_xyz_m
  manifest.json
```

- 外部 fresh：旧 starts 中真实 XYZ，只用于 holdout 邻域排除和未来独立外部测试，不直接成为新的 Grid25 行索引
- fresh guard：按原 XYZ 到新网格的实际欧氏距离 75mm 排除，**包括靠近 Grid25 边缘的旧域外坐标**
- validation：先从剩余候选空间的 8×8×8 网格块按确定性种子选择约 10% 的位置作为 Grid25 开发验证
- val guard：再排除距 validation 75mm 内的剩余位置，防止邻近验证泄漏
- train：排除外部 fresh 邻域、验证集、验证集 buffer 后的其余点
- 输出不承诺 train/val 分布平衡或训练精度；训练前需审核各组 sensor support 与类别分布
- 显式固定全部划分参数，拒绝覆盖已经存在的输出目录。若希望变更半径或随机种子，应选全新 `--out` 路径并记录实验版本。

## 第二阶段（可选，可能需要较长的文件系统 I/O）

**只有 plan/verify 完成，磁盘剩余空间和使用规范允许时，再单独执行：**

```bash
bash "$SCRIPT" cache --root "$GRID25_ROOT"
```

- 在执行前核对 merged NPZ 与 published `complete.json` 中的 SHA256
- 复用仓库已有 `cache_io.import_bank()` 以小块流式将必要的 `.npy` 成员写入 `bank_cache`，不让每个 worker 各自解压 36GB
- 使用 `np.load(..., mmap_mode="r")` 读取需要的 Q0、valid mask 和元信息；只解包 bank 必需字段而非整个 ZIP 的 56GB 原始数据
- 需要至少必要成员的未压缩大小 + 5GiB 可用空间；具体耗时由共享文件系统吞吐量决定；此命令在登录节点可能产生大量 I/O，应遵守集群 CPU/I/O 规则，必要时改在单节点 CPU 批作业中执行
- 不修改源数据、不重跑采集、不算 V4 监督标签；即使 cache 构建成功，**仍然 `training_ready=false`**
- 输入来源或缓存已存在时拒绝覆写。程序异常或服务器重启后先检查 `bank_cache` 的完整性，不要盲目删除目录。

## 仍然不能直接运行的旧命令

**不要直接用** `hierarchical9_full_dataset_v4_v1/prepare_full.sh` **给新 Grid25 生成标签**。旧 `prepare.py` 依赖 old `cache_io.open_job` 的旧来源及旧 `x_index`。下一步还要为 Grid25 新 bank 和本次划分编写 V4 专用 adapter、少量 pilot 与数值对比，再进入正式训练。

测试为轻量 CPU 函数测试；不意味着当前生产数据上的 GPU 训练兼容性已通过。
