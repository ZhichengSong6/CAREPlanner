# R1 离线连续 FOV 重标注 v1

**状态：代码交付；已做 CPU 合成测试。尚未在用户的真实 URDF、q₀ 数据或 GPU 上运行。**

本包只新增一个实验目录，不修改旧数据、权重、训练器、runtime、URDF、FOV/LOS/VBC/GCDF/tracker。没有自动训练、启动机器人或推送 Git 的操作。输出是经过数值质量检查的**近似连续边界标签**，不是全局最近点证书，也不是机器人安全认证。

## 1. 本轮具体实现了什么

对固定的 `(x_index, q_query)`，计算八个 sensor 的两套标签：

- `old_value / old_grad`：保持旧 FP32 最近 q₀ 距离、1e-8 平方距离 floor 和解析 FOV 符号。
- `new_value / new_grad`：以原 q₀ 和查询自身作初始化，用连续约束优化寻找更近的 FOV 边界点。

搜索目标：`min 0.5 * ||q_active_star - q_active_query||²`。
约束：枚举六个 FOV 面，每次一个面等于零，其余面非负，同时满足原关节限位。
非相关关节保持原查询值。距离度量使用原始有效关节欧氏距离，不引入角度 wrap、归一化、sensor 平均或工作空间距离替代。

每个查询默认最多 4 个起点（查询本身 + 最近/多样化库点），每个起点枚举六个面；库种子从最近 32 个有效候选中选择。SLSQP 最多 100 次迭代。完整参数在 `solver_config.json`，训练/运行的安全参数不受它影响。

**为什么不是只调用旧 refine_boundary：** 旧精修只寻找零点；这里同时最小化与指定查询 q 的距离。使用多个起点不是平均多个梯度。新标签通过最终选中的近似最近点计算。

## 2. 代码依赖和核对过的参考

参考仓库：`ZhichengSong6/CAREPlanner`
参考 commit：`e9ada9d502fd622418f5fc1c28a8a52beb863364`

依赖本地已有源码：

- `src/care_visibility_cdf/scripts/extract_visibility_zero_level_sets.py`
- `src/care_visibility_cdf/scripts/validate_visibility_oracle.py`
- `src/care_visibility_cdf/scripts/check_visibility_self_occlusion.py`（上游模块的 import 依赖，不运行 LOS 标注）
- `src/care_visibility_cdf/scripts/train_signed_visibility_cdf_pairwise_replace.py`
- `src/care_visibility_cdf/scripts/train_per_sensor_visibility_cdf.py`
- `src/arm_description/urdf/Arm.urdf`

不依赖 R1 checkpoint，不加载 R1 网络，不依赖本地 ROS/Gazebo 服务。
使用现有 `viscdf` Python 环境；脚本不自动安装或升级任何依赖。`requirements.txt` 是依赖说明，不建议直接对现有环境执行全量升级。

### 为什么离线使用 float64 FK

上游 FK 有 FP32 矩阵分配，直接把 SciPy 的 float64 查询传进去不能保证真正的 float64 优化。因此本包从上游 `prepare_chain_specs` 取得同样的 URDF 常量，使用独立、dtype-safe 的 float64 运动学计算。

真实预检必须检查：

1. 归一化 FOV margin 与未修改上游 FP32 oracle 的最大误差 <= 2e-6 m；delta 只减一次。
2. float64 Jacobian 与中心有限差分的相对误差 <= 2e-4。
3. NumPy 旧标签与原 Torch decoder/target 逐值对照：值误差 <= 5e-6，梯度误差 <= 3e-5。
4. sensor/关节顺序、有效关节 mask、关节限位、数据 FOV 元信息一致。
5. 每个新候选在保存为 FP32 后，再用原 oracle 检查边界残差。

若当前 Git 中含固定参考 commit，会比对 URDF 内容；不同则拒绝。如果该对象不存在，报告 `REFERENCE_OBJECT_NOT_AVAILABLE`，不会声称历史 URDF 身份已经验证。此时训练前需独立确认原始 URDF。

首次预检冻结源码、URDF、Python/NumPy/SciPy/Torch 版本与 solver 配置；同一输出目录不能混入改过版本的标签。

## 3. 从现有主线 B 分支获取代码

本目录位于专用分支 `mainline-b/h9-scratch50k-r012-v1`，不需要上传或解压 ZIP。
保持当前服务器 worktree，不切换到 master，不合并其他研究分支。
先确认没有待处理的源码改动；下面的命令只允许 fast-forward，有冲突或分叉就停止，不能 reset/clean 强行覆盖。

```bash
(
set -euo pipefail
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
BRANCH=mainline-b/h9-scratch50k-r012-v1
if [[ "$(git branch --show-current)" != "$BRANCH" ]]; then
  echo "[STOP] 当前分支不是 $BRANCH；先检查 git worktree list，不自动切分支。" >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  git status --short
  echo '[STOP] 有未提交源码改动，先妥善保存；不自动 stash/reset。' >&2
  exit 2
fi
git fetch origin "$BRANCH"
git merge --ff-only "origin/$BRANCH"
sha256sum -c experiments/hierarchical9_offline_relabel_v1/SHA256SUMS
)
```

未跟踪文件与新目录有重名冲突时，Git 会拒绝覆盖。不要删除旧结果来强行合并。
上述 Git 操作由使用者手动执行；标注脚本本身不执行 Git 写操作。

## 4. 先提交 smoke，不直接生成大规模训练标签

在服务器代码仓库中：

```bash
export REPO=/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
export ARTIFACT_REPO=/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner
export VIS_PYTHON=$HOME/miniforge3/envs/viscdf/bin/python

cd "$REPO"
bash experiments/hierarchical9_offline_relabel_v1/submit.sh smoke
```

默认申请 `3090node1`、4 GPU、16 CPU、1 小时 walltime；这是资源上限，不是运行时间预测。节点资源是否可用由 Slurm 决定。不会申请 `--mem`。

脚本默认查找数据：

```text
$ARTIFACT_REPO/src/care_visibility_cdf/data/visibility_yiming_style_grid30_q20000_k500_fovonly.npz
```

实际位置不同请先 `export DATA=实际绝对路径`；不会自动猜另一份数据。
可设置 `NODE`、`TIME_LIMIT`、`OUT`、`BANK_CACHE`、`URDF`、`VIS_PYTHON`。修改 FOV 参数或 solver 配置必须使用新输出目录。

smoke 默认：原训练划分中 2 个 x，原验证划分中 2 个 x；每个 x 一个 uniform 查询和一个 near-bank 查询。通常目标是 8 个查询、最多 64 个 sensor 标注，但 near-bank 越界生成失败会被记录并排除，因此真实分母以 manifest 为准。**smoke 只检查流程，不足以评估标签收益或训练模型。**

执行阶段：

```text
合成测试
→ 原始 NPZ 数值数组缓存化
→ 固定查询与空间划分
→ 真实 FOV/梯度/旧标签一致性预检
→ 四个 worker 各自处理固定分片
→ 完整性检查、汇总
```

提交后终端打印 `[out]` 和 `[log]`。读取实际日志路径即可：

```bash
tail -F 终端打印的实际log路径
```

worker 的细节在：`<OUT>/logs/worker_0.log` 至 `worker_3.log`。
不要在登录节点运行正式标注。

### 不使用 Slurm 的直接运行

在已获计算资源的终端中，可以显式使用 CPU，避免假设小批量优化一定在 GPU 更快：

```bash
REPO="$PWD" \
DATA="/absolute/path/to/the/original.npz" \
ARTIFACT_REPO="/absolute/path/to/artifacts" \
VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python" \
DEVICE=cpu WORKERS=4 \
bash experiments/hierarchical9_offline_relabel_v1/run.sh smoke
```

SLSQP 优化器在 CPU 上调度；`DEVICE=cuda` 只把 FK/自动微分移到各 GPU。Python 调度、小矩阵和同步可能是瓶颈，本包不承诺 GPU 高利用率或线性加速。

## 5. 原 NPZ 如何被使用

必须存在的数值字段：

- `x[P,3]`、`q[P,K,7,8]`、`valid_fov[P,K,8]`、`sensor_chain_masks[8,7]`；
- `k[P]`（原网格索引，不是每点 q₀ 数量）；
- `q_min[7]`、`q_max[7]`；
- `joint_names`、`sensor_frames`；
- `horizontal_fov_deg`、`vertical_fov_deg`、`z_min`、`z_max`、`delta`。

不根据文件名推断 P 或 K，不默认缺失的 joint limits，不读取 pickle object。使用原始 x/q=float32、valid_fov=bool 格式。

大型压缩 NPZ 在准备阶段**只解出所需字段一次**到独立 `bank_numeric_cache/`；四个 worker 读取只读 `.npy` memory map，避免四次完整解压原大文件。原 NPZ 和提取后的数组都有 SHA256；常规重启验证缓存统计信息和 manifest，`verify-bank` 可重新全量校验数组 SHA。

提取占用额外磁盘，脚本先按归档的未压缩大小检查空间。首次准备被强制杀死可能留下 `.building.*` 临时目录；确认没有运行任务后再人工清理，不要删原始 NPZ。

## 6. 查询和空间划分

复用旧 split 算法：有效 x 索引按 `seed=0` 打乱，`val_count=min(1000,max(1,有效x数//10))`；先冻结全部 train/val x 列表，再抽本轮子集。

同一个 x 派生的所有查询保持在同一 split。后续标注失败不会触发重新划分。这里的 val 是开发验证，不声称是新的独立 fresh holdout。

查询组：

- `query_group=0`：原关节范围内 uniform，包含 FOV 内外两侧，不能误称全是 outside；
- `query_group=1`：随机有效库点附近，对选中 sensor 的有效关节添加 Gaussian 扰动。越界不 clamp 成另一条标签，最多按声明预算重试。

`near_bank` **不是**前述已经验证两侧符号的 local-boundary benchmark。不能直接把这里的覆盖率与历史 local/uniform 求解成功率比较。

所有查询在连续标签优化之前固定；不根据 R1 结果选取有利样本，不使用 R1 权重。每个查询尝试原库支持的全部八个 head；无库支持保持明确缺失。

## 7. 质量标记不是全局证书

每个 head 保存 `status`、值/梯度 mask、q*、数值残差和完整优化尝试。

`new_value_valid=True` 要求找到可行候选、优化器收敛、相对一阶 KKT 残差合格、查询符号可靠，并通过原 oracle 的保存精度复核。

`new_grad_valid=True` 还要求单一规则活跃面、远离关节限位、法向非退化、查询到 q* 的方向与法向一致、有限差分检查通过、没有检测到明显多解歧义。

- 这些是近似标签的数值筛选，不是全局最近点证明。
- SLSQP 成功和 KKT 残差小都是不足以单独证明全局最优的条件。
- 多面角点/限位点可能保留值标签而不保留梯度标签；并不是声明所有此类距离都不可微。
- 检测到几乎等距、明显不同的 q*，不平均梯度。
- 未找到边界不表示不可达；没有标签不表示不可见。
- `global_nearest_certified` 始终为 false。
- 低可信候选的数值可以留在原始审计文件，但 validity=false；无候选使用 NaN，不伪造 0。

默认优化只从有限种子出发，确实可能漏掉另一片更近的边界。这是本方法剩余的近似误差，后续需要提高预算、独立核验和距离变化检查来衡量。

## 8. 输出和训练接口

```text
<OUT>/
  manifest.json
  queries.npz                 # query_id, x_index, q_query, split, 来源
  spatial_splits.npz
  run_spec.json               # 冻结源码/URDF/软件版本/solver
  preflight.json
  preflight.rank*.json
  logs/
  shards/shard_000000/
    labels.npz
    audit.jsonl.gz
    complete.json
  ...
  dataset_index.json          # 只有所有分片完整才生成
  label_summary.json
  label_summary.md
```

每个标签分片主要字段：

| 字段 | 形状 |
|---|---|
| query_id, x_index, source_grid_index, split, query_group | [N] |
| x, q_query | [N,3], [N,7] |
| old_value, new_value | [N,8] |
| old_grad, new_grad, q_star | [N,8,7] |
| support, old_value_valid, old_grad_regular | [N,8] |
| new_value_valid, new_grad_valid | [N,8] |
| paired_value_mask, paired_grad_mask | [N,8] |
| union_old_value, union_new_value | [N] |
| union_old_grad, union_new_grad | [N,7] |
| paired_union_value_mask, paired_union_grad_mask | [N] |

还有 status、真实 query g、边界残差、KKT 残差、normal cosine、求解耗时和调用次数。

**Union 规则：** 延续原有有效 sensor 的 max-envelope，不声称它是真实几何并集 SDF。如果原支持集合中一个 sensor 新标签缺失，默认整条 union value 监督不可用，不允许静默丢掉该 sensor 后重新定义 winner。union 梯度还要求旧/新各自 winner 的梯度均可信，且不存在接近并列。

`dataset.py` 给出纯缓存读取器：

```python
from dataset import OfflineLabelDataset

old = OfflineLabelDataset('/absolute/path/to/OUT', label_set='old', split='train')
new = OfflineLabelDataset('/absolute/path/to/OUT', label_set='new', split='train')
a, b = old[0], new[0]
assert (a['inputs'] == b['inputs']).all()
# inputs: [10]；sensor_value: [8]；sensor_grad: [8,7]
# 两组拿到相同的 paired value/gradient/union masks。
```

读取器把 mask=false 的值转成安全占位 0，但**始终返回 mask**；这不是把缺失标成零距离。训练 loss 必须先正确筛选，不能 `NaN * 0`。

这不是完整训练器。原 objective 只有一套 sensor mask、会重算 union，不能不加修改就吞入这个字典。下一步训练接入需显式使用分开的值/梯度/union mask、对应的全局归一化计数，保持两组监督集合一致。

训练时可以重排/重复读取查询；不能改变 x/q、做随机抖动或重新 Cartesian 拼接后继续使用原标签。预测 q-gradient 仍需网络 autograd；缓存只提供参考梯度。高频随机访问很多压缩分片可能受 IO 限制，大规模时宜使用分片级打乱或进一步设计 mmap 训练格式，不改变配对身份。

## 9. resume 和 pilot

任务中断后，重复**同一模式和 OUT**：

```bash
OUT=/actual/existing/output RESUME=true \
bash experiments/hierarchical9_offline_relabel_v1/submit.sh smoke
```

已完成分片会校验 SHA 和 query_id 后跳过，未完成分片重新计算。worker 使用分片文件锁防止重叠写入；不以分片内已经计算了几条为准恢复。被硬杀死可能遗留 `.shard_*` 临时目录，不会作为成功结果。
代码、URDF、solver 参数、软件版本改变后，应选择新 OUT；不要用 resume 混接新旧标签。

smoke 通过且质量/覆盖分母合理后，才扩到 pilot：

```bash
TIME_LIMIT=04:00:00 \
bash experiments/hierarchical9_offline_relabel_v1/submit.sh pilot
```

这个 4 小时是示例 Slurm walltime 申请，不是完成时间预测；按集群规则调整。pilot 默认 32 个训练 x、8 个验证 x，每点 4 uniform + 2 near-bank，目标最多 240 查询。仍是标注试验，不是足以替代 50k 训练的数据量。

## 10. 更高预算抽查（建议在训练前做）

完成 merge 后，可从有库支持的 query/sensor 中随机抽样，包含之前失败的样本，增加起点和迭代次数重新计算：

```bash
PKG="$REPO/experiments/hierarchical9_offline_relabel_v1"
"$VIS_PYTHON" "$PKG/cli.py" audit \
  --repo "$REPO" --out "$OUT" --device cpu \
  --samples 8 --starts 8 --maxiter 200
```

生成 `high_budget_audit_seed77491.json`，不改原标签或 mask。如果经常找到显著更近的合格点，当前预算的标签还不可靠，应重新设计预算/种子并生成新版本，不能直接宣称 continuous labels 已改善。
重复抽查需要新 seed，已有报告不覆盖。高预算也不是全局证明。

## 11. 先看哪些结果

1. preflight 是否全部通过，参考 URDF 身份是否能确认；
2. 每个 sensor 的 value/gradient 有效比例与失败类型；
3. near-bank 和 uniform 的来源与真实分母（完整信息在查询和审计文件）；
4. 新边界 residual、old-new 距离差、old/new 梯度方向差；
5. 完整 union 监督还剩多少；
6. 更高预算抽查是否推翻很多当前标签。

“标注完成”不等于“所有标签有效”，更不等于“R1 已有提升”。本包完全没有测 R1 的值/梯度/solver 表现，也没有跑训练。

## 12. 外部接口依据

- SciPy SLSQP 文档： https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html
- SciPy 约束 / callable Jacobian： https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
- NumPy 数值加载与 mmap： https://numpy.org/doc/stable/reference/generated/numpy.load.html

脚本使用显式目标/约束 Jacobian，不依赖较新 SciPy 才有的 `workers` 或 SLSQP `multipliers` 返回字段。数值最优性残差另行计算。
