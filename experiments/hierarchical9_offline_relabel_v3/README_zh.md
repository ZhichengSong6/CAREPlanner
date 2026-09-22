# R1 批量离线配对标签 v3

**本轮是有限批量 pilot，不是完整数据集重标，不训练 R1，不改变网络结构。**
保留原 q0 数据、R1、v1/v2 源码及其所有结果。只按 `(query_id, sensor_id)` 组织标签；
同一 sensor 的六个面仍是其完整 FOV 的内部约束，不新增按面划分的数据或输出 head。

## 1. 首批规模与“全量”的含义

默认 `sampling.json`：64 个训练 x、16 个开发验证 x，每个 x 两个 uniform q，
再从一个原库锚点沿解析法向的正负方向各取一个 q。目标为 **320 个 (x,q)**，
最多 **2560 个查询-sensor 标注**。无原库支持的 sensor 不求解；原库锚点无效、
梯度退化或扰动越界会记录跳过原因，不 clamp、不补采容易样本，因此实际数可少于320。

- x 继续使用原空间点和原 train/val 划分，不加密空间网格。
- 排除原 smoke 用过的 x，避免重复六案例式诊断。
- 在原 split 内按 sensor 库支持覆盖选点，要求每个 sensor 至少由2个所选 x 支持；
  这是明确的支持均衡 pilot，不是均匀 workspace benchmark。支持不等于标注成功。
- bank-normal pair 不是精确边界法向样本，也不保证两点跨过真实零面；实际 FOV 符号逐 sensor 计算并记录。
- 全部查询在优化前冻结。训练只能读取已标查询，不能扰动/重组输入后沿用标签。

原始 x 有限、查询 q 连续，“所有 (x,q) 的完整标签”不是原NPZ已有的一张有限表。
扩展应明确每个 x 的查询预算、逐 sensor 支持和计算成本。当前不提供隐式 `all` 模式。
改变采样规模或策略必须创建独立输出与配置；不能改变已有 batch 的冻结查询。

## 2. 两阶段执行，不把候选梯度自动升级成真值

### base：批量基础标签

复用 v2 的 `solve_attempts` / `screen_candidates` 和数值等价候选选择。
求解预算继承已完成 v1 smoke 的 solver 设置，不减少约束、重设几何阈值或缩减搜索。
每个查询-sensor 保存原方法的 old_value/old_grad，以及 new_value、q_star、gradient_candidate、
候选质量、所有尝试及耗时。原FP32 oracle再检查保存候选的FOV和符号。

**这一层 `grad_valid=false`、`verification=NOT_RUN`。** 有候选方向不代表已做距离差分复核。
每个查询-sensor独立原子保存、独立哈希与锁；中断仅丢失尚未完成的那个求解，已完成记录恢复时不重算。

### audit：预先固定的分层深度复核

在生成查询时就按 `split × sensor × 实际FOV正负号` 各抽1条，共最多32条；
抽样不读取新标签成功与否，不因失败换一个样本。原始失败/歧义案例保留SKIPPED或HOLD。
近零符号保护区不纳入该两侧抽查，但数量单独报告。审计计划缺某一层是覆盖缺口，不是该层通过。

只对这份固定名单复用中心查询已有attempts，运行 v2 两尺度、双侧、单侧斜率检查。
每次扰动重新优化完整约束；不固定 q_star 冒充距离求解。
未抽中的梯度仍 NOT_RUN，不因同一个sensor其他样本通过而被放行。
审计发现更近的中心候选时，其新值有效性也被撤回到待复核状态，不悄悄替换成未经验证答案。

这些都是近似局部数值证据，`global_nearest_certified=false`。
本轮不声明CPU/GPU的速度优劣，也不改变原求解预算制造提速结果。

## 3. 使用原来的已完成 smoke

默认只读：

```
/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_offline_labels_v1/smoke_20260921_202241
```

其manifest指向已存在的 numeric bank cache；不重新解压7GB原数据。
验证源查询/空间划分/v1源码哈希、bank身份和几何文件身份。没有原始数值cache就明确报错。
不修改 master，不自动checkout、stash、reset、安装包、训练、提交/取消其他任务。

## 4. 更新后先跑代码测试

在服务器 worktree 根目录：

```bash
sha256sum -c experiments/hierarchical9_offline_relabel_v3/SHA256SUMS
"$HOME/miniforge3/envs/viscdf/bin/python" -m unittest discover \
  -s experiments/hierarchical9_offline_relabel_v3 -p test_batch.py -v
```

测试中的临时目录和合成oracle不是实际机器人验证结果。

## 5. 本轮只提交一次 base

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
export REPO="$PWD"
export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
NODE=3090node1 DEVICE=cuda WORKERS=4 \
  bash experiments/hierarchical9_offline_relabel_v3/submit.sh base
```

默认申请4GPU、16CPU、24小时walltime：这是资源申请上限，不是预计完成时间。
依赖在sbatch之前检查；环境仍用已有viscdf Python。`DEVICE=cpu`不请求GPU，是否可调度取决于集群策略。
新输出在 `outputs/mainline_b/r1_offline_labels_v3/pilot_时间戳_PID`。
提交器**忽略旧 OUT/VERIFY_OUT/LOG_DIR**。自定义或恢复用 `BATCH_OUT`；不会覆盖旧latest文件。
同一BATCH_OUT同阶段已有活动作业时拒绝重复提交。日志打印实际job、out、log、state。

### 只看当前输出，不再运行 submit

```bash
# 默认看最近一次base任务：
bash experiments/hierarchical9_offline_relabel_v3/watch.sh

STATE=/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_offline_labels_v3/latest_base.env
bash experiments/hierarchical9_offline_relabel_v3/watch.sh "$STATE" status
bash experiments/hierarchical9_offline_relabel_v3/watch.sh "$STATE" workers
bash experiments/hierarchical9_offline_relabel_v3/watch.sh "$STATE" plan
bash experiments/hierarchical9_offline_relabel_v3/watch.sh "$STATE" summary
```

需要固定某个job时，用提交器打印的base_job_JOBID.env代替latest文件。
watch只读取，不提交、不取消、不merge。Ctrl+C只退出查看。
主日志每20秒汇集各worker最后阶段；基础阶段每个查询-sensor开始和完成都会打印。

base结束后返回 `sampling_report.json`、`base_cache/summary.md`、`base_cache/summary.json`，
以及 `preflight.base.rank*.json`。先看真实内外侧与sensor覆盖和成本，再启动独立audit。
不要重新跑16519或旧v1 pilot。

## 6. 后续明确执行 audit（不要和base同时提交）

base完成且检查覆盖/成本后，使用**同一批次**：

```bash
export BATCH_OUT=/实际base输出目录
NODE=3090node1 DEVICE=cuda WORKERS=4 \
  bash experiments/hierarchical9_offline_relabel_v3/submit.sh audit

STATE=/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_offline_labels_v3/latest_audit.env
bash experiments/hierarchical9_offline_relabel_v3/watch.sh "$STATE" main
```

audit不会自动执行，也不会抽查全部2560条。名单早已冻结，无需重新生成中心标签。
完成后生成 `paired_cache/summary.md/json` 和 `paired_cache/dataset_index.json`。

## 7. 中断恢复

确认原任务已退出后，保留相同的代码、配置、SOURCE_OUT、BATCH_OUT和依赖版本：

```bash
BATCH_OUT=/原v3输出目录 RESUME=true \
  bash experiments/hierarchical9_offline_relabel_v3/submit.sh base
# 恢复audit时把base换为audit。
```

完成的逐task记录检查哈希后跳过，不覆盖。不兼容记录、缺失文件或损坏均报错。
不同阶段可采用不同worker数量或CPU/GPU，但数值文件/配置必须一致，且每个设备都重新预检。
不要指向原v1/v2目录；不要删结果重来。原代码/几何修改后需要新批次。

## 8. 缓存字段和读取器

paired_cache分片包含：

```
query_id[N], x_index[N], x[N,3], q_query[N,7], split[N], support[N,8]
old_value/new_value[N,8], old_grad/new_grad[N,8,7], q_star[N,8,7]
gradient_candidate[N,8,7], gradient_candidate_valid[N,8]
new_value_valid/new_grad_valid[N,8], audit_selected[N,8], gradient_status[N,8]
paired_value_mask/paired_grad_mask[N,8]
union_old/new_value[N], union_old/new_grad[N,7], paired_union_value/grad_mask[N]
```

FP32数值、原7D关节度量、原sensor順序；未定义结果用NaN加明确mask，reader仅在保留mask前提下提供零占位。
只有实际audit PASS且值仍有效的记录，才有new_grad_valid。共同mask让old/new两组获得相同监督机会。
任何原本有bank支持的sensor缺新值时，不删除这个sensor重算假的union winner。
union梯度还要求old/new各自winner的梯度均有共同监督、且winner唯一。

```python
from batch_dataset import PairedLabels
old = PairedLabels('/实际批次/paired_cache', label_set='old', split='train')
new = PairedLabels('/实际批次/paired_cache', label_set='new', split='train')
# 两组的query、value/gradient/union mask完全匹配。
```

base_cache只是基础结果，reader拒绝把它当作已经审计的训练缓存。
**本包不实现训练器**：原单mask objective不能直接接入；后续需分别按值、梯度和union有效性归一化。
`training_ready=false`不是流程失败，而是要求先审查覆盖/质量和监督策略。
最多32条审计也不能自动提供大规模梯度监督，更不能代表全部样本的准确率。
后续扩大缓存和梯度验证覆盖，应基于本轮报告；当前R1冻结，不覆盖任何checkpoint。

## 9. 规模边界

这不是全量重标，也不保证每个sensor每类查询都有足够成功标签。
分层抽查比例、缺失层、失败、内外侧比例、每个有效标签的成本必须报告。
诊断容差不修改运行时安全阈值。R1模型收益只能由后续同查询/同预算的旧新标签训练对照回答。
