# 第三轮受控实验：真实 FOV 连续边界两侧新增 V3 距离标签

## 唯一研究问题

前两轮在同一个冻结 Grid25 Val 验证集上：
- 第一轮：全局 RMSE 0.0971569，近边界 RMSE 0.0582709、近边界符号81.20%，FOV-only 64-case成功44/64。
- 第二轮：全局 RMSE 0.0980717，近边界 RMSE 0.0479560、近边界符号84.33%，FOV-only 64-case成功43/64。

第二轮**降低距离预测误差，但没有提高FOV搜索成功率**。第三轮不再调整Loss权重；采用真正位于连续 FOV 零面两侧的新查询，并用原V3求解器重新计算合格的关节空间signed distance标签，检验改善训练样本分布能否提升FOV搜索。

**不涉及LOS、自遮挡、碰撞、VBC、真实机器人运动、梯度伪标签或改动GCDF安全设置。**

## 方案

- 冻结原124段全部数据：504804个(x,q)行、3644719个有效sensor距离标签；split、R1九分支架构、optimizer与20k步全部保留。
- 使用原冻结split确定性选择256个Train x及64个Val x；每个x/sensor选一个已经通过原V3验证的真实q_star作为连续 FOV 边界锚点。
- 按原解析 FOV 的**运动学链法向**做±0.04/0.06/0.08 rad两侧扰动；只有解析FOV符号一正一负的配对才进入离线V3标注。
- 两个新查询都调用**原** `core.solve_attempts → core.screen_candidates → v3.reference_gate`，要求两侧都经过V3 value_valid认证、符号正确且`|signed_distance|≤0.20`。绝不拿解析FOV margin代替距离监督，也不把未认证梯度当目标。
- 只有Train新增标签混入训练，Val新增标签单独评估。新查询是**sensor-specific**：每行只有所属sensor一个有效标签，其余sensor掩码为false。
- 第三轮每个batch 75%原训练query、25%新增Train边界query；对原、新部分分别计算相同masked MSE并按采样比例混合。原union proxy仍使用原训练行与0.2权重；新单sensor行不生成虚假union标签。
- checkpoint依旧由**原冻结Val全局sensor RMSE**决定（保持与第一、第二轮一致），另外独立报告新的配对Val RMSE、MAE和符号准确率。这样是数据增强实验，**不是加LOS Loss**。
- 这是第三轮**受控数据增强试验**，不是全Grid25每个x都额外标注，且Val原来已用于选择checkpoint，不能称为独立最终Test。

## 服务器直接操作（分两次Slurm作业，避免同账号并行）

在已经有前两轮完整结果的HPC仓库：

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
git -c http.version=HTTP/1.1 fetch origin mainline-b/h9-scratch50k-r012-v1
git merge --ff-only origin/mainline-b/h9-scratch50k-r012-v1
export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
export GRID25_ROOT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"
THIRD=experiments/hierarchical9_grid25_v3_boundary_pairs_v1

# 只做一次：建立不可覆盖的320个原x的边界锚点任务计划
bash "$THIRD/run.sh" prepare

# CPU离线V3原算法重标注，4CPU、0GPU，可断点恢复
bash "$THIRD/run.sh" submit-labels
```

查看标签任务实际状态：

```bash
squeue -u "$USER"
bash "$THIRD/run.sh" label-status
```

标签Slurm完成后，执行：

```bash
# 合并有合格正负两侧的新标签；如果不够，拒绝训练
bash "$THIRD/run.sh" merge
# 提交一次H9 R1正式受控训练（3090node3, RTX3090×1, 20000步）
bash "$THIRD/run.sh" submit-train
```

正式训练完成后：

```bash
bash "$THIRD/run.sh" train-status
bash "$THIRD/run.sh" fov-compare
```

最后一行只比较冻结的**同64个FOV-only Val场景**下第一轮、第二轮、第三轮的搜索成功率：相同原算法预算、相同模型无关的(x,q)起点，排除自遮挡。它只应在第三轮训练完成后运行。

## 输出位置

```
$GRID25_ROOT/derived/grid25_v3_boundary_pairs_v1/
  plan.json                    # 冻结来源/solver/标签任务元数据
  tasks.npz                    # 原qualified q_star和原空间split
  tasks/task_00000/...         # 每个配对重标注任务原子保存，可以恢复
  train_pairs.npz              # 只含训练x的qualified双侧标签
  val_pairs.npz                # 只含验证x的qualified双侧标签
  augmentation_manifest.json  # train/val摘要和SHA
  run_r1_v3_pairs_aug/
    best.pt
    final.pt
    metrics.jsonl
    run.json
  fov_val_three_models_v1.json
  logs/
```

*保留原V3数据、第一轮best.pt和第二轮best.pt全部不动。脚本不会自动修改任何在线规划配置。*

## 分析边界

一个新的V3标签依然是best-found且通过原资格审查的signed distance，并非数学上认证的全局精确SDF。对正负两侧添加相同x的新q，检验的是空间分布能否改善预测；使用现有FOV梯度搜索仍然需要之后的固定预算比较，不可仅凭训练RMSE推断规划成功。
