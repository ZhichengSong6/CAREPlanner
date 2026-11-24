# 正式 Grid25 H9 第二轮：V3 零边界加权（单一变量对照）

## 目的

第一轮124段正式 H9 R1 V3-only 验证 RMSE 0.09715688490905942，Case026 在旧scalar投影+H9 S4候选中得到预测 S4 +0.0200，但解析 conservative FOV margin -0.00629。原先 Case026 还包含 LOS 检查，但此处**只针对 FOV 零边界的值/符号精度**，不考虑自遮挡、碰撞或轨迹。旧8-head的checkpoint不在该HPC上，不能声称旧网络更好。

原 V3 标签及运行时保守FOV几何参数一致：水平50°、垂直66°、z_min=.2、z_max=.7、delta=.01。但V3标签是**best-found关节空间signed距离**，运行时保守FOV margin 是几何平面距离，它们的绝对数值不应直接相减。两者符号和零面附近行为是关键。

原 `select_v3_indices` 先覆盖每个sensor的正负状态、再鼓励q多样性，最后以平均 |FOV margin| 破局；不是强制靠近零边界采样。原V3训练不直接优化接近零面的准确性，也无FD-gradient标签。

本轮*只改变一项训练超参数*：在原masked sensor MSE中，将 `|V3标签| <= 0.10` 的**有效**标签权重从1提升到3；非边界仍为1。损失按权重总和归一。没有新的标签、没有额外梯度损失。所有8个sensor遵循同一个规则。union代理MSE及其权重0.2保持不变。

**保留不变**：原R1九分支模型、数据126201个x/504804个(x,q)/3644719个有效传感器标签、109490/16711的固定空间split、随机种子0、20000步、batch1024、Adam 1e-3、cosine到1e-5、每500步验证。checkpoint选择改为 `val_boundary_sensor_rmse`（这是与研究目标一致的选择准则），同时报告global RMSE、MAE、sign accuracy和各sensor结果。这意味着它的best.pt选择规则和第一轮不同，比较时应固定检查点和同一验证指标。

**不覆盖**第一轮任何文件。第二轮数据仍引用：

```
$GRID25_ROOT/derived/grid25_v3_full_training_v1/dataset_segments_000_123
```

新输出：

```
$GRID25_ROOT/derived/grid25_v3_boundary_focus_v1/run_r1_v3_boundary_focus/
  best.pt
  final.pt
  run.json
  metrics.jsonl
```

## 直接提交一个正式GPU作业（不用再次生成数据或pilot测试）

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
git -c http.version=HTTP/1.1 fetch origin mainline-b/h9-scratch50k-r012-v1
git merge --ff-only origin/mainline-b/h9-scratch50k-r012-v1
export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
export GRID25_ROOT="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"
NEXT=experiments/hierarchical9_grid25_v3_boundary_focus_v1
bash "$NEXT/submit.sh"
```

提交器会确认自己没有其他Slurm作业，只做train.py Python语法检查，不运行额外的dataset re-verify、重复label生产或小型pilot。正式训练本身会验证原冻结labels.npz SHA；数据格式或split不符直接拒绝。申请3090node3一张RTX3090、4CPU、最多4小时。

查看：`squeue -u "$USER"`，完成后 `bash "$NEXT/run.sh" status`。

## 比较问题

唯一需要回答的是：**和固定第一轮best.pt相比，零附近的qualified V3值预测RMSE与符号准确率是否改善，同时全局RMSE是否被过度牺牲。** 新训练不是为了自遮挡，也不意味着S4 Case026一定能过；后续须单独检查FOV-only目标的边界穿越和q梯度有效性。不可将这次零距离区间加权称为新label精度提高，也不能声称已满足实时规划。
