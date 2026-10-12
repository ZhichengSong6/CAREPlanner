# R1-Outcome v1：同结构、原训练保留、候选几何效果监督

目标只有一个：训练一个能在原接口和原 solver 下优于 R1 的 checkpoint。
这是一项待验证实验，不保证提升，不修改本地控制、solver 阈值或安全链。

## 固定配置

- Private-tail H9，2,184,329 参数，输入10维、完整输出9维，与 R1 完全相同。
- 从头随机初始化；不加载 R1/PS/Normal 权重作初始化。
- 原 R1 每步4000×100=400,000查询组合，50,000成功更新；原标签、SDF/gradient/Eikonal/tension/consistency损失以及Adam/ReduceLROnPlateau/FP16 AMP规则保留。
- 前10k步不启用新增项；之后5k步线性增加到0.01，此后固定。
- 每卡每sensor固定采4对已有V4正负配对，共128对/全局更新；两侧都作为起点，共256条训练rollout。只读取此前PS建立的pairs缓存，**不使用PS模型或PS损失**。
- 仅训练原R1训练x分区；val只用原R1验证x分区。每个抽样使用独立随机流，必须复现原R1完整50k的x/q流SHA。

## 新增项是什么

训练期间，把网络的场值和关节梯度放入与原runtime一致的投影、二分和上升更新，得到候选；再以独立解析FOV的真实margin监督候选。

同一对记录的可见侧q+必须是解析检查合格的局部见证。每条目标裕量为 min(0.005m, 0.5*g(q+))，候选不足该裕量的部分按0.01m归一化，用SmoothL1惩罚。q-与q+都进入训练，避免只修复外侧失败而忽略已有可见侧。

训练采样中解析符号不稳健的配对按**预测之前**的几何条件mask掉，不替换成更容易成功的样本，也不把失败当不可行；日志报告valid_rows和rejected_witness_fraction。所有新候选的FOV都重新解析计算，不把旧offset标签赋给新q。

离散分支选择不可微：代码只对当前被选中的连续更新路径求导，不宣称整个solver全局光滑。包括q更新在内的梯度保持连接，新增项的参数梯度经过有限差分测试。原在线solver文件未修改。这个目标不保证每一步margin单调改善，也不提供碰撞、LOS或可执行性认证。

## 单次正式训练

在原分支、仓库根目录：

```bash
export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
env -u NODE bash experiments/hierarchical9_r1_outcome_v1/run.sh train
```

入口运行CPU测试、检查原R1/checkpoint/URDF/cache/pair字段与哈希，使用真实数据测试几何值和原始runtime的候选位置一致性，再提交单节点4×3090，默认排除3090node1。无独立GPU smoke/pilot作业。Slurm脚本使用显式源码快照，避免spool相对路径问题。

真实数据校验失败会在提交前停止；不得关闭校验强行提交。默认读取此前已建好的 h9_paired_slope_scratch_v1/formal/pairs，不会重建全量标签。

默认新输出：outputs/mainline_b/h9_r1_outcome_v1。所有旧模型、数据及实验不修改。可以通过OUTCOME_ROOT指定新的独立根；改变已冻结训练源码或配置不能复用旧输出。

## 查看、恢复与评估

```bash
bash experiments/hierarchical9_r1_outcome_v1/run.sh status
bash experiments/hierarchical9_r1_outcome_v1/run.sh log
bash experiments/hierarchical9_r1_outcome_v1/run.sh follow
bash experiments/hierarchical9_r1_outcome_v1/run.sh summary
```

仅当旧作业已结束，且本实验有latest.pt、尚无final.pt时可resume。恢复全部优化器/scaler/RNG和新增项step，不能加载历史R1、PS或Normal继续训练。

```bash
bash experiments/hierarchical9_r1_outcome_v1/run.sh resume
```

正式50k完成后：

```bash
bash experiments/hierarchical9_r1_outcome_v1/run.sh verify
bash experiments/hierarchical9_r1_outcome_v1/run.sh eval
```

另提交一次四卡只读评估：实际原runtime、相同1963个固定起点，不调阈值；原R1必须复现local618/683、uniform673/1280、all1291/1963。输出逐sensor和paired得失。这个集合只作回归，未用于训练或checkpoint选择；已被历史讨论使用，不是新的泛化证据。

```bash
bash experiments/hierarchical9_r1_outcome_v1/run.sh summary
bash experiments/hierarchical9_r1_outcome_v1/run.sh pack
```

final.pt保存完整可恢复训练状态；runtime.pt保留相同网络结构、采用现有R012运行时可识别的格式，并明确记载本实验来源，不伪称历史R1。生成后实际调用现有模型加载器，验证输出和输入梯度一致。不因文件名runtime就自动晋升：只有相同solver表现改善，才考虑替换本地R1。

## 范围与预算

增加的rollout和高阶求导有额外计算开销，不称为等算力比较。前10k完全调用原步函数；后40k保留全部400k原查询并增加几何目标。学习率调度仍看原R1目标，不在1963回归集上选模型。首版固定final50k，不做隐含checkpoint挑选。

已实际执行的CPU测试见TEST_RESULTS.md。没有在这里运行真实服务器168分片/完整cache，也没有CUDA/NCCL；真实数据与原runtime一致性检查由提交前入口执行，不能声称四卡训练已通过。
