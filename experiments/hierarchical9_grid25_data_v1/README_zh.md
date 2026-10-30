# VisCDF 25 mm：复用原始 Q0 采样器，4卡/8卡分片采集

第一步只重建原始 FOV-only 边界配置库，不训练网络，不使用任何CASE成功/失败日志，不读取模型权重。
原 `src/care_visibility_cdf/scripts/data_generator.py` 及其采样、候选筛选、LBFGS/Adam边界提取、最远点下采样函数完全不改；新入口调用这些旧函数。

## 唯一数据域调整

- base_link，位置米、关节弧度；x/y=[-0.8,0.8]m，z=[0,1]m。
- 整数毫米构网格后转FP32，XYZ均25mm，含边界：65×65×41=173225位置。
- 七关节顺序、上下限、八个ToF optical frames及祖先关节mask与现有URDF和旧数据逐一检查。
- FOV仍为水平50°/垂直66°、传感器深度0.2–0.7m、delta=0.01m，不计算LOS。
- q_init_per_p、optimize_top_k_per_sensor、q0_per_sensor、epsilon、边界优化器、迭代数、lr、lbfgs_lr、seed从原始NPZ元数据继承，不用脚本默认100去替换原K=500。
- epsilon仍为原始边界采集验收阈值，不宣称精确根或25mm预测误差。没有收紧标签定义或改变关节距离。
- RNG改为按全局grid_id独立确定；下采样seed按全局索引计算，4/8卡和恢复的任务分配不会改变同一点的随机输入。同设备/软件栈应一致，不承诺跨CUDA版本逐位相同。

## Git分支拉取后：只选择一个提交方式

8卡（推荐可用两节点时）：

    export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
    NODES=3090node2,3090node3 bash experiments/hierarchical9_grid25_data_v1/run.sh submit

4卡：

    NODES=3090node2 bash experiments/hierarchical9_grid25_data_v1/run.sh submit

也可指定3090node3。8卡是两个 `--nodes=1 --gres=gpu:3090:4 --exclusive` 独立作业，不是一个跨节点8卡DDP；避免此前单作业节点数限制。是否能同时运行受当前Slurm账户/资源/分区策略控制，不是保证。两个节点必须能读写同一个共享输出目录，并支持跨主机文件锁。

默认旧数据：
`.../CAREPlanner/src/care_visibility_cdf/data/visibility_yiming_style_grid30_q20000_k500_fovonly.npz`
允许GRID25_REFERENCE_DATA更换路径，但文件SHA必须与已验证原数据一致。只读取元信息并校验整文件SHA，不继承旧x_index标签。

默认输出：
`.../CAREPlanner/outputs/mainline_b/h9_grid25_data_v1/formal/`
可用GRID25_ROOT指定全新输出根，不覆盖任何旧结果。原始CDF完整数据的实际边界并未被本代码核验；这里实现的是已约定目标数据域。

提交前执行CPU单元测试，检查原始NPZ及采样器SHA，URDF限位/传感器链和小批原始Torch/NumPy几何一致性。没有独立GPU pilot。实际分配后打印GPU占用并要求四卡每卡至少6GiB空闲；不杀进程，不偷偷缩减采样量。Slurm默认3天是作业时限，不是完成时间估计。

## 任务、断点与资源

每64个空间点一个不可变分片，共2707片。两个节点的每张卡独立计算原单点采样器，通过共享文件锁领取分片；先启动的节点可以独立完成整个任务，不等待另一节点。中断只损失未发布分片，完成片哈希校验后复用。随机输入不依赖领取顺序。

    bash experiments/hierarchical9_grid25_data_v1/run.sh status
    bash experiments/hierarchical9_grid25_data_v1/run.sh summary
    bash experiments/hierarchical9_grid25_data_v1/run.sh log
    bash experiments/hierarchical9_grid25_data_v1/run.sh ranks

只有所有已提交作业退出后才能恢复，4/8卡可互换；不要删除formal目录：

    NODES=3090node2,3090node3 bash experiments/hierarchical9_grid25_data_v1/run.sh resume

未找到边界记录的x/sensor也保留，valid_fov=False和+inf槽位，不声明物理不可见。保留所有173225个空间索引；本步骤不依据任何模型筛选数据，也不宣称已经建立新的训练/验证隔离协议。

分片保存原训练器使用的canonical x/q/k/valid_fov/mask等布局及g、sensor_margins、active_planes和计数，不保存原来的冗余转置调试副本。只有has_occlusion_labels=False，不伪造遮挡认证。

## 自动合并与验收

最后完成的节点自动核验2707片、逐点覆盖和完整保留的Q0，再流式ZIP64合并；不调用旧的全量np.concatenate合并器，避免把全网格和冗余数组一次放进RAM。

    bash experiments/hierarchical9_grid25_data_v1/run.sh verify
    bash experiments/hierarchical9_grid25_data_v1/run.sh summary
    bash experiments/hierarchical9_grid25_data_v1/run.sh pack

最终文件：
`formal/dataset/visibility_grid25_fovonly.npz`
`formal/dataset/complete.json`

summary在采集中也可以运行。Slurm COMPLETED仅表示某个节点任务退出；以dataset/complete.json及verify通过为整批完成依据。两节点中一个先完成而另一个仍工作时，可以暂时打印NOT COMPLETE，这是正常的。

完整K=500数组的canonical未压缩上界约52.3GiB，分片与合并同时保留需要额外空间；prepare采用两份未压缩上界加4GiB的保守磁盘准入。实际压缩文件大小未知，不按压缩率做无证据的保证。训练时直接加载完整q数组也会增大主机RAM需求，旧训练入口不能盲目换路径后立刻开跑。

**本次只完成原Q0库这一步。**旧的V4零边界增广、正负pair cache、train/val划分、成功保持参考库依赖旧x_index，必须在新网格上重建。它们不能直接复用或覆盖。本代码不修改训练loss、50:50比例、模型架构、checkpoint或线上solver；不自动启动下一轮训练。

25mm是位置采样间距，不是网络精度或线上地图体素。所有参考数据只读，诊断归档不包含大NPZ或权重。
