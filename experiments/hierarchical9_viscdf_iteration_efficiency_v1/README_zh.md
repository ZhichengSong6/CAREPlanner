# R1 / B_keep：迭代引导效率 v1

**Uniform 是起点分布，不是一次梯度下降。**已有 Uniform 分数来自随机不可见关节配置上的完整多步 solver。本目录让两个概念分开：行/分组为 Local、Uniform；预算为 K=1、2、3、5、10，另有完整原 solver 的 FULL 对照。

只新增评估代码；不训练，不修改任何旧实验、数据、权重、缓存、运行时或安全链。比较原 R1 与正式验收的 B_keep/runtime.pt。教师、A组和Parent无需在线参与。

## 预先固定的语义

K 是允许的 branch_value_and_grad 调用次数，也就是梯度指导的投影或上升更新尝试预算。若梯度退化或关节限位导致零位移，该次查询仍消耗预算；实际次数、零位移次数另报。它不是 K 次完整 solver，也不是 K 次网络前向。

继承原 runtime 的 _optimize_branch、_projection_step、_ascent_step、_refine_branch_root 和 _clamp。只在下一次梯度查询会超预算时停止，并返回这次请求所对应的当前配置。若原算法提前完成，返回原算法按预测值规则选择的候选。没有用真实 FOV 判断决定提前停下，没有事后挑真实几何最好的点，没有重新随机启动。

**值查询的二分细化保留。**二分不调用梯度，但会改变根候选并消耗值查询及时间；因此 K=1 不能表述成严格仅一次计算或一次位置变换。CSV 记录 gradient_calls、value_calls、root_value_calls。论文横轴建议写“梯度查询/更新尝试预算”，并在表注说明二分预算（最多12）不变。

投影/上升参数和原来的提前停止、回退规则不变。FULL 是原 solver 的总预算（最多10次投影、根后1次上升或最多8次回退上升），不是把它称为一步。

每个 K 从相同原始起点独立运行。每个模型/案例都测所有预算；不根据某模型先前是否成功筛选样本。预算运行次序和模型次序轮转，预热不计入测量；权重在前后做参数哈希，拒绝隐式改变。

## 输入来源及结论范围

默认读取：

    .../outputs/mainline_b/h9_solver_goal_keep_v2/formal/

必须包含已完成的 confirmation/inputs、confirmation/evaluation、confirmation/report、dependencies，以及 B_keep/runtime.pt。原 R1 路径由当时的 job.json 读取。

使用已经冻结的 512 个新非网格点、22,518 个起点，starts SHA256：

    6d653a48f1e00805b64108a72a4d799153c5c476c3b55346c76230f326e451cc

这是**复用过的测试集上的冻结收敛特征评测**，不是新生成的、未见过的最终测试集。Local 和 Uniform 都报告；Uniform 不是本地框架完整任务成功率。

R1 SHA256：

    4f395926fa79c29474be8748cef4733ec400d155cd8fadb76c632c2838864002

B_keep/runtime.pt SHA256：

    45730438eb9cdc29418a86e0d2f933595ce0626b4171b19144145141d72c2a93

加载时核验 B 的真实 70k/额外20k 微调来源，strict=True 加载原 R1 架构；不篡改成50k scratch，不修改旧线上加载器。

## 一次提交

仓库原分支拉取后：

    export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
    env -u NODE bash experiments/hierarchical9_viscdf_iteration_efficiency_v1/run.sh submit

先执行 CPU 单元测试、原完整结果逐例对齐、权重和依赖哈希检查，以及真实权重/URDF上的只读 CPU 预算检查。通过后提交一个单节点4张3090的评估作业，默认排除3090node1，申请 --exclusive。分配后检查每卡至少6 GiB全局空闲显存；失败就报错，不杀进程、不改参数。6 GiB是准入门槛，不保证峰值内存。

无单独GPU pilot。Slurm默认12小时为时限，不是预估耗时。输出：

    .../outputs/mainline_b/h9_viscdf_iteration_efficiency_v1/formal/

可使用 IE_ROOT 修改新输出根、K2_ROOT 指向已完成的Keep-v2根。旧实验数据只读，依赖复制到新目录；Slurm spool 不用于查找源码。

    bash experiments/hierarchical9_viscdf_iteration_efficiency_v1/run.sh status
    bash experiments/hierarchical9_viscdf_iteration_efficiency_v1/run.sh log
    bash experiments/hierarchical9_viscdf_iteration_efficiency_v1/run.sh ranks

只有旧任务已退出、评估未完成时才用 resume。已完成并哈希验证的分片不重跑；未完成分片重新评估。不得删除旧结果来调参重测。

    bash experiments/hierarchical9_viscdf_iteration_efficiency_v1/run.sh resume

## 结果

    bash experiments/hierarchical9_viscdf_iteration_efficiency_v1/run.sh summary
    bash experiments/hierarchical9_viscdf_iteration_efficiency_v1/run.sh verify
    bash experiments/hierarchical9_viscdf_iteration_efficiency_v1/run.sh pack

report/ 内：summary.md、report.json、curve.csv，以及安装 matplotlib 时自动生成的 success_local.svg、success_uniform.svg。SVG为矢量图，原始CSV可以直接用于论文作图。

主表报告所有案例中的返回候选成功率、B−R1成对新增/丢失和按空间点分组95%区间、解析FOV残余违约 max(0,-g) 的均值/p50/p95、完整的本次带记录开销的求解耗时、梯度/值/二分查询数、多个裕量门槛。失败案例不删除。

FULL 必须逐案例复现上轮候选配置（数值容差）、FOV成功、根来源和调用次数；否则停止，不给出看似正常的报告。FULL 不是1、2、3、5、10中的一个固定次数，图上不把它当数值横坐标。

首次访问合格状态只作 first_hit_diagnostic 附加统计，未成功记为 null，不混入平均0步。只记录实际更新后的状态和二分最终选中的根，不把二分中所有试探点当作已采用的轨迹。主结果不以这项诊断选择候选；返回候选成功率不保证随K单调。

计时含原求解器及少量记录开销、值查询二分，不含后验解析检查和加载。不是单次网络前向延迟，也不是四卡吞吐量换算的延迟。此处“可见”仅指目标进入指定sensor的解析FOV，不是无遮挡、碰撞安全、轨迹可执行或真实传感器已观测。

归档含源码、协议、CSV/SVG、逐例结果和日志；不包含.pt权重或原始大数据。原R1、B、旧结果均不覆盖，也不自动晋升checkpoint。
