# VisCDF 25 mm：原始 Q0 采样器，单作业单节点4卡

**集群不支持同时提交两个任务。本入口只提交一个单节点4张3090的作业，不拆成两个任务，不尝试绕过限制。**
此前的双节点双作业说明撤销。默认3090node2，可改为3090node3，但只能选一个。

## 数据与标签不变

本步骤仅重建原始FOV-only边界配置库，不训练网络、不使用CASE日志、不读取模型权重。原data_generator.py及采样、候选筛选、LBFGS/Adam边界提取、最远点下采样函数不修改。

- base_link；位置米、关节弧度；x/y=[-0.8,0.8]m，z=[0,1]m。
- XYZ均25mm，含边界：65×65×41=173225个位置。
- 七关节顺序/限位、八个ToF optical frames、祖先关节mask与原数据和URDF核对。
- FOV保持50°/66°、深度0.2–0.7m、delta=0.01m；不计算LOS。
- 采样量、K=500、边界容差、优化器等从原NPZ元信息继承，不用默认100替换。
- 每点随机输入与下采样种子由全局grid_id确定，不依赖领取顺序；不承诺不同CUDA版本逐位一致。
- 25mm为位置采样间距，不是预测误差或地图体素大小。

## 只提交一个作业

    export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
    NODES=3090node2 bash experiments/hierarchical9_grid25_data_v1/run.sh submit

需要使用3090node3时，仅将上面的3090node2改成3090node3。不要执行两遍。
NODES=3090node2,3090node3或节点范围表达式会在提交前直接拒绝。Shell入口与Python提交器都检查账户队列；已有排队/运行任务时停止，不自动取消、不提交依赖任务、不排第二个作业。队列查询失败也停止。

资源为 --nodes=1 --ntasks=1 --gres=gpu:3090:4 --exclusive。一个Slurm作业内启动四个采集进程，不是四个Slurm作业。分配后检查显存；不杀进程、不降低采样量。3天是作业时限，不是预计耗时。

默认旧数据：
`.../CAREPlanner/src/care_visibility_cdf/data/visibility_yiming_style_grid30_q20000_k500_fovonly.npz`
GRID25_REFERENCE_DATA可以指定路径，但SHA必须匹配已验证原数据。

默认输出：
`.../CAREPlanner/outputs/mainline_b/h9_grid25_data_v1/formal/`
GRID25_ROOT可以指定独立输出根；不覆盖旧数据。新CDF实际训练数据边界仍未由本程序核验。

## 进度、恢复和已准备的数据

    bash experiments/hierarchical9_grid25_data_v1/run.sh status
    bash experiments/hierarchical9_grid25_data_v1/run.sh summary
    bash experiments/hierarchical9_grid25_data_v1/run.sh log
    bash experiments/hierarchical9_grid25_data_v1/run.sh ranks

每64个位置一个原子发布分片，共2707片。四个进程通过文件锁领取全部分片。
已完成分片保留并校验。未找到合格Q0的x/sensor保留valid_fov=False与+inf槽位，不声称物理不可见。

只有旧任务已经退出、账户队列为空且数据未完成时才恢复：

    NODES=3090node2 bash experiments/hierarchical9_grid25_data_v1/run.sh resume

已存在formal/job.json时，新Shell入口复用source.env指向的原冻结采集代码并显式传入单节点，不重写其代码哈希、job.json、随机种子或已完成分片。不要删除formal/source.env，也不要直接调用旧冻结快照的提交入口。

## 完成与下一阶段

一个作业中的四个进程全部结束后自动检查分片和逐点覆盖，再流式ZIP64合并，避免全量np.concatenate占用内存。

    bash experiments/hierarchical9_grid25_data_v1/run.sh verify
    bash experiments/hierarchical9_grid25_data_v1/run.sh summary
    bash experiments/hierarchical9_grid25_data_v1/run.sh pack

最终文件为formal/dataset/visibility_grid25_fovonly.npz。以dataset/complete.json和verify通过为整批完成依据。
K=500的canonical未压缩上界约52.3GiB；分片和合并共存，prepare按两份上界加4GiB检查磁盘。实际压缩大小未知。

本步骤只完成基础Q0库。V4增广、正负pair、训练划分/缓存和保持参考仍需对应新网格重建，不能套旧x_index。本次不改训练loss、模型、线上solver，也不自动启动训练。

原采样测试见TEST_RESULTS.md；此次单作业修正的13项本地CPU/模拟调度器测试见SINGLE_JOB_TESTS.md。真实HPC/CUDA采集尚未在本地运行。
