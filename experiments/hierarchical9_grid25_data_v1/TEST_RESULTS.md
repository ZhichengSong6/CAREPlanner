# 本地执行记录

2026-10-30，在隔离目录运行：

    PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s . -p 'test_*.py' -v

**20/20通过，零跳过**，最后一轮约17.3秒。另运行全部Python语法检查及bash -n。

覆盖173225点/25mm坐标、2707片无遗漏、全局ID种子与下采样公式、继承旧K=500而非默认100、缺失/冲突元信息拒绝、跨进程锁、原子发布和损坏检测、未确认sensor保留、流式NPZ各字段回读、原R1数据读取器兼容、最终合并与CRC、单节点4卡/双独立作业、重复提交阻断和Slurm spool路径。

实际调用原始data_generator/torch FOV/边界优化函数，在合成运动学链上做了一个CPU分片采集，保留32个Q0，并通过几何重验。没有在本环境访问HPC的7.8GB旧NPZ、真实URDF依赖或CUDA。环境没有urdf_parser_py，测试只对未使用的URDF导入类做shim；原Torch几何和优化函数并未替换。生产prepare不会安装shim，必须加载真实URDF并做独立NumPy/Torch FOV一致性检查。

调度器相关测试是命令构造、模拟状态和spool执行，不代表真实Slurm允许并行运行两个独占作业。跨节点共享文件锁、真实GPU和全量采集仍须在服务器执行验证。CPU通过不代表已采完真实数据。
