# 单作业修正测试记录

2026-10-30实际执行：

    PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s . -p test_single_job.py -v

13/13通过，零跳过，约12.8秒；bash -n通过。

测试使用真实临时Git仓库、Shell入口、模拟squeue和模拟冻结采集器；Python调度函数从实际run.py的AST提取后以模拟调度器执行。未使用GPU或真实Slurm，不声称已在服务器提交作业。

覆盖默认单节点、显式3090node3、两个节点/hostlist拒绝、账户RUNNING/PENDING/COMPLETING阻止新提交、队列查询失败时停止、Python入口仅一次sbatch、既有source.env/job.json/分片逐字节保持、缺少冻结来源时拒绝、新任务源码快照、未准备的resume拒绝、prepare不提交和Shell语法。

本次只修改调度入口及文档，新增调度测试；采集算法、protocol.json、merge.py、g25_common.py、collect.py均未修改。原20项采样/合并测试记录保留在TEST_RESULTS.md；本次没有在该环境重新运行完整原20项，也没有执行真实HPC数据预检、真实URDF或CUDA采集。生产入口仍执行目录内全部测试后才进入数据预检。
