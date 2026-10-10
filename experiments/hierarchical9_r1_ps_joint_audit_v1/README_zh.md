# R1 / Paired-Slope 合并只读评估 v1

## 本次做什么

一个四卡 Slurm 作业同时完成三部分：

1. **全部 1963 个已冻结 starts 的等 solver 回归。** 原 R1 与 Paired-Slope best/final 使用完全相同的原 runtime_probe，包括 epsilon_f=0.03。没有阈值扫描。best/final 都校验独立文件哈希、checkpoint/run 元数据；仅当 model_state 逐张量内容哈希相同，才合并其计算，并在报告中保留两个文件名的对应关系。
2. **同点局部场及独立几何检查。** 从原始168个 V4 分片中，按 train/val × sensor × radius 固定随机优先级选取最多128对/单元（默认6144对）。不按模型预测筛选。全部原始分片校验 SHA256；所选记录逐项验证与训练 pair cache、V4 cache、boundary cache 的精确对应。重新用原 FOV oracle 检查符号、活跃面、法向、固定活跃面的双尺度有限差分；以相同点比较 R1/PS 的零值、斜率、符号、梯度及固定长度方向探针。**几何检查是预定子集，不是对全部V4做全局最近距离证明。**
3. **真实 checkpoint 的参数梯度诊断。** 四个确定性 train batch，逐一拼回训练时四个rank的全局抽样，调用原Paired-Slope objective。逐损失计算共享层、union、S0–S7私有层的梯度范数和方向关系，验证各分项梯度之和复现实际总梯度。记录当前裁剪条件，但不执行梯度裁剪更新或 Adam.step。R1也使用当前PS objective作诊断，**不冒充R1历史objective**。只报告原始梯度关系，不把负余弦直接判为病因或Adam实际更新效果。

附带记录所选点上的ReLU神经元活跃情况；“这些点上未激活”不是“神经元全局死亡”的证明。

## 不会做什么

不训练；不生成训练标签；不更换checkpoint；不写回旧cache或原始V3/V4；不改原solver、GCDF/VBC/LOS和真实执行流程。FD与方向探针产生的配置仅用于诊断，不作为新训练样本。评估前后再次核验模型参数及两个缓存哈希。原始大型q0 bank只记录并复核大小/mtime，不宣称重新完成其SHA验证。

本实验只支持已确认的R1和job17460 checkpoint。最终PS文件哈希为4872dc937224a1c20962d03388809e5f8e8436f90e0d2f00516f0b7bfda84dee。依赖接口对应仓库基线7a8eb246bb76feda2f3a0f4fe486c3e986e23633；关键上游源码blob哈希被固定，其他依赖提交时全量记录并在作业前后复查。

## 使用

在 mainline-b/h9-scratch50k-r012-v1 分支应用并提交本目录后，仓库根目录执行：

```bash
export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
env -u NODE bash experiments/hierarchical9_r1_ps_joint_audit_v1/run.sh submit
```

默认申请单节点四张3090（排除3090node1），16CPU。提交前在CPU执行静态/回归测试及真实文件预检；不是额外的GPU smoke/pilot。Slurm启动脚本明确使用冻结的绝对源码路径，不从spool寻找Python脚本。

只读查看，不要为了看输出重复submit：

```bash
bash experiments/hierarchical9_r1_ps_joint_audit_v1/run.sh status
bash experiments/hierarchical9_r1_ps_joint_audit_v1/run.sh log
bash experiments/hierarchical9_r1_ps_joint_audit_v1/run.sh ranks
bash experiments/hierarchical9_r1_ps_joint_audit_v1/run.sh follow
bash experiments/hierarchical9_r1_ps_joint_audit_v1/run.sh summary
```

默认输出根目录：

```text
/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_ps_joint_audit_v1/
```

每次提交创建独立run目录；具体job state保存在job_JOBID.env。可通过JEA_STATE显式选择历史任务，避免latest指向新的任务。

完成后执行 `run.sh pack`，一次创建review tar.gz（不向压缩tar追加），仅含评估记录和代码，不含原模型或大缓存。

## 结果怎么看

summary.md 给出相同solver下的local/uniform/all、paired得失、逐sensor同点指标；report.json保存完整分支、几何标记、4组梯度矩阵和活动统计。逐条solver和boundary记录、抽样来源/索引及全部trace也保留。

R1必须复现618/683 local、673/1280 uniform和1291/1963 all，才会完成基线门禁。复现失败会保留报告并标记BASELINE_MISMATCH，而不是假报COMPLETE。几何发现会独立标为DATA_REVIEW_REQUIRED；成功跑完也永远不自动promote模型。

这1963个starts已经参与过方案设计，只是回归集。paired sign p是描述性结果，没有校正同一x下多个start的相关性。完整泛化结论还需要未用于选择的测试集。所有FOV结果均不是执行、安全或实际可见认证。

## 本地检查

```bash
JEA_TEST_REPO="$PWD" python -m unittest discover \
  -s experiments/hierarchical9_r1_ps_joint_audit_v1 -p 'test_*.py' -v
```

单测涵盖原始value/grad字段、原始记录到三个cache的精确映射、数据不可变、分层抽样、合并完整性、NaN记录、法向FD、模型参数不更新、实际R1结构反传、梯度分解，以及Slurm spool路径与失败rank回收。CPU测试不等于在用户真实模型/168分片/CUDA/NCCL上已通过。
