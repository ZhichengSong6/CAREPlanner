# v3 批量标签交付测试

## 实际执行

环境：Python 3.13.5、NumPy 2.3.5、SciPy 1.17.0、PyTorch 2.10.0+cpu；无CUDA。

- v3 `unittest discover -s .../hierarchical9_offline_relabel_v3 -p test_batch.py -v`：20 tests PASS。
- 原 v1 的18个合成测试 PASS；v1文件未改。
- v2 verified_core.py 与固定Git blob `87d52825d7ab0abc2e13262c911879c962f60703` 完全一致；未重写其优化/差分算法。
- v1八个Python文件SHA256与用户已完成smoke manifest中的源码身份一致。
- Python编译、CLI --help、Bash/Sbatch语法检查通过。

覆盖：确定性查询、原split和旧诊断点排除、sensor支持覆盖、实际符号分层审计、
抽查计划不读取优化成功状态、无标签时不补采、不可覆盖源数据/原bank、
逐任务原子保存、锁、损坏检查、输入/配置/源码身份、resume不重复求解、
基础标签梯度默认NOT_RUN、真实执行差分后才设置PASS（合成几何）、
两阶段完整性及配对reader、缺sensor的union屏蔽、winner并列梯度屏蔽、
零抽查不自动启用候选梯度、依赖失败不提交、同输出活动任务拒绝重复提交、
旧OUT不会被使用、watch只读。Slurm相关为mock测试，不是实际调度。

## 未执行 / 不声称完成

未在本环境执行真实80个空间点、最多320条查询的批量标注。
未执行服务器CUDA/Slurm新任务。未测得本轮真实标注有效率、耗时或任何提速。
未生成真实paired_cache；未训练R1；未修改runtime、URDF、FOV阈值、原数据或权重。
本轮没有再次运行既有v2六案例；用户已提供的v2结果只作为方法与接口依据。
局部优化与有限差分通过均不等于全局最近距离证明。
