# v2 交付前测试

## 本次实际运行

环境：Python 3.13.5、NumPy 2.3.5、SciPy 1.17.0、PyTorch 2.10.0+cpu；无 CUDA。

- 原 v1：18 个 CPU 合成测试全部 PASS，源码未修改。
- v2：26 个测试全部 PASS。
- Python compileall、CLI --help、所有 Bash/Sbatch 语法检查 PASS。
- 用户上传的真实 smoke 归档：8个query的分片/审计/输入一致性与18个源文件哈希检查通过。
- 真实 query 4/S0 的两条候选用于 selector 数值等价回归；成功候选按新规则得到选择。
- 源不可覆盖、完整case恢复不重写、校验失败停止、mock提交仅一次、watch不提交、依赖失败不提交均测试通过。

## v2几何测试覆盖

平面、稀疏库、内外符号、光滑零点、非零距离角点、投影到关节限位、曲面、多最近点、
同一sensor全部FOV不等式、inactive joint保持、原始query不clamp、近零符号保护、
零距离角点拒绝、query自身限位双侧差分拒绝、故意反向梯度拒绝、对称差分掩盖尖点的单侧检查、
扰动发现更近中心候选时值/梯度保持无效、数值等价聚类不跨不同端点、不链式扩张、
未收敛竞争解保守标注为uncertainty、真实记录的等价候选修复。

## 没有运行 / 不声称完成

- 未运行真实 Arm.urdf 的新距离扰动验证（交付环境无用户完整仓库/原始bank）。
- 未运行服务器CUDA/Slurm实际作业；mock测试不是实际调度。
- 未取得S4/S6/S7新梯度通过率，不预测会恢复多少条。
- 未生成大规模v2训练集、未训练R1、未改runtime/URDF/safety/checkpoints。
- 合成或有限差分PASS都不是全局最近距离证明。

用户应运行独立v2固定案例复核，并返回 verification_summary、preflight和case结果。
