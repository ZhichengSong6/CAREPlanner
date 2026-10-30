# PIQP 有限约束索引修复

固定上游提交为 `de79713d37fb38560a187ac0c3e1b034f35a13fb`。修复避免读取有限上下界索引数组未初始化的尾部，防止无穷约束在对偶恢复中产生无效运算。

`manifest.json` 保存原始和修复后头文件SHA；`finite-bound-indices.patch` 保存完整补丁。`scripts/prepare_piqp.py` 复制头文件到构建目录并应用补丁。主工程CMake使用该副本，上游submodule保持原始提交。

现有回归为 `scripts/test_phase_e_piqp_finite_bounds.cpp`，覆盖上下界、复用求解器、未使用索引槽，以及真实不可行情形。
