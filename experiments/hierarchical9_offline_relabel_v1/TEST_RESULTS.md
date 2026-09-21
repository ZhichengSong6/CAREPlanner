# 交付前测试记录

## 实际执行

- Python: 3.13.5
- NumPy: 2.3.5
- SciPy: 1.17.0
- PyTorch: 2.10.0+cpu
- CUDA available: False
- `python -m unittest -v test_labels.py`: **18 tests PASS**。
- `python -m compileall -q .`: PASS。
- CLI help: 主命令、prepare、preflight、worker、audit 均可解析。
- `bash -n run.sh submit.sh worker.sbatch`: 分别检查通过。

## 覆盖范围

平面稀疏库例子、正负号、边界零值及法向、曲面最近点、多个等距最近点、
所有 FOV 不等式的共同可行性、有效关节 mask、不改变原查询、无可行边界、
近边界符号保护、关节限位最近点、float64 FK 数值及解析梯度、
旧标签 floor/符号跳变、确定性查询与原始 split、缺失 sensor 的 union 屏蔽、
分片恢复/完整汇总、旧新 reader 的配对身份、文件损坏检测、高预算抽查的不可覆盖性。

## 没有执行

**用户真实 7GB 级 NPZ、真实 Arm.urdf、服务器环境、CUDA 多卡标注和 Slurm 提交均 NOT_RUN。**
本环境没有这些数据和 GPU，不能把合成 fixture 的 PASS 写成真实机械臂预检已通过。
包内真实 preflight 会在服务器运行时强制检查几何、Jacobian、旧标签 parity、mask 和限位。

没有进行 R1 推理、网络训练、求解成功率比较、ROS/Gazebo、碰撞或执行认证。

## 原始测试输出（合成数据）

```text
test_all_other_fov_constraints_enforced (test_labels.CoreTests.test_all_other_fov_constraints_enforced) ... ok
test_ambiguous_nearest_gradient_masked (test_labels.CoreTests.test_ambiguous_nearest_gradient_masked) ... ok
test_curved_boundary_projection (test_labels.CoreTests.test_curved_boundary_projection) ... ok
test_fk64_dtype_and_analytic_derivative (test_labels.CoreTests.test_fk64_dtype_and_analytic_derivative) ... ok
test_inside_sign (test_labels.CoreTests.test_inside_sign) ... ok
test_mask_keeps_inactive_joint (test_labels.CoreTests.test_mask_keeps_inactive_joint) ... ok
test_near_boundary_sign_guard (test_labels.CoreTests.test_near_boundary_sign_guard) ... ok
test_nearest_at_joint_limit_masks_gradient (test_labels.CoreTests.test_nearest_at_joint_limit_masks_gradient) ... ok
test_no_boundary_failure_not_zero (test_labels.CoreTests.test_no_boundary_failure_not_zero) ... ok
test_no_query_clamping (test_labels.CoreTests.test_no_query_clamping) ... ok
test_old_distance_continuous_sign_jump (test_labels.CoreTests.test_old_distance_continuous_sign_jump) ... ok
test_old_floor_and_empty (test_labels.CoreTests.test_old_floor_and_empty) ... ok
test_sparse_bank_example (test_labels.CoreTests.test_sparse_bank_example) ... ok
test_zero_boundary_normal_not_zero (test_labels.CoreTests.test_zero_boundary_normal_not_zero) ... ok
test_missing_sensor_never_creates_union_winner (test_labels.PipelineTests.test_missing_sensor_never_creates_union_winner) ... ok
test_prepare_reproducibility_original_split_and_no_overwrite (test_labels.PipelineTests.test_prepare_reproducibility_original_split_and_no_overwrite) ... ok
test_required_metadata_not_guessed (test_labels.PipelineTests.test_required_metadata_not_guessed) ... ok
test_resume_merge_cache_reader_and_corruption (test_labels.PipelineTests.test_resume_merge_cache_reader_and_corruption) ... [preflight] PASS /tmp/tmp15arz4c4/job/preflight.json
[query] id=0 supported=8 new_value_valid=8 new_grad_valid=0
[shard] COMPLETE 0 rows=1
[query] id=1 supported=8 new_value_valid=8 new_grad_valid=8
[shard] COMPLETE 1 rows=1
[resume] verified shard=0
[merge] COMPLETE /tmp/tmp15arz4c4/job/label_summary.md
[audit] query=1 sensor=5 materially_closer=False
[audit] query=0 sensor=5 materially_closer=False
[audit] /tmp/tmp15arz4c4/job/high_budget_audit_seed77491.json
ok

----------------------------------------------------------------------
Ran 18 tests in 0.168s

OK
```
