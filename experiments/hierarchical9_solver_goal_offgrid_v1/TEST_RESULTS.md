# Executed local verification

Environment: Python 3.13.5, torch 2.10.0+cpu, NumPy 2.3.5, SciPy 1.17.0.

- **18/18 CPU unit tests passed, zero skips.** Includes fixed 512 off-grid selection, exact local +/- construction on controlled geometry, actual SLSQP feasibility search, unresolved-vs-infeasible semantics, no-overwrite/hash rejection, atomic input freeze with unresolved x retained, full 28,672-case synthetic report, x-cluster bootstrap, corruption rejection, and frozen actual R1 architecture input gradients without changing parameters.
- The actual R1 architecture source used in the test matches Git blob `0b9dab1986114d4b95b33bb0fa01fd8e0c68e53b`.
- Python compilation and `bash -n` passed.
- Separate shell lifecycle harness passed: four-GPU resource arguments, duplicate submission refusal, failed-job resume, completed-job refusal, Slurm-spooled script resolving absolute frozen source, and archive contents plus full read/CRC check.
- The lifecycle harness STUBS Slurm and real-data preflight; it does not claim a real job ran.

## Not executed here

No access to the HPC's two trained checkpoint files or complete cache in this local runtime. No real URDF parser/FK pipeline, CUDA solver, Slurm allocation, or full off-grid result has run here. The server entrypoint performs actual-input geometry/Jacobian checks before requesting GPU resources. A CPU test pass is not a CUDA/NCCL or scientific performance result.

## Unit-test output

```text
test_01_frozen_protocol_and_maximum_scale (test_offgrid.Tests.test_01_frozen_protocol_and_maximum_scale) ... ok
test_02_offgrid_sampling_is_reproducible_after_float32 (test_offgrid.Tests.test_02_offgrid_sampling_is_reproducible_after_float32) ... ok
test_03_witness_search_uses_feasibility_not_solver_status (test_offgrid.Tests.test_03_witness_search_uses_feasibility_not_solver_status) ... ok
test_04_bounded_failed_witness_is_unresolved_not_infeasible (test_offgrid.Tests.test_04_bounded_failed_witness_is_unresolved_not_infeasible) ... ok
test_05_local_pairs_are_new_geometry_checked_exact_offsets (test_offgrid.Tests.test_05_local_pairs_are_new_geometry_checked_exact_offsets) ... ok
test_06_local_joint_limit_excludes_without_replacement (test_offgrid.Tests.test_06_local_joint_limit_excludes_without_replacement) ... ok
test_07_atomic_no_overwrite_and_stage_hash_rejection (test_offgrid.Tests.test_07_atomic_no_overwrite_and_stage_hash_rejection) ... ok
test_08_full_input_freeze_records_all_unresolved_and_never_uses_models (test_offgrid.Tests.test_08_full_input_freeze_records_all_unresolved_and_never_uses_models) ... [FROZEN] x=4 starts=168 sha=83dc2bfadf7f5b8a01d75af1d4ba06e86ecd2c6857423e8380ff4a6662dcda77; model loading begins only after this gate
ok
test_09_report_does_not_require_historical_1291_success_count (test_offgrid.Tests.test_09_report_does_not_require_historical_1291_success_count) ... ok
test_10_case_corruption_and_nonfinite_results_rejected (test_offgrid.Tests.test_10_case_corruption_and_nonfinite_results_rejected) ... ok
test_11_cluster_bootstrap_uses_x_not_cases (test_offgrid.Tests.test_11_cluster_bootstrap_uses_x_not_cases) ... ok
test_12_full_512_x_report_scale_and_sensor_counts (test_offgrid.Tests.test_12_full_512_x_report_scale_and_sensor_counts) ... ok
test_13_seedbank_uses_old_data_as_guesses_and_is_readonly (test_offgrid.Tests.test_13_seedbank_uses_old_data_as_guesses_and_is_readonly) ... ok
test_14_real_r1_model_frozen_still_has_input_gradients (test_offgrid.Tests.test_14_real_r1_model_frozen_still_has_input_gradients) ... ok
test_15_syntax_and_shell_parser (test_offgrid.Tests.test_15_syntax_and_shell_parser) ... ok
test_16_git_blob_and_sha256_are_different (test_offgrid.Tests.test_16_git_blob_and_sha256_are_different) ... ok
test_17_empty_subgroup_reports_NA_not_zero (test_offgrid.Tests.test_17_empty_subgroup_reports_NA_not_zero) ... ok
test_18_freeze_failure_does_not_load_models (test_offgrid.Tests.test_18_freeze_failure_does_not_load_models) ... ok

----------------------------------------------------------------------
Ran 18 tests in 2.689s

OK
```

## Shell-harness output

```text
PASS: submit resources, duplicate protection, failed-job resume, completed-job refusal, absolute Slurm-spool source, and pack contents/CRC.
This uses stubbed Slurm and preflight, NOT actual GPU/HPC execution.
```
