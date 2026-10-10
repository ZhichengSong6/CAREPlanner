# Validation record (local CPU, not the user's GPU job)

- 23/23 CPU unit/integration checks executed; 0 skipped.
- Python: 3.13.5; PyTorch: 2.10.0+cpu; NumPy: 2.3.5; SciPy: 1.17.0.
- CUDA available here: False.
- All package Python files compiled in memory; every shell file passed `bash -n`.
- Used the exact R1 model source and PS reader/objective blobs from the reviewed repository.
- Cache/geometry/checkpoint tests use synthetic fixtures. The 1963-row merger test uses a synthetic table with the known baseline counts; it is NOT a real solver evaluation.
- Actual R1 model input-derivative/backward check passed, without parameter updates.
- Actual PS objective's component gradients sum to its full gradient on a synthetic complete batch.
- Slurm spool-path and failed-child cleanup were simulated locally, not on a Slurm cluster.
- User checkpoint files, full 168-shard production data and CUDA runtime: NOT_RUN locally.
- GitHub publication: NOT_PERFORMED in this session (available connector is read-only).

## Executed output

```text
test_alias_context_restores_imports (test_audit.CoreTests.test_alias_context_restores_imports) ... ok
test_bad_normal_is_flagged_not_silently_fixed (test_audit.CoreTests.test_bad_normal_is_flagged_not_silently_fixed) ... ok
test_checkpoint_dedup_uses_actual_weights_and_rejects_bad_guard (test_audit.CoreTests.test_checkpoint_dedup_uses_actual_weights_and_rejects_bad_guard) ... ok
test_exact_p_no_large_integer_overflow (test_audit.CoreTests.test_exact_p_no_large_integer_overflow) ... ok
test_failed_rank_does_not_leave_other_workers_running (test_audit.CoreTests.test_failed_rank_does_not_leave_other_workers_running) ... [prepare] verify existing caches and match original pairs; no labels generated
===== rank0: last 50 log lines =====
fake rank 0
===== rank1: last 50 log lines =====
fake rank 1
===== rank2: last 50 log lines =====
fake rank 2
===== rank3: last 50 log lines =====
fake rank 3
ok
test_full_1963_merge_and_baseline_gate (test_audit.CoreTests.test_full_1963_merge_and_baseline_gate) ... ok
test_independent_plane_fd_and_signs (test_audit.CoreTests.test_independent_plane_fd_and_signs) ... ok
test_json_nonfinite_is_null_not_fabricated_zero (test_audit.CoreTests.test_json_nonfinite_is_null_not_fabricated_zero) ... ok
test_known_field_metrics_and_no_weight_change (test_audit.CoreTests.test_known_field_metrics_and_no_weight_change) ... ok
test_merge_duplicate_or_missing_ids_rejected (test_audit.CoreTests.test_merge_duplicate_or_missing_ids_rejected) ... ok
test_model_content_digest_ignores_serialization_container (test_audit.CoreTests.test_model_content_digest_ignores_serialization_container) ... ok
test_output_may_not_overlap_source (test_audit.CoreTests.test_output_may_not_overlap_source) ... ok
test_priority_sampler_streaming_and_seed (test_audit.CoreTests.test_priority_sampler_streaming_and_seed) ... ok
test_private_parameter_partition_complete (test_audit.CoreTests.test_private_parameter_partition_complete) ... ok
test_slurm_spooled_worker_uses_pinned_absolute_source (test_audit.CoreTests.test_slurm_spooled_worker_uses_pinned_absolute_source) ... ok
test_urdf_ancestry_order (test_audit.CoreTests.test_urdf_ancestry_order) ... ok
test_zero_gradient_cosine_is_missing_not_zero (test_audit.CoreTests.test_zero_gradient_cosine_is_missing_not_zero) ... ok
test_zero_gradient_explicit_status (test_audit.CoreTests.test_zero_gradient_explicit_status) ... ok
test_actual_R1_model_input_gradient_and_parameter_backward (test_audit.UpstreamTests.test_actual_R1_model_input_gradient_and_parameter_backward) ... ok
test_current_objective_gradient_decomposition_read_only (test_audit.UpstreamTests.test_current_objective_gradient_decomposition_read_only) ... ok
test_modified_original_shard_rejected_before_selection (test_audit.UpstreamTests.test_modified_original_shard_rejected_before_selection) ... [pairs] pass=1/2 shards=2/2
[pairs] pass=2/2 shards=2/2
[pairs] complete {'train': 24, 'val': 24}
ok
test_real_source_schema_and_all_cache_links (test_audit.UpstreamTests.test_real_source_schema_and_all_cache_links) ... [pairs] pass=1/2 shards=2/2
[pairs] pass=2/2 shards=2/2
[pairs] complete {'train': 24, 'val': 24}
[prepare] original V4 shards verified 2/2
ok
test_wrong_production_alias_fields_rejected (test_audit.UpstreamTests.test_wrong_production_alias_fields_rejected) ... [pairs] pass=1/2 shards=2/2
[pairs] pass=2/2 shards=2/2
[pairs] complete {'train': 24, 'val': 24}
ok

----------------------------------------------------------------------
Ran 23 tests in 2.144s

OK
```
