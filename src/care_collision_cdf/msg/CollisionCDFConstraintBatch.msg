std_msgs/Header header

int32 num_pairs
int32 dof

# One entry per paired constraint.
int32[] original_timestep

# Flat arrays:
# point_flat              : num_pairs * 3
# q_linearization_flat    : num_pairs * dof
# distance                : num_pairs
# gradient_flat           : num_pairs * dof
float64[] point_flat
float64[] q_linearization_flat
float64[] distance
float64[] gradient_flat

# Profiling copied from the C++ selector / GPU worker.
float64 pair_selection_ms
float64 pair_buffer_build_ms
float64 ipc_roundtrip_ms
float64 gpu_h2d_ms
float64 gpu_inference_ms
float64 gpu_d2h_ms
float64 gpu_worker_total_ms
float64 online_pipeline_ms
