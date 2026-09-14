# Query-scoped identities, never cached distances/gradients. LOCAL only.
std_msgs/Header header
uint64 plan_sequence
uint64 mode_epoch
uint64 target_revision
uint64 progress_epoch
int32 dof
int32[] original_timestep
float64[] point_flat
float64[] q_flat
