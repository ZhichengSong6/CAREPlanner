# header.stamp is the ORIGINAL tracker generation time. Intermediaries preserve
# it, source_sequence, and every lease field. They may only limit velocity.
std_msgs/Header header
uint64 source_sequence
uint64 controller_session
uint64 execution_stamp_ns
uint64 lease_sequence
time valid_until
string certificate_sha256
bool target_range_enforced
float64[7] target_abs_velocity
string[] joint_names
float64[] velocity
# Terminal for this execution. The limiter repeats it until a NEW token arrives.
bool stop
