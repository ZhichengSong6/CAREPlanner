# Issued by a separate, qualified stopping-envelope authorizer. Transport only:
# publishing this message does not establish that its certificate is safe.
uint64 controller_session
uint64 execution_stamp_ns
uint64 lease_sequence
time issued_at
time valid_until
string certificate_sha256
# If true, every accepted target is inside +/- target_abs_velocity. A zero
# bound permits only zero targets; existing output still brakes at its rate.
bool target_range_enforced
float64[7] target_abs_velocity
bool revoke
