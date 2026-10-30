std_msgs/Header header
uint8 WAITING=0
uint8 TRACKING=1
uint8 BRAKING=2
uint8 LATCHED=3
uint8 mode
bool clock_fault
time command_stamp
uint32 command_sequence
time fault_stamp
time stopped_stamp
uint64 rejected_commands
uint64 rearm_successes
uint64 rearm_rejections
float64[7] applied_velocity
float64[7] measured_velocity
# Diagnostics at header.stamp, before this update's applied_velocity takes effect.
float64[7] measured_position
float64[7] applied_position_integral
bool integral_valid
float64 max_update_gap
# Lease mode reports the identity actually accepted by the controller, not the
# latest request observed by a Python subscriber. Zero session means legacy mode.
bool execution_lease_required
uint64 controller_session
uint64 execution_stamp_ns
uint64 lease_sequence
uint64 source_sequence
time lease_valid_until
string certificate_sha256
# Range from the APPLIED grant, never the latest requested grant.
bool target_range_enforced
float64[7] target_abs_velocity
uint64 retired_execution_through_ns
uint64 rejected_leases
# Limits actually loaded by this controller instance, for budget binding.
float64[7] velocity_limits
float64[7] acceleration_limits
float64 command_timeout_s
float64 max_execution_lease_s
float64 max_braking_step_s
