# Transport identity is separate from the raw and audited trajectory stamps.
# Retransmission preserves identity/payload and only reduces remaining_budget_s.
string action
string session
uint64 sequence
string receiver_boot_id
string payload_sha256
float64 remaining_budget_s
trajectory_msgs/JointTrajectory trajectory
# Immutable planning view, included in payload_sha256 when nonempty.
string planned_prefix_context
