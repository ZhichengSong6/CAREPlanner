# Planning intent only. Exact GCDF/VBC/commit gates still authorize execution.
# revision changes only when the selected prefix changes; Header.seq is not identity.
std_msgs/Header header
string session
uint64 revision
float64 prefix_s
float64[] acceleration
float64 brake_dt_s
uint32 brake_steps
