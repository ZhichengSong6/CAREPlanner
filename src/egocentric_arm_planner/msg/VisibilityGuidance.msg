# Soft observation guidance. Never an observation or execution certificate.
time measured_stamp
uint64 sequence
string observation_token
string region_token
int32 sensor_id
float64[7] q_eval
float64[7] q_target
float64[7] joint_mask
float64[7] gradient
float64 value
bool drive_enabled
# Fresh measured-pose FOV progress, scheduling only. It cannot complete an obligation.
bool geometry_valid
string geometry_signature
float64 conservative_fov_margin
