# Negative evidence only; never a safety certificate. Header identifies the
# exact audited executable. No old distance/gradient may become a QP row.
std_msgs/Header header
time raw_candidate_stamp
float64[] point_flat
# More than 32 distinct rejected points: do not silently truncate coverage.
bool overflow
