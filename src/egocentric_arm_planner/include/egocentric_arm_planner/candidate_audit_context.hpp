#pragma once
#include <cmath>
#include <iomanip>
#include <ostream>
#include <string>
#include <trajectory_msgs/JointTrajectory.h>
#include <care_collision_cdf/CollisionCDFConstraintBatch.h>

namespace egocentric_arm_planner {
// Candidate audit metadata must not depend on the local QP snapshot writer.
inline void auditNumber(std::ostream& out, double v) {
  if (std::isfinite(v)) out << std::setprecision(17) << v;
  else out << (std::isnan(v) ? "\"nan\"" : v > 0 ? "\"inf\"" : "\"-inf\"");
}
template<class Vector> void auditVector(std::ostream& out, const Vector& v) {
  out << '[';
  for (std::size_t i = 0; i < static_cast<std::size_t>(v.size()); ++i) {
    if (i) out << ',';
    auditNumber(out, v[i]);
  }
  out << ']';
}
inline void auditString(std::ostream& out, const std::string& text) {
  out << '"';
  for (unsigned char c : text) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u00" << "0123456789abcdef"[c >> 4] << "0123456789abcdef"[c & 15];
    else out << c;
  }
  out << '"';
}
inline void auditTrajectory(std::ostream& out, const trajectory_msgs::JointTrajectory& t) {
  out << "{\"stamp_ns\":\"" << t.header.stamp.toNSec() << "\",\"frame_id\":";
  auditString(out, t.header.frame_id);
  out << ",\"joint_names\":[";
  for (std::size_t i=0; i<t.joint_names.size(); ++i) { if(i) out << ','; auditString(out,t.joint_names[i]); }
  out << "],\"points\":[";
  for (std::size_t i=0; i<t.points.size(); ++i) {
    if(i) out << ',';
    const auto& p=t.points[i];
    out << "{\"time_ns\":\"" << p.time_from_start.toNSec() << "\",\"positions\":";
    auditVector(out,p.positions); out << ",\"velocities\":"; auditVector(out,p.velocities);
    out << ",\"accelerations\":"; auditVector(out,p.accelerations);
    out << ",\"effort\":"; auditVector(out,p.effort); out << '}';
  }
  out << "]}";
}
inline void auditBatch(std::ostream& out, const care_collision_cdf::CollisionCDFConstraintBatch& b) {
  out << "{\"stamp_ns\":\"" << b.header.stamp.toNSec() << "\",\"dof\":" << b.dof
      << ",\"num_pairs\":" << b.num_pairs;
#define AUDIT_ARRAY(field) out << ",\"" #field "\":"; auditVector(out,b.field)
  AUDIT_ARRAY(original_timestep); AUDIT_ARRAY(source_type); AUDIT_ARRAY(point_flat);
  AUDIT_ARRAY(q_linearization_flat); AUDIT_ARRAY(distance); AUDIT_ARRAY(gradient_flat);
  AUDIT_ARRAY(approx_body_clearance_m);
#undef AUDIT_ARRAY
  out << '}';
}
}  // namespace egocentric_arm_planner
