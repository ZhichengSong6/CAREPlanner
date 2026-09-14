#pragma once

#include <cmath>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

namespace care_confidence_map {
// Diagnostic serialization only. All values come from the SAME completed
// audit; never rerun visibility or change its decision to populate this trace.
inline std::string evidenceNumber(double value) {
  if (!std::isfinite(value)) return "null";
  std::ostringstream s;
  s.imbue(std::locale::classic());
  s << std::setprecision(17) << value;
  return s.str();
}

inline std::string evidenceString(const std::string& value) {
  std::ostringstream s;
  s << '"';
  for (unsigned char c : value) {
    // Escape spaces too: this JSON is one token in the legacy key=value wire.
    if (c <= 32 || c == '"' || c == '\\')
      s << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
    else s << c;
  }
  s << '"';
  return s.str();
}

template <class Candidates, class Layers>
std::string vbcEvidence(unsigned long long stamp, unsigned long long bundle,
                        double required_margin, double resolution,
                        const Candidates& candidates, const Layers& layers,
                        const std::string& geometry_backend = "samples") {
  std::ostringstream s;
  s.imbue(std::locale::classic());
  s << "{\"schema\":1,\"trajectory_stamp_ns\":\"" << stamp
    << "\",\"bundle_seq\":" << bundle
    << ",\"required_margin_s\":" << evidenceNumber(required_margin)
    << ",\"voxel_resolution_m\":" << evidenceNumber(resolution)
    << ",\"geometry_backend\":" << evidenceString(geometry_backend)
    << ",\"spatial_semantics\":\"voxel_centers_discrete_knots\""
    << ",\"violations\":[";
  bool first = true;
  for (std::size_t layer = 0; layer < layers.size(); ++layer) {
    for (int index : layers[layer].member_indices) {
      const auto& c = candidates.at(index);
      if (!first) s << ',';
      first = false;
      s << "{\"layer\":" << layer << ",\"point\":["
        << evidenceNumber(c.point_base.x()) << ','
        << evidenceNumber(c.point_base.y()) << ','
        << evidenceNumber(c.point_base.z()) << ']'
        << ",\"sweeping_link\":" << evidenceString(c.link_name)
        << ",\"sample_index\":" << c.sample_index_in_link
        << ",\"body_source\":" << evidenceString(c.source_type)
        << ",\"source_collision_index\":" << c.source_collision_index
        << ",\"source_collision_name\":" << evidenceString(c.source_collision_name)
        << ",\"primitive_signed_distance_m\":" << evidenceNumber(c.primitive_signed_distance_m)
        << ",\"confidence\":" << evidenceNumber(c.confidence)
        << ",\"current_visibility\":" << evidenceNumber(c.current_visibility)
        << ",\"raw_radius_m\":" << evidenceNumber(c.raw_sample_radius_m)
        << ",\"swept_radius_m\":" << evidenceNumber(c.swept_radius_m)
        << ",\"center_distance_m\":" << evidenceNumber(c.point_center_distance_m)
        << ",\"sweep_eval_k\":" << c.sweep_eval_timestep
        << ",\"sweep_original_k\":" << c.sweep_original_timestep
        << ",\"sweep_s\":" << evidenceNumber(c.sweep_time_s)
        << ",\"latest_see_s\":" << evidenceNumber(c.sweep_time_s - required_margin)
        << ",\"nominally_visible\":" << (c.nominally_visible ? "true" : "false")
        << ",\"see_eval_k\":" << c.see_eval_timestep
        << ",\"see_original_k\":" << c.see_original_timestep
        << ",\"see_s\":" << evidenceNumber(c.see_time_s)
        << ",\"margin_s\":" << evidenceNumber(c.margin_s) << '}';
    }
  }
  s << "]}";
  return s.str();
}
}  // namespace care_confidence_map
