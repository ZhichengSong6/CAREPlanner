// Test-only frozen pre-optimization rasterizer (2026-09-11).
// Kept independent of the optimized function to compare every evidence field.
#pragma once
#include <care_confidence_map/vbc_primitive_model.hpp>
namespace care_confidence_map {
template <class Voxel>
std::vector<Voxel> buildPrimitiveSweptVoxelsReference(
    const std::vector<VbcPrimitiveFrame>& frames,
    const std::vector<int>& original_indices, const std::vector<double>& times,
    const VbcGrid& grid, double margin) {
  if (!grid.min.allFinite() || !grid.max.allFinite() ||
      !std::isfinite(grid.resolution) || grid.resolution <= 0 ||
      !std::isfinite(margin) || margin < 0 || (grid.max.array() < grid.min.array()).any() ||
      frames.empty() || frames.size() != times.size() || frames.size() != original_indices.size())
    throw std::invalid_argument("Invalid primitive sweep grid/timing/margin");
  const Eigen::Array3d dims = ((grid.max - grid.min) / grid.resolution).array().floor() + 1;
  if ((dims > double(std::numeric_limits<int>::max())).any())
    throw std::invalid_argument("Primitive sweep grid exceeds integer bounds");
  const Eigen::Array3i n = dims.cast<int>();
  using Key = std::tuple<int, int, int>;
  std::map<Key, Voxel> earliest;
  for (std::size_t k = 0; k < frames.size(); ++k) {
    if (!std::isfinite(times[k]) || times[k] < 0 || original_indices[k] < 0)
      throw std::invalid_argument("Invalid primitive sweep time/index");
    for (const auto& p : frames[k].primitives) {
      // Bounds must include the same floating-point tolerance as the SDF test,
      // otherwise ceil/floor can discard an exactly-on-surface grid center.
      const Eigen::Vector3d extent = p.aabbHalfExtent(margin + 1e-12);
      if (!p.center.allFinite() || !p.rotation.allFinite() || !extent.allFinite())
        throw std::invalid_argument("Nonfinite primitive FK geometry");
      // Clip in floating point BEFORE integer conversion, including distant bodies.
      const Eigen::Array3d lo = ((p.center - extent - grid.min) / grid.resolution).array();
      const Eigen::Array3d hi = ((p.center + extent - grid.min) / grid.resolution).array();
      if ((hi < 0).any() || (lo > (n - 1).cast<double>()).any()) continue;
      const Eigen::Array3i begin = lo.ceil().max(0).cast<int>();
      const Eigen::Array3i end = hi.floor().min((n - 1).cast<double>()).cast<int>();
      for (int ix = begin.x(); ix <= end.x(); ++ix)
        for (int iy = begin.y(); iy <= end.y(); ++iy)
          for (int iz = begin.z(); iz <= end.z(); ++iz) {
            const Key key(ix, iy, iz);
            const auto it = earliest.find(key);
            if (it != earliest.end() && it->second.sweep_time_s <= times[k] + 1e-12) continue;
            const Eigen::Vector3d point = grid.min + grid.resolution * Eigen::Vector3d(ix, iy, iz);
            const double distance = p.signedDistance(point);
            if (distance > margin + 1e-12) continue;
            Voxel v;
            v.point_base = point;
            v.link_name = p.link_name;
            v.source_type = p.sourceType();
            v.source_collision_index = p.collision_index;
            v.source_collision_name = p.collision_name;
            v.sample_index_in_link = -1;
            v.sample_center_base = p.center;
            v.raw_sample_radius_m = v.swept_radius_m = std::numeric_limits<double>::quiet_NaN();
            v.point_center_distance_m = (point - p.center).norm();
            v.primitive_signed_distance_m = distance;
            v.sweep_eval_timestep = static_cast<int>(k);
            v.sweep_original_timestep = original_indices[k];
            v.sweep_time_s = times[k];
            earliest[key] = std::move(v);
          }
    }
  }
  std::vector<Voxel> out;
  out.reserve(earliest.size());
  for (auto& item : earliest) out.push_back(std::move(item.second));
  return out;
}
}  // namespace care_confidence_map
