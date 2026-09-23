#pragma once

#include <care_confidence_map/trajectory_risk_evaluator.hpp>
#include <care_confidence_map/primitive_geometry.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>

namespace care_confidence_map {


struct VbcPrimitiveFrame {
  std::vector<VbcPrimitive> primitives;
  // Per-solid displacement enclosure about this representative configuration.
  // Empty means an exact static pose (legacy discrete API).
  std::vector<double> motion_bounds;
  int original_eval_timestep = -1;
};

class VbcPrimitiveModel {
 public:
  bool load(const std::string& urdf_file, const std::vector<std::string>& ignored_links,
            std::string* error);
  bool computeTrajectory(const TrajectoryRiskEvaluator& fk,
                         const std::vector<Eigen::VectorXd>& trajectory,
                         std::vector<VbcPrimitiveFrame>* out, std::string* error) const;
  const std::vector<VbcPrimitive>& primitives() const { return primitives_; }
  const std::vector<std::string>& frames() const { return frames_; }
  bool computeContinuousTrajectory(const TrajectoryRiskEvaluator& fk,
      const std::vector<Eigen::VectorXd>& q, const std::vector<int>& indices,
      const std::vector<double>& times, std::vector<VbcPrimitiveFrame>* out,
      std::vector<int>* out_indices, std::vector<double>* out_times,
      std::string* error, bool use_motion_bounds = false) const;
  std::vector<double> displacementBounds(const std::vector<std::string>& names,
      const Eigen::VectorXd& delta) const;

  // Configuration-aware runtime body displacement.  Each returned value is a
  // conservative bound for one primitive between q_measured and q_reference:
  // center translation plus the relative rotation of the primitive times its
  // local enclosing radius.  This is intentionally separate from
  // displacementBounds(), which remains the continuous-sweep certificate used
  // by VBC.
  bool initializeRelativeFk(const std::string& urdf_file,
                            const std::vector<std::string>& joint_names,
                            std::string* error = nullptr);
  std::vector<double> relativeFkDisplacementBounds(
      const Eigen::VectorXd& q_measured,
      const Eigen::VectorXd& q_reference) const;

 private:
  std::vector<std::string> frames_;
  std::vector<VbcPrimitive> primitives_;
  std::vector<std::map<std::string, double>> joint_radii_;

  bool relative_fk_initialized_ = false;
  pinocchio::Model relative_fk_model_;
  mutable pinocchio::Data relative_fk_measured_data_;
  mutable pinocchio::Data relative_fk_reference_data_;
  std::vector<int> relative_fk_q_indices_;
  std::vector<pinocchio::FrameIndex> relative_fk_frame_ids_;
  std::vector<double> relative_fk_primitive_radii_;
};

struct VbcGrid {
  Eigen::Vector3d min, max;
  double resolution = 0.05;
};

// Grid-center rasterization of static poses or conservative interval enclosures.
// Continuous coverage requires frames from computeContinuousTrajectory; this
// does not certify voxel-cube intersections or unmodelled actuator motion.
// Template output keeps the selector's evidence/temporal pipeline unchanged.
template <class Voxel>
std::vector<Voxel> buildPrimitiveSweptVoxels(
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
  const int ny = n.y(), nz = n.z();
  const double grid_x = grid.min.x(), grid_y = grid.min.y(), grid_z = grid.min.z();
  using Key = std::tuple<int, int, int>;
  // A grid-local slot table avoids a tree lookup at EVERY AABB grid center.
  // Bound scratch memory (16 MiB) independently of user grid size; very large
  // grids retain the sparse algorithm, never truncate geometry or change resolution.
  constexpr std::size_t max_dense_cells = 4 * 1024 * 1024;
  std::size_t cell_count = 1;
  bool dense = true;
  for (int axis = 0; axis < 3; ++axis) {
    if (static_cast<std::size_t>(n[axis]) > max_dense_cells / cell_count) {
      dense = false; break;
    }
    cell_count *= static_cast<std::size_t>(n[axis]);
  }
  std::vector<int> slots(dense ? cell_count : 0, -1);
  std::vector<std::pair<std::size_t, Voxel>> occupied;
  std::map<Key, Voxel> earliest;
  for (std::size_t k = 0; k < frames.size(); ++k) {
    if (!std::isfinite(times[k]) || times[k] < 0 || original_indices[k] < 0)
      throw std::invalid_argument("Invalid primitive sweep time/index");
    if (!frames[k].motion_bounds.empty() && frames[k].motion_bounds.size() != frames[k].primitives.size())
      throw std::invalid_argument("Incomplete continuous primitive enclosure");
    for (std::size_t pi = 0; pi < frames[k].primitives.size(); ++pi) {
      const auto& p = frames[k].primitives[pi];
      const double bound = frames[k].motion_bounds.empty() ? 0.0 : frames[k].motion_bounds[pi];
      if (!std::isfinite(bound) || bound < 0) throw std::invalid_argument("Invalid motion bound");
      const double effective_margin = margin + bound;
      // Bounds must include the same floating-point tolerance as the SDF test,
      // otherwise ceil/floor can discard an exactly-on-surface grid center.
      const Eigen::Vector3d extent = p.aabbHalfExtent(effective_margin + 1e-12);
      if (!p.center.allFinite() || !p.rotation.allFinite() || !extent.allFinite())
        throw std::invalid_argument("Nonfinite primitive FK geometry");
      // Clip in floating point BEFORE integer conversion, including distant bodies.
      const Eigen::Array3d lo = ((p.center - extent - grid.min) / grid.resolution).array();
      const Eigen::Array3d hi = ((p.center + extent - grid.min) / grid.resolution).array();
      if ((hi < 0).any() || (lo > (n - 1).cast<double>()).any()) continue;
      const Eigen::Array3i begin = lo.ceil().max(0).cast<int>();
      const Eigen::Array3i end = hi.floor().min((n - 1).cast<double>()).cast<int>();
      const int bx = begin.x(), by = begin.y(), bz = begin.z();
      const int ex = end.x(), ey = end.y(), ez = end.z();
      for (int ix = bx; ix <= ex; ++ix) {
        const double x = grid_x + grid.resolution * ix;
        for (int iy = by; iy <= ey; ++iy) {
          const double y = grid_y + grid.resolution * iy;
          const std::size_t row = dense ? (static_cast<std::size_t>(ix) * ny + iy) * nz : 0;
          for (int iz = bz; iz <= ez; ++iz) {
            std::size_t flat = 0;
            int slot = -1;
            Voxel* previous = nullptr;
            if (dense) {
              flat = row + iz;
              slot = slots[flat];
              if (slot >= 0) previous = &occupied[slot].second;
            } else {
              const auto it = earliest.find(Key(ix, iy, iz));
              if (it != earliest.end()) previous = &it->second;
            }
            // Preserve the original time tolerance and first-in-traversal tie.
            if (previous && previous->sweep_time_s <= times[k] + 1e-12) continue;
            const Eigen::Vector3d point(x, y, grid_z + grid.resolution * iz);
            const double distance = p.signedDistance(point);
            if (distance > effective_margin + 1e-12) continue;
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
            v.sweep_eval_timestep = frames[k].original_eval_timestep >= 0
                ? frames[k].original_eval_timestep : static_cast<int>(k);
            v.sweep_original_timestep = original_indices[k];
            v.sweep_time_s = times[k];
            if (previous) *previous = std::move(v);
            else if (dense) {
              slots[flat] = static_cast<int>(occupied.size());
              occupied.emplace_back(flat, std::move(v));
            } else earliest.emplace(Key(ix, iy, iz), std::move(v));
          }
        }
      }
    }
  }
  std::vector<Voxel> out;
  if (dense) {
    // Flattening is lexicographic (x,y,z). Sorting only occupied cells retains
    // exactly the old map's output order, without scanning all grid cells.
    std::sort(occupied.begin(), occupied.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    out.reserve(occupied.size());
    for (auto& item : occupied) out.push_back(std::move(item.second));
  } else {
    out.reserve(earliest.size());
    for (auto& item : earliest) out.push_back(std::move(item.second));
  }
  return out;
}
}  // namespace care_confidence_map
