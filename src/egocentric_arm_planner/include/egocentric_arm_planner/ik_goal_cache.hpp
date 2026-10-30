#pragma once
#include <Eigen/Geometry>
#include <string>

namespace egocentric_arm_planner {
// A converged target configuration is only an IK seed. Execution boundaries,
// trajectory dynamics and safety are always rebuilt from the measured state.
struct IKGoalCache {
  std::string frame;
  Eigen::Isometry3d target = Eigen::Isometry3d::Identity();
  Eigen::VectorXd q;

  bool matches(const std::string& candidate_frame,
               const Eigen::Isometry3d& candidate, int dof) const {
    return q.size() == dof && q.allFinite() && frame == candidate_frame &&
        candidate.matrix().allFinite() &&
        (target.translation()-candidate.translation()).norm() <= 1e-8 &&
        (target.rotation()-candidate.rotation()).norm() <= 1e-8;
  }
  void store(const std::string& target_frame, const Eigen::Isometry3d& pose,
             const Eigen::VectorXd& goal) {
    if (!pose.matrix().allFinite() || goal.size() == 0 || !goal.allFinite()) return;
    frame=target_frame; target=pose; q=goal;
  }
};
}  // namespace egocentric_arm_planner
