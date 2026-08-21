#pragma once

#include <care_confidence_map/body_sample_model.hpp>

#include <Eigen/Dense>

#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace care_confidence_map
{

struct TrajectoryBodySample
{
  int timestep_index = -1;

  std::string link_name;
  std::string frame_name;

  Eigen::Vector3d center_base = Eigen::Vector3d::Zero();
  double radius = 0.0;

  std::string source_type;
  int source_collision_index = -1;
  int sample_index_in_link = -1;

  bool include_for_risk = true;
};

struct TrajectoryFrameSamples
{
  int timestep_index = -1;
  Eigen::VectorXd q;

  std::vector<TrajectoryBodySample> samples;
};

struct TrajectorySampleResult
{
  bool success = false;
  std::string message;

  int num_timesteps = 0;
  int num_samples_per_timestep = 0;
  int total_samples = 0;

  std::vector<TrajectoryFrameSamples> frames;
};

// Pose of an arbitrary URDF/Pinocchio frame expressed in base_frame_.
struct FramePoseInBase
{
  std::string frame_name;
  Eigen::Vector3d translation_base = Eigen::Vector3d::Zero();
  Eigen::Matrix3d rotation_base = Eigen::Matrix3d::Identity();
};

// Geometry required by the original predicted-VBC verifier for one future q.
struct ConfigurationAuditGeometry
{
  int timestep_index = -1;
  std::vector<TrajectoryBodySample> body_samples;
  std::vector<FramePoseInBase> frame_poses;
};

// Allocation-light sensor pose used by the optimized C4 verifier.  The sensor
// identity is fixed at prepareFastAudit() time, so no strings are copied in the
// per-configuration hot path.
struct FastAuditSensorPose
{
  Eigen::Vector3d translation_base = Eigen::Vector3d::Zero();
  Eigen::Matrix3d rotation_base = Eigen::Matrix3d::Identity();
};

class TrajectoryRiskEvaluator
{
public:
  TrajectoryRiskEvaluator() = default;

  bool initialize(const std::string& robot_urdf_file,
                  const std::string& body_samples_file,
                  const std::string& base_frame,
                  std::string* error_msg = nullptr);

  bool isInitialized() const
  {
    return initialized_;
  }

  int nq() const
  {
    return model_.nq;
  }

  int nv() const
  {
    return model_.nv;
  }

  const std::vector<std::string>& activeJointNames() const
  {
    return active_joint_names_;
  }

  const BodySampleModel& bodySampleModel() const
  {
    return body_sample_model_;
  }

  TrajectorySampleResult computeTrajectorySamples(
      const std::vector<Eigen::VectorXd>& q_traj) const;

  bool computeSamplesForConfiguration(
      const Eigen::VectorXd& q,
      int timestep_index,
      TrajectoryFrameSamples* out,
      std::string* error_msg = nullptr) const;

  bool computeFramePosesForConfiguration(
      const Eigen::VectorXd& q,
      const std::vector<std::string>& frame_names,
      std::vector<FramePoseInBase>* out,
      std::string* error_msg = nullptr) const;

  bool computeAuditGeometryForConfiguration(
      const Eigen::VectorXd& q,
      int timestep_index,
      const std::vector<std::string>& frame_names,
      ConfigurationAuditGeometry* out,
      std::string* error_msg = nullptr) const;

  // Prepare the real-time C4 audit path once.  Frame IDs, ignored-link
  // filtering, body sample local centers/radii and body-frame grouping are all
  // cached here so evaluateFastAuditForConfiguration() performs no string/frame
  // lookup in the normal per-MPC-cycle hot path.
  bool prepareFastAudit(
      const std::vector<std::string>& sensor_frame_names,
      const std::vector<std::string>& ignored_risk_links,
      std::string* error_msg = nullptr);

  // One FK/updateFramePlacements pass for q, followed by:
  //   * cached 8-sensor pose extraction;
  //   * cached/grouped body sample transforms and target clearance.
  // sensor_poses capacity is reused by the caller across horizon states.
  bool evaluateFastAuditForConfiguration(
      const Eigen::VectorXd& q,
      const Eigen::Vector3d& target_base,
      std::vector<FastAuditSensorPose>* sensor_poses,
      double* min_clearance_m,
      std::string* error_msg = nullptr) const;

  std::size_t fastAuditBodySampleCount() const
  {
    return fast_audit_body_sample_count_;
  }

  std::size_t fastAuditBodyFrameCount() const
  {
    return fast_audit_body_frames_.size();
  }

  std::size_t fastAuditSensorCount() const
  {
    return fast_audit_sensor_frame_ids_.size();
  }

private:
  struct CachedAuditBodySample
  {
    Eigen::Vector3d center_link = Eigen::Vector3d::Zero();
    double radius = 0.0;
  };

  struct CachedAuditBodyFrame
  {
    pinocchio::FrameIndex frame_id = 0;
    std::vector<CachedAuditBodySample> samples;
  };

  bool buildPinocchioModel(const std::string& robot_urdf_file,
                           std::string* error_msg);

  bool validateBodySampleFrames(std::string* error_msg) const;

  void extractActiveJointNames();

  bool checkConfigurationSize(const Eigen::VectorXd& q,
                              std::string* error_msg) const;

private:
  bool initialized_ = false;

  std::string robot_urdf_file_;
  std::string body_samples_file_;
  std::string base_frame_ = "base_link";

  BodySampleModel body_sample_model_;

  pinocchio::Model model_;
  mutable pinocchio::Data data_;

  std::vector<std::string> active_joint_names_;

  bool fast_audit_prepared_ = false;
  pinocchio::FrameIndex fast_audit_base_frame_id_ = 0;
  std::vector<pinocchio::FrameIndex> fast_audit_sensor_frame_ids_;
  std::vector<CachedAuditBodyFrame> fast_audit_body_frames_;
  std::size_t fast_audit_body_sample_count_ = 0;
};

}  // namespace care_confidence_map