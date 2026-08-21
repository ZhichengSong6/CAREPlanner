#pragma once

#include <care_confidence_map/body_sample_model.hpp>

#include <Eigen/Dense>

#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

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

// Pose of an arbitrary URDF/Pinocchio frame expressed in base_frame_.  This
// lightweight representation intentionally keeps Pinocchio types out of the
// public VBC selector interface.
struct FramePoseInBase
{
  std::string frame_name;
  Eigen::Vector3d translation_base = Eigen::Vector3d::Zero();
  Eigen::Matrix3d rotation_base = Eigen::Matrix3d::Identity();
};

// Geometry required by the predicted-VBC verifier for one future q.  Body
// samples and sensor poses are produced from the same Pinocchio FK/update pass
// so the real-time verifier does not pay for duplicated forward kinematics.
struct ConfigurationAuditGeometry
{
  int timestep_index = -1;
  std::vector<TrajectoryBodySample> body_samples;
  std::vector<FramePoseInBase> frame_poses;
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

  // Compute poses of requested frames for a future configuration.  This is
  // used by visibility-before-contact analysis to predict when a workspace
  // point enters any arm-mounted sensor FOV along the nominal trajectory.
  bool computeFramePosesForConfiguration(
      const Eigen::VectorXd& q,
      const std::vector<std::string>& frame_names,
      std::vector<FramePoseInBase>* out,
      std::string* error_msg = nullptr) const;

  // Real-time helper for C4 predicted-trajectory VBC auditing.  Performs one
  // FK/updateFramePlacements call and extracts both body samples and the
  // requested sensor poses from that same kinematic state.
  bool computeAuditGeometryForConfiguration(
      const Eigen::VectorXd& q,
      int timestep_index,
      const std::vector<std::string>& frame_names,
      ConfigurationAuditGeometry* out,
      std::string* error_msg = nullptr) const;

private:
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
};

}  // namespace care_confidence_map
