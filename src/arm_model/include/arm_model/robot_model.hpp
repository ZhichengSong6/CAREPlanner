#pragma once

#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

#include "arm_model/sensor_frame_manager.hpp"

#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <geometry_msgs/PoseStamped.h>

namespace arm_model {

class RobotModel {
public:
  RobotModel();
  ~RobotModel() = default;

  bool initializeFromRosParam(const ros::NodeHandle& nh);

  bool initialized() const;

  int dof() const;
  int nq() const;
  int nv() const;

  const std::string& baseFrame() const;
  const std::string& eeFrame() const;
  const std::vector<std::string>& jointNames() const;

  bool getJointQIndex(const std::string& joint_name, int& q_idx) const;
  bool getJointVIndex(const std::string& joint_name, int& v_idx) const;

  bool updateJointState(const sensor_msgs::JointState& msg);

  bool hasCurrentState() const;
  Eigen::VectorXd getCurrentQ() const;
  Eigen::VectorXd getCurrentDq() const;

  bool buildQVectorFromJointState(const sensor_msgs::JointState& msg,
                                  Eigen::VectorXd& q,
                                  Eigen::VectorXd& dq) const;

  bool getFramePose(const std::string& frame_name,
                    const Eigen::VectorXd& q,
                    Eigen::Isometry3d& T_base_frame) const;

  bool getEndEffectorPose(const Eigen::VectorXd& q,
                          Eigen::Isometry3d& T_base_ee) const;

  bool getSensorPoseByName(const std::string& sensor_name,
                           const Eigen::VectorXd& q,
                           Eigen::Isometry3d& T_base_sensor) const;

  bool getSensorPoseById(int sensor_id,
                         const Eigen::VectorXd& q,
                         Eigen::Isometry3d& T_base_sensor) const;

  bool solveIK(const Eigen::Isometry3d& target_pose,
               const Eigen::VectorXd& q_seed,
               Eigen::VectorXd& q_solution) const;

  bool poseMsgToEigen(const geometry_msgs::PoseStamped& pose_msg,
                      Eigen::Isometry3d& T) const;

  geometry_msgs::PoseStamped eigenToPoseMsg(const Eigen::Isometry3d& T,
                                            const std::string& frame_id) const;

  const SensorFrameManager& sensorFrameManager() const;

private:
  bool loadUrdfFromParam(const ros::NodeHandle& nh,
                         const std::string& param_name,
                         std::string& urdf_string) const;

  bool loadControlledJoints(const ros::NodeHandle& nh);

  bool fillNeutralConfiguration(Eigen::VectorXd& q) const;

  bool getFrameId(const std::string& frame_name,
                  pinocchio::FrameIndex& frame_id) const;

  Eigen::Isometry3d pinocchioSE3ToEigen(const pinocchio::SE3& M) const;

  void clampToPositionLimits(Eigen::VectorXd& q) const;

private:
  bool initialized_ = false;
  bool has_current_state_ = false;

  std::string base_frame_ = "base_link";
  std::string ee_frame_ = "EE_link";
  std::string urdf_path_;

  std::vector<std::string> joint_names_;
  std::unordered_map<std::string, int> joint_name_to_index_;

  pinocchio::Model model_;
  mutable pinocchio::Data data_;

  SensorFrameManager sensor_frame_manager_;

  Eigen::VectorXd q_current_;
  Eigen::VectorXd dq_current_;

  double ik_position_weight_ = 1.0;
  double ik_orientation_weight_ = 0.5;
  double ik_damping_ = 1e-4;
  double ik_step_size_ = 0.5;
  double ik_pos_tol_ = 1e-3;
  double ik_rot_tol_ = 2e-2;
  int ik_max_iters_ = 100;
};

}  // namespace arm_model