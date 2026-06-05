#include "arm_model/robot_model.hpp"

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/spatial/explog.hpp>

#include <XmlRpcValue.h>

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <limits>

namespace arm_model {

RobotModel::RobotModel() = default;

bool RobotModel::initializeFromRosParam(const ros::NodeHandle& nh) {
  std::string robot_description_param = "/robot_description";
  nh.param<std::string>("robot_description_param", robot_description_param, robot_description_param);

  nh.param<std::string>("base_frame", base_frame_, base_frame_);
  nh.param<std::string>("ee_frame", ee_frame_, ee_frame_);
  nh.param<std::string>("urdf_path", urdf_path_, std::string(""));

  nh.param<double>("ik/position_weight", ik_position_weight_, ik_position_weight_);
  nh.param<double>("ik/orientation_weight", ik_orientation_weight_, ik_orientation_weight_);
  nh.param<double>("ik/damping", ik_damping_, ik_damping_);
  nh.param<double>("ik/step_size", ik_step_size_, ik_step_size_);
  nh.param<double>("ik/position_tolerance", ik_pos_tol_, ik_pos_tol_);
  nh.param<double>("ik/orientation_tolerance", ik_rot_tol_, ik_rot_tol_);
  nh.param<int>("ik/max_iterations", ik_max_iters_, ik_max_iters_);

  try {
    if (!urdf_path_.empty()) {
      ROS_INFO_STREAM("[RobotModel] Loading URDF from file: " << urdf_path_);
      pinocchio::urdf::buildModel(urdf_path_, model_);
    } else {
      std::string urdf_string;
      if (!loadUrdfFromParam(nh, robot_description_param, urdf_string)) {
        return false;
      }

      ROS_INFO_STREAM("[RobotModel] Loading URDF from ROS param: " << robot_description_param);
      pinocchio::urdf::buildModelFromXML(urdf_string, model_);
    }

    data_ = pinocchio::Data(model_);
  } catch (const std::exception& e) {
    ROS_ERROR_STREAM("[RobotModel] Failed to build Pinocchio model from URDF: " << e.what());
    return false;
  }

  if (!loadControlledJoints(nh)) {
    return false;
  }

  if (static_cast<int>(joint_names_.size()) != model_.nq) {
    ROS_WARN_STREAM("[RobotModel] controlled_joints size (" << joint_names_.size()
                    << ") != model.nq (" << model_.nq << "). This is okay only if your URDF has fixed/mimic/floating joints handled separately.");
  }

  q_current_ = Eigen::VectorXd::Zero(model_.nq);
  dq_current_ = Eigen::VectorXd::Zero(model_.nv);

  if (!fillNeutralConfiguration(q_current_)) {
    q_current_.setZero();
  }

  if (!sensor_frame_manager_.loadFromRosParam(nh, "sensors")) {
    ROS_WARN("[RobotModel] Sensor frames not loaded. Phase I can still run without sensor poses.");
  }

  for (const auto& sensor : sensor_frame_manager_.sensors()) {
    pinocchio::FrameIndex sensor_frame_id;
    if (!getFrameId(sensor.frame, sensor_frame_id)) {
      ROS_WARN_STREAM("[RobotModel] Sensor frame not found in model: "
                      << sensor.frame
                      << " for sensor "
                      << sensor.name);
    } else {
      ROS_INFO_STREAM("[RobotModel] Sensor "
                      << sensor.id << " / " << sensor.name
                      << " uses frame " << sensor.frame);
    }
  }

  pinocchio::FrameIndex ee_id;
  if (!getFrameId(ee_frame_, ee_id)) {
    ROS_ERROR_STREAM("[RobotModel] EE frame not found in model: " << ee_frame_);
    return false;
  }

  initialized_ = true;

  ROS_INFO_STREAM("[RobotModel] Initialized.");
  ROS_INFO_STREAM("[RobotModel] base_frame: " << base_frame_);
  ROS_INFO_STREAM("[RobotModel] ee_frame: " << ee_frame_);
  ROS_INFO_STREAM("[RobotModel] nq: " << model_.nq << ", nv: " << model_.nv);
  ROS_INFO_STREAM("[RobotModel] controlled joints: " << joint_names_.size());

  return true;
}

bool RobotModel::initialized() const {
  return initialized_;
}

int RobotModel::dof() const {
  return static_cast<int>(joint_names_.size());
}

int RobotModel::nq() const {
  return model_.nq;
}

int RobotModel::nv() const {
  return model_.nv;
}

const std::string& RobotModel::baseFrame() const {
  return base_frame_;
}

const std::string& RobotModel::eeFrame() const {
  return ee_frame_;
}

const std::vector<std::string>& RobotModel::jointNames() const {
  return joint_names_;
}

bool RobotModel::getJointQIndex(const std::string& joint_name,
                                int& q_idx) const {
  q_idx = -1;

  if (!initialized_) {
    ROS_ERROR("[RobotModel] getJointQIndex called before initialization.");
    return false;
  }

  if (!model_.existJointName(joint_name)) {
    ROS_ERROR_STREAM("[RobotModel] Joint name not found in model: " << joint_name);
    return false;
  }

  const pinocchio::JointIndex jid = model_.getJointId(joint_name);
  if (jid == 0 || jid >= model_.joints.size()) {
    ROS_ERROR_STREAM("[RobotModel] Invalid joint id for joint: " << joint_name);
    return false;
  }

  if (model_.nqs[jid] != 1) {
    ROS_ERROR_STREAM("[RobotModel] getJointQIndex currently expects 1-DoF joint. "
                     << joint_name
                     << " has nq = "
                     << model_.nqs[jid]);
    return false;
  }

  q_idx = model_.idx_qs[jid];

  if (q_idx < 0 || q_idx >= model_.nq) {
    ROS_ERROR_STREAM("[RobotModel] Invalid q index for joint: "
                     << joint_name
                     << ", q_idx = "
                     << q_idx);
    q_idx = -1;
    return false;
  }

  return true;
}

bool RobotModel::getJointVIndex(const std::string& joint_name,
                                int& v_idx) const {
  v_idx = -1;

  if (!initialized_) {
    ROS_ERROR("[RobotModel] getJointVIndex called before initialization.");
    return false;
  }

  if (!model_.existJointName(joint_name)) {
    ROS_ERROR_STREAM("[RobotModel] Joint name not found in model: " << joint_name);
    return false;
  }

  const pinocchio::JointIndex jid = model_.getJointId(joint_name);
  if (jid == 0 || jid >= model_.joints.size()) {
    ROS_ERROR_STREAM("[RobotModel] Invalid joint id for joint: " << joint_name);
    return false;
  }

  if (model_.nvs[jid] != 1) {
    ROS_ERROR_STREAM("[RobotModel] getJointVIndex currently expects 1-DoF joint. "
                     << joint_name
                     << " has nv = "
                     << model_.nvs[jid]);
    return false;
  }

  v_idx = model_.idx_vs[jid];

  if (v_idx < 0 || v_idx >= model_.nv) {
    ROS_ERROR_STREAM("[RobotModel] Invalid v index for joint: "
                     << joint_name
                     << ", v_idx = "
                     << v_idx);
    v_idx = -1;
    return false;
  }

  return true;
}

bool RobotModel::loadUrdfFromParam(const ros::NodeHandle& nh,
                                   const std::string& param_name,
                                   std::string& urdf_string) const {
  if (!nh.getParam(param_name, urdf_string)) {
    ROS_ERROR_STREAM("[RobotModel] Failed to read URDF from param: " << param_name);
    return false;
  }

  if (urdf_string.empty()) {
    ROS_ERROR_STREAM("[RobotModel] URDF string is empty at param: " << param_name);
    return false;
  }

  return true;
}

bool RobotModel::loadControlledJoints(const ros::NodeHandle& nh) {
  joint_names_.clear();
  joint_name_to_index_.clear();

  XmlRpc::XmlRpcValue joint_list;
  std::string param_used;

  if (nh.getParam("joint_names", joint_list)) {
    param_used = "joint_names";
  } else if (nh.getParam("controlled_joints", joint_list)) {
    param_used = "controlled_joints";
  } else {
    ROS_ERROR("[RobotModel] Please provide either 'joint_names' or 'controlled_joints' parameter.");
    return false;
  }

  if (joint_list.getType() != XmlRpc::XmlRpcValue::TypeArray) {
    ROS_ERROR_STREAM("[RobotModel] Param '" << param_used << "' must be a list.");
    return false;
  }

  for (int i = 0; i < joint_list.size(); ++i) {
    const std::string name = static_cast<std::string>(joint_list[i]);

    if (!model_.existJointName(name)) {
      ROS_ERROR_STREAM("[RobotModel] Joint not found in URDF/Pinocchio model: " << name);
      return false;
    }

    const pinocchio::JointIndex jid = model_.getJointId(name);
    const int nq_joint = model_.nqs[jid];
    const int nv_joint = model_.nvs[jid];

    if (nq_joint != 1 || nv_joint != 1) {
      ROS_ERROR_STREAM("[RobotModel] Phase-I RobotModel expects 1-DoF joints. Joint "
                       << name << " has nq=" << nq_joint << ", nv=" << nv_joint);
      return false;
    }

    joint_name_to_index_[name] = static_cast<int>(joint_names_.size());
    joint_names_.push_back(name);

    ROS_INFO_STREAM("[RobotModel] joint " << name
                    << " -> q index " << model_.idx_qs[jid]
                    << ", v index " << model_.idx_vs[jid]);
  }

  if (static_cast<int>(joint_names_.size()) != model_.nq) {
    ROS_WARN_STREAM("[RobotModel] joint_names size = " << joint_names_.size()
                    << ", model.nq = " << model_.nq
                    << ". This is okay only if fixed/mimic/non-controlled joints exist.");
  }

  return !joint_names_.empty();
}

bool RobotModel::fillNeutralConfiguration(Eigen::VectorXd& q) const {
  if (model_.nq <= 0) {
    return false;
  }

  try {
    q = pinocchio::neutral(model_);
    return true;
  } catch (...) {
    q = Eigen::VectorXd::Zero(model_.nq);
    return false;
  }
}

bool RobotModel::updateJointState(const sensor_msgs::JointState& msg) {
  if (!initialized_) {
    ROS_ERROR("[RobotModel] updateJointState called before initialization.");
    return false;
  }

  Eigen::VectorXd q;
  Eigen::VectorXd dq;
  if (!buildQVectorFromJointState(msg, q, dq)) {
    return false;
  }

  q_current_ = q;
  dq_current_ = dq;
  has_current_state_ = true;

  return true;
}

bool RobotModel::hasCurrentState() const {
  return has_current_state_;
}

Eigen::VectorXd RobotModel::getCurrentQ() const {
  return q_current_;
}

Eigen::VectorXd RobotModel::getCurrentDq() const {
  return dq_current_;
}

bool RobotModel::buildQVectorFromJointState(const sensor_msgs::JointState& msg,
                                            Eigen::VectorXd& q,
                                            Eigen::VectorXd& dq) const {
  q = q_current_;
  dq = Eigen::VectorXd::Zero(model_.nv);

  if (q.size() != model_.nq) {
    if (!fillNeutralConfiguration(q)) {
      q = Eigen::VectorXd::Zero(model_.nq);
    }
  }

  std::unordered_map<std::string, std::size_t> msg_index;
  for (std::size_t i = 0; i < msg.name.size(); ++i) {
    msg_index[msg.name[i]] = i;
  }

  for (const auto& joint_name : joint_names_) {
    const auto msg_it = msg_index.find(joint_name);
    if (msg_it == msg_index.end()) {
      ROS_ERROR_STREAM_THROTTLE(
          1.0,
          "[RobotModel] JointState missing controlled joint: " << joint_name);
      return false;
    }

    const std::size_t src_idx = msg_it->second;

    const pinocchio::JointIndex jid = model_.getJointId(joint_name);
    if (jid == 0 || jid >= model_.joints.size()) {
      ROS_ERROR_STREAM("[RobotModel] Invalid joint id for: " << joint_name);
      return false;
    }

    const int q_idx = model_.idx_qs[jid];
    const int v_idx = model_.idx_vs[jid];

    if (q_idx < 0 || q_idx >= model_.nq || v_idx < 0 || v_idx >= model_.nv) {
      ROS_ERROR_STREAM("[RobotModel] Invalid q/v index for joint: " << joint_name);
      return false;
    }

    // Position is required for planning. If it is missing, the current state is invalid.
    if (src_idx >= msg.position.size()) {
      ROS_ERROR_STREAM_THROTTLE(
          1.0,
          "[RobotModel] JointState has no position for controlled joint: "
              << joint_name);
      return false;
    }

    q[q_idx] = msg.position[src_idx];

    // Velocity is useful but not required in Phase I.
    // If missing, assume zero velocity for this joint.
    if (src_idx < msg.velocity.size()) {
      dq[v_idx] = msg.velocity[src_idx];
    } else {
      dq[v_idx] = 0.0;
    }
  }

  return true;
}

bool RobotModel::getFrameId(const std::string& frame_name,
                            pinocchio::FrameIndex& frame_id) const {
  if (!model_.existFrame(frame_name)) {
    return false;
  }

  frame_id = model_.getFrameId(frame_name);
  return frame_id < model_.frames.size();
}

Eigen::Isometry3d RobotModel::pinocchioSE3ToEigen(const pinocchio::SE3& M) const {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = M.rotation();
  T.translation() = M.translation();
  return T;
}

bool RobotModel::getFramePose(const std::string& frame_name,
                              const Eigen::VectorXd& q,
                              Eigen::Isometry3d& T_base_frame) const {
  if (!initialized_) {
    return false;
  }

  if (q.size() != model_.nq) {
    ROS_ERROR_STREAM("[RobotModel] getFramePose q size mismatch. q.size = "
                     << q.size() << ", model.nq = " << model_.nq);
    return false;
  }

  pinocchio::FrameIndex frame_id;
  if (!getFrameId(frame_name, frame_id)) {
    ROS_ERROR_STREAM("[RobotModel] Frame not found: " << frame_name);
    return false;
  }

  pinocchio::forwardKinematics(model_, data_, q);
  pinocchio::updateFramePlacements(model_, data_);

  T_base_frame = pinocchioSE3ToEigen(data_.oMf[frame_id]);
  return true;
}

bool RobotModel::getEndEffectorPose(const Eigen::VectorXd& q,
                                    Eigen::Isometry3d& T_base_ee) const {
  return getFramePose(ee_frame_, q, T_base_ee);
}

bool RobotModel::getSensorPoseByName(const std::string& sensor_name,
                                     const Eigen::VectorXd& q,
                                     Eigen::Isometry3d& T_base_sensor) const {
  SensorFrame sensor;
  if (!sensor_frame_manager_.getSensorByName(sensor_name, sensor)) {
    ROS_ERROR_STREAM("[RobotModel] Sensor name not found: " << sensor_name);
    return false;
  }

  return getFramePose(sensor.frame, q, T_base_sensor);
}

bool RobotModel::getSensorPoseById(int sensor_id,
                                   const Eigen::VectorXd& q,
                                   Eigen::Isometry3d& T_base_sensor) const {
  SensorFrame sensor;
  if (!sensor_frame_manager_.getSensorById(sensor_id, sensor)) {
    ROS_ERROR_STREAM("[RobotModel] Sensor id not found: " << sensor_id);
    return false;
  }

  return getFramePose(sensor.frame, q, T_base_sensor);
}

void RobotModel::clampToPositionLimits(Eigen::VectorXd& q) const {
  if (q.size() != model_.nq) {
    return;
  }

  for (pinocchio::JointIndex jid = 1; jid < static_cast<pinocchio::JointIndex>(model_.joints.size()); ++jid) {
    const int q_idx = model_.idx_qs[jid];
    const int nq_joint = model_.nqs[jid];

    if (nq_joint != 1) {
      continue;
    }

    const double lower = model_.lowerPositionLimit[q_idx];
    const double upper = model_.upperPositionLimit[q_idx];

    if (std::isfinite(lower) && std::isfinite(upper) && lower < upper) {
      q[q_idx] = std::min(std::max(q[q_idx], lower), upper);
    }
  }
}

bool RobotModel::solveIK(const Eigen::Isometry3d& target_pose,
                         const Eigen::VectorXd& q_seed,
                         Eigen::VectorXd& q_solution) const {
  if (!initialized_) {
    ROS_ERROR("[RobotModel] solveIK called before initialization.");
    return false;
  }

  if (q_seed.size() != model_.nq) {
    ROS_ERROR_STREAM("[RobotModel] solveIK q_seed size mismatch. q_seed.size = "
                     << q_seed.size() << ", model.nq = " << model_.nq);
    return false;
  }

  pinocchio::FrameIndex ee_id;
  if (!getFrameId(ee_frame_, ee_id)) {
    ROS_ERROR_STREAM("[RobotModel] EE frame not found: " << ee_frame_);
    return false;
  }

  Eigen::VectorXd q = q_seed;

  for (int iter = 0; iter < ik_max_iters_; ++iter) {
    pinocchio::forwardKinematics(model_, data_, q);
    pinocchio::updateFramePlacements(model_, data_);
    pinocchio::computeJointJacobians(model_, data_, q);

    const Eigen::Isometry3d T_current = pinocchioSE3ToEigen(data_.oMf[ee_id]);

    const Eigen::Vector3d pos_err = target_pose.translation() - T_current.translation();

    const Eigen::Matrix3d R_err = T_current.linear().transpose() * target_pose.linear();
    const Eigen::AngleAxisd aa(R_err);
    Eigen::Vector3d rot_err_local = aa.angle() * aa.axis();
    if (!rot_err_local.allFinite()) {
      rot_err_local.setZero();
    }

    const Eigen::Vector3d rot_err_world = T_current.linear() * rot_err_local;

    const double pos_norm = pos_err.norm();
    const double rot_norm = rot_err_world.norm();

    if (pos_norm < ik_pos_tol_ && rot_norm < ik_rot_tol_) {
      q_solution = q;
      return true;
    }

    Eigen::MatrixXd J6(6, model_.nv);
    J6.setZero();

    pinocchio::getFrameJacobian(model_,
                                data_,
                                ee_id,
                                pinocchio::LOCAL_WORLD_ALIGNED,
                                J6);

    Eigen::MatrixXd J_weighted(6, model_.nv);
    J_weighted.topRows<3>() = ik_position_weight_ * J6.topRows<3>();
    J_weighted.bottomRows<3>() = ik_orientation_weight_ * J6.bottomRows<3>();

    Eigen::Matrix<double, 6, 1> err;
    err.head<3>() = ik_position_weight_ * pos_err;
    err.tail<3>() = ik_orientation_weight_ * rot_err_world;

    const Eigen::MatrixXd A =
        J_weighted * J_weighted.transpose()
        + ik_damping_ * Eigen::MatrixXd::Identity(6, 6);

    Eigen::VectorXd v = J_weighted.transpose() * A.ldlt().solve(err);

    if (!v.allFinite()) {
      ROS_WARN("[RobotModel] IK produced non-finite velocity update.");
      return false;
    }

    const double max_step = 0.2;
    if (v.norm() > max_step) {
      v *= max_step / v.norm();
    }

    q = pinocchio::integrate(model_, q, ik_step_size_ * v);
    clampToPositionLimits(q);
  }

  q_solution = q;

  Eigen::Isometry3d T_final;
  getEndEffectorPose(q_solution, T_final);

  const double final_pos_err =
      (target_pose.translation() - T_final.translation()).norm();

  const Eigen::Matrix3d R_err =
      T_final.linear().transpose() * target_pose.linear();

  const Eigen::AngleAxisd aa_final(R_err);
  double final_rot_err = std::abs(aa_final.angle());
  if (!std::isfinite(final_rot_err)) {
    final_rot_err = std::numeric_limits<double>::infinity();
  }

  ROS_WARN_STREAM("[RobotModel] IK did not fully converge. final position error = "
                  << final_pos_err
                  << ", final rotation error = "
                  << final_rot_err);

  return final_pos_err < 5.0 * ik_pos_tol_
      && final_rot_err < 5.0 * ik_rot_tol_;
}

bool RobotModel::poseMsgToEigen(const geometry_msgs::PoseStamped& pose_msg,
                                Eigen::Isometry3d& T) const {
  const auto& p = pose_msg.pose.position;
  const auto& q_msg = pose_msg.pose.orientation;

  Eigen::Quaterniond q(q_msg.w, q_msg.x, q_msg.y, q_msg.z);
  if (q.norm() < 1e-12) {
    ROS_ERROR("[RobotModel] Invalid quaternion in poseMsgToEigen.");
    return false;
  }

  q.normalize();

  T = Eigen::Isometry3d::Identity();
  T.translation() = Eigen::Vector3d(p.x, p.y, p.z);
  T.linear() = q.toRotationMatrix();

  return true;
}

geometry_msgs::PoseStamped RobotModel::eigenToPoseMsg(const Eigen::Isometry3d& T,
                                                      const std::string& frame_id) const {
  geometry_msgs::PoseStamped msg;
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = frame_id;

  msg.pose.position.x = T.translation().x();
  msg.pose.position.y = T.translation().y();
  msg.pose.position.z = T.translation().z();

  Eigen::Quaterniond q(T.linear());
  q.normalize();

  msg.pose.orientation.x = q.x();
  msg.pose.orientation.y = q.y();
  msg.pose.orientation.z = q.z();
  msg.pose.orientation.w = q.w();

  return msg;
}

const SensorFrameManager& RobotModel::sensorFrameManager() const {
  return sensor_frame_manager_;
}

}  // namespace arm_model