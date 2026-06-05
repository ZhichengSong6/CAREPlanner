#include "egocentric_arm_planner/receding_horizon_planner.hpp"

#include <algorithm>
#include <cmath>

namespace egocentric_arm_planner {

bool RecedingHorizonPlanner::initialize(const ros::NodeHandle& nh,
                                        const ros::NodeHandle& pnh) {
  nh_ = nh;
  pnh_ = pnh;

  pnh_.param<double>("receding_horizon/planning_rate",
                     planning_rate_,
                     planning_rate_);

  pnh_.param<std::string>("topics/joint_states",
                          joint_state_topic_,
                          joint_state_topic_);

  pnh_.param<std::string>("topics/target_pose",
                          target_pose_topic_,
                          target_pose_topic_);

  pnh_.param<std::string>("topics/task_trajectory",
                          task_trajectory_topic_,
                          task_trajectory_topic_);

  pnh_.param<std::string>("topics/command_trajectory",
                          command_trajectory_topic_,
                          command_trajectory_topic_);

  pnh_.param<bool>("debug/publish_task_trajectory",
                   publish_task_trajectory_,
                   publish_task_trajectory_);

  pnh_.param<double>("debug/overrun_warn_ratio",
                     overrun_warn_ratio_,
                     overrun_warn_ratio_);

  if (planning_rate_ <= 0.0) {
    ROS_ERROR("[RecedingHorizonPlanner] planning_rate must be positive.");
    return false;
  }

  robot_model_ = std::make_shared<arm_model::RobotModel>();
  if (!robot_model_->initializeFromRosParam(pnh_)) {
    ROS_ERROR("[RecedingHorizonPlanner] Failed to initialize RobotModel.");
    return false;
  }

  if (!task_generator_.initialize(pnh_, robot_model_)) {
    ROS_ERROR("[RecedingHorizonPlanner] Failed to initialize TaskTrajectoryGenerator.");
    return false;
  }

  if (!dummy_evaluator_.initialize(pnh_)) {
    ROS_ERROR("[RecedingHorizonPlanner] Failed to initialize DummyTrajectoryEvaluator.");
    return false;
  }

  if (!intervention_manager_.initialize(pnh_)) {
    ROS_ERROR("[RecedingHorizonPlanner] Failed to initialize InterventionManager.");
    return false;
  }

  joint_state_sub_ = nh_.subscribe(
      joint_state_topic_,
      1,
      &RecedingHorizonPlanner::jointStateCallback,
      this);

  target_pose_sub_ = nh_.subscribe(
      target_pose_topic_,
      1,
      &RecedingHorizonPlanner::targetPoseCallback,
      this);

  task_traj_pub_ = nh_.advertise<trajectory_msgs::JointTrajectory>(
      task_trajectory_topic_,
      1,
      false);

  command_traj_pub_ = nh_.advertise<trajectory_msgs::JointTrajectory>(
      command_trajectory_topic_,
      1,
      false);

  planning_timer_ = nh_.createTimer(
      ros::Duration(1.0 / planning_rate_),
      &RecedingHorizonPlanner::planningTimerCallback,
      this);

  ROS_INFO("[RecedingHorizonPlanner] Initialized.");
  ROS_INFO_STREAM("[RecedingHorizonPlanner] planning_rate = " << planning_rate_);
  ROS_INFO_STREAM("[RecedingHorizonPlanner] joint_state_topic = " << joint_state_topic_);
  ROS_INFO_STREAM("[RecedingHorizonPlanner] target_pose_topic = " << target_pose_topic_);
  ROS_INFO_STREAM("[RecedingHorizonPlanner] task_trajectory_topic = " << task_trajectory_topic_);
  ROS_INFO_STREAM("[RecedingHorizonPlanner] command_trajectory_topic = " << command_trajectory_topic_);
  ROS_WARN("[RecedingHorizonPlanner] Phase I node only publishes trajectory messages. It does NOT directly control Gazebo or robot states.");

  return true;
}

void RecedingHorizonPlanner::jointStateCallback(
    const sensor_msgs::JointStateConstPtr& msg) {
  if (!msg) {
    return;
  }

  std::lock_guard<std::mutex> lock(data_mutex_);
  latest_joint_state_ = *msg;
  has_joint_state_ = true;
}

void RecedingHorizonPlanner::targetPoseCallback(
    const geometry_msgs::PoseStampedConstPtr& msg) {
  if (!msg) {
    return;
  }

  std::lock_guard<std::mutex> lock(data_mutex_);
  latest_target_pose_ = *msg;
  has_target_pose_ = true;
}

bool RecedingHorizonPlanner::hasValidInputs() const {
  if (!has_joint_state_) {
    ROS_WARN_THROTTLE(1.0, "[RecedingHorizonPlanner] Waiting for joint state.");
    return false;
  }

  if (!has_target_pose_) {
    ROS_WARN_THROTTLE(1.0, "[RecedingHorizonPlanner] Waiting for target pose.");
    return false;
  }

  return true;
}

void RecedingHorizonPlanner::planningTimerCallback(const ros::TimerEvent&) {
  const ros::WallTime t_start = ros::WallTime::now();

  const bool ok = runOnePlanningStep();

  const double elapsed_ms =
      (ros::WallTime::now() - t_start).toSec() * 1000.0;

  const double period_ms = 1000.0 / planning_rate_;

  if (elapsed_ms > overrun_warn_ratio_ * period_ms) {
    ROS_WARN_STREAM_THROTTLE(
        1.0,
        "[RecedingHorizonPlanner] Planning overrun. elapsed = "
            << elapsed_ms
            << " ms, period = "
            << period_ms
            << " ms.");
  }

  if (!ok) {
    ROS_WARN_THROTTLE(
        1.0,
        "[RecedingHorizonPlanner] Planning step failed. No new command trajectory published.");
  }
}

bool RecedingHorizonPlanner::runOnePlanningStep() {
  sensor_msgs::JointState joint_state;
  geometry_msgs::PoseStamped target_pose;

  {
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (!hasValidInputs()) {
      return false;
    }

    joint_state = latest_joint_state_;
    target_pose = latest_target_pose_;
  }

  if (!robot_model_->updateJointState(joint_state)) {
    ROS_WARN_THROTTLE(1.0, "[RecedingHorizonPlanner] Failed to update RobotModel from JointState.");
    return false;
  }

  const Eigen::VectorXd q_current = robot_model_->getCurrentQ();
  const Eigen::VectorXd dq_current = robot_model_->getCurrentDq();

  arm_trajectory::JointTrajectory tau_task;
  const PlannerStatus gen_status =
      task_generator_.generate(q_current, dq_current, target_pose, tau_task);

  if (gen_status != PlannerStatus::SUCCESS) {
    ROS_WARN_STREAM_THROTTLE(
        1.0,
        "[RecedingHorizonPlanner] Task trajectory generation failed: "
            << plannerStatusToString(gen_status));
    return false;
  }

  EvaluationResult eval;
  const PlannerStatus eval_status =
      dummy_evaluator_.evaluate(tau_task, eval);

  if (eval_status != PlannerStatus::SUCCESS) {
    ROS_WARN_STREAM_THROTTLE(
        1.0,
        "[RecedingHorizonPlanner] Dummy evaluation failed: "
            << plannerStatusToString(eval_status)
            << ", message: "
            << eval.message);
    return false;
  }

  arm_trajectory::JointTrajectory tau_cmd;
  const PlannerStatus intervention_status =
      intervention_manager_.decideCommand(
          tau_task,
          eval,
          q_current,
          tau_cmd);

  if (intervention_status != PlannerStatus::SUCCESS) {
    ROS_WARN_STREAM_THROTTLE(
        1.0,
        "[RecedingHorizonPlanner] Intervention failed: "
            << plannerStatusToString(intervention_status));
    return false;
  }

  trajectory_msgs::JointTrajectory task_msg;
  trajectory_msgs::JointTrajectory cmd_msg;

  if (publish_task_trajectory_) {
    if (convertToRosTrajectory(tau_task, robot_model_->baseFrame(), task_msg)) {
      task_traj_pub_.publish(task_msg);
    }
  }

  if (!convertToRosTrajectory(tau_cmd, robot_model_->baseFrame(), cmd_msg)) {
    ROS_WARN_THROTTLE(
        1.0,
        "[RecedingHorizonPlanner] Failed to convert tau_cmd to ROS JointTrajectory.");
    return false;
  }

  command_traj_pub_.publish(cmd_msg);

  ROS_INFO_STREAM_THROTTLE(
      1.0,
      "[RecedingHorizonPlanner] Published command trajectory. "
          << "mode = "
          << interventionModeToString(eval.mode)
          << ", tau_task duration = "
          << tau_task.duration()
          << ", tau_cmd duration = "
          << tau_cmd.duration()
          << ", tau_cmd points = "
          << tau_cmd.size());

  return true;
}

bool RecedingHorizonPlanner::convertToRosTrajectory(
    const arm_trajectory::JointTrajectory& traj,
    const std::string& frame_id,
    trajectory_msgs::JointTrajectory& msg) const {
  msg = trajectory_msgs::JointTrajectory();

  if (traj.empty()) {
    ROS_WARN("[RecedingHorizonPlanner] Cannot convert empty JointTrajectory.");
    return false;
  }

  const auto& joint_names = robot_model_->jointNames();

  if (joint_names.empty()) {
    ROS_ERROR("[RecedingHorizonPlanner] RobotModel joint_names is empty.");
    return false;
  }

  const int nq = robot_model_->nq();
  const int nv = robot_model_->nv();

  if (traj.dof() != nq) {
    ROS_ERROR_STREAM(
        "[RecedingHorizonPlanner] Trajectory dof does not match RobotModel nq. "
            << "traj.dof = "
            << traj.dof()
            << ", nq = "
            << nq);
    return false;
  }

  std::vector<int> q_indices;
  std::vector<int> v_indices;

  q_indices.reserve(joint_names.size());
  v_indices.reserve(joint_names.size());

  for (const auto& joint_name : joint_names) {
    int q_idx = -1;
    int v_idx = -1;

    if (!robot_model_->getJointQIndex(joint_name, q_idx)) {
      ROS_ERROR_STREAM("[RecedingHorizonPlanner] Failed to get q index for joint: "
                       << joint_name);
      return false;
    }

    if (!robot_model_->getJointVIndex(joint_name, v_idx)) {
      ROS_ERROR_STREAM("[RecedingHorizonPlanner] Failed to get v index for joint: "
                       << joint_name);
      return false;
    }

    if (q_idx < 0 || q_idx >= nq || v_idx < 0 || v_idx >= nv) {
      ROS_ERROR_STREAM("[RecedingHorizonPlanner] Invalid q/v index for joint: "
                       << joint_name
                       << ", q_idx = "
                       << q_idx
                       << ", v_idx = "
                       << v_idx);
      return false;
    }

    q_indices.push_back(q_idx);
    v_indices.push_back(v_idx);
  }

  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = frame_id;
  msg.joint_names = joint_names;

  msg.points.reserve(traj.size());

  const auto& times = traj.times();
  const auto& positions = traj.positions();
  const auto& velocities = traj.velocities();
  const auto& accelerations = traj.accelerations();

  for (std::size_t i = 0; i < traj.size(); ++i) {
    trajectory_msgs::JointTrajectoryPoint point;

    point.time_from_start = ros::Duration(times[i] - traj.startTime());

    point.positions.resize(joint_names.size());
    point.velocities.resize(joint_names.size());
    point.accelerations.resize(joint_names.size());

    for (std::size_t j = 0; j < joint_names.size(); ++j) {
      const int q_idx = q_indices[j];
      const int v_idx = v_indices[j];

      point.positions[j] = positions[i][q_idx];
      point.velocities[j] = velocities[i][v_idx];
      point.accelerations[j] = accelerations[i][v_idx];
    }

    msg.points.push_back(point);
  }

  return true;
}

}  // namespace egocentric_arm_planner