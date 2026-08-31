#include "egocentric_arm_planner/local_sparse_scp_planner.hpp"

#include <piqp/piqp.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace egocentric_arm_planner {
namespace {

bool finiteVector(const Eigen::VectorXd& x) {
  for (int i = 0; i < x.size(); ++i) {
    if (!std::isfinite(x[i])) return false;
  }
  return true;
}

bool finiteMatrix(const Eigen::MatrixXd& x) {
  for (int r = 0; r < x.rows(); ++r) {
    for (int c = 0; c < x.cols(); ++c) {
      if (!std::isfinite(x(r, c))) return false;
    }
  }
  return true;
}

double clampValue(double x, double lo, double hi) {
  return std::max(lo, std::min(hi, x));
}

}  // namespace

LocalSparseSCPPlanner::~LocalSparseSCPPlanner() {
  stopWorker();
}

bool LocalSparseSCPPlanner::initialize(
    const ros::NodeHandle& nh,
    const ros::NodeHandle& pnh) {
  nh_ = nh;
  pnh_ = pnh;

  if (!loadConfig()) return false;
  if (!loadJointLimits()) return false;

  latest_executed_command_ = Eigen::VectorXd::Zero(dof_);
  latest_single_waypoint_q_ = Eigen::VectorXd::Zero(dof_);

  joint_state_sub_ = nh_.subscribe(
      joint_state_topic_, 1,
      &LocalSparseSCPPlanner::jointStateCallback, this);
  reference_sub_ = nh_.subscribe(
      reference_topic_, 1,
      &LocalSparseSCPPlanner::referenceCallback, this);
  waypoint_schedule_sub_ = nh_.subscribe(
      waypoint_schedule_topic_, 1,
      &LocalSparseSCPPlanner::waypointScheduleCallback, this);
  single_waypoint_active_sub_ = nh_.subscribe(
      single_waypoint_active_topic_, 1,
      &LocalSparseSCPPlanner::singleWaypointActiveCallback, this);
  single_waypoint_q_sub_ = nh_.subscribe(
      single_waypoint_q_topic_, 1,
      &LocalSparseSCPPlanner::singleWaypointQCallback, this);
  recovery_sub_ = nh_.subscribe(
      recovery_topic_, 1,
      &LocalSparseSCPPlanner::recoveryCallback, this);
  probe_active_sub_ = nh_.subscribe(
      probe_active_topic_, 1,
      &LocalSparseSCPPlanner::probeActiveCallback, this);
  replan_request_sub_ = nh_.subscribe(
      replan_request_topic_, 10,
      &LocalSparseSCPPlanner::replanRequestCallback, this);
  executed_command_sub_ = nh_.subscribe(
      executed_command_topic_, 2,
      &LocalSparseSCPPlanner::executedCommandCallback, this);
  execution_summary_sub_ = nh_.subscribe(
      execution_summary_topic_, 10,
      &LocalSparseSCPPlanner::executionSummaryCallback, this);
  cdf_batch_sub_ = nh_.subscribe(
      cdf_batch_topic_, 2,
      &LocalSparseSCPPlanner::cdfConstraintBatchCallback, this);

  query_trajectory_pub_ =
      nh_.advertise<trajectory_msgs::JointTrajectory>(
          query_trajectory_topic_, 1);
  candidate_trajectory_pub_ =
      nh_.advertise<trajectory_msgs::JointTrajectory>(
          candidate_trajectory_topic_, 1);
  summary_pub_ =
      nh_.advertise<std_msgs::String>(summary_topic_, 20, true);
  task_infeasible_pub_ =
      nh_.advertise<std_msgs::Bool>(task_infeasible_topic_, 10, false);
  task_uncertified_pub_ =
      nh_.advertise<std_msgs::Bool>(task_uncertified_topic_, 10, false);
  force_vbc_bootstrap_pub_ =
      nh_.advertise<std_msgs::Bool>(force_vbc_bootstrap_topic_, 1, true);

  std_msgs::Bool bootstrap_init;
  bootstrap_init.data = false;
  force_vbc_bootstrap_pub_.publish(bootstrap_init);

  timer_ = nh_.createTimer(
      ros::Duration(1.0 / planner_poll_rate_),
      &LocalSparseSCPPlanner::timerCallback, this);

  startWorker();

  ROS_WARN_STREAM(
      "[LocalSparseSCPPlanner] C5.4 EVENT-TRIGGERED LOCAL PLANNER ENABLED: "
      "explicit q/u multiple shooting + Sparse PIQP + CDF slacks + SCP. "
      "This node does NOT publish actuator commands.");
  ROS_INFO_STREAM(
      "[LocalSparseSCPPlanner] K=" << num_intervals_
      << " dt=" << dt_
      << " SCP=" << max_scp_iterations_
      << " query=" << query_trajectory_topic_
      << " batch=" << cdf_batch_topic_
      << " candidate=" << candidate_trajectory_topic_);

  return true;
}

bool LocalSparseSCPPlanner::loadConfig() {
  if (!pnh_.getParam("joint_names", joint_names_) ||
      joint_names_.empty()) {
    ROS_ERROR("[LocalSparseSCPPlanner] missing joint_names");
    return false;
  }
  dof_ = static_cast<int>(joint_names_.size());

  pnh_.param<int>("local_planner/num_intervals",
                  num_intervals_, num_intervals_);
  pnh_.param<double>("local_planner/horizon_duration",
                     horizon_duration_, horizon_duration_);
  if (num_intervals_ <= 0 || horizon_duration_ <= 0.0) {
    ROS_ERROR("[LocalSparseSCPPlanner] invalid horizon");
    return false;
  }
  dt_ = horizon_duration_ / static_cast<double>(num_intervals_);

  pnh_.param<double>("local_planner/poll_rate",
                     planner_poll_rate_, planner_poll_rate_);
  pnh_.param<double>("local_planner/min_replan_interval",
                     min_replan_interval_s_, min_replan_interval_s_);
  pnh_.param<double>("local_planner/cdf_wait_timeout",
                     cdf_wait_timeout_s_, cdf_wait_timeout_s_);
  pnh_.param<double>("local_planner/cdf_stamp_tolerance",
                     cdf_stamp_tolerance_s_, cdf_stamp_tolerance_s_);

  pnh_.param<int>("local_planner/scp/max_iterations",
                  max_scp_iterations_, max_scp_iterations_);
  pnh_.param<double>("local_planner/scp/step_tolerance_inf",
                     scp_step_tolerance_inf_,
                     scp_step_tolerance_inf_);
  pnh_.param<double>("local_planner/scp/trust_region_initial",
                     trust_region_initial_, trust_region_initial_);
  pnh_.param<double>("local_planner/scp/trust_region_min",
                     trust_region_min_, trust_region_min_);
  pnh_.param<double>("local_planner/scp/trust_region_max",
                     trust_region_max_, trust_region_max_);
  pnh_.param<double>("local_planner/scp/trust_region_grow",
                     trust_region_grow_, trust_region_grow_);
  pnh_.param<double>("local_planner/scp/trust_region_shrink",
                     trust_region_shrink_, trust_region_shrink_);
  pnh_.param<double>("local_planner/scp/improvement_tolerance",
                     trust_region_improvement_tol_,
                     trust_region_improvement_tol_);

  pnh_.param<double>("local_planner/q_tracking_weight",
                     q_tracking_weight_, q_tracking_weight_);
  pnh_.param<double>("local_planner/terminal_q_tracking_weight",
                     terminal_q_tracking_weight_,
                     terminal_q_tracking_weight_);
  pnh_.param<double>("local_planner/u_tracking_weight",
                     u_tracking_weight_, u_tracking_weight_);
  pnh_.param<bool>("local_planner/u_reference_tracking_enabled",
                   u_reference_tracking_enabled_,
                   u_reference_tracking_enabled_);
  pnh_.param<double>("local_planner/u_smooth_weight",
                     u_smooth_weight_, u_smooth_weight_);
  pnh_.param<bool>("local_planner/enforce_acceleration_constraints",
                   enforce_acceleration_constraints_,
                   enforce_acceleration_constraints_);
  pnh_.param<bool>("local_planner/repair_hold_initialization_enabled",
                   repair_hold_initialization_enabled_,
                   repair_hold_initialization_enabled_);
  pnh_.param<double>("local_planner/repair_task_tracking_scale",
                     repair_task_tracking_scale_,
                     repair_task_tracking_scale_);
  pnh_.param<double>("local_planner/visibility_waypoint_weight",
                     visibility_waypoint_weight_,
                     visibility_waypoint_weight_);

  pnh_.param<double>("local_planner/cdf/safety_margin",
                     cdf_safety_margin_, cdf_safety_margin_);
  pnh_.param<double>("local_planner/cdf/slack_linear_weight",
                     cdf_slack_linear_weight_,
                     cdf_slack_linear_weight_);
  pnh_.param<double>("local_planner/cdf/slack_quadratic_weight",
                     cdf_slack_quadratic_weight_,
                     cdf_slack_quadratic_weight_);
  pnh_.param<double>("local_planner/cdf/slack_upper_bound",
                     cdf_slack_upper_bound_,
                     cdf_slack_upper_bound_);
  pnh_.param<bool>("local_planner/cdf/per_constraint_slack",
                   cdf_per_constraint_slack_,
                   cdf_per_constraint_slack_);
  pnh_.param<bool>("local_planner/cdf/slack_use_upper_bound",
                   cdf_slack_use_upper_bound_,
                   cdf_slack_use_upper_bound_);
  pnh_.param<bool>("local_planner/cdf/slack_enabled",
                   cdf_slack_enabled_,
                   cdf_slack_enabled_);
  pnh_.param<bool>("local_planner/cdf/task_failure_slack_diagnostic_enabled",
                   task_failure_slack_diagnostic_enabled_,
                   task_failure_slack_diagnostic_enabled_);
  pnh_.param<double>("local_planner/cdf/task_failure_slack_diagnostic_weight",
                     task_failure_slack_diagnostic_weight_,
                     task_failure_slack_diagnostic_weight_);
  pnh_.param<bool>("local_planner/cdf/adaptive_slack_penalty",
                   cdf_adaptive_slack_penalty_,
                   cdf_adaptive_slack_penalty_);
  pnh_.param<double>("local_planner/cdf/slack_penalty_multiplier",
                     cdf_slack_penalty_multiplier_,
                     cdf_slack_penalty_multiplier_);
  pnh_.param<double>("local_planner/cdf/slack_penalty_max",
                     cdf_slack_penalty_max_,
                     cdf_slack_penalty_max_);
  pnh_.param<double>("local_planner/cdf/slack_tolerance",
                     cdf_slack_tolerance_,
                     cdf_slack_tolerance_);
  pnh_.param<bool>("local_planner/cdf/safe_row_screening",
                   cdf_safe_row_screening_,
                   cdf_safe_row_screening_);
  pnh_.param<double>("local_planner/cdf/linearization_tolerance_inf",
                     cdf_linearization_tolerance_inf_,
                     cdf_linearization_tolerance_inf_);
  pnh_.param<int>("local_planner/cdf/constraint_horizon_steps",
                  cdf_constraint_horizon_steps_,
                  cdf_constraint_horizon_steps_);

  pnh_.param<int>("local_planner/piqp/max_iterations",
                  piqp_max_iterations_, piqp_max_iterations_);
  pnh_.param<double>("local_planner/piqp/eps_abs",
                     piqp_eps_abs_, piqp_eps_abs_);
  pnh_.param<double>("local_planner/piqp/eps_rel",
                     piqp_eps_rel_, piqp_eps_rel_);
  pnh_.param<bool>("local_planner/piqp/verbose",
                   piqp_verbose_, piqp_verbose_);

  pnh_.param<double>("mpc/joint_position_margin",
                     joint_position_margin_,
                     joint_position_margin_);

  pnh_.param<std::string>("local_planner/joint_states",
                          joint_state_topic_, joint_state_topic_);
  pnh_.param<std::string>("local_planner/reference_trajectory",
                          reference_topic_, reference_topic_);
  pnh_.param<std::string>("local_planner/waypoint_schedule_topic",
                          waypoint_schedule_topic_,
                          waypoint_schedule_topic_);
  pnh_.param<std::string>("local_planner/single_waypoint_active_topic",
                          single_waypoint_active_topic_,
                          single_waypoint_active_topic_);
  pnh_.param<std::string>("local_planner/single_waypoint_q_topic",
                          single_waypoint_q_topic_,
                          single_waypoint_q_topic_);
  pnh_.param<std::string>("local_planner/recovery_topic",
                          recovery_topic_, recovery_topic_);
  pnh_.param<std::string>("local_planner/probe_active_topic",
                          probe_active_topic_, probe_active_topic_);
  pnh_.param<std::string>("local_planner/replan_request_topic",
                          replan_request_topic_, replan_request_topic_);
  pnh_.param<std::string>("local_planner/executed_command_topic",
                          executed_command_topic_,
                          executed_command_topic_);
  pnh_.param<std::string>("local_planner/execution_summary_topic",
                          execution_summary_topic_,
                          execution_summary_topic_);
  pnh_.param<std::string>("local_planner/cdf_batch_topic",
                          cdf_batch_topic_, cdf_batch_topic_);
  pnh_.param<std::string>("local_planner/query_trajectory_topic",
                          query_trajectory_topic_,
                          query_trajectory_topic_);
  pnh_.param<std::string>("local_planner/candidate_trajectory_topic",
                          candidate_trajectory_topic_,
                          candidate_trajectory_topic_);
  pnh_.param<std::string>("local_planner/summary_topic",
                          summary_topic_, summary_topic_);
  pnh_.param<std::string>("local_planner/task_infeasible_topic",
                          task_infeasible_topic_,
                          task_infeasible_topic_);
  pnh_.param<std::string>("local_planner/task_uncertified_topic",
                          task_uncertified_topic_,
                          task_uncertified_topic_);
  pnh_.param<std::string>("local_planner/force_vbc_bootstrap_topic",
                          force_vbc_bootstrap_topic_,
                          force_vbc_bootstrap_topic_);

  if (planner_poll_rate_ <= 0.0 ||
      max_scp_iterations_ < 1 ||
      trust_region_initial_ <= 0.0 ||
      trust_region_min_ <= 0.0 ||
      trust_region_max_ < trust_region_min_ ||
      cdf_slack_linear_weight_ < 0.0 ||
      cdf_slack_quadratic_weight_ < 0.0 ||
      (cdf_slack_use_upper_bound_ && cdf_slack_upper_bound_ <= 0.0) ||
      cdf_slack_penalty_multiplier_ < 1.0 ||
      cdf_slack_penalty_max_ < cdf_slack_linear_weight_ ||
      cdf_slack_tolerance_ < 0.0 ||
      cdf_constraint_horizon_steps_ < 1 ||
      cdf_constraint_horizon_steps_ > num_intervals_) {
    ROS_ERROR("[LocalSparseSCPPlanner] invalid local_planner parameters");
    return false;
  }

  return true;
}

bool LocalSparseSCPPlanner::loadJointLimits() {
  velocity_limits_ = Eigen::VectorXd::Zero(dof_);
  acceleration_limits_ = Eigen::VectorXd::Zero(dof_);
  q_min_ = Eigen::VectorXd::Zero(dof_);
  q_max_ = Eigen::VectorXd::Zero(dof_);

  for (int j = 0; j < dof_; ++j) {
    const std::string& name = joint_names_[static_cast<std::size_t>(j)];
    double v = 0.0, a = 0.0, lo = 0.0, hi = 0.0;
    if (!pnh_.getParam("mpc/joint_velocity_limits/" + name, v) ||
        !pnh_.getParam("mpc/joint_acceleration_limits/" + name, a) ||
        !pnh_.getParam("mpc/joint_position_limits/" + name + "/lower", lo) ||
        !pnh_.getParam("mpc/joint_position_limits/" + name + "/upper", hi)) {
      ROS_ERROR_STREAM(
          "[LocalSparseSCPPlanner] missing limits for " << name);
      return false;
    }
    if (!(v > 0.0) || !(a > 0.0) || !(lo < hi)) {
      ROS_ERROR_STREAM(
          "[LocalSparseSCPPlanner] invalid limits for " << name);
      return false;
    }
    velocity_limits_[j] = v;
    acceleration_limits_[j] = a;
    q_min_[j] = lo;
    q_max_[j] = hi;
  }
  return true;
}

void LocalSparseSCPPlanner::jointStateCallback(
    const sensor_msgs::JointStateConstPtr& msg) {
  if (!msg) return;
  std::lock_guard<std::mutex> lock(mutex_);
  latest_joint_state_ = *msg;
  latest_joint_state_received_ = ros::Time::now();
  has_joint_state_ = true;
}

void LocalSparseSCPPlanner::referenceCallback(
    const trajectory_msgs::JointTrajectoryConstPtr& msg) {
  if (!msg || msg->points.empty()) return;
  std::lock_guard<std::mutex> lock(mutex_);
  latest_reference_ = *msg;
  latest_reference_received_ = ros::Time::now();
  has_reference_ = true;
  requestPlanLocked("new_nominal_reference");
}

void LocalSparseSCPPlanner::waypointScheduleCallback(
    const std_msgs::Float64MultiArrayConstPtr& msg) {
  if (!msg || msg->data.size() % 9 != 0) return;

  std::vector<DeadlineWaypoint> incoming;
  incoming.reserve(msg->data.size() / 9);
  for (std::size_t r = 0; r < msg->data.size() / 9; ++r) {
    const std::size_t off = 9 * r;
    DeadlineWaypoint wp;
    wp.id = static_cast<long long>(std::llround(msg->data[off]));
    wp.deadline_abs_s = msg->data[off + 1];
    wp.q = Eigen::VectorXd::Zero(dof_);
    if (!std::isfinite(wp.deadline_abs_s) || wp.deadline_abs_s <= 0.0)
      return;
    for (int j = 0; j < dof_; ++j) {
      wp.q[j] = msg->data[off + 2 + static_cast<std::size_t>(j)];
    }
    if (!finiteVector(wp.q)) return;
    incoming.push_back(wp);
  }

  std::lock_guard<std::mutex> lock(mutex_);
  bool changed = incoming.size() != latest_schedule_.size();
  if (!changed) {
    for (std::size_t i = 0; i < incoming.size(); ++i) {
      if (incoming[i].id != latest_schedule_[i].id ||
          std::fabs(incoming[i].deadline_abs_s -
                    latest_schedule_[i].deadline_abs_s) > 1e-5 ||
          (incoming[i].q - latest_schedule_[i].q)
                  .lpNorm<Eigen::Infinity>() > 1e-5) {
        changed = true;
        break;
      }
    }
  }
  latest_schedule_ = incoming;
  if (changed && repair_mode_ && visibility_waypoint_weight_ > 0.0)
    requestPlanLocked("visibility_schedule_changed");
}

void LocalSparseSCPPlanner::singleWaypointActiveCallback(
    const std_msgs::BoolConstPtr& msg) {
  if (!msg) return;

  std::lock_guard<std::mutex> lock(mutex_);
  if (latest_single_waypoint_active_ == msg->data) return;

  latest_single_waypoint_active_ = msg->data;
  if (repair_mode_ && visibility_waypoint_weight_ > 0.0) {
    requestPlanLocked(
        msg->data
            ? "single_visibility_waypoint_activated"
            : "single_visibility_waypoint_deactivated");
  }
}

void LocalSparseSCPPlanner::singleWaypointQCallback(
    const std_msgs::Float64MultiArrayConstPtr& msg) {
  if (!msg ||
      msg->data.size() != static_cast<std::size_t>(dof_)) {
    return;
  }

  Eigen::VectorXd q(dof_);
  for (int j = 0; j < dof_; ++j) {
    q[j] = msg->data[static_cast<std::size_t>(j)];
  }
  if (!finiteVector(q)) return;

  std::lock_guard<std::mutex> lock(mutex_);
  const bool changed =
      !has_single_waypoint_q_ ||
      latest_single_waypoint_q_.size() != dof_ ||
      (q - latest_single_waypoint_q_)
              .lpNorm<Eigen::Infinity>() > 1e-5;

  latest_single_waypoint_q_ = q;
  has_single_waypoint_q_ = true;

  if (changed && repair_mode_ && visibility_waypoint_weight_ > 0.0) {
    requestPlanLocked("single_visibility_waypoint_q_changed");
  }
}

void LocalSparseSCPPlanner::recoveryCallback(
    const std_msgs::BoolConstPtr& msg) {
  if (!msg) return;
  bool clear_bootstrap = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (repair_mode_ != msg->data) {
      repair_mode_ = msg->data;
      requestPlanLocked(
          repair_mode_ ? "enter_repair" : "leave_repair");
      clear_bootstrap = !repair_mode_;
    }
  }
  if (clear_bootstrap) {
    std_msgs::Bool bootstrap_msg;
    bootstrap_msg.data = false;
    force_vbc_bootstrap_pub_.publish(bootstrap_msg);
  }
}

void LocalSparseSCPPlanner::probeActiveCallback(
    const std_msgs::BoolConstPtr& msg) {
  if (!msg) return;
  std::lock_guard<std::mutex> lock(mutex_);
  if (probe_mode_ == msg->data) return;
  probe_mode_ = msg->data;
  requestPlanLocked(
      probe_mode_ ? "enter_probe_normal" : "leave_probe_normal");
}

void LocalSparseSCPPlanner::replanRequestCallback(
    const std_msgs::BoolConstPtr& msg) {
  if (!msg || !msg->data) return;
  std::lock_guard<std::mutex> lock(mutex_);
  requestPlanLocked("external_replan_request");
}

void LocalSparseSCPPlanner::executedCommandCallback(
    const std_msgs::Float64MultiArrayConstPtr& msg) {
  if (!msg || msg->data.size() != static_cast<std::size_t>(dof_)) return;
  Eigen::VectorXd u(dof_);
  for (int j = 0; j < dof_; ++j)
    u[j] = msg->data[static_cast<std::size_t>(j)];
  if (!finiteVector(u)) return;

  std::lock_guard<std::mutex> lock(mutex_);
  latest_executed_command_ = u;
  latest_executed_command_received_ = ros::Time::now();
}

void LocalSparseSCPPlanner::executionSummaryCallback(
    const std_msgs::StringConstPtr& msg) {
  if (!msg) return;

  const bool complete =
      msg->data.find(" complete=1") != std::string::npos ||
      msg->data.rfind("complete=1", 0) == 0;

  std::lock_guard<std::mutex> lock(mutex_);
  const bool rising = complete && !latest_execution_complete_;
  latest_execution_complete_ = complete;

  if (rising && repair_mode_) {
    requestPlanLocked("repair_trajectory_complete");
  }
  // PROBE_NORMAL replanning is owned by the regime manager and is released
  // only after tracker complete=1 for the exact committed trajectory seq.
  // Do not use a bare complete edge here: it can belong to the previous mode.
}

void LocalSparseSCPPlanner::cdfConstraintBatchCallback(
    const care_collision_cdf::CollisionCDFConstraintBatchConstPtr& msg) {
  if (!msg) return;

  bool accepted = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++cdf_batch_received_;

    if (!plan_running_ || !waiting_for_cdf_) return;

    const double dt =
        std::fabs((msg->header.stamp - current_query_stamp_).toSec());
    if (dt > cdf_stamp_tolerance_s_) {
      ++cdf_stamp_miss_;
      return;
    }

    pending_batch_ = msg;
    waiting_for_cdf_ = false;
    accepted = true;
  }

  if (accepted) {
    // Latched diagnostic: if PIQP later spends too long in setup/solve, the
    // benchmark still tells us that timestamp matching and ROS transport worked.
    publishSummary("cdf_batch_accepted");
    worker_cv_.notify_one();
  }
}

void LocalSparseSCPPlanner::timerCallback(const ros::TimerEvent&) {
  bool should_start = false;
  bool should_abort = false;

  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (plan_running_ && waiting_for_cdf_ &&
        current_query_wall_.toSec() > 0.0 &&
        (ros::WallTime::now() - current_query_wall_).toSec() >
            cdf_wait_timeout_s_) {
      should_abort = true;
    }

    if (!plan_running_ && plan_requested_) {
      const bool interval_ok =
          last_plan_finish_time_.isZero() ||
          (ros::Time::now() - last_plan_finish_time_).toSec() >=
              min_replan_interval_s_;
      should_start = interval_ok;
    }
  }

  if (should_abort) {
    abortPlan("cdf_wait_timeout");
    return;
  }
  if (should_start) startPlan();
}

bool LocalSparseSCPPlanner::extractMeasuredQ(
    const sensor_msgs::JointState& msg,
    Eigen::VectorXd& q) const {
  std::unordered_map<std::string, std::size_t> index;
  for (std::size_t i = 0; i < msg.name.size(); ++i)
    index[msg.name[i]] = i;

  q = Eigen::VectorXd::Zero(dof_);
  for (int j = 0; j < dof_; ++j) {
    const auto it = index.find(joint_names_[static_cast<std::size_t>(j)]);
    if (it == index.end() || it->second >= msg.position.size())
      return false;
    q[j] = msg.position[it->second];
  }
  return finiteVector(q);
}

bool LocalSparseSCPPlanner::buildTrajectoryMapping(
    const trajectory_msgs::JointTrajectory& msg,
    std::vector<int>& mapping) const {
  std::unordered_map<std::string, int> index;
  for (std::size_t i = 0; i < msg.joint_names.size(); ++i)
    index[msg.joint_names[i]] = static_cast<int>(i);

  mapping.resize(static_cast<std::size_t>(dof_));
  for (int j = 0; j < dof_; ++j) {
    const auto it = index.find(joint_names_[static_cast<std::size_t>(j)]);
    if (it == index.end()) return false;
    mapping[static_cast<std::size_t>(j)] = it->second;
  }
  return true;
}

bool LocalSparseSCPPlanner::sampleReferencePosition(
    const trajectory_msgs::JointTrajectory& msg,
    const std::vector<int>& mapping,
    double t,
    Eigen::VectorXd& q) const {
  if (msg.points.empty()) return false;

  auto copy =
      [&](const trajectory_msgs::JointTrajectoryPoint& p) -> bool {
        if (p.positions.size() < msg.joint_names.size()) return false;
        q = Eigen::VectorXd::Zero(dof_);
        for (int j = 0; j < dof_; ++j) {
          const int idx = mapping[static_cast<std::size_t>(j)];
          q[j] = p.positions[static_cast<std::size_t>(idx)];
        }
        return finiteVector(q);
      };

  const double first = msg.points.front().time_from_start.toSec();
  const double last = msg.points.back().time_from_start.toSec();
  if (t <= first) return copy(msg.points.front());
  if (t >= last) return copy(msg.points.back());

  std::size_t hi = 1;
  while (hi < msg.points.size() &&
         msg.points[hi].time_from_start.toSec() < t) ++hi;
  if (hi >= msg.points.size()) return copy(msg.points.back());

  Eigen::VectorXd q0, q1;
  const auto& p0 = msg.points[hi - 1];
  const auto& p1 = msg.points[hi];
  if (!copy(p0)) return false;
  q0 = q;
  if (!copy(p1)) return false;
  q1 = q;

  const double t0 = p0.time_from_start.toSec();
  const double t1 = p1.time_from_start.toSec();
  const double h = t1 - t0;
  if (h <= 1e-12) return false;

  const double alpha = (t - t0) / h;
  q = (1.0 - alpha) * q0 + alpha * q1;
  return finiteVector(q);
}

bool LocalSparseSCPPlanner::buildReferenceHorizon(
    const Eigen::VectorXd& q_current,
    const trajectory_msgs::JointTrajectory& reference,
    const ros::Time& now,
    Eigen::MatrixXd& q_ref,
    Eigen::MatrixXd& u_ref,
    Eigen::MatrixXd& q_init,
    Eigen::MatrixXd& u_init) const {
  std::vector<int> mapping;
  if (!buildTrajectoryMapping(reference, mapping)) return false;

  q_ref = Eigen::MatrixXd::Zero(dof_, num_intervals_ + 1);
  u_ref = Eigen::MatrixXd::Zero(dof_, num_intervals_);
  q_init = Eigen::MatrixXd::Zero(dof_, num_intervals_ + 1);
  u_init = Eigen::MatrixXd::Zero(dof_, num_intervals_);

  q_ref.col(0) = q_current;
  q_init.col(0) = q_current;

  double reference_age = 0.0;
  if (!reference.header.stamp.isZero()) {
    reference_age =
        std::max(0.0, (now - reference.header.stamp).toSec());
  }

  for (int k = 1; k <= num_intervals_; ++k) {
    Eigen::VectorXd qk;
    if (!sampleReferencePosition(
            reference, mapping, reference_age + k * dt_, qk)) {
      return false;
    }
    q_ref.col(k) = qk;
  }

  for (int k = 0; k < num_intervals_; ++k) {
    Eigen::VectorXd ur =
        (q_ref.col(k + 1) - q_ref.col(k)) / dt_;
    for (int j = 0; j < dof_; ++j) {
      ur[j] = clampValue(
          ur[j], -velocity_limits_[j], velocity_limits_[j]);
    }
    u_ref.col(k) = ur;

    Eigen::VectorXd ui =
        (q_ref.col(k + 1) - q_init.col(k)) / dt_;
    for (int j = 0; j < dof_; ++j) {
      ui[j] = clampValue(
          ui[j], -velocity_limits_[j], velocity_limits_[j]);
    }
    u_init.col(k) = ui;
    q_init.col(k + 1) = q_init.col(k) + dt_ * ui;
  }

  return finiteMatrix(q_ref) && finiteMatrix(q_init) &&
         finiteMatrix(u_ref) && finiteMatrix(u_init);
}

void LocalSparseSCPPlanner::requestPlanLocked(
    const std::string& reason) {
  plan_requested_ = true;
  plan_request_reason_ = reason;
}

bool LocalSparseSCPPlanner::startPlan() {
  sensor_msgs::JointState joint_state;
  trajectory_msgs::JointTrajectory reference;
  std::vector<DeadlineWaypoint> schedule;
  bool single_waypoint_active = false;
  bool has_single_waypoint_q = false;
  Eigen::VectorXd single_waypoint_q;
  Eigen::VectorXd previous_command;
  bool repair = false;
  bool probe = false;
  std::string reason;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (plan_running_ || !plan_requested_) return false;
    if (!has_joint_state_ || !has_reference_) {
      ROS_WARN_THROTTLE(
          1.0,
          "[LocalSparseSCPPlanner] waiting for joint state/reference");
      return false;
    }

    joint_state = latest_joint_state_;
    reference = latest_reference_;
    schedule = latest_schedule_;
    single_waypoint_active = latest_single_waypoint_active_;
    has_single_waypoint_q = has_single_waypoint_q_;
    single_waypoint_q = latest_single_waypoint_q_;
    previous_command = latest_executed_command_;
    if (previous_command.size() != dof_)
      previous_command = Eigen::VectorXd::Zero(dof_);
    repair = repair_mode_;
    probe = probe_mode_;
    reason = plan_request_reason_;

    plan_requested_ = false;
    plan_running_ = true;
    waiting_for_cdf_ = false;
    pending_batch_.reset();
    ++plan_sequence_;
  }

  // C4.7/C4.9 blocker-aware acquisition intentionally publishes only the
  // currently selected persistent q_vis through the legacy single-waypoint
  // topics; its multi-deadline schedule topic stays empty.  The C5.4 local
  // planner accepts both interfaces. In REPAIR, fall back to a synthetic
  // terminal-horizon obligation for that active q_vis.
  if (!repair) {
    schedule.clear();
  } else if (schedule.empty() &&
             single_waypoint_active &&
             has_single_waypoint_q &&
             single_waypoint_q.size() == dof_) {
    DeadlineWaypoint wp;
    wp.id = -1;
    wp.deadline_abs_s =
        ros::Time::now().toSec() + horizon_duration_;
    wp.q = single_waypoint_q;
    schedule.push_back(std::move(wp));
  }

  // A REPAIR plan without an active visibility obligation has no steering
  // objective. Solving it would produce a no-op hold candidate, which then
  // stays "fresh" on the predicted-trajectory topic and prevents the VBC
  // selector from falling back to the task trajectory to discover the next
  // blocker. Treat this as a normal waiting state instead: request an immediate
  // task-trajectory bootstrap and publish no candidate until q_vis arrives.
  if (repair && schedule.empty()) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      plan_running_ = false;
      waiting_for_cdf_ = false;
      plan_repair_mode_ = true;
      plan_initialization_mode_ = "waiting_visibility_obligation";
      plan_schedule_.clear();
      last_plan_finish_time_ = ros::Time::now();
    }
    std_msgs::Bool bootstrap_msg;
    bootstrap_msg.data = true;
    force_vbc_bootstrap_pub_.publish(bootstrap_msg);
    publishSummary("waiting_visibility_obligation");
    ROS_INFO_THROTTLE(
        0.5,
        "[LocalSparseSCPPlanner] REPAIR waiting for visibility obligation; "
        "forcing task-trajectory VBC bootstrap");
    return false;
  }

  // A real visibility target is available, so normal predicted-trajectory VBC
  // selection may resume.
  std_msgs::Bool bootstrap_msg;
  bootstrap_msg.data = false;
  force_vbc_bootstrap_pub_.publish(bootstrap_msg);

  Eigen::VectorXd q_current;
  if (!extractMeasuredQ(joint_state, q_current)) {
    abortPlan("joint_state_decode_failed");
    return false;
  }

  Eigen::MatrixXd q_ref, u_ref, q_init, u_init;
  if (!buildReferenceHorizon(
          q_current, reference, ros::Time::now(),
          q_ref, u_ref, q_init, u_init)) {
    abortPlan("reference_horizon_failed");
    return false;
  }

  std::string initialization_mode = "task_reference";
  if (repair && repair_hold_initialization_enabled_) {
    for (int k = 0; k <= num_intervals_; ++k) {
      q_init.col(k) = q_current;
    }
    u_init.setZero();
    initialization_mode = "repair_hold";
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    plan_q_current_ = q_current;
    plan_previous_command_ = previous_command;
    plan_q_ref_ = q_ref;
    plan_u_ref_ = u_ref;
    plan_q_bar_ = q_init;
    plan_u_bar_ = u_init;
    plan_schedule_ = schedule;
    plan_repair_mode_ = repair;
    plan_probe_mode_ = probe;
    plan_initialization_mode_ = initialization_mode;
    scp_iteration_ = 0;
    trust_radius_ = trust_region_initial_;
    plan_cdf_slack_linear_weight_ = cdf_slack_linear_weight_;
    previous_query_min_distance_ =
        std::numeric_limits<double>::quiet_NaN();
    current_frame_id_ =
        reference.header.frame_id.empty()
            ? "base_link"
            : reference.header.frame_id;
    current_plan_start_wall_ = ros::WallTime::now();
  }

  const ros::Time stamp =
      publishQueryTrajectory(q_init, u_init, current_frame_id_);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    current_query_stamp_ = stamp;
    current_query_wall_ = ros::WallTime::now();
    waiting_for_cdf_ = true;
  }

  ROS_INFO_STREAM(
      "[LocalSparseSCPPlanner] plan " << plan_sequence_
      << " started reason=" << reason
      << " repair=" << static_cast<int>(repair)
      << " probe=" << static_cast<int>(probe)
      << " init=" << initialization_mode
      << " vis_obligations=" << schedule.size());
  publishSummary("plan_started");
  return true;
}

void LocalSparseSCPPlanner::abortPlan(const std::string& reason) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    plan_running_ = false;
    waiting_for_cdf_ = false;
    pending_batch_.reset();
    last_plan_finish_time_ = ros::Time::now();
    ++solve_failure_count_;
  }
  publishSummary("aborted_" + reason);
  ROS_WARN_STREAM(
      "[LocalSparseSCPPlanner] plan aborted: " << reason);
}

void LocalSparseSCPPlanner::startWorker() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (worker_.joinable()) return;
  worker_stop_ = false;
  worker_ = std::thread(
      &LocalSparseSCPPlanner::workerLoop, this);
}

void LocalSparseSCPPlanner::stopWorker() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    worker_stop_ = true;
    pending_batch_.reset();
  }
  worker_cv_.notify_all();
  if (worker_.joinable()) worker_.join();
}

void LocalSparseSCPPlanner::workerLoop() {
  while (true) {
    care_collision_cdf::CollisionCDFConstraintBatchConstPtr batch;
    Eigen::MatrixXd q_bar, u_bar, q_ref, u_ref;
    Eigen::VectorXd previous_command;
    std::vector<DeadlineWaypoint> schedule;
    bool repair = false;
    bool probe = false;
    double trust = 0.0;
    double slack_linear_weight = 0.0;
    int iteration = 0;
    ros::Time expected_stamp;

    {
      std::unique_lock<std::mutex> lock(mutex_);
      worker_cv_.wait(
          lock,
          [&]() {
            return worker_stop_ ||
                   static_cast<bool>(pending_batch_);
          });
      if (worker_stop_) return;

      batch = pending_batch_;
      pending_batch_.reset();
      if (!batch || !plan_running_) continue;

      repair = plan_repair_mode_;
      probe = plan_probe_mode_;
      const bool executable_prefix_mode = repair || probe;

      double current_min_d =
          std::numeric_limits<double>::infinity();
      for (std::size_t i = 0; i < batch->distance.size(); ++i) {
        if (i >= batch->original_timestep.size()) break;
        const int k = batch->original_timestep[i];
        if (k < 1 || k > num_intervals_) continue;
        if (executable_prefix_mode && k > cdf_constraint_horizon_steps_)
          continue;
        const double d = batch->distance[i];
        if (std::isfinite(d))
          current_min_d = std::min(current_min_d, d);
      }

      if (scp_iteration_ > 0 &&
          std::isfinite(previous_query_min_distance_) &&
          std::isfinite(current_min_d)) {
        const double improvement =
            current_min_d - previous_query_min_distance_;
        if (improvement > trust_region_improvement_tol_) {
          trust_radius_ = std::min(
              trust_region_max_,
              trust_radius_ * trust_region_grow_);
        } else if (improvement <
                   -trust_region_improvement_tol_) {
          trust_radius_ = std::max(
              trust_region_min_,
              trust_radius_ * trust_region_shrink_);
        }
      }

      q_bar = plan_q_bar_;
      u_bar = plan_u_bar_;
      q_ref = plan_q_ref_;
      u_ref = plan_u_ref_;
      previous_command = plan_previous_command_;
      schedule = plan_schedule_;
      trust = trust_radius_;
      slack_linear_weight = plan_cdf_slack_linear_weight_;
      iteration = scp_iteration_;
      expected_stamp = current_query_stamp_;
    }

    if (std::fabs(
            (batch->header.stamp - expected_stamp).toSec()) >
        cdf_stamp_tolerance_s_) {
      continue;
    }

    publishSummary("sparse_qp_started");

    SparseSolveResult result =
        solveSparseSubproblem(
            *batch,
            q_bar,
            u_bar,
            q_ref,
            u_ref,
            previous_command,
            schedule,
            repair,
            probe,
            trust,
            slack_linear_weight);

    bool publish_candidate = false;
    bool publish_next_query = false;
    Eigen::MatrixXd q_next, u_next;
    std::string frame;
    double total_ms = 0.0;

    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++solve_count_;
      if (!result.solved) {
        ++solve_failure_count_;
        plan_running_ = false;
        waiting_for_cdf_ = false;
        last_plan_finish_time_ = ros::Time::now();
      } else {
        plan_q_bar_ = result.q;
        plan_u_bar_ = result.u;
        previous_query_min_distance_ = result.min_distance;
        ++scp_iteration_;

        const bool slack_satisfied =
            !cdf_adaptive_slack_penalty_ ||
            result.max_slack <= cdf_slack_tolerance_;
        const bool converged =
            result.step_inf <= scp_step_tolerance_inf_ &&
            slack_satisfied;
        const bool exhausted =
            scp_iteration_ >= max_scp_iterations_;

        if (cdf_adaptive_slack_penalty_ &&
            !slack_satisfied && !exhausted) {
          plan_cdf_slack_linear_weight_ =
              std::min(
                  cdf_slack_penalty_max_,
                  std::max(
                      plan_cdf_slack_linear_weight_,
                      cdf_slack_linear_weight_) *
                      cdf_slack_penalty_multiplier_);
        }

        if (converged || exhausted) {
          publish_candidate = true;
          plan_running_ = false;
          waiting_for_cdf_ = false;
          last_plan_finish_time_ = ros::Time::now();
          total_ms =
              (ros::WallTime::now() - current_plan_start_wall_)
                  .toSec() * 1000.0;
        } else {
          publish_next_query = true;
          q_next = plan_q_bar_;
          u_next = plan_u_bar_;
          frame = current_frame_id_;
        }
      }
    }

    publishSummary(
        result.solved ? "scp_solved" : "scp_failed",
        &result,
        total_ms);

    if (!result.solved) {
      ROS_WARN_STREAM(
          "[LocalSparseSCPPlanner] sparse PIQP failed: "
          << result.status);

      if (!repair && task_failure_slack_diagnostic_enabled_) {
        const SparseSolveResult diagnostic =
            solveSparseSubproblem(
                *batch,
                q_bar,
                u_bar,
                q_ref,
                u_ref,
                previous_command,
                schedule,
                repair,
                probe,
                trust,
                task_failure_slack_diagnostic_weight_,
                true);
        publishSummary(
            "task_failure_slack_diagnostic",
            &diagnostic,
            0.0);
        ROS_WARN_STREAM(
            "[LocalSparseSCPPlanner] task failure slack diagnostic: "
            << "mode=" << (probe ? "PROBE_NORMAL" : "NORMAL")
            << " hard_status='" << result.status << "'"
            << " soft_solved=" << (diagnostic.solved ? 1 : 0)
            << " soft_status='" << diagnostic.status << "'"
            << " required_max_slack=" << diagnostic.max_slack
            << " required_mean_slack=" << diagnostic.mean_slack
            << " soft_primal=" << diagnostic.primal_residual);
      }

      if (!repair) {
        if (result.status.find("primal infeasible") != std::string::npos) {
          std_msgs::Bool infeasible_msg;
          infeasible_msg.data = true;
          task_infeasible_pub_.publish(infeasible_msg);
          ROS_WARN_STREAM(
              "[LocalSparseSCPPlanner] task QP infeasible in "
              << (probe ? "PROBE_NORMAL" : "NORMAL")
              << " -> publish task infeasible signal");
        } else if (
            result.status.find("max iterations") != std::string::npos ||
            result.status.find("maximum iterations") != std::string::npos) {
          std_msgs::Bool uncertified_msg;
          uncertified_msg.data = true;
          task_uncertified_pub_.publish(uncertified_msg);
          ROS_WARN_STREAM(
              "[LocalSparseSCPPlanner] task QP uncertified in "
              << (probe ? "PROBE_NORMAL" : "NORMAL")
              << " status='" << result.status
              << "' primal_res=" << result.primal_residual
              << " -> publish task uncertified signal");
        }
      }
      continue;
    }

    if (publish_candidate) {
      publishCandidateTrajectory(
          result.q, result.u, current_frame_id_);
      publishSummary("candidate_published", &result, total_ms);
      continue;
    }

    if (publish_next_query) {
      const ros::Time stamp =
          publishQueryTrajectory(q_next, u_next, frame);
      std::lock_guard<std::mutex> lock(mutex_);
      if (plan_running_) {
        current_query_stamp_ = stamp;
        current_query_wall_ = ros::WallTime::now();
        waiting_for_cdf_ = true;
      }
    }

    (void)iteration;
  }
}

int LocalSparseSCPPlanner::qIndex(int k, int j) const {
  // q_0 is measured/fixed; decision q starts at q_1.
  return (k - 1) * dof_ + j;
}

int LocalSparseSCPPlanner::uIndex(int k, int j) const {
  return num_intervals_ * dof_ + k * dof_ + j;
}

LocalSparseSCPPlanner::SparseSolveResult
LocalSparseSCPPlanner::solveSparseSubproblem(
    const care_collision_cdf::CollisionCDFConstraintBatch& batch,
    const Eigen::MatrixXd& q_bar,
    const Eigen::MatrixXd& u_bar,
    const Eigen::MatrixXd& q_ref,
    const Eigen::MatrixXd& u_ref,
    const Eigen::VectorXd& previous_command,
    const std::vector<DeadlineWaypoint>& schedule,
    bool repair_mode,
    bool probe_mode,
    double trust_radius,
    double slack_linear_weight,
    bool force_diagnostic_slack) const {
  SparseSolveResult out;
  const bool slack_enabled =
      cdf_slack_enabled_ || force_diagnostic_slack;
  const bool per_constraint_slack =
      cdf_per_constraint_slack_ || force_diagnostic_slack;
  out.slack_linear_weight_used = slack_linear_weight;
  out.batch_pairs = batch.num_pairs;
  out.min_distance = std::numeric_limits<double>::infinity();

  if (q_bar.rows() != dof_ ||
      q_bar.cols() != num_intervals_ + 1 ||
      u_bar.rows() != dof_ ||
      u_bar.cols() != num_intervals_ ||
      q_ref.rows() != dof_ ||
      q_ref.cols() != num_intervals_ + 1 ||
      u_ref.rows() != dof_ ||
      u_ref.cols() != num_intervals_) {
    out.status = "trajectory_dimension_error";
    return out;
  }

  const int n_pairs = batch.num_pairs;
  if (n_pairs < 0 ||
      batch.dof != dof_ ||
      batch.original_timestep.size() !=
          static_cast<std::size_t>(n_pairs) ||
      batch.distance.size() !=
          static_cast<std::size_t>(n_pairs) ||
      batch.q_linearization_flat.size() !=
          static_cast<std::size_t>(n_pairs * dof_) ||
      batch.gradient_flat.size() !=
          static_cast<std::size_t>(n_pairs * dof_)) {
    out.status = "cdf_batch_dimension_error";
    return out;
  }

  struct SelectedRow {
    int pair = -1;
    int k = -1;
    double d = 0.0;
    Eigen::VectorXd g;
    Eigen::VectorXd qlin;
  };
  std::vector<SelectedRow> selected;
  selected.reserve(static_cast<std::size_t>(n_pairs));

  for (int i = 0; i < n_pairs; ++i) {
    const int k =
        batch.original_timestep[static_cast<std::size_t>(i)];
    if (k == 0) {
      ++out.skipped_step0_rows;
      continue;
    }
    if (k < 1 || k > num_intervals_) {
      ++out.skipped_horizon_rows;
      continue;
    }
    const bool executable_prefix_mode = repair_mode || probe_mode;
    if (executable_prefix_mode && k > cdf_constraint_horizon_steps_) {
      ++out.skipped_safety_horizon_rows;
      continue;
    }

    const double d =
        batch.distance[static_cast<std::size_t>(i)];
    Eigen::VectorXd g(dof_);
    Eigen::VectorXd qlin(dof_);
    bool finite = std::isfinite(d);
    for (int j = 0; j < dof_; ++j) {
      g[j] = batch.gradient_flat[
          static_cast<std::size_t>(i * dof_ + j)];
      qlin[j] = batch.q_linearization_flat[
          static_cast<std::size_t>(i * dof_ + j)];
      finite = finite &&
               std::isfinite(g[j]) &&
               std::isfinite(qlin[j]);
    }
    if (!finite) {
      ++out.skipped_horizon_rows;
      continue;
    }

    out.min_distance = std::min(out.min_distance, d);
    out.qlin_error_inf = std::max(
        out.qlin_error_inf,
        (q_bar.col(k) - qlin).lpNorm<Eigen::Infinity>());

    if (cdf_safe_row_screening_) {
      const double worst_linearized =
          d - trust_radius * g.lpNorm<1>();
      if (worst_linearized >= cdf_safety_margin_) {
        ++out.screened_safe_rows;
        continue;
      }
    }

    SelectedRow row;
    row.pair = i;
    row.k = k;
    row.d = d;
    row.g = g;
    row.qlin = qlin;
    selected.push_back(std::move(row));
  }

  out.selected_cdf_rows = static_cast<int>(selected.size());

  if (!std::isfinite(out.min_distance))
    out.min_distance = std::numeric_limits<double>::quiet_NaN();

  if (n_pairs == 0) {
    ROS_INFO(
        "[LocalSparseSCPPlanner] received explicit empty CDF batch: no active forbidden pairs");
  }

  if (out.qlin_error_inf > cdf_linearization_tolerance_inf_) {
    out.status = "cdf_linearization_mismatch";
    return out;
  }

  // CARE historically shared one user slack across all CDF rows at a
  // timestep. G0 can switch to the GCDF convention: one slack per inequality.
  std::vector<int> step_to_slack(
      static_cast<std::size_t>(num_intervals_ + 1), -1);
  int n_s = 0;
  if (slack_enabled) {
    if (per_constraint_slack) {
      n_s = static_cast<int>(selected.size());
    } else {
      for (const auto& row : selected) {
        if (step_to_slack[static_cast<std::size_t>(row.k)] < 0) {
          step_to_slack[static_cast<std::size_t>(row.k)] = n_s++;
        }
      }
    }
  }

  const int n_q = num_intervals_ * dof_;
  const int n_u = num_intervals_ * dof_;
  const int n = n_q + n_u + n_s;

  const int n_eq = num_intervals_ * dof_;
  // Acceleration rows include the executed-command -> u0 boundary, all
  // inter-stage velocity changes, and u_{K-1} -> 0 terminal braking.
  const int n_acc =
      enforce_acceleration_constraints_
          ? (num_intervals_ + 1) * dof_
          : 0;
  const int n_cdf_rows = static_cast<int>(selected.size());
  const int n_ineq = n_acc + n_cdf_rows;

  using Triplet = Eigen::Triplet<double>;
  std::vector<Triplet> p_triplets;
  std::vector<Triplet> a_triplets;
  std::vector<Triplet> g_triplets;

  p_triplets.reserve(
      static_cast<std::size_t>(n * 4));
  a_triplets.reserve(
      static_cast<std::size_t>(n_eq * 3));
  g_triplets.reserve(
      static_cast<std::size_t>(
          n_acc * 2 + n_cdf_rows * (dof_ + 1)));

  Eigen::VectorXd c = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(n_eq);
  Eigen::VectorXd h_l =
      Eigen::VectorXd::Constant(n_ineq, -PIQP_INF);
  Eigen::VectorXd h_u =
      Eigen::VectorXd::Constant(n_ineq, PIQP_INF);
  Eigen::VectorXd x_l =
      Eigen::VectorXd::Constant(n, -PIQP_INF);
  Eigen::VectorXd x_u =
      Eigen::VectorXd::Constant(n, PIQP_INF);

  const double task_scale =
      repair_mode ? repair_task_tracking_scale_ : 1.0;

  auto addQuadraticTarget =
      [&](int idx, double weight, double target) {
        if (weight <= 0.0) return;
        p_triplets.emplace_back(idx, idx, 2.0 * weight);
        c[idx] += -2.0 * weight * target;
      };

  for (int k = 1; k <= num_intervals_; ++k) {
    const double w =
        task_scale *
        (q_tracking_weight_ +
         (k == num_intervals_
              ? terminal_q_tracking_weight_
              : 0.0));

    for (int j = 0; j < dof_; ++j) {
      const int qi = qIndex(k, j);
      addQuadraticTarget(qi, w, q_ref(j, k));

      const double lo = std::max(
          q_min_[j] + joint_position_margin_,
          q_bar(j, k) - trust_radius);
      const double hi = std::min(
          q_max_[j] - joint_position_margin_,
          q_bar(j, k) + trust_radius);
      if (lo > hi) {
        out.status = "trust_region_joint_limit_empty";
        return out;
      }
      x_l[qi] = lo;
      x_u[qi] = hi;
    }
  }

  // Visibility obligations are trajectory-level objectives. Each absolute
  // deadline selects the nearest future q_k in this local horizon.
  const double now_s = ros::Time::now().toSec();
  for (const auto& wp : schedule) {
    if (wp.q.size() != dof_) continue;
    const double rel = wp.deadline_abs_s - now_s;
    int k = static_cast<int>(std::ceil(rel / dt_));
    k = std::max(1, k);
    if (k > num_intervals_) continue;
    for (int j = 0; j < dof_; ++j) {
      addQuadraticTarget(
          qIndex(k, j),
          visibility_waypoint_weight_,
          wp.q[j]);
    }
  }

  for (int k = 0; k < num_intervals_; ++k) {
    for (int j = 0; j < dof_; ++j) {
      const int ui = uIndex(k, j);
      addQuadraticTarget(
          ui, u_tracking_weight_,
          u_reference_tracking_enabled_ ? u_ref(j, k) : 0.0);
      x_l[ui] = -velocity_limits_[j];
      x_u[ui] = velocity_limits_[j];
    }
  }

  // Smoothness: ||u0-u_executed||^2 + sum ||uk-u(k-1)||^2.
  for (int j = 0; j < dof_; ++j) {
    const int u0 = uIndex(0, j);
    p_triplets.emplace_back(
        u0, u0, 2.0 * u_smooth_weight_);
    const double prev =
        previous_command.size() == dof_
            ? previous_command[j]
            : 0.0;
    c[u0] += -2.0 * u_smooth_weight_ * prev;
  }
  for (int k = 1; k < num_intervals_; ++k) {
    for (int j = 0; j < dof_; ++j) {
      const int ua = uIndex(k - 1, j);
      const int ub = uIndex(k, j);
      p_triplets.emplace_back(
          ua, ua, 2.0 * u_smooth_weight_);
      p_triplets.emplace_back(
          ub, ub, 2.0 * u_smooth_weight_);
      p_triplets.emplace_back(
          ua, ub, -2.0 * u_smooth_weight_);
      p_triplets.emplace_back(
          ub, ua, -2.0 * u_smooth_weight_);
    }
  }

  // User CDF slacks guarantee every convex subproblem remains feasible even
  // when the current iterate is deeply inside forbidden space. G0 uses one
  // slack per CDF inequality, matching GCDF.
  const int slack0 = n_q + n_u;
  for (int s = 0; s < n_s; ++s) {
    const int si = slack0 + s;
    c[si] += slack_linear_weight;
    p_triplets.emplace_back(
        si, si, 2.0 * cdf_slack_quadratic_weight_);
    x_l[si] = 0.0;
    if (cdf_slack_use_upper_bound_ && !force_diagnostic_slack)
      x_u[si] = cdf_slack_upper_bound_;
  }

  // Small diagonal regularization makes P strictly positive definite enough
  // for stable sparse factorization without changing the optimizer materially.
  for (int i = 0; i < n; ++i)
    p_triplets.emplace_back(i, i, 1e-8);

  // Multiple-shooting dynamics:
  // q_{k+1} - q_k - dt*u_k = 0, with measured q0 moved to b.
  for (int k = 0; k < num_intervals_; ++k) {
    for (int j = 0; j < dof_; ++j) {
      const int row = k * dof_ + j;
      a_triplets.emplace_back(
          row, qIndex(k + 1, j), 1.0);
      a_triplets.emplace_back(
          row, uIndex(k, j), -dt_);
      if (k == 0) {
        b[row] = q_bar(j, 0);
      } else {
        a_triplets.emplace_back(
            row, qIndex(k, j), -1.0);
      }
    }
  }

  if (enforce_acceleration_constraints_) {
    // CARE acceleration envelope.
    for (int k = 0; k < num_intervals_; ++k) {
      for (int j = 0; j < dof_; ++j) {
        const int row = k * dof_ + j;
        g_triplets.emplace_back(
            row, uIndex(k, j), 1.0);

        const double du =
            acceleration_limits_[j] * dt_;
        if (k == 0) {
          const double prev =
              previous_command.size() == dof_
                  ? previous_command[j]
                  : 0.0;
          h_l[row] = prev - du;
          h_u[row] = prev + du;
        } else {
          g_triplets.emplace_back(
              row, uIndex(k - 1, j), -1.0);
          h_l[row] = -du;
          h_u[row] = du;
        }
      }
    }

    // CARE terminal braking row.
    for (int j = 0; j < dof_; ++j) {
      const int row = num_intervals_ * dof_ + j;
      const double du = acceleration_limits_[j] * dt_;
      g_triplets.emplace_back(
          row, uIndex(num_intervals_ - 1, j), 1.0);
      h_l[row] = -du;
      h_u[row] = du;
    }
  }

  // Linearized CDF:
  // d + g'(q_k-qbar) + s >= d_safe
  // -> g' q_k + s >= d_safe - d + g' qbar.
  for (int r = 0; r < n_cdf_rows; ++r) {
    const auto& row_data =
        selected[static_cast<std::size_t>(r)];
    const int row = n_acc + r;
    for (int j = 0; j < dof_; ++j) {
      g_triplets.emplace_back(
          row,
          qIndex(row_data.k, j),
          row_data.g[j]);
    }
    if (slack_enabled) {
      const int slack_slot =
          per_constraint_slack
              ? r
              : step_to_slack[static_cast<std::size_t>(row_data.k)];
      if (slack_slot < 0 || slack_slot >= n_s) {
        out.status = "cdf_slack_mapping_error";
        return out;
      }
      g_triplets.emplace_back(
          row, slack0 + slack_slot, 1.0);
    }
    h_l[row] =
        cdf_safety_margin_ - row_data.d +
        row_data.g.dot(row_data.qlin);
    h_u[row] = PIQP_INF;
  }

  Eigen::SparseMatrix<double> P(n, n);
  Eigen::SparseMatrix<double> A(n_eq, n);
  Eigen::SparseMatrix<double> G(n_ineq, n);
  P.setFromTriplets(
      p_triplets.begin(), p_triplets.end());
  A.setFromTriplets(
      a_triplets.begin(), a_triplets.end());
  G.setFromTriplets(
      g_triplets.begin(), g_triplets.end());
  P.makeCompressed();
  A.makeCompressed();
  G.makeCompressed();

  ROS_WARN_STREAM(
      "[LocalSparseSCPPlanner] sparse QP dimensions n=" << n
      << " q_vars=" << n_q
      << " u_vars=" << n_u
      << " cdf_slacks=" << n_s
      << " slack_enabled=" << (slack_enabled ? 1 : 0)
      << " per_constraint_slack=" << (per_constraint_slack ? 1 : 0)
      << " diagnostic_slack=" << (force_diagnostic_slack ? 1 : 0)
      << " accel_constraints="
      << (enforce_acceleration_constraints_ ? 1 : 0)
      << " eq=" << n_eq
      << " ineq=" << n_ineq
      << " selected_cdf_rows=" << n_cdf_rows
      << " nnz(P)=" << P.nonZeros()
      << " nnz(A)=" << A.nonZeros()
      << " nnz(G)=" << G.nonZeros());

  piqp::SparseSolver<double> solver;
  auto& settings = solver.settings();
  settings.max_iter =
      force_diagnostic_slack
          ? std::max(piqp_max_iterations_, 1000)
          : piqp_max_iterations_;
  settings.eps_abs = piqp_eps_abs_;
  settings.eps_rel = piqp_eps_rel_;
  settings.verbose = piqp_verbose_;
  settings.compute_timings = true;
  settings.kkt_solver = piqp::KKTSolver::sparse_ldlt;

  const ros::WallTime tic = ros::WallTime::now();
  ROS_WARN("[LocalSparseSCPPlanner] sparse PIQP setup begin");
  solver.setup(
      P, c,
      A, b,
      G, h_l, h_u,
      x_l, x_u);
  const double setup_ms =
      (ros::WallTime::now() - tic).toSec() * 1000.0;
  ROS_WARN_STREAM(
      "[LocalSparseSCPPlanner] sparse PIQP setup done in "
      << setup_ms << " ms; solve begin");
  const piqp::Status status = solver.solve();
  out.setup_and_solve_ms =
      (ros::WallTime::now() - tic).toSec() * 1000.0;
  ROS_WARN_STREAM(
      "[LocalSparseSCPPlanner] sparse PIQP solve returned status="
      << piqp::status_to_string(status)
      << " total_ms=" << out.setup_and_solve_ms);

  const auto& result = solver.result();
  out.iterations = static_cast<int>(result.info.iter);
  out.primal_residual = result.info.primal_res;
  out.dual_residual = result.info.dual_res;
  out.status = piqp::status_to_string(status);

  if (status != piqp::PIQP_SOLVED ||
      result.x.size() != n ||
      !finiteVector(result.x)) {
    return out;
  }

  out.q = Eigen::MatrixXd::Zero(
      dof_, num_intervals_ + 1);
  out.u = Eigen::MatrixXd::Zero(
      dof_, num_intervals_);
  out.q.col(0) = q_bar.col(0);

  for (int k = 1; k <= num_intervals_; ++k)
    for (int j = 0; j < dof_; ++j)
      out.q(j, k) = result.x[qIndex(k, j)];

  for (int k = 0; k < num_intervals_; ++k)
    for (int j = 0; j < dof_; ++j)
      out.u(j, k) = result.x[uIndex(k, j)];

  out.step_inf =
      (out.q - q_bar).cwiseAbs().maxCoeff();

  double slack_sum = 0.0;
  out.max_slack = 0.0;
  for (int s = 0; s < n_s; ++s) {
    const double value = result.x[slack0 + s];
    slack_sum += value;
    out.max_slack = std::max(out.max_slack, value);
  }
  out.mean_slack =
      n_s > 0 ? slack_sum / static_cast<double>(n_s) : 0.0;

  out.solved = true;
  return out;
}

trajectory_msgs::JointTrajectory
LocalSparseSCPPlanner::makeTrajectoryMessage(
    const Eigen::MatrixXd& q,
    const Eigen::MatrixXd& u,
    const std::string& frame_id,
    const ros::Time& stamp) const {
  trajectory_msgs::JointTrajectory msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;
  msg.joint_names = joint_names_;
  msg.points.resize(
      static_cast<std::size_t>(num_intervals_ + 1));

  for (int k = 0; k <= num_intervals_; ++k) {
    auto& p = msg.points[static_cast<std::size_t>(k)];
    p.time_from_start = ros::Duration(k * dt_);
    p.positions.resize(static_cast<std::size_t>(dof_));
    p.velocities.resize(static_cast<std::size_t>(dof_));
    p.accelerations.resize(static_cast<std::size_t>(dof_));

    for (int j = 0; j < dof_; ++j) {
      p.positions[static_cast<std::size_t>(j)] = q(j, k);
      p.velocities[static_cast<std::size_t>(j)] =
          k < num_intervals_ ? u(j, k) : 0.0;

      double a = 0.0;
      if (k == 0) {
        a = 0.0;
      } else if (k < num_intervals_) {
        a = (u(j, k) - u(j, k - 1)) / dt_;
      } else {
        a = -u(j, k - 1) / dt_;
      }
      p.accelerations[static_cast<std::size_t>(j)] = a;
    }
  }
  return msg;
}

ros::Time LocalSparseSCPPlanner::publishQueryTrajectory(
    const Eigen::MatrixXd& q,
    const Eigen::MatrixXd& u,
    const std::string& frame_id) {
  const ros::Time stamp = ros::Time::now();
  query_trajectory_pub_.publish(
      makeTrajectoryMessage(q, u, frame_id, stamp));
  return stamp;
}

void LocalSparseSCPPlanner::publishCandidateTrajectory(
    const Eigen::MatrixXd& q,
    const Eigen::MatrixXd& u,
    const std::string& frame_id) {
  candidate_trajectory_pub_.publish(
      makeTrajectoryMessage(
          q, u, frame_id, ros::Time::now()));
}

void LocalSparseSCPPlanner::publishSummary(
    const std::string& event,
    const SparseSolveResult* result,
    double total_plan_ms) {
  unsigned long long plan_seq = 0;
  unsigned long long batches = 0;
  unsigned long long misses = 0;
  unsigned long long solves = 0;
  unsigned long long failures = 0;
  int scp_iter = 0;
  double trust = 0.0;
  bool running = false;
  bool repair = false;
  bool probe = false;
  double active_slack_weight = 0.0;
  std::string init_mode = "unknown";

  {
    std::lock_guard<std::mutex> lock(mutex_);
    plan_seq = plan_sequence_;
    batches = cdf_batch_received_;
    misses = cdf_stamp_miss_;
    solves = solve_count_;
    failures = solve_failure_count_;
    scp_iter = scp_iteration_;
    trust = trust_radius_;
    running = plan_running_;
    repair = plan_repair_mode_;
    probe = plan_probe_mode_;
    active_slack_weight = plan_cdf_slack_linear_weight_;
    init_mode = plan_initialization_mode_;
  }

  std::ostringstream oss;
  oss << "C5_4_LOCAL_SCP"
      << " event=" << event
      << " plan_seq=" << plan_seq
      << " running=" << static_cast<int>(running)
      << " repair=" << static_cast<int>(repair)
      << " probe=" << static_cast<int>(probe)
      << " cdf_horizon_steps="
      << ((repair || probe) ? cdf_constraint_horizon_steps_ : num_intervals_)
      << " init=" << init_mode
      << " scp_iter=" << scp_iter
      << " trust_q_inf=" << trust
      << " slack_mu=" << active_slack_weight
      << " batches=" << batches
      << " stamp_miss=" << misses
      << " solves=" << solves
      << " solve_failures=" << failures;

  if (result) {
    oss << " solved=" << static_cast<int>(result->solved)
        << " status=" << result->status
        << " piqp_iter=" << result->iterations
        << " solve_ms=" << result->setup_and_solve_ms
        << " batch_pairs=" << result->batch_pairs
        << " cdf_rows=" << result->selected_cdf_rows
        << " screened_safe=" << result->screened_safe_rows
        << " skipped_step0=" << result->skipped_step0_rows
        << " skipped_horizon=" << result->skipped_horizon_rows
        << " skipped_safety_horizon="
        << result->skipped_safety_horizon_rows
        << " qlin_error_inf=" << result->qlin_error_inf
        << " min_d=" << result->min_distance
        << " max_slack=" << result->max_slack
        << " mean_slack=" << result->mean_slack
        << " slack_mu_used=" << result->slack_linear_weight_used
        << " step_inf=" << result->step_inf
        << " primal=" << result->primal_residual
        << " dual=" << result->dual_residual;
  }
  if (total_plan_ms > 0.0)
    oss << " total_plan_ms=" << total_plan_ms;

  std_msgs::String msg;
  msg.data = oss.str();
  summary_pub_.publish(msg);
  ROS_INFO_STREAM_THROTTLE(
      0.25, "[LocalSparseSCPPlanner] " << msg.data);
}

}  // namespace egocentric_arm_planner
