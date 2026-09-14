#include "egocentric_arm_planner/local_sparse_scp_planner.hpp"
#include "egocentric_arm_planner/visibility_objective_step.hpp"
#include "egocentric_arm_planner/measured_braking_seed.hpp"
#include "egocentric_arm_planner/qp_box_conflict.hpp"
#include "egocentric_arm_planner/candidate_audit_context.hpp"
#include <iomanip>

#include <piqp/piqp.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <tuple>
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

bool parseUnsignedToken(
    const std::string& text,
    const std::string& key,
    unsigned long long* value) {
  if (value == nullptr) return false;
  const std::string needle = key + "=";
  const std::size_t pos = text.find(needle);
  if (pos == std::string::npos) return false;
  const std::size_t begin = pos + needle.size();
  std::size_t end = begin;
  while (end < text.size() &&
         text[end] >= '0' && text[end] <= '9') {
    ++end;
  }
  if (end == begin) return false;
  try {
    *value = std::stoull(text.substr(begin, end - begin));
    return true;
  } catch (...) {
    return false;
  }
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
  latest_frontier_.q = Eigen::VectorXd::Zero(dof_);
  plan_frontier_.q = Eigen::VectorXd::Zero(dof_);

  joint_state_sub_ = nh_.subscribe(
      joint_state_topic_, 1,
      &LocalSparseSCPPlanner::jointStateCallback, this);
  reference_sub_ = nh_.subscribe(
      reference_topic_, 1,
      &LocalSparseSCPPlanner::referenceCallback, this);
  task_reference_status_sub_ = nh_.subscribe(
      "/care_planner/task_reference_status", 10,
      &LocalSparseSCPPlanner::taskReferenceStatusCallback, this);
  waypoint_schedule_sub_ = nh_.subscribe(
      waypoint_schedule_topic_, 1,
      &LocalSparseSCPPlanner::waypointScheduleCallback, this);
  visibility_frontier_sub_ = nh_.subscribe(
      visibility_frontier_topic_, 1,
      &LocalSparseSCPPlanner::visibilityFrontierCallback, this);
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
  smooth_replan_request_sub_ = nh_.subscribe(
      smooth_replan_request_topic_, 10,
      &LocalSparseSCPPlanner::smoothReplanRequestCallback, this);
  executed_command_sub_ = nh_.subscribe(
      executed_command_topic_, 2,
      &LocalSparseSCPPlanner::executedCommandCallback, this);
  execution_summary_sub_ = nh_.subscribe(
      execution_summary_topic_, 10,
      &LocalSparseSCPPlanner::executionSummaryCallback, this);
  cdf_batch_sub_ = nh_.subscribe(
      cdf_batch_topic_, 2,
      &LocalSparseSCPPlanner::cdfConstraintBatchCallback, this);
  witness_request_pub_ = nh_.advertise<care_collision_cdf::CollisionCDFWitnessRequest>(
      cdf_batch_topic_ + "/witness_request", 2);
  witness_response_sub_ = nh_.subscribe(cdf_batch_topic_ + "/witness_response", 2,
      &LocalSparseSCPPlanner::witnessResponseCallback, this);

  query_trajectory_pub_ =
      nh_.advertise<trajectory_msgs::JointTrajectory>(
          query_trajectory_topic_, 1);
  candidate_trajectory_pub_ =
      nh_.advertise<trajectory_msgs::JointTrajectory>(
          candidate_trajectory_topic_, 1);
  observation_identity_pub_ = nh_.advertise<std_msgs::String>(
      "/care_planner/local_planner/observation_candidate_identity", 100);
  std::string final_outcome_topic;
  pnh_.param<std::string>("local_planner/verification_outcome_topic", final_outcome_topic,
                         "/care_planner/verification_outcome");
  final_verification_sub_ = nh_.subscribe(final_outcome_topic, 100,
      &LocalSparseSCPPlanner::finalVerificationCallback, this);
  gcdf_rejection_feedback_sub_ = nh_.subscribe(
      candidate_trajectory_topic_ + "/gcdf_rejection_feedback", 4,
      &LocalSparseSCPPlanner::gcdfRejectionFeedbackCallback, this);
  pnh_.param("local_planner/rejection_snapshots_enabled", rejection_snapshots_enabled_, false);
  candidate_audit_context_pub_ = nh_.advertise<std_msgs::String>(
      candidate_trajectory_topic_ + "/audit_context", 4);
  summary_pub_ =
      nh_.advertise<std_msgs::String>(summary_topic_, 20, true);
  witness_diagnostic_pub_ = nh_.advertise<std_msgs::String>(summary_topic_ + "/witness", 100);
  task_infeasible_pub_ =
      nh_.advertise<std_msgs::Bool>(task_infeasible_topic_, 10, false);
  task_obstacle_blocked_pub_ =
      nh_.advertise<std_msgs::Bool>(task_obstacle_blocked_topic_, 10, false);
  task_uncertified_pub_ =
      nh_.advertise<std_msgs::Bool>(task_uncertified_topic_, 10, false);
  task_stall_pub_ = nh_.advertise<std_msgs::String>(
      "/care_planner/local_planner/task_stall", 10, true);
  force_vbc_bootstrap_pub_ =
      nh_.advertise<std_msgs::Bool>(force_vbc_bootstrap_topic_, 1, true);
  gcdf_recovery_trajectory_pub_ =
      nh_.advertise<trajectory_msgs::JointTrajectory>(
          gcdf_recovery_trajectory_topic_, 2, false);
  gcdf_recovery_event_pub_ =
      nh_.advertise<std_msgs::Float64MultiArray>(
          gcdf_recovery_event_topic_, 2, false);

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
  pnh_.param<double>("local_planner/handoff_velocity_weight",
                     handoff_velocity_weight_, handoff_velocity_weight_);
  pnh_.param<bool>("local_planner/enforce_acceleration_constraints",
                   enforce_acceleration_constraints_,
                   enforce_acceleration_constraints_);
  pnh_.param<bool>("local_planner/repair_hold_initialization_enabled",
                   repair_hold_initialization_enabled_,
                   repair_hold_initialization_enabled_);
  pnh_.param<double>("local_planner/repair_task_tracking_scale",
                     repair_task_tracking_scale_,
                     repair_task_tracking_scale_);
  pnh_.param<int>("local_planner/probe_task_horizon_steps",
                  probe_task_horizon_steps_,
                  probe_task_horizon_steps_);
  pnh_.param<double>("local_planner/visibility_waypoint_weight",
                     visibility_waypoint_weight_,
                     visibility_waypoint_weight_);
  pnh_.param<int>("local_planner/visibility_frontier_horizon_step",
                  visibility_frontier_horizon_step_,
                  visibility_frontier_horizon_step_);

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
  pnh_.param<bool>("local_planner/cdf/probe_feasibility_restoration_enabled",
                   probe_feasibility_restoration_enabled_,
                   probe_feasibility_restoration_enabled_);
  pnh_.param<int>("local_planner/cdf/probe_feasibility_restoration_max_attempts",
                  probe_feasibility_restoration_max_attempts_,
                  probe_feasibility_restoration_max_attempts_);
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
  pnh_.param<std::string>("local_planner/visibility_frontier_topic",
                          visibility_frontier_topic_,
                          visibility_frontier_topic_);
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
  pnh_.param<std::string>("local_planner/smooth_replan_request_topic",
                          smooth_replan_request_topic_,
                          smooth_replan_request_topic_);
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
  pnh_.param<std::string>("local_planner/task_obstacle_blocked_topic",
                          task_obstacle_blocked_topic_,
                          task_obstacle_blocked_topic_);
  pnh_.param<std::string>("local_planner/task_uncertified_topic",
                          task_uncertified_topic_,
                          task_uncertified_topic_);
  pnh_.param<std::string>("local_planner/force_vbc_bootstrap_topic",
                          force_vbc_bootstrap_topic_,
                          force_vbc_bootstrap_topic_);
  pnh_.param<std::string>("local_planner/gcdf_recovery_trajectory_topic",
                          gcdf_recovery_trajectory_topic_,
                          gcdf_recovery_trajectory_topic_);
  pnh_.param<std::string>("local_planner/gcdf_recovery_event_topic",
                          gcdf_recovery_event_topic_,
                          gcdf_recovery_event_topic_);

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
      probe_feasibility_restoration_max_attempts_ < 0 ||
      probe_task_horizon_steps_ < 1 ||
      probe_task_horizon_steps_ > num_intervals_ ||
      visibility_frontier_horizon_step_ < 1 ||
      visibility_frontier_horizon_step_ > num_intervals_ ||
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
  Eigen::VectorXd measured;
  if (extractMeasuredQ(*msg, measured)) {
    const bool vbc_was_blocked = final_vbc_no_progress_.progress.blocked();
    if (final_vbc_no_progress_.progress.observe(measured) && vbc_was_blocked && repair_mode_)
      requestPlanLocked("measured_progress_after_final_vbc_stall");
    const bool repair_was_blocked = repair_no_progress_.blocked();
    if (repair_no_progress_.observe(measured) && repair_was_blocked && repair_mode_)
      requestPlanLocked("measured_progress_after_repair_qp_stall");
    const bool was_blocked = task_no_progress_.blocked();
    if (task_no_progress_.observe(measured) && was_blocked) {
      std_msgs::String status; status.data = "status=reset reason=measured_progress";
      task_stall_pub_.publish(status);
      requestPlanLocked("measured_progress_after_qp_stall");
    }
  }
}

void LocalSparseSCPPlanner::taskReferenceStatusCallback(
    const std_msgs::StringConstPtr& msg) {
  if (!msg) return;
  unsigned long long id = 0, stamp = 0;
  if (!parseUnsignedToken(msg->data, "request_id", &id) ||
      !parseUnsignedToken(msg->data, "request_ros_ns", &stamp)) return;
  std::lock_guard<std::mutex> lock(mutex_);
  const bool exhausted = msg->data.find("status=exhausted") != std::string::npos;
  if (!task_reference_receipt_.status(id, stamp, exhausted)) return;
  // Different ROS topics may be delivered out of order: do not invalidate a
  // fresh reference just because its earlier 'pending' status arrives later.
  if (!task_reference_receipt_.pending && !task_reference_receipt_.failed &&
      latest_reference_.header.stamp.toNSec() >= stamp && !latest_reference_.points.empty()) {
    const bool reference_was_unavailable = !has_reference_;
    has_reference_ = true;
    if (reference_was_unavailable ||
        (normal_reference_refresh_pending_ && !probe_mode_ && !repair_mode_)) {
      normal_reference_refresh_pending_ = false;
      last_normal_completed_execution_stamp_ns_ = latest_execution_stamp_ns_;
      requestPlanLocked("fresh_reference_status_after_reference");
    }
    return;
  }
  normal_reference_refresh_pending_ = !exhausted;
  has_reference_ = false;
  plan_requested_ = false;
  ++mode_epoch_;  // in-flight candidates based on the old reference are stale
  plan_request_reason_ = exhausted ? "normal_task_reference_retry_exhausted"
                                   : "waiting_fresh_normal_task_reference";
  ROS_WARN_STREAM("[LocalSparseSCPPlanner] " << plan_request_reason_ << " " << msg->data);
}

void LocalSparseSCPPlanner::referenceCallback(
    const trajectory_msgs::JointTrajectoryConstPtr& msg) {
  if (!msg || msg->points.empty()) return;
  std::lock_guard<std::mutex> lock(mutex_);
  if (!task_reference_receipt_.reference(msg->header.stamp.toNSec())) {
    if (msg->header.stamp.toNSec() >= task_reference_receipt_.min_stamp) {
      // Buffer only; an exhausted request cannot authorize planning. A newer
      // request's status may arrive after its reference on the other topic.
      latest_reference_ = *msg;
      latest_reference_received_ = ros::Time::now();
    }
    return;
  }
  latest_reference_ = *msg;
  latest_reference_received_ = ros::Time::now();
  has_reference_ = true;
  task_no_progress_.reset();
  std_msgs::String reset; reset.data = "status=reset reason=fresh_reference";
  task_stall_pub_.publish(reset);

  if (normal_reference_refresh_pending_) {
    // C5.33: the first fresh /task_trajectory after PROBE->NORMAL is the
    // measured-state task rebase. Baseline the just-completed PROBE execution
    // stamp so a delayed duplicate tracker summary cannot be interpreted as a
    // NORMAL completion event.
    normal_reference_refresh_pending_ = false;
    last_normal_completed_execution_stamp_ns_ =
        latest_execution_stamp_ns_;
    ++normal_reference_refresh_count_;
    requestPlanLocked("fresh_normal_task_reference");
  } else {
    requestPlanLocked("new_nominal_reference");
  }
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
  if (changed) selectRepairTargetLocked();
  if (changed && repair_mode_ && visibility_waypoint_weight_ > 0.0)
    requestPlanLocked("visibility_schedule_changed");
}

void LocalSparseSCPPlanner::visibilityFrontierCallback(
    const std_msgs::Float64MultiArrayConstPtr& msg) {
  // Message layout:
  // [active, frontier_weight_scale, qvis_weight_scale, q_frontier(7)]
  if (!msg || msg->data.size() != 10) return;

  const bool active = msg->data[0] > 0.5;
  const double frontier_scale = msg->data[1];
  const double qvis_scale = msg->data[2];
  if (!std::isfinite(frontier_scale) ||
      !std::isfinite(qvis_scale) ||
      frontier_scale < 0.0 ||
      qvis_scale < 0.0) {
    return;
  }

  Eigen::VectorXd q = Eigen::VectorXd::Zero(dof_);
  for (int j = 0; j < dof_; ++j) {
    q[j] = msg->data[3 + static_cast<std::size_t>(j)];
  }
  if (active && !finiteVector(q)) return;
  if (!active) q.setZero();

  std::lock_guard<std::mutex> lock(mutex_);
  const bool changed =
      active != latest_frontier_.active ||
      std::fabs(frontier_scale -
                latest_frontier_.frontier_weight_scale) > 1e-5 ||
      std::fabs(qvis_scale -
                latest_frontier_.qvis_weight_scale) > 1e-5 ||
      latest_frontier_.q.size() != dof_ ||
      (q - latest_frontier_.q).lpNorm<Eigen::Infinity>() > 1e-4;

  latest_frontier_.active = active;
  latest_frontier_.frontier_weight_scale = frontier_scale;
  latest_frontier_.qvis_weight_scale = qvis_scale;
  latest_frontier_.q = q;

  if (changed && repair_mode_ && visibility_waypoint_weight_ > 0.0) {
    requestPlanLocked(
        active
            ? "visibility_frontier_changed"
            : "visibility_frontier_cleared");
  }
}

void LocalSparseSCPPlanner::singleWaypointActiveCallback(
    const std_msgs::BoolConstPtr& msg) {
  if (!msg) return;

  std::lock_guard<std::mutex> lock(mutex_);
  if (latest_single_waypoint_active_ == msg->data) return;

  latest_single_waypoint_active_ = msg->data;
  selectRepairTargetLocked();
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

  const std::string previous_token = latest_observation_token_;
  latest_single_waypoint_q_ = q;
  latest_observation_token_ = "none";
  if (!msg->layout.dim.empty() &&
      msg->layout.dim.front().label.find("care_obs_v1_") == 0 &&
      msg->layout.dim.front().label.find_first_not_of(
          "abcdefghijklmnopqrstuvwxyz0123456789_") == std::string::npos)
    latest_observation_token_ = msg->layout.dim.front().label;
  has_single_waypoint_q_ = true;
  if (changed || previous_token != latest_observation_token_)
    selectRepairTargetLocked();

  if ((changed || previous_token != latest_observation_token_) &&
      repair_mode_ && visibility_waypoint_weight_ > 0.0) {
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
      ++mode_epoch_;
      if (repair_mode_) {
        task_no_progress_.reset();
        std_msgs::String reset; reset.data = "status=reset reason=real_repair_transition";
        task_stall_pub_.publish(reset);
        // C5.32: the execution that was already active/completed before REPAIR
        // is not a REPAIR completion event. Baseline the stamp here so a
        // repeated latched complete=1 from the previous mode cannot trigger a
        // spurious REPAIR replan.
        last_repair_completed_execution_stamp_ns_ =
            latest_execution_stamp_ns_;
        // PROBE->REPAIR owns its next plan through the REPAIR trigger, so any
        // pending wait for a fresh NORMAL task reference is cancelled.
        normal_reference_refresh_pending_ = false;
      }
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

  const bool was_probe = probe_mode_;
  probe_mode_ = msg->data;
  ++mode_epoch_;

  if (probe_mode_) {
    probe_reference_request_id_ = task_reference_receipt_.id;
    normal_reference_refresh_pending_ = false;
    requestPlanLocked("enter_probe_normal");
    return;
  }

  if (!was_probe) return;

  if (repair_mode_) {
    // PROBE->REPAIR is already replanned by recoveryCallback().
    normal_reference_refresh_pending_ = false;
    return;
  }

  // C5.33 PROBE->NORMAL: do NOT immediately plan against the stale one-shot
  // task trajectory. Wait for the regime-manager pulse -> repeated EE goal ->
  // fresh measured-state /task_trajectory callback.
  if (has_reference_ && task_reference_receipt_.id > probe_reference_request_id_ &&
      !task_reference_receipt_.pending && !task_reference_receipt_.failed) {
    normal_reference_refresh_pending_ = false;
    requestPlanLocked("fresh_reference_before_normal_transition");
    return;
  }
  normal_reference_refresh_pending_ = !task_reference_receipt_.failed;
  plan_requested_ = false;
  plan_request_reason_ = "waiting_fresh_normal_task_reference";
}

void LocalSparseSCPPlanner::replanRequestCallback(
    const std_msgs::BoolConstPtr& msg) {
  if (!msg || !msg->data) return;
  std::lock_guard<std::mutex> lock(mutex_);
  if (normal_reference_refresh_pending_ && !repair_mode_ && !probe_mode_) {
    ++normal_refresh_blocked_replan_count_;
    return;
  }
  requestPlanLocked("external_replan_request");
}

void LocalSparseSCPPlanner::smoothReplanRequestCallback(
    const std_msgs::BoolConstPtr& msg) {
  if (!msg || !msg->data) return;
  std::lock_guard<std::mutex> lock(mutex_);

  if (probe_mode_ || normal_reference_refresh_pending_) {
    ++smooth_handoff_replan_suppressed_probe_count_;
    return;
  }
  if (plan_running_ || plan_requested_) {
    ++smooth_handoff_replan_suppressed_busy_count_;
    return;
  }

  ++smooth_handoff_replan_count_;
  requestPlanLocked(
      repair_mode_ ? "smooth_handoff_repair" : "smooth_handoff_normal");
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

  unsigned long long execution_stamp_ns = 0;
  const bool has_execution_stamp =
      parseUnsignedToken(
          msg->data, "execution_stamp_ns", &execution_stamp_ns) &&
      execution_stamp_ns > 0;

  bool request_repair_replan = false;
  bool request_normal_replan = false;
  bool duplicate_repair_completion = false;
  bool duplicate_normal_completion = false;
  unsigned long long repair_completion_replan_count_snapshot = 0;
  unsigned long long normal_completion_replan_count_snapshot = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (has_execution_stamp) {
      latest_execution_stamp_ns_ = execution_stamp_ns;
    }

    const bool rising = complete && !latest_execution_complete_;
    latest_execution_complete_ = complete;

    if (complete && repair_mode_) {
      if (has_execution_stamp) {
        // C5.32: completion identity is the immutable committed execution
        // stamp, not a lossy Boolean edge. Faster C5.31 planning can replace
        // trajectories quickly enough that an intermediate complete=0 tracker
        // sample is dropped.
        if (execution_stamp_ns !=
            last_repair_completed_execution_stamp_ns_) {
          last_repair_completed_execution_stamp_ns_ =
              execution_stamp_ns;
          ++repair_completion_replan_count_;
          repair_completion_replan_count_snapshot =
              repair_completion_replan_count_;
          request_repair_replan = true;
        } else {
          ++repair_completion_duplicate_count_;
          duplicate_repair_completion = true;
        }
      } else if (rising) {
        ++repair_completion_replan_count_;
        repair_completion_replan_count_snapshot =
            repair_completion_replan_count_;
        request_repair_replan = true;
      }
    } else if (complete && !probe_mode_ &&
               !normal_reference_refresh_pending_) {
      // C5.33: NORMAL is also a receding-horizon execution mode. Once one
      // certified local trajectory finishes, request the next plan from the
      // latest measured state/reference. Use execution_stamp_ns exactly once,
      // mirroring C5.32, so dropped complete=0 samples cannot deadlock NORMAL.
      if (has_execution_stamp) {
        if (execution_stamp_ns !=
            last_normal_completed_execution_stamp_ns_) {
          last_normal_completed_execution_stamp_ns_ =
              execution_stamp_ns;
          ++normal_completion_replan_count_;
          normal_completion_replan_count_snapshot =
              normal_completion_replan_count_;
          request_normal_replan = true;
        } else {
          ++normal_completion_duplicate_count_;
          duplicate_normal_completion = true;
        }
      } else if (rising) {
        ++normal_completion_replan_count_;
        normal_completion_replan_count_snapshot =
            normal_completion_replan_count_;
        request_normal_replan = true;
      }
    }

    if (request_repair_replan) {
      requestPlanLocked("repair_trajectory_complete");
    } else if (request_normal_replan) {
      requestPlanLocked("normal_trajectory_complete");
    }
  }

  if (request_repair_replan) {
    ROS_INFO_STREAM(
        "[LocalSparseSCPPlanner] REPAIR execution completion -> replan"
        << " stamp_ns="
        << (has_execution_stamp ? execution_stamp_ns : 0ULL)
        << " count=" << repair_completion_replan_count_snapshot);
  } else if (request_normal_replan) {
    ROS_INFO_STREAM(
        "[LocalSparseSCPPlanner] NORMAL execution completion -> replan"
        << " stamp_ns="
        << (has_execution_stamp ? execution_stamp_ns : 0ULL)
        << " count=" << normal_completion_replan_count_snapshot);
  } else if (duplicate_repair_completion) {
    ROS_DEBUG_STREAM_THROTTLE(
        1.0,
        "[LocalSparseSCPPlanner] duplicate REPAIR completion ignored stamp_ns="
            << execution_stamp_ns);
  } else if (duplicate_normal_completion) {
    ROS_DEBUG_STREAM_THROTTLE(
        1.0,
        "[LocalSparseSCPPlanner] duplicate NORMAL completion ignored stamp_ns="
            << execution_stamp_ns);
  }

  // PROBE_NORMAL replanning remains owned by the regime manager and is
  // released only after tracker complete=1 for the exact committed token.
}

void LocalSparseSCPPlanner::cdfConstraintBatchCallback(
    const care_collision_cdf::CollisionCDFConstraintBatchConstPtr& msg) {
  acceptCdfBatch(msg, nullptr);
}

void LocalSparseSCPPlanner::witnessResponseCallback(
    const care_collision_cdf::CollisionCDFWitnessResponseConstPtr& msg) {
  if (!msg) return;
  acceptCdfBatch(boost::make_shared<care_collision_cdf::CollisionCDFConstraintBatch>(msg->batch), msg.get());
}

void LocalSparseSCPPlanner::acceptCdfBatch(
    const care_collision_cdf::CollisionCDFConstraintBatchConstPtr& msg,
    const care_collision_cdf::CollisionCDFWitnessResponse* response) {
  if (!msg) return;

  bool accepted = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // A legacy batch cannot stand in for an explicit witness acknowledgement.
    // Empty requests keep the legacy batch path (NORMAL/PROBE included).
    if (witness_response_required_ != static_cast<bool>(response)) return;
    ++cdf_batch_received_;

    if (!plan_running_ || !waiting_for_cdf_) return;

    // Stamp is a request identity, not a noisy measurement timestamp. In
    // simulation adjacent requests may differ by only 1 ns; a tolerance
    // would allow a delayed previous (possibly empty) batch into a new solve.
    if (msg->header.stamp != current_query_stamp_) {
      ++cdf_stamp_miss_;
      return;
    }

    const double roundtrip_ms =
        current_query_wall_.toSec() > 0.0
            ? (ros::WallTime::now() - current_query_wall_).toSec() * 1000.0
            : 0.0;
    last_cdf_roundtrip_ms_ = roundtrip_ms;
    plan_cdf_roundtrip_sum_ms_ += roundtrip_ms;
    plan_cdf_roundtrip_max_ms_ =
        std::max(plan_cdf_roundtrip_max_ms_, roundtrip_ms);
    ++plan_cdf_roundtrip_count_;

    bool response_error = false;
    if (response) {
      const auto& a = response->request;
      const auto& b = current_witness_request_;
      if (a.header.stamp != b.header.stamp || a.header.frame_id != b.header.frame_id ||
          a.plan_sequence != b.plan_sequence || a.mode_epoch != b.mode_epoch ||
          a.target_revision != b.target_revision || a.progress_epoch != b.progress_epoch ||
          a.dof != b.dof || a.original_timestep != b.original_timestep ||
          a.point_flat != b.point_flat || a.q_flat != b.q_flat) {
        publishWitnessDiagnosticLocked(-1, -1, Eigen::Vector3d::Zero(), "response_identity_mismatch");
        return;
      }
      response_error = response->status.size() != b.original_timestep.size();
      if (response_error)
        publishWitnessDiagnosticLocked(-1, -1, Eigen::Vector3d::Zero(), "response_dimension_error");
      // Only a fresh map disposition in this exact query may retire a point.
      // A missing pair in a plain batch is never a free-space certificate.
      if (!response_error && plan_mode_epoch_ == mode_epoch_ &&
          repair_no_progress_.current(plan_repair_ticket_)) {
        for (std::size_t i = 0; i < response->status.size(); ++i) {
          const Eigen::Vector3d p(b.point_flat[3*i], b.point_flat[3*i+1], b.point_flat[3*i+2]);
          const auto& status = response->status[i];
          publishWitnessDiagnosticLocked(static_cast<int>(i), b.original_timestep[i], p, "upstream_"+status);
          if (status == "resolved_free") {
            repair_unknown_witnesses_.erase(std::remove_if(repair_unknown_witnesses_.begin(),
                repair_unknown_witnesses_.end(), [&](const RepairWitness& w) {
                  return w.timestep == b.original_timestep[i] &&
                      (w.point-p).lpNorm<Eigen::Infinity>() <= 1e-5;
                }), repair_unknown_witnesses_.end());
          } else if (status != "evaluated_unknown" && status != "evaluated_occupied") {
            response_error = true;
          }
        }
      }
    }
    pending_batch_ = msg;
    if (plan_repair_mode_ && repair_mode_ && plan_mode_epoch_ == mode_epoch_ &&
        repair_no_progress_.current(plan_repair_ticket_))
      updateRepairWitnessesLocked(*msg);
    if (response_error) repair_witness_requires_reobserve_ = true;
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

void LocalSparseSCPPlanner::publishWitnessDiagnosticLocked(
    int index, int timestep, const Eigen::Vector3d& point,
    const std::string& reason, int pair, double q_error) {
  std::ostringstream s;
  s << std::setprecision(17) << "C5_4_WITNESS event=repair_witness_check"
    << " query_stamp_ns=" << current_query_stamp_.toNSec()
    << " plan_seq=" << plan_sequence_ << " mode_epoch=" << plan_mode_epoch_
    << " target_revision=" << plan_repair_ticket_.revision
    << " progress_epoch=" << plan_repair_ticket_.progress
    << " phase=" << (plan_repair_observation_phase_ ? "observation" : "retreat")
    << " witness_index=" << index << " timestep=" << timestep
    << " point=[" << point[0] << ',' << point[1] << ',' << point[2] << ']'
    << " reason=" << reason << " batch_pair=" << pair << " q_error_inf=" << q_error;
  if (timestep >= 1 && timestep < plan_q_bar_.cols()) {
    s << " q_expected=[";
    for (int j = 0; j < dof_; ++j) { if (j) s << ','; s << plan_q_bar_(j,timestep); }
    s << ']';
  }
  std_msgs::String msg; msg.data = s.str(); witness_diagnostic_pub_.publish(msg);
  // Unthrottled evidence with exact identities also survives CSV packet loss.
  ROS_INFO_STREAM(msg.data);
}

void LocalSparseSCPPlanner::updateRepairWitnessesLocked(
    const care_collision_cdf::CollisionCDFConstraintBatch& batch) {
  // Identity only is retained. Every accepted query must supply its own finite
  // distance and full 7D gradient at the current (point, timestep, q_bar).
  // This runs in BOTH phases, and freshness is recomputed, never latched.
  repair_witness_requires_reobserve_ = true;
  const std::size_t n = batch.distance.size();
  if (batch.num_pairs < 0 || n != static_cast<std::size_t>(batch.num_pairs) ||
      batch.dof != dof_ || batch.original_timestep.size() != n ||
      (!batch.source_type.empty() && batch.source_type.size() != n) ||
      batch.point_flat.size() != n * 3 ||
      batch.gradient_flat.size() != n * dof_ ||
      batch.q_linearization_flat.size() != n * dof_) {
    publishWitnessDiagnosticLocked(-1, -1, Eigen::Vector3d::Zero(), "batch_dimension_error");
    for (std::size_t i = 0; i < repair_unknown_witnesses_.size(); ++i)
      publishWitnessDiagnosticLocked(i, repair_unknown_witnesses_[i].timestep,
          repair_unknown_witnesses_[i].point, "batch_dimension_error");
    return;
  }

  std::vector<RepairWitness> fresh;
  std::vector<std::string> reasons(n, "outside_safety_horizon");
  std::vector<double> q_errors(n, 0.0);
  bool valid_batch = true;
  auto same = [](const RepairWitness& a, const RepairWitness& b) {
    return a.timestep == b.timestep &&
        (a.point - b.point).lpNorm<Eigen::Infinity>() <= 1e-5;
  };
  for (std::size_t i = 0; i < n; ++i) {
    const int k = batch.original_timestep[i];
    if (k < 1 || k > std::min(num_intervals_, cdf_constraint_horizon_steps_)) continue;
    RepairWitness w{Eigen::Vector3d(batch.point_flat[3*i],
        batch.point_flat[3*i+1], batch.point_flat[3*i+2]), k};
    std::string reason = "fresh";
    if (!w.point.allFinite()) reason = "nonfinite_point";
    else if (!std::isfinite(batch.distance[i])) reason = "nonfinite_distance";
    double gradient_l1 = 0.0;
    for (int j = 0; j < dof_; ++j) {
      const double q = batch.q_linearization_flat[i*dof_+j];
      const double g = batch.gradient_flat[i*dof_+j];
      if (!std::isfinite(q)) reason = "nonfinite_q";
      else if (!std::isfinite(g)) reason = "nonfinite_gradient";
      else {
        q_errors[i] = std::max(q_errors[i], std::fabs(q-plan_q_bar_(j,k)));
        if (q_errors[i] > cdf_linearization_tolerance_inf_ && reason == "fresh")
          reason = "q_linearization_mismatch";
      }
      gradient_l1 += std::fabs(g);
    }
    const uint8_t source = batch.source_type.empty()
        ? static_cast<uint8_t>(care_collision_cdf::CollisionCDFConstraintBatch::SOURCE_UNKNOWN) : batch.source_type[i];
    if (source != care_collision_cdf::CollisionCDFConstraintBatch::SOURCE_UNKNOWN &&
        source != care_collision_cdf::CollisionCDFConstraintBatch::SOURCE_OCCUPIED)
      reason = "invalid_source";
    reasons[i] = reason;
    if (reason != "fresh") {
      valid_batch = false;
      publishWitnessDiagnosticLocked(-1, k, w.point, reason, i, q_errors[i]);
      continue;
    }
    fresh.push_back(w);
    if (source == care_collision_cdf::CollisionCDFConstraintBatch::SOURCE_UNKNOWN &&
        batch.distance[i] - trust_radius_ * gradient_l1 < cdf_safety_margin_ &&
        std::none_of(repair_unknown_witnesses_.begin(), repair_unknown_witnesses_.end(),
                     [&](const RepairWitness& old) { return same(old, w); })) {
      if (repair_unknown_witnesses_.size() < 32) repair_unknown_witnesses_.push_back(w);
      else {
        repair_witness_overflow_ = true;
        publishWitnessDiagnosticLocked(-1, k, w.point, "witness_capacity_exceeded", i);
      }
    }
  }
  const bool all_seen = std::all_of(repair_unknown_witnesses_.begin(), repair_unknown_witnesses_.end(),
      [&](const RepairWitness& old) {
        return std::any_of(fresh.begin(), fresh.end(),
                          [&](const RepairWitness& w) { return same(old, w); });
      });
  repair_witness_requires_reobserve_ = repair_witness_overflow_ || !valid_batch || !all_seen;
  for (std::size_t wi = 0; wi < repair_unknown_witnesses_.size(); ++wi) {
    const auto& w = repair_unknown_witnesses_[wi];
    int matched = -1;
    std::string reason = "missing_point";
    for (std::size_t i = 0; i < n; ++i) {
      const Eigen::Vector3d p(batch.point_flat[3*i], batch.point_flat[3*i+1], batch.point_flat[3*i+2]);
      if (!p.allFinite() || (p-w.point).lpNorm<Eigen::Infinity>() > 1e-5) continue;
      if (batch.original_timestep[i] != w.timestep) {
        if (matched < 0) reason = "wrong_timestep";
        continue;
      }
      matched = static_cast<int>(i); reason = reasons[i];
      if (reason == "fresh") break;
    }
    publishWitnessDiagnosticLocked(wi, w.timestep, w.point, reason, matched,
        matched >= 0 ? q_errors[matched] : 0.0);
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
    bool probe_mode,
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

  // C5.24: a PROBE is a local task-continuation test, not a full NORMAL
  // horizon solve whose suffix happens to be discarded later. Advance the
  // nominal task reference only through probe_task_horizon_steps_, then hold
  // that local target for the rest of the optimization horizon. This keeps the
  // fixed QP dimension while making terminal/state tracking consistent with
  // the short executable PROBE semantics.
  if (probe_mode) {
    const int hold_k =
        std::max(1, std::min(probe_task_horizon_steps_, num_intervals_));
    const Eigen::VectorXd q_hold = q_ref.col(hold_k);
    for (int k = hold_k + 1; k <= num_intervals_; ++k) {
      q_ref.col(k) = q_hold;
    }
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

std::string LocalSparseSCPPlanner::repairTargetKeyLocked() const {
  // Deadlines may be refreshed without changing the actual observation goal.
  std::ostringstream s;
  s << std::setprecision(17);
  if (!latest_schedule_.empty()) {
    s << "schedule";
    for (const auto& wp : latest_schedule_) {
      s << ':' << wp.id;
      for (int j = 0; j < wp.q.size(); ++j) s << ':' << wp.q[j];
    }
  } else if (latest_single_waypoint_active_ && has_single_waypoint_q_) {
    s << latest_observation_token_;
    for (int j = 0; j < latest_single_waypoint_q_.size(); ++j)
      s << ':' << latest_single_waypoint_q_[j];
  }
  return s.str();
}

void LocalSparseSCPPlanner::selectRepairTargetLocked() {
  const auto key = repairTargetKeyLocked();
  repair_no_progress_.select(key);
  final_vbc_no_progress_.progress.select(key);
}

void LocalSparseSCPPlanner::finalVerificationCallback(const std_msgs::StringConstPtr& msg) {
  if (!msg) return;
  // Exact field tokens, not substring/nearest-time matching. No malformed or
  // positive outcome can become a final VBC failure.
  std::map<std::string, std::string> fields;
  std::istringstream input(msg->data);
  std::string word;
  while (input >> word) {
    const auto eq = word.find('=');
    if (eq != std::string::npos && !fields.emplace(word.substr(0,eq),word.substr(eq+1)).second) return;
  }
  if (fields["result"] != "unsafe" || fields["safety_gate"] != "vbc" || fields["committed"] != "0") return;
  unsigned long long raw = 0, audited = 0;
  try {
    for (const auto& key : {"raw_candidate_stamp_ns", "audited_trajectory_stamp_ns"}) {
      const auto& value = fields[key];
      if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) return;
    }
    raw = std::stoull(fields["raw_candidate_stamp_ns"]);
    audited = std::stoull(fields["audited_trajectory_stamp_ns"]);
  } catch (...) { return; }
  if (!raw || !audited) return;
  std::string event;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!repair_mode_ || probe_mode_) return;
    const auto outcome = final_vbc_no_progress_.reject(raw, mode_epoch_);
    if (outcome == RepairNoProgress::Failure::Stale) return;
    event = outcome == RepairNoProgress::Failure::Exhausted
        ? "repair_final_vbc_no_progress_hold" : "repair_final_vbc_retry";
    if (outcome == RepairNoProgress::Failure::Exhausted) {
      // Invalidate an already-started solve as well as future Bool replan
      // pulses. Never cancel/commandeer a trajectory owned by the tracker.
      ++plan_sequence_;
      plan_running_ = false;
      waiting_for_cdf_ = false;
      pending_batch_.reset();
    }
    requestPlanLocked(event);
    ROS_WARN_STREAM("[LocalSparseSCPPlanner] " << event << " raw_candidate_stamp_ns=" << raw
        << " audited_trajectory_stamp_ns=" << audited
        << " failures=" << final_vbc_no_progress_.progress.failures());
  }
  publishSummary(event);
}

void LocalSparseSCPPlanner::gcdfRejectionFeedbackCallback(
    const care_collision_cdf::CollisionCDFRejectionFeedbackConstPtr& msg) {
  if (!msg || msg->header.stamp.isZero() || msg->raw_candidate_stamp.isZero() ||
      msg->point_flat.size() % 3 || msg->point_flat.size() > 96 ||
      (!msg->overflow && msg->point_flat.empty()) ||
      !std::all_of(msg->point_flat.begin(), msg->point_flat.end(),
                   [](double v) { return std::isfinite(v); })) return;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = gcdf_feedback_pending_.find(msg->raw_candidate_stamp.toNSec());
    if (it == gcdf_feedback_pending_.end() || !repair_mode_ || probe_mode_ ||
        msg->header.frame_id != current_frame_id_ || it->second.mode_epoch != mode_epoch_ ||
        !repair_no_progress_.current(it->second.ticket)) return;
    gcdf_feedback_pending_.erase(it);  // one feedback per exact raw candidate
    repair_witness_overflow_ = repair_witness_overflow_ || msg->overflow;
    // Carry only point identity. A handoff/braking knot has no general
    // one-to-one QP index: explicitly re-query every EXISTING hard-prefix
    // knot at its new q. Never reuse the executable's distance or gradient.
    for (std::size_t i = 0; i < msg->point_flat.size(); i += 3) {
      const Eigen::Vector3d p(msg->point_flat[i], msg->point_flat[i+1], msg->point_flat[i+2]);
      for (int k = 1; k <= std::min(num_intervals_, cdf_constraint_horizon_steps_); ++k) {
        const bool present = std::any_of(repair_unknown_witnesses_.begin(), repair_unknown_witnesses_.end(),
            [&](const RepairWitness& w) { return w.timestep == k &&
                (w.point-p).lpNorm<Eigen::Infinity>() <= 1e-5; });
        if (present) continue;
        if (repair_unknown_witnesses_.size() >= 32) {
          repair_witness_overflow_ = true;
          break;
        }
        repair_unknown_witnesses_.push_back(RepairWitness{p, k});
      }
    }
    repair_witness_requires_reobserve_ = true;
    // Invalidate any solve started without the newly discovered identities.
    ++plan_sequence_;
    plan_running_ = false;
    waiting_for_cdf_ = false;
    pending_batch_.reset();
    requestPlanLocked("repair_final_gcdf_witness_requery");
    ROS_WARN_STREAM("[LocalSparseSCPPlanner] repair_final_gcdf_witness_requery raw_stamp_ns="
        << msg->raw_candidate_stamp.toNSec() << " audited_stamp_ns=" << msg->header.stamp.toNSec()
        << " witnesses=" << repair_unknown_witnesses_.size()
        << " overflow=" << repair_witness_overflow_);
  }
  publishSummary("repair_final_gcdf_witness_requery");
}

void LocalSparseSCPPlanner::requestPlanLocked(
    const std::string& reason) {
  if (repair_mode_ && final_vbc_no_progress_.progress.blocked()) {
    plan_requested_ = false;
    plan_request_reason_ = "repair_final_vbc_no_progress_hold";
    return;
  }
  if (repair_mode_ && repair_no_progress_.blocked()) {
    plan_requested_ = false;
    plan_request_reason_ = "repair_qp_no_progress_hold";
    return;
  }
  if (!repair_mode_ && task_no_progress_.blocked()) {
    plan_requested_ = false;
    plan_request_reason_ = "qp_no_progress_hold";
    return;
  }
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
  FrontierObjective frontier;
  bool repair = false;
  bool observation_phase = true;
  bool probe = false;
  std::string reason;
  unsigned long long start_mode_epoch = 0;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (plan_running_ || !plan_requested_) return false;
    if (repair_mode_ && (repair_no_progress_.blocked() || final_vbc_no_progress_.progress.blocked())) {
      plan_requested_ = false;
      return false;
    }
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
    plan_observation_token_ = repair_mode_ && schedule.empty() &&
        single_waypoint_active && has_single_waypoint_q
        ? latest_observation_token_ : "none";
    previous_command = latest_executed_command_;
    if (previous_command.size() != dof_)
      previous_command = Eigen::VectorXd::Zero(dof_);
    frontier = latest_frontier_;
    repair = repair_mode_;
    probe = probe_mode_;
    start_mode_epoch = mode_epoch_;
    plan_repair_ticket_ = repair_no_progress_.active;
    if (repair && (repair_witness_mode_epoch_ != start_mode_epoch ||
                   repair_witness_ticket_.target != plan_repair_ticket_.target ||
                   repair_witness_ticket_.revision != plan_repair_ticket_.revision)) {
      repair_unknown_witnesses_.clear();
      repair_witness_overflow_ = false;
      repair_witness_requires_reobserve_ = false;
      repair_observation_phase_ = true;
      repair_witness_ticket_ = plan_repair_ticket_;
      repair_witness_mode_epoch_ = start_mode_epoch;
    }
    plan_repair_mode_ = repair;
    plan_repair_observation_phase_ = repair_observation_phase_;
    observation_phase = plan_repair_observation_phase_;
    if (repair && !observation_phase) plan_observation_token_ = "none";
    plan_probe_mode_ = probe;
    plan_mode_epoch_ = start_mode_epoch;
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
  if (!repair || !observation_phase) {
    schedule.clear();
    frontier.active = false;
    frontier.frontier_weight_scale = 0.0;
    frontier.qvis_weight_scale = 1.0;
  } else if (schedule.empty() &&
             single_waypoint_active &&
             has_single_waypoint_q &&
             single_waypoint_q.size() == dof_) {
    DeadlineWaypoint wp;
    wp.id = -1;
    wp.terminal_objective = true;
    wp.q = single_waypoint_q;
    schedule.push_back(std::move(wp));
  }

  // A REPAIR plan without an active visibility obligation has no steering
  // objective. Solving it would produce a no-op hold candidate, which then
  // stays "fresh" on the predicted-trajectory topic and prevents the VBC
  // selector from falling back to the task trajectory to discover the next
  // blocker. Treat this as a normal waiting state instead: request an immediate
  // task-trajectory bootstrap and publish no candidate until q_vis arrives.
  if (repair && observation_phase && schedule.empty()) {
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
          q_current, reference, ros::Time::now(), probe,
          q_ref, u_ref, q_init, u_init)) {
    abortPlan("reference_horizon_failed");
    return false;
  }

  std::string initialization_mode =
      probe ? "probe_short_horizon_hold_tail" : "task_reference";
  if (repair && repair_hold_initialization_enabled_) {
    for (int k = 0; k <= num_intervals_; ++k) {
      q_init.col(k) = q_current;
    }
    u_init.setZero();
    initialization_mode = "repair_hold";
    if (!observation_phase) initialization_mode = "repair_safety_retreat_hold";
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (start_mode_epoch != mode_epoch_ ||
        (repair && !repair_no_progress_.targetCurrent(plan_repair_ticket_))) {
      plan_running_ = false;
      return false;
    }
    plan_q_current_ = q_current;
    plan_previous_command_ = previous_command;
    plan_q_ref_ = q_ref;
    plan_u_ref_ = u_ref;
    plan_q_bar_ = q_init;
    plan_normal_reseed_used_ = false;
    plan_u_bar_ = u_init;
    plan_schedule_ = schedule;
    plan_frontier_ = frontier;
    plan_repair_mode_ = repair;
    plan_probe_mode_ = probe;
    plan_initialization_mode_ = initialization_mode;
    plan_mode_epoch_ = mode_epoch_;
    plan_probe_feasibility_restore_attempts_ = 0;
    plan_probe_restore_pending_hard_recheck_ = false;
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
    last_cdf_roundtrip_ms_ = 0.0;
    plan_cdf_roundtrip_sum_ms_ = 0.0;
    plan_cdf_roundtrip_max_ms_ = 0.0;
    plan_cdf_roundtrip_count_ = 0;
  }

  publishQueryTrajectory(q_init, u_init, current_frame_id_);

  ROS_INFO_STREAM(
      "[LocalSparseSCPPlanner] plan " << plan_sequence_
      << " started reason=" << reason
      << " repair=" << static_cast<int>(repair)
      << " probe=" << static_cast<int>(probe)
      << " init=" << initialization_mode
      << " vis_obligations=" << schedule.size()
      << " frontier_active=" << static_cast<int>(frontier.active)
      << " frontier_weight_scale=" << frontier.frontier_weight_scale
      << " qvis_weight_scale=" << frontier.qvis_weight_scale);
  publishSummary("plan_started");
  return true;
}

void LocalSparseSCPPlanner::abortPlan(const std::string& reason) {
  SparseSolveResult trace;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    trace.trace_plan_sequence = plan_sequence_;
    trace.trace_observation_token = plan_observation_token_;
    trace.trace_repair = plan_repair_mode_;
    trace.trace_repair_observation_phase = plan_repair_observation_phase_;
    trace.trace_repair_witnesses = static_cast<int>(repair_unknown_witnesses_.size());
    trace.trace_repair_witnesses_fresh = !repair_witness_requires_reobserve_;
    trace.trace_probe = plan_probe_mode_;
    trace.trace_repair_ticket = plan_repair_ticket_;
    trace.status = reason;
    if (witness_response_required_ && reason == "cdf_wait_timeout") {
      for (std::size_t i = 0; i < repair_unknown_witnesses_.size(); ++i)
        publishWitnessDiagnosticLocked(i, repair_unknown_witnesses_[i].timestep,
            repair_unknown_witnesses_[i].point, "witness_response_timeout");
    }
    plan_running_ = false;
    waiting_for_cdf_ = false;
    pending_batch_.reset();
    last_plan_finish_time_ = ros::Time::now();
    ++solve_failure_count_;
    if (repair_mode_ && plan_repair_mode_ && plan_mode_epoch_ == mode_epoch_ &&
        repair_no_progress_.targetCurrent(plan_repair_ticket_)) {
      const auto outcome = repair_no_progress_.fail(plan_repair_ticket_);
      trace.repair_failures = repair_no_progress_.failures();
      trace.repair_feedback = outcome == RepairNoProgress::Failure::Exhausted
          ? "repair_plan_exhausted" : "repair_plan_retry_scheduled";
      requestPlanLocked(trace.repair_feedback);
    }
  }
  publishSummary("aborted_" + reason, &trace);
  if (reason == "cdf_wait_timeout" && !trace.trace_repair && !trace.trace_probe) {
    // Transport failure is not evidence of collision or geometric infeasibility.
    // Remain fail-closed; no unbounded automatic retry of the same reference.
    std_msgs::Bool uncertified; uncertified.data = true;
    task_uncertified_pub_.publish(uncertified);
    std_msgs::String status;
    status.data = "status=uncertified reason=cdf_wait_timeout action=hold_wait_external_replan";
    task_stall_pub_.publish(status);
  }
  if (!trace.repair_feedback.empty()) publishSummary(trace.repair_feedback, &trace);
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
    FrontierObjective frontier;
    bool repair = false;
    bool probe = false;
    double trust = 0.0;
    double slack_linear_weight = 0.0;
    int iteration = 0;
    unsigned long long solve_mode_epoch = 0;
    unsigned long long solve_plan_sequence = 0;
    bool observation_phase = true;
    bool witnesses_fresh = true;
    int witness_count = 0;
    std::string solve_observation_token;
    RepairNoProgress::Ticket solve_repair_ticket;
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
      solve_mode_epoch = plan_mode_epoch_;
      solve_plan_sequence = plan_sequence_;
      solve_observation_token = plan_observation_token_;
      solve_repair_ticket = plan_repair_ticket_;

      repair = plan_repair_mode_;
      observation_phase = plan_repair_observation_phase_;
      witnesses_fresh = !repair_witness_requires_reobserve_;
      witness_count = static_cast<int>(repair_unknown_witnesses_.size());
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
      frontier = plan_frontier_;
      trust = trust_radius_;
      slack_linear_weight = plan_cdf_slack_linear_weight_;
      iteration = scp_iteration_;
      expected_stamp = current_query_stamp_;
    }

    if (batch->header.stamp != expected_stamp) {
      continue;
    }

    publishSummary("sparse_qp_started");

    SparseSolveResult result;
    if (repair && !witnesses_fresh) {
      result.status = "repair_witness_missing_or_invalid";
    } else if (repair && !observation_phase && cdf_slack_enabled_) {
      result.status = "repair_retreat_requires_hard_gcdf";
    } else {
      result = solveSparseSubproblem(
            *batch,
            q_bar,
            u_bar,
            q_ref,
            u_ref,
            previous_command,
            schedule,
            frontier,
            repair,
            probe,
            trust,
            slack_linear_weight);
    }
    result.trace_repair_observation_phase = observation_phase;
    result.trace_repair_witnesses = witness_count;
    result.trace_repair_witnesses_fresh = witnesses_fresh;

    // C5.25: compute the existing soft diagnostic before deciding whether a
    // failed PROBE plan is terminal. The soft solve is NON-EXECUTABLE; when
    // enabled it may only provide the next SCP linearization point.
    SparseSolveResult diagnostic;
    bool diagnostic_ran = false;
    if (!result.solved && !repair &&
        task_failure_slack_diagnostic_enabled_) {
      diagnostic =
          solveSparseSubproblem(
              *batch,
              q_bar,
              u_bar,
              q_ref,
              u_ref,
              previous_command,
              schedule,
              frontier,
              repair,
              probe,
              trust,
              task_failure_slack_diagnostic_weight_,
              true);
      diagnostic_ran = true;
    }

    result.trace_plan_sequence = solve_plan_sequence;
    result.trace_observation_token = solve_observation_token;
    result.trace_repair = repair;
    result.trace_probe = probe;
    result.trace_repair_ticket = solve_repair_ticket;
    diagnostic.trace_plan_sequence = solve_plan_sequence;
    diagnostic.trace_observation_token = solve_observation_token;
    diagnostic.trace_repair = repair;
    diagnostic.trace_probe = probe;
    bool publish_candidate = false;
    bool publish_next_query = false;
    bool restoration_applied = false;
    bool normal_reseed_applied = false;
    Eigen::MatrixXd q_next, u_next;
    std::string frame;
    double total_ms = 0.0;

    {
      std::lock_guard<std::mutex> lock(mutex_);
      // Failure evidence/restoration is as request-sensitive as candidates.
      // An old worker result must not clear a newer plan's running state.
      if (solve_plan_sequence != plan_sequence_) continue;
      if (solve_mode_epoch != mode_epoch_ || repair != repair_mode_ ||
          probe != probe_mode_ ||
          (repair && !repair_no_progress_.current(solve_repair_ticket))) {
        plan_running_ = false;
        waiting_for_cdf_ = false;
        if (repair && repair_mode_ && solve_mode_epoch == mode_epoch_ &&
            repair_no_progress_.targetCurrent(solve_repair_ticket))
          requestPlanLocked("repair_stale_measured_progress");
        continue;
      }
      ++solve_count_;
      if (!result.solved) {
        ++solve_failure_count_;

        const bool restoration_available =
            probe &&
            probe_feasibility_restoration_enabled_ &&
            diagnostic_ran &&
            diagnostic.solved &&
            plan_probe_feasibility_restore_attempts_ <
                probe_feasibility_restoration_max_attempts_;

        // One bounded NORMAL relinearization only after an independently
        // proven row/box conflict. Preserve the task objective, measured q_0,
        // trust radius, PIQP settings and total SCP budget. This seed is never
        // published as an executable candidate; fresh GCDF + hard QP required.
        if (!repair && !probe && result.box_conflicting_cdf_rows > 0 &&
            !plan_normal_reseed_used_ && scp_iteration_ + 1 < max_scp_iterations_ &&
            measuredBrakingSeed(q_bar.col(0), previous_command, acceleration_limits_,
                                num_intervals_, dt_, q_next, u_next)) {
          plan_normal_reseed_used_ = true;
          ++scp_iteration_;  // failed solve consumes an iteration too
          plan_q_bar_ = q_next;
          plan_u_bar_ = u_next;
          previous_query_min_distance_ = std::numeric_limits<double>::quiet_NaN();
          plan_initialization_mode_ = "normal_conflict_measured_reseed";
          normal_reseed_applied = true;
          publish_next_query = true;
          frame = current_frame_id_;
        } else if (restoration_available) {
          // The soft trajectory stays internal. Re-query GCDF at this iterate,
          // then require a hard-QP solve before any candidate publication.
          plan_q_bar_ = diagnostic.q;
          plan_u_bar_ = diagnostic.u;
          previous_query_min_distance_ =
              std::numeric_limits<double>::quiet_NaN();
          ++plan_probe_feasibility_restore_attempts_;
          ++probe_feasibility_restore_count_;
          plan_probe_restore_pending_hard_recheck_ = true;
          restoration_applied = true;
          publish_next_query = true;
          q_next = plan_q_bar_;
          u_next = plan_u_bar_;
          frame = current_frame_id_;
        } else {
          plan_running_ = false;
          waiting_for_cdf_ = false;
          plan_probe_restore_pending_hard_recheck_ = false;
          last_plan_finish_time_ = ros::Time::now();
          if (repair) {
            if (observation_phase && (result.selected_unknown_cdf_rows > 0 || !witnesses_fresh)) {
              repair_witness_requires_reobserve_ = true;
              repair_observation_phase_ = false;
            }
            const auto outcome = repair_no_progress_.fail(solve_repair_ticket);
            result.repair_failures = repair_no_progress_.failures();
            result.repair_feedback = outcome == RepairNoProgress::Failure::Exhausted
                ? "repair_qp_exhausted" : outcome == RepairNoProgress::Failure::Retry
                ? "repair_qp_retry_scheduled" : "repair_qp_stale_progress_retry";
            // A fresh GCDF query and hard solve are required for every retry.
            // Measured motion during this solve belongs to a new episode.
            requestPlanLocked(result.repair_feedback);
          }
        }
      } else {
        if (probe && plan_probe_restore_pending_hard_recheck_) {
          ++probe_feasibility_restore_success_count_;
          plan_probe_restore_pending_hard_recheck_ = false;
        }

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
          if (repair && !observation_phase) {
            // The active plan remains retreat through publication. Only the
            // NEXT plan regains visibility objectives, after this complete
            // hard solve with fresh witness coverage. UNKNOWN rows may remain.
            repair_observation_phase_ = true;
          }
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

      if (diagnostic_ran) {
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

      if (normal_reseed_applied) {
        ROS_WARN_STREAM("[LocalSparseSCPPlanner] NORMAL_BOX_RESEED conflicts="
                        << result.box_conflicting_cdf_rows << " used_iteration=" << iteration+1
                        << " fresh_gcdf_hard_qp_required=1");
        publishQueryTrajectory(q_next, u_next, frame);
        continue;
      }

      if (restoration_applied) {
        publishSummary(
            "probe_feasibility_restore_applied",
            diagnostic_ran ? &diagnostic : nullptr,
            0.0);
        ROS_WARN_STREAM(
            "[LocalSparseSCPPlanner] C5.25 PROBE feasibility restoration "
            "applied; soft iterate is internal only, fresh GCDF + hard solve "
            "required before candidate publication");

        publishQueryTrajectory(q_next, u_next, frame);
        continue;
      }

      if (repair) {
        publishSummary(result.repair_feedback, &result, total_ms);
      }
      if (!repair) {
        bool stalled = false;
        int failures = 0;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          if (solve_mode_epoch != mode_epoch_ || solve_plan_sequence != plan_sequence_ || repair != repair_mode_ ||
              probe != probe_mode_) continue;
          Eigen::VectorXd measured;
          if (!extractMeasuredQ(latest_joint_state_, measured)) measured = q_bar.col(0);
          stalled = task_no_progress_.fail(measured);
          failures = task_no_progress_.failures;
          if (stalled) plan_requested_ = false;
        }
        if (stalled) {
          std::lock_guard<std::mutex> lock(mutex_);
          if (solve_mode_epoch != mode_epoch_ || solve_plan_sequence != plan_sequence_ ||
              !task_no_progress_.blocked()) continue;
          std_msgs::String msg;
          std::ostringstream s;
          s << "status=stalled classification=QP_NO_PROGRESS_UNCERTIFIED"
            << " failures=" << failures << " query_stamp_ns=" << batch->header.stamp.toNSec()
            << " snapshot=disabled";
          msg.data = s.str(); task_stall_pub_.publish(msg);
          ROS_ERROR_STREAM("[LocalSparseSCPPlanner] " << msg.data);
          // No infeasibility claim or soft candidate. The regime manager
          // requests same-query exact VBC; row presence is not blocker evidence.
          continue;
        }
        if (result.status.find("primal infeasible") != std::string::npos) {
          // A certificate for this linearized QP does not identify which
          // UNKNOWN/OCCUPIED rows caused it. Never turn row presence into a
          // visibility obligation (the final-GCDF rejection path is unchanged).
          std::lock_guard<std::mutex> lock(mutex_);
          if (solve_mode_epoch != mode_epoch_ || solve_plan_sequence != plan_sequence_) continue;
          std_msgs::String msg;
          msg.data = "status=retry classification=QP_PRIMAL_INFEASIBLE_UNATTRIBUTED";
          task_stall_pub_.publish(msg);
          ROS_WARN_STREAM("[LocalSparseSCPPlanner] " << msg.data
                          << " unknown_rows=" << result.selected_unknown_cdf_rows
                          << " occupied_rows=" << result.selected_occupied_cdf_rows
                          << " -> bounded retry; no recovery evidence published");
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
      bool mode_stale = false;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        mode_stale =
            solve_mode_epoch != mode_epoch_ ||
            solve_plan_sequence != plan_sequence_ ||
            repair != repair_mode_ ||
            probe != probe_mode_ ||
            (repair && !repair_no_progress_.current(solve_repair_ticket));
        if (mode_stale) {
          ++stale_mode_candidate_discard_count_;
        } else {
          // Publish under the same identity lock as the check: a callback
          // cannot replace the target in between validation and publication.
          publishCandidateTrajectory(
              result.q, result.u, current_frame_id_, solve_observation_token, *batch, q_bar, u_bar);
          if (repair && !observation_phase)
            requestPlanLocked("repair_observation_after_safe_retreat");
        }
      }
      if (mode_stale) {
        publishSummary("stale_mode_candidate_discarded", &result, total_ms);
        ROS_INFO_STREAM(
            "[LocalSparseSCPPlanner] stale-mode candidate discarded before "
            "commit pipeline");
        continue;
      }

      publishSummary("candidate_published", &result, total_ms);
      continue;
    }

    if (publish_next_query) {
      publishQueryTrajectory(q_next, u_next, frame);
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
    const FrontierObjective& frontier,
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
      (!batch.source_type.empty() &&
       batch.source_type.size() != static_cast<std::size_t>(n_pairs)) ||
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

    const uint8_t source_type =
        batch.source_type.empty()
            ? care_collision_cdf::CollisionCDFConstraintBatch::SOURCE_UNKNOWN
            : batch.source_type[static_cast<std::size_t>(i)];
    if (source_type ==
        care_collision_cdf::CollisionCDFConstraintBatch::SOURCE_OCCUPIED) {
      ++out.selected_occupied_cdf_rows;
    } else {
      ++out.selected_unknown_cdf_rows;
      out.selected_unknown_pair_indices.push_back(i);
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

  // Visibility obligations are long-range trajectory objectives. A learned
  // frontier target may simultaneously lower their weight and add a short
  // local visibility-improvement objective; neither changes hard safety.
  const double qvis_weight =
      visibility_waypoint_weight_ *
      ((repair_mode && frontier.active)
           ? frontier.qvis_weight_scale
           : 1.0);
  const double now_s = ros::Time::now().toSec();
  for (const auto& wp : schedule) {
    if (wp.q.size() != dof_) continue;
    const int k = visibilityObjectiveStep(wp.terminal_objective,
        wp.deadline_abs_s, now_s, dt_, num_intervals_);
    if (k < 1) continue;
    out.visibility_objective_step = k;
    for (int j = 0; j < dof_; ++j) {
      addQuadraticTarget(
          qIndex(k, j),
          qvis_weight,
          wp.q[j]);
    }
  }

  if (repair_mode &&
      frontier.active &&
      frontier.q.size() == dof_ &&
      finiteVector(frontier.q) &&
      frontier.frontier_weight_scale > 0.0) {
    const int k_frontier = std::max(
        1, std::min(visibility_frontier_horizon_step_, num_intervals_));
    const double frontier_weight =
        visibility_waypoint_weight_ * frontier.frontier_weight_scale;
    for (int j = 0; j < dof_; ++j) {
      addQuadraticTarget(
          qIndex(k_frontier, j),
          frontier_weight,
          frontier.q[j]);
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

  // Certified-handoff boundary continuity. Keep this boundary-only term
  // active even when general intra-horizon smoothness is disabled.
  const double boundary_velocity_weight =
      u_smooth_weight_ + handoff_velocity_weight_;
  for (int j = 0; j < dof_; ++j) {
    const int u0 = uIndex(0, j);
    p_triplets.emplace_back(
        u0, u0, 2.0 * boundary_velocity_weight);
    const double prev =
        previous_command.size() == dof_
            ? previous_command[j]
            : 0.0;
    c[u0] += -2.0 * boundary_velocity_weight * prev;
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
      << " unknown_rows=" << out.selected_unknown_cdf_rows
      << " occupied_rows=" << out.selected_occupied_cdf_rows
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

  if (!force_diagnostic_slack && status != piqp::PIQP_SOLVED) {
    out.box_conflicting_cdf_rows = lowerBoxConflicts(G, h_l, x_l, x_u, n_acc);
  }

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
  ros::Time stamp;
  bool armed = false;
  bool invalid_clock = false;
  care_collision_cdf::CollisionCDFWitnessRequest request;
  {
    // C5.31: arm the expected stamp BEFORE publishing. In the event-driven
    // GCDF path a batch can return within a few milliseconds; publishing first
    // creates a race where the callback sees waiting_for_cdf_=false.
    std::lock_guard<std::mutex> lock(mutex_);
    if (plan_running_) {
      const ros::Time now = ros::Time::now();
      // Keep the existing stamp-correlated wire API while avoiding collisions
      // when several SCP iterations run during one /clock update. Do not reset
      // identity between plans. A backward clock must not mint future evidence.
      invalid_clock = now.isZero() || now < last_query_ros_time_;
      if (!invalid_clock) {
        stamp = now;
        if (stamp <= current_query_stamp_) {
          stamp = current_query_stamp_ + ros::Duration(0, 1);
          ++query_stamp_adjustments_;
        }
        last_query_ros_time_ = now;
        current_query_stamp_ = stamp;
        current_query_wall_ = ros::WallTime::now();
        waiting_for_cdf_ = true;
        request.header.stamp = stamp;
        request.header.frame_id = frame_id;
        request.dof = dof_;
        request.plan_sequence = plan_sequence_;
        request.mode_epoch = plan_mode_epoch_;
        request.target_revision = plan_repair_ticket_.revision;
        request.progress_epoch = plan_repair_ticket_.progress;
        if (plan_repair_mode_) {
          for (const auto& witness : repair_unknown_witnesses_) {
            request.original_timestep.push_back(witness.timestep);
            for (int j = 0; j < 3; ++j) request.point_flat.push_back(witness.point[j]);
            for (int j = 0; j < dof_; ++j) request.q_flat.push_back(q(j, witness.timestep));
          }
        }
        current_witness_request_ = request;
        witness_response_required_ = !request.original_timestep.empty();
        if (witness_response_required_) repair_witness_requires_reobserve_ = true;
        armed = true;
      }
    }
  }
  if (armed) {
    publishSummary("cdf_query_armed");
    witness_request_pub_.publish(request);
    query_trajectory_pub_.publish(
        makeTrajectoryMessage(q, u, frame_id, stamp));
  }
  if (invalid_clock) abortPlan("cdf_query_clock_invalid");
  return stamp;
}

void LocalSparseSCPPlanner::publishCandidateTrajectory(
    const Eigen::MatrixXd& q,
    const Eigen::MatrixXd& u,
    const std::string& frame_id,
    const std::string& observation_token,
    const care_collision_cdf::CollisionCDFConstraintBatch& batch,
    const Eigen::MatrixXd& query_q, const Eigen::MatrixXd& query_u) {
  const auto msg = makeTrajectoryMessage(q, u, frame_id, ros::Time::now());
  if (plan_repair_mode_ && !plan_probe_mode_ &&
      !final_vbc_no_progress_.remember(msg.header.stamp.toNSec(), mode_epoch_, plan_sequence_)) {
    ROS_ERROR("[LocalSparseSCPPlanner] duplicate/zero candidate identity; not published");
    return;
  }
  if (plan_repair_mode_ && !plan_probe_mode_) {
    while (gcdf_feedback_pending_.size() >= FinalVbcNoProgress::max_pending)
      gcdf_feedback_pending_.erase(gcdf_feedback_pending_.begin());
    gcdf_feedback_pending_.emplace(msg.header.stamp.toNSec(),
        FinalVbcNoProgress::Candidate{plan_repair_ticket_, mode_epoch_, plan_sequence_});
  }
  if (rejection_snapshots_enabled_) {
    std::ostringstream out;
    out << "{\"schema\":1,\"raw_candidate_stamp_ns\":\"" << msg.header.stamp.toNSec()
        << "\",\"plan_seq\":" << plan_sequence_ << ",\"mode_epoch\":" << mode_epoch_
        << ",\"target_revision\":" << final_vbc_no_progress_.progress.active.revision
        << ",\"progress_epoch\":" << final_vbc_no_progress_.progress.active.progress
        << ",\"observation_token\":";
    auditString(out,observation_token);
    out << ",\"query_trajectory\":";
    auditTrajectory(out,makeTrajectoryMessage(query_q,query_u,frame_id,batch.header.stamp));
    out << ",\"local_gcdf_batch\":"; auditBatch(out,batch); out << '}';
    std_msgs::String context; context.data = out.str();
    if (context.data.size() <= 2*1024*1024) candidate_audit_context_pub_.publish(context);
    else ROS_WARN("[LocalSparseSCPPlanner] audit context exceeds 2 MiB cap; diagnostic omitted");
  }
  std_msgs::String identity;
  identity.data = "observation_token=" + observation_token +
      " raw_candidate_stamp_ns=" + std::to_string(msg.header.stamp.toNSec());
  observation_identity_pub_.publish(identity);
  ROS_INFO_STREAM("[OBS_CANDIDATE] observation_token=" << observation_token
                  << " raw_candidate_stamp_ns=" << msg.header.stamp.toNSec());
  candidate_trajectory_pub_.publish(msg);
}


bool LocalSparseSCPPlanner::publishLocalGcdfRecoveryEvidence(
    const care_collision_cdf::CollisionCDFConstraintBatch& batch,
    const SparseSolveResult& result,
    const Eigen::MatrixXd& q_bar,
    const Eigen::MatrixXd& u_bar,
    const std::string& frame_id) {
  if (result.selected_unknown_pair_indices.empty()) {
    ++local_gcdf_recovery_drop_count_;
    return false;
  }
  if (batch.header.stamp.isZero()) {
    ++local_gcdf_recovery_drop_count_;
    ROS_WARN(
        "[LocalSparseSCPPlanner] local-GCDF recovery dropped: zero batch stamp");
    return false;
  }

  int earliest = std::numeric_limits<int>::max();
  for (const int pair : result.selected_unknown_pair_indices) {
    if (pair < 0 ||
        pair >= static_cast<int>(batch.original_timestep.size())) {
      continue;
    }
    earliest = std::min(
        earliest,
        batch.original_timestep[static_cast<std::size_t>(pair)]);
  }
  if (earliest < 1 || earliest > num_intervals_) {
    ++local_gcdf_recovery_drop_count_;
    ROS_WARN_STREAM(
        "[LocalSparseSCPPlanner] local-GCDF recovery dropped: invalid earliest="
        << earliest);
    return false;
  }

  std::set<std::tuple<double, double, double>> seen;
  std::vector<std::tuple<double, double, double>> points;
  for (const int pair : result.selected_unknown_pair_indices) {
    if (pair < 0 ||
        pair >= static_cast<int>(batch.original_timestep.size()) ||
        batch.original_timestep[static_cast<std::size_t>(pair)] != earliest) {
      continue;
    }
    const std::size_t j = static_cast<std::size_t>(3 * pair);
    if (j + 2 >= batch.point_flat.size()) {
      continue;
    }
    const double x = batch.point_flat[j];
    const double y = batch.point_flat[j + 1];
    const double z = batch.point_flat[j + 2];
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      continue;
    }
    const auto key = std::make_tuple(x, y, z);
    if (seen.insert(key).second) {
      points.push_back(key);
    }
  }

  if (points.empty()) {
    ++local_gcdf_recovery_drop_count_;
    ROS_WARN(
        "[LocalSparseSCPPlanner] local-GCDF recovery dropped: no earliest UNKNOWN points");
    return false;
  }

  const unsigned long long seq = ++local_gcdf_recovery_event_count_;
  trajectory_msgs::JointTrajectory trajectory =
      makeTrajectoryMessage(
          q_bar, u_bar, frame_id, batch.header.stamp);
  trajectory.header.seq = static_cast<uint32_t>(
      seq & 0xffffffffULL);

  std_msgs::Float64MultiArray event;
  const double sweep_s = static_cast<double>(earliest) * dt_;
  event.data.reserve(6 + 3 * points.size());
  event.data.push_back(static_cast<double>(seq));
  event.data.push_back(static_cast<double>(batch.header.stamp.sec));
  event.data.push_back(static_cast<double>(batch.header.stamp.nsec));
  event.data.push_back(sweep_s);
  event.data.push_back(static_cast<double>(earliest));
  event.data.push_back(static_cast<double>(points.size()));
  for (const auto& p : points) {
    event.data.push_back(std::get<0>(p));
    event.data.push_back(std::get<1>(p));
    event.data.push_back(std::get<2>(p));
  }

  // Publish trajectory first so the blocker-aware acquisition cache can match
  // the subsequent exact-voxel event by immutable header stamp.
  gcdf_recovery_trajectory_pub_.publish(trajectory);
  gcdf_recovery_event_pub_.publish(event);

  std::ostringstream point_oss;
  bool first = true;
  for (const auto& p : points) {
    if (!first) point_oss << ";";
    first = false;
    point_oss << "[" << std::get<0>(p) << ","
              << std::get<1>(p) << ","
              << std::get<2>(p) << "]";
  }
  ROS_WARN_STREAM(
      "[LOCAL_GCDF_RECOVERY] seq=" << seq
      << " stamp=" << batch.header.stamp
      << " earliest_timestep=" << earliest
      << " sweep_s=" << sweep_s
      << " point_count=" << points.size()
      << " points=" << point_oss.str());
  return true;
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
  int probe_restore_attempts = 0;
  bool observation_phase = true;
  int witness_count = 0;
  bool witnesses_fresh = true;
  int final_vbc_failures = 0;
  bool final_vbc_hold = false;
  bool probe_restore_pending_recheck = false;
  unsigned long long probe_restore_total = 0;
  unsigned long long probe_restore_hard_success_total = 0;
  double active_slack_weight = 0.0;
  unsigned long long latest_execution_stamp_ns = 0;
  unsigned long long last_repair_completion_stamp_ns = 0;
  unsigned long long repair_completion_replan_count = 0;
  unsigned long long repair_completion_duplicate_count = 0;
  unsigned long long last_normal_completion_stamp_ns = 0;
  unsigned long long normal_completion_replan_count = 0;
  unsigned long long normal_completion_duplicate_count = 0;
  unsigned long long normal_reference_refresh_count = 0;
  unsigned long long normal_refresh_blocked_replan_count = 0;
  bool normal_reference_refresh_pending = false;
  unsigned long long smooth_handoff_replan_count = 0;
  unsigned long long smooth_handoff_replan_suppressed_probe_count = 0;
  unsigned long long smooth_handoff_replan_suppressed_busy_count = 0;
  unsigned long long stale_mode_candidate_discard_count = 0;
  double last_cdf_roundtrip_ms = 0.0;
  unsigned long long query_stamp_ns = 0, query_ros_time_ns = 0, query_stamp_adjustments = 0;
  double plan_cdf_roundtrip_sum_ms = 0.0;
  double plan_cdf_roundtrip_max_ms = 0.0;
  int plan_cdf_roundtrip_count = 0;
  std::string init_mode = "unknown";
  std::string observation_token;
  bool frontier_active = false;
  double frontier_weight_scale = 0.0;
  double frontier_qvis_weight_scale = 1.0;
  double frontier_target_shift_inf =
      std::numeric_limits<double>::quiet_NaN();

  {
    std::lock_guard<std::mutex> lock(mutex_);
    plan_seq = plan_sequence_;
    final_vbc_failures = final_vbc_no_progress_.progress.failures();
    final_vbc_hold = final_vbc_no_progress_.progress.blocked();
    observation_token = plan_observation_token_;
    batches = cdf_batch_received_;
    query_stamp_ns = current_query_stamp_.toNSec();
    query_ros_time_ns = last_query_ros_time_.toNSec();
    query_stamp_adjustments = query_stamp_adjustments_;
    misses = cdf_stamp_miss_;
    solves = solve_count_;
    failures = solve_failure_count_;
    scp_iter = scp_iteration_;
    trust = trust_radius_;
    running = plan_running_;
    repair = plan_repair_mode_;
    probe = plan_probe_mode_;
    observation_phase = plan_repair_observation_phase_;
    witness_count = static_cast<int>(repair_unknown_witnesses_.size());
    witnesses_fresh = !repair_witness_requires_reobserve_;
    probe_restore_attempts = plan_probe_feasibility_restore_attempts_;
    probe_restore_pending_recheck =
        plan_probe_restore_pending_hard_recheck_;
    probe_restore_total = probe_feasibility_restore_count_;
    probe_restore_hard_success_total =
        probe_feasibility_restore_success_count_;
    active_slack_weight = plan_cdf_slack_linear_weight_;
    latest_execution_stamp_ns = latest_execution_stamp_ns_;
    last_repair_completion_stamp_ns =
        last_repair_completed_execution_stamp_ns_;
    repair_completion_replan_count = repair_completion_replan_count_;
    repair_completion_duplicate_count =
        repair_completion_duplicate_count_;
    last_normal_completion_stamp_ns =
        last_normal_completed_execution_stamp_ns_;
    normal_completion_replan_count =
        normal_completion_replan_count_;
    normal_completion_duplicate_count =
        normal_completion_duplicate_count_;
    normal_reference_refresh_count =
        normal_reference_refresh_count_;
    normal_refresh_blocked_replan_count =
        normal_refresh_blocked_replan_count_;
    normal_reference_refresh_pending =
        normal_reference_refresh_pending_;
    smooth_handoff_replan_count = smooth_handoff_replan_count_;
    smooth_handoff_replan_suppressed_probe_count =
        smooth_handoff_replan_suppressed_probe_count_;
    smooth_handoff_replan_suppressed_busy_count =
        smooth_handoff_replan_suppressed_busy_count_;
    stale_mode_candidate_discard_count =
        stale_mode_candidate_discard_count_;
    last_cdf_roundtrip_ms = last_cdf_roundtrip_ms_;
    plan_cdf_roundtrip_sum_ms = plan_cdf_roundtrip_sum_ms_;
    plan_cdf_roundtrip_max_ms = plan_cdf_roundtrip_max_ms_;
    plan_cdf_roundtrip_count = plan_cdf_roundtrip_count_;
    init_mode = plan_initialization_mode_;
    frontier_active = plan_frontier_.active;
    frontier_weight_scale = plan_frontier_.frontier_weight_scale;
    frontier_qvis_weight_scale = plan_frontier_.qvis_weight_scale;
    if (plan_frontier_.q.size() == dof_ &&
        plan_q_current_.size() == dof_ &&
        finiteVector(plan_frontier_.q) &&
        finiteVector(plan_q_current_)) {
      frontier_target_shift_inf =
          (plan_frontier_.q - plan_q_current_)
              .lpNorm<Eigen::Infinity>();
    }
  }

  std::ostringstream oss;
  if (result && result->trace_plan_sequence) {
    plan_seq = result->trace_plan_sequence;
    observation_token = result->trace_observation_token;
    repair = result->trace_repair;
    probe = result->trace_probe;
    observation_phase = result->trace_repair_observation_phase;
    witness_count = result->trace_repair_witnesses;
    witnesses_fresh = result->trace_repair_witnesses_fresh;
  }
  oss << "C5_4_LOCAL_SCP"
      << " event=" << event
      << " query_stamp_ns=" << query_stamp_ns
      << " query_ros_time_ns=" << query_ros_time_ns
      << " query_stamp_adjustments=" << query_stamp_adjustments
      << " final_vbc_failures=" << final_vbc_failures
      << " final_vbc_hold=" << static_cast<int>(final_vbc_hold)
      << " plan_seq=" << plan_seq
      << " observation_token=" << observation_token
      << " running=" << static_cast<int>(running)
      << " repair=" << static_cast<int>(repair)
      << " repair_phase=" << (observation_phase ? "observation" : "retreat")
      << " repair_witness_count=" << witness_count
      << " repair_witness_fresh=" << static_cast<int>(witnesses_fresh)
      << " probe=" << static_cast<int>(probe)
      << " task_ref_horizon_steps="
      << (probe ? probe_task_horizon_steps_ : num_intervals_)
      << " probe_hold_tail="
      << static_cast<int>(probe && probe_task_horizon_steps_ < num_intervals_)
      << " probe_restore_enabled="
      << static_cast<int>(probe_feasibility_restoration_enabled_)
      << " probe_restore_attempts=" << probe_restore_attempts
      << " probe_restore_pending_recheck="
      << static_cast<int>(probe_restore_pending_recheck)
      << " probe_restore_total=" << probe_restore_total
      << " probe_restore_hard_success_total="
      << probe_restore_hard_success_total
      << " cdf_horizon_steps="
      << ((repair || probe) ? cdf_constraint_horizon_steps_ : num_intervals_)
      << " init=" << init_mode
      << " scp_iter=" << scp_iter
      << " trust_q_inf=" << trust
      << " slack_mu=" << active_slack_weight
      << " latest_execution_stamp_ns=" << latest_execution_stamp_ns
      << " last_repair_completion_stamp_ns="
      << last_repair_completion_stamp_ns
      << " repair_completion_replan_count="
      << repair_completion_replan_count
      << " repair_completion_duplicate_count="
      << repair_completion_duplicate_count
      << " last_normal_completion_stamp_ns="
      << last_normal_completion_stamp_ns
      << " normal_completion_replan_count="
      << normal_completion_replan_count
      << " normal_completion_duplicate_count="
      << normal_completion_duplicate_count
      << " normal_reference_refresh_pending="
      << static_cast<int>(normal_reference_refresh_pending)
      << " normal_reference_refresh_count="
      << normal_reference_refresh_count
      << " normal_refresh_blocked_replan_count="
      << normal_refresh_blocked_replan_count
      << " smooth_handoff_replan_count="
      << smooth_handoff_replan_count
      << " smooth_handoff_replan_suppressed_probe_count="
      << smooth_handoff_replan_suppressed_probe_count
      << " smooth_handoff_replan_suppressed_busy_count="
      << smooth_handoff_replan_suppressed_busy_count
      << " stale_mode_candidate_discard_count="
      << stale_mode_candidate_discard_count
      << " frontier_active=" << static_cast<int>(frontier_active)
      << " frontier_horizon_step=" << visibility_frontier_horizon_step_
      << " frontier_weight_scale=" << frontier_weight_scale
      << " frontier_qvis_weight_scale=" << frontier_qvis_weight_scale
      << " frontier_target_shift_inf=" << frontier_target_shift_inf
      << " handoff_velocity_weight=" << handoff_velocity_weight_
      << " batches=" << batches
      << " cdf_roundtrip_ms=" << last_cdf_roundtrip_ms
      << " cdf_roundtrip_count=" << plan_cdf_roundtrip_count
      << " cdf_roundtrip_mean_ms="
      << (plan_cdf_roundtrip_count > 0
              ? plan_cdf_roundtrip_sum_ms /
                    static_cast<double>(plan_cdf_roundtrip_count)
              : 0.0)
      << " cdf_roundtrip_max_ms=" << plan_cdf_roundtrip_max_ms
      << " stamp_miss=" << misses
      << " solves=" << solves
      << " solve_failures=" << failures;

  if (result) {
    oss << " solved=" << static_cast<int>(result->solved)
        << " repair_failures=" << result->repair_failures
        << " repair_max_failures=" << RepairNoProgress::max_failures
        << " repair_target_revision=" << result->trace_repair_ticket.revision
        << " repair_progress_epoch=" << result->trace_repair_ticket.progress
        << " snapshot=disabled"
        << " visibility_objective_step=" << result->visibility_objective_step
        << " status=" << result->status
        << " piqp_iter=" << result->iterations
        << " solve_ms=" << result->setup_and_solve_ms
        << " batch_pairs=" << result->batch_pairs
        << " cdf_rows=" << result->selected_cdf_rows
        << " unknown_cdf_rows=" << result->selected_unknown_cdf_rows
        << " occupied_cdf_rows=" << result->selected_occupied_cdf_rows
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
