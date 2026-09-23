#include "egocentric_arm_planner/trajectory_execution_manager.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace egocentric_arm_planner {

bool TrajectoryExecutionManager::initialize(const ros::NodeHandle& nh,
                                            const ros::NodeHandle& pnh) {
  nh_ = nh;
  pnh_ = pnh;

  if (!loadConfig()) {
    ROS_ERROR("[TrajectoryExecutionManager] Failed to load config.");
    return false;
  }

  if (joint_names_.empty()) {
    ROS_ERROR("[TrajectoryExecutionManager] joint_names is empty.");
    return false;
  }

  const int dof = static_cast<int>(joint_names_.size());
  pnh_.param("execution/primitive_tracking_guard", primitive_tracking_guard_, false);
  pnh_.param("execution/primitive_tracking_relative_fk",
             primitive_tracking_relative_fk_, false);
  pnh_.param("execution/certified_tracking_margin_m", certified_tracking_margin_m_, 0.020);
  if (primitive_tracking_guard_) {
    std::string path, error;
    pnh_.param<std::string>("execution/primitive_urdf_file", path, "");
    if (!std::isfinite(certified_tracking_margin_m_) || certified_tracking_margin_m_ < 0 ||
        !tracking_geometry_.load(path, {"base_link", "link1"}, &error)) {
      ROS_ERROR_STREAM("Invalid primitive tracking envelope: " << error); return false;
    }
    if (primitive_tracking_relative_fk_) {
      std::string fk_error;
      primitive_tracking_relative_fk_ready_ =
          tracking_geometry_.initializeRelativeFk(path, joint_names_, &fk_error);
      if (!primitive_tracking_relative_fk_ready_) {
        ROS_ERROR_STREAM("Invalid relative FK tracking envelope: " << fk_error);
        return false;
      }
    }
  }
  q_measured_ = Eigen::VectorXd::Zero(dof);
  dq_measured_ = Eigen::VectorXd::Zero(dof);
  last_q_ref_ = Eigen::VectorXd::Zero(dof);
  last_dq_ref_ = Eigen::VectorXd::Zero(dof);

  joint_state_sub_ = nh_.subscribe(
      joint_state_topic_, 1,
      &TrajectoryExecutionManager::jointStateCallback, this);

  trajectory_sub_ = nh_.subscribe(
      input_trajectory_topic_, 1,
      &TrajectoryExecutionManager::trajectoryCallback, this);

  safety_hold_sub_ = nh_.subscribe(
      external_safety_hold_topic_, 1,
      &TrajectoryExecutionManager::safetyHoldCallback, this);

  velocity_command_pub_ = nh_.advertise<std_msgs::Float64MultiArray>(
      output_velocity_command_topic_, 1);

  reference_state_pub_ = nh_.advertise<sensor_msgs::JointState>(
      reference_state_topic_, 1);

  summary_pub_ = nh_.advertise<std_msgs::String>(
      summary_topic_, 10);

  replan_request_pub_ = nh_.advertise<std_msgs::Bool>(
      replan_request_topic_, 10);
  smooth_replan_request_pub_ = nh_.advertise<std_msgs::Bool>(
      smooth_replan_request_topic_, 10);

  execution_timer_ = nh_.createTimer(
      ros::Duration(1.0 / execution_rate_),
      &TrajectoryExecutionManager::executionTimerCallback,
      this);

  ROS_WARN(
      "[TrajectoryExecutionManager] C5.4 FULL-TRAJECTORY TRACKING ENABLED: "
      "execution advances through a committed trajectory independently of planner rate.");
  ROS_INFO_STREAM("[TrajectoryExecutionManager] execution_rate="
                  << execution_rate_
                  << " Hz input=" << input_trajectory_topic_
                  << " command=" << output_velocity_command_topic_
                  << " summary=" << summary_topic_);
  ROS_INFO_STREAM("[TrajectoryExecutionManager] Kp="
                  << position_feedback_gain_
                  << " vmax=" << max_command_velocity_
                  << " tracking_replan_threshold="
                  << replan_tracking_error_inf_);
  ROS_INFO_STREAM(
      "[TrajectoryExecutionManager] smooth_handoff="
      << static_cast<int>(smooth_handoff_enabled_)
      << " lead_time=" << smooth_replan_lead_time_s_
      << " min_phase=" << smooth_replan_min_phase_s_
      << " topic=" << smooth_replan_request_topic_);
  ROS_INFO_STREAM(
      "[TrajectoryExecutionManager] Phase E5 external safety hold topic="
      << external_safety_hold_topic_);
  ROS_INFO_STREAM(
      "[TrajectoryExecutionManager] phase-aligned spatial tracking="
      << static_cast<int>(tracking_phase_alignment_enabled_)
      << " search_window=" << tracking_phase_search_window_s_ << " s");
  ROS_INFO_STREAM(
      "[TrajectoryExecutionManager] relative-FK tracking guard="
      << static_cast<int>(primitive_tracking_relative_fk_)
      << " ready=" << static_cast<int>(primitive_tracking_relative_fk_ready_)
      << " (legacy bound is used as the fast-path precheck)");

  return true;
}

bool TrajectoryExecutionManager::loadConfig() {
  pnh_.param<double>("execution/rate", execution_rate_, execution_rate_);
  pnh_.param<double>("execution/control_dt", control_dt_, control_dt_);

  pnh_.param<double>("execution/max_start_error",
                     max_start_error_, max_start_error_);
  pnh_.param<double>("execution/max_tracking_error",
                     max_tracking_error_, max_tracking_error_);

  pnh_.param<bool>("execution/hold_when_no_trajectory",
                   hold_when_no_trajectory_, hold_when_no_trajectory_);
  pnh_.param<bool>("execution/hold_when_tracking_error_large",
                   hold_when_tracking_error_large_,
                   hold_when_tracking_error_large_);
  pnh_.param<bool>("execution/reject_large_start_error",
                   reject_large_start_error_,
                   reject_large_start_error_);
  pnh_.param<bool>("execution/hold_initial_zero_pose",
                   hold_initial_zero_pose_,
                   hold_initial_zero_pose_);
  pnh_.param<bool>("execution/hold_last_reference_when_no_trajectory",
                   hold_last_reference_when_no_trajectory_,
                   hold_last_reference_when_no_trajectory_);
  pnh_.param<double>("execution/reference_timeout",
                     reference_timeout_, reference_timeout_);
  pnh_.param<bool>("execution/tracking_phase_alignment_enabled",
                   tracking_phase_alignment_enabled_,
                   tracking_phase_alignment_enabled_);
  pnh_.param<double>("execution/tracking_phase_search_window_s",
                     tracking_phase_search_window_s_,
                     tracking_phase_search_window_s_);

  // Backward-compatible fallback.
  pnh_.param<double>("execution/velocity_tracking_kp",
                     position_feedback_gain_,
                     position_feedback_gain_);
  pnh_.param<double>("execution/position_feedback_gain",
                     position_feedback_gain_,
                     position_feedback_gain_);
  pnh_.param<double>("execution/max_command_velocity",
                     max_command_velocity_,
                     max_command_velocity_);

  pnh_.param<double>("execution/replan_tracking_error_inf",
                     replan_tracking_error_inf_,
                     replan_tracking_error_inf_);
  pnh_.param<double>("execution/replan_request_min_interval",
                     replan_request_min_interval_s_,
                     replan_request_min_interval_s_);
  pnh_.param<bool>("execution/smooth_handoff_enabled",
                   smooth_handoff_enabled_, smooth_handoff_enabled_);
  pnh_.param<double>("execution/smooth_replan_lead_time",
                     smooth_replan_lead_time_s_, smooth_replan_lead_time_s_);
  pnh_.param<double>("execution/smooth_replan_min_phase",
                     smooth_replan_min_phase_s_, smooth_replan_min_phase_s_);

  pnh_.param<std::string>("execution/joint_states",
                          joint_state_topic_, joint_state_topic_);
  pnh_.param<std::string>("execution/input_trajectory",
                          input_trajectory_topic_, input_trajectory_topic_);
  pnh_.param<std::string>("execution/output_velocity_command",
                          output_velocity_command_topic_,
                          output_velocity_command_topic_);
  pnh_.param<std::string>("execution/reference_state",
                          reference_state_topic_, reference_state_topic_);
  pnh_.param<std::string>("execution/summary_topic",
                          summary_topic_, summary_topic_);
  pnh_.param<std::string>("execution/replan_request_topic",
                          replan_request_topic_, replan_request_topic_);
  pnh_.param<std::string>("execution/smooth_replan_request_topic",
                          smooth_replan_request_topic_,
                          smooth_replan_request_topic_);
  pnh_.param<std::string>("execution/external_safety_hold_topic",
                          external_safety_hold_topic_,
                          external_safety_hold_topic_);

  if (!pnh_.getParam("joint_names", joint_names_)) {
    ROS_ERROR("[TrajectoryExecutionManager] Missing param: joint_names");
    return false;
  }

  if (execution_rate_ <= 0.0) {
    ROS_ERROR("[TrajectoryExecutionManager] execution/rate must be positive.");
    return false;
  }
  if (control_dt_ <= 0.0) {
    ROS_ERROR("[TrajectoryExecutionManager] execution/control_dt must be positive.");
    return false;
  }
  if (max_command_velocity_ <= 0.0) {
    ROS_ERROR("[TrajectoryExecutionManager] execution/max_command_velocity must be positive.");
    return false;
  }
  if (position_feedback_gain_ < 0.0) {
    ROS_ERROR("[TrajectoryExecutionManager] execution/position_feedback_gain must be non-negative.");
    return false;
  }
  if (reference_timeout_ < 0.0 ||
      replan_request_min_interval_s_ < 0.0 ||
      replan_tracking_error_inf_ < 0.0 ||
      tracking_phase_search_window_s_ < 0.0 ||
      smooth_replan_lead_time_s_ <= 0.0 ||
      smooth_replan_min_phase_s_ < 0.0) {
    ROS_ERROR("[TrajectoryExecutionManager] invalid timeout/replan configuration.");
    return false;
  }

  return true;
}

void TrajectoryExecutionManager::jointStateCallback(
    const sensor_msgs::JointStateConstPtr& msg) {
  if (!msg) return;

  Eigen::VectorXd q;
  Eigen::VectorXd dq;
  if (!extractMeasuredState(*msg, q, dq)) {
    ROS_WARN_THROTTLE(
        1.0,
        "[TrajectoryExecutionManager] Failed to extract measured joint state.");
    return;
  }

  std::lock_guard<std::mutex> lock(data_mutex_);
  latest_joint_state_ = *msg;
  q_measured_ = q;
  dq_measured_ = dq;
  has_joint_state_ = true;
}

void TrajectoryExecutionManager::safetyHoldCallback(
    const std_msgs::BoolConstPtr& msg) {
  if (!msg) return;

  const ros::Time now = ros::Time::now();
  std::lock_guard<std::mutex> lock(data_mutex_);
  const bool requested = msg->data;
  if (requested == external_safety_hold_) return;

  if (requested) {
    external_safety_hold_ = true;
    external_safety_hold_start_time_ = now;
    ++external_safety_hold_event_count_;
    ROS_ERROR_STREAM(
        "[TrajectoryExecutionManager] Phase E5 external GCDF HARD HOLD engaged"
        << " count=" << external_safety_hold_event_count_);
  } else {
    if (external_safety_hold_ &&
        !external_safety_hold_start_time_.isZero() &&
        has_active_trajectory_) {
      // Freeze trajectory phase while externally held so release never jumps
      // forward along an unexecuted reference segment.
      active_trajectory_start_time_ +=
          (now - external_safety_hold_start_time_);
    }
    external_safety_hold_ = false;
    external_safety_hold_start_time_ = ros::Time(0);
    ROS_WARN(
        "[TrajectoryExecutionManager] Phase E5 external GCDF hold released");
  }
}

void TrajectoryExecutionManager::trajectoryCallback(
    const trajectory_msgs::JointTrajectoryConstPtr& msg) {
  if (!msg || msg->points.empty()) {
    ROS_WARN_THROTTLE(
        1.0, "[TrajectoryExecutionManager] Received empty trajectory.");
    return;
  }
  if (!trajectoryHasExpectedJoints(*msg)) {
    ROS_WARN_THROTTLE(
        1.0,
        "[TrajectoryExecutionManager] Received trajectory with unexpected joints.");
    return;
  }

  std::vector<int> mapping;
  if (!buildTrajectoryJointIndexMap(*msg, mapping)) {
    ROS_WARN_THROTTLE(
        1.0,
        "[TrajectoryExecutionManager] Failed to build trajectory joint map.");
    return;
  }

  const double duration = getTrajectoryEndTime(*msg);
  if (!std::isfinite(duration) || duration < 0.0) {
    ROS_WARN_THROTTLE(
        1.0, "[TrajectoryExecutionManager] Invalid trajectory duration.");
    return;
  }

  Eigen::VectorXd q_start, dq_start, ddq_start;
  if (!sampleTrajectory(*msg, mapping, 0.0, q_start, dq_start, ddq_start)) {
    ROS_WARN_THROTTLE(
        1.0,
        "[TrajectoryExecutionManager] Failed to sample trajectory start.");
    return;
  }

  const ros::Time now = ros::Time::now();
  {
    std::lock_guard<std::mutex> lock(data_mutex_);

    const uint64_t execution_stamp_ns = msg->header.stamp.toNSec();
    if (execution_stamp_ns > 0 &&
        aborted_execution_stamps_.count(execution_stamp_ns) != 0) {
      ROS_WARN_STREAM_THROTTLE(
          1.0,
          "[TrajectoryExecutionManager] Rejecting delayed duplicate of "
          "aborted execution token=" << execution_stamp_ns);
      return;
    }

    if (has_joint_state_ && reject_large_start_error_) {
      const double start_error =
          (q_start - q_measured_).lpNorm<Eigen::Infinity>();
      if (start_error > max_start_error_) {
        ROS_WARN_THROTTLE(
            1.0,
            "[TrajectoryExecutionManager] Reject trajectory: start error %.4f > %.4f",
            start_error, max_start_error_);
        return;
      }
    }

    active_trajectory_ = *msg;
    active_trajectory_mapping_ = mapping;
    active_trajectory_start_time_ = now;
    active_trajectory_received_time_ = now;
    active_trajectory_duration_s_ = duration;
    active_trajectory_seq_ = msg->header.seq;
    active_execution_stamp_ns_ = execution_stamp_ns;
    has_active_trajectory_ = true;
    has_received_trajectory_ = true;

    last_q_ref_ = q_start;
    last_dq_ref_ = dq_start;
    has_last_reference_ = true;
  }

  ROS_INFO_STREAM_THROTTLE(
      0.5,
      "[TrajectoryExecutionManager] accepted committed trajectory seq="
          << msg->header.seq
          << " execution_stamp_ns=" << active_execution_stamp_ns_
          << " duration="
          << duration << " s points=" << msg->points.size());
}

void TrajectoryExecutionManager::executionTimerCallback(
    const ros::TimerEvent&) {
  const ros::Time now = ros::Time::now();

  // These diagnostics describe the current timer tick. Reset them before any
  // early-return hold path so a previous trajectory cannot be mistaken for
  // the current one.
  last_tracking_bound_m_ = 0.0;
  last_same_phase_tracking_bound_m_ = 0.0;
  last_spatial_tracking_error_inf_ = 0.0;
  last_spatial_tracking_bound_m_ = 0.0;
  last_tracking_phase_match_s_ = 0.0;
  last_tracking_phase_lag_s_ = 0.0;
  last_tracking_phase_alignment_used_ = false;
  last_relative_fk_tracking_bound_m_ = 0.0;
  last_relative_fk_used_ = false;

  // A safety-triggered request may arrive during the ordinary request
  // coalescing interval.  Do not lose it when the tracker has already moved
  // into a zero-velocity hold and therefore no longer has a large tracking
  // error to retrigger the request path.
  flushPendingReplanRequest();

  Eigen::VectorXd q_measured;
  Eigen::VectorXd q_ref;
  Eigen::VectorXd dq_ref;
  Eigen::VectorXd ddq_ref;

  bool has_joint_state = false;
  bool has_received_trajectory = false;
  bool external_safety_hold = false;
  ros::Time external_safety_hold_start;
  unsigned long long external_safety_hold_event_count = 0;
  bool active = false;
  bool complete = false;
  uint32_t trajectory_seq = 0;
  uint64_t execution_stamp_ns = 0;
  bool have_reference = false;
  double measured_age_s = std::numeric_limits<double>::infinity();
  double phase_s = 0.0;
  double remaining_s = 0.0;
  double matched_phase_s = 0.0;
  Eigen::VectorXd q_spatial_ref;
  bool have_spatial_reference = false;
  std::string source = "none";

  {
    std::lock_guard<std::mutex> lock(data_mutex_);

    has_joint_state = has_joint_state_;
    has_received_trajectory = has_received_trajectory_;
    external_safety_hold = external_safety_hold_;
    external_safety_hold_start = external_safety_hold_start_time_;
    external_safety_hold_event_count = external_safety_hold_event_count_;
    if (has_joint_state) {
      q_measured = q_measured_;
      if (!latest_joint_state_.header.stamp.isZero()) {
        measured_age_s =
            (now - latest_joint_state_.header.stamp).toSec();
      }
    }

    if (has_active_trajectory_) {
      trajectory_seq = active_trajectory_seq_;
      execution_stamp_ns = active_execution_stamp_ns_;
      const ros::Time phase_now =
          (external_safety_hold && !external_safety_hold_start.isZero())
              ? external_safety_hold_start
              : now;
      phase_s = std::max(
          0.0, (phase_now - active_trajectory_start_time_).toSec());
      const double sample_t =
          std::min(phase_s, active_trajectory_duration_s_);

      if (sampleTrajectory(
              active_trajectory_,
              active_trajectory_mapping_,
              sample_t,
              q_ref,
              dq_ref,
              ddq_ref)) {
        have_reference = true;
        source = "active_trajectory";
        matched_phase_s = phase_s;
        q_spatial_ref = q_ref;
        if (has_joint_state && tracking_phase_alignment_enabled_) {
          have_spatial_reference = findNearestTrajectoryReference(
              active_trajectory_, active_trajectory_mapping_, phase_s,
              tracking_phase_search_window_s_, q_measured, matched_phase_s,
              q_spatial_ref);
        }
        remaining_s =
            std::max(0.0, active_trajectory_duration_s_ - phase_s);

        last_q_ref_ = q_ref;
        last_dq_ref_ = dq_ref;
        has_last_reference_ = true;

        if (phase_s >= active_trajectory_duration_s_ - 1e-9) {
          complete = true;
          has_active_trajectory_ = false;
          dq_ref.setZero();
          last_dq_ref_.setZero();
          source = "trajectory_complete_hold";
        } else {
          active = true;
        }
      } else {
        ROS_WARN_THROTTLE(
            1.0,
            "[TrajectoryExecutionManager] Failed to sample active trajectory.");
      }
    }

    if (!have_reference && has_last_reference_ &&
        hold_last_reference_when_no_trajectory_) {
      q_ref = last_q_ref_;
      dq_ref = Eigen::VectorXd::Zero(last_dq_ref_.size());
      have_reference = true;
      source = "last_reference_hold";
    }
  }

  if (!has_joint_state) {
    ROS_WARN_THROTTLE(
        1.0, "[TrajectoryExecutionManager] Waiting for joint state.");
    return;
  }

  if (external_safety_hold) {
    const Eigen::VectorXd dq_zero = makeZeroVelocityCommand();
    publishVelocityCommand(dq_zero);
    publishReferenceState(q_measured, dq_zero);
    publishSummary(
        active, complete, trajectory_seq, execution_stamp_ns,
        phase_s, remaining_s, 0.0,
        -1, std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        "external_gcdf_hard_hold");
    ROS_ERROR_STREAM_THROTTLE(
        0.5,
        "[TrajectoryExecutionManager] Phase E5 HARD HOLD active"
        << " events=" << external_safety_hold_event_count);
    return;
  }

  if (!have_reference) {
    if (!hold_when_no_trajectory_) return;

    const Eigen::VectorXd dq_zero = makeZeroVelocityCommand();
    if (hold_initial_zero_pose_ && !has_received_trajectory) {
      q_ref = Eigen::VectorXd::Zero(
          static_cast<int>(joint_names_.size()));
      dq_ref = dq_zero;
      source = "initial_zero_hold";
      have_reference = true;
    } else {
      publishVelocityCommand(dq_zero);
      publishReferenceState(q_measured, dq_zero);
      publishSummary(
          false, false, 0, 0, 0.0, 0.0, 0.0,
          -1, std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          "zero_velocity_hold");
      return;
    }
  }

  // The initial-zero and last-reference hold paths create q_ref after the
  // active-trajectory sampling block above.  Initialize the spatial reference
  // after all such paths have run, otherwise an empty vector could reach the
  // spatial residual calculation.
  if (have_reference && !have_spatial_reference) {
    q_spatial_ref = q_ref;
    matched_phase_s = phase_s;
  }

  const Eigen::VectorXd tracking_error_vec = q_ref - q_measured;
  Eigen::Index tracking_error_joint_index = -1;
  const double tracking_error =
      tracking_error_vec.cwiseAbs().maxCoeff(&tracking_error_joint_index);
  const double tracking_error_q_ref =
      (tracking_error_joint_index >= 0)
          ? q_ref[tracking_error_joint_index]
          : std::numeric_limits<double>::quiet_NaN();
  const double tracking_error_q_measured =
      (tracking_error_joint_index >= 0)
          ? q_measured[tracking_error_joint_index]
          : std::numeric_limits<double>::quiet_NaN();

  const Eigen::VectorXd spatial_tracking_error_vec =
      q_spatial_ref - q_measured;
  const double spatial_tracking_error =
      spatial_tracking_error_vec.cwiseAbs().maxCoeff();
  const double phase_lag_s = phase_s - matched_phase_s;
  last_spatial_tracking_error_inf_ = spatial_tracking_error;
  last_tracking_phase_match_s_ = matched_phase_s;
  last_tracking_phase_lag_s_ = phase_lag_s;
  last_tracking_phase_alignment_used_ = have_spatial_reference;

  const double replan_tracking_error =
      std::isfinite(spatial_tracking_error) ? spatial_tracking_error
                                            : tracking_error;
  maybePublishReplanRequest(replan_tracking_error);
  if (primitive_tracking_guard_ && has_received_trajectory) {
    try {
      const auto same_phase_bounds = tracking_geometry_.displacementBounds(
          joint_names_, tracking_error_vec);
      last_same_phase_tracking_bound_m_ = *std::max_element(
          same_phase_bounds.begin(), same_phase_bounds.end());

      const auto spatial_bounds = tracking_geometry_.displacementBounds(
          joint_names_, spatial_tracking_error_vec);
      last_spatial_tracking_bound_m_ = *std::max_element(
          spatial_bounds.begin(), spatial_bounds.end());
      const bool measured_state_fresh =
          measured_age_s >= 0 && measured_age_s <= reference_timeout_;
      if (!measured_state_fresh) {
        last_same_phase_tracking_bound_m_ =
            std::numeric_limits<double>::infinity();
        last_spatial_tracking_bound_m_ =
            std::numeric_limits<double>::infinity();
      }
      last_tracking_bound_m_ = last_spatial_tracking_bound_m_;
      // The legacy sum-of-radii bound is a cheap upper bound and remains the
      // fast path.  Only when it would trip the guard do we evaluate the
      // configuration-aware maximum primitive displacement.  This preserves
      // the normal 100 Hz cost while avoiding unnecessary conservative holds.
      if (primitive_tracking_relative_fk_ &&
          primitive_tracking_relative_fk_ready_ &&
          measured_state_fresh &&
          std::isfinite(last_spatial_tracking_bound_m_) &&
          last_spatial_tracking_bound_m_ > certified_tracking_margin_m_) {
        const auto relative_fk_bounds =
            tracking_geometry_.relativeFkDisplacementBounds(
                q_measured, q_spatial_ref);
        last_relative_fk_tracking_bound_m_ = *std::max_element(
            relative_fk_bounds.begin(), relative_fk_bounds.end());
        last_tracking_bound_m_ = last_relative_fk_tracking_bound_m_;
        last_relative_fk_used_ = true;
      }
    } catch (const std::exception&) {
      last_same_phase_tracking_bound_m_ =
          std::numeric_limits<double>::infinity();
      last_spatial_tracking_bound_m_ =
          std::numeric_limits<double>::infinity();
      last_relative_fk_tracking_bound_m_ =
          std::numeric_limits<double>::infinity();
      last_tracking_bound_m_ = std::numeric_limits<double>::infinity();
    }
    if (last_tracking_bound_m_ > certified_tracking_margin_m_) {
      // The reference sweep certificate does not authorize chasing a reference
      // outside its body-space envelope. Stop commanding that trajectory and
      // request a newly certified plan from the measured state. This is a
      // sampled runtime guard, not a bound on unmodelled actuator stopping.
      {
        std::lock_guard<std::mutex> lock(data_mutex_);
        if (active_execution_stamp_ns_ > 0) {
          aborted_execution_stamps_.insert(active_execution_stamp_ns_);
          // Keep stale-token memory bounded if a faulty upstream repeatedly
          // creates fresh tokens without restarting the tracker.
          if (aborted_execution_stamps_.size() > 128) {
            aborted_execution_stamps_.erase(aborted_execution_stamps_.begin());
          }
        }
        has_active_trajectory_ = false;
        // Do not let a protected trajectory become the fallback reference.
        // The next timer tick must issue a zero-velocity hold until a fresh,
        // newly certified trajectory is accepted from the measured state.
        has_last_reference_ = false;
        last_dq_ref_.setZero();
        replan_request_pending_execution_stamp_ns_ = execution_stamp_ns;
      }
      const auto zero = makeZeroVelocityCommand();
      publishVelocityCommand(zero);
      publishReferenceState(q_measured, zero);
      maybePublishReplanRequest(std::numeric_limits<double>::infinity());
      publishSummary(
          false, false, trajectory_seq, execution_stamp_ns, phase_s,
          remaining_s, tracking_error,
          static_cast<int>(tracking_error_joint_index), tracking_error_q_ref,
          tracking_error_q_measured, "primitive_tracking_envelope_hold");
      return;
    }
  }
  maybePublishSmoothHandoffReplanRequest(
      active, execution_stamp_ns, phase_s, remaining_s);

  if (tracking_error > max_tracking_error_) {
    ROS_WARN_THROTTLE(
        1.0,
        "[TrajectoryExecutionManager] Tracking error %.4f > %.4f; holding.",
        tracking_error, max_tracking_error_);

    if (hold_when_tracking_error_large_) {
      const Eigen::VectorXd dq_zero = makeZeroVelocityCommand();
      publishVelocityCommand(dq_zero);
      publishReferenceState(q_measured, dq_zero);
      publishSummary(
          active, complete, trajectory_seq, execution_stamp_ns,
          phase_s, remaining_s, tracking_error,
          static_cast<int>(tracking_error_joint_index),
          tracking_error_q_ref, tracking_error_q_measured,
          "tracking_error_hold");
      return;
    }
  }

  Eigen::VectorXd dq_cmd =
      computeVelocityCommand(q_ref, dq_ref, q_measured);
  dq_cmd = clampVelocityCommand(dq_cmd);

  publishVelocityCommand(dq_cmd);
  publishReferenceState(q_ref, dq_ref);
  publishSummary(
      active, complete, trajectory_seq, execution_stamp_ns,
      phase_s, remaining_s, tracking_error,
      static_cast<int>(tracking_error_joint_index),
      tracking_error_q_ref, tracking_error_q_measured, source);
}

bool TrajectoryExecutionManager::extractMeasuredState(
    const sensor_msgs::JointState& msg,
    Eigen::VectorXd& q,
    Eigen::VectorXd& dq) const {
  if (joint_names_.empty()) return false;

  std::unordered_map<std::string, std::size_t> name_to_index;
  for (std::size_t i = 0; i < msg.name.size(); ++i) {
    name_to_index[msg.name[i]] = i;
  }

  const int dof = static_cast<int>(joint_names_.size());
  q = Eigen::VectorXd::Zero(dof);
  dq = Eigen::VectorXd::Zero(dof);

  for (int i = 0; i < dof; ++i) {
    const auto it = name_to_index.find(joint_names_[i]);
    if (it == name_to_index.end()) return false;

    const std::size_t src_idx = it->second;
    if (src_idx >= msg.position.size()) return false;

    q[i] = msg.position[src_idx];
    dq[i] = src_idx < msg.velocity.size() ? msg.velocity[src_idx] : 0.0;
  }
  return true;
}

bool TrajectoryExecutionManager::trajectoryHasExpectedJoints(
    const trajectory_msgs::JointTrajectory& traj) const {
  if (traj.joint_names.empty()) return false;
  for (const auto& joint_name : joint_names_) {
    if (std::find(traj.joint_names.begin(),
                  traj.joint_names.end(),
                  joint_name) == traj.joint_names.end()) {
      return false;
    }
  }
  return true;
}

bool TrajectoryExecutionManager::buildTrajectoryJointIndexMap(
    const trajectory_msgs::JointTrajectory& traj,
    std::vector<int>& mapping) const {
  mapping.clear();
  mapping.reserve(joint_names_.size());
  for (const auto& joint_name : joint_names_) {
    const auto it = std::find(
        traj.joint_names.begin(), traj.joint_names.end(), joint_name);
    if (it == traj.joint_names.end()) return false;
    mapping.push_back(static_cast<int>(it - traj.joint_names.begin()));
  }
  return true;
}

bool TrajectoryExecutionManager::getPointVector(
    const trajectory_msgs::JointTrajectoryPoint&,
    const std::vector<double>& field,
    const std::vector<int>& mapping,
    Eigen::VectorXd& out,
    bool allow_missing_as_zero) {
  const int dof = static_cast<int>(mapping.size());
  out = Eigen::VectorXd::Zero(dof);

  if (field.empty()) return allow_missing_as_zero;

  for (int i = 0; i < dof; ++i) {
    const int src_idx = mapping[static_cast<std::size_t>(i)];
    if (src_idx < 0 ||
        static_cast<std::size_t>(src_idx) >= field.size()) {
      if (!allow_missing_as_zero) return false;
      continue;
    }
    out[i] = field[static_cast<std::size_t>(src_idx)];
  }
  return true;
}

bool TrajectoryExecutionManager::sampleTrajectory(
    const trajectory_msgs::JointTrajectory& traj,
    const std::vector<int>& mapping,
    double t,
    Eigen::VectorXd& q_ref,
    Eigen::VectorXd& dq_ref,
    Eigen::VectorXd& ddq_ref) {
  if (traj.points.empty() || mapping.empty()) return false;

  auto copyPoint =
      [&](const trajectory_msgs::JointTrajectoryPoint& p) -> bool {
        if (!getPointVector(p, p.positions, mapping, q_ref, false)) return false;
        getPointVector(p, p.velocities, mapping, dq_ref, true);
        getPointVector(p, p.accelerations, mapping, ddq_ref, true);
        return true;
      };

  if (traj.points.size() == 1) return copyPoint(traj.points.front());

  const double first_t = traj.points.front().time_from_start.toSec();
  const double last_t = traj.points.back().time_from_start.toSec();

  if (t <= first_t) return copyPoint(traj.points.front());
  if (t >= last_t) return copyPoint(traj.points.back());

  std::size_t hi = 1;
  while (hi < traj.points.size() &&
         traj.points[hi].time_from_start.toSec() < t) {
    ++hi;
  }
  if (hi >= traj.points.size()) return copyPoint(traj.points.back());

  const auto& p0 = traj.points[hi - 1];
  const auto& p1 = traj.points[hi];
  const double t0 = p0.time_from_start.toSec();
  const double t1 = p1.time_from_start.toSec();
  const double h = t1 - t0;
  if (h <= 1e-12) return false;

  const double s = (t - t0) / h;

  Eigen::VectorXd q0, q1, v0, v1, a0, a1;
  if (!getPointVector(p0, p0.positions, mapping, q0, false) ||
      !getPointVector(p1, p1.positions, mapping, q1, false)) {
    return false;
  }
  getPointVector(p0, p0.velocities, mapping, v0, true);
  getPointVector(p1, p1.velocities, mapping, v1, true);
  getPointVector(p0, p0.accelerations, mapping, a0, true);
  getPointVector(p1, p1.accelerations, mapping, a1, true);

  q_ref = (1.0 - s) * q0 + s * q1;
  dq_ref = (1.0 - s) * v0 + s * v1;
  ddq_ref = (1.0 - s) * a0 + s * a1;
  return true;
}

bool TrajectoryExecutionManager::findNearestTrajectoryReference(
    const trajectory_msgs::JointTrajectory& traj,
    const std::vector<int>& mapping,
    double center_t,
    double search_window_s,
    const Eigen::VectorXd& q_measured,
    double& matched_t,
    Eigen::VectorXd& q_match) {
  if (traj.points.empty() || mapping.empty() || !q_measured.allFinite() ||
      !std::isfinite(center_t) || !std::isfinite(search_window_s) ||
      search_window_s < 0.0) {
    return false;
  }

  const double first_t = traj.points.front().time_from_start.toSec();
  const double last_t = traj.points.back().time_from_start.toSec();
  if (!std::isfinite(first_t) || !std::isfinite(last_t) ||
      first_t < 0.0 || last_t < first_t) {
    return false;
  }

  const double window_lo = std::max(first_t, center_t - search_window_s);
  const double window_hi = std::min(last_t, center_t + search_window_s);
  if (!std::isfinite(window_lo) || !std::isfinite(window_hi) ||
      window_lo > window_hi + 1e-12) {
    return false;
  }

  bool found = false;
  double best_distance_sq = std::numeric_limits<double>::infinity();

  auto consider = [&](double t, const Eigen::VectorXd& q) {
    if (!std::isfinite(t) || q.size() != q_measured.size() ||
        !q.allFinite()) {
      return;
    }
    const double distance_sq = (q - q_measured).squaredNorm();
    if (!std::isfinite(distance_sq)) return;
    if (!found || distance_sq < best_distance_sq) {
      found = true;
      best_distance_sq = distance_sq;
      matched_t = t;
      q_match = q;
    }
  };

  if (traj.points.size() == 1) {
    Eigen::VectorXd q;
    Eigen::VectorXd dq;
    Eigen::VectorXd ddq;
    if (!getPointVector(traj.points.front(), traj.points.front().positions,
                        mapping, q, false)) {
      return false;
    }
    consider(std::max(first_t, std::min(last_t, center_t)), q);
    return found;
  }

  for (std::size_t i = 0; i + 1 < traj.points.size(); ++i) {
    const auto& p0 = traj.points[i];
    const auto& p1 = traj.points[i + 1];
    const double t0 = p0.time_from_start.toSec();
    const double t1 = p1.time_from_start.toSec();
    if (!std::isfinite(t0) || !std::isfinite(t1) || t1 <= t0) continue;
    if (t1 < window_lo - 1e-12 || t0 > window_hi + 1e-12) continue;

    Eigen::VectorXd q0;
    Eigen::VectorXd q1;
    if (!getPointVector(p0, p0.positions, mapping, q0, false) ||
        !getPointVector(p1, p1.positions, mapping, q1, false) ||
        q0.size() != q_measured.size() || q1.size() != q_measured.size()) {
      return false;
    }

    const double segment_lo_t = std::max(t0, window_lo);
    const double segment_hi_t = std::min(t1, window_hi);
    if (segment_lo_t > segment_hi_t + 1e-12) continue;
    const double h = t1 - t0;
    const double lambda_lo = std::max(0.0, (segment_lo_t - t0) / h);
    const double lambda_hi = std::min(1.0, (segment_hi_t - t0) / h);
    const Eigen::VectorXd delta = q1 - q0;

    auto considerLambda = [&](double lambda) {
      const double clipped = std::max(lambda_lo, std::min(lambda_hi, lambda));
      const Eigen::VectorXd q = (1.0 - clipped) * q0 + clipped * q1;
      consider(t0 + clipped * h, q);
    };

    // The tracker itself linearly interpolates positions, so the closest
    // point in Euclidean joint space on this segment has a closed form.
    // Evaluate the overlap endpoints as well so a clipped local window is
    // handled exactly.
    considerLambda(lambda_lo);
    considerLambda(lambda_hi);
    const double denominator = delta.squaredNorm();
    if (std::isfinite(denominator) && denominator > 1e-18) {
      considerLambda((q_measured - q0).dot(delta) / denominator);
    }
  }

  if (!found) {
    Eigen::VectorXd q;
    Eigen::VectorXd dq;
    Eigen::VectorXd ddq;
    const double fallback_t = std::max(first_t, std::min(last_t, center_t));
    if (!sampleTrajectory(traj, mapping, fallback_t, q, dq, ddq)) return false;
    consider(fallback_t, q);
  }
  return found;
}

double TrajectoryExecutionManager::getTrajectoryEndTime(
    const trajectory_msgs::JointTrajectory& traj) const {
  return traj.points.empty()
             ? 0.0
             : traj.points.back().time_from_start.toSec();
}

Eigen::VectorXd TrajectoryExecutionManager::computeVelocityCommand(
    const Eigen::VectorXd& q_ref,
    const Eigen::VectorXd& dq_ref,
    const Eigen::VectorXd& q_measured) const {
  Eigen::VectorXd dq_cmd = Eigen::VectorXd::Zero(q_ref.size());
  if (dq_ref.size() == q_ref.size()) dq_cmd = dq_ref;
  dq_cmd += position_feedback_gain_ * (q_ref - q_measured);
  return dq_cmd;
}

Eigen::VectorXd TrajectoryExecutionManager::clampVelocityCommand(
    const Eigen::VectorXd& dq_cmd) const {
  Eigen::VectorXd out = dq_cmd;
  for (int i = 0; i < out.size(); ++i) {
    out[i] = std::max(
        -max_command_velocity_,
        std::min(max_command_velocity_, out[i]));
  }
  return out;
}

void TrajectoryExecutionManager::publishVelocityCommand(
    const Eigen::VectorXd& dq_cmd) {
  std_msgs::Float64MultiArray msg;
  msg.data.resize(static_cast<std::size_t>(dq_cmd.size()));
  for (int i = 0; i < dq_cmd.size(); ++i) {
    msg.data[static_cast<std::size_t>(i)] = dq_cmd[i];
  }
  velocity_command_pub_.publish(msg);
}

void TrajectoryExecutionManager::publishReferenceState(
    const Eigen::VectorXd& q_ref,
    const Eigen::VectorXd& dq_ref) {
  sensor_msgs::JointState msg;
  msg.header.stamp = ros::Time::now();
  msg.name = joint_names_;
  msg.position.resize(static_cast<std::size_t>(q_ref.size()));
  msg.velocity.resize(static_cast<std::size_t>(dq_ref.size()));
  for (int i = 0; i < q_ref.size(); ++i)
    msg.position[static_cast<std::size_t>(i)] = q_ref[i];
  for (int i = 0; i < dq_ref.size(); ++i)
    msg.velocity[static_cast<std::size_t>(i)] = dq_ref[i];
  reference_state_pub_.publish(msg);
}

void TrajectoryExecutionManager::publishSummary(
    bool trajectory_active,
    bool trajectory_complete,
    uint32_t trajectory_seq,
    uint64_t execution_stamp_ns,
    double phase_s,
    double remaining_s,
    double tracking_error_inf,
    int tracking_error_joint_index,
    double tracking_error_q_ref,
    double tracking_error_q_measured,
    const std::string& source) {
  std_msgs::String msg;
  std::ostringstream oss;
  const bool execution_aborted =
      source == "primitive_tracking_envelope_hold" &&
      !trajectory_active && execution_stamp_ns > 0;
  oss << "TRACKER"
      << " active=" << static_cast<int>(trajectory_active)
      << " complete=" << static_cast<int>(trajectory_complete)
      << " seq=" << trajectory_seq
      << " execution_stamp_ns=" << execution_stamp_ns
      << " execution_aborted=" << static_cast<int>(execution_aborted)
      << " phase_s=" << phase_s
      << " remaining_s=" << remaining_s
      << " tracking_error_inf=" << tracking_error_inf
      << " tracking_error_joint_index=" << tracking_error_joint_index
      << " tracking_error_joint_name="
      << ((tracking_error_joint_index >= 0 &&
           static_cast<std::size_t>(tracking_error_joint_index) < joint_names_.size())
              ? joint_names_[static_cast<std::size_t>(tracking_error_joint_index)]
              : "none")
      << " tracking_error_q_ref=" << tracking_error_q_ref
      << " primitive_tracking_bound_m=" << last_tracking_bound_m_
      << " primitive_tracking_legacy_bound_m="
      << last_spatial_tracking_bound_m_
      << " primitive_tracking_relative_fk_bound_m="
      << last_relative_fk_tracking_bound_m_
      << " primitive_tracking_relative_fk_used="
      << static_cast<int>(last_relative_fk_used_)
      << " primitive_tracking_same_phase_bound_m="
      << last_same_phase_tracking_bound_m_
      << " spatial_tracking_error_inf="
      << last_spatial_tracking_error_inf_
      << " spatial_tracking_bound_m=" << last_spatial_tracking_bound_m_
      << " tracking_phase_match_s=" << last_tracking_phase_match_s_
      << " tracking_phase_lag_s=" << last_tracking_phase_lag_s_
      << " tracking_phase_alignment_used="
      << static_cast<int>(last_tracking_phase_alignment_used_)
      << " tracking_phase_search_window_s="
      << tracking_phase_search_window_s_
      << " certified_tracking_margin_m=" << certified_tracking_margin_m_
      << " replan_request_pending="
      << static_cast<int>(replan_request_pending_)
      << " replan_request_pending_execution_stamp_ns="
      << replan_request_pending_execution_stamp_ns_
      << " replan_request_count=" << replan_request_count_
      << " replan_request_deferred_count="
      << replan_request_deferred_count_
      << " tracking_error_q_measured=" << tracking_error_q_measured
      << " smooth_handoff_enabled="
      << static_cast<int>(smooth_handoff_enabled_)
      << " smooth_replan_request_count="
      << smooth_replan_request_count_
      << " smooth_replan_lead_time_s="
      << smooth_replan_lead_time_s_
      << " source=" << source;
  msg.data = oss.str();
  summary_pub_.publish(msg);
}

void TrajectoryExecutionManager::maybePublishReplanRequest(
    double tracking_error_inf) {
  if (replan_tracking_error_inf_ <= 0.0 ||
      tracking_error_inf <= replan_tracking_error_inf_) {
    return;
  }

  const ros::Time now = ros::Time::now();
  if (!last_replan_request_time_.isZero() &&
      (now - last_replan_request_time_).toSec() <
          replan_request_min_interval_s_) {
    replan_request_pending_ = true;
    if (replan_request_pending_time_.isZero()) {
      replan_request_pending_time_ = now;
    }
    ++replan_request_deferred_count_;
    return;
  }

  std_msgs::Bool msg;
  msg.data = true;
  replan_request_pub_.publish(msg);
  last_replan_request_time_ = now;
  replan_request_pending_ = false;
  replan_request_pending_time_ = ros::Time(0);
  replan_request_pending_execution_stamp_ns_ = 0;
  ++replan_request_count_;

  ROS_WARN_STREAM_THROTTLE(
      0.5,
      "[TrajectoryExecutionManager] requested local replan: tracking_error_inf="
          << tracking_error_inf);
}

void TrajectoryExecutionManager::flushPendingReplanRequest() {
  if (!replan_request_pending_ || last_replan_request_time_.isZero()) {
    return;
  }

  const ros::Time now = ros::Time::now();
  if ((now - last_replan_request_time_).toSec() <
      replan_request_min_interval_s_) {
    return;
  }

  std_msgs::Bool msg;
  msg.data = true;
  replan_request_pub_.publish(msg);
  last_replan_request_time_ = now;
  replan_request_pending_ = false;
  replan_request_pending_time_ = ros::Time(0);
  replan_request_pending_execution_stamp_ns_ = 0;
  ++replan_request_count_;

  ROS_WARN_STREAM(
      "[TrajectoryExecutionManager] flushed deferred local replan request");
}

void TrajectoryExecutionManager::maybePublishSmoothHandoffReplanRequest(
    bool trajectory_active,
    uint64_t execution_stamp_ns,
    double phase_s,
    double remaining_s) {
  if (!smooth_handoff_enabled_ || !trajectory_active ||
      execution_stamp_ns == 0 ||
      phase_s < smooth_replan_min_phase_s_ ||
      remaining_s > smooth_replan_lead_time_s_) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (last_smooth_replan_execution_stamp_ns_ == execution_stamp_ns) return;
    last_smooth_replan_execution_stamp_ns_ = execution_stamp_ns;
    ++smooth_replan_request_count_;
  }

  std_msgs::Bool msg;
  msg.data = true;
  smooth_replan_request_pub_.publish(msg);
  ROS_INFO_STREAM(
      "[TrajectoryExecutionManager] smooth look-ahead replan requested"
      << " execution_stamp_ns=" << execution_stamp_ns
      << " phase_s=" << phase_s
      << " remaining_s=" << remaining_s
      << " count=" << smooth_replan_request_count_);
}

Eigen::VectorXd TrajectoryExecutionManager::makeZeroVelocityCommand() const {
  return Eigen::VectorXd::Zero(static_cast<int>(joint_names_.size()));
}

}  // namespace egocentric_arm_planner
