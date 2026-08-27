#pragma once

#include <ros/ros.h>

#include <care_collision_cdf/CollisionCDFConstraintBatch.h>

#include <sensor_msgs/JointState.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/String.h>
#include <trajectory_msgs/JointTrajectory.h>

#include <Eigen/Dense>
#include <piqp/piqp.hpp>

#include <cmath>
#include <thread>
#include <deque>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

namespace egocentric_arm_planner {

/**
 * CAREPlanner VBC deadline-waypoint candidate planner.
 *
 * NORMAL:
 *   J = J_task + J_control + J_smooth
 *
 * INTERVENTION (fresh active q_vis):
 *   J = J_task + J_control + J_smooth
 *       + w_vis ||q_kd - q_vis||^2
 *
 * VERIFICATION_HOLD (C4 audit temporarily unavailable):
 *   J = J_control_effort + J_smooth
 *
 * REPAIR, legacy single-waypoint mode:
 *   J = J_control_effort + J_smooth
 *       + w_vis ||q_K - q_vis||^2
 *
 * C4.6 accumulated multi-deadline REPAIR:
 *   J = J_control_effort + J_smooth
 *       + sum_r w_vis ||q_{k_r} - q_vis^(r)||^2
 *
 * where k_r is computed from each obligation's absolute VBC deadline.  Expired
 * obligations are applied at k=1 (act immediately); obligations beyond the
 * current horizon are applied at k=K.  The exact VBC verifier remains the hard
 * commit authority, so these waypoint costs only shape candidate generation.
 *
 * Legacy experiments treat RECOVERY as a controller episode: clearing it enters
 * RECOVERY_HOLD, publishes recovery_complete, and waits for replan_ready.
 *
 * C4.4+ can enable ``planner_mode_semantics``. In that mode RECOVERY is only a
 * candidate-generation objective (REPAIR): trigger selects the repair objective
 * and clear selects the normal objective. Switching objective never creates a
 * hold, never requests a measured-state replan, and never changes the currently
 * committed execution trajectory; commit/reject is owned downstream by the
 * verified-commit layer.
 *
 * Task position/terminal tracking and nominal velocity tracking are removed in
 * VERIFICATION_HOLD and REPAIR. Hard position/velocity/acceleration constraints
 * are unchanged.
 */
class VelocityQPMPCWaypoint {
public:
  VelocityQPMPCWaypoint() = default;
  ~VelocityQPMPCWaypoint();

  bool initialize(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);

  // Planner/controller split support. In verified-commit mode a raw candidate
  // may be rejected and never executed, so the first-step acceleration and
  // smoothness anchor must come from the command that the low-level controller
  // actually sent to the robot, not from the previous raw MPC solution.
  void setExecutedCommandAnchor(
      const std_msgs::Float64MultiArrayConstPtr& msg) {
    if (!msg || msg->data.size() != static_cast<std::size_t>(dof_)) return;
    Eigen::VectorXd command(dof_);
    for (int i = 0; i < dof_; ++i) {
      const double value = msg->data[static_cast<std::size_t>(i)];
      if (!std::isfinite(value)) return;
      command[i] = value;
    }
    std::lock_guard<std::mutex> lock(data_mutex_);
    previous_command_ = command;
  }

private:
  struct DeadlineWaypoint {
    long long id = -1;
    double deadline_abs_s = 0.0;
    Eigen::VectorXd q;
  };

  struct CDFQPSnapshot {
    ros::Time prediction_stamp;
    ros::WallTime created_wall;
    Eigen::VectorXd q_current;
    Eigen::MatrixXd hessian;
    Eigen::VectorXd gradient;
    Eigen::VectorXd lower;
    Eigen::VectorXd upper;
    Eigen::VectorXd raw_solution;
    std::string frame_id;
    std::string control_mode;
  };

  struct CDFShadowJob {
    CDFQPSnapshot snapshot;
    care_collision_cdf::CollisionCDFConstraintBatchConstPtr batch;
  };

  void jointStateCallback(const sensor_msgs::JointStateConstPtr& msg);
  void referenceCallback(const trajectory_msgs::JointTrajectoryConstPtr& msg);
  void waypointActiveCallback(const std_msgs::BoolConstPtr& msg);
  void waypointQCallback(const std_msgs::Float64MultiArrayConstPtr& msg);
  void waypointDeadlineCallback(const std_msgs::Float64ConstPtr& msg);
  void waypointScheduleCallback(const std_msgs::Float64MultiArrayConstPtr& msg);
  void verificationHoldCallback(const std_msgs::BoolConstPtr& msg);
  void recoveryTriggerCallback(const std_msgs::BoolConstPtr& msg);
  void recoveryClearCallback(const std_msgs::BoolConstPtr& msg);
  void replanReadyCallback(const std_msgs::BoolConstPtr& msg);
  void cdfConstraintBatchCallback(
      const care_collision_cdf::CollisionCDFConstraintBatchConstPtr& msg);
  void timerCallback(const ros::TimerEvent& event);

  bool loadConfig();
  bool loadJointVectorParam(const std::string& param_name,
                            const Eigen::VectorXd& fallback,
                            Eigen::VectorXd& value) const;
  bool loadPositionLimitsFromUrdf();
  bool buildStaticQP();

  bool buildReferenceHorizon(const trajectory_msgs::JointTrajectory& msg,
                             Eigen::MatrixXd& q_ref,
                             Eigen::MatrixXd& u_ref) const;
  bool sampleReferencePosition(const trajectory_msgs::JointTrajectory& msg,
                               const std::vector<int>& mapping,
                               double t,
                               Eigen::VectorXd& q) const;
  bool buildTrajectoryMapping(const trajectory_msgs::JointTrajectory& msg,
                              std::vector<int>& mapping) const;
  bool extractMeasuredQ(const sensor_msgs::JointState& msg,
                        Eigen::VectorXd& q) const;

  void buildBounds(const Eigen::VectorXd& q_current,
                   Eigen::VectorXd& lower,
                   Eigen::VectorXd& upper) const;
  void buildBaseCycleQP(const Eigen::VectorXd& q_current,
                        const Eigen::MatrixXd& q_ref,
                        const Eigen::MatrixXd& u_ref,
                        Eigen::VectorXd& gradient,
                        Eigen::VectorXd& lower,
                        Eigen::VectorXd& upper) const;
  void buildRegularizationCycleQP(const Eigen::VectorXd& q_current,
                                  Eigen::VectorXd& gradient,
                                  Eigen::VectorXd& lower,
                                  Eigen::VectorXd& upper) const;

  bool solveWithPIQP(const Eigen::MatrixXd& hessian,
                     const Eigen::VectorXd& gradient,
                     const Eigen::VectorXd& lower,
                     const Eigen::VectorXd& upper,
                     Eigen::VectorXd& solution,
                     int& iterations,
                     double& primal_residual,
                     double& dual_residual,
                     std::string& status_string) const;

  void startCDFShadowWorker();
  void stopCDFShadowWorker();
  void cdfShadowWorkerLoop();
  bool solveCDFShadowJob(
      const CDFShadowJob& job,
      Eigen::VectorXd& solution,
      int& iterations,
      double& primal_residual,
      double& dual_residual,
      std::string& status_string,
      int& active_constraint_rows,
      int& skipped_step0_rows,
      int& skipped_horizon_rows,
      double& max_linearization_q_error_inf,
      double& min_raw_distance,
      double& min_linearized_shadow_distance,
      double& solve_ms) const;
  void publishCDFShadowPrediction(
      const Eigen::MatrixXd& q_pred,
      const Eigen::VectorXd& u_stack,
      const std::string& frame_id,
      const ros::Time& source_stamp);
  void publishCDFShadowSummary(
      const CDFShadowJob& job,
      bool solved,
      const std::string& status,
      int iterations,
      double primal,
      double dual,
      int active_constraint_rows,
      int skipped_step0_rows,
      int skipped_horizon_rows,
      double max_linearization_q_error_inf,
      double min_raw_distance,
      double min_linearized_shadow_distance,
      double solve_ms,
      double end_to_end_age_ms);

  Eigen::MatrixXd reconstructPredictedQ(const Eigen::VectorXd& q_current,
                                        const Eigen::VectorXd& u_stack) const;
  void publishVelocity(const Eigen::VectorXd& command);
  void publishSafeStop(const std::string& reason);
  void publishPrediction(const Eigen::MatrixXd& q_pred,
                         const Eigen::VectorXd& u_stack,
                         const std::string& frame_id,
                         const ros::Time& stamp);
  void publishRecoveryActive(bool active);
  void publishRecoveryComplete();

private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  ros::Subscriber joint_state_sub_;
  ros::Subscriber reference_sub_;
  ros::Subscriber waypoint_active_sub_;
  ros::Subscriber waypoint_q_sub_;
  ros::Subscriber waypoint_deadline_sub_;
  ros::Subscriber waypoint_schedule_sub_;
  ros::Subscriber verification_hold_sub_;
  ros::Subscriber recovery_trigger_sub_;
  ros::Subscriber recovery_clear_sub_;
  ros::Subscriber replan_ready_sub_;
  ros::Subscriber cdf_constraint_batch_sub_;

  ros::Publisher velocity_command_pub_;
  ros::Publisher prediction_pub_;
  ros::Publisher solve_time_pub_;
  ros::Publisher summary_pub_;
  ros::Publisher recovery_active_pub_;
  ros::Publisher recovery_complete_pub_;
  ros::Publisher cdf_shadow_prediction_pub_;
  ros::Publisher cdf_shadow_summary_pub_;
  ros::Timer timer_;

  mutable std::mutex data_mutex_;
  sensor_msgs::JointState latest_joint_state_;
  trajectory_msgs::JointTrajectory latest_reference_;
  ros::Time latest_joint_state_received_;
  ros::Time latest_reference_received_;
  bool has_joint_state_ = false;
  bool has_reference_ = false;

  bool latest_waypoint_active_ = false;
  bool has_waypoint_active_ = false;
  bool has_waypoint_q_ = false;
  bool has_waypoint_deadline_ = false;
  ros::Time latest_waypoint_active_received_;
  ros::Time latest_waypoint_q_received_;
  ros::Time latest_waypoint_deadline_received_;
  Eigen::VectorXd latest_waypoint_q_;
  double latest_waypoint_deadline_abs_s_ = 0.0;

  // C4.6 schedule message format is a flat sequence of 9 doubles per record:
  //   [obligation_id, absolute_deadline_ros_s, q1, ..., q7]
  // The schedule producer accumulates obligations across rejected candidates and
  // clears them only after an exact predicted-VBC SAFE verdict.
  bool multi_deadline_enabled_ = false;
  bool has_waypoint_schedule_ = false;
  ros::Time latest_waypoint_schedule_received_;
  std::vector<DeadlineWaypoint> latest_waypoint_schedule_;
  int max_repair_waypoints_ = 8;

  bool latest_verification_hold_ = false;
  bool has_verification_hold_ = false;
  ros::Time latest_verification_hold_received_;

  bool use_external_recovery_trigger_ = false;
  bool latest_recovery_trigger_ = false;
  bool has_recovery_trigger_ = false;
  ros::Time latest_recovery_trigger_received_;

  bool use_external_recovery_clear_ = false;
  bool latest_recovery_clear_ = false;
  bool has_recovery_clear_ = false;
  ros::Time latest_recovery_clear_received_;
  double recovery_signal_timeout_ = 0.25;

  bool recovery_enabled_ = true;
  // C4.4: when true, RECOVERY is only a candidate-planning objective selector.
  // Clearing it never enters recovery_hold or emits replan handshakes.
  bool planner_mode_semantics_ = false;
  bool recovery_active_ = false;
  bool recovery_hold_ = false;
  bool recovery_complete_published_ = false;
  bool replan_ready_received_ = false;
  double recovery_weight_scale_ = 1.0;

  std::vector<std::string> joint_names_;
  int dof_ = 7;

  double rate_ = 20.0;
  double horizon_duration_ = 1.0;
  int num_intervals_ = 20;
  double dt_ = 0.05;
  double control_period_ = 0.05;

  double q_tracking_weight_ = 50.0;
  double terminal_q_tracking_weight_ = 100.0;
  double u_tracking_weight_ = 2.0;
  double u_smooth_weight_ = 1.0;

  bool waypoint_enabled_ = true;
  double waypoint_weight_ = 100.0;
  double waypoint_timeout_ = 0.20;
  double waypoint_horizon_slack_ = 0.025;

  Eigen::VectorXd velocity_limits_;
  Eigen::VectorXd acceleration_limits_;
  Eigen::VectorXd q_min_;
  Eigen::VectorXd q_max_;
  Eigen::VectorXd previous_command_;

  double joint_position_margin_ = 0.01;
  double joint_state_timeout_ = 0.20;
  double reference_timeout_ = 0.20;

  int piqp_max_iterations_ = 100;
  double piqp_eps_abs_ = 1e-6;
  double piqp_eps_rel_ = 1e-6;
  bool piqp_verbose_ = false;
  bool piqp_compute_timings_ = true;

  std::string robot_description_param_ = "/robot_description";
  std::string joint_state_topic_ = "/care_arm/joint_states";
  std::string reference_topic_ = "/care_planner/command_trajectory_candidate";
  std::string velocity_command_topic_ =
      "/care_arm/arm_group_velocity_controller/command";
  std::string prediction_topic_ = "/care_planner/mpc/predicted_trajectory";

  std::string waypoint_active_topic_ =
      "/care_planner/active_sensing/visibility_waypoint_active";
  std::string waypoint_q_topic_ =
      "/care_planner/active_sensing/visibility_waypoint_q";
  std::string waypoint_deadline_topic_ =
      "/care_planner/active_sensing/visibility_waypoint_deadline";
  std::string waypoint_schedule_topic_ =
      "/care_planner/active_sensing/visibility_waypoint_schedule";
  std::string verification_hold_topic_ =
      "/care_planner/execution/predicted_vbc_verification_hold";
  std::string recovery_trigger_topic_ =
      "/care_planner/execution/predicted_vbc_recovery_triggered";
  std::string recovery_clear_topic_ =
      "/care_planner/execution/predicted_vbc_recovery_clear";
  std::string recovery_active_topic_ =
      "/care_planner/execution/visibility_recovery_active";
  std::string recovery_complete_topic_ =
      "/care_planner/execution/visibility_recovery_complete";
  std::string replan_ready_topic_ =
      "/care_planner/execution/visibility_replan_ready";

  bool cdf_shadow_enabled_ = false;
  double cdf_safety_margin_ = 0.0;
  int cdf_constraint_horizon_steps_ = 20;
  double cdf_snapshot_timeout_s_ = 0.75;
  int cdf_snapshot_history_size_ = 12;
  double cdf_stamp_tolerance_s_ = 1e-6;
  std::string cdf_constraint_batch_topic_ =
      "/care_planner/collision_cdf/constraint_batch";
  std::string cdf_shadow_prediction_topic_ =
      "/care_planner/mpc/cdf_shadow_predicted_trajectory";
  std::string cdf_shadow_summary_topic_ =
      "/care_planner/mpc/cdf_shadow_summary";

  Eigen::MatrixXd S_;
  Eigen::MatrixXd S_terminal_;
  Eigen::MatrixXd D_;
  Eigen::MatrixXd G_;
  Eigen::MatrixXd H_base_;
  Eigen::MatrixXd H_regularization_;
  Eigen::VectorXd x_lower_;
  Eigen::VectorXd x_upper_;

  int n_u_ = 0;
  int n_constraints_ = 0;
  int acceleration_row0_ = 0;
  int position_row0_ = 0;

  mutable std::mutex cdf_shadow_mutex_;
  std::condition_variable cdf_shadow_cv_;
  std::deque<CDFQPSnapshot> cdf_shadow_snapshots_;
  std::unique_ptr<CDFShadowJob> cdf_shadow_pending_job_;
  std::thread cdf_shadow_worker_;
  bool cdf_shadow_worker_stop_ = false;
  unsigned long long cdf_shadow_job_received_ = 0;
  unsigned long long cdf_shadow_job_processed_ = 0;
  unsigned long long cdf_shadow_job_dropped_ = 0;
  unsigned long long cdf_shadow_stamp_miss_ = 0;

  unsigned long long sequence_ = 0;
};

}  // namespace egocentric_arm_planner
