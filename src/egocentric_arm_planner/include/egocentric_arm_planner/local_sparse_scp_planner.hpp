#pragma once

#include <ros/ros.h>

#include <care_collision_cdf/CollisionCDFConstraintBatch.h>

#include <sensor_msgs/JointState.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/String.h>
#include <trajectory_msgs/JointTrajectory.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace egocentric_arm_planner {

class LocalSparseSCPPlanner {
public:
  LocalSparseSCPPlanner() = default;
  ~LocalSparseSCPPlanner();

  bool initialize(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);

private:
  struct DeadlineWaypoint {
    long long id = 0;
    double deadline_abs_s = 0.0;
    Eigen::VectorXd q;
  };

  struct SparseSolveResult {
    bool solved = false;
    std::string status = "not_solved";
    int iterations = 0;
    double primal_residual = 0.0;
    double dual_residual = 0.0;
    double setup_and_solve_ms = 0.0;
    int batch_pairs = 0;
    int selected_cdf_rows = 0;
    int screened_safe_rows = 0;
    int skipped_step0_rows = 0;
    int skipped_horizon_rows = 0;
    int skipped_safety_horizon_rows = 0;
    double qlin_error_inf = 0.0;
    double min_distance = 0.0;
    double max_slack = 0.0;
    double mean_slack = 0.0;
    double slack_linear_weight_used = 0.0;
    double step_inf = 0.0;
    Eigen::MatrixXd q;
    Eigen::MatrixXd u;
  };

  void jointStateCallback(const sensor_msgs::JointStateConstPtr& msg);
  void referenceCallback(
      const trajectory_msgs::JointTrajectoryConstPtr& msg);
  void waypointScheduleCallback(
      const std_msgs::Float64MultiArrayConstPtr& msg);
  void singleWaypointActiveCallback(const std_msgs::BoolConstPtr& msg);
  void singleWaypointQCallback(
      const std_msgs::Float64MultiArrayConstPtr& msg);
  void recoveryCallback(const std_msgs::BoolConstPtr& msg);
  void probeActiveCallback(const std_msgs::BoolConstPtr& msg);
  void replanRequestCallback(const std_msgs::BoolConstPtr& msg);
  void executedCommandCallback(
      const std_msgs::Float64MultiArrayConstPtr& msg);
  void executionSummaryCallback(const std_msgs::StringConstPtr& msg);
  void cdfConstraintBatchCallback(
      const care_collision_cdf::CollisionCDFConstraintBatchConstPtr& msg);
  void timerCallback(const ros::TimerEvent&);

  bool loadConfig();
  bool loadJointLimits();
  bool extractMeasuredQ(const sensor_msgs::JointState& msg,
                        Eigen::VectorXd& q) const;
  bool buildTrajectoryMapping(
      const trajectory_msgs::JointTrajectory& msg,
      std::vector<int>& mapping) const;
  bool sampleReferencePosition(
      const trajectory_msgs::JointTrajectory& msg,
      const std::vector<int>& mapping,
      double t,
      Eigen::VectorXd& q) const;
  bool buildReferenceHorizon(
      const Eigen::VectorXd& q_current,
      const trajectory_msgs::JointTrajectory& reference,
      const ros::Time& now,
      Eigen::MatrixXd& q_ref,
      Eigen::MatrixXd& u_ref,
      Eigen::MatrixXd& q_init,
      Eigen::MatrixXd& u_init) const;

  void requestPlanLocked(const std::string& reason);
  bool startPlan();
  void abortPlan(const std::string& reason);

  void startWorker();
  void stopWorker();
  void workerLoop();

  SparseSolveResult solveSparseSubproblem(
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
      bool force_diagnostic_slack = false) const;

  ros::Time publishQueryTrajectory(
      const Eigen::MatrixXd& q,
      const Eigen::MatrixXd& u,
      const std::string& frame_id);
  void publishCandidateTrajectory(
      const Eigen::MatrixXd& q,
      const Eigen::MatrixXd& u,
      const std::string& frame_id);
  trajectory_msgs::JointTrajectory makeTrajectoryMessage(
      const Eigen::MatrixXd& q,
      const Eigen::MatrixXd& u,
      const std::string& frame_id,
      const ros::Time& stamp) const;

  void publishSummary(const std::string& event,
                      const SparseSolveResult* result = nullptr,
                      double total_plan_ms = 0.0);

  int qIndex(int k, int j) const;
  int uIndex(int k, int j) const;

private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  ros::Subscriber joint_state_sub_;
  ros::Subscriber reference_sub_;
  ros::Subscriber waypoint_schedule_sub_;
  ros::Subscriber single_waypoint_active_sub_;
  ros::Subscriber single_waypoint_q_sub_;
  ros::Subscriber recovery_sub_;
  ros::Subscriber probe_active_sub_;
  ros::Subscriber replan_request_sub_;
  ros::Subscriber executed_command_sub_;
  ros::Subscriber execution_summary_sub_;
  ros::Subscriber cdf_batch_sub_;

  ros::Publisher query_trajectory_pub_;
  ros::Publisher candidate_trajectory_pub_;
  ros::Publisher summary_pub_;
  ros::Publisher task_infeasible_pub_;
  ros::Publisher task_uncertified_pub_;
  ros::Publisher force_vbc_bootstrap_pub_;

  ros::Timer timer_;

  mutable std::mutex mutex_;
  sensor_msgs::JointState latest_joint_state_;
  trajectory_msgs::JointTrajectory latest_reference_;
  std::vector<DeadlineWaypoint> latest_schedule_;
  bool latest_single_waypoint_active_ = false;
  bool has_single_waypoint_q_ = false;
  Eigen::VectorXd latest_single_waypoint_q_;
  Eigen::VectorXd latest_executed_command_;
  ros::Time latest_joint_state_received_;
  ros::Time latest_reference_received_;
  ros::Time latest_executed_command_received_;
  bool latest_execution_complete_ = false;

  bool has_joint_state_ = false;
  bool has_reference_ = false;
  bool repair_mode_ = false;
  bool probe_mode_ = false;
  bool plan_requested_ = false;
  std::string plan_request_reason_ = "none";
  ros::Time last_plan_finish_time_;

  // Active SCP state.
  bool plan_running_ = false;
  bool waiting_for_cdf_ = false;
  int scp_iteration_ = 0;
  double trust_radius_ = 0.20;
  double previous_query_min_distance_ =
      std::numeric_limits<double>::quiet_NaN();
  double plan_cdf_slack_linear_weight_ = 10.0;
  ros::Time current_query_stamp_;
  ros::WallTime current_query_wall_;
  ros::WallTime current_plan_start_wall_;
  std::string current_frame_id_ = "base_link";

  Eigen::VectorXd plan_q_current_;
  Eigen::VectorXd plan_previous_command_;
  Eigen::MatrixXd plan_q_ref_;
  Eigen::MatrixXd plan_u_ref_;
  Eigen::MatrixXd plan_q_bar_;
  Eigen::MatrixXd plan_u_bar_;
  std::vector<DeadlineWaypoint> plan_schedule_;
  bool plan_repair_mode_ = false;
  bool plan_probe_mode_ = false;
  std::string plan_initialization_mode_ = "task_reference";

  // One CDF batch per SCP iterate. The callback only hands ownership to the
  // worker; PIQP never runs in a ROS callback.
  std::condition_variable worker_cv_;
  std::thread worker_;
  bool worker_stop_ = false;
  care_collision_cdf::CollisionCDFConstraintBatchConstPtr pending_batch_;

  std::vector<std::string> joint_names_;
  int dof_ = 7;
  int num_intervals_ = 20;
  double horizon_duration_ = 1.0;
  double dt_ = 0.05;
  double planner_poll_rate_ = 50.0;
  double min_replan_interval_s_ = 0.10;
  double cdf_wait_timeout_s_ = 0.50;
  double cdf_stamp_tolerance_s_ = 1e-6;

  int max_scp_iterations_ = 3;
  double scp_step_tolerance_inf_ = 0.01;
  double trust_region_initial_ = 0.20;
  double trust_region_min_ = 0.05;
  double trust_region_max_ = 0.50;
  double trust_region_grow_ = 1.50;
  double trust_region_shrink_ = 0.50;
  double trust_region_improvement_tol_ = 0.002;

  double q_tracking_weight_ = 50.0;
  double terminal_q_tracking_weight_ = 100.0;
  double u_tracking_weight_ = 2.0;
  // Backward-compatible switch: true reproduces the existing CARE
  // ||u-u_ref||^2 objective; false gives the GCDF-style ||u||^2 control cost.
  bool u_reference_tracking_enabled_ = true;
  double u_smooth_weight_ = 1.0;
  // CARE execution normally enforces hard acceleration and terminal-braking
  // rows. G0 disables them to match the GCDF formulation more closely.
  bool enforce_acceleration_constraints_ = true;
  // In REPAIR, initialize SCP from a measured hold trajectory instead of the
  // known-infeasible task interpolation. The visibility objective then moves
  // outward only as far as hard GCDF safety permits.
  bool repair_hold_initialization_enabled_ = false;
  double repair_task_tracking_scale_ = 0.0;
  double visibility_waypoint_weight_ = 3000.0;

  double cdf_safety_margin_ = 0.0;
  double cdf_slack_linear_weight_ = 500.0;
  double cdf_slack_quadratic_weight_ = 1e-3;
  double cdf_slack_upper_bound_ = 2.0;
  // Existing CARE runs share one user slack across all CDF rows at a
  // timestep. G0 can switch to the GCDF convention: one slack per inequality.
  bool cdf_per_constraint_slack_ = false;
  // GCDF slacks are nonnegative and not artificially capped. Keep the old
  // upper bound available for backwards compatibility.
  bool cdf_slack_use_upper_bound_ = true;
  // G0.2 diagnostic can remove user slacks entirely and test whether the
  // linearized CDF-constrained subproblem is genuinely feasible.
  bool cdf_slack_enabled_ = true;
  // Diagnostic-only fallback: after a hard task QP failure, solve the same
  // convex subproblem with per-constraint nonnegative CDF slacks. The result is
  // never executable; max_slack only diagnoses numerical failure vs. genuine
  // local hard-GCDF infeasibility.
  bool task_failure_slack_diagnostic_enabled_ = true;
  double task_failure_slack_diagnostic_weight_ = 1.0;
  bool cdf_adaptive_slack_penalty_ = false;
  double cdf_slack_penalty_multiplier_ = 10.0;
  double cdf_slack_penalty_max_ = 1e8;
  double cdf_slack_tolerance_ = 5e-3;
  bool cdf_safe_row_screening_ = true;
  double cdf_linearization_tolerance_inf_ = 1e-4;
  // Unknown/low-confidence CDF is enforced over the actually executable
  // prefix in REPAIR/PROBE. NORMAL keeps the full planning horizon.
  int cdf_constraint_horizon_steps_ = 20;

  int piqp_max_iterations_ = 100;
  double piqp_eps_abs_ = 1e-5;
  double piqp_eps_rel_ = 1e-5;
  bool piqp_verbose_ = false;

  Eigen::VectorXd velocity_limits_;
  Eigen::VectorXd acceleration_limits_;
  Eigen::VectorXd q_min_;
  Eigen::VectorXd q_max_;
  double joint_position_margin_ = 0.01;

  std::string joint_state_topic_ = "/care_arm/joint_states";
  std::string reference_topic_ = "/care_planner/task_trajectory";
  std::string waypoint_schedule_topic_ =
      "/care_planner/active_sensing/visibility_waypoint_schedule";
  std::string single_waypoint_active_topic_ =
      "/care_planner/active_sensing/visibility_waypoint_active";
  std::string single_waypoint_q_topic_ =
      "/care_planner/active_sensing/visibility_waypoint_q";
  std::string recovery_topic_ =
      "/care_planner/execution/predicted_vbc_recovery_triggered";
  std::string probe_active_topic_ =
      "/care_planner/c4_4/probe_active";
  std::string replan_request_topic_ =
      "/care_planner/local_planner/replan_request";
  std::string executed_command_topic_ =
      "/care_arm/arm_group_velocity_controller/command";
  std::string execution_summary_topic_ =
      "/care_planner/execution/tracker_summary";
  std::string cdf_batch_topic_ =
      "/care_planner/local_planner/cdf_constraint_batch";
  std::string query_trajectory_topic_ =
      "/care_planner/local_planner/scp_query_trajectory";
  std::string candidate_trajectory_topic_ =
      "/care_planner/local_planner/candidate_trajectory";
  std::string summary_topic_ =
      "/care_planner/local_planner/summary";
  std::string task_infeasible_topic_ =
      "/care_planner/local_planner/task_infeasible";
  std::string task_uncertified_topic_ =
      "/care_planner/local_planner/task_uncertified";
  std::string force_vbc_bootstrap_topic_ =
      "/care_planner/trajectory_risk/force_bootstrap";

  unsigned long long plan_sequence_ = 0;
  unsigned long long cdf_batch_received_ = 0;
  unsigned long long cdf_stamp_miss_ = 0;
  unsigned long long solve_count_ = 0;
  unsigned long long solve_failure_count_ = 0;
};

}  // namespace egocentric_arm_planner
