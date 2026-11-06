#pragma once
#include "egocentric_arm_planner/fixed_start_clearance.hpp"
#include "egocentric_arm_planner/known_ground_hold.hpp"

#include <ros/ros.h>

#include <care_collision_cdf/CollisionCDFConstraintBatch.h>
#include <care_collision_cdf/CollisionCDFWitnessRequest.h>
#include <care_collision_cdf/CollisionCDFWitnessResponse.h>
#include <care_collision_cdf/CollisionCDFRejectionFeedback.h>
#include <egocentric_arm_planner/RepairPrefixSelection.h>
#include <egocentric_arm_planner/CandidateTransfer.h>
#include <egocentric_arm_planner/VisibilityGuidance.h>
#include "egocentric_arm_planner/visibility_guidance.hpp"

#include <sensor_msgs/JointState.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/String.h>
#include "egocentric_arm_planner/task_reference_retry.hpp"
#include "egocentric_arm_planner/qp_no_progress.hpp"
#include "egocentric_arm_planner/normal_active_no_progress.hpp"
#include "egocentric_arm_planner/normal_unknown_no_progress.hpp"
#include "egocentric_arm_planner/normal_unknown_task_progress.hpp"
#include "egocentric_arm_planner/qp_backtrack_policy.hpp"
#include "egocentric_arm_planner/repair_no_progress.hpp"
#include "egocentric_arm_planner/final_vbc_no_progress.hpp"
#include "egocentric_arm_planner/vbc_failure_direction_memory.hpp"
#include "egocentric_arm_planner/safe_frontier_recovery.hpp"
#include "egocentric_arm_planner/state_evidence_retry.hpp"
#include "egocentric_arm_planner/execution_response_monitor.hpp"
#include "egocentric_arm_planner/repair_candidate_failure.hpp"
#include "egocentric_arm_planner/repair_conflict_reseed.hpp"
#include "egocentric_arm_planner/primitive_observation_guard.hpp"
#include "egocentric_arm_planner/normal_terminal_retention.hpp"
#include <geometry_msgs/PoseStamped.h>
#include <care_confidence_map/initial_shell_recovery.hpp>
#include "egocentric_arm_planner/repair_executable_view.hpp"
#include "egocentric_arm_planner/probe_executable_task.hpp"
#include "egocentric_arm_planner/repair_executable_iterates.hpp"
#include <trajectory_msgs/JointTrajectory.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <condition_variable>
#include <atomic>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <set>
#include <thread>
#include <vector>

namespace egocentric_arm_planner {

class LocalSparseSCPPlanner {
public:
  LocalSparseSCPPlanner() = default;
  ~LocalSparseSCPPlanner();

  bool initialize(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);

private:
  friend struct KnownGroundFixture;
  struct DeadlineWaypoint {
    long long id = 0;
    double deadline_abs_s = 0.0;
    bool terminal_objective = false;
    Eigen::VectorXd q;
    Eigen::VectorXd joint_mask;
  };

  // Learned visibility-frontier steering is an objective-only hint. The
  // producer supplies a short local target plus relative weights; hard GCDF
  // and exact VBC remain the only executable safety authorities.
  struct FrontierObjective {
    bool active = false;
    double frontier_weight_scale = 0.0;
    double qvis_weight_scale = 1.0;
    Eigen::VectorXd q;
    Eigen::VectorXd joint_mask;
    int recovery_attempt = 0;
    Eigen::VectorXd recovery_target;
    Eigen::VectorXd visibility_gradient;
    std::vector<Eigen::VectorXd> failed_directions;
  };

  struct SparseSolveResult {
    bool solved = false;
    bool repair_unknown_soft = false;
    std::string status = "not_solved";
    int box_conflicting_cdf_rows = 0;
    int reachable_conflicting_cdf_rows = 0;
    int first_step_conflicting_cdf_rows = 0;
    int later_conflicting_cdf_rows = 0;
    int first_step_conflicting_occupied_rows = 0;
    std::vector<Eigen::Vector3d> first_step_unknown_conflict_points;
    std::vector<Eigen::Vector3d> geometric_restore_guard_points;
    std::vector<PrimitiveObservationPoint> geometric_restore_path_guards;
    int primitive_conflicting_rows = 0;
    std::vector<Eigen::Vector3d> primitive_margin_recovery_points;
    std::vector<Eigen::Vector3d> fixed_start_recovery_points;
    int reachable_conflicting_repair_witness_rows = 0;
    int first_step_conflicting_repair_witness_rows = 0;
    int later_conflicting_repair_witness_rows = 0;
    double worst_repair_witness_reachable_deficit = 0.0;
    std::string repair_hard_qp_snapshot;
    int visibility_objective_step = -1;
    unsigned long long trace_plan_sequence = 0;
    std::string trace_observation_token = "none";
  bool trace_repair = false;
    bool trace_repair_observation_phase = true;
    int trace_repair_witnesses = 0;
    bool trace_repair_witnesses_fresh = true;
    bool trace_probe = false;
    RepairNoProgress::Ticket trace_repair_ticket;
    int repair_failures = 0;
    std::string repair_feedback;
    int iterations = 0;
    double primal_residual = 0.0;
    double dual_residual = 0.0;
    double setup_and_solve_ms = 0.0;
    int batch_pairs = 0;
    int selected_cdf_rows = 0;
    int primitive_observation_rows = 0;
    int primitive_observation_points = 0;
    int primitive_executable_rows = 0;
    int primitive_executable_samples = 0;
    bool primitive_executable_valid = true;
    bool normal_curve_checked=false,normal_curve_valid=true,normal_live_checked=false,normal_live_valid=true;
    double normal_curve_violation=0.,normal_live_violation=0.;
    Eigen::VectorXd normal_live_q,normal_live_v;

    bool primitive_live_rebase_checked = false;
    bool primitive_live_rebase_required = false;
    Eigen::VectorXd primitive_live_rebase_start;
    Eigen::VectorXd primitive_live_rebase_velocity;
    double primitive_live_rebase_shift_inf = 0.;
    double primitive_live_rebase_min_m = std::numeric_limits<double>::quiet_NaN();
    double primitive_executable_min_distance_m = std::numeric_limits<double>::quiet_NaN();
    double primitive_executable_numerical_tightening_m = 0.;
    double executable_step_scale = 1.;
    bool executable_recheck_only = false;
    bool executable_trial_restored = false;
    bool executable_retained_available = false;
    double primitive_observation_start_distance_m = std::numeric_limits<double>::quiet_NaN();
    bool fixed_start_unknown_blocked = false;
    Eigen::Vector3d fixed_start_unknown_point = Eigen::Vector3d::Zero();
    double fixed_start_unknown_distance_m = 0., fixed_start_unknown_floor_m = 0.;
    int primitive_shell_recovery_points = 0;
    int primitive_shell_penalty_rows = 0;
    int selected_unknown_cdf_rows = 0;
    int selected_occupied_cdf_rows = 0;
    int selected_obligation_cdf_rows = 0;
    // Exact batch-pair identities that actually entered the hard QP as
    // UNKNOWN rows. Used to route local-GCDF blockers directly to sensing.
    std::vector<int> selected_unknown_pair_indices;
    std::vector<Eigen::Vector3d> normal_blocking_unknown_points;
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
    Eigen::VectorXd recovery_frontier;
    double recovery_start_clearance=0.;
    std::string recovery_clearance_signature;
    int recovery_normals = 0;
    int recovery_tight_normals = 0;
    bool recovery_rejoining = false;
    std::vector<Eigen::Vector3d> observation_dependency_points;
    Eigen::MatrixXd q;
    Eigen::MatrixXd u;
  };

  void jointStateCallback(const sensor_msgs::JointStateConstPtr& msg);
  void repairPrefixSelectionCallback(const egocentric_arm_planner::RepairPrefixSelectionConstPtr& msg);
  bool prefixSelectionCompatible(const egocentric_arm_planner::RepairPrefixSelection& msg) const;
  bool prefixSelectionCurrentLocked(const egocentric_arm_planner::RepairPrefixSelection& msg) const;
  void referenceCallback(
      const trajectory_msgs::JointTrajectoryConstPtr& msg);
  void taskReferenceStatusCallback(const std_msgs::StringConstPtr& msg);
  void waypointScheduleCallback(
      const std_msgs::Float64MultiArrayConstPtr& msg);
  void vbcObligationPointsCallback(
      const std_msgs::Float64MultiArrayConstPtr& msg);
  void visibilityFrontierCallback(
      const std_msgs::Float64MultiArrayConstPtr& msg);
  void visibilityGuidanceCallback(const egocentric_arm_planner::VisibilityGuidanceConstPtr& msg);
  void singleWaypointActiveCallback(const std_msgs::BoolConstPtr& msg);
  void singleWaypointQCallback(
      const std_msgs::Float64MultiArrayConstPtr& msg);
  void recoveryCallback(const std_msgs::BoolConstPtr& msg);
  void probeActiveCallback(const std_msgs::BoolConstPtr& msg);
  void replanRequestCallback(const std_msgs::BoolConstPtr& msg);
  void qpRetryRequestCallback(const std_msgs::StringConstPtr& msg);
  void smoothReplanRequestCallback(const std_msgs::BoolConstPtr& msg);
  void terminalGoalCallback(const geometry_msgs::PoseStampedConstPtr& msg);
  void terminalAuditCallback(const trajectory_msgs::JointTrajectoryConstPtr& msg);
  void terminalCommitCallback(const trajectory_msgs::JointTrajectoryConstPtr& msg);
  std::string terminalContextLocked() const;
  bool terminalPoseInsideLocked(const Eigen::VectorXd& q) const;
  void armTerminalLocked();
  bool retainTerminalLocked();
  void executedCommandCallback(
      const std_msgs::Float64MultiArrayConstPtr& msg);
  void executionSummaryCallback(const std_msgs::StringConstPtr& msg);
  void desiredCommandCallback(const std_msgs::Float64MultiArrayConstPtr& msg);
  void limiterSummaryCallback(const std_msgs::StringConstPtr& msg);
  void vbcExecutionHoldCallback(const std_msgs::BoolConstPtr& msg);
  void gcdfExecutionHoldCallback(const std_msgs::BoolConstPtr& msg);
  void observationProgressCallback(const std_msgs::StringConstPtr& msg);
  void finalVerificationCallback(const std_msgs::StringConstPtr& msg);
  void admissionTerminalCallback(const egocentric_arm_planner::CandidateTransferConstPtr& msg);
  void candidateReplacementCallback(const std_msgs::StringConstPtr& msg);
  void cdfConstraintBatchCallback(
      const care_collision_cdf::CollisionCDFConstraintBatchConstPtr& msg);
  void updateRepairWitnessesLocked(
      const care_collision_cdf::CollisionCDFConstraintBatch& batch);
  void gcdfRejectionFeedbackCallback(
      const care_collision_cdf::CollisionCDFRejectionFeedbackConstPtr& msg);
  void witnessResponseCallback(
      const care_collision_cdf::CollisionCDFWitnessResponseConstPtr& msg);
  void acceptCdfBatch(
      const care_collision_cdf::CollisionCDFConstraintBatchConstPtr& msg,
      const care_collision_cdf::CollisionCDFWitnessResponse* response);
  void publishWitnessDiagnosticLocked(int index, int timestep,
      const Eigen::Vector3d& point, const std::string& reason,
      int pair = -1, double q_error = 0.0);
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
      bool probe_mode,
      Eigen::MatrixXd& q_ref,
      Eigen::MatrixXd& u_ref,
      Eigen::MatrixXd& q_init,
      Eigen::MatrixXd& u_init) const;

  void requestPlanLocked(const std::string& reason);
  std::string repairTargetKeyLocked() const;
  void selectRepairTargetLocked();
  VbcFailureDirectionMemory::Context vbcFailureContextLocked() const;
  bool startPlan();
  void abortPlan(const std::string& reason);

  void startWorker();
  void stopWorker();
  void workerLoop();

  struct ViewPreviewSnapshot;
  struct ViewLocalContext;
  void alignedViewPreview(const std_msgs::StringConstPtr& message);
  bool view_preview_aligned_enabled_=false;
  std::mutex view_preview_request_mutex_;
  std::shared_ptr<const ViewLocalContext> view_local_context_;
  std::shared_ptr<const ViewLocalContext> view_pending_context_;
  std::string view_pending_id_;
  ros::WallTime view_pending_started_;
  void viewPreviewCallback(const std_msgs::StringConstPtr& msg);
  ros::Subscriber view_preview_sub_;
  ros::Publisher view_preview_pub_;
  care_collision_cdf::CollisionCDFConstraintBatchConstPtr view_preview_batch_;
  ros::Time view_preview_batch_received_;

  SparseSolveResult solveSparseSubproblem(
      const care_collision_cdf::CollisionCDFConstraintBatch& batch,
      const Eigen::MatrixXd& q_bar,
      const Eigen::MatrixXd& u_bar,
      const Eigen::MatrixXd& q_ref,
      const Eigen::MatrixXd& u_ref,
      const Eigen::VectorXd& previous_command,
      const std::vector<DeadlineWaypoint>& schedule,
      const FrontierObjective& frontier,
      const std::vector<Eigen::Vector3d>& obligation_points,
      bool repair_mode,
      bool probe_mode,
      double trust_radius,
      double slack_linear_weight,
      bool force_diagnostic_slack = false,
      bool previous_vbc_margins = false,
      double executable_step_scale = 1.,
      bool executable_recheck_only = false,
      const egocentric_arm_planner::RepairPrefixSelection* selected_prefix = nullptr,
      bool free_joint_task = false,
      const ViewPreviewSnapshot* preview = nullptr, ViewPreviewSnapshot* capture = nullptr) const;
  bool freeJointTaskImproves(const SparseSolveResult& primary,
      const SparseSolveResult& trial, const std::vector<DeadlineWaypoint>& schedule,
      const FrontierObjective& frontier, const Eigen::MatrixXd& q_ref) const;

  std::string repairExecutableContextLocked() const;

  ros::Time publishQueryTrajectory(
      const Eigen::MatrixXd& q,
      const Eigen::MatrixXd& u,
      const std::string& frame_id,
      unsigned long long expected_plan_sequence = 0,
      unsigned long long expected_mode_epoch = 0,
      const RepairNoProgress::Ticket* expected_repair_ticket = nullptr);
  void publishCandidateTrajectory(
      const Eigen::MatrixXd& q,
      const Eigen::MatrixXd& u,
      const std::string& frame_id,
      const std::string& observation_token,
      const care_collision_cdf::CollisionCDFConstraintBatch& batch,
      const Eigen::MatrixXd& query_q, const Eigen::MatrixXd& query_u,
      const std::vector<Eigen::Vector3d>& normal_blocking_unknown_points,
      const std::vector<Eigen::Vector3d>& geometric_restore_points = {},
      bool margin_recovery = false,
      const std::vector<Eigen::Vector3d>& fixed_start_recovery_points = {});
  bool publishLocalGcdfRecoveryEvidence(
      const care_collision_cdf::CollisionCDFConstraintBatch& batch,
      const SparseSolveResult& result,
      const Eigen::MatrixXd& q_bar,
      const Eigen::MatrixXd& u_bar,
      const std::string& frame_id);
  trajectory_msgs::JointTrajectory makeTrajectoryMessage(
      const Eigen::MatrixXd& q,
      const Eigen::MatrixXd& u,
      const std::string& frame_id,
      const ros::Time& stamp,
      const Eigen::VectorXd* normal_initial_velocity = nullptr) const;
  bool buildNormalMotionSeed(const Eigen::VectorXd& q_current,
      const Eigen::VectorXd& v_current, const Eigen::MatrixXd& q_ref,
      Eigen::MatrixXd& q, Eigen::MatrixXd& u) const;
  std::string repairWarmStartKeyLocked() const;
  bool takeRepairWarmStartLocked(const Eigen::VectorXd& q_current,
      const Eigen::VectorXd& v_current, Eigen::MatrixXd& q,
      Eigen::MatrixXd& u, int& charged_iterations);

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
  ros::Subscriber task_reference_status_sub_;
  ros::Subscriber waypoint_schedule_sub_;
  ros::Subscriber vbc_obligation_points_sub_;
  ros::Subscriber visibility_frontier_sub_;
  ros::Subscriber single_waypoint_active_sub_;
  ros::Subscriber single_waypoint_q_sub_;
  ros::Subscriber recovery_sub_;
  ros::Subscriber probe_active_sub_;
  ros::Subscriber replan_request_sub_;
  ros::Subscriber smooth_replan_request_sub_;
  ros::Subscriber terminal_goal_sub_,terminal_audit_sub_,terminal_commit_sub_;
  bool terminal_retention_enabled_=false,terminal_goal_valid_=false;
  bool repair_goal_stopping_enabled_=false;
  unsigned long long repair_goal_handoff_execution_=0;
  NormalTerminalRetention terminal_retention_;
  std::string plan_terminal_context_;
  Eigen::Isometry3d terminal_goal_=Eigen::Isometry3d::Identity();
  std::string terminal_base_frame_="base_link",terminal_ee_frame_="EE_link";
  unsigned long long terminal_goal_revision_=0,terminal_tracker_token_=0;
  ros::Time terminal_tracker_received_;
  std::string terminal_tracker_source_;
  bool terminal_tracker_completed_=false,terminal_tracker_aborted_=false;
  ros::Subscriber executed_command_sub_;
  ros::Subscriber execution_summary_sub_;
  ros::Subscriber desired_command_sub_, limiter_summary_sub_;
  ros::Subscriber vbc_execution_hold_sub_, gcdf_execution_hold_sub_;
  ros::Subscriber observation_progress_sub_;
  ros::Subscriber cdf_batch_sub_;

  ros::Publisher query_trajectory_pub_;
  ros::Publisher witness_request_pub_;
  ros::Publisher witness_diagnostic_pub_;
  ros::Subscriber witness_response_sub_;
  care_collision_cdf::CollisionCDFWitnessRequest current_witness_request_;
  bool witness_response_required_ = false;
  ros::Publisher candidate_trajectory_pub_;
  ros::Publisher observation_identity_pub_;
  ros::Publisher observation_dependency_pub_;
  ros::Subscriber candidate_replacement_sub_;
  ros::Publisher candidate_replacement_grant_pub_;
  ros::Publisher candidate_replacement_trigger_pub_;
  std::string handleHandoffResidualLocked(unsigned long long raw,unsigned long long audit,
      const HandoffResidualFailure& failure);
  bool requestQpReplacementLocked(const std::string& reason,
      unsigned long long rejected_raw = 0, unsigned long long rejected_audit = 0);
  bool qpReplacementRetryCurrentLocked() const;
  std::string finishReplacementLocked(bool refresh_failed_target, const std::string& reason,
                                      bool unapplied_resource = false);
  RepairCandidateFailure repair_candidate_failure_;
  FixedStartClearanceHold fixed_start_clearance_hold_;
  KnownGroundHold known_ground_hold_;
  unsigned long long ground_retry_goal_=0;
  unsigned ground_resource_retries_=0;
  bool task_first_research_=false;
  bool executable_task_research_=false;
  bool ground_resource_blocked_=false;
  std::string geometry_resource_blocked_reason_="ground_proof_unavailable_hold";
  std::string groundContextLocked() const;
  void checkKnownGroundHoldLocked();
  struct GroundAdmission {
    CandidateTransfer planned;
    unsigned long long goal=0,epoch=0;
    bool repair=false,probe=false;
    std::string context;
    ros::WallTime deadline;
  };
  std::map<unsigned long long,GroundAdmission> ground_admission_;

  RepairCandidateFailure candidate_replacement_reserved_failure_;
  RepairCandidateFailure candidate_replacement_retry_context_;
  bool candidate_replacement_reserved_qp_ = false;
  bool candidate_replacement_retry_pending_ = false;
  bool plan_candidate_replacement_refresh_only_ = false;
  std::string candidate_replacement_trigger_reason_ = "final_vbc_repeat";
  std::string candidate_replacement_trigger_observation_token_;
  bool candidate_replacement_enabled_ = false;
  bool candidate_replacement_execution_v2_ = false;
  SafeFrontierRecovery::ReplacementCharge candidate_replacement_charge_;
  bool candidate_replacement_pending_ = false;
  bool candidate_replacement_trigger_pending_ = false;
  bool candidate_replacement_trigger_direction_advanced_ = false;
  unsigned long long candidate_replacement_id_ = 0;
  unsigned long long candidate_replacement_last_id_ = 0;
  unsigned long long candidate_replacement_epoch_ = 0;
  unsigned long long candidate_replacement_trigger_raw_ = 0;
  unsigned long long candidate_replacement_trigger_count_ = 0;
  long long candidate_replacement_trigger_obligation_id_ = -1;
  ros::Time candidate_replacement_deadline_;
  ros::WallTime candidate_replacement_wall_deadline_;
  ros::Time candidate_replacement_trigger_deadline_;
  ros::WallTime candidate_replacement_trigger_wall_deadline_;
  std::string candidate_replacement_new_token_;
  std::string final_vbc_repeat_target_key_ = "none";
  std::vector<Eigen::Vector3d> final_vbc_previous_points_;
  std::vector<Eigen::Vector3d> final_gcdf_previous_points_;
  VbcFailureDirectionMemory vbc_failure_direction_memory_;
  ros::Publisher summary_pub_;
  ros::Publisher task_infeasible_pub_;
  ros::Publisher task_obstacle_blocked_pub_;
  ros::Publisher task_uncertified_pub_;
  ros::Publisher task_stall_pub_;
  ros::Subscriber qp_retry_request_sub_;
  ros::Publisher qp_retry_ack_pub_;
  std::string qp_retry_token_;
  std::string qp_retry_consumed_token_;
  unsigned long long qp_retry_plan_sequence_ = 0;
  unsigned long long qp_retry_mode_epoch_ = 0;
  QPNoProgress task_no_progress_;
  NormalActiveNoProgress normal_active_no_progress_;
  NormalUnknownNoProgress normal_unknown_no_progress_;
  NormalUnknownTaskProgress normal_unknown_task_progress_;
  unsigned long long normal_task_progress_logged_sequence_ = 0;
  bool normal_task_stall_pending_ = false;
  std::string normal_task_stall_context_;
  double normal_task_stall_credited_error_ = -1.;
  std::string normalUnknownContextLocked() const;
  std::string normalTaskProgressContextLocked() const;
  void logNormalTaskProgressLocked();
  void resetNormalTaskProgressLocked(const std::string& reason);
  // Count measured progress after distinct certified NORMAL executions too.
  // A hard QP that recovers to a stationary hold must not reset liveness.
  double normal_no_progress_target_gap_inf_ = 0.10;
  unsigned long long normal_measured_stall_count_ = 0;
  RepairNoProgress repair_no_progress_;
  FinalVbcNoProgress final_vbc_no_progress_;
  ros::Subscriber final_verification_sub_;
  ros::Subscriber admission_terminal_sub_;
  struct AdmissionCandidate {
    egocentric_arm_planner::CandidateTransfer planned;
    egocentric_arm_planner::RepairPrefixSelection prefix;
    unsigned long long epoch;
    std::string observation_token, recovery_key;
  };
  std::map<unsigned long long, AdmissionCandidate> admission_pending_;
  ros::Subscriber gcdf_rejection_feedback_sub_;
  struct GcdfCandidate {
    FinalVbcNoProgress::Candidate identity;
    std::string observation_token;
    ros::Time raw_stamp;
    ros::WallTime deadline;
  };
  std::map<std::uint64_t, GcdfCandidate> gcdf_feedback_pending_;
  ros::Publisher candidate_audit_context_pub_;
  bool rejection_snapshots_enabled_ = false;
  RepairNoProgress::Ticket plan_repair_ticket_;
  ros::Publisher force_vbc_bootstrap_pub_;
  ros::Publisher gcdf_recovery_trajectory_pub_;
  ros::Publisher gcdf_recovery_event_pub_;

  ros::Timer timer_;

  mutable std::mutex mutex_;
  sensor_msgs::JointState latest_joint_state_;
  trajectory_msgs::JointTrajectory latest_reference_;
  std::vector<DeadlineWaypoint> latest_schedule_;
  // Current blocker-aware visibility target. This is deliberately separate
  // from local GCDF witnesses: queued obligations must not preempt the active
  // target. Its points receive only the configured near-term CDF guard;
  // confirmed GCDF witnesses are carried separately in repair_unknown_witnesses_.
  std::vector<Eigen::Vector3d> latest_vbc_obligation_points_;
  unsigned long long latest_vbc_obligation_points_seq_ = 0;
  long long latest_vbc_obligation_id_ = -1;
  FrontierObjective latest_frontier_;
  ros::Subscriber visibility_guidance_sub_;
  bool rolling_visibility_guidance_enabled_=false;
  VisibilityGuidanceState latest_visibility_guidance_;
  std::string latest_observation_region_;
  int latest_observation_sensor_=-1;
  bool safe_frontier_recovery_enabled_ = false;
  SafeFrontierRecovery safe_frontier_recovery_;
  std::map<std::string,bool> state_evidence_retry_used_;
  std::string state_evidence_retry_pending_;
  ros::WallTime state_evidence_retry_deadline_;
  StateEvidenceRetryWait state_evidence_retry_wait_;
  unsigned long long state_residual_retry_raw_=0,state_residual_retry_audit_=0,state_residual_retry_epoch_=0;
  std::string state_residual_retry_token_;
  HandoffResidualFailure state_residual_retry_failure_;
  std::string geometric_restore_scope_;
  std::set<std::string> geometric_restore_snapshot_scopes_;
  bool buildGeometricRestoreProposalLocked(const SparseSolveResult& failure,
      const std::string& observation_token, double prefix_s,
      Eigen::MatrixXd* q, Eigen::MatrixXd* u);
  ExecutionResponseMonitor execution_response_monitor_;
  Eigen::VectorXd latest_desired_command_;
  ros::Time latest_desired_command_received_, latest_limiter_summary_received_;
  bool latest_limiter_intentional_hold_ = false;
  bool vbc_execution_hold_ = false, gcdf_execution_hold_ = false;
  std::string safe_frontier_margin_geometry_key_;
  bool latest_single_waypoint_active_ = false;
  bool has_single_waypoint_q_ = false;
  Eigen::VectorXd latest_single_waypoint_q_;
  Eigen::VectorXd latest_single_waypoint_joint_mask_;
  std::string latest_observation_token_ = "none";
  bool latest_observation_feedback_unseen_ = false;
  std::string plan_observation_token_ = "none";
  Eigen::VectorXd latest_executed_command_;
  ros::Time latest_joint_state_received_;
  ros::Time latest_reference_received_;
  ros::Time latest_executed_command_received_;
  bool latest_execution_complete_ = false;
  // C5.32 REPAIR liveness: identify tracker completions by immutable committed
  // execution stamp rather than relying on a potentially dropped complete=0
  // sample between two fast executions.
  unsigned long long latest_execution_stamp_ns_ = 0;
  unsigned long long last_stopped_replan_stamp_ns_ = 0, stopped_execution_replan_count_ = 0;
  unsigned long long last_repair_completed_execution_stamp_ns_ = 0;
  unsigned long long repair_completion_replan_count_ = 0;
  unsigned long long repair_completion_duplicate_count_ = 0;
  // C5.33 NORMAL liveness/reference refresh. The first NORMAL plan after
  // PROBE waits for a fresh measured-state /task_trajectory; subsequent
  // certified executions replan exactly once per execution stamp.
  unsigned long long last_normal_completed_execution_stamp_ns_ = 0;
  unsigned long long normal_completion_replan_count_ = 0;
  unsigned long long normal_completion_duplicate_count_ = 0;
  unsigned long long normal_reference_refresh_count_ = 0;
  unsigned long long normal_refresh_blocked_replan_count_ = 0;
  bool normal_reference_refresh_pending_ = false;
  TaskReferenceReceipt task_reference_receipt_;
  unsigned long long probe_reference_request_id_ = 0;

  // Smooth handoff requests are advisory and mode-aware. PROBE remains owned
  // by the verified single-flight state machine.
  unsigned long long smooth_handoff_replan_count_ = 0;
  unsigned long long smooth_handoff_replan_suppressed_probe_count_ = 0;
  unsigned long long smooth_handoff_replan_suppressed_busy_count_ = 0;
  unsigned long long mode_epoch_ = 0;
  unsigned long long plan_mode_epoch_ = 0;
  unsigned long long stale_mode_candidate_discard_count_ = 0;

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
  ros::Time last_query_ros_time_;
  unsigned long long query_stamp_adjustments_ = 0;
  ros::WallTime current_query_wall_;
  ros::WallTime current_plan_start_wall_;
  // C5.31 latency accounting for the event-driven trajectory->GCDF roundtrip.
  double last_cdf_roundtrip_ms_ = 0.0;
  double plan_cdf_roundtrip_sum_ms_ = 0.0;
  double plan_cdf_roundtrip_max_ms_ = 0.0;
  int plan_cdf_roundtrip_count_ = 0;
  std::string current_frame_id_ = "base_link";

  Eigen::VectorXd plan_q_current_;
  Eigen::VectorXd plan_previous_command_;
  Eigen::MatrixXd plan_q_ref_;
  Eigen::MatrixXd plan_u_ref_;
  Eigen::MatrixXd plan_q_bar_;
  // One-use shape only. Fresh q/v, query, executable and audit are required.
  Eigen::MatrixXd repair_warm_q_;
  std::string repair_warm_key_;
  ros::Time repair_warm_stamp_;
  int repair_warm_iterations_ = 0;
  // The last accepted hard-SCP center is retained so a CDF point-set change
  // that makes the next hard QP fail can be retried once from a known center.
  Eigen::MatrixXd plan_previous_q_bar_;
  Eigen::MatrixXd plan_previous_u_bar_;
  bool plan_has_hard_solution_ = false;
  QPBacktrackPolicy plan_qp_backtrack_policy_;
  bool plan_normal_reseed_used_ = false;
  RepairConflictReseed plan_repair_conflict_reseed_;
  RepairExecutableIterates plan_executable_iterates_;
  Eigen::MatrixXd plan_normal_retained_q_,plan_normal_retained_u_;
  std::atomic<int> repair_hard_qp_snapshots_written_{0};
  // One replayable failed hard QP before and after a live obligation appears.
  mutable std::atomic<bool> normal_qp_snapshot_unowned_written_{false};
  mutable std::atomic<bool> normal_qp_snapshot_owned_written_{false};
  Eigen::MatrixXd plan_u_bar_;
  std::vector<DeadlineWaypoint> plan_schedule_;
  FrontierObjective plan_frontier_;
  bool plan_repair_mode_ = false;
  // REPAIR alternates between safe retreat and visibility observation. A
  // retreat never carries a q_vis objective back into the hard GCDF solve.
  bool repair_observation_phase_ = true;
  bool plan_repair_observation_phase_ = true;
  struct RepairWitness {
    Eigen::Vector3d point;
    // Diagnostic anchor only. Persistent identity is point-scoped; every
    // current trajectory knot is re-queried at its own q linearization.
    int timestep;
    // The guard is always expressed in native CDF units. A point's VBC
    // rejection may advance this local CDF ladder, but cannot derive its
    // value from the VBC's meter-valued result.
    double safety_margin = 0.0;
    // VBC feedback uses the same point identity, with its independently
    // elevated native-CDF margin retained across fresh queries.
    bool vbc_feedback = false;
    double previous_margin = -1.;
    double margin_ceiling = 0.30;
    bool has_primitive_guard = false;
    PrimitiveObservationPoint primitive_guard;
  };
  unsigned long long native_snapshot_probe_seq_=0,native_snapshot_normal_seq_=0;
  int native_snapshot_count_=0;
  bool native_snapshot_observation_seen_=false;
  struct ViewPreviewSnapshot {
    bool native_task_replay=false,normal_live_fresh=false,primitive_live_fresh=false;
    Eigen::VectorXd normal_live_q,normal_live_v,primitive_live_q,primitive_live_v;
    std::string native_matrix_json;
    std::vector<RepairWitness> witnesses;
    std::vector<PrimitiveObservationPoint> path_guards;
    std::vector<Eigen::Vector3d> points;
    Eigen::Isometry3d goal = Eigen::Isometry3d::Identity();
    bool goal_valid = false;
    care_confidence_map::InitialShellRecovery shell;
    std::shared_ptr<PrimitiveObservationGuard> guard;
  };
  struct ViewLocalContext {
    care_collision_cdf::CollisionCDFConstraintBatch batch;
    Eigen::MatrixXd q_bar,u_bar,q_ref,u_ref;
    Eigen::VectorXd previous,measured;
    std::vector<Eigen::Vector3d> obligation_points;
    RepairPrefixSelection prefix;
    ViewPreviewSnapshot snapshot;
    double trust=0.,slack=0.,step_scale=1.;
    int iteration=0;
    bool repair=false;
    unsigned long long epoch=0,plan_sequence=0,goal_revision=0;
    std::string owner,region;
    ros::Time measured_stamp;
    ros::WallTime wall;
  };
  std::vector<RepairWitness> repair_unknown_witnesses_;
  bool primitive_observation_guard_enabled_ = false;
  bool unknown_corridor_guard_enabled_ = true;
  double primitive_observation_margin_m_ = 0.;
  PrimitiveObservationGuard primitive_observation_guard_;
  std::string restore_path_guard_scope_;
  std::vector<PrimitiveObservationPoint> restore_path_guard_points_;
  bool initial_shell_recovery_enabled_ = false;
  bool fixed_start_recovery_enabled_ = false;
  double initial_shell_execution_margin_m_ = 0.;
  mutable care_confidence_map::InitialShellRecovery initial_shell_recovery_;
  bool repair_executable_view_enabled_ = false;
  bool probe_task_objective_enabled_ = false;
  ProbeTaskSettings probe_task_settings_;
  RepairExecutableSettings repair_executable_settings_;
  bool repair_prefix_binding_enabled_ = false;
  ros::Subscriber repair_prefix_selection_sub_;
  ros::Publisher planned_candidate_pub_;
  egocentric_arm_planner::RepairPrefixSelection latest_prefix_selection_, plan_prefix_selection_;
  std::set<std::string> closed_prefix_sessions_;
  bool repair_witness_requires_reobserve_ = false;
  bool repair_witness_overflow_ = false;
  RepairNoProgress::Ticket repair_witness_ticket_;
  unsigned long long repair_witness_mode_epoch_ = 0;
  bool plan_probe_mode_ = false;
  std::string plan_initialization_mode_ = "task_reference";
  // C5.25: a failed PROBE hard-QP may use one soft solve only as an
  // internal feasibility-restoration iterate. The soft trajectory is never
  // executable and must be followed by a fresh GCDF query + hard solve.
  int plan_probe_feasibility_restore_attempts_ = 0;
  bool plan_probe_restore_pending_hard_recheck_ = false;

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
  // Separate boundary-only penalty for certified trajectory replacement:
  // ||u0-u_executed||^2. This remains active even when intra-horizon
  // u_smooth_weight is zero.
  double handoff_velocity_weight_ = 0.0;
  // CARE execution normally enforces hard acceleration and terminal-braking
  // rows. G0 disables them to match the GCDF formulation more closely.
  bool enforce_acceleration_constraints_ = true;
  // In REPAIR, initialize SCP from a measured hold trajectory instead of the
  // known-infeasible task interpolation. The visibility objective then moves
  // outward only as far as hard GCDF safety permits.
  bool repair_hold_initialization_enabled_ = false;
  double repair_task_tracking_scale_ = 0.0;
  double repair_free_joint_task_tracking_scale_ = 0.0;
  // C5.24: PROBE advances only a short task-reference horizon and holds the
  // remaining tail at that local target. This is deliberately separate from
  // the GCDF constraint horizon even when both are configured to five steps.
  int probe_task_horizon_steps_ = 5;
  double probe_velocity_cap_ = 0.5;
  // C5.25 PROBE-only hard-GCDF feasibility restoration. A soft-QP solution
  // may seed the next SCP linearization, but can never be published.
  bool probe_feasibility_restoration_enabled_ = false;
  int probe_feasibility_restoration_max_attempts_ = 1;
  double visibility_waypoint_weight_ = 3000.0;
  // The frontier target acts on an early local state while q_vis remains the
  // long-range visibility attractor.
  int visibility_frontier_horizon_step_ = 2;

  double cdf_safety_margin_ = 0.0;
  // Active visibility targets use a margin in the learned CDF output units.
  // This is deliberately separate from the primitive VBC margin in meters;
  // no meter-to-CDF/radian conversion is attempted. Ordinary CDF rows keep
  // cdf_safety_margin_ unchanged. The numeric value has no physical-clearance
  // interpretation without independent CDF-side calibration.
  double visibility_obligation_cdf_margin_ = 0.05;
  // VBC increments only the rejected point's RepairWitness::safety_margin.
  // The shared observation margin remains the configured base. All values
  // are native CDF units, never converted from VBC meter-valued distances.
  double visibility_obligation_cdf_margin_base_ = 0.05;
  std::atomic<double> visibility_obligation_cdf_margin_effective_{0.05};
  double vbc_feedback_cdf_margin_step_ = 0.15;
  double vbc_feedback_cdf_margin_max_ = 0.30;
  std::string vbc_feedback_target_key_ = "none";
  int vbc_feedback_rejection_count_ = 0;
  int vbc_feedback_matched_point_count_ = 0;
  int vbc_feedback_repair_witness_count_ = 0;
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
  // An active visibility target is not a confirmed collision witness. Keep its
  // identity over the configured receding-horizon guard while the visibility
  // objective moves toward q_vis. Confirmed GCDF repair witnesses retain their
  // existing hard horizon.
  int visibility_obligation_cdf_horizon_steps_ = 2;

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
  std::string vbc_obligation_points_topic_ =
      "/care_planner/active_sensing/current_visibility_obligation_points";
  std::string visibility_frontier_topic_ =
      "/care_planner/active_sensing/visibility_frontier_target";
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
  std::string smooth_replan_request_topic_ =
      "/care_planner/local_planner/smooth_handoff_replan_request";
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
  std::string task_obstacle_blocked_topic_ =
      "/care_planner/local_planner/task_obstacle_blocked";
  std::string task_uncertified_topic_ =
      "/care_planner/local_planner/task_uncertified";
  std::string force_vbc_bootstrap_topic_ =
      "/care_planner/trajectory_risk/force_bootstrap";
  // Shared GCDF->sensing evidence channel. Final-GCDF already publishes here;
  // local hard-GCDF uses the same wire format so blocker-aware acquisition can
  // consume either source without re-running a smaller VBC geometry filter.
  std::string gcdf_recovery_trajectory_topic_ =
      "/care_planner/final_gcdf/recovery_trajectory";
  std::string gcdf_recovery_event_topic_ =
      "/care_planner/final_gcdf/recovery_visibility_event";

  unsigned long long plan_sequence_ = 0;
  unsigned long long cdf_batch_received_ = 0;
  unsigned long long cdf_stamp_miss_ = 0;
  unsigned long long solve_count_ = 0;
  unsigned long long solve_failure_count_ = 0;
  unsigned long long normal_qp_backtrack_count_ = 0;
  unsigned long long probe_feasibility_restore_count_ = 0;
  unsigned long long probe_feasibility_restore_success_count_ = 0;
  unsigned long long local_gcdf_recovery_event_count_ = 0;
  unsigned long long local_gcdf_recovery_drop_count_ = 0;
};

}  // namespace egocentric_arm_planner
