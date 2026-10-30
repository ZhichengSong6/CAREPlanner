#include "planner_diagnostic_access.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

using Planner = egocentric_arm_planner::LocalSparseSCPPlanner;
using namespace egocentric_arm_planner;
static std::string scenario;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(scenario + ": " + #x); } while (false)

static void prepare(Planner& p) {
  p.repair_mode_ = p.plan_repair_mode_ = true;
  p.probe_mode_ = p.plan_probe_mode_ = false;
  p.repair_observation_phase_ = p.plan_repair_observation_phase_ = false;
  p.mode_epoch_ = p.plan_mode_epoch_ = 1;
  p.plan_sequence_ = 100;
  p.plan_running_ = true;
  p.waiting_for_cdf_ = p.plan_requested_ = false;
  p.plan_request_reason_ = "fixture_idle";
  p.pending_batch_.reset();
  p.scp_iteration_ = 1;
  p.repair_no_progress_ = RepairNoProgress{};
  p.final_vbc_no_progress_ = FinalVbcNoProgress{};
  p.safe_frontier_recovery_ = SafeFrontierRecovery{};
  p.repair_candidate_failure_ = RepairCandidateFailure{};
  p.execution_response_monitor_ = ExecutionResponseMonitor{};
  p.candidate_replacement_pending_ = p.candidate_replacement_trigger_pending_ = false;
  p.candidate_replacement_retry_pending_ = false;
  p.plan_candidate_replacement_refresh_only_ = false;
  p.repair_unknown_witnesses_.clear();
  p.repair_witness_overflow_ = p.repair_witness_requires_reobserve_ = false;
  p.latest_vbc_obligation_points_.clear();
  p.latest_vbc_obligation_id_ = -1;
  p.latest_schedule_.clear();
  p.latest_single_waypoint_active_ = p.has_single_waypoint_q_ = true;
  p.latest_single_waypoint_q_ = Eigen::VectorXd::Zero(7);
  p.latest_single_waypoint_joint_mask_ = Eigen::VectorXd::Ones(7);
  p.latest_observation_token_ = p.plan_observation_token_ = "care_obs_v1_1_fixture";
  p.selectRepairTargetLocked();
  p.repair_no_progress_.observe(Eigen::VectorXd::Zero(7));
  p.plan_repair_ticket_ = p.repair_no_progress_.active;
  p.repair_witness_ticket_ = p.plan_repair_ticket_;
  p.repair_witness_mode_epoch_ = 1;
  const auto q = Eigen::MatrixXd::Zero(7, 21).eval();
  const auto u = Eigen::MatrixXd::Zero(7, 20).eval();
  p.has_joint_state_ = p.has_reference_ = true;
  p.latest_joint_state_.name = p.joint_names_;
  p.latest_joint_state_.position = std::vector<double>(7, 0.);
  p.latest_joint_state_.header.stamp = ros::Time::now();
  p.latest_joint_state_received_ = p.latest_joint_state_.header.stamp;
  p.latest_reference_ = p.makeTrajectoryMessage(q, u, "base_link", ros::Time::now());
  p.plan_q_current_ = Eigen::VectorXd::Zero(7);
  p.plan_previous_command_ = Eigen::VectorXd::Zero(7);
  p.plan_q_bar_ = p.plan_previous_q_bar_ = q;
  p.plan_u_bar_ = p.plan_previous_u_bar_ = u;
  p.plan_executable_iterates_ = RepairExecutableIterates{};
  p.plan_executable_iterates_.bind("saved_old_context");
  CHECK(p.plan_executable_iterates_.retain(q, u));
  p.plan_has_hard_solution_ = true;
  p.plan_probe_restore_pending_hard_recheck_ = true;
  p.current_query_stamp_ = ros::Time::now();
  p.last_query_ros_time_ = p.current_query_stamp_;
  p.current_query_wall_ = ros::WallTime::now();
  p.current_witness_request_ = care_collision_cdf::CollisionCDFWitnessRequest{};
  p.current_witness_request_.header.stamp = p.current_query_stamp_;
  p.witness_response_required_ = false;
  auto& prefix = p.latest_prefix_selection_;
  prefix.session = "0123456789abcdef0123456789abcdef";
  prefix.revision = 3;
  prefix.prefix_s = .25;
  prefix.brake_dt_s = p.repair_executable_settings_.brake_dt;
  prefix.brake_steps = p.repair_executable_settings_.max_brake_steps;
  prefix.acceleration.assign(p.repair_executable_settings_.acceleration.data(),
                             p.repair_executable_settings_.acceleration.data() + 7);
  p.plan_prefix_selection_ = prefix;
  CHECK(p.prefixSelectionCompatible(prefix));
}

int main(int argc, char** argv) {
  try {
    ros::init(argc, argv, "query_handoff_test", ros::init_options::NoSigintHandler);
    ros::NodeHandle nh, pnh("~");
    Planner p;
    CHECK(p.initialize(nh, pnh));
    CHECK(p.candidate_replacement_enabled_ && p.safe_frontier_recovery_enabled_);
    CHECK(p.repair_prefix_binding_enabled_ && p.primitive_observation_guard_enabled_);
    p.stopWorker();
    p.timer_.stop();
    // Allow the real TCPROS recorders to attach; no task or actuator exists.
    ros::WallDuration(.2).sleep();
    const auto q = Eigen::MatrixXd::Zero(7, 21).eval();
    const auto u = Eigen::MatrixXd::Zero(7, 20).eval();
    std::ofstream out(argv[1]);
    out << "{\"fixed\":" << (CARE_FIXED ? "true" : "false") << ",\"checks\":[";
    bool first = true;
    auto passed = [&]() {
      if (!first) out << ',';
      first = false;
      out << '"' << scenario << '"';
    };

    scenario = "measured_progress_between_acceptance_and_publication";
    prepare(p);
    const auto ticket = p.plan_repair_ticket_;
    auto measured = boost::make_shared<sensor_msgs::JointState>(p.latest_joint_state_);
    measured->position[0] = .02;
    p.jointStateCallback(measured); // Actual callback, not a counter edit.
    CHECK(!p.repair_no_progress_.current(ticket));
    const auto old_stamp = p.current_query_stamp_;
    CHECK(p.publishQueryTrajectory(q, u, "base_link", 100, 1, &ticket).isZero());
    CHECK(!p.waiting_for_cdf_ && p.scp_iteration_ == 1 && p.max_scp_iterations_ == 6);
    CHECK(p.current_query_stamp_ == old_stamp); // Rejected seed was not armed.
    CHECK(p.plan_running_ == !CARE_FIXED);
    CHECK(p.plan_executable_iterates_.available() == !CARE_FIXED);
    CHECK(p.plan_requested_ == bool(CARE_FIXED)); // Automatic handoff required.
    if (CARE_FIXED) {
      CHECK(!p.plan_has_hard_solution_ && p.plan_q_bar_.size() == 0);
      CHECK(p.current_witness_request_.header.stamp.isZero());
      CHECK(!p.plan_probe_restore_pending_hard_recheck_);
    }
    // The existing external replan request cannot recover the old gap.
    p.requestPlanLocked("fixture_external_replan");
    CHECK(p.plan_requested_);
    CHECK(p.startPlan() == bool(CARE_FIXED));
    if (CARE_FIXED) {
      CHECK(p.plan_sequence_ == 101 && p.waiting_for_cdf_);
      CHECK(p.plan_repair_ticket_.progress == p.repair_no_progress_.active.progress);
      CHECK(p.current_query_stamp_ > old_stamp);
      CHECK(p.plan_q_bar_(0, 0) == .02);
      auto late = boost::make_shared<care_collision_cdf::CollisionCDFConstraintBatch>();
      late->dof = 7;
      late->header.stamp = old_stamp;
      p.cdfConstraintBatchCallback(late);
      CHECK(p.waiting_for_cdf_ && !p.pending_batch_);
      auto fresh = boost::make_shared<care_collision_cdf::CollisionCDFConstraintBatch>();
      fresh->dof = 7;
      fresh->header.stamp = p.current_query_stamp_;
      p.cdfConstraintBatchCallback(fresh);
      CHECK(!p.waiting_for_cdf_ && p.pending_batch_ == fresh);
      CHECK(!p.repair_witness_requires_reobserve_);
    }
    passed();

    for (const auto& name : {"newer_plan_protected", "armed_query_protected", "queued_response_protected", "already_cancelled_protected"}) {
      scenario = name;
      prepare(p);
      auto stale = p.plan_repair_ticket_;
      --stale.progress;
      if (scenario == "armed_query_protected") p.waiting_for_cdf_ = true;
      if (scenario == "queued_response_protected")
        p.pending_batch_ = boost::make_shared<care_collision_cdf::CollisionCDFConstraintBatch>();
      if (scenario == "already_cancelled_protected") p.plan_running_ = false;
      const auto pending = p.pending_batch_;
      const bool running = p.plan_running_, waiting = p.waiting_for_cdf_;
      const auto stamp = p.current_query_stamp_;
      CHECK(p.publishQueryTrajectory(q, u, "base_link",
              scenario == "newer_plan_protected" ? 99 : 100, 1, &stale).isZero());
      CHECK(p.plan_running_ == running && p.waiting_for_cdf_ == waiting);
      CHECK(p.pending_batch_ == pending && p.current_query_stamp_ == stamp);
      CHECK(p.plan_executable_iterates_.available() && !p.plan_requested_);
      CHECK(p.plan_sequence_ == 100 && p.scp_iteration_ == 1);
      passed();
    }

    for (const auto& name : {"repair_failure_budget_hold", "final_vbc_budget_hold", "direction_segment_budget_hold",
                            "replacement_pending_hold", "execution_no_response_hold", "failed_candidate_hold"}) {
      scenario = name;
      prepare(p);
      auto stale = p.plan_repair_ticket_;
      --stale.progress;
      if (scenario == "repair_failure_budget_hold")
        p.repair_no_progress_.counts[p.repair_no_progress_.active.target] = RepairNoProgress::max_failures;
      if (scenario == "final_vbc_budget_hold")
        for (int i = 0; i < 20 && !p.final_vbc_no_progress_.progress.blocked(); ++i)
          p.final_vbc_no_progress_.progress.fail(p.final_vbc_no_progress_.progress.active);
      if (scenario == "direction_segment_budget_hold") {
        p.safe_frontier_recovery_.active = true;
        p.safe_frontier_recovery_.attempts = SafeFrontierRecovery::max_attempts + 1;
        p.safe_frontier_recovery_.segments = SafeFrontierRecovery::max_segments;
      }
      if (scenario == "replacement_pending_hold") p.candidate_replacement_pending_ = true;
      if (scenario == "execution_no_response_hold") p.execution_response_monitor_.blocked = true;
      if (scenario == "failed_candidate_hold")
        p.repair_candidate_failure_.record(p.repairTargetKeyLocked(), p.latest_observation_token_, 1,
                                           p.repair_no_progress_.active.progress);
      const auto counts = p.repair_no_progress_.counts;
      const auto vbc_counts = p.final_vbc_no_progress_.progress.counts;
      const auto attempts = p.safe_frontier_recovery_.attempts, segments = p.safe_frontier_recovery_.segments;
      CHECK(p.publishQueryTrajectory(q, u, "base_link", 100, 1, &stale).isZero());
      CHECK(p.plan_running_ == !CARE_FIXED && !p.waiting_for_cdf_);
      CHECK(!p.plan_requested_ && !p.startPlan());
      CHECK(p.repair_no_progress_.counts == counts && p.final_vbc_no_progress_.progress.counts == vbc_counts);
      CHECK(p.safe_frontier_recovery_.attempts == attempts && p.safe_frontier_recovery_.segments == segments);
      CHECK(p.scp_iteration_ == 1 && p.max_scp_iterations_ == 6);
      passed();
    }

    scenario = "fresh_identity_positive_control";
    prepare(p);
    CHECK(!p.publishQueryTrajectory(q, u, "base_link", 100, 1, &p.plan_repair_ticket_).isZero());
    CHECK(p.plan_running_ && p.waiting_for_cdf_ && !p.plan_requested_);
    CHECK(p.scp_iteration_ == 1 && p.plan_sequence_ == 100);
    passed();
    scenario = "missing_ticket_rejected";
    prepare(p);
    CHECK(p.publishQueryTrajectory(q, u, "base_link", 100, 1, nullptr).isZero());
    CHECK(p.plan_running_ == !CARE_FIXED && !p.waiting_for_cdf_);
    passed();
    scenario = "same_plan_mode_change_rejected";
    prepare(p);
    CHECK(p.publishQueryTrajectory(q, u, "base_link", 100, 2, &p.plan_repair_ticket_).isZero());
    CHECK(p.plan_running_ == !CARE_FIXED && !p.waiting_for_cdf_);
    passed();
    scenario = "persistent_witness_and_margin_preserved";
    prepare(p);
    p.repair_unknown_witnesses_.resize(1);
    p.repair_unknown_witnesses_[0].point = Eigen::Vector3d(.1, .2, .3);
    p.repair_unknown_witnesses_[0].safety_margin = .05;
    p.repair_unknown_witnesses_[0].vbc_feedback = true;
    auto stale = p.plan_repair_ticket_;
    --stale.progress;
    CHECK(p.publishQueryTrajectory(q, u, "base_link", 100, 1, &stale).isZero());
    CHECK(p.repair_unknown_witnesses_.size() == 1 && p.repair_unknown_witnesses_[0].safety_margin == .05);
    CHECK(p.repair_unknown_witnesses_[0].point == Eigen::Vector3d(.1, .2, .3));
    CHECK(p.repair_unknown_witnesses_[0].vbc_feedback);
    passed();
    out << "],\"max_scp_iterations\":" << p.max_scp_iterations_
        << ",\"max_directions\":" << SafeFrontierRecovery::max_attempts
        << ",\"max_segments\":" << SafeFrontierRecovery::max_segments << "}\n";
    ros::WallDuration(.1).sleep();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
