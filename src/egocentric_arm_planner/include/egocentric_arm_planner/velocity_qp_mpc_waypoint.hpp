#pragma once

#include <ros/ros.h>

#include <sensor_msgs/JointState.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/String.h>
#include <trajectory_msgs/JointTrajectory.h>

#include <Eigen/Dense>
#include <piqp/piqp.hpp>

#include <limits>
#include <mutex>
#include <string>
#include <vector>

namespace egocentric_arm_planner {

/**
 * CAREPlanner VBC deadline-waypoint controller with deadline-miss recovery.
 *
 * NORMAL:
 *   J = J_task + J_control + J_smooth
 *
 * INTERVENTION (fresh active q_vis before deadline):
 *   J = J_task + J_control + J_smooth
 *       + w_vis ||q_kd - q_vis||^2
 *
 * RECOVERY (deadline missed while target is still unseen):
 *   J = J_control_effort + J_smooth
 *       + w_vis ||q_K - q_vis||^2
 *
 * Task position/terminal tracking and nominal velocity tracking are removed in
 * RECOVERY. Hard position/velocity/acceleration constraints are unchanged.
 * Once the real visibility gate sets waypoint active=false, the controller
 * enters RECOVERY_HOLD (regularization-only deceleration), emits a one-shot
 * recovery-complete event, and waits for a newly replanned nominal reference.
 */
class VelocityQPMPCWaypoint {
public:
  VelocityQPMPCWaypoint() = default;

  bool initialize(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);

private:
  void jointStateCallback(const sensor_msgs::JointStateConstPtr& msg);
  void referenceCallback(const trajectory_msgs::JointTrajectoryConstPtr& msg);
  void waypointActiveCallback(const std_msgs::BoolConstPtr& msg);
  void waypointQCallback(const std_msgs::Float64MultiArrayConstPtr& msg);
  void waypointDeadlineCallback(const std_msgs::Float64ConstPtr& msg);
  void replanReadyCallback(const std_msgs::BoolConstPtr& msg);
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

  Eigen::MatrixXd reconstructPredictedQ(const Eigen::VectorXd& q_current,
                                        const Eigen::VectorXd& u_stack) const;
  void publishVelocity(const Eigen::VectorXd& command);
  void publishSafeStop(const std::string& reason);
  void publishPrediction(const Eigen::MatrixXd& q_pred,
                         const Eigen::VectorXd& u_stack,
                         const std::string& frame_id);
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
  ros::Subscriber replan_ready_sub_;

  ros::Publisher velocity_command_pub_;
  ros::Publisher prediction_pub_;
  ros::Publisher solve_time_pub_;
  ros::Publisher summary_pub_;
  ros::Publisher recovery_active_pub_;
  ros::Publisher recovery_complete_pub_;
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

  bool recovery_enabled_ = true;
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
  std::string recovery_active_topic_ =
      "/care_planner/execution/visibility_recovery_active";
  std::string recovery_complete_topic_ =
      "/care_planner/execution/visibility_recovery_complete";
  std::string replan_ready_topic_ =
      "/care_planner/execution/visibility_replan_ready";

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

  unsigned long long sequence_ = 0;
};

}  // namespace egocentric_arm_planner
