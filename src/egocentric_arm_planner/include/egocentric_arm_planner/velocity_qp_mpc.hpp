#pragma once

#include <ros/ros.h>

#include <sensor_msgs/JointState.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/String.h>
#include <trajectory_msgs/JointTrajectory.h>

#include <Eigen/Dense>
#include <piqp/piqp.hpp>

#include <mutex>
#include <string>
#include <vector>

namespace egocentric_arm_planner {

/**
 * Phase-A joint-space velocity QP-MPC.
 *
 * The nominal quintic is a moving task reference only. The controller uses the
 * latest measured q as the closed-loop state, solves a condensed convex QP over
 * stacked joint velocity commands, and publishes only the first command.
 *
 * Phase A intentionally contains no NCDF/CDF terms. The PIQP problem is kept in
 * the generic form
 *
 *   min 0.5 u' H u + g' u
 *   s.t. h_l <= G u <= h_u
 *        x_l <= u <= x_u
 *
 * so later visibility gradients can alter the linear objective and Yiming CDF
 * linearizations can add collision-avoidance rows without changing the
 * controller/executor boundary.
 */
class VelocityQPMPC {
public:
  VelocityQPMPC() = default;

  bool initialize(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);

private:
  void jointStateCallback(const sensor_msgs::JointStateConstPtr& msg);
  void referenceCallback(const trajectory_msgs::JointTrajectoryConstPtr& msg);
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

  void buildCycleQP(const Eigen::VectorXd& q_current,
                    const Eigen::MatrixXd& q_ref,
                    const Eigen::MatrixXd& u_ref,
                    Eigen::VectorXd& gradient,
                    Eigen::VectorXd& lower,
                    Eigen::VectorXd& upper) const;

  bool solveWithPIQP(const Eigen::VectorXd& gradient,
                     const Eigen::VectorXd& lower,
                     const Eigen::VectorXd& upper,
                     Eigen::VectorXd& solution,
                     int& iterations,
                     double& primal_residual,
                     double& dual_residual,
                     std::string& status_string);

  Eigen::MatrixXd reconstructPredictedQ(const Eigen::VectorXd& q_current,
                                        const Eigen::VectorXd& u_stack) const;

  void publishVelocity(const Eigen::VectorXd& command);
  void publishSafeStop(const std::string& reason);
  void publishPrediction(const Eigen::MatrixXd& q_pred,
                         const Eigen::VectorXd& u_stack,
                         const std::string& frame_id);

private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  ros::Subscriber joint_state_sub_;
  ros::Subscriber reference_sub_;
  ros::Publisher velocity_command_pub_;
  ros::Publisher prediction_pub_;
  ros::Publisher solve_time_pub_;
  ros::Publisher summary_pub_;
  ros::Timer timer_;

  mutable std::mutex data_mutex_;
  sensor_msgs::JointState latest_joint_state_;
  trajectory_msgs::JointTrajectory latest_reference_;
  ros::Time latest_joint_state_received_;
  ros::Time latest_reference_received_;
  bool has_joint_state_ = false;
  bool has_reference_ = false;

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

  Eigen::VectorXd q_tracking_trust_;
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
  bool piqp_setup_done_ = false;

  std::string robot_description_param_ = "/robot_description";
  std::string joint_state_topic_ = "/care_arm/joint_states";
  std::string reference_topic_ = "/care_planner/command_trajectory_candidate";
  std::string velocity_command_topic_ =
      "/care_arm/arm_group_velocity_controller/command";
  std::string prediction_topic_ = "/care_planner/mpc/predicted_trajectory";

  // Condensed MPC matrices. u_stack=[u0;...;uK-1]. S maps u_stack to
  // [q1-q0;...;qK-q0], and D maps it to
  // [u0; u1-u0; ...; uK-1-uK-2].
  Eigen::MatrixXd S_;
  Eigen::MatrixXd S_terminal_;
  Eigen::MatrixXd D_;
  Eigen::MatrixXd G_;
  Eigen::MatrixXd H_;
  Eigen::VectorXd x_lower_;
  Eigen::VectorXd x_upper_;

  int n_u_ = 0;
  int n_constraints_ = 0;
  int acceleration_row0_ = 0;
  int position_row0_ = 0;
  int trust_row0_ = 0;

  piqp::DenseSolver<double> piqp_solver_;

  unsigned long long sequence_ = 0;
};

}  // namespace egocentric_arm_planner
