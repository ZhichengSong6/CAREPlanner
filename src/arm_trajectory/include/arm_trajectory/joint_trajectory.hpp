#pragma once

#include <Eigen/Dense>

#include <string>
#include <vector>
#include <stdexcept>
#include <limits>

namespace arm_trajectory {

class JointTrajectory {
public:
  JointTrajectory() = default;
  explicit JointTrajectory(int dof);

  void setDof(int dof);

  int dof() const;
  bool empty() const;
  std::size_t size() const;

  double startTime() const;
  double endTime() const;
  double duration() const;

  void clear();

  bool addPoint(double t,
                const Eigen::VectorXd& q,
                const Eigen::VectorXd& dq,
                const Eigen::VectorXd& ddq);

  bool sample(double t,
              Eigen::VectorXd& q,
              Eigen::VectorXd& dq,
              Eigen::VectorXd& ddq) const;

  JointTrajectory truncate(double t0, double t1) const;

  bool append(const JointTrajectory& other,
              double time_gap = 0.0,
              bool skip_duplicate_boundary = true);

  const std::vector<double>& times() const;
  const std::vector<Eigen::VectorXd>& positions() const;
  const std::vector<Eigen::VectorXd>& velocities() const;
  const std::vector<Eigen::VectorXd>& accelerations() const;

  static JointTrajectory makeHold(const Eigen::VectorXd& q,
                                  double duration,
                                  double dt);

  static JointTrajectory makeLinear(const Eigen::VectorXd& q0,
                                    const Eigen::VectorXd& q1,
                                    double duration,
                                    double dt);

  static JointTrajectory makeQuinticZeroVelocityAcceleration(
      const Eigen::VectorXd& q0,
      const Eigen::VectorXd& q1,
      double duration,
      double dt);

private:
  bool checkVectorSize(const Eigen::VectorXd& q,
                       const Eigen::VectorXd& dq,
                       const Eigen::VectorXd& ddq) const;

  int dof_ = 0;

  std::vector<double> times_;
  std::vector<Eigen::VectorXd> q_;
  std::vector<Eigen::VectorXd> dq_;
  std::vector<Eigen::VectorXd> ddq_;
};

}  // namespace arm_trajectory