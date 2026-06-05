#include "arm_trajectory/joint_trajectory.hpp"

#include <algorithm>
#include <cmath>

namespace arm_trajectory {

JointTrajectory::JointTrajectory(int dof) {
  setDof(dof);
}

void JointTrajectory::setDof(int dof) {
  if (dof <= 0) {
    throw std::runtime_error("[JointTrajectory] dof must be positive.");
  }

  if (!empty() && dof != dof_) {
    throw std::runtime_error("[JointTrajectory] cannot change dof after points are added.");
  }

  dof_ = dof;
}

int JointTrajectory::dof() const {
  return dof_;
}

bool JointTrajectory::empty() const {
  return times_.empty();
}

std::size_t JointTrajectory::size() const {
  return times_.size();
}

double JointTrajectory::startTime() const {
  if (empty()) {
    return 0.0;
  }
  return times_.front();
}

double JointTrajectory::endTime() const {
  if (empty()) {
    return 0.0;
  }
  return times_.back();
}

double JointTrajectory::duration() const {
  if (empty()) {
    return 0.0;
  }
  return endTime() - startTime();
}

void JointTrajectory::clear() {
  times_.clear();
  q_.clear();
  dq_.clear();
  ddq_.clear();
}

bool JointTrajectory::checkVectorSize(const Eigen::VectorXd& q,
                                      const Eigen::VectorXd& dq,
                                      const Eigen::VectorXd& ddq) const {
  if (dof_ <= 0) {
    return false;
  }

  return q.size() == dof_ && dq.size() == dof_ && ddq.size() == dof_;
}

bool JointTrajectory::addPoint(double t,
                               const Eigen::VectorXd& q,
                               const Eigen::VectorXd& dq,
                               const Eigen::VectorXd& ddq) {
  if (dof_ <= 0) {
    setDof(static_cast<int>(q.size()));
  }

  if (!checkVectorSize(q, dq, ddq)) {
    return false;
  }

  if (!times_.empty() && t <= times_.back()) {
    return false;
  }

  times_.push_back(t);
  q_.push_back(q);
  dq_.push_back(dq);
  ddq_.push_back(ddq);

  return true;
}

bool JointTrajectory::sample(double t,
                             Eigen::VectorXd& q,
                             Eigen::VectorXd& dq,
                             Eigen::VectorXd& ddq) const {
  if (empty() || dof_ <= 0) {
    return false;
  }

  if (t <= times_.front()) {
    q = q_.front();
    dq = dq_.front();
    ddq = ddq_.front();
    return true;
  }

  if (t >= times_.back()) {
    q = q_.back();
    dq = dq_.back();
    ddq = ddq_.back();
    return true;
  }

  auto upper_it = std::upper_bound(times_.begin(), times_.end(), t);
  const std::size_t idx1 = static_cast<std::size_t>(upper_it - times_.begin());
  const std::size_t idx0 = idx1 - 1;

  const double t0 = times_[idx0];
  const double t1 = times_[idx1];
  const double h = t1 - t0;

  if (h <= 1e-12) {
    return false;
  }

  const double s = (t - t0) / h;

  q = (1.0 - s) * q_[idx0] + s * q_[idx1];
  dq = (1.0 - s) * dq_[idx0] + s * dq_[idx1];
  ddq = (1.0 - s) * ddq_[idx0] + s * ddq_[idx1];

  return true;
}

JointTrajectory JointTrajectory::truncate(double t0, double t1) const {
  JointTrajectory out(dof_);

  if (empty() || t1 < t0) {
    return out;
  }

  const double t_start = std::max(t0, startTime());
  const double t_end = std::min(t1, endTime());

  if (t_end < t_start) {
    return out;
  }

  Eigen::VectorXd q_sample, dq_sample, ddq_sample;

  if (sample(t_start, q_sample, dq_sample, ddq_sample)) {
    out.addPoint(0.0, q_sample, dq_sample, ddq_sample);
  }

  for (std::size_t i = 0; i < times_.size(); ++i) {
    if (times_[i] > t_start && times_[i] < t_end) {
      out.addPoint(times_[i] - t_start, q_[i], dq_[i], ddq_[i]);
    }
  }

  if (t_end > t_start && sample(t_end, q_sample, dq_sample, ddq_sample)) {
    out.addPoint(t_end - t_start, q_sample, dq_sample, ddq_sample);
  }

  return out;
}

bool JointTrajectory::append(const JointTrajectory& other,
                             double time_gap,
                             bool skip_duplicate_boundary) {
  if (other.empty()) {
    return true;
  }

  if (empty()) {
    *this = other;
    return true;
  }

  if (dof_ != other.dof()) {
    return false;
  }

  const double offset = endTime() + time_gap - other.startTime();

  std::size_t start_idx = 0;
  if (skip_duplicate_boundary && other.size() > 0) {
    start_idx = 1;
  }

  for (std::size_t i = start_idx; i < other.size(); ++i) {
    const double new_t = other.times()[i] + offset;
    if (!addPoint(new_t,
                  other.positions()[i],
                  other.velocities()[i],
                  other.accelerations()[i])) {
      return false;
    }
  }

  return true;
}

const std::vector<double>& JointTrajectory::times() const {
  return times_;
}

const std::vector<Eigen::VectorXd>& JointTrajectory::positions() const {
  return q_;
}

const std::vector<Eigen::VectorXd>& JointTrajectory::velocities() const {
  return dq_;
}

const std::vector<Eigen::VectorXd>& JointTrajectory::accelerations() const {
  return ddq_;
}

JointTrajectory JointTrajectory::makeHold(const Eigen::VectorXd& q,
                                          double duration,
                                          double dt) {
  if (duration < 0.0 || dt <= 0.0) {
    throw std::runtime_error("[JointTrajectory] invalid duration or dt in makeHold.");
  }

  const int dof = static_cast<int>(q.size());
  JointTrajectory traj(dof);

  const Eigen::VectorXd zero = Eigen::VectorXd::Zero(dof);

  const int steps = std::max(1, static_cast<int>(std::ceil(duration / dt)));
  for (int i = 0; i <= steps; ++i) {
    const double t = std::min(i * dt, duration);
    traj.addPoint(t, q, zero, zero);
  }

  return traj;
}

JointTrajectory JointTrajectory::makeLinear(const Eigen::VectorXd& q0,
                                            const Eigen::VectorXd& q1,
                                            double duration,
                                            double dt) {
  if (q0.size() != q1.size()) {
    throw std::runtime_error("[JointTrajectory] q0 and q1 size mismatch in makeLinear.");
  }

  if (duration <= 0.0 || dt <= 0.0) {
    throw std::runtime_error("[JointTrajectory] invalid duration or dt in makeLinear.");
  }

  const int dof = static_cast<int>(q0.size());
  JointTrajectory traj(dof);

  const Eigen::VectorXd v = (q1 - q0) / duration;
  const Eigen::VectorXd a = Eigen::VectorXd::Zero(dof);

  const int steps = std::max(1, static_cast<int>(std::ceil(duration / dt)));
  for (int i = 0; i <= steps; ++i) {
    const double t = std::min(i * dt, duration);
    const double s = t / duration;
    const Eigen::VectorXd q = (1.0 - s) * q0 + s * q1;
    traj.addPoint(t, q, v, a);
  }

  return traj;
}

JointTrajectory JointTrajectory::makeQuinticZeroVelocityAcceleration(
    const Eigen::VectorXd& q0,
    const Eigen::VectorXd& q1,
    double duration,
    double dt) {
  if (q0.size() != q1.size()) {
    throw std::runtime_error(
        "[JointTrajectory] q0 and q1 size mismatch in makeQuinticZeroVelocityAcceleration.");
  }

  if (duration <= 0.0 || dt <= 0.0) {
    throw std::runtime_error(
        "[JointTrajectory] invalid duration or dt in makeQuinticZeroVelocityAcceleration.");
  }

  const int dof = static_cast<int>(q0.size());
  JointTrajectory traj(dof);

  const Eigen::VectorXd delta = q1 - q0;

  const int steps = std::max(1, static_cast<int>(std::ceil(duration / dt)));

  for (int i = 0; i <= steps; ++i) {
    const double t = std::min(i * dt, duration);
    const double s = t / duration;

    // Quintic smoothstep:
    // h(s) = 10s^3 - 15s^4 + 6s^5
    // h(0)=0, h(1)=1
    // h'(0)=h'(1)=0
    // h''(0)=h''(1)=0
    const double s2 = s * s;
    const double s3 = s2 * s;
    const double s4 = s3 * s;
    const double s5 = s4 * s;

    const double h = 10.0 * s3 - 15.0 * s4 + 6.0 * s5;

    const double dh_ds =
        30.0 * s2 - 60.0 * s3 + 30.0 * s4;

    const double d2h_ds2 =
        60.0 * s - 180.0 * s2 + 120.0 * s3;

    const double dh_dt = dh_ds / duration;
    const double d2h_dt2 = d2h_ds2 / (duration * duration);

    const Eigen::VectorXd q = q0 + h * delta;
    const Eigen::VectorXd dq = dh_dt * delta;
    const Eigen::VectorXd ddq = d2h_dt2 * delta;

    traj.addPoint(t, q, dq, ddq);
  }

  return traj;
}

}  // namespace arm_trajectory