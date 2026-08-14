#include <ros/ros.h>

#include <geometry_msgs/PoseStamped.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void lerpVector(const std::vector<double>& a,
                const std::vector<double>& b,
                double alpha,
                std::vector<double>& out) {
  if (a.size() != b.size()) {
    out.clear();
    return;
  }
  out.resize(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    out[i] = (1.0 - alpha) * a[i] + alpha * b[i];
  }
}

trajectory_msgs::JointTrajectoryPoint interpolatePoint(
    const trajectory_msgs::JointTrajectoryPoint& p0,
    const trajectory_msgs::JointTrajectoryPoint& p1,
    double alpha) {
  trajectory_msgs::JointTrajectoryPoint out;
  lerpVector(p0.positions, p1.positions, alpha, out.positions);
  lerpVector(p0.velocities, p1.velocities, alpha, out.velocities);
  lerpVector(p0.accelerations, p1.accelerations, alpha, out.accelerations);
  lerpVector(p0.effort, p1.effort, alpha, out.effort);
  return out;
}

class PersistentReferenceRelay {
public:
  PersistentReferenceRelay()
      : nh_(), pnh_("~") {
    pnh_.param<std::string>(
        "input_trajectory",
        input_topic_,
        "/care_planner/command_trajectory_candidate");
    pnh_.param<std::string>(
        "target_pose",
        target_topic_,
        "/care_planner/ee_target_pose");
    pnh_.param<std::string>(
        "output_trajectory",
        output_topic_,
        "/care_planner/command_trajectory_persistent");
    pnh_.param<double>("publish_rate", publish_rate_, 20.0);

    if (publish_rate_ <= 0.0) {
      throw std::runtime_error("publish_rate must be positive");
    }

    input_sub_ = nh_.subscribe(
        input_topic_, 1, &PersistentReferenceRelay::inputCallback, this);
    target_sub_ = nh_.subscribe(
        target_topic_, 1, &PersistentReferenceRelay::targetCallback, this);
    output_pub_ = nh_.advertise<trajectory_msgs::JointTrajectory>(
        output_topic_, 1, false);
    timer_ = nh_.createTimer(
        ros::Duration(1.0 / publish_rate_),
        &PersistentReferenceRelay::timerCallback,
        this);

    ROS_INFO_STREAM(
        "[PersistentReferenceRelay] input=" << input_topic_
        << ", output=" << output_topic_
        << ", target=" << target_topic_
        << ", rate=" << publish_rate_ << " Hz");
    ROS_WARN(
        "[PersistentReferenceRelay] Diagnostic Stage 2: latch only the first successful nominal trajectory after each new target and ignore later replans.");
  }

private:
  void targetCallback(const geometry_msgs::PoseStampedConstPtr& msg) {
    if (!msg) return;
    std::lock_guard<std::mutex> lock(mutex_);
    waiting_for_reference_ = true;
    has_reference_ = false;
    latched_reference_ = trajectory_msgs::JointTrajectory();
    ROS_INFO("[PersistentReferenceRelay] New target received. Waiting for first successful nominal trajectory.");
  }

  void inputCallback(const trajectory_msgs::JointTrajectoryConstPtr& msg) {
    if (!msg || msg->points.empty()) return;

    std::lock_guard<std::mutex> lock(mutex_);
    if (!waiting_for_reference_) {
      return;
    }

    latched_reference_ = *msg;
    reference_start_time_ = ros::Time::now();
    waiting_for_reference_ = false;
    has_reference_ = true;

    ROS_INFO_STREAM(
        "[PersistentReferenceRelay] Latched first nominal trajectory. duration="
        << latched_reference_.points.back().time_from_start.toSec()
        << " s, points=" << latched_reference_.points.size());
  }

  bool makeSuffix(const trajectory_msgs::JointTrajectory& reference,
                  double elapsed,
                  trajectory_msgs::JointTrajectory& suffix) const {
    suffix = trajectory_msgs::JointTrajectory();
    if (reference.points.empty()) return false;

    suffix.header = reference.header;
    suffix.header.stamp = ros::Time::now();
    suffix.joint_names = reference.joint_names;

    const double first_t = reference.points.front().time_from_start.toSec();
    const double last_t = reference.points.back().time_from_start.toSec();
    const double phase = std::max(first_t, std::min(elapsed, last_t));

    trajectory_msgs::JointTrajectoryPoint first;
    if (phase <= first_t + 1e-12) {
      first = reference.points.front();
    } else if (phase >= last_t - 1e-12) {
      first = reference.points.back();
    } else {
      std::size_t hi = 1;
      while (hi < reference.points.size() &&
             reference.points[hi].time_from_start.toSec() < phase) {
        ++hi;
      }
      if (hi >= reference.points.size()) {
        first = reference.points.back();
      } else {
        const std::size_t lo = hi - 1;
        const double t0 = reference.points[lo].time_from_start.toSec();
        const double t1 = reference.points[hi].time_from_start.toSec();
        const double h = t1 - t0;
        if (h <= 1e-12) return false;
        const double alpha = (phase - t0) / h;
        first = interpolatePoint(reference.points[lo], reference.points[hi], alpha);
      }
    }
    first.time_from_start = ros::Duration(0.0);
    suffix.points.push_back(first);

    for (const auto& point : reference.points) {
      const double t = point.time_from_start.toSec();
      if (t <= phase + 1e-9) continue;
      auto shifted = point;
      shifted.time_from_start = ros::Duration(t - phase);
      suffix.points.push_back(shifted);
    }

    return !suffix.points.empty();
  }

  void timerCallback(const ros::TimerEvent&) {
    trajectory_msgs::JointTrajectory reference;
    ros::Time start_time;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!has_reference_) return;
      reference = latched_reference_;
      start_time = reference_start_time_;
    }

    const double elapsed = std::max(
        0.0, (ros::Time::now() - start_time).toSec());
    trajectory_msgs::JointTrajectory suffix;
    if (!makeSuffix(reference, elapsed, suffix)) {
      ROS_WARN_THROTTLE(
          1.0,
          "[PersistentReferenceRelay] Failed to construct persistent reference suffix.");
      return;
    }

    output_pub_.publish(suffix);
    ROS_INFO_STREAM_THROTTLE(
        0.5,
        "[PersistentReferenceRelay] phase=" << elapsed
        << " s, remaining="
        << suffix.points.back().time_from_start.toSec()
        << " s, points=" << suffix.points.size());
  }

private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber input_sub_;
  ros::Subscriber target_sub_;
  ros::Publisher output_pub_;
  ros::Timer timer_;

  mutable std::mutex mutex_;
  bool waiting_for_reference_ = false;
  bool has_reference_ = false;
  trajectory_msgs::JointTrajectory latched_reference_;
  ros::Time reference_start_time_;

  std::string input_topic_;
  std::string target_topic_;
  std::string output_topic_;
  double publish_rate_ = 20.0;
};

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "persistent_reference_relay_node");
  try {
    PersistentReferenceRelay node;
    ros::spin();
  } catch (const std::exception& e) {
    ROS_FATAL_STREAM("[PersistentReferenceRelay] Initialization failed: " << e.what());
    return 1;
  }
  return 0;
}
