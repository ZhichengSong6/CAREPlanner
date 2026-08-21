#include <care_confidence_map/trajectory_risk_evaluator.hpp>

#include <ros/ros.h>

#include <geometry_msgs/PointStamped.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/String.h>
#include <trajectory_msgs/JointTrajectory.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

class PredictedVbcAuditorNode
{
public:
  PredictedVbcAuditorNode()
    : nh_()
    , pnh_("~")
  {
  }

  bool initialize()
  {
    loadParams();

    std::string error_msg;
    if (!evaluator_.initialize(
            robot_urdf_file_, body_samples_file_, base_frame_, &error_msg))
    {
      ROS_ERROR_STREAM("[predicted_vbc_auditor] evaluator init failed: " << error_msg);
      return false;
    }

    if (sensor_frames_.empty())
    {
      ROS_ERROR("[predicted_vbc_auditor] trajectory_vbc/sensor_frames is empty");
      return false;
    }

    if (!buildSensorBasis())
    {
      return false;
    }

    if (fallback_dt_ <= 0.0 || min_margin_s_ < 0.0 ||
        tof_min_range_ < 0.0 || tof_max_range_ <= tof_min_range_ ||
        horizontal_fov_deg_ <= 0.0 || vertical_fov_deg_ <= 0.0)
    {
      ROS_ERROR("[predicted_vbc_auditor] invalid geometry/timing parameters");
      return false;
    }

    tan_half_h_fov_ = std::tan(0.5 * horizontal_fov_deg_ * M_PI / 180.0);
    tan_half_v_fov_ = std::tan(0.5 * vertical_fov_deg_ * M_PI / 180.0);

    // Validate the sensor frame names only once at startup. Runtime calls use
    // the same fixed list and the optimized single-FK evaluator helper.
    Eigen::VectorXd q0(evaluator_.nq());
    q0.setZero();
    care_confidence_map::ConfigurationAuditGeometry geometry;
    if (!evaluator_.computeAuditGeometryForConfiguration(
            q0, 0, sensor_frames_, &geometry, &error_msg))
    {
      ROS_ERROR_STREAM("[predicted_vbc_auditor] invalid sensor/body geometry: " << error_msg);
      return false;
    }

    target_sub_ = nh_.subscribe(
        target_topic_, 1, &PredictedVbcAuditorNode::targetCallback, this);
    active_sub_ = nh_.subscribe(
        active_topic_, 1, &PredictedVbcAuditorNode::activeCallback, this);
    prediction_sub_ = nh_.subscribe(
        prediction_topic_, 1, &PredictedVbcAuditorNode::predictionCallback, this);

    summary_pub_ = nh_.advertise<std_msgs::String>(summary_topic_, 1);
    violation_pub_ = nh_.advertise<std_msgs::Bool>(violation_topic_, 1);
    margin_pub_ = nh_.advertise<std_msgs::Float32>(margin_topic_, 1);
    audit_time_pub_ = nh_.advertise<std_msgs::Float32>(audit_time_topic_, 1);

    ROS_WARN_STREAM(
        "[predicted_vbc_auditor] C4.1 DIAGNOSTIC ONLY; no control output. prediction="
        << prediction_topic_ << " target=" << target_topic_
        << " required_margin=" << min_margin_s_ << "s"
        << " warn_budget=" << audit_warn_ms_ << "ms");
    ROS_INFO_STREAM(
        "[predicted_vbc_auditor] horizon audit uses full MPC prediction (normally 21 q), "
        "one Pinocchio FK per evaluated q, " << sensor_frames_.size()
        << " sensor FOV checks, and " << evaluator_.bodySampleModel().riskSampleCount()
        << " body samples per q");
    return true;
  }

private:
  void loadParams()
  {
    pnh_.param<std::string>(
        "predicted_vbc/robot_urdf_file", robot_urdf_file_, "");
    pnh_.param<std::string>(
        "predicted_vbc/body_samples_file", body_samples_file_, "");
    pnh_.param<std::string>(
        "predicted_vbc/base_frame", base_frame_, "base_link");
    pnh_.param<std::string>(
        "predicted_vbc/prediction_topic", prediction_topic_,
        "/care_planner/mpc/predicted_trajectory");
    pnh_.param<std::string>(
        "predicted_vbc/target_topic", target_topic_,
        "/care_planner/active_sensing/target_point");
    pnh_.param<std::string>(
        "predicted_vbc/active_topic", active_topic_,
        "/care_planner/active_sensing/visibility_waypoint_active");
    pnh_.param<std::string>(
        "predicted_vbc/summary_topic", summary_topic_,
        "/care_planner/execution/predicted_vbc_audit_summary");
    pnh_.param<std::string>(
        "predicted_vbc/violation_topic", violation_topic_,
        "/care_planner/execution/predicted_vbc_violation");
    pnh_.param<std::string>(
        "predicted_vbc/margin_topic", margin_topic_,
        "/care_planner/execution/predicted_vbc_margin_s");
    pnh_.param<std::string>(
        "predicted_vbc/audit_time_topic", audit_time_topic_,
        "/care_planner/execution/predicted_vbc_audit_ms");

    pnh_.param("predicted_vbc/enabled", enabled_, true);
    pnh_.param("predicted_vbc/require_active", require_active_, true);
    pnh_.param("predicted_vbc/fallback_dt", fallback_dt_, 0.05);
    pnh_.param("predicted_vbc/min_margin_s", min_margin_s_, 0.30);
    pnh_.param("predicted_vbc/sweep_extra_margin_m", sweep_extra_margin_m_, 0.0);
    pnh_.param("predicted_vbc/audit_warn_ms", audit_warn_ms_, 5.0);

    pnh_.param<std::string>(
        "trajectory_vbc/sensor_model/forward_axis", forward_axis_, "z");
    pnh_.param(
        "trajectory_vbc/sensor_model/tof_min_range", tof_min_range_, 0.15);
    pnh_.param(
        "trajectory_vbc/sensor_model/tof_max_range", tof_max_range_, 0.75);
    pnh_.param(
        "trajectory_vbc/sensor_model/horizontal_fov_deg", horizontal_fov_deg_, 55.0);
    pnh_.param(
        "trajectory_vbc/sensor_model/vertical_fov_deg", vertical_fov_deg_, 72.0);
    sensor_frames_.clear();
    pnh_.getParam("trajectory_vbc/sensor_frames", sensor_frames_);

    ignored_risk_links_.clear();
    if (!pnh_.getParam("trajectory_vbc/ignored_risk_links", ignored_risk_links_))
    {
      ignored_risk_links_.push_back("base_link");
      ignored_risk_links_.push_back("link1");
    }
  }

  bool buildSensorBasis()
  {
    if (forward_axis_ == "x") forward_dir_ = Eigen::Vector3d::UnitX();
    else if (forward_axis_ == "-x") forward_dir_ = -Eigen::Vector3d::UnitX();
    else if (forward_axis_ == "y") forward_dir_ = Eigen::Vector3d::UnitY();
    else if (forward_axis_ == "-y") forward_dir_ = -Eigen::Vector3d::UnitY();
    else if (forward_axis_ == "z") forward_dir_ = Eigen::Vector3d::UnitZ();
    else if (forward_axis_ == "-z") forward_dir_ = -Eigen::Vector3d::UnitZ();
    else
    {
      ROS_ERROR_STREAM("[predicted_vbc_auditor] invalid forward_axis=" << forward_axis_);
      return false;
    }

    if (forward_axis_ == "x" || forward_axis_ == "-x")
    {
      right_dir_ = Eigen::Vector3d::UnitY();
      up_dir_ = Eigen::Vector3d::UnitZ();
    }
    else if (forward_axis_ == "y" || forward_axis_ == "-y")
    {
      right_dir_ = Eigen::Vector3d::UnitX();
      up_dir_ = Eigen::Vector3d::UnitZ();
    }
    else
    {
      right_dir_ = Eigen::Vector3d::UnitX();
      up_dir_ = Eigen::Vector3d::UnitY();
    }
    return true;
  }

  bool isIgnoredRiskLink(const std::string& link_name) const
  {
    return std::find(ignored_risk_links_.begin(), ignored_risk_links_.end(), link_name)
        != ignored_risk_links_.end();
  }

  bool pointVisibleFromSensor(
      const Eigen::Vector3d& point_base,
      const care_confidence_map::FramePoseInBase& sensor_pose) const
  {
    const Eigen::Vector3d p_sensor =
        sensor_pose.rotation_base.transpose() *
        (point_base - sensor_pose.translation_base);
    const double depth = forward_dir_.dot(p_sensor);
    if (depth < tof_min_range_ || depth > tof_max_range_)
    {
      return false;
    }
    const double horizontal = right_dir_.dot(p_sensor);
    const double vertical = up_dir_.dot(p_sensor);
    return std::fabs(horizontal) <= depth * tan_half_h_fov_ &&
           std::fabs(vertical) <= depth * tan_half_v_fov_;
  }

  bool pointVisibleFromAnySensor(
      const Eigen::Vector3d& point_base,
      const std::vector<care_confidence_map::FramePoseInBase>& poses) const
  {
    for (const auto& pose : poses)
    {
      if (pointVisibleFromSensor(point_base, pose))
      {
        return true;
      }
    }
    return false;
  }

  double minBodyClearance(
      const Eigen::Vector3d& point_base,
      const std::vector<care_confidence_map::TrajectoryBodySample>& samples) const
  {
    double best = std::numeric_limits<double>::infinity();
    for (const auto& sample : samples)
    {
      if (isIgnoredRiskLink(sample.link_name))
      {
        continue;
      }
      const double clearance =
          (sample.center_base - point_base).norm() - sample.radius;
      best = std::min(best, clearance);
    }
    return best;
  }

  bool buildJointMap(
      const trajectory_msgs::JointTrajectory& traj,
      std::vector<int>* required_to_input,
      std::string* error_msg) const
  {
    std::map<std::string, int> input_index;
    for (int i = 0; i < static_cast<int>(traj.joint_names.size()); ++i)
    {
      input_index[traj.joint_names[static_cast<std::size_t>(i)]] = i;
    }

    const auto& required = evaluator_.activeJointNames();
    required_to_input->assign(required.size(), -1);
    for (std::size_t i = 0; i < required.size(); ++i)
    {
      const auto it = input_index.find(required[i]);
      if (it == input_index.end())
      {
        if (error_msg) *error_msg = "prediction missing joint " + required[i];
        return false;
      }
      (*required_to_input)[i] = it->second;
    }
    return true;
  }

  void targetCallback(const geometry_msgs::PointStampedConstPtr& msg)
  {
    if (!msg) return;
    const Eigen::Vector3d p(msg->point.x, msg->point.y, msg->point.z);
    if (!p.allFinite()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    target_base_ = p;
    has_target_ = true;
  }

  void activeCallback(const std_msgs::BoolConstPtr& msg)
  {
    if (!msg) return;
    std::lock_guard<std::mutex> lock(mutex_);
    active_ = msg->data;
    has_active_ = true;
  }

  void publishInactive(const std::string& status)
  {
    std_msgs::Bool violation_msg;
    violation_msg.data = false;
    violation_pub_.publish(violation_msg);

    std_msgs::String summary;
    std::ostringstream oss;
    oss << "enabled=" << static_cast<int>(enabled_)
        << " active=0 status=" << status
        << " violation=0 audit_ms=0 evaluated_q=0";
    summary.data = oss.str();
    summary_pub_.publish(summary);
  }

  void predictionCallback(const trajectory_msgs::JointTrajectoryConstPtr& msg)
  {
    if (!msg || !enabled_)
    {
      return;
    }

    Eigen::Vector3d target;
    bool active = false;
    bool has_active = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!has_target_)
      {
        publishInactive("waiting_target");
        return;
      }
      target = target_base_;
      active = active_;
      has_active = has_active_;
    }

    if (require_active_ && (!has_active || !active))
    {
      publishInactive("inactive");
      return;
    }

    if (msg->joint_names.empty() || msg->points.empty())
    {
      publishError("empty_prediction", 0.0, 0);
      return;
    }

    std::vector<int> required_to_input;
    std::string error_msg;
    if (!buildJointMap(*msg, &required_to_input, &error_msg))
    {
      publishError("joint_map_error", 0.0, 0);
      ROS_WARN_STREAM_THROTTLE(1.0, "[predicted_vbc_auditor] " << error_msg);
      return;
    }

    const ros::WallTime tic = ros::WallTime::now();

    double first_see_s = std::numeric_limits<double>::infinity();
    double first_sweep_s = std::numeric_limits<double>::infinity();
    double min_clearance_m = std::numeric_limits<double>::infinity();
    double margin_s = std::numeric_limits<double>::quiet_NaN();
    int see_k = -1;
    int sweep_k = -1;
    int evaluated_q = 0;
    bool violation = false;
    bool outcome_decided = false;
    std::string status = "no_sweep_in_horizon";

    Eigen::VectorXd q(evaluator_.nq());
    care_confidence_map::ConfigurationAuditGeometry geometry;

    for (std::size_t k = 0; k < msg->points.size(); ++k)
    {
      const auto& pt = msg->points[k];
      if (pt.positions.size() < msg->joint_names.size())
      {
        publishError("invalid_prediction_point", elapsedMs(tic), evaluated_q);
        return;
      }

      for (int j = 0; j < evaluator_.nq(); ++j)
      {
        q(j) = pt.positions[static_cast<std::size_t>(
            required_to_input[static_cast<std::size_t>(j)])];
      }
      if (!q.allFinite())
      {
        publishError("nonfinite_prediction", elapsedMs(tic), evaluated_q);
        return;
      }

      double t = pt.time_from_start.toSec();
      if (k > 0 && t <= 0.0)
      {
        t = static_cast<double>(k) * fallback_dt_;
      }

      if (!evaluator_.computeAuditGeometryForConfiguration(
              q, static_cast<int>(k), sensor_frames_, &geometry, &error_msg))
      {
        publishError("fk_error", elapsedMs(tic), evaluated_q);
        ROS_WARN_STREAM_THROTTLE(1.0, "[predicted_vbc_auditor] FK error: " << error_msg);
        return;
      }
      evaluated_q += 1;

      if (!std::isfinite(first_see_s) &&
          pointVisibleFromAnySensor(target, geometry.frame_poses))
      {
        first_see_s = t;
        see_k = static_cast<int>(k);
      }

      const double clearance = minBodyClearance(target, geometry.body_samples);
      min_clearance_m = std::min(min_clearance_m, clearance);
      if (clearance <= sweep_extra_margin_m_)
      {
        first_sweep_s = t;
        sweep_k = static_cast<int>(k);
        if (!std::isfinite(first_see_s))
        {
          violation = true;
          status = "violation_unseen_before_sweep";
        }
        else
        {
          margin_s = first_sweep_s - first_see_s;
          violation = margin_s + 1e-9 < min_margin_s_;
          status = violation ? "violation_margin" : "safe_margin";
        }
        outcome_decided = true;
        break;
      }

      // If visibility was established at least min_margin_s_ ago and there has
      // been no sweep up to the current q, every possible later sweep in this
      // prediction necessarily satisfies the VBC lead requirement.  Stop here
      // instead of spending FK on the rest of the horizon.
      if (std::isfinite(first_see_s) && t - first_see_s + 1e-9 >= min_margin_s_)
      {
        margin_s = t - first_see_s;  // guaranteed lower bound, not exact sweep margin
        violation = false;
        status = "safe_margin_guaranteed";
        outcome_decided = true;
        break;
      }
    }

    if (!outcome_decided)
    {
      violation = false;
      status = "no_sweep_in_horizon";
    }

    const double audit_ms = elapsedMs(tic);
    audit_count_ += 1;
    audit_ms_sum_ += audit_ms;
    audit_ms_max_ = std::max(audit_ms_max_, audit_ms);
    const double audit_ms_mean = audit_ms_sum_ / static_cast<double>(audit_count_);

    std_msgs::Bool violation_msg;
    violation_msg.data = violation;
    violation_pub_.publish(violation_msg);

    std_msgs::Float32 margin_msg;
    margin_msg.data = std::isfinite(margin_s)
        ? static_cast<float>(margin_s)
        : std::numeric_limits<float>::quiet_NaN();
    margin_pub_.publish(margin_msg);

    std_msgs::Float32 timing_msg;
    timing_msg.data = static_cast<float>(audit_ms);
    audit_time_pub_.publish(timing_msg);

    std_msgs::String summary_msg;
    std::ostringstream oss;
    oss << "enabled=1 active=1"
        << " status=" << status
        << " violation=" << static_cast<int>(violation)
        << " predicted_seen=" << static_cast<int>(std::isfinite(first_see_s))
        << " predicted_sweep=" << static_cast<int>(std::isfinite(first_sweep_s))
        << " see_k=" << see_k
        << " sweep_k=" << sweep_k
        << " see_time_s=" << finiteOrNan(first_see_s)
        << " sweep_time_s=" << finiteOrNan(first_sweep_s)
        << " margin_s=" << finiteOrNan(margin_s)
        << " min_required_margin_s=" << min_margin_s_
        << " min_clearance_m=" << finiteOrNan(min_clearance_m)
        << " evaluated_q=" << evaluated_q
        << " prediction_q=" << msg->points.size()
        << " audit_ms=" << audit_ms
        << " audit_ms_mean=" << audit_ms_mean
        << " audit_ms_max=" << audit_ms_max_;
    summary_msg.data = oss.str();
    summary_pub_.publish(summary_msg);

    if (audit_ms > audit_warn_ms_)
    {
      ROS_WARN_STREAM_THROTTLE(
          1.0,
          "[predicted_vbc_auditor] audit exceeded budget: " << audit_ms
          << "ms > " << audit_warn_ms_ << "ms; evaluated_q=" << evaluated_q
          << "/" << msg->points.size());
    }
    else
    {
      ROS_INFO_STREAM_THROTTLE(
          1.0,
          "[predicted_vbc_auditor] " << status
          << " violation=" << violation
          << " see=" << finiteOrNan(first_see_s)
          << " sweep=" << finiteOrNan(first_sweep_s)
          << " margin=" << finiteOrNan(margin_s)
          << " q=" << evaluated_q << "/" << msg->points.size()
          << " audit=" << audit_ms << "ms mean=" << audit_ms_mean
          << "ms max=" << audit_ms_max_ << "ms");
    }
  }

  static double elapsedMs(const ros::WallTime& tic)
  {
    return (ros::WallTime::now() - tic).toSec() * 1000.0;
  }

  static double finiteOrNan(double x)
  {
    return std::isfinite(x) ? x : std::numeric_limits<double>::quiet_NaN();
  }

  void publishError(const std::string& status, double audit_ms, int evaluated_q)
  {
    std_msgs::Bool violation_msg;
    violation_msg.data = false;
    violation_pub_.publish(violation_msg);

    std_msgs::String summary_msg;
    std::ostringstream oss;
    oss << "enabled=1 active=1 status=" << status
        << " violation=0 audit_ms=" << audit_ms
        << " evaluated_q=" << evaluated_q;
    summary_msg.data = oss.str();
    summary_pub_.publish(summary_msg);
  }

private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber target_sub_;
  ros::Subscriber active_sub_;
  ros::Subscriber prediction_sub_;
  ros::Publisher summary_pub_;
  ros::Publisher violation_pub_;
  ros::Publisher margin_pub_;
  ros::Publisher audit_time_pub_;

  care_confidence_map::TrajectoryRiskEvaluator evaluator_;

  std::string robot_urdf_file_;
  std::string body_samples_file_;
  std::string base_frame_ = "base_link";
  std::string prediction_topic_;
  std::string target_topic_;
  std::string active_topic_;
  std::string summary_topic_;
  std::string violation_topic_;
  std::string margin_topic_;
  std::string audit_time_topic_;

  bool enabled_ = true;
  bool require_active_ = true;
  double fallback_dt_ = 0.05;
  double min_margin_s_ = 0.30;
  double sweep_extra_margin_m_ = 0.0;
  double audit_warn_ms_ = 5.0;

  std::string forward_axis_ = "z";
  double tof_min_range_ = 0.15;
  double tof_max_range_ = 0.75;
  double horizontal_fov_deg_ = 55.0;
  double vertical_fov_deg_ = 72.0;
  double tan_half_h_fov_ = 0.0;
  double tan_half_v_fov_ = 0.0;
  Eigen::Vector3d forward_dir_ = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d right_dir_ = Eigen::Vector3d::UnitX();
  Eigen::Vector3d up_dir_ = Eigen::Vector3d::UnitY();
  std::vector<std::string> sensor_frames_;
  std::vector<std::string> ignored_risk_links_;

  std::mutex mutex_;
  bool has_target_ = false;
  Eigen::Vector3d target_base_ = Eigen::Vector3d::Zero();
  bool has_active_ = false;
  bool active_ = false;

  std::size_t audit_count_ = 0;
  double audit_ms_sum_ = 0.0;
  double audit_ms_max_ = 0.0;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "predicted_vbc_auditor_node");
  PredictedVbcAuditorNode node;
  if (!node.initialize())
  {
    return 1;
  }
  ros::spin();
  return 0;
}
