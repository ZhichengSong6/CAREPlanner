#include <care_confidence_map/trajectory_risk_evaluator.hpp>
#include <care_confidence_map/QueryConfidence.h>

#include <ros/ros.h>

#include <geometry_msgs/Point.h>
#include <geometry_msgs/PointStamped.h>
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
#include <tuple>
#include <vector>

class TrajectoryVbcSelectorNode
{
public:
  TrajectoryVbcSelectorNode()
    : nh_()
    , pnh_("~")
  {
  }

  bool initialize()
  {
    loadParams();

    std::string error_msg;
    if (!evaluator_.initialize(
            robot_urdf_file_,
            body_samples_file_,
            base_frame_,
            &error_msg))
    {
      ROS_ERROR_STREAM("[trajectory_vbc] Failed to initialize evaluator: "
                       << error_msg);
      return false;
    }

    if (sensor_frames_.empty())
    {
      ROS_ERROR("[trajectory_vbc] trajectory_vbc/sensor_frames is empty.");
      return false;
    }

    if (!buildSensorBasis())
    {
      return false;
    }

    if (candidate_resolution_ <= 0.0 ||
        tof_min_range_ < 0.0 ||
        tof_max_range_ <= tof_min_range_ ||
        horizontal_fov_deg_ <= 0.0 ||
        vertical_fov_deg_ <= 0.0)
    {
      ROS_ERROR("[trajectory_vbc] Invalid VBC geometry parameters.");
      return false;
    }

    tan_half_h_fov_ =
        std::tan(0.5 * horizontal_fov_deg_ * M_PI / 180.0);
    tan_half_v_fov_ =
        std::tan(0.5 * vertical_fov_deg_ * M_PI / 180.0);

    // Validate sensor frame names before accepting trajectories.
    Eigen::VectorXd q0(evaluator_.nq());
    q0.setZero();
    std::vector<care_confidence_map::FramePoseInBase> poses;
    if (!evaluator_.computeFramePosesForConfiguration(
            q0, sensor_frames_, &poses, &error_msg))
    {
      ROS_ERROR_STREAM("[trajectory_vbc] Invalid sensor frame configuration: "
                       << error_msg);
      return false;
    }

    confidence_query_client_ =
        nh_.serviceClient<care_confidence_map::QueryConfidence>(
            confidence_query_service_);

    trajectory_sub_ =
        nh_.subscribe(
            input_trajectory_topic_,
            1,
            &TrajectoryVbcSelectorNode::trajectoryCallback,
            this);

    target_pub_ =
        nh_.advertise<geometry_msgs::PointStamped>(
            output_target_topic_, 1, false);

    summary_pub_ =
        nh_.advertise<std_msgs::String>(
            "/care_planner/trajectory_risk/vbc_summary", 1, true);

    selected_margin_pub_ =
        nh_.advertise<std_msgs::Float32>(
            "/care_planner/trajectory_risk/vbc_selected_margin_s", 1, true);

    selected_sweep_time_pub_ =
        nh_.advertise<std_msgs::Float32>(
            "/care_planner/trajectory_risk/vbc_selected_sweep_time_s", 1, true);

    selected_see_time_pub_ =
        nh_.advertise<std_msgs::Float32>(
            "/care_planner/trajectory_risk/vbc_selected_see_time_s", 1, true);

    timer_ =
        nh_.createTimer(
            ros::Duration(1.0 / std::max(0.1, eval_rate_)),
            &TrajectoryVbcSelectorNode::timerCallback,
            this);

    printSummary();
    return true;
  }

private:
  struct Candidate
  {
    Eigen::Vector3d point_base = Eigen::Vector3d::Zero();
    std::string link_name = "none";
    int sample_index_in_link = -1;

    double confidence = 0.0;
    double current_visibility = 0.0;

    int sweep_eval_timestep = -1;
    int sweep_original_timestep = -1;
    double sweep_time_s = 0.0;

    bool nominally_visible = false;
    int see_eval_timestep = -1;
    int see_original_timestep = -1;
    double see_time_s = std::numeric_limits<double>::infinity();

    // Positive means nominal visibility happens before sweep.  Negative means
    // the nominal body arrives first.  -inf means never visible in horizon.
    double margin_s = -std::numeric_limits<double>::infinity();
  };

  using CandidateKey = std::tuple<long long, long long, long long>;

  void loadParams()
  {
    pnh_.param<std::string>(
        "trajectory_vbc/robot_urdf_file", robot_urdf_file_, "");
    pnh_.param<std::string>(
        "trajectory_vbc/body_samples_file", body_samples_file_, "");
    pnh_.param<std::string>(
        "trajectory_vbc/base_frame", base_frame_, "base_link");
    pnh_.param<std::string>(
        "trajectory_vbc/input_trajectory_topic",
        input_trajectory_topic_,
        "/care_planner/task_trajectory");
    pnh_.param<std::string>(
        "trajectory_vbc/output_target_topic",
        output_target_topic_,
        "/care_planner/active_sensing/target_candidate");
    pnh_.param<std::string>(
        "trajectory_vbc/confidence_query_service",
        confidence_query_service_,
        "/care_planner/confidence_map/query");

    pnh_.param("trajectory_vbc/enabled", enabled_, true);
    pnh_.param("trajectory_vbc/eval_rate", eval_rate_, 20.0);
    pnh_.param("trajectory_vbc/max_eval_timesteps", max_eval_timesteps_, 50);
    pnh_.param("trajectory_vbc/query_timeout", query_timeout_, 0.10);
    pnh_.param("trajectory_vbc/fallback_dt", fallback_dt_, 0.05);

    pnh_.param(
        "trajectory_vbc/frontier_confidence_threshold",
        frontier_confidence_threshold_,
        0.50);
    pnh_.param(
        "trajectory_vbc/candidate_resolution",
        candidate_resolution_,
        0.05);
    pnh_.param(
        "trajectory_vbc/min_visibility_before_sweep_margin_s",
        min_margin_s_,
        0.30);

    pnh_.param<std::string>(
        "trajectory_vbc/sensor_model/forward_axis",
        forward_axis_,
        "z");
    pnh_.param(
        "trajectory_vbc/sensor_model/tof_min_range",
        tof_min_range_,
        0.15);
    pnh_.param(
        "trajectory_vbc/sensor_model/tof_max_range",
        tof_max_range_,
        0.75);
    pnh_.param(
        "trajectory_vbc/sensor_model/horizontal_fov_deg",
        horizontal_fov_deg_,
        55.0);
    pnh_.param(
        "trajectory_vbc/sensor_model/vertical_fov_deg",
        vertical_fov_deg_,
        72.0);

    sensor_frames_.clear();
    pnh_.getParam("trajectory_vbc/sensor_frames", sensor_frames_);

    ignored_risk_links_.clear();
    if (!pnh_.getParam(
            "trajectory_vbc/ignored_risk_links", ignored_risk_links_))
    {
      ignored_risk_links_.push_back("base_link");
      ignored_risk_links_.push_back("link1");
    }
  }

  bool buildSensorBasis()
  {
    if (forward_axis_ == "x")
    {
      forward_dir_ = Eigen::Vector3d::UnitX();
    }
    else if (forward_axis_ == "-x")
    {
      forward_dir_ = -Eigen::Vector3d::UnitX();
    }
    else if (forward_axis_ == "y")
    {
      forward_dir_ = Eigen::Vector3d::UnitY();
    }
    else if (forward_axis_ == "-y")
    {
      forward_dir_ = -Eigen::Vector3d::UnitY();
    }
    else if (forward_axis_ == "z")
    {
      forward_dir_ = Eigen::Vector3d::UnitZ();
    }
    else if (forward_axis_ == "-z")
    {
      forward_dir_ = -Eigen::Vector3d::UnitZ();
    }
    else
    {
      ROS_ERROR_STREAM("[trajectory_vbc] Invalid forward_axis: "
                       << forward_axis_);
      return false;
    }

    // Match confidence_map_node's basis convention exactly.
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
    return std::find(
               ignored_risk_links_.begin(),
               ignored_risk_links_.end(),
               link_name) != ignored_risk_links_.end();
  }

  std::vector<int> makeDownsampleIndices(int input_size) const
  {
    std::vector<int> indices;
    if (input_size <= 0)
    {
      return indices;
    }
    if (input_size == 1)
    {
      indices.push_back(0);
      return indices;
    }

    const int max_steps = std::max(2, max_eval_timesteps_);
    if (input_size <= max_steps)
    {
      indices.reserve(static_cast<std::size_t>(input_size));
      for (int i = 0; i < input_size; ++i)
      {
        indices.push_back(i);
      }
      return indices;
    }

    indices.reserve(static_cast<std::size_t>(max_steps));
    for (int k = 0; k < max_steps; ++k)
    {
      const double s =
          static_cast<double>(k) / static_cast<double>(max_steps - 1);
      int idx = static_cast<int>(
          std::round(s * static_cast<double>(input_size - 1)));
      idx = std::max(0, std::min(input_size - 1, idx));
      if (indices.empty() || indices.back() != idx)
      {
        indices.push_back(idx);
      }
    }
    return indices;
  }

  bool convertTrajectory(
      const trajectory_msgs::JointTrajectory& traj,
      std::vector<Eigen::VectorXd>* q_traj,
      std::vector<int>* original_indices,
      std::vector<double>* eval_times_s,
      std::string* error_msg) const
  {
    q_traj->clear();
    original_indices->clear();
    eval_times_s->clear();

    if (traj.joint_names.empty() || traj.points.empty())
    {
      if (error_msg)
      {
        *error_msg = "JointTrajectory has no names or points.";
      }
      return false;
    }

    std::map<std::string, int> input_joint_index;
    for (int i = 0; i < static_cast<int>(traj.joint_names.size()); ++i)
    {
      input_joint_index[traj.joint_names[static_cast<std::size_t>(i)]] = i;
    }

    const auto& required_names = evaluator_.activeJointNames();
    if (static_cast<int>(required_names.size()) != evaluator_.nq())
    {
      if (error_msg)
      {
        *error_msg = "Unexpected evaluator active-joint count.";
      }
      return false;
    }

    std::vector<int> required_to_input(required_names.size(), -1);
    for (int i = 0; i < static_cast<int>(required_names.size()); ++i)
    {
      const auto it = input_joint_index.find(required_names[static_cast<std::size_t>(i)]);
      if (it == input_joint_index.end())
      {
        if (error_msg)
        {
          *error_msg = "Trajectory missing joint: " +
              required_names[static_cast<std::size_t>(i)];
        }
        return false;
      }
      required_to_input[static_cast<std::size_t>(i)] = it->second;
    }

    const std::vector<int> selected =
        makeDownsampleIndices(static_cast<int>(traj.points.size()));

    const bool has_timing =
        traj.points.size() > 1 &&
        traj.points.back().time_from_start.toSec() > 1e-9;

    q_traj->reserve(selected.size());
    original_indices->reserve(selected.size());
    eval_times_s->reserve(selected.size());

    for (const int original_index : selected)
    {
      const auto& pt = traj.points[static_cast<std::size_t>(original_index)];
      if (pt.positions.size() < traj.joint_names.size())
      {
        if (error_msg)
        {
          *error_msg = "Trajectory point has insufficient positions.";
        }
        return false;
      }

      Eigen::VectorXd q(evaluator_.nq());
      for (int i = 0; i < evaluator_.nq(); ++i)
      {
        q(i) = pt.positions[static_cast<std::size_t>(
            required_to_input[static_cast<std::size_t>(i)])];
      }

      double t = has_timing
          ? pt.time_from_start.toSec()
          : static_cast<double>(original_index) * fallback_dt_;

      if (!eval_times_s->empty() && t < eval_times_s->back())
      {
        t = eval_times_s->back();
      }

      q_traj->push_back(q);
      original_indices->push_back(original_index);
      eval_times_s->push_back(t);
    }

    return !q_traj->empty();
  }

  bool queryConfidence(
      const care_confidence_map::TrajectorySampleResult& samples,
      care_confidence_map::QueryConfidence* srv) const
  {
    srv->request.points.clear();
    srv->request.points.reserve(static_cast<std::size_t>(samples.total_samples));

    for (const auto& frame : samples.frames)
    {
      for (const auto& sample : frame.samples)
      {
        geometry_msgs::Point p;
        p.x = sample.center_base.x();
        p.y = sample.center_base.y();
        p.z = sample.center_base.z();
        srv->request.points.push_back(p);
      }
    }

    if (!confidence_query_client_.waitForExistence(ros::Duration(query_timeout_)))
    {
      return false;
    }
    if (!confidence_query_client_.call(*srv))
    {
      return false;
    }

    const std::size_t n = srv->request.points.size();
    return srv->response.confidence.size() == n &&
           srv->response.current_visibility.size() == n &&
           srv->response.inside_map.size() == n;
  }

  CandidateKey candidateKey(const Eigen::Vector3d& p) const
  {
    const double inv = 1.0 / candidate_resolution_;
    return std::make_tuple(
        static_cast<long long>(std::llround(p.x() * inv)),
        static_cast<long long>(std::llround(p.y() * inv)),
        static_cast<long long>(std::llround(p.z() * inv)));
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

  bool betterCandidate(const Candidate& a, const Candidate& b) const
  {
    // Never-visible-before-end candidates are treated as the strongest
    // violations; among them, protect the earliest impending sweep first.
    if (a.nominally_visible != b.nominally_visible)
    {
      return !a.nominally_visible;
    }
    if (!a.nominally_visible)
    {
      return a.sweep_time_s < b.sweep_time_s;
    }
    if (std::fabs(a.margin_s - b.margin_s) > 1e-9)
    {
      return a.margin_s < b.margin_s;
    }
    return a.sweep_time_s < b.sweep_time_s;
  }

  void evaluate(
      const trajectory_msgs::JointTrajectory& traj)
  {
    if (!enabled_)
    {
      publishNoTargetSummary("disabled", 0, 0);
      return;
    }

    std::vector<Eigen::VectorXd> q_traj;
    std::vector<int> original_indices;
    std::vector<double> eval_times_s;
    std::string error_msg;

    if (!convertTrajectory(
            traj, &q_traj, &original_indices, &eval_times_s, &error_msg))
    {
      ROS_WARN_STREAM_THROTTLE(2.0, "[trajectory_vbc] " << error_msg);
      return;
    }

    const care_confidence_map::TrajectorySampleResult sample_result =
        evaluator_.computeTrajectorySamples(q_traj);
    if (!sample_result.success)
    {
      ROS_WARN_STREAM_THROTTLE(
          2.0, "[trajectory_vbc] body-sweep FK failed: " << sample_result.message);
      return;
    }

    care_confidence_map::QueryConfidence confidence_srv;
    if (!queryConfidence(sample_result, &confidence_srv))
    {
      ROS_WARN_THROTTLE(
          2.0,
          "[trajectory_vbc] confidence query failed or returned invalid sizes");
      return;
    }

    std::map<CandidateKey, Candidate> candidate_map;
    std::size_t flat_index = 0;

    for (const auto& frame : sample_result.frames)
    {
      const int k = frame.timestep_index;
      if (k < 0 || k >= static_cast<int>(eval_times_s.size()))
      {
        flat_index += frame.samples.size();
        continue;
      }

      for (const auto& sample : frame.samples)
      {
        const bool inside = confidence_srv.response.inside_map[flat_index] != 0;
        const double confidence =
            static_cast<double>(confidence_srv.response.confidence[flat_index]);
        const double current_visibility =
            static_cast<double>(confidence_srv.response.current_visibility[flat_index]);

        flat_index += 1;

        if (isIgnoredRiskLink(sample.link_name) || !inside)
        {
          continue;
        }
        if (confidence > frontier_confidence_threshold_ ||
            current_visibility > 0.5)
        {
          continue;
        }

        const CandidateKey key = candidateKey(sample.center_base);
        auto it = candidate_map.find(key);
        if (it == candidate_map.end())
        {
          Candidate c;
          c.point_base = sample.center_base;
          c.link_name = sample.link_name;
          c.sample_index_in_link = sample.sample_index_in_link;
          c.confidence = confidence;
          c.current_visibility = current_visibility;
          c.sweep_eval_timestep = k;
          c.sweep_original_timestep = original_indices[static_cast<std::size_t>(k)];
          c.sweep_time_s = eval_times_s[static_cast<std::size_t>(k)];
          candidate_map.emplace(key, c);
        }
        else
        {
          it->second.confidence = std::min(it->second.confidence, confidence);
          if (eval_times_s[static_cast<std::size_t>(k)] < it->second.sweep_time_s)
          {
            it->second.point_base = sample.center_base;
            it->second.link_name = sample.link_name;
            it->second.sample_index_in_link = sample.sample_index_in_link;
            it->second.sweep_eval_timestep = k;
            it->second.sweep_original_timestep =
                original_indices[static_cast<std::size_t>(k)];
            it->second.sweep_time_s = eval_times_s[static_cast<std::size_t>(k)];
          }
        }
      }
    }

    if (candidate_map.empty())
    {
      publishNoTargetSummary("no_low_confidence_sweep_candidates", 0, 0);
      return;
    }

    std::vector<std::vector<care_confidence_map::FramePoseInBase>> sensor_poses;
    sensor_poses.resize(q_traj.size());
    for (std::size_t k = 0; k < q_traj.size(); ++k)
    {
      if (!evaluator_.computeFramePosesForConfiguration(
              q_traj[k], sensor_frames_, &sensor_poses[k], &error_msg))
      {
        ROS_WARN_STREAM_THROTTLE(
            2.0, "[trajectory_vbc] sensor FK failed: " << error_msg);
        return;
      }
    }

    std::vector<Candidate> candidates;
    candidates.reserve(candidate_map.size());
    int violation_count = 0;

    for (auto& kv : candidate_map)
    {
      Candidate c = kv.second;
      for (std::size_t k = 0; k < sensor_poses.size(); ++k)
      {
        if (pointVisibleFromAnySensor(c.point_base, sensor_poses[k]))
        {
          c.nominally_visible = true;
          c.see_eval_timestep = static_cast<int>(k);
          c.see_original_timestep = original_indices[k];
          c.see_time_s = eval_times_s[k];
          c.margin_s = c.sweep_time_s - c.see_time_s;
          break;
        }
      }

      const bool violation =
          !c.nominally_visible || c.margin_s < min_margin_s_;
      if (violation)
      {
        violation_count += 1;
      }
      candidates.push_back(c);
    }

    const Candidate* selected = nullptr;
    for (const auto& c : candidates)
    {
      const bool violation =
          !c.nominally_visible || c.margin_s < min_margin_s_;
      if (!violation)
      {
        continue;
      }
      if (!selected || betterCandidate(c, *selected))
      {
        selected = &c;
      }
    }

    if (!selected)
    {
      publishNoTargetSummary(
          "all_low_confidence_points_seen_before_deadline",
          static_cast<int>(candidates.size()),
          violation_count);
      ROS_INFO_THROTTLE(
          1.0,
          "[trajectory_vbc] no intervention: all %zu low-confidence sweep candidates have VBC margin >= %.3fs",
          candidates.size(),
          min_margin_s_);
      return;
    }

    geometry_msgs::PointStamped target_msg;
    target_msg.header.stamp = ros::Time::now();
    target_msg.header.frame_id = base_frame_;
    target_msg.point.x = selected->point_base.x();
    target_msg.point.y = selected->point_base.y();
    target_msg.point.z = selected->point_base.z();
    target_pub_.publish(target_msg);

    std_msgs::Float32 sweep_msg;
    sweep_msg.data = static_cast<float>(selected->sweep_time_s);
    selected_sweep_time_pub_.publish(sweep_msg);

    std_msgs::Float32 see_msg;
    see_msg.data = selected->nominally_visible
        ? static_cast<float>(selected->see_time_s)
        : std::numeric_limits<float>::infinity();
    selected_see_time_pub_.publish(see_msg);

    std_msgs::Float32 margin_msg;
    margin_msg.data = selected->nominally_visible
        ? static_cast<float>(selected->margin_s)
        : -std::numeric_limits<float>::infinity();
    selected_margin_pub_.publish(margin_msg);

    std::ostringstream oss;
    oss << "vbc success=1"
        << " has_violation=1"
        << " candidate_count=" << candidates.size()
        << " violation_count=" << violation_count
        << " min_required_margin_s=" << min_margin_s_
        << " target=[" << selected->point_base.x() << ","
        << selected->point_base.y() << ","
        << selected->point_base.z() << "]"
        << " confidence=" << selected->confidence
        << " link=" << selected->link_name
        << " sweep_eval_t=" << selected->sweep_eval_timestep
        << " sweep_original_t=" << selected->sweep_original_timestep
        << " sweep_time_s=" << selected->sweep_time_s
        << " nominally_visible=" << selected->nominally_visible
        << " see_eval_t=" << selected->see_eval_timestep
        << " see_original_t=" << selected->see_original_timestep;
    if (selected->nominally_visible)
    {
      oss << " see_time_s=" << selected->see_time_s
          << " margin_s=" << selected->margin_s;
    }
    else
    {
      oss << " see_time_s=inf margin_s=-inf";
    }

    std_msgs::String summary_msg;
    summary_msg.data = oss.str();
    summary_pub_.publish(summary_msg);

    ROS_WARN_THROTTLE(
        1.0,
        "[trajectory_vbc] VBC VIOLATION target=[%.3f %.3f %.3f] sweep=%.3fs see=%s margin=%s required=%.3fs candidates=%zu violations=%d",
        selected->point_base.x(),
        selected->point_base.y(),
        selected->point_base.z(),
        selected->sweep_time_s,
        selected->nominally_visible
            ? std::to_string(selected->see_time_s).c_str()
            : "never",
        selected->nominally_visible
            ? std::to_string(selected->margin_s).c_str()
            : "-inf",
        min_margin_s_,
        candidates.size(),
        violation_count);
  }

  void publishNoTargetSummary(
      const std::string& reason,
      int candidate_count,
      int violation_count)
  {
    std::ostringstream oss;
    oss << "vbc success=1 has_violation=0"
        << " reason=" << reason
        << " candidate_count=" << candidate_count
        << " violation_count=" << violation_count
        << " min_required_margin_s=" << min_margin_s_;
    std_msgs::String msg;
    msg.data = oss.str();
    summary_pub_.publish(msg);
  }

  void trajectoryCallback(const trajectory_msgs::JointTrajectoryConstPtr& msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_traj_ = *msg;
    latest_traj_time_ = ros::Time::now();
    has_traj_ = true;
  }

  void timerCallback(const ros::TimerEvent&)
  {
    trajectory_msgs::JointTrajectory traj;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!has_traj_)
      {
        ROS_WARN_THROTTLE(
            2.0,
            "[trajectory_vbc] waiting for nominal trajectory on %s",
            input_trajectory_topic_.c_str());
        return;
      }
      traj = latest_traj_;
    }
    evaluate(traj);
  }

  void printSummary() const
  {
    ROS_INFO_STREAM("");
    ROS_INFO_STREAM("========== trajectory_vbc_selector ==========");
    ROS_INFO_STREAM("enabled: " << enabled_);
    ROS_INFO_STREAM("input trajectory: " << input_trajectory_topic_);
    ROS_INFO_STREAM("output target: " << output_target_topic_);
    ROS_INFO_STREAM("candidate resolution: " << candidate_resolution_);
    ROS_INFO_STREAM("frontier confidence threshold: "
                    << frontier_confidence_threshold_);
    ROS_INFO_STREAM("required VBC margin: " << min_margin_s_ << " s");
    ROS_INFO_STREAM("sensor model: FOV=" << horizontal_fov_deg_ << " x "
                    << vertical_fov_deg_ << " deg, range=["
                    << tof_min_range_ << "," << tof_max_range_
                    << "], forward=" << forward_axis_);
    ROS_INFO_STREAM("sensor frames:");
    for (const auto& frame : sensor_frames_)
    {
      ROS_INFO_STREAM("  " << frame);
    }
    ROS_INFO_STREAM("Interpretation: intervene only if nominal see-before-sweep "
                    "margin is below the required safety buffer.");
    ROS_INFO_STREAM("=============================================");
  }

private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  ros::Subscriber trajectory_sub_;
  ros::ServiceClient confidence_query_client_;
  ros::Publisher target_pub_;
  ros::Publisher summary_pub_;
  ros::Publisher selected_margin_pub_;
  ros::Publisher selected_sweep_time_pub_;
  ros::Publisher selected_see_time_pub_;
  ros::Timer timer_;

  care_confidence_map::TrajectoryRiskEvaluator evaluator_;

  std::mutex mutex_;
  trajectory_msgs::JointTrajectory latest_traj_;
  ros::Time latest_traj_time_;
  bool has_traj_ = false;

  bool enabled_ = true;
  double eval_rate_ = 20.0;
  int max_eval_timesteps_ = 50;
  double query_timeout_ = 0.10;
  double fallback_dt_ = 0.05;

  double frontier_confidence_threshold_ = 0.50;
  double candidate_resolution_ = 0.05;
  double min_margin_s_ = 0.30;

  std::string robot_urdf_file_;
  std::string body_samples_file_;
  std::string base_frame_ = "base_link";
  std::string input_trajectory_topic_ = "/care_planner/task_trajectory";
  std::string output_target_topic_ =
      "/care_planner/active_sensing/target_candidate";
  std::string confidence_query_service_ =
      "/care_planner/confidence_map/query";

  std::vector<std::string> ignored_risk_links_;
  std::vector<std::string> sensor_frames_;

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
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "trajectory_vbc_selector_node");

  TrajectoryVbcSelectorNode node;
  if (!node.initialize())
  {
    ROS_ERROR("[trajectory_vbc] initialization failed.");
    return 1;
  }

  ros::spin();
  return 0;
}
