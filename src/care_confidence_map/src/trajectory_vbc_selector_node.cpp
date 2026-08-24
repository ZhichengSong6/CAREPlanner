#include <care_confidence_map/trajectory_risk_evaluator.hpp>
#include <care_confidence_map/QueryConfidence.h>

#include <ros/ros.h>

#include <geometry_msgs/Point.h>
#include <geometry_msgs/PointStamped.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/String.h>
#include <trajectory_msgs/JointTrajectory.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

class TrajectoryVbcSelectorNode
{
public:
  TrajectoryVbcSelectorNode() : nh_(), pnh_("~") {}

  bool initialize()
  {
    loadParams();

    std::string error_msg;
    if (!evaluator_.initialize(
            robot_urdf_file_, body_samples_file_, base_frame_, &error_msg))
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
    if (!buildSensorBasis()) return false;

    if (candidate_resolution_ <= 0.0 || fallback_dt_ <= 0.0 ||
        predicted_trajectory_timeout_ <= 0.0 ||
        primary_frontier_window_s_ < 0.0 ||
        target_switch_hysteresis_s_ < 0.0 ||
        region_spatial_radius_m_ <= 0.0 ||
        region_sweep_window_s_ < 0.0 ||
        region_match_radius_m_ <= 0.0 ||
        region_representative_count_ < 1 ||
        tof_min_range_ < 0.0 || tof_max_range_ <= tof_min_range_ ||
        horizontal_fov_deg_ <= 0.0 || vertical_fov_deg_ <= 0.0)
    {
      ROS_ERROR("[trajectory_vbc] Invalid VBC geometry/timing/region parameters.");
      return false;
    }

    tan_half_h_fov_ =
        std::tan(0.5 * horizontal_fov_deg_ * M_PI / 180.0);
    tan_half_v_fov_ =
        std::tan(0.5 * vertical_fov_deg_ * M_PI / 180.0);

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

    trajectory_sub_ = nh_.subscribe(
        input_trajectory_topic_, 1,
        &TrajectoryVbcSelectorNode::trajectoryCallback, this);
    predicted_trajectory_sub_ = nh_.subscribe(
        predicted_trajectory_topic_, 1,
        &TrajectoryVbcSelectorNode::predictedTrajectoryCallback, this);

    target_pub_ = nh_.advertise<geometry_msgs::PointStamped>(
        output_target_topic_, 1, false);
    candidate_active_pub_ = nh_.advertise<std_msgs::Bool>(
        candidate_active_topic_, 1, true);
    summary_pub_ = nh_.advertise<std_msgs::String>(
        "/care_planner/trajectory_risk/vbc_summary", 1, true);
    selected_margin_pub_ = nh_.advertise<std_msgs::Float32>(
        "/care_planner/trajectory_risk/vbc_selected_margin_s", 1, true);
    selected_sweep_time_pub_ = nh_.advertise<std_msgs::Float32>(
        "/care_planner/trajectory_risk/vbc_selected_sweep_time_s", 1, true);
    selected_see_time_pub_ = nh_.advertise<std_msgs::Float32>(
        "/care_planner/trajectory_risk/vbc_selected_see_time_s", 1, true);
    selected_region_representatives_pub_ =
        nh_.advertise<std_msgs::Float64MultiArray>(
            selected_region_representatives_topic_, 1, true);

    publishCandidateActive(false);

    timer_ = nh_.createTimer(
        ros::Duration(1.0 / std::max(0.1, eval_rate_)),
        &TrajectoryVbcSelectorNode::timerCallback, this);

    printSummary();
    return true;
  }

private:
  enum class FrontierGroup
  {
    Primary,
    Secondary
  };

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
    double margin_s = -std::numeric_limits<double>::infinity();
  };

  using CandidateKey = std::tuple<long long, long long, long long>;

  struct RiskRegion
  {
    FrontierGroup group = FrontierGroup::Primary;
    std::vector<int> member_indices;
    std::vector<CandidateKey> member_keys;
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    double earliest_sweep_time_s = std::numeric_limits<double>::infinity();
    int anchor_index = -1;
    int distinct_link_count = 0;
  };

  static const char* groupName(FrontierGroup group)
  {
    return group == FrontierGroup::Primary ? "primary" : "secondary";
  }

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
        input_trajectory_topic_, "/care_planner/task_trajectory");
    pnh_.param<std::string>(
        "trajectory_vbc/predicted_trajectory_topic",
        predicted_trajectory_topic_, "/care_planner/mpc/predicted_trajectory");
    pnh_.param<std::string>(
        "trajectory_vbc/output_target_topic",
        output_target_topic_, "/care_planner/active_sensing/target_candidate");
    pnh_.param<std::string>(
        "trajectory_vbc/candidate_active_topic",
        candidate_active_topic_,
        "/care_planner/active_sensing/target_candidate_active");
    pnh_.param<std::string>(
        "trajectory_vbc/confidence_query_service",
        confidence_query_service_, "/care_planner/confidence_map/query");
    pnh_.param<std::string>(
        "trajectory_vbc/selected_region_representatives_topic",
        selected_region_representatives_topic_,
        "/care_planner/trajectory_risk/vbc_selected_region_representatives");

    pnh_.param("trajectory_vbc/enabled", enabled_, true);
    pnh_.param("trajectory_vbc/eval_rate", eval_rate_, 20.0);
    pnh_.param("trajectory_vbc/max_eval_timesteps", max_eval_timesteps_, 50);
    pnh_.param("trajectory_vbc/query_timeout", query_timeout_, 0.10);
    pnh_.param("trajectory_vbc/fallback_dt", fallback_dt_, 0.05);
    pnh_.param(
        "trajectory_vbc/prefer_predicted_trajectory",
        prefer_predicted_trajectory_, false);
    pnh_.param(
        "trajectory_vbc/predicted_trajectory_timeout",
        predicted_trajectory_timeout_, 0.20);
    pnh_.param(
        "trajectory_vbc/frontier_confidence_threshold",
        frontier_confidence_threshold_, 0.50);
    pnh_.param(
        "trajectory_vbc/candidate_resolution",
        candidate_resolution_, 0.05);
    pnh_.param(
        "trajectory_vbc/min_visibility_before_sweep_margin_s",
        min_margin_s_, 0.30);
    pnh_.param(
        "trajectory_vbc/primary_frontier_window_s",
        primary_frontier_window_s_, 0.25);
    pnh_.param(
        "trajectory_vbc/target_switch_hysteresis_s",
        target_switch_hysteresis_s_, 0.05);

    // C4.3 risk-region clustering. Link identity is deliberately NOT a hard
    // connectivity condition: nearby swept points from different links may be
    // one physical low-confidence region. Link names remain diagnostics only.
    pnh_.param(
        "trajectory_vbc/region_spatial_radius_m",
        region_spatial_radius_m_, 0.12);
    pnh_.param(
        "trajectory_vbc/region_sweep_window_s",
        region_sweep_window_s_, 0.15);
    pnh_.param(
        "trajectory_vbc/region_match_radius_m",
        region_match_radius_m_, 0.15);
    pnh_.param(
        "trajectory_vbc/region_representative_count",
        region_representative_count_, 3);

    pnh_.param<std::string>(
        "trajectory_vbc/sensor_model/forward_axis", forward_axis_, "z");
    pnh_.param(
        "trajectory_vbc/sensor_model/tof_min_range", tof_min_range_, 0.15);
    pnh_.param(
        "trajectory_vbc/sensor_model/tof_max_range", tof_max_range_, 0.75);
    pnh_.param(
        "trajectory_vbc/sensor_model/horizontal_fov_deg",
        horizontal_fov_deg_, 55.0);
    pnh_.param(
        "trajectory_vbc/sensor_model/vertical_fov_deg",
        vertical_fov_deg_, 72.0);

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
    if (forward_axis_ == "x") forward_dir_ = Eigen::Vector3d::UnitX();
    else if (forward_axis_ == "-x") forward_dir_ = -Eigen::Vector3d::UnitX();
    else if (forward_axis_ == "y") forward_dir_ = Eigen::Vector3d::UnitY();
    else if (forward_axis_ == "-y") forward_dir_ = -Eigen::Vector3d::UnitY();
    else if (forward_axis_ == "z") forward_dir_ = Eigen::Vector3d::UnitZ();
    else if (forward_axis_ == "-z") forward_dir_ = -Eigen::Vector3d::UnitZ();
    else
    {
      ROS_ERROR_STREAM("[trajectory_vbc] Invalid forward_axis: " << forward_axis_);
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
    return std::find(
               ignored_risk_links_.begin(), ignored_risk_links_.end(), link_name)
        != ignored_risk_links_.end();
  }

  std::vector<int> makeDownsampleIndices(int input_size) const
  {
    std::vector<int> indices;
    if (input_size <= 0) return indices;
    if (input_size == 1)
    {
      indices.push_back(0);
      return indices;
    }
    const int max_steps = std::max(2, max_eval_timesteps_);
    if (input_size <= max_steps)
    {
      for (int i = 0; i < input_size; ++i) indices.push_back(i);
      return indices;
    }
    indices.reserve(static_cast<std::size_t>(max_steps));
    for (int k = 0; k < max_steps; ++k)
    {
      const double s = static_cast<double>(k) /
                       static_cast<double>(max_steps - 1);
      int idx = static_cast<int>(
          std::round(s * static_cast<double>(input_size - 1)));
      idx = std::max(0, std::min(input_size - 1, idx));
      if (indices.empty() || indices.back() != idx) indices.push_back(idx);
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
      if (error_msg) *error_msg = "JointTrajectory has no names or points.";
      return false;
    }

    std::map<std::string, int> input_joint_index;
    for (int i = 0; i < static_cast<int>(traj.joint_names.size()); ++i)
      input_joint_index[traj.joint_names[static_cast<std::size_t>(i)]] = i;

    const auto& required_names = evaluator_.activeJointNames();
    if (static_cast<int>(required_names.size()) != evaluator_.nq())
    {
      if (error_msg) *error_msg = "Unexpected evaluator active-joint count.";
      return false;
    }

    std::vector<int> required_to_input(required_names.size(), -1);
    for (int i = 0; i < static_cast<int>(required_names.size()); ++i)
    {
      const auto it = input_joint_index.find(required_names[static_cast<std::size_t>(i)]);
      if (it == input_joint_index.end())
      {
        if (error_msg)
          *error_msg = "Trajectory missing joint: " +
              required_names[static_cast<std::size_t>(i)];
        return false;
      }
      required_to_input[static_cast<std::size_t>(i)] = it->second;
    }

    const std::vector<int> selected =
        makeDownsampleIndices(static_cast<int>(traj.points.size()));
    const bool has_timing = traj.points.size() > 1 &&
        traj.points.back().time_from_start.toSec() > 1e-9;
    for (const int original_index : selected)
    {
      const auto& pt = traj.points[static_cast<std::size_t>(original_index)];
      if (pt.positions.size() < traj.joint_names.size())
      {
        if (error_msg) *error_msg = "Trajectory point has insufficient positions.";
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
        t = eval_times_s->back();
      q_traj->push_back(q);
      original_indices->push_back(original_index);
      eval_times_s->push_back(t);
    }
    return !q_traj->empty();
  }

  bool queryConfidence(
      const care_confidence_map::TrajectorySampleResult& samples,
      care_confidence_map::QueryConfidence* srv)
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
      return false;
    if (!confidence_query_client_.call(*srv)) return false;
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
    if (depth < tof_min_range_ || depth > tof_max_range_) return false;
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
      if (pointVisibleFromSensor(point_base, pose)) return true;
    return false;
  }

  bool isViolation(const Candidate& c) const
  {
    return !c.nominally_visible || c.margin_s < min_margin_s_;
  }

  bool betterViolation(const Candidate& a, const Candidate& b) const
  {
    if (a.nominally_visible != b.nominally_visible)
      return !a.nominally_visible;
    if (!a.nominally_visible)
      return a.sweep_time_s < b.sweep_time_s;
    if (std::fabs(a.margin_s - b.margin_s) > 1e-9)
      return a.margin_s < b.margin_s;
    return a.sweep_time_s < b.sweep_time_s;
  }

  bool keepCurrentUnderHysteresis(
      const Candidate& current,
      const Candidate& best) const
  {
    if (candidateKey(current.point_base) == candidateKey(best.point_base))
      return true;
    if (current.nominally_visible != best.nominally_visible)
      return !current.nominally_visible;
    if (!current.nominally_visible)
      return current.sweep_time_s <=
          best.sweep_time_s + target_switch_hysteresis_s_;
    return current.margin_s <= best.margin_s + target_switch_hysteresis_s_;
  }

  FrontierGroup groupFor(const Candidate& c, double first_sweep_time_s) const
  {
    return c.sweep_time_s <=
            first_sweep_time_s + primary_frontier_window_s_ + 1e-9
        ? FrontierGroup::Primary
        : FrontierGroup::Secondary;
  }

  bool regionConnected(const Candidate& a, const Candidate& b) const
  {
    if ((a.point_base - b.point_base).norm() > region_spatial_radius_m_ + 1e-9)
      return false;
    return std::fabs(a.sweep_time_s - b.sweep_time_s) <=
        region_sweep_window_s_ + 1e-9;
  }

  std::vector<RiskRegion> buildRiskRegions(
      const std::vector<Candidate>& candidates,
      double first_sweep_time_s) const
  {
    const int n = static_cast<int>(candidates.size());
    std::vector<bool> visited(static_cast<std::size_t>(n), false);
    std::vector<RiskRegion> regions;

    for (int seed = 0; seed < n; ++seed)
    {
      if (visited[static_cast<std::size_t>(seed)] ||
          !isViolation(candidates[static_cast<std::size_t>(seed)]))
        continue;

      RiskRegion region;
      region.group = groupFor(
          candidates[static_cast<std::size_t>(seed)], first_sweep_time_s);
      std::queue<int> frontier;
      frontier.push(seed);
      visited[static_cast<std::size_t>(seed)] = true;

      while (!frontier.empty())
      {
        const int i = frontier.front();
        frontier.pop();
        const Candidate& ci = candidates[static_cast<std::size_t>(i)];
        region.member_indices.push_back(i);
        region.member_keys.push_back(candidateKey(ci.point_base));
        region.centroid += ci.point_base;
        region.earliest_sweep_time_s =
            std::min(region.earliest_sweep_time_s, ci.sweep_time_s);
        if (region.anchor_index < 0 ||
            betterViolation(
                ci, candidates[static_cast<std::size_t>(region.anchor_index)]))
          region.anchor_index = i;

        for (int j = 0; j < n; ++j)
        {
          if (visited[static_cast<std::size_t>(j)] ||
              !isViolation(candidates[static_cast<std::size_t>(j)]))
            continue;
          const Candidate& cj = candidates[static_cast<std::size_t>(j)];
          if (groupFor(cj, first_sweep_time_s) != region.group) continue;
          if (!regionConnected(ci, cj)) continue;
          visited[static_cast<std::size_t>(j)] = true;
          frontier.push(j);
        }
      }

      if (!region.member_indices.empty())
      {
        region.centroid /= static_cast<double>(region.member_indices.size());
        std::set<std::string> links;
        for (const int idx : region.member_indices)
          links.insert(candidates[static_cast<std::size_t>(idx)].link_name);
        region.distinct_link_count = static_cast<int>(links.size());
        regions.push_back(region);
      }
    }
    return regions;
  }

  int overlapCount(
      const RiskRegion& region,
      const std::vector<CandidateKey>& old_keys) const
  {
    int count = 0;
    for (const auto& key : region.member_keys)
      if (std::find(old_keys.begin(), old_keys.end(), key) != old_keys.end())
        ++count;
    return count;
  }

  int findPreviousRegion(
      const std::vector<RiskRegion>& regions,
      FrontierGroup desired_group) const
  {
    if (!has_selected_region_) return -1;

    int best_index = -1;
    int best_overlap = 0;
    double best_distance = std::numeric_limits<double>::infinity();
    for (int i = 0; i < static_cast<int>(regions.size()); ++i)
    {
      const auto& region = regions[static_cast<std::size_t>(i)];
      if (region.group != desired_group) continue;
      const int overlap = overlapCount(region, selected_region_keys_);
      const double distance = (region.centroid - selected_region_centroid_).norm();
      if (overlap > best_overlap ||
          (overlap == best_overlap && overlap > 0 && distance < best_distance))
      {
        best_index = i;
        best_overlap = overlap;
        best_distance = distance;
      }
    }
    if (best_overlap > 0) return best_index;

    best_index = -1;
    best_distance = region_match_radius_m_;
    for (int i = 0; i < static_cast<int>(regions.size()); ++i)
    {
      const auto& region = regions[static_cast<std::size_t>(i)];
      if (region.group != desired_group) continue;
      const double distance = (region.centroid - selected_region_centroid_).norm();
      if (distance <= best_distance)
      {
        best_index = i;
        best_distance = distance;
      }
    }
    return best_index;
  }

  bool betterRegion(
      const RiskRegion& a,
      const RiskRegion& b,
      const std::vector<Candidate>& candidates) const
  {
    const Candidate& ca = candidates[static_cast<std::size_t>(a.anchor_index)];
    const Candidate& cb = candidates[static_cast<std::size_t>(b.anchor_index)];
    if (betterViolation(ca, cb)) return true;
    if (betterViolation(cb, ca)) return false;
    return a.member_indices.size() > b.member_indices.size();
  }

  const Candidate* findCandidateByKey(
      const RiskRegion& region,
      const std::vector<Candidate>& candidates,
      const CandidateKey& key) const
  {
    for (const int idx : region.member_indices)
    {
      const Candidate& c = candidates[static_cast<std::size_t>(idx)];
      if (candidateKey(c.point_base) == key) return &c;
    }
    return nullptr;
  }

  const Candidate* regionMedoid(
      const RiskRegion& region,
      const std::vector<Candidate>& candidates) const
  {
    const Candidate* best = nullptr;
    double best_dist = std::numeric_limits<double>::infinity();
    for (const int idx : region.member_indices)
    {
      const Candidate& c = candidates[static_cast<std::size_t>(idx)];
      const double d = (c.point_base - region.centroid).norm();
      if (!best || d < best_dist)
      {
        best = &c;
        best_dist = d;
      }
    }
    return best;
  }

  std::vector<const Candidate*> regionRepresentatives(
      const RiskRegion& region,
      const std::vector<Candidate>& candidates,
      const Candidate* steering) const
  {
    std::vector<const Candidate*> reps;
    if (!steering) return reps;
    reps.push_back(steering);
    const int desired = std::min(
        region_representative_count_,
        static_cast<int>(region.member_indices.size()));

    while (static_cast<int>(reps.size()) < desired)
    {
      const Candidate* best = nullptr;
      double best_min_distance = -1.0;
      for (const int idx : region.member_indices)
      {
        const Candidate& c = candidates[static_cast<std::size_t>(idx)];
        if (std::find(reps.begin(), reps.end(), &c) != reps.end()) continue;
        double min_distance = std::numeric_limits<double>::infinity();
        for (const Candidate* r : reps)
          min_distance = std::min(
              min_distance, (c.point_base - r->point_base).norm());
        if (!best || min_distance > best_min_distance)
        {
          best = &c;
          best_min_distance = min_distance;
        }
      }
      if (!best) break;
      reps.push_back(best);
    }
    return reps;
  }

  void publishCandidateActive(bool active)
  {
    std_msgs::Bool msg;
    msg.data = active;
    candidate_active_pub_.publish(msg);
  }

  void publishRegionRepresentatives(
      const std::vector<const Candidate*>& reps)
  {
    std_msgs::Float64MultiArray msg;
    msg.data.reserve(reps.size() * 3);
    for (const Candidate* c : reps)
    {
      if (!c) continue;
      msg.data.push_back(c->point_base.x());
      msg.data.push_back(c->point_base.y());
      msg.data.push_back(c->point_base.z());
    }
    selected_region_representatives_pub_.publish(msg);
  }

  void clearStickySelection()
  {
    has_selected_region_ = false;
    selected_region_keys_.clear();
    selected_region_centroid_.setZero();
    has_selected_steering_key_ = false;
    selected_group_ = FrontierGroup::Primary;
  }

  void publishNoTargetSummary(
      const std::string& reason,
      const std::string& trajectory_source,
      int candidate_count,
      int primary_count,
      int secondary_count,
      int primary_violation_count,
      int secondary_violation_count)
  {
    publishCandidateActive(false);
    publishRegionRepresentatives(std::vector<const Candidate*>());
    clearStickySelection();

    std::ostringstream oss;
    oss << "vbc success=1 has_violation=0"
        << " reason=" << reason
        << " trajectory_source=" << trajectory_source
        << " candidate_count=" << candidate_count
        << " primary_count=" << primary_count
        << " secondary_count=" << secondary_count
        << " primary_violation_count=" << primary_violation_count
        << " secondary_violation_count=" << secondary_violation_count
        << " violation_count="
        << primary_violation_count + secondary_violation_count
        << " region_count=0 primary_region_count=0 secondary_region_count=0"
        << " min_required_margin_s=" << min_margin_s_;
    std_msgs::String msg;
    msg.data = oss.str();
    summary_pub_.publish(msg);
  }

  void publishSelected(
      const Candidate& selected,
      const RiskRegion& selected_region,
      const std::vector<const Candidate*>& representatives,
      const std::string& selection_reason,
      const std::string& trajectory_source,
      const ros::Time& trajectory_epoch,
      int candidate_count,
      int primary_count,
      int secondary_count,
      int primary_violation_count,
      int secondary_violation_count,
      int region_count,
      int primary_region_count,
      int secondary_region_count)
  {
    geometry_msgs::PointStamped target_msg;
    target_msg.header.stamp = trajectory_epoch.isZero()
        ? ros::Time::now() : trajectory_epoch;
    target_msg.header.frame_id = base_frame_;
    target_msg.point.x = selected.point_base.x();
    target_msg.point.y = selected.point_base.y();
    target_msg.point.z = selected.point_base.z();
    target_pub_.publish(target_msg);

    std_msgs::Float32 sweep_msg;
    sweep_msg.data = static_cast<float>(selected.sweep_time_s);
    selected_sweep_time_pub_.publish(sweep_msg);
    std_msgs::Float32 see_msg;
    see_msg.data = selected.nominally_visible
        ? static_cast<float>(selected.see_time_s)
        : std::numeric_limits<float>::infinity();
    selected_see_time_pub_.publish(see_msg);
    std_msgs::Float32 margin_msg;
    margin_msg.data = selected.nominally_visible
        ? static_cast<float>(selected.margin_s)
        : -std::numeric_limits<float>::infinity();
    selected_margin_pub_.publish(margin_msg);
    publishRegionRepresentatives(representatives);
    publishCandidateActive(true);

    std::ostringstream oss;
    oss << "vbc success=1 has_violation=1"
        << " reason=selected_region"
        << " trajectory_source=" << trajectory_source
        << " candidate_count=" << candidate_count
        << " primary_count=" << primary_count
        << " secondary_count=" << secondary_count
        << " primary_violation_count=" << primary_violation_count
        << " secondary_violation_count=" << secondary_violation_count
        << " violation_count="
        << primary_violation_count + secondary_violation_count
        << " region_count=" << region_count
        << " primary_region_count=" << primary_region_count
        << " secondary_region_count=" << secondary_region_count
        << " selected_group=" << groupName(selected_region.group)
        << " selection_reason=" << selection_reason
        << " selected_region_size=" << selected_region.member_indices.size()
        << " selected_region_link_count=" << selected_region.distinct_link_count
        << " selected_region_cross_link="
        << static_cast<int>(selected_region.distinct_link_count > 1)
        << " selected_region_centroid=["
        << selected_region.centroid.x() << ","
        << selected_region.centroid.y() << ","
        << selected_region.centroid.z() << "]"
        << " selected_region_earliest_sweep_s="
        << selected_region.earliest_sweep_time_s
        << " selected_region_rep_count=" << representatives.size()
        << " min_required_margin_s=" << min_margin_s_
        << " target=[" << selected.point_base.x() << ","
        << selected.point_base.y() << ","
        << selected.point_base.z() << "]"
        << " confidence=" << selected.confidence
        << " link=" << selected.link_name
        << " sample_index=" << selected.sample_index_in_link
        << " sweep_eval_t=" << selected.sweep_eval_timestep
        << " sweep_original_t=" << selected.sweep_original_timestep
        << " sweep_time_s=" << selected.sweep_time_s
        << " nominally_visible=" << selected.nominally_visible
        << " see_eval_t=" << selected.see_eval_timestep
        << " see_original_t=" << selected.see_original_timestep;
    if (selected.nominally_visible)
      oss << " see_time_s=" << selected.see_time_s
          << " margin_s=" << selected.margin_s;
    else
      oss << " see_time_s=inf margin_s=-inf";

    std_msgs::String summary_msg;
    summary_msg.data = oss.str();
    summary_pub_.publish(summary_msg);

    ROS_WARN_STREAM_THROTTLE(
        1.0,
        "[trajectory_vbc] " << groupName(selected_region.group)
            << " risk region size=" << selected_region.member_indices.size()
            << " links=" << selected_region.distinct_link_count
            << " target=[" << selected.point_base.x() << " "
            << selected.point_base.y() << " " << selected.point_base.z() << "]"
            << " reps=" << representatives.size()
            << " source=" << trajectory_source
            << " selection=" << selection_reason);
  }

  void evaluate(
      const trajectory_msgs::JointTrajectory& traj,
      const std::string& trajectory_source)
  {
    if (!enabled_)
    {
      publishNoTargetSummary("disabled", trajectory_source, 0, 0, 0, 0, 0);
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
          2.0, "[trajectory_vbc] body-sweep FK failed: "
                   << sample_result.message);
      return;
    }

    care_confidence_map::QueryConfidence confidence_srv;
    if (!queryConfidence(sample_result, &confidence_srv))
    {
      ROS_WARN_THROTTLE(
          2.0, "[trajectory_vbc] confidence query failed or returned invalid sizes");
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
        const double confidence = static_cast<double>(
            confidence_srv.response.confidence[flat_index]);
        const double current_visibility = static_cast<double>(
            confidence_srv.response.current_visibility[flat_index]);
        flat_index += 1;

        if (isIgnoredRiskLink(sample.link_name) || !inside) continue;
        if (confidence > frontier_confidence_threshold_ ||
            current_visibility > 0.5) continue;

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
          c.sweep_original_timestep =
              original_indices[static_cast<std::size_t>(k)];
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
      publishNoTargetSummary(
          "no_low_confidence_sweep_candidates",
          trajectory_source, 0, 0, 0, 0, 0);
      return;
    }

    std::vector<std::vector<care_confidence_map::FramePoseInBase>> sensor_poses(
        q_traj.size());
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

    double first_sweep_time_s = std::numeric_limits<double>::infinity();
    for (const auto& kv : candidate_map)
      first_sweep_time_s = std::min(first_sweep_time_s, kv.second.sweep_time_s);

    int primary_count = 0;
    int secondary_count = 0;
    int primary_violation_count = 0;
    int secondary_violation_count = 0;
    std::vector<Candidate> candidates;
    candidates.reserve(candidate_map.size());

    for (const auto& kv : candidate_map)
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
      const FrontierGroup group = groupFor(c, first_sweep_time_s);
      if (group == FrontierGroup::Primary)
      {
        ++primary_count;
        if (isViolation(c)) ++primary_violation_count;
      }
      else
      {
        ++secondary_count;
        if (isViolation(c)) ++secondary_violation_count;
      }
      candidates.push_back(c);
    }

    if (primary_violation_count + secondary_violation_count == 0)
    {
      publishNoTargetSummary(
          "all_low_confidence_points_seen_before_deadline",
          trajectory_source,
          static_cast<int>(candidates.size()),
          primary_count, secondary_count,
          primary_violation_count, secondary_violation_count);
      return;
    }

    std::vector<RiskRegion> regions =
        buildRiskRegions(candidates, first_sweep_time_s);
    int primary_region_count = 0;
    int secondary_region_count = 0;
    int best_primary = -1;
    int best_secondary = -1;
    for (int i = 0; i < static_cast<int>(regions.size()); ++i)
    {
      const auto& r = regions[static_cast<std::size_t>(i)];
      if (r.group == FrontierGroup::Primary)
      {
        ++primary_region_count;
        if (best_primary < 0 || betterRegion(
                r, regions[static_cast<std::size_t>(best_primary)], candidates))
          best_primary = i;
      }
      else
      {
        ++secondary_region_count;
        if (best_secondary < 0 || betterRegion(
                r, regions[static_cast<std::size_t>(best_secondary)], candidates))
          best_secondary = i;
      }
    }

    int selected_region_index = best_primary >= 0 ? best_primary : best_secondary;
    if (selected_region_index < 0)
    {
      publishNoTargetSummary(
          "no_violation_regions", trajectory_source,
          static_cast<int>(candidates.size()),
          primary_count, secondary_count,
          primary_violation_count, secondary_violation_count);
      return;
    }

    const FrontierGroup desired_group =
        best_primary >= 0 ? FrontierGroup::Primary : FrontierGroup::Secondary;
    std::string selection_reason = "best_new_region";
    const int previous_region_index = findPreviousRegion(regions, desired_group);
    if (previous_region_index >= 0)
    {
      const auto& previous = regions[static_cast<std::size_t>(previous_region_index)];
      const auto& best = regions[static_cast<std::size_t>(selected_region_index)];
      const Candidate& previous_anchor =
          candidates[static_cast<std::size_t>(previous.anchor_index)];
      const Candidate& best_anchor =
          candidates[static_cast<std::size_t>(best.anchor_index)];
      if (keepCurrentUnderHysteresis(previous_anchor, best_anchor))
      {
        selected_region_index = previous_region_index;
        selection_reason = "retained_region_hysteresis";
      }
      else
      {
        selection_reason = "region_risk_preempt";
      }
    }
    else if (has_selected_region_)
    {
      if (selected_group_ == FrontierGroup::Secondary &&
          desired_group == FrontierGroup::Primary)
        selection_reason = "primary_region_preempt";
      else
        selection_reason = "previous_region_left_set";
    }

    const RiskRegion& selected_region =
        regions[static_cast<std::size_t>(selected_region_index)];

    const Candidate* steering = nullptr;
    if (has_selected_steering_key_)
      steering = findCandidateByKey(
          selected_region, candidates, selected_steering_key_);
    if (!steering)
      steering = regionMedoid(selected_region, candidates);
    if (!steering)
      steering = &candidates[static_cast<std::size_t>(selected_region.anchor_index)];

    selected_region_keys_ = selected_region.member_keys;
    selected_region_centroid_ = selected_region.centroid;
    has_selected_region_ = true;
    selected_group_ = selected_region.group;
    selected_steering_key_ = candidateKey(steering->point_base);
    has_selected_steering_key_ = true;

    const std::vector<const Candidate*> representatives =
        regionRepresentatives(selected_region, candidates, steering);

    ros::Time epoch = traj.header.stamp;
    if (epoch.isZero()) epoch = ros::Time::now();
    publishSelected(
        *steering, selected_region, representatives,
        selection_reason, trajectory_source, epoch,
        static_cast<int>(candidates.size()),
        primary_count, secondary_count,
        primary_violation_count, secondary_violation_count,
        static_cast<int>(regions.size()),
        primary_region_count, secondary_region_count);
  }

  void trajectoryCallback(const trajectory_msgs::JointTrajectoryConstPtr& msg)
  {
    if (!msg || msg->points.empty()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    latest_traj_ = *msg;
    has_traj_ = true;
  }

  void predictedTrajectoryCallback(
      const trajectory_msgs::JointTrajectoryConstPtr& msg)
  {
    if (!msg || msg->points.empty()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    latest_predicted_traj_ = *msg;
    predicted_traj_received_ = ros::Time::now();
    has_predicted_traj_ = true;
  }

  void timerCallback(const ros::TimerEvent&)
  {
    trajectory_msgs::JointTrajectory traj;
    std::string source = "bootstrap";
    const ros::Time now = ros::Time::now();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const bool predicted_fresh =
          prefer_predicted_trajectory_ && has_predicted_traj_ &&
          (now - predicted_traj_received_).toSec() >= 0.0 &&
          (now - predicted_traj_received_).toSec() <=
              predicted_trajectory_timeout_;
      if (predicted_fresh)
      {
        traj = latest_predicted_traj_;
        source = "predicted";
      }
      else if (has_traj_)
      {
        traj = latest_traj_;
        source = "bootstrap";
      }
      else
      {
        ROS_WARN_THROTTLE(
            2.0,
            "[trajectory_vbc] waiting for bootstrap trajectory on %s",
            input_trajectory_topic_.c_str());
        return;
      }
    }
    evaluate(traj, source);
  }

  void printSummary() const
  {
    ROS_INFO_STREAM("");
    ROS_INFO_STREAM("========== trajectory_vbc_selector ==========");
    ROS_INFO_STREAM("enabled: " << enabled_);
    ROS_INFO_STREAM("bootstrap trajectory: " << input_trajectory_topic_);
    ROS_INFO_STREAM("predicted trajectory: " << predicted_trajectory_topic_);
    ROS_INFO_STREAM("prefer predicted: " << prefer_predicted_trajectory_
                    << ", timeout=" << predicted_trajectory_timeout_ << " s");
    ROS_INFO_STREAM("candidate resolution: " << candidate_resolution_);
    ROS_INFO_STREAM("primary frontier window: "
                    << primary_frontier_window_s_ << " s");
    ROS_INFO_STREAM("region connectivity: spatial<="
                    << region_spatial_radius_m_ << " m, |dt_sweep|<="
                    << region_sweep_window_s_ << " s");
    ROS_INFO_STREAM("region matching radius: " << region_match_radius_m_ << " m");
    ROS_INFO_STREAM("region representative count: "
                    << region_representative_count_);
    ROS_INFO_STREAM("target-switch hysteresis: "
                    << target_switch_hysteresis_s_ << " s");
    ROS_INFO_STREAM("required VBC margin: " << min_margin_s_ << " s");
    ROS_INFO_STREAM("Rule: audit every candidate; cluster violating candidates "
                    "by spatiotemporal connectivity within Primary/Secondary; "
                    "body-link identity is diagnostic, not a hard cluster boundary; "
                    "steer one sticky representative from the selected risk region.");
    ROS_INFO_STREAM("=============================================");
  }

private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber trajectory_sub_;
  ros::Subscriber predicted_trajectory_sub_;
  ros::ServiceClient confidence_query_client_;
  ros::Publisher target_pub_;
  ros::Publisher candidate_active_pub_;
  ros::Publisher summary_pub_;
  ros::Publisher selected_margin_pub_;
  ros::Publisher selected_sweep_time_pub_;
  ros::Publisher selected_see_time_pub_;
  ros::Publisher selected_region_representatives_pub_;
  ros::Timer timer_;

  care_confidence_map::TrajectoryRiskEvaluator evaluator_;
  std::mutex mutex_;
  trajectory_msgs::JointTrajectory latest_traj_;
  trajectory_msgs::JointTrajectory latest_predicted_traj_;
  ros::Time predicted_traj_received_;
  bool has_traj_ = false;
  bool has_predicted_traj_ = false;

  bool has_selected_region_ = false;
  std::vector<CandidateKey> selected_region_keys_;
  Eigen::Vector3d selected_region_centroid_ = Eigen::Vector3d::Zero();
  bool has_selected_steering_key_ = false;
  CandidateKey selected_steering_key_;
  FrontierGroup selected_group_ = FrontierGroup::Primary;

  bool enabled_ = true;
  bool prefer_predicted_trajectory_ = false;
  double eval_rate_ = 20.0;
  int max_eval_timesteps_ = 50;
  double query_timeout_ = 0.10;
  double fallback_dt_ = 0.05;
  double predicted_trajectory_timeout_ = 0.20;
  double frontier_confidence_threshold_ = 0.50;
  double candidate_resolution_ = 0.05;
  double min_margin_s_ = 0.30;
  double primary_frontier_window_s_ = 0.25;
  double target_switch_hysteresis_s_ = 0.05;
  double region_spatial_radius_m_ = 0.12;
  double region_sweep_window_s_ = 0.15;
  double region_match_radius_m_ = 0.15;
  int region_representative_count_ = 3;

  std::string robot_urdf_file_;
  std::string body_samples_file_;
  std::string base_frame_ = "base_link";
  std::string input_trajectory_topic_ = "/care_planner/task_trajectory";
  std::string predicted_trajectory_topic_ =
      "/care_planner/mpc/predicted_trajectory";
  std::string output_target_topic_ =
      "/care_planner/active_sensing/target_candidate";
  std::string candidate_active_topic_ =
      "/care_planner/active_sensing/target_candidate_active";
  std::string confidence_query_service_ =
      "/care_planner/confidence_map/query";
  std::string selected_region_representatives_topic_ =
      "/care_planner/trajectory_risk/vbc_selected_region_representatives";

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
