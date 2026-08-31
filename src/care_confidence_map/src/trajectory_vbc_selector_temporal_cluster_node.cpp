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
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

// C4.3 selector with bounded temporal clustering.
//
// Safety semantics remain point-wise: every low-confidence future body-sweep
// candidate is evaluated independently with analytic VBC.  Steering semantics
// are coarser: violating candidates are first grouped into ordered temporal
// layers, each layer spanning at most temporal_layer_mpc_steps original MPC
// intervals.  Every temporal layer is then spatially clustered only for
// diagnostics / physical interpretation.  ALL points from ALL spatial regions
// in the earliest unsafe temporal layer are published as one learned-steering
// active set.
class TrajectoryVbcTemporalClusterNode
{
public:
  TrajectoryVbcTemporalClusterNode() : nh_(), pnh_("~") {}

  bool initialize()
  {
    loadParams();

    std::string error_msg;
    if (!evaluator_.initialize(
            robot_urdf_file_, body_samples_file_, base_frame_, &error_msg))
    {
      ROS_ERROR_STREAM(
          "[trajectory_vbc_temporal] Failed to initialize evaluator: "
          << error_msg);
      return false;
    }
    if (sensor_frames_.empty())
    {
      ROS_ERROR(
          "[trajectory_vbc_temporal] trajectory_vbc/sensor_frames is empty.");
      return false;
    }
    if (!buildSensorBasis()) return false;

    if (candidate_resolution_ <= 0.0 || fallback_dt_ <= 0.0 ||
        predicted_trajectory_timeout_ <= 0.0 ||
        temporal_layer_mpc_steps_ < 0 || region_max_diameter_m_ <= 0.0 ||
        tof_min_range_ < 0.0 || tof_max_range_ <= tof_min_range_ ||
        horizontal_fov_deg_ <= 0.0 || vertical_fov_deg_ <= 0.0)
    {
      ROS_ERROR(
          "[trajectory_vbc_temporal] Invalid geometry/timing/cluster parameters.");
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
      ROS_ERROR_STREAM(
          "[trajectory_vbc_temporal] Invalid sensor frame configuration: "
          << error_msg);
      return false;
    }

    confidence_query_client_ =
        nh_.serviceClient<care_confidence_map::QueryConfidence>(
            confidence_query_service_);

    trajectory_sub_ = nh_.subscribe(
        input_trajectory_topic_, 1,
        &TrajectoryVbcTemporalClusterNode::trajectoryCallback, this);
    predicted_trajectory_sub_ = nh_.subscribe(
        predicted_trajectory_topic_, 1,
        &TrajectoryVbcTemporalClusterNode::predictedTrajectoryCallback, this);
    force_bootstrap_sub_ = nh_.subscribe(
        force_bootstrap_topic_, 1,
        &TrajectoryVbcTemporalClusterNode::forceBootstrapCallback, this);

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
    active_set_points_pub_ = nh_.advertise<std_msgs::Float64MultiArray>(
        active_set_points_topic_, 1, true);

    publishCandidateActive(false);
    publishActiveSet({});

    timer_ = nh_.createTimer(
        ros::Duration(1.0 / std::max(0.1, eval_rate_)),
        &TrajectoryVbcTemporalClusterNode::timerCallback, this);

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
    double margin_s = -std::numeric_limits<double>::infinity();
  };

  using CandidateKey = std::tuple<long long, long long, long long>;

  struct SpatialRegion
  {
    std::vector<int> member_indices;
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    double diameter_m = 0.0;
    int distinct_link_count = 0;
  };

  struct TemporalLayer
  {
    int start_eval_timestep = -1;
    int end_eval_timestep = -1;
    int start_original_timestep = -1;
    int end_original_timestep = -1;
    double min_sweep_time_s = std::numeric_limits<double>::infinity();
    double max_sweep_time_s = -std::numeric_limits<double>::infinity();
    std::vector<int> member_indices;
    std::vector<SpatialRegion> regions;
  };

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
        "trajectory_vbc/active_set_points_topic",
        active_set_points_topic_,
        "/care_planner/trajectory_risk/vbc_active_set_points");
    pnh_.param<std::string>(
        "trajectory_vbc/force_bootstrap_topic",
        force_bootstrap_topic_,
        "/care_planner/trajectory_risk/force_bootstrap");

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
        "trajectory_vbc/temporal_layer_mpc_steps",
        temporal_layer_mpc_steps_, 2);
    pnh_.param(
        "trajectory_vbc/region_max_diameter_m",
        region_max_diameter_m_, 0.12);

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
      ROS_ERROR_STREAM(
          "[trajectory_vbc_temporal] Invalid forward_axis: " << forward_axis_);
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
      indices.reserve(static_cast<std::size_t>(input_size));
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

    q_traj->reserve(selected.size());
    original_indices->reserve(selected.size());
    eval_times_s->reserve(selected.size());
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

  double regionDiameter(
      const std::vector<int>& members,
      const std::vector<Candidate>& candidates) const
  {
    double diameter = 0.0;
    for (std::size_t a = 0; a < members.size(); ++a)
    {
      for (std::size_t b = a + 1; b < members.size(); ++b)
      {
        diameter = std::max(
            diameter,
            (candidates[static_cast<std::size_t>(members[a])].point_base -
             candidates[static_cast<std::size_t>(members[b])].point_base).norm());
      }
    }
    return diameter;
  }

  SpatialRegion finalizeRegion(
      const std::vector<int>& members,
      const std::vector<Candidate>& candidates) const
  {
    SpatialRegion region;
    region.member_indices = members;
    std::set<std::string> links;
    for (const int idx : members)
    {
      const Candidate& c = candidates[static_cast<std::size_t>(idx)];
      region.centroid += c.point_base;
      links.insert(c.link_name);
    }
    if (!members.empty())
      region.centroid /= static_cast<double>(members.size());
    region.diameter_m = regionDiameter(members, candidates);
    region.distinct_link_count = static_cast<int>(links.size());
    return region;
  }

  std::vector<SpatialRegion> clusterSpatially(
      const std::vector<int>& layer_members,
      const std::vector<Candidate>& candidates) const
  {
    std::vector<std::vector<int>> clusters;
    clusters.reserve(layer_members.size());
    for (const int idx : layer_members) clusters.push_back({idx});

    while (true)
    {
      int best_a = -1;
      int best_b = -1;
      double best_centroid_distance = std::numeric_limits<double>::infinity();

      std::vector<Eigen::Vector3d> centroids(
          clusters.size(), Eigen::Vector3d::Zero());
      for (std::size_t i = 0; i < clusters.size(); ++i)
      {
        for (const int idx : clusters[i])
          centroids[i] += candidates[static_cast<std::size_t>(idx)].point_base;
        centroids[i] /= static_cast<double>(clusters[i].size());
      }

      for (int a = 0; a < static_cast<int>(clusters.size()); ++a)
      {
        for (int b = a + 1; b < static_cast<int>(clusters.size()); ++b)
        {
          std::vector<int> merged = clusters[static_cast<std::size_t>(a)];
          merged.insert(
              merged.end(),
              clusters[static_cast<std::size_t>(b)].begin(),
              clusters[static_cast<std::size_t>(b)].end());
          if (regionDiameter(merged, candidates) > region_max_diameter_m_ + 1e-9)
            continue;
          const double d =
              (centroids[static_cast<std::size_t>(a)] -
               centroids[static_cast<std::size_t>(b)]).norm();
          if (d < best_centroid_distance)
          {
            best_centroid_distance = d;
            best_a = a;
            best_b = b;
          }
        }
      }

      if (best_a < 0 || best_b < 0) break;
      clusters[static_cast<std::size_t>(best_a)].insert(
          clusters[static_cast<std::size_t>(best_a)].end(),
          clusters[static_cast<std::size_t>(best_b)].begin(),
          clusters[static_cast<std::size_t>(best_b)].end());
      clusters.erase(clusters.begin() + best_b);
    }

    std::vector<SpatialRegion> regions;
    regions.reserve(clusters.size());
    for (const auto& members : clusters)
      regions.push_back(finalizeRegion(members, candidates));
    return regions;
  }

  std::vector<TemporalLayer> buildTemporalLayers(
      const std::vector<Candidate>& candidates) const
  {
    std::vector<int> violations;
    for (int i = 0; i < static_cast<int>(candidates.size()); ++i)
    {
      if (isViolation(candidates[static_cast<std::size_t>(i)]))
        violations.push_back(i);
    }

    std::sort(
        violations.begin(), violations.end(),
        [&candidates](int a, int b) {
          const Candidate& ca = candidates[static_cast<std::size_t>(a)];
          const Candidate& cb = candidates[static_cast<std::size_t>(b)];
          if (ca.sweep_original_timestep != cb.sweep_original_timestep)
            return ca.sweep_original_timestep < cb.sweep_original_timestep;
          if (std::fabs(ca.sweep_time_s - cb.sweep_time_s) > 1e-12)
            return ca.sweep_time_s < cb.sweep_time_s;
          return a < b;
        });

    std::vector<TemporalLayer> layers;
    std::size_t cursor = 0;
    while (cursor < violations.size())
    {
      const Candidate& first =
          candidates[static_cast<std::size_t>(violations[cursor])];
      const int start_original = first.sweep_original_timestep;
      const int max_original = start_original + temporal_layer_mpc_steps_;

      TemporalLayer layer;
      layer.start_original_timestep = start_original;
      layer.start_eval_timestep = first.sweep_eval_timestep;

      while (cursor < violations.size())
      {
        const int idx = violations[cursor];
        const Candidate& c = candidates[static_cast<std::size_t>(idx)];
        if (c.sweep_original_timestep > max_original) break;

        layer.member_indices.push_back(idx);
        layer.end_original_timestep = std::max(
            layer.end_original_timestep, c.sweep_original_timestep);
        layer.end_eval_timestep = std::max(
            layer.end_eval_timestep, c.sweep_eval_timestep);
        layer.min_sweep_time_s = std::min(
            layer.min_sweep_time_s, c.sweep_time_s);
        layer.max_sweep_time_s = std::max(
            layer.max_sweep_time_s, c.sweep_time_s);
        ++cursor;
      }

      layer.regions = clusterSpatially(layer.member_indices, candidates);
      layers.push_back(layer);
    }
    return layers;
  }

  const Candidate* chooseDiagnosticRepresentative(
      const TemporalLayer& active_layer,
      const std::vector<Candidate>& candidates,
      std::string* reason)
  {
    if (has_selected_key_)
    {
      for (const int idx : active_layer.member_indices)
      {
        const Candidate& c = candidates[static_cast<std::size_t>(idx)];
        if (candidateKey(c.point_base) == selected_key_)
        {
          if (reason) *reason = "retained_active_layer_rep";
          return &c;
        }
      }
    }

    const Candidate* best = nullptr;
    for (const int idx : active_layer.member_indices)
    {
      const Candidate& c = candidates[static_cast<std::size_t>(idx)];
      if (!best || betterViolation(c, *best)) best = &c;
    }
    if (reason) *reason = "active_layer_rep_new";
    return best;
  }

  void publishCandidateActive(bool active)
  {
    std_msgs::Bool msg;
    msg.data = active;
    candidate_active_pub_.publish(msg);
  }

  void publishActiveSet(const std::vector<const Candidate*>& points)
  {
    std::vector<const Candidate*> ordered = points;
    std::sort(
        ordered.begin(), ordered.end(),
        [this](const Candidate* a, const Candidate* b) {
          return candidateKey(a->point_base) < candidateKey(b->point_base);
        });

    std_msgs::Float64MultiArray msg;
    msg.data.reserve(ordered.size() * 3);
    for (const Candidate* c : ordered)
    {
      if (!c) continue;
      msg.data.push_back(c->point_base.x());
      msg.data.push_back(c->point_base.y());
      msg.data.push_back(c->point_base.z());
    }
    active_set_points_pub_.publish(msg);
  }

  static std::string joinInts(const std::vector<int>& values)
  {
    std::ostringstream oss;
    for (std::size_t i = 0; i < values.size(); ++i)
    {
      if (i) oss << ":";
      oss << values[i];
    }
    return oss.str();
  }

  void publishNoTargetSummary(
      const std::string& reason,
      const std::string& trajectory_source,
      int candidate_count)
  {
    publishCandidateActive(false);
    publishActiveSet({});
    has_selected_key_ = false;

    std::ostringstream oss;
    oss << "vbc success=1 has_violation=0"
        << " reason=" << reason
        << " trajectory_source=" << trajectory_source
        << " candidate_count=" << candidate_count
        << " primary_count=0 secondary_count=" << candidate_count
        << " primary_violation_count=0 secondary_violation_count=0"
        << " violation_count=0"
        << " temporal_layer_mpc_steps=" << temporal_layer_mpc_steps_
        << " temporal_layer_count=0 active_layer_index=-1"
        << " active_layer_start_original_t=-1 active_layer_end_original_t=-1"
        << " active_layer_step_span=-1"
        << " active_layer_sweep_time_min_s=nan active_layer_sweep_time_max_s=nan"
        << " active_layer_time_span_s=nan"
        << " active_layer_point_count=0 active_layer_region_count=0"
        << " active_layer_cross_link_region_count=0 active_set_point_count=0"
        << " layer_point_counts=none active_region_sizes=none"
        << " min_required_margin_s=" << min_margin_s_;
    std_msgs::String msg;
    msg.data = oss.str();
    summary_pub_.publish(msg);
  }

  void publishSelected(
      const Candidate& representative,
      const TemporalLayer& active_layer,
      const std::vector<TemporalLayer>& layers,
      const std::vector<Candidate>& candidates,
      const std::string& selection_reason,
      const std::string& trajectory_source,
      const ros::Time& trajectory_epoch)
  {
    geometry_msgs::PointStamped target_msg;
    target_msg.header.stamp = trajectory_epoch.isZero()
        ? ros::Time::now() : trajectory_epoch;
    target_msg.header.frame_id = base_frame_;
    target_msg.point.x = representative.point_base.x();
    target_msg.point.y = representative.point_base.y();
    target_msg.point.z = representative.point_base.z();
    target_pub_.publish(target_msg);

    // The steering deadline is tied to the earliest point in the active layer.
    std_msgs::Float32 sweep_msg;
    sweep_msg.data = static_cast<float>(active_layer.min_sweep_time_s);
    selected_sweep_time_pub_.publish(sweep_msg);

    std_msgs::Float32 see_msg;
    see_msg.data = representative.nominally_visible
        ? static_cast<float>(representative.see_time_s)
        : std::numeric_limits<float>::infinity();
    selected_see_time_pub_.publish(see_msg);

    std_msgs::Float32 margin_msg;
    margin_msg.data = representative.nominally_visible
        ? static_cast<float>(representative.margin_s)
        : -std::numeric_limits<float>::infinity();
    selected_margin_pub_.publish(margin_msg);

    std::vector<const Candidate*> active_points;
    active_points.reserve(active_layer.member_indices.size());
    for (const int idx : active_layer.member_indices)
      active_points.push_back(&candidates[static_cast<std::size_t>(idx)]);
    publishActiveSet(active_points);
    publishCandidateActive(true);

    int violation_count = 0;
    for (const auto& c : candidates)
      if (isViolation(c)) ++violation_count;
    const int active_count = static_cast<int>(active_layer.member_indices.size());

    std::vector<int> layer_sizes;
    layer_sizes.reserve(layers.size());
    for (const auto& layer : layers)
      layer_sizes.push_back(static_cast<int>(layer.member_indices.size()));

    std::vector<int> active_region_sizes;
    int active_cross_link_regions = 0;
    for (const auto& region : active_layer.regions)
    {
      active_region_sizes.push_back(static_cast<int>(region.member_indices.size()));
      if (region.distinct_link_count > 1) ++active_cross_link_regions;
    }

    const int step_span =
        active_layer.end_original_timestep - active_layer.start_original_timestep;
    const double time_span_s =
        active_layer.max_sweep_time_s - active_layer.min_sweep_time_s;

    std::ostringstream oss;
    oss << "vbc success=1 has_violation=1"
        << " reason=selected_temporal_cluster"
        << " trajectory_source=" << trajectory_source
        << " candidate_count=" << candidates.size()
        << " primary_count=" << active_count
        << " secondary_count="
        << std::max(0, static_cast<int>(candidates.size()) - active_count)
        << " primary_violation_count=" << active_count
        << " secondary_violation_count=" << std::max(0, violation_count - active_count)
        << " violation_count=" << violation_count
        << " selected_group=primary"
        << " selection_reason=" << selection_reason
        << " temporal_layer_mpc_steps=" << temporal_layer_mpc_steps_
        << " temporal_layer_count=" << layers.size()
        << " active_layer_index=0"
        << " active_layer_start_eval_t=" << active_layer.start_eval_timestep
        << " active_layer_end_eval_t=" << active_layer.end_eval_timestep
        << " active_layer_start_original_t=" << active_layer.start_original_timestep
        << " active_layer_end_original_t=" << active_layer.end_original_timestep
        << " active_layer_step_span=" << step_span
        << " active_layer_sweep_time_min_s=" << active_layer.min_sweep_time_s
        << " active_layer_sweep_time_max_s=" << active_layer.max_sweep_time_s
        << " active_layer_time_span_s=" << time_span_s
        << " active_layer_sweep_time_s=" << active_layer.min_sweep_time_s
        << " active_layer_point_count=" << active_count
        << " active_layer_region_count=" << active_layer.regions.size()
        << " active_layer_cross_link_region_count=" << active_cross_link_regions
        << " active_set_point_count=" << active_count
        << " layer_point_counts=" << joinInts(layer_sizes)
        << " active_region_sizes=" << joinInts(active_region_sizes)
        << " min_required_margin_s=" << min_margin_s_
        << " target=[" << representative.point_base.x() << ","
        << representative.point_base.y() << ","
        << representative.point_base.z() << "]"
        << " confidence=" << representative.confidence
        << " link=" << representative.link_name
        << " sample_index=" << representative.sample_index_in_link
        << " sweep_eval_t=" << representative.sweep_eval_timestep
        << " sweep_original_t=" << representative.sweep_original_timestep
        // Keep the legacy guard diagnostic key.  In temporal-cluster mode this
        // is intentionally the earliest sweep time of the whole active layer.
        << " sweep_time_s=" << active_layer.min_sweep_time_s
        << " nominally_visible=" << representative.nominally_visible
        << " see_eval_t=" << representative.see_eval_timestep
        << " see_original_t=" << representative.see_original_timestep;
    if (representative.nominally_visible)
      oss << " see_time_s=" << representative.see_time_s
          << " margin_s=" << representative.margin_s;
    else
      oss << " see_time_s=inf margin_s=-inf";

    std_msgs::String summary_msg;
    summary_msg.data = oss.str();
    summary_pub_.publish(summary_msg);

    ROS_WARN_STREAM_THROTTLE(
        1.0,
        "[trajectory_vbc_temporal] layers=" << layers.size()
            << " active_steps=" << active_layer.start_original_timestep
            << ".." << active_layer.end_original_timestep
            << " span=" << step_span
            << " active_points=" << active_count
            << " active_regions=" << active_layer.regions.size()
            << " region_sizes=" << joinInts(active_region_sizes)
            << " source=" << trajectory_source);
  }

  void evaluate(
      const trajectory_msgs::JointTrajectory& traj,
      const std::string& trajectory_source)
  {
    if (!enabled_)
    {
      publishNoTargetSummary("disabled", trajectory_source, 0);
      return;
    }

    std::vector<Eigen::VectorXd> q_traj;
    std::vector<int> original_indices;
    std::vector<double> eval_times_s;
    std::string error_msg;
    if (!convertTrajectory(
            traj, &q_traj, &original_indices, &eval_times_s, &error_msg))
    {
      ROS_WARN_STREAM_THROTTLE(2.0, "[trajectory_vbc_temporal] " << error_msg);
      return;
    }

    const care_confidence_map::TrajectorySampleResult sample_result =
        evaluator_.computeTrajectorySamples(q_traj);
    if (!sample_result.success)
    {
      ROS_WARN_STREAM_THROTTLE(
          2.0, "[trajectory_vbc_temporal] body-sweep FK failed: "
                   << sample_result.message);
      return;
    }

    care_confidence_map::QueryConfidence confidence_srv;
    if (!queryConfidence(sample_result, &confidence_srv))
    {
      ROS_WARN_THROTTLE(
          2.0,
          "[trajectory_vbc_temporal] confidence query failed or returned invalid sizes");
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
          "no_low_confidence_sweep_candidates", trajectory_source, 0);
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
            2.0, "[trajectory_vbc_temporal] sensor FK failed: " << error_msg);
        return;
      }
    }

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
      candidates.push_back(c);
    }

    const std::vector<TemporalLayer> layers = buildTemporalLayers(candidates);
    if (layers.empty())
    {
      publishNoTargetSummary(
          "all_low_confidence_points_seen_before_deadline",
          trajectory_source,
          static_cast<int>(candidates.size()));
      return;
    }

    const TemporalLayer& active_layer = layers.front();
    std::string selection_reason;
    const Candidate* representative =
        chooseDiagnosticRepresentative(active_layer, candidates, &selection_reason);
    if (!representative)
    {
      publishNoTargetSummary(
          "active_layer_has_no_representative",
          trajectory_source,
          static_cast<int>(candidates.size()));
      return;
    }

    selected_key_ = candidateKey(representative->point_base);
    has_selected_key_ = true;

    ros::Time epoch = traj.header.stamp;
    if (epoch.isZero()) epoch = ros::Time::now();
    publishSelected(
        *representative,
        active_layer,
        layers,
        candidates,
        selection_reason,
        trajectory_source,
        epoch);
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

  void forceBootstrapCallback(const std_msgs::BoolConstPtr& msg)
  {
    if (!msg) return;
    std::lock_guard<std::mutex> lock(mutex_);
    const bool changed = force_bootstrap_ != msg->data;
    force_bootstrap_ = msg->data;
    if (changed)
    {
      ROS_INFO_STREAM(
          "[trajectory_vbc_temporal] force_bootstrap="
          << static_cast<int>(force_bootstrap_));
    }
  }

  void timerCallback(const ros::TimerEvent&)
  {
    trajectory_msgs::JointTrajectory traj;
    std::string source = "bootstrap";
    const ros::Time now = ros::Time::now();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const bool predicted_fresh =
          !force_bootstrap_ &&
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
            "[trajectory_vbc_temporal] waiting for bootstrap trajectory on %s",
            input_trajectory_topic_.c_str());
        return;
      }
    }
    evaluate(traj, source);
  }

  void printSummary() const
  {
    ROS_INFO_STREAM("");
    ROS_INFO_STREAM(
        "========== trajectory_vbc_selector: bounded temporal clusters ==========");
    ROS_INFO_STREAM("enabled: " << enabled_);
    ROS_INFO_STREAM("bootstrap trajectory: " << input_trajectory_topic_);
    ROS_INFO_STREAM("predicted trajectory: " << predicted_trajectory_topic_);
    ROS_INFO_STREAM("prefer predicted: " << prefer_predicted_trajectory_
                    << ", timeout=" << predicted_trajectory_timeout_ << " s");
    ROS_INFO_STREAM("candidate resolution: " << candidate_resolution_ << " m");
    ROS_INFO_STREAM("temporal layer maximum span: "
                    << temporal_layer_mpc_steps_ << " MPC steps");
    ROS_INFO_STREAM("spatial region maximum diameter: "
                    << region_max_diameter_m_ << " m");
    ROS_INFO_STREAM("active set topic: " << active_set_points_topic_);
    ROS_INFO_STREAM("force bootstrap topic: " << force_bootstrap_topic_);
    ROS_INFO_STREAM("required VBC margin: " << min_margin_s_ << " s");
    ROS_INFO_STREAM(
        "Rule: audit every candidate; sort violations by t_sweep; greedily form "
        "ordered temporal layers with bounded MPC-step span; spatially cluster "
        "inside each layer; ALL regions and ALL points in the earliest unsafe "
        "layer form the steering active set.");
    ROS_INFO_STREAM(
        "=========================================================================");
  }

private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber trajectory_sub_;
  ros::Subscriber predicted_trajectory_sub_;
  ros::Subscriber force_bootstrap_sub_;
  ros::ServiceClient confidence_query_client_;
  ros::Publisher target_pub_;
  ros::Publisher candidate_active_pub_;
  ros::Publisher summary_pub_;
  ros::Publisher selected_margin_pub_;
  ros::Publisher selected_sweep_time_pub_;
  ros::Publisher selected_see_time_pub_;
  ros::Publisher active_set_points_pub_;
  ros::Timer timer_;

  care_confidence_map::TrajectoryRiskEvaluator evaluator_;
  std::mutex mutex_;
  trajectory_msgs::JointTrajectory latest_traj_;
  trajectory_msgs::JointTrajectory latest_predicted_traj_;
  ros::Time predicted_traj_received_;
  bool has_traj_ = false;
  bool has_predicted_traj_ = false;
  bool force_bootstrap_ = false;

  bool has_selected_key_ = false;
  CandidateKey selected_key_;

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
  int temporal_layer_mpc_steps_ = 2;
  double region_max_diameter_m_ = 0.12;

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
  std::string active_set_points_topic_ =
      "/care_planner/trajectory_risk/vbc_active_set_points";
  std::string force_bootstrap_topic_ =
      "/care_planner/trajectory_risk/force_bootstrap";

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
  TrajectoryVbcTemporalClusterNode node;
  if (!node.initialize())
  {
    ROS_ERROR("[trajectory_vbc_temporal] initialization failed.");
    return 1;
  }
  ros::spin();
  return 0;
}
