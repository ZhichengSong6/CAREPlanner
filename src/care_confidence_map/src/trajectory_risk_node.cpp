#include <care_confidence_map/trajectory_risk_evaluator.hpp>
#include <care_confidence_map/QueryConfidence.h>

#include <std_srvs/Trigger.h>

#include <ros/ros.h>

#include <std_msgs/Float32.h>
#include <std_msgs/Int32.h>
#include <std_msgs/String.h>

#include <geometry_msgs/Point.h>
#include <geometry_msgs/PointStamped.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

class TrajectoryRiskNode
{
public:
  TrajectoryRiskNode()
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
      ROS_ERROR_STREAM("[trajectory_risk_node] Failed to initialize evaluator: "
                       << error_msg);
      return false;
    }

    confidence_query_client_ =
        nh_.serviceClient<care_confidence_map::QueryConfidence>(
            confidence_query_service_);

    refresh_body_prior_client_ =
        nh_.serviceClient<std_srvs::Trigger>(
            refresh_body_prior_service_);

    trajectory_sub_ =
        nh_.subscribe(
            input_trajectory_topic_,
            1,
            &TrajectoryRiskNode::trajectoryCallback,
            this);

    score_pub_ =
        nh_.advertise<std_msgs::Float32>(
            outputTopic("score"), 1, true);

    mean_confidence_pub_ =
        nh_.advertise<std_msgs::Float32>(
            outputTopic("mean_confidence"), 1, true);

    min_confidence_pub_ =
        nh_.advertise<std_msgs::Float32>(
            outputTopic("min_confidence"), 1, true);

    visible_ratio_pub_ =
        nh_.advertise<std_msgs::Float32>(
            outputTopic("visible_ratio"), 1, true);

    worst_timestep_pub_ =
        nh_.advertise<std_msgs::Int32>(
            outputTopic("worst_timestep"), 1, true);

    summary_pub_ =
        nh_.advertise<std_msgs::String>(
            outputTopic("summary"), 1, true);

    timestep_summary_pub_ =
        nh_.advertise<std_msgs::String>(
            outputTopic("timestep_query_summary"), 1, true);

    worst_timestep_summary_pub_ =
        nh_.advertise<std_msgs::String>(
            outputTopic("worst_timestep_summary"), 1, true);

    full_trajectory_marker_pub_ =
        nh_.advertise<visualization_msgs::MarkerArray>(
            outputTopic("full_trajectory_markers"), 1, true);

    worst_timestep_marker_pub_ =
        nh_.advertise<visualization_msgs::MarkerArray>(
            outputTopic("worst_timestep_markers"), 1, true);

    attribution_marker_pub_ =
        nh_.advertise<visualization_msgs::MarkerArray>(
            outputTopic("attribution_markers"), 1, true);

    primary_frontier_marker_pub_ =
        nh_.advertise<visualization_msgs::MarkerArray>(
            outputTopic("primary_frontier_markers"), 1, true);

    secondary_frontier_marker_pub_ =
        nh_.advertise<visualization_msgs::MarkerArray>(
            outputTopic("secondary_frontier_markers"), 1, true);

    active_sensing_target_pub_ =
        nh_.advertise<geometry_msgs::PointStamped>(
            active_sensing_target_topic_, 1, false);

    eval_time_pub_ =
        nh_.advertise<std_msgs::Float32>(
            outputTopic("eval_time_ms"), 1, true);

    fk_time_pub_ =
        nh_.advertise<std_msgs::Float32>(
            outputTopic("fk_time_ms"), 1, true);

    query_time_pub_ =
        nh_.advertise<std_msgs::Float32>(
            outputTopic("query_time_ms"), 1, true);

    risk_compute_time_pub_ =
        nh_.advertise<std_msgs::Float32>(
            outputTopic("risk_compute_time_ms"), 1, true);

    marker_time_pub_ =
        nh_.advertise<std_msgs::Float32>(
            outputTopic("marker_time_ms"), 1, true);

    input_age_pub_ =
        nh_.advertise<std_msgs::Float32>(
            outputTopic("input_age_ms"), 1, true);

    timing_summary_pub_ =
        nh_.advertise<std_msgs::String>(
            outputTopic("timing_summary"), 1, true);

    attribution_summary_pub_ =
        nh_.advertise<std_msgs::String>(
            outputTopic("attribution_summary"), 1, true);

    timestep_attribution_pub_ =
        nh_.advertise<std_msgs::String>(
            outputTopic("timestep_attribution"), 1, true);

    link_attribution_pub_ =
        nh_.advertise<std_msgs::String>(
            outputTopic("link_attribution"), 1, true);

    topk_sample_attribution_pub_ =
        nh_.advertise<std_msgs::String>(
            outputTopic("topk_sample_attribution"), 1, true);

    risk_frontier_attribution_pub_ =
        nh_.advertise<std_msgs::String>(
            outputTopic("risk_frontier_attribution"), 1, true);

    if (forbidden_space_pair_publish_enabled_)
    {
      forbidden_space_pair_pub_ =
          nh_.advertise<sensor_msgs::PointCloud2>(
              forbidden_space_pair_topic_, 1, false);
    }

    if (!event_driven_eval_)
    {
      eval_timer_ =
          nh_.createTimer(
              ros::Duration(1.0 / std::max(0.1, eval_rate_)),
              &TrajectoryRiskNode::evalTimerCallback,
              this);
    }

    printSummary();
    return true;
  }

private:
  struct TimestepStats
  {
    int timestep_index = -1;
    int original_timestep_index = -1;

    int total_count = 0;
    int inside_count = 0;
    int outside_count = 0;
    int visible_count = 0;

    double mean_confidence = 0.0;
    double min_confidence = 0.0;
    double visible_ratio = 0.0;
    double risk = 1.0;

    std::string worst_link = "none";
    int worst_sample_index_in_link = -1;
    int worst_source_collision_index = -1;
    double worst_confidence = 1.0;
  };

  struct RiskResult
  {
    bool success = false;
    std::string message = "not evaluated";

    int input_num_timesteps = 0;
    int eval_num_timesteps = 0;
    int samples_per_timestep = 0;
    int total_samples = 0;

    int inside_count = 0;
    int outside_count = 0;
    int visible_count = 0;

    double mean_confidence = 0.0;
    double min_confidence = 0.0;
    double visible_ratio = 0.0;

    double score = 1.0;
    int worst_risk_timestep = -1;
    int worst_risk_original_timestep = -1;
    double worst_timestep_risk = 1.0;

    int worst_sample_timestep = -1;
    int worst_sample_original_timestep = -1;
    std::string worst_link = "none";
    int worst_sample_index_in_link = -1;
    int worst_source_collision_index = -1;
    double worst_confidence = 1.0;

    std::vector<int> eval_to_original_index;
    std::vector<TimestepStats> timestep_stats;
  };

  struct TimingStats
  {
    double input_age_ms = 0.0;
    double convert_time_ms = 0.0;
    double fk_time_ms = 0.0;
    double query_time_ms = 0.0;
    double risk_compute_time_ms = 0.0;
    double attribution_time_ms = 0.0;
    double marker_time_ms = 0.0;
    double total_time_ms = 0.0;
    bool published_markers = false;
  };

  struct LinkAttributionStats
  {
    std::string link_name = "none";

    int sample_count = 0;
    int inside_count = 0;
    int outside_count = 0;
    int visible_count = 0;

    double confidence_sum = 0.0;
    double mean_confidence = 0.0;
    double min_confidence = 0.0;
    double visible_ratio = 0.0;
    double gap = 1.0;
  };

  struct SampleAttributionItem
  {
    int eval_timestep = -1;
    int original_timestep = -1;

    std::string link_name = "none";
    int sample_index_in_link = -1;
    int source_collision_index = -1;

    double confidence = 0.0;
    double gap = 1.0;
    double current_visibility = 0.0;
    bool inside = false;

    Eigen::Vector3d center_base = Eigen::Vector3d::Zero();
    double radius = 0.01;
  };

  struct AttributionResult
  {
    bool success = false;
    std::string message = "not evaluated";

    double trajectory_gap = 1.0;

    int worst_eval_timestep = -1;
    int worst_original_timestep = -1;
    double worst_timestep_gap = 1.0;
    double worst_timestep_mean_confidence = 0.0;
    double worst_timestep_visible_ratio = 0.0;

    std::string worst_link_over_trajectory = "none";
    double worst_link_gap_over_trajectory = 1.0;
    double worst_link_mean_confidence_over_trajectory = 0.0;
    double worst_link_visible_ratio_over_trajectory = 0.0;

    std::string worst_link_at_worst_timestep = "none";
    double worst_link_gap_at_worst_timestep = 1.0;
    double worst_link_mean_confidence_at_worst_timestep = 0.0;
    double worst_link_visible_ratio_at_worst_timestep = 0.0;

    std::vector<LinkAttributionStats> link_stats_over_trajectory;
    std::vector<LinkAttributionStats> link_stats_at_worst_timestep;

    bool has_risk_frontier = false;
    int first_risky_eval_timestep = -1;
    int first_risky_original_timestep = -1;
    int safe_prefix_end_eval_timestep = -1;
    int safe_prefix_end_original_timestep = -1;
    int risk_frontier_window_start_eval = -1;
    int risk_frontier_window_end_eval = -1;
    int risk_frontier_window_start_original = -1;
    int risk_frontier_window_end_original = -1;
    double risk_frontier_threshold = 0.0;
    double risk_frontier_mean_gap = 0.0;
    double risk_frontier_total_weight = 0.0;
    Eigen::Vector3d risk_frontier_centroid_base = Eigen::Vector3d::Zero();
    double risk_frontier_radius = 0.0;

    std::vector<SampleAttributionItem> topk_low_confidence_samples;
    std::vector<SampleAttributionItem> worst_timestep_worst_link_samples;

    // Primary temporal frontier group: earliest time window where any
    // considered body/collision sample enters low-confidence space.
    // This is the required active-sensing target group.
    std::vector<SampleAttributionItem> risk_frontier_samples;
    std::vector<LinkAttributionStats> risk_frontier_link_stats;

    // Secondary temporal frontier group: all low-confidence samples after
    // the primary window. This is an optional long-horizon bonus.
    bool has_secondary_frontier = false;
    Eigen::Vector3d secondary_frontier_centroid_base = Eigen::Vector3d::Zero();
    double secondary_frontier_radius = 0.0;
    double secondary_frontier_mean_gap = 0.0;
    double secondary_frontier_total_weight = 0.0;
    int secondary_frontier_window_start_eval = -1;
    int secondary_frontier_window_end_eval = -1;
    int secondary_frontier_window_start_original = -1;
    int secondary_frontier_window_end_original = -1;
    std::vector<SampleAttributionItem> secondary_frontier_samples;
    std::vector<LinkAttributionStats> secondary_frontier_link_stats;
  };

  void loadParams()
  {
    pnh_.param<std::string>(
        "trajectory_risk/robot_urdf_file",
        robot_urdf_file_,
        "");

    pnh_.param<std::string>(
        "trajectory_risk/body_samples_file",
        body_samples_file_,
        "");

    pnh_.param<std::string>(
        "trajectory_risk/base_frame",
        base_frame_,
        "base_link");

    pnh_.param<std::string>(
        "trajectory_risk/input_trajectory_topic",
        input_trajectory_topic_,
        "/care_planner/task_trajectory");

    pnh_.param<std::string>(
        "trajectory_risk/active_sensing_target_topic",
        active_sensing_target_topic_,
        "/care_planner/active_sensing/target_point");

    pnh_.param<std::string>(
        "trajectory_risk/output_namespace",
        output_namespace_,
        "/care_planner/trajectory_risk");

    pnh_.param<std::string>(
        "trajectory_risk/confidence_query_service",
        confidence_query_service_,
        "/care_planner/confidence_map/query");

    pnh_.param<std::string>(
        "trajectory_risk/refresh_body_prior_service",
        refresh_body_prior_service_,
        "/care_planner/confidence_map/refresh_body_prior");

    pnh_.param(
        "trajectory_risk/refresh_body_prior_before_query",
        refresh_body_prior_before_query_,
        true);

    pnh_.param(
        "trajectory_risk/refresh_body_prior_timeout",
        refresh_body_prior_timeout_,
        0.10);

    pnh_.param(
        "trajectory_risk/eval_rate",
        eval_rate_,
        20.0);

    pnh_.param(
        "trajectory_risk/event_driven_eval",
        event_driven_eval_,
        false);

    pnh_.param(
        "trajectory_risk/max_eval_timesteps",
        max_eval_timesteps_,
        12);

    pnh_.param(
        "trajectory_risk/query_timeout",
        query_timeout_,
        0.10);

    pnh_.param(
        "trajectory_risk/marker_alpha",
        marker_alpha_,
        0.45);

    pnh_.param(
        "trajectory_risk/worst_marker_alpha",
        worst_marker_alpha_,
        0.85);

    pnh_.param(
        "trajectory_risk/attribution_marker_alpha",
        attribution_marker_alpha_,
        0.90);

    pnh_.param(
        "trajectory_risk/marker_publish_rate",
        marker_publish_rate_,
        2.0);

    pnh_.param(
        "trajectory_risk/show_full_trajectory_markers",
        show_full_trajectory_markers_,
        false);

    pnh_.param(
        "trajectory_risk/show_worst_timestep_markers",
        show_worst_timestep_markers_,
        true);

    pnh_.param(
        "trajectory_risk/show_attribution_markers",
        show_attribution_markers_,
        true);

    pnh_.param(
        "trajectory_risk/top_k_samples",
        top_k_samples_,
        20);

    pnh_.param(
        "trajectory_risk/risk_frontier_threshold",
        risk_frontier_threshold_,
        0.30);

    pnh_.param(
        "trajectory_risk/risk_frontier_window_steps",
        risk_frontier_window_steps_,
        5);

    pnh_.param(
        "trajectory_risk/safe_prefix_margin_steps",
        safe_prefix_margin_steps_,
        2);

    pnh_.param(
        "trajectory_risk/frontier_confidence_threshold",
        frontier_confidence_threshold_,
        0.50);

    pnh_.param(
        "trajectory_risk/frontier_gap_threshold",
        frontier_gap_threshold_,
        0.50);

    pnh_.param(
        "trajectory_risk/frontier_radius_margin",
        frontier_radius_margin_,
        0.05);

    pnh_.param(
        "trajectory_risk/evaluate_stale_trajectory",
        evaluate_stale_trajectory_,
        true);

    pnh_.param(
        "trajectory_risk/stale_trajectory_timeout",
        stale_trajectory_timeout_,
        1.0);

    pnh_.param(
        "trajectory_risk/forbidden_space_pair_publish_enabled",
        forbidden_space_pair_publish_enabled_,
        false);

    pnh_.param<std::string>(
        "trajectory_risk/forbidden_space_pair_topic",
        forbidden_space_pair_topic_,
        "/care_planner/trajectory_risk/body_sweep_anchors");

    pnh_.param(
        "trajectory_risk/forbidden_space_confidence_threshold",
        forbidden_space_confidence_threshold_,
        0.50);

    pnh_.param(
        "trajectory_risk/forbidden_space_body_inflation_m",
        forbidden_space_body_inflation_m_,
        0.0);

    if (forbidden_space_body_inflation_m_ < 0.0)
    {
      ROS_ERROR_STREAM(
          "[trajectory_risk_node] forbidden_space_body_inflation_m must be nonnegative, got "
          << forbidden_space_body_inflation_m_);
      forbidden_space_body_inflation_m_ = 0.0;
    }

    ignored_risk_links_.clear();
    if (!pnh_.getParam("trajectory_risk/ignored_risk_links",
                       ignored_risk_links_))
    {
      ROS_WARN_STREAM("[trajectory_risk_node] "
                      "trajectory_risk/ignored_risk_links is not set. "
                      "No links will be ignored.");
    }

    ROS_INFO_STREAM("[trajectory_risk_node] ignored_risk_links: "
                    << ignoredRiskLinksToString());
  }

  std::string outputTopic(const std::string& leaf) const
  {
    std::string base = output_namespace_;
    if (base.empty()) base = "/care_planner/trajectory_risk";
    while (base.size() > 1 && base.back() == '/') base.pop_back();
    if (leaf.empty()) return base;
    return base + "/" + leaf;
  }

  double wallMs(const ros::WallTime& start, const ros::WallTime& end) const
  {
    return (end - start).toSec() * 1000.0;
  }

  geometry_msgs::Point toPointMsg(const Eigen::Vector3d& p) const
  {
    geometry_msgs::Point out;
    out.x = p.x();
    out.y = p.y();
    out.z = p.z();
    return out;
  }

  std_msgs::Float32 makeFloatMsg(double value) const
  {
    std_msgs::Float32 msg;
    msg.data = static_cast<float>(value);
    return msg;
  }

  std_msgs::Int32 makeIntMsg(int value) const
  {
    std_msgs::Int32 msg;
    msg.data = value;
    return msg;
  }

  bool isIgnoredRiskLink(const std::string& link_name) const
  {
    return std::find(
               ignored_risk_links_.begin(),
               ignored_risk_links_.end(),
               link_name) != ignored_risk_links_.end();
  }

  std::string ignoredRiskLinksToString() const
  {
    std::ostringstream oss;

    for (std::size_t i = 0; i < ignored_risk_links_.size(); ++i)
    {
      if (i > 0)
      {
        oss << ",";
      }
      oss << ignored_risk_links_[i];
    }

    return oss.str();
  }

  std::vector<int> makeDownsampleIndices(int input_size) const
  {
    std::vector<int> indices;

    if (input_size <= 0)
    {
      return indices;
    }

    const int max_steps = std::max(1, max_eval_timesteps_);

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

      int idx =
          static_cast<int>(
              std::round(s * static_cast<double>(input_size - 1)));

      idx = std::max(0, std::min(input_size - 1, idx));

      if (indices.empty() || indices.back() != idx)
      {
        indices.push_back(idx);
      }
    }

    if (indices.empty() || indices.front() != 0)
    {
      indices.insert(indices.begin(), 0);
    }

    if (indices.back() != input_size - 1)
    {
      indices.push_back(input_size - 1);
    }

    return indices;
  }

  bool jointTrajectoryToQTrajectory(
      const trajectory_msgs::JointTrajectory& traj_msg,
      std::vector<Eigen::VectorXd>* q_traj,
      std::vector<int>* eval_to_original_index,
      std::string* error_msg) const
  {
    q_traj->clear();
    eval_to_original_index->clear();

    if (traj_msg.joint_names.empty())
    {
      if (error_msg)
      {
        *error_msg = "JointTrajectory.joint_names is empty.";
      }
      return false;
    }

    if (traj_msg.points.empty())
    {
      if (error_msg)
      {
        *error_msg = "JointTrajectory.points is empty.";
      }
      return false;
    }

    std::map<std::string, int> input_joint_index;
    for (int i = 0; i < static_cast<int>(traj_msg.joint_names.size()); ++i)
    {
      input_joint_index[traj_msg.joint_names[static_cast<std::size_t>(i)]] = i;
    }

    const auto& required_names = evaluator_.activeJointNames();

    if (static_cast<int>(required_names.size()) != evaluator_.nq())
    {
      if (error_msg)
      {
        std::ostringstream oss;
        oss << "Unexpected active joint count. activeJointNames.size()="
            << required_names.size()
            << ", nq="
            << evaluator_.nq();
        *error_msg = oss.str();
      }
      return false;
    }

    std::vector<int> map_required_to_input;
    map_required_to_input.resize(required_names.size(), -1);

    for (int i = 0; i < static_cast<int>(required_names.size()); ++i)
    {
      const std::string& joint_name =
          required_names[static_cast<std::size_t>(i)];

      const auto it = input_joint_index.find(joint_name);

      if (it == input_joint_index.end())
      {
        if (error_msg)
        {
          std::ostringstream oss;
          oss << "Input JointTrajectory is missing required joint: "
              << joint_name;
          *error_msg = oss.str();
        }
        return false;
      }

      map_required_to_input[static_cast<std::size_t>(i)] = it->second;
    }

    const std::vector<int> selected_indices =
        makeDownsampleIndices(static_cast<int>(traj_msg.points.size()));

    q_traj->reserve(selected_indices.size());
    eval_to_original_index->reserve(selected_indices.size());

    for (const int original_index : selected_indices)
    {
      const auto& pt =
          traj_msg.points[static_cast<std::size_t>(original_index)];

      if (pt.positions.size() < traj_msg.joint_names.size())
      {
        if (error_msg)
        {
          std::ostringstream oss;
          oss << "Trajectory point " << original_index
              << " has positions.size()=" << pt.positions.size()
              << ", but joint_names.size()=" << traj_msg.joint_names.size();
          *error_msg = oss.str();
        }
        return false;
      }

      Eigen::VectorXd q(evaluator_.nq());
      q.setZero();

      for (int i = 0; i < evaluator_.nq(); ++i)
      {
        const int input_index =
            map_required_to_input[static_cast<std::size_t>(i)];

        q(i) = pt.positions[static_cast<std::size_t>(input_index)];
      }

      q_traj->push_back(q);
      eval_to_original_index->push_back(original_index);
    }

    return true;
  }

  bool refreshBodyPriorBeforeQuery()
  {
    if (!refresh_body_prior_before_query_)
    {
      return true;
    }

    if (!refresh_body_prior_client_.waitForExistence(
            ros::Duration(refresh_body_prior_timeout_)))
    {
      ROS_WARN_THROTTLE(
          2.0,
          "[trajectory_risk_node] Waiting for refresh body prior service: %s",
          refresh_body_prior_service_.c_str());
      return false;
    }

    std_srvs::Trigger srv;
    if (!refresh_body_prior_client_.call(srv))
    {
      ROS_WARN_THROTTLE(
          2.0,
          "[trajectory_risk_node] Failed to call refresh body prior service: %s",
          refresh_body_prior_service_.c_str());
      return false;
    }

    if (!srv.response.success)
    {
      ROS_WARN_THROTTLE(
          2.0,
          "[trajectory_risk_node] Refresh body prior service returned failure: %s",
          srv.response.message.c_str());
      return false;
    }

    ROS_INFO_THROTTLE(
        2.0,
        "[trajectory_risk_node] Refreshed current-body confidence prior before trajectory query: %s",
        srv.response.message.c_str());

    return true;
  }

  bool queryTrajectoryConfidence(
      const care_confidence_map::TrajectorySampleResult& sample_result,
      care_confidence_map::QueryConfidence* srv)
  {
    srv->request.points.clear();
    srv->request.points.reserve(
        static_cast<std::size_t>(sample_result.total_samples));

    for (const auto& frame : sample_result.frames)
    {
      for (const auto& sample : frame.samples)
      {
        srv->request.points.push_back(toPointMsg(sample.center_base));
      }
    }

    if (!confidence_query_client_.waitForExistence(
            ros::Duration(query_timeout_)))
    {
      ROS_WARN_THROTTLE(
          2.0,
          "[trajectory_risk_node] Waiting for confidence query service: %s",
          confidence_query_service_.c_str());
      return false;
    }

    if (!confidence_query_client_.call(*srv))
    {
      ROS_WARN_THROTTLE(
          2.0,
          "[trajectory_risk_node] Failed to call confidence query service: %s",
          confidence_query_service_.c_str());
      return false;
    }

    const std::size_t n = srv->request.points.size();

    if (srv->response.confidence.size() != n ||
        srv->response.current_visibility.size() != n ||
        srv->response.inside_map.size() != n)
    {
      ROS_WARN_THROTTLE(
          2.0,
          "[trajectory_risk_node] Invalid query response size. "
          "request=%zu, confidence=%zu, visibility=%zu, inside_map=%zu",
          n,
          srv->response.confidence.size(),
          srv->response.current_visibility.size(),
          srv->response.inside_map.size());
      return false;
    }

    return true;
  }

  RiskResult computeRiskResult(
      const care_confidence_map::TrajectorySampleResult& sample_result,
      const care_confidence_map::QueryConfidence& srv,
      int input_num_timesteps,
      const std::vector<int>& eval_to_original_index) const
  {
    RiskResult result;
    result.success = true;
    result.message = "OK";

    result.input_num_timesteps = input_num_timesteps;
    result.eval_num_timesteps = sample_result.num_timesteps;
    result.samples_per_timestep = sample_result.num_samples_per_timestep;
    result.total_samples = sample_result.total_samples;
    result.eval_to_original_index = eval_to_original_index;

    result.timestep_stats.resize(
        static_cast<std::size_t>(sample_result.num_timesteps));

    for (int k = 0; k < sample_result.num_timesteps; ++k)
    {
      TimestepStats& ts =
          result.timestep_stats[static_cast<std::size_t>(k)];

      ts.timestep_index = k;

      if (k < static_cast<int>(eval_to_original_index.size()))
      {
        ts.original_timestep_index =
            eval_to_original_index[static_cast<std::size_t>(k)];
      }
    }

    double confidence_sum = 0.0;
    double min_confidence = std::numeric_limits<double>::infinity();

    std::size_t flat_index = 0;

    for (const auto& frame : sample_result.frames)
    {
      const int k = frame.timestep_index;
      TimestepStats& ts =
          result.timestep_stats[static_cast<std::size_t>(k)];

      double timestep_confidence_sum = 0.0;
      double timestep_min_confidence =
          std::numeric_limits<double>::infinity();

      ts.total_count = 0;

      for (const auto& sample : frame.samples)
      {
        const bool inside =
            (srv.response.inside_map[flat_index] != 0);

        const double confidence =
            static_cast<double>(srv.response.confidence[flat_index]);

        const double visible =
            static_cast<double>(
                srv.response.current_visibility[flat_index]);

        if (isIgnoredRiskLink(sample.link_name))
        {
          flat_index += 1;
          continue;
        }

        ts.total_count += 1;

        if (!inside)
        {
          result.outside_count += 1;
          ts.outside_count += 1;
          flat_index += 1;
          continue;
        }

        result.inside_count += 1;
        ts.inside_count += 1;

        confidence_sum += confidence;
        timestep_confidence_sum += confidence;

        if (visible > 0.5)
        {
          result.visible_count += 1;
          ts.visible_count += 1;
        }

        if (confidence < min_confidence)
        {
          min_confidence = confidence;
        }

        if (confidence < timestep_min_confidence)
        {
          timestep_min_confidence = confidence;
        }

        if (confidence < result.worst_confidence)
        {
          result.worst_confidence = confidence;
          result.worst_sample_timestep = k;

          if (k < static_cast<int>(eval_to_original_index.size()))
          {
            result.worst_sample_original_timestep =
                eval_to_original_index[static_cast<std::size_t>(k)];
          }

          result.worst_link = sample.link_name;
          result.worst_sample_index_in_link = sample.sample_index_in_link;
          result.worst_source_collision_index = sample.source_collision_index;
        }

        if (confidence < ts.worst_confidence)
        {
          ts.worst_confidence = confidence;
          ts.worst_link = sample.link_name;
          ts.worst_sample_index_in_link = sample.sample_index_in_link;
          ts.worst_source_collision_index = sample.source_collision_index;
        }

        flat_index += 1;
      }

      if (ts.inside_count > 0)
      {
        ts.mean_confidence =
            timestep_confidence_sum / static_cast<double>(ts.inside_count);
        ts.min_confidence = timestep_min_confidence;
        ts.visible_ratio =
            static_cast<double>(ts.visible_count) /
            static_cast<double>(ts.inside_count);
      }
      else
      {
        ts.mean_confidence = 0.0;
        ts.min_confidence = 0.0;
        ts.visible_ratio = 0.0;
        ts.worst_confidence = 0.0;
        ts.worst_link = "none";
      }

      if (!std::isfinite(ts.min_confidence))
      {
        ts.min_confidence = 0.0;
      }

      ts.risk =
          std::max(0.0, std::min(1.0, 1.0 - ts.mean_confidence));

      if (ts.risk > result.score || result.worst_risk_timestep < 0)
      {
        result.score = ts.risk;
        result.worst_timestep_risk = ts.risk;
        result.worst_risk_timestep = k;

        if (k < static_cast<int>(eval_to_original_index.size()))
        {
          result.worst_risk_original_timestep =
              eval_to_original_index[static_cast<std::size_t>(k)];
        }
      }
    }

    if (result.inside_count > 0)
    {
      result.mean_confidence =
          confidence_sum / static_cast<double>(result.inside_count);
      result.min_confidence = min_confidence;
      result.visible_ratio =
          static_cast<double>(result.visible_count) /
          static_cast<double>(result.inside_count);
    }
    else
    {
      result.mean_confidence = 0.0;
      result.min_confidence = 0.0;
      result.visible_ratio = 0.0;
      result.worst_confidence = 0.0;
      result.score = 1.0;
      result.worst_timestep_risk = 1.0;
      result.worst_link = "none";
    }

    if (!std::isfinite(result.min_confidence))
    {
      result.min_confidence = 0.0;
    }

    return result;
  }

  LinkAttributionStats finalizeLinkAttributionStats(
      const LinkAttributionStats& in) const
  {
    LinkAttributionStats out = in;

    if (out.inside_count > 0)
    {
      out.mean_confidence =
          out.confidence_sum / static_cast<double>(out.inside_count);

      out.visible_ratio =
          static_cast<double>(out.visible_count) /
          static_cast<double>(out.inside_count);

      out.gap =
          std::max(0.0, std::min(1.0, 1.0 - out.mean_confidence));

      if (!std::isfinite(out.min_confidence))
      {
        out.min_confidence = 0.0;
      }
    }
    else
    {
      out.mean_confidence = 0.0;
      out.min_confidence = 0.0;
      out.visible_ratio = 0.0;
      out.gap = 1.0;
    }

    return out;
  }

  void updateLinkAttributionStats(
      const std::string& link_name,
      bool inside,
      double confidence,
      double current_visibility,
      std::map<std::string, LinkAttributionStats>* stats_map) const
  {
    LinkAttributionStats& stats = (*stats_map)[link_name];

    if (stats.link_name == "none")
    {
      stats.link_name = link_name;
      stats.min_confidence = std::numeric_limits<double>::infinity();
    }

    stats.sample_count += 1;

    if (!inside)
    {
      stats.outside_count += 1;
      return;
    }

    stats.inside_count += 1;
    stats.confidence_sum += confidence;

    if (current_visibility > 0.5)
    {
      stats.visible_count += 1;
    }

    if (confidence < stats.min_confidence)
    {
      stats.min_confidence = confidence;
    }
  }

  std::vector<LinkAttributionStats> finalizeAndSortLinkStats(
      const std::map<std::string, LinkAttributionStats>& stats_map) const
  {
    std::vector<LinkAttributionStats> out;
    out.reserve(stats_map.size());

    for (const auto& kv : stats_map)
    {
      out.push_back(finalizeLinkAttributionStats(kv.second));
    }

    std::sort(
        out.begin(),
        out.end(),
        [](const LinkAttributionStats& a,
           const LinkAttributionStats& b)
        {
          if (std::fabs(a.gap - b.gap) > 1e-9)
          {
            return a.gap > b.gap;
          }
          return a.link_name < b.link_name;
        });

    return out;
  }

  static bool sampleAttributionLess(
      const SampleAttributionItem& a,
      const SampleAttributionItem& b)
  {
    if (std::fabs(a.gap - b.gap) > 1e-9)
    {
      return a.gap > b.gap;
    }

    if (std::fabs(a.confidence - b.confidence) > 1e-9)
    {
      return a.confidence < b.confidence;
    }

    if (a.eval_timestep != b.eval_timestep)
    {
      return a.eval_timestep < b.eval_timestep;
    }

    if (a.link_name != b.link_name)
    {
      return a.link_name < b.link_name;
    }

    return a.sample_index_in_link < b.sample_index_in_link;
  }

  bool isRiskFrontierSample(const SampleAttributionItem& item) const
  {
    if (!item.inside)
    {
      return false;
    }

    if (item.confidence > frontier_confidence_threshold_)
    {
      return false;
    }

    if (item.gap < frontier_gap_threshold_)
    {
      return false;
    }

    return true;
  }

  int originalIndexForEvalTimestep(
      const RiskResult& risk_result,
      int eval_timestep) const
  {
    if (eval_timestep < 0 ||
        eval_timestep >=
            static_cast<int>(risk_result.eval_to_original_index.size()))
    {
      return -1;
    }

    return risk_result.eval_to_original_index[
        static_cast<std::size_t>(eval_timestep)];
  }

  void computeRiskFrontierAttribution(
      const RiskResult& risk_result,
      const std::vector<SampleAttributionItem>& all_sample_items,
      AttributionResult* attr) const
  {
    if (!attr)
    {
      return;
    }

    attr->risk_frontier_threshold = risk_frontier_threshold_;

    const int num_eval_timesteps =
        static_cast<int>(risk_result.timestep_stats.size());

    if (num_eval_timesteps <= 0)
    {
      attr->has_risk_frontier = false;
      return;
    }

    // Sample-temporal frontier detection:
    // Find the earliest eval timestep where any considered body/collision
    // sample sphere becomes low-confidence. This is more sensitive than
    // whole-body mean timestep risk and is the correct trigger for active sensing.
    std::vector<std::vector<SampleAttributionItem>> bad_samples_by_timestep(
        static_cast<std::size_t>(num_eval_timesteps));

    for (const auto& item : all_sample_items)
    {
      if (item.eval_timestep < 0 || item.eval_timestep >= num_eval_timesteps)
      {
        continue;
      }

      if (!isRiskFrontierSample(item))
      {
        continue;
      }

      bad_samples_by_timestep[static_cast<std::size_t>(item.eval_timestep)].push_back(item);
    }

    int first_bad_eval_timestep = -1;
    for (int t = 0; t < num_eval_timesteps; ++t)
    {
      if (!bad_samples_by_timestep[static_cast<std::size_t>(t)].empty())
      {
        first_bad_eval_timestep = t;
        break;
      }
    }

    if (first_bad_eval_timestep < 0)
    {
      attr->has_risk_frontier = false;
      return;
    }

    const int primary_window_start = first_bad_eval_timestep;
    const int primary_window_end =
        std::min(num_eval_timesteps - 1,
                 first_bad_eval_timestep +
                     std::max(0, risk_frontier_window_steps_));

    const int safe_prefix_eval =
        std::max(0,
                 first_bad_eval_timestep -
                     std::max(0, safe_prefix_margin_steps_));

    attr->has_risk_frontier = true;
    attr->first_risky_eval_timestep = first_bad_eval_timestep;
    attr->first_risky_original_timestep =
        originalIndexForEvalTimestep(risk_result, first_bad_eval_timestep);
    attr->safe_prefix_end_eval_timestep = safe_prefix_eval;
    attr->safe_prefix_end_original_timestep =
        originalIndexForEvalTimestep(risk_result, safe_prefix_eval);

    attr->risk_frontier_window_start_eval = primary_window_start;
    attr->risk_frontier_window_end_eval = primary_window_end;
    attr->risk_frontier_window_start_original =
        originalIndexForEvalTimestep(risk_result, primary_window_start);
    attr->risk_frontier_window_end_original =
        originalIndexForEvalTimestep(risk_result, primary_window_end);

    for (int t = primary_window_start; t <= primary_window_end; ++t)
    {
      const auto& samples = bad_samples_by_timestep[static_cast<std::size_t>(t)];
      attr->risk_frontier_samples.insert(
          attr->risk_frontier_samples.end(), samples.begin(), samples.end());
    }

    if (primary_window_end + 1 < num_eval_timesteps)
    {
      attr->secondary_frontier_window_start_eval = primary_window_end + 1;
      attr->secondary_frontier_window_end_eval = num_eval_timesteps - 1;
      attr->secondary_frontier_window_start_original =
          originalIndexForEvalTimestep(
              risk_result, attr->secondary_frontier_window_start_eval);
      attr->secondary_frontier_window_end_original =
          originalIndexForEvalTimestep(
              risk_result, attr->secondary_frontier_window_end_eval);

      for (int t = primary_window_end + 1; t < num_eval_timesteps; ++t)
      {
        const auto& samples = bad_samples_by_timestep[static_cast<std::size_t>(t)];
        attr->secondary_frontier_samples.insert(
            attr->secondary_frontier_samples.end(), samples.begin(), samples.end());
      }
    }

    auto summarizeGroup =
        [this](const std::vector<SampleAttributionItem>& samples,
               Eigen::Vector3d* centroid,
               double* radius,
               double* mean_gap,
               double* total_weight,
               std::vector<LinkAttributionStats>* link_stats)
    {
      if (!centroid || !radius || !mean_gap || !total_weight || !link_stats)
      {
        return;
      }

      *centroid = Eigen::Vector3d::Zero();
      *radius = 0.0;
      *mean_gap = 0.0;
      *total_weight = 0.0;
      link_stats->clear();

      std::map<std::string, LinkAttributionStats> stats_map;
      Eigen::Vector3d weighted_sum = Eigen::Vector3d::Zero();
      double weight_sum = 0.0;
      double gap_sum = 0.0;

      for (const auto& item : samples)
      {
        updateLinkAttributionStats(
            item.link_name,
            item.inside,
            item.confidence,
            item.current_visibility,
            &stats_map);

        const double w = std::max(1e-6, item.gap);
        weighted_sum += w * item.center_base;
        weight_sum += w;
        gap_sum += item.gap;
      }

      *link_stats = finalizeAndSortLinkStats(stats_map);

      if (samples.empty() || weight_sum <= 0.0)
      {
        return;
      }

      *centroid = weighted_sum / weight_sum;
      *total_weight = weight_sum;
      *mean_gap = gap_sum / static_cast<double>(samples.size());

      double r = 0.0;
      for (const auto& item : samples)
      {
        r = std::max(
            r,
            (item.center_base - *centroid).norm() + item.radius);
      }

      *radius = r + frontier_radius_margin_;
    };

    summarizeGroup(
        attr->risk_frontier_samples,
        &attr->risk_frontier_centroid_base,
        &attr->risk_frontier_radius,
        &attr->risk_frontier_mean_gap,
        &attr->risk_frontier_total_weight,
        &attr->risk_frontier_link_stats);

    summarizeGroup(
        attr->secondary_frontier_samples,
        &attr->secondary_frontier_centroid_base,
        &attr->secondary_frontier_radius,
        &attr->secondary_frontier_mean_gap,
        &attr->secondary_frontier_total_weight,
        &attr->secondary_frontier_link_stats);

    attr->has_secondary_frontier = !attr->secondary_frontier_samples.empty();

    std::sort(
        attr->risk_frontier_samples.begin(),
        attr->risk_frontier_samples.end(),
        sampleAttributionLess);

    std::sort(
        attr->secondary_frontier_samples.begin(),
        attr->secondary_frontier_samples.end(),
        sampleAttributionLess);
  }

  AttributionResult computeAttributionResult(
      const care_confidence_map::TrajectorySampleResult& sample_result,
      const care_confidence_map::QueryConfidence& srv,
      const RiskResult& risk_result) const
  {
    AttributionResult attr;

    if (!sample_result.success || !risk_result.success)
    {
      attr.success = false;
      attr.message = "sample_result or risk_result is not successful";
      return attr;
    }

    attr.success = true;
    attr.message = "OK";

    attr.trajectory_gap = risk_result.score;

    attr.worst_eval_timestep = risk_result.worst_risk_timestep;
    attr.worst_original_timestep = risk_result.worst_risk_original_timestep;
    attr.worst_timestep_gap = risk_result.worst_timestep_risk;

    if (risk_result.worst_risk_timestep >= 0 &&
        risk_result.worst_risk_timestep <
            static_cast<int>(risk_result.timestep_stats.size()))
    {
      const TimestepStats& ts =
          risk_result.timestep_stats[
              static_cast<std::size_t>(risk_result.worst_risk_timestep)];

      attr.worst_timestep_mean_confidence = ts.mean_confidence;
      attr.worst_timestep_visible_ratio = ts.visible_ratio;
    }

    std::map<std::string, LinkAttributionStats> link_stats_all;
    std::map<std::string, LinkAttributionStats> link_stats_worst_timestep;

    std::vector<SampleAttributionItem> all_sample_items;
    all_sample_items.reserve(
        static_cast<std::size_t>(sample_result.total_samples));

    std::size_t flat_index = 0;

    for (const auto& frame : sample_result.frames)
    {
      int original_timestep = -1;
      if (frame.timestep_index >= 0 &&
          frame.timestep_index <
              static_cast<int>(risk_result.eval_to_original_index.size()))
      {
        original_timestep =
            risk_result.eval_to_original_index[
                static_cast<std::size_t>(frame.timestep_index)];
      }

      for (const auto& sample : frame.samples)
      {
        const bool inside =
            (srv.response.inside_map[flat_index] != 0);

        const double confidence =
            static_cast<double>(srv.response.confidence[flat_index]);

        const double current_visibility =
            static_cast<double>(
                srv.response.current_visibility[flat_index]);

        if (isIgnoredRiskLink(sample.link_name))
        {
          flat_index += 1;
          continue;
        }

        updateLinkAttributionStats(
            sample.link_name,
            inside,
            confidence,
            current_visibility,
            &link_stats_all);

        if (frame.timestep_index == risk_result.worst_risk_timestep)
        {
          updateLinkAttributionStats(
              sample.link_name,
              inside,
              confidence,
              current_visibility,
              &link_stats_worst_timestep);
        }

        SampleAttributionItem item;
        item.eval_timestep = frame.timestep_index;
        item.original_timestep = original_timestep;
        item.link_name = sample.link_name;
        item.sample_index_in_link = sample.sample_index_in_link;
        item.source_collision_index = sample.source_collision_index;
        item.confidence = inside ? confidence : 0.0;
        item.gap =
            inside
                ? std::max(0.0, std::min(1.0, 1.0 - confidence))
                : 1.0;
        item.current_visibility = current_visibility;
        item.inside = inside;
        item.center_base = sample.center_base;
        item.radius = sample.radius;

        all_sample_items.push_back(item);

        flat_index += 1;
      }
    }

    computeRiskFrontierAttribution(
        risk_result,
        all_sample_items,
        &attr);

    attr.link_stats_over_trajectory =
        finalizeAndSortLinkStats(link_stats_all);

    attr.link_stats_at_worst_timestep =
        finalizeAndSortLinkStats(link_stats_worst_timestep);

    if (!attr.link_stats_over_trajectory.empty())
    {
      const LinkAttributionStats& worst =
          attr.link_stats_over_trajectory.front();

      attr.worst_link_over_trajectory = worst.link_name;
      attr.worst_link_gap_over_trajectory = worst.gap;
      attr.worst_link_mean_confidence_over_trajectory =
          worst.mean_confidence;
      attr.worst_link_visible_ratio_over_trajectory =
          worst.visible_ratio;
    }

    if (!attr.link_stats_at_worst_timestep.empty())
    {
      const LinkAttributionStats& worst =
          attr.link_stats_at_worst_timestep.front();

      attr.worst_link_at_worst_timestep = worst.link_name;
      attr.worst_link_gap_at_worst_timestep = worst.gap;
      attr.worst_link_mean_confidence_at_worst_timestep =
          worst.mean_confidence;
      attr.worst_link_visible_ratio_at_worst_timestep =
          worst.visible_ratio;
    }

    std::sort(
        all_sample_items.begin(),
        all_sample_items.end(),
        sampleAttributionLess);

    const int top_k =
        std::max(0,
                 std::min(top_k_samples_,
                          static_cast<int>(all_sample_items.size())));

    attr.topk_low_confidence_samples.assign(
        all_sample_items.begin(),
        all_sample_items.begin() + top_k);

    for (const auto& item : all_sample_items)
    {
      if (item.eval_timestep == attr.worst_eval_timestep &&
          item.link_name == attr.worst_link_at_worst_timestep)
      {
        attr.worst_timestep_worst_link_samples.push_back(item);
      }
    }

    return attr;
  }

  visualization_msgs::Marker makeDeleteAllMarker(const std::string& ns) const
  {
    visualization_msgs::Marker marker;
    marker.header.frame_id = base_frame_;
    marker.header.stamp = ros::Time::now();
    marker.ns = ns;
    marker.id = 0;
    marker.action = visualization_msgs::Marker::DELETEALL;
    return marker;
  }

  void setColorForConfidence(
      double confidence,
      bool inside,
      double alpha,
      visualization_msgs::Marker* marker) const
  {
    if (!inside)
    {
      marker->color.r = 0.4f;
      marker->color.g = 0.4f;
      marker->color.b = 0.4f;
      marker->color.a = static_cast<float>(alpha);
      return;
    }

    const double q =
        std::max(0.0, std::min(1.0, confidence));

    marker->color.r = static_cast<float>(1.0 - q);
    marker->color.g = static_cast<float>(q);
    marker->color.b = 0.05f;
    marker->color.a = static_cast<float>(alpha);
  }

  visualization_msgs::Marker makeSampleMarker(
      const care_confidence_map::TrajectoryBodySample& sample,
      const std::string& ns_prefix,
      int marker_id,
      double confidence,
      bool inside,
      bool is_worst_marker) const
  {
    visualization_msgs::Marker marker;

    marker.header.frame_id = base_frame_;
    marker.header.stamp = ros::Time::now();

    marker.ns =
        ns_prefix + "/t" +
        std::to_string(sample.timestep_index) +
        "/" +
        sample.link_name;

    marker.id = marker_id;

    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;

    marker.pose.position = toPointMsg(sample.center_base);
    marker.pose.orientation.w = 1.0;

    const double scale_multiplier = is_worst_marker ? 2.4 : 2.0;
    marker.scale.x = scale_multiplier * sample.radius;
    marker.scale.y = scale_multiplier * sample.radius;
    marker.scale.z = scale_multiplier * sample.radius;

    setColorForConfidence(
        confidence,
        inside,
        is_worst_marker ? worst_marker_alpha_ : marker_alpha_,
        &marker);

    marker.lifetime = ros::Duration(0.0);
    return marker;
  }

  visualization_msgs::Marker makeAttributionSampleMarker(
      const SampleAttributionItem& item,
      const std::string& ns_prefix,
      int marker_id,
      bool is_topk_marker) const
  {
    visualization_msgs::Marker marker;

    marker.header.frame_id = base_frame_;
    marker.header.stamp = ros::Time::now();

    marker.ns =
        ns_prefix +
        "/t" +
        std::to_string(item.eval_timestep) +
        "/" +
        item.link_name;

    marker.id = marker_id;

    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;

    marker.pose.position = toPointMsg(item.center_base);
    marker.pose.orientation.w = 1.0;

    const double scale_multiplier = is_topk_marker ? 3.0 : 2.5;
    marker.scale.x = scale_multiplier * item.radius;
    marker.scale.y = scale_multiplier * item.radius;
    marker.scale.z = scale_multiplier * item.radius;

    if (is_topk_marker)
    {
      marker.color.r = 1.0f;
      marker.color.g = 0.0f;
      marker.color.b = 0.0f;
      marker.color.a = static_cast<float>(attribution_marker_alpha_);
    }
    else
    {
      marker.color.r = 1.0f;
      marker.color.g = 0.8f;
      marker.color.b = 0.0f;
      marker.color.a = static_cast<float>(attribution_marker_alpha_);
    }

    marker.lifetime = ros::Duration(0.0);
    return marker;
  }

  visualization_msgs::Marker makeColoredFrontierSampleMarker(
      const SampleAttributionItem& item,
      const std::string& ns_prefix,
      int marker_id,
      float r,
      float g,
      float b,
      float alpha,
      double scale_multiplier) const
  {
    visualization_msgs::Marker marker;

    marker.header.frame_id = base_frame_;
    marker.header.stamp = ros::Time::now();
    marker.ns =
        ns_prefix +
        "/t" +
        std::to_string(item.eval_timestep) +
        "/" +
        item.link_name;
    marker.id = marker_id;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position = toPointMsg(item.center_base);
    marker.pose.orientation.w = 1.0;
    marker.scale.x = scale_multiplier * item.radius;
    marker.scale.y = scale_multiplier * item.radius;
    marker.scale.z = scale_multiplier * item.radius;
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;
    marker.color.a = alpha;
    marker.lifetime = ros::Duration(0.0);
    return marker;
  }

  visualization_msgs::Marker makeFrontierSummarySphereMarker(
      const Eigen::Vector3d& centroid,
      double radius,
      const std::string& ns,
      int marker_id,
      float r,
      float g,
      float b,
      float alpha) const
  {
    visualization_msgs::Marker marker;

    marker.header.frame_id = base_frame_;
    marker.header.stamp = ros::Time::now();
    marker.ns = ns;
    marker.id = marker_id;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position = toPointMsg(centroid);
    marker.pose.orientation.w = 1.0;
    const double diameter = 2.0 * std::max(0.02, radius);
    marker.scale.x = diameter;
    marker.scale.y = diameter;
    marker.scale.z = diameter;
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;
    marker.color.a = alpha;
    marker.lifetime = ros::Duration(0.0);
    return marker;
  }

  visualization_msgs::Marker makeFrontierGroupTextMarker(
      const Eigen::Vector3d& centroid,
      double radius,
      const std::string& ns,
      int marker_id,
      const std::string& text,
      float r,
      float g,
      float b) const
  {
    visualization_msgs::Marker marker;

    marker.header.frame_id = base_frame_;
    marker.header.stamp = ros::Time::now();
    marker.ns = ns;
    marker.id = marker_id;
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position = toPointMsg(centroid);
    marker.pose.position.z += std::max(0.05, radius + 0.04);
    marker.pose.orientation.w = 1.0;
    marker.scale.z = 0.035;
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;
    marker.color.a = 1.0f;
    marker.text = text;
    marker.lifetime = ros::Duration(0.0);
    return marker;
  }

  visualization_msgs::Marker makeAttributionTextMarker(
      const AttributionResult& attr,
      int marker_id) const
  {
    visualization_msgs::Marker marker;

    marker.header.frame_id = base_frame_;
    marker.header.stamp = ros::Time::now();

    marker.ns = "trajectory_risk/attribution_text";
    marker.id = marker_id;

    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::Marker::ADD;

    marker.pose.position.x = 0.0;
    marker.pose.position.y = 0.0;
    marker.pose.position.z = 1.25;
    marker.pose.orientation.w = 1.0;

    marker.scale.z = 0.045;

    marker.color.r = 1.0f;
    marker.color.g = 1.0f;
    marker.color.b = 1.0f;
    marker.color.a = 1.0f;

    std::ostringstream oss;
    oss << "U_gap attribution"
        << "\ntrajectory_gap = " << attr.trajectory_gap
        << "\nignored links = " << ignoredRiskLinksToString()
        << "\nworst eval timestep = " << attr.worst_eval_timestep
        << "\nworst original timestep = " << attr.worst_original_timestep
        << "\nworst link over trajectory = " << attr.worst_link_over_trajectory
        << "\nworst link at worst timestep = " << attr.worst_link_at_worst_timestep
        << "\ntop-K samples = " << attr.topk_low_confidence_samples.size();

    marker.text = oss.str();

    marker.lifetime = ros::Duration(0.0);
    return marker;
  }

  visualization_msgs::Marker makeWorstTimestepTextMarker(
      const RiskResult& result,
      int marker_id) const
  {
    visualization_msgs::Marker marker;

    marker.header.frame_id = base_frame_;
    marker.header.stamp = ros::Time::now();

    marker.ns = "trajectory_risk/worst_timestep_text";
    marker.id = marker_id;

    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::Marker::ADD;

    marker.pose.position.x = 0.0;
    marker.pose.position.y = 0.0;
    marker.pose.position.z = 1.15;
    marker.pose.orientation.w = 1.0;

    marker.scale.z = 0.045;

    marker.color.r = 1.0f;
    marker.color.g = 1.0f;
    marker.color.b = 1.0f;
    marker.color.a = 1.0f;

    std::ostringstream oss;
    oss << "Trajectory risk = " << result.score
        << "\nIgnored links = " << ignoredRiskLinksToString()
        << "\nWorst eval timestep = " << result.worst_risk_timestep
        << "\nWorst original timestep = "
        << result.worst_risk_original_timestep
        << "\nWorst timestep risk = " << result.worst_timestep_risk
        << "\nMean confidence = " << result.mean_confidence
        << "\nMin confidence = " << result.min_confidence
        << "\nWorst link = " << result.worst_link;

    marker.text = oss.str();
    marker.lifetime = ros::Duration(0.0);
    return marker;
  }

  std_msgs::String makeRiskSummaryMsg(const RiskResult& result) const
  {
    std::ostringstream oss;

    oss << "trajectory_risk: "
        << "success=" << result.success
        << ", message=" << result.message
        << ", score=" << result.score
        << ", rule=max_timestep(1-mean_confidence)"
        << ", ignored_risk_links=" << ignoredRiskLinksToString()
        << ", mean_confidence=" << result.mean_confidence
        << ", min_confidence=" << result.min_confidence
        << ", visible_ratio=" << result.visible_ratio
        << ", input_num_timesteps=" << result.input_num_timesteps
        << ", eval_num_timesteps=" << result.eval_num_timesteps
        << ", max_eval_timesteps=" << max_eval_timesteps_
        << ", samples_per_timestep=" << result.samples_per_timestep
        << ", total_samples_queried=" << result.total_samples
        << ", risk_inside=" << result.inside_count
        << ", risk_outside=" << result.outside_count
        << ", worst_risk_timestep=" << result.worst_risk_timestep
        << ", worst_risk_original_timestep="
        << result.worst_risk_original_timestep
        << ", worst_timestep_risk=" << result.worst_timestep_risk
        << ", worst_sample_timestep=" << result.worst_sample_timestep
        << ", worst_sample_original_timestep="
        << result.worst_sample_original_timestep
        << ", worst_link=" << result.worst_link
        << ", worst_sample_index_in_link="
        << result.worst_sample_index_in_link
        << ", worst_source_collision_index="
        << result.worst_source_collision_index
        << ", worst_confidence=" << result.worst_confidence;

    std_msgs::String msg;
    msg.data = oss.str();
    return msg;
  }

  std_msgs::String makeTimestepSummaryMsg(const RiskResult& result) const
  {
    std::ostringstream oss;
    oss << "trajectory_risk_timestep_query_summary:";

    for (const auto& ts : result.timestep_stats)
    {
      oss << "\n  eval_t=" << ts.timestep_index
          << ", original_t=" << ts.original_timestep_index
          << ": risk=" << ts.risk
          << ", mean=" << ts.mean_confidence
          << ", min=" << ts.min_confidence
          << ", visible_ratio=" << ts.visible_ratio
          << ", risk_inside=" << ts.inside_count
          << ", risk_outside=" << ts.outside_count
          << ", risk_total=" << ts.total_count
          << ", worst_link=" << ts.worst_link
          << ", worst_conf=" << ts.worst_confidence
          << ", worst_sample_index_in_link="
          << ts.worst_sample_index_in_link
          << ", worst_collision_index="
          << ts.worst_source_collision_index;
    }

    std_msgs::String msg;
    msg.data = oss.str();
    return msg;
  }

  std_msgs::String makeWorstTimestepSummaryMsg(const RiskResult& result) const
  {
    std::ostringstream oss;

    oss << "trajectory_risk_worst_timestep: "
        << "worst_risk_timestep=" << result.worst_risk_timestep
        << ", worst_risk_original_timestep="
        << result.worst_risk_original_timestep
        << ", worst_timestep_risk=" << result.worst_timestep_risk
        << ", trajectory_score=" << result.score
        << ", rule=max_timestep(1-mean_confidence)"
        << ", ignored_risk_links=" << ignoredRiskLinksToString()
        << ", worst_sample_timestep=" << result.worst_sample_timestep
        << ", worst_sample_original_timestep="
        << result.worst_sample_original_timestep
        << ", worst_link=" << result.worst_link
        << ", worst_sample_index_in_link="
        << result.worst_sample_index_in_link
        << ", worst_source_collision_index="
        << result.worst_source_collision_index
        << ", worst_confidence=" << result.worst_confidence;

    if (result.worst_risk_timestep >= 0 &&
        result.worst_risk_timestep <
            static_cast<int>(result.timestep_stats.size()))
    {
      const TimestepStats& ts =
          result.timestep_stats[static_cast<std::size_t>(
              result.worst_risk_timestep)];

      oss << ", timestep_mean_confidence=" << ts.mean_confidence
          << ", timestep_min_confidence=" << ts.min_confidence
          << ", timestep_visible_ratio=" << ts.visible_ratio
          << ", timestep_risk_inside=" << ts.inside_count
          << ", timestep_risk_outside=" << ts.outside_count
          << ", timestep_risk_total=" << ts.total_count
          << ", timestep_worst_link=" << ts.worst_link
          << ", timestep_worst_confidence=" << ts.worst_confidence;
    }

    std_msgs::String msg;
    msg.data = oss.str();
    return msg;
  }

  std_msgs::String makeTimingSummaryMsg(
      const RiskResult& result,
      const TimingStats& timing) const
  {
    std::ostringstream oss;

    oss << "trajectory_risk_timing: "
        << "total_ms=" << timing.total_time_ms
        << ", fk_ms=" << timing.fk_time_ms
        << ", query_ms=" << timing.query_time_ms
        << ", risk_compute_ms=" << timing.risk_compute_time_ms
        << ", attribution_ms=" << timing.attribution_time_ms
        << ", marker_ms=" << timing.marker_time_ms
        << ", convert_ms=" << timing.convert_time_ms
        << ", input_age_ms=" << timing.input_age_ms
        << ", eval_rate_target_hz=" << eval_rate_
        << ", target_period_ms=" << (1000.0 / std::max(0.1, eval_rate_))
        << ", input_num_timesteps=" << result.input_num_timesteps
        << ", eval_num_timesteps=" << result.eval_num_timesteps
        << ", total_samples_queried=" << result.total_samples
        << ", risk_inside=" << result.inside_count
        << ", markers_published=" << timing.published_markers;

    std_msgs::String msg;
    msg.data = oss.str();
    return msg;
  }

  std_msgs::String makeAttributionSummaryMsg(
      const AttributionResult& attr) const
  {
    std::ostringstream oss;

    oss << "trajectory_attribution: "
        << "success=" << attr.success
        << ", message=" << attr.message
        << ", ignored_risk_links=" << ignoredRiskLinksToString()
        << ", trajectory_gap=" << attr.trajectory_gap
        << ", worst_eval_timestep=" << attr.worst_eval_timestep
        << ", worst_original_timestep=" << attr.worst_original_timestep
        << ", worst_timestep_gap=" << attr.worst_timestep_gap
        << ", worst_timestep_mean_confidence="
        << attr.worst_timestep_mean_confidence
        << ", worst_timestep_visible_ratio="
        << attr.worst_timestep_visible_ratio
        << ", worst_link_over_trajectory="
        << attr.worst_link_over_trajectory
        << ", worst_link_gap_over_trajectory="
        << attr.worst_link_gap_over_trajectory
        << ", worst_link_mean_confidence_over_trajectory="
        << attr.worst_link_mean_confidence_over_trajectory
        << ", worst_link_visible_ratio_over_trajectory="
        << attr.worst_link_visible_ratio_over_trajectory
        << ", worst_link_at_worst_timestep="
        << attr.worst_link_at_worst_timestep
        << ", worst_link_gap_at_worst_timestep="
        << attr.worst_link_gap_at_worst_timestep
        << ", worst_link_mean_confidence_at_worst_timestep="
        << attr.worst_link_mean_confidence_at_worst_timestep
        << ", worst_link_visible_ratio_at_worst_timestep="
        << attr.worst_link_visible_ratio_at_worst_timestep
        << ", topk_sample_count="
        << attr.topk_low_confidence_samples.size()
        << ", worst_timestep_worst_link_sample_count="
        << attr.worst_timestep_worst_link_samples.size()
        << ", has_risk_frontier=" << attr.has_risk_frontier
        << ", first_risky_eval_timestep="
        << attr.first_risky_eval_timestep
        << ", first_risky_original_timestep="
        << attr.first_risky_original_timestep
        << ", safe_prefix_end_eval_timestep="
        << attr.safe_prefix_end_eval_timestep
        << ", safe_prefix_end_original_timestep="
        << attr.safe_prefix_end_original_timestep
        << ", risk_frontier_window_eval=["
        << attr.risk_frontier_window_start_eval << ","
        << attr.risk_frontier_window_end_eval << "]"
        << ", risk_frontier_window_original=["
        << attr.risk_frontier_window_start_original << ","
        << attr.risk_frontier_window_end_original << "]"
        << ", risk_frontier_sample_count="
        << attr.risk_frontier_samples.size()
        << ", risk_frontier_radius="
        << attr.risk_frontier_radius;

    std_msgs::String msg;
    msg.data = oss.str();
    return msg;
  }

  std_msgs::String makeTimestepAttributionMsg(
      const RiskResult& result) const
  {
    std::ostringstream oss;

    oss << "trajectory_timestep_attribution:";

    for (const auto& ts : result.timestep_stats)
    {
      const double timestep_gap =
          std::max(0.0, std::min(1.0, 1.0 - ts.mean_confidence));

      oss << "\n  eval_t=" << ts.timestep_index
          << ", original_t=" << ts.original_timestep_index
          << ", gap=" << timestep_gap
          << ", mean_confidence=" << ts.mean_confidence
          << ", min_confidence=" << ts.min_confidence
          << ", visible_ratio=" << ts.visible_ratio
          << ", risk_inside=" << ts.inside_count
          << ", risk_outside=" << ts.outside_count
          << ", risk_total=" << ts.total_count
          << ", worst_link=" << ts.worst_link
          << ", worst_confidence=" << ts.worst_confidence;
    }

    std_msgs::String msg;
    msg.data = oss.str();
    return msg;
  }

  std_msgs::String makeLinkAttributionMsg(
      const AttributionResult& attr) const
  {
    std::ostringstream oss;

    oss << "trajectory_link_attribution:"
        << "\n  ignored_risk_links=" << ignoredRiskLinksToString();

    oss << "\n  over_trajectory:";

    for (const auto& link : attr.link_stats_over_trajectory)
    {
      oss << "\n    " << link.link_name
          << ": gap=" << link.gap
          << ", mean_confidence=" << link.mean_confidence
          << ", min_confidence=" << link.min_confidence
          << ", visible_ratio=" << link.visible_ratio
          << ", inside=" << link.inside_count
          << ", outside=" << link.outside_count
          << ", total=" << link.sample_count;
    }

    oss << "\n  at_worst_timestep:";

    for (const auto& link : attr.link_stats_at_worst_timestep)
    {
      oss << "\n    " << link.link_name
          << ": gap=" << link.gap
          << ", mean_confidence=" << link.mean_confidence
          << ", min_confidence=" << link.min_confidence
          << ", visible_ratio=" << link.visible_ratio
          << ", inside=" << link.inside_count
          << ", outside=" << link.outside_count
          << ", total=" << link.sample_count;
    }

    std_msgs::String msg;
    msg.data = oss.str();
    return msg;
  }

  std_msgs::String makeRiskFrontierAttributionMsg(
      const AttributionResult& attr) const
  {
    std::ostringstream oss;

    oss << "trajectory_risk_frontier_attribution:"
        << "\n  ignored_risk_links=" << ignoredRiskLinksToString()
        << "\n  has_risk_frontier=" << attr.has_risk_frontier
        << "\n  risk_frontier_threshold=" << attr.risk_frontier_threshold
        << "\n  first_risky_eval_timestep="
        << attr.first_risky_eval_timestep
        << "\n  first_risky_original_timestep="
        << attr.first_risky_original_timestep
        << "\n  safe_prefix_end_eval_timestep="
        << attr.safe_prefix_end_eval_timestep
        << "\n  safe_prefix_end_original_timestep="
        << attr.safe_prefix_end_original_timestep
        << "\n  risk_frontier_window_eval=["
        << attr.risk_frontier_window_start_eval << ","
        << attr.risk_frontier_window_end_eval << "]"
        << "\n  risk_frontier_window_original=["
        << attr.risk_frontier_window_start_original << ","
        << attr.risk_frontier_window_end_original << "]"
        << "\n  target_centroid_base=["
        << attr.risk_frontier_centroid_base.x() << ","
        << attr.risk_frontier_centroid_base.y() << ","
        << attr.risk_frontier_centroid_base.z() << "]"
        << "\n  target_radius=" << attr.risk_frontier_radius
        << "\n  mean_gap=" << attr.risk_frontier_mean_gap
        << "\n  total_weight=" << attr.risk_frontier_total_weight
        << "\n  sample_count=" << attr.risk_frontier_samples.size()
        << "\n  detection_mode=sample_temporal"
        << "\n  primary_group:"
        << "\n    role=required_immediate_active_sensing_target"
        << "\n    eval_window=[" << attr.risk_frontier_window_start_eval
        << "," << attr.risk_frontier_window_end_eval << "]"
        << "\n    original_window=[" << attr.risk_frontier_window_start_original
        << "," << attr.risk_frontier_window_end_original << "]"
        << "\n    centroid_base=["
        << attr.risk_frontier_centroid_base.x() << ","
        << attr.risk_frontier_centroid_base.y() << ","
        << attr.risk_frontier_centroid_base.z() << "]"
        << "\n    radius=" << attr.risk_frontier_radius
        << "\n    sample_count=" << attr.risk_frontier_samples.size()
        << "\n  secondary_group:"
        << "\n    role=optional_long_horizon_bonus"
        << "\n    has_secondary=" << attr.has_secondary_frontier
        << "\n    eval_window=[" << attr.secondary_frontier_window_start_eval
        << "," << attr.secondary_frontier_window_end_eval << "]"
        << "\n    original_window=[" << attr.secondary_frontier_window_start_original
        << "," << attr.secondary_frontier_window_end_original << "]"
        << "\n    centroid_base=["
        << attr.secondary_frontier_centroid_base.x() << ","
        << attr.secondary_frontier_centroid_base.y() << ","
        << attr.secondary_frontier_centroid_base.z() << "]"
        << "\n    radius=" << attr.secondary_frontier_radius
        << "\n    sample_count=" << attr.secondary_frontier_samples.size()
        << "\n  link_groups:";

    for (const auto& link : attr.risk_frontier_link_stats)
    {
      oss << "\n    " << link.link_name
          << ": gap=" << link.gap
          << ", mean_confidence=" << link.mean_confidence
          << ", min_confidence=" << link.min_confidence
          << ", visible_ratio=" << link.visible_ratio
          << ", inside=" << link.inside_count
          << ", outside=" << link.outside_count
          << ", total=" << link.sample_count;
    }

    oss << "\n  samples:";
    int rank = 0;
    for (const auto& item : attr.risk_frontier_samples)
    {
      oss << "\n    #" << rank
          << ": eval_t=" << item.eval_timestep
          << ", original_t=" << item.original_timestep
          << ", link=" << item.link_name
          << ", sample_index_in_link=" << item.sample_index_in_link
          << ", source_collision_index=" << item.source_collision_index
          << ", confidence=" << item.confidence
          << ", gap=" << item.gap
          << ", current_visibility=" << item.current_visibility
          << ", inside=" << item.inside
          << ", position_base=["
          << item.center_base.x() << ","
          << item.center_base.y() << ","
          << item.center_base.z() << "]";
      rank += 1;
    }

    std_msgs::String msg;
    msg.data = oss.str();
    return msg;
  }

  std_msgs::String makeTopKSampleAttributionMsg(
      const AttributionResult& attr) const
  {
    std::ostringstream oss;

    oss << "trajectory_topk_sample_attribution:"
        << "\n  ignored_risk_links=" << ignoredRiskLinksToString()
        << "\n  top_k=" << attr.topk_low_confidence_samples.size()
        << "\n  samples:";

    int rank = 0;
    for (const auto& item : attr.topk_low_confidence_samples)
    {
      oss << "\n    #" << rank
          << ": eval_t=" << item.eval_timestep
          << ", original_t=" << item.original_timestep
          << ", link=" << item.link_name
          << ", sample_index_in_link=" << item.sample_index_in_link
          << ", source_collision_index=" << item.source_collision_index
          << ", confidence=" << item.confidence
          << ", gap=" << item.gap
          << ", current_visibility=" << item.current_visibility
          << ", inside=" << item.inside
          << ", position_base=["
          << item.center_base.x() << ","
          << item.center_base.y() << ","
          << item.center_base.z() << "]";
      rank += 1;
    }

    oss << "\n  worst_timestep_worst_link="
        << attr.worst_link_at_worst_timestep
        << "\n  worst_timestep_worst_link_samples=";

    rank = 0;
    for (const auto& item : attr.worst_timestep_worst_link_samples)
    {
      oss << "\n    #" << rank
          << ": eval_t=" << item.eval_timestep
          << ", original_t=" << item.original_timestep
          << ", link=" << item.link_name
          << ", sample_index_in_link=" << item.sample_index_in_link
          << ", source_collision_index=" << item.source_collision_index
          << ", confidence=" << item.confidence
          << ", gap=" << item.gap
          << ", current_visibility=" << item.current_visibility
          << ", inside=" << item.inside
          << ", position_base=["
          << item.center_base.x() << ","
          << item.center_base.y() << ","
          << item.center_base.z() << "]";
      rank += 1;
    }

    std_msgs::String msg;
    msg.data = oss.str();
    return msg;
  }

  void publishForbiddenSpaceSweepPairs(
      const care_confidence_map::TrajectorySampleResult& sample_result,
      const care_confidence_map::QueryConfidence& srv,
      const RiskResult& risk_result,
      const ros::Time& source_trajectory_stamp)
  {
    if (!forbidden_space_pair_publish_enabled_)
    {
      return;
    }

    std::size_t selected_count = 0;
    std::size_t flat_index = 0;
    for (const auto& frame : sample_result.frames)
    {
      for (const auto& sample : frame.samples)
      {
        const bool inside = (srv.response.inside_map[flat_index] != 0);
        const double confidence =
            static_cast<double>(srv.response.confidence[flat_index]);

        // C5.2: export every in-map risk-body sample as a spatial anchor.
        // The anchor center itself is NOT a forbidden CDF point.  The shadow
        // diagnostic uses it only to retrieve nearby low-confidence voxel
        // centers from the confidence map.
        if (!isIgnoredRiskLink(sample.link_name) && inside)
        {
          selected_count += 1;
        }
        flat_index += 1;
      }
    }

    sensor_msgs::PointCloud2 cloud;
    cloud.header.frame_id = base_frame_;
    cloud.header.stamp = source_trajectory_stamp.isZero()
        ? ros::Time::now()
        : source_trajectory_stamp;
    cloud.height = 1;
    cloud.width = static_cast<uint32_t>(selected_count);

    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2Fields(
        15,
        "x", 1, sensor_msgs::PointField::FLOAT32,
        "y", 1, sensor_msgs::PointField::FLOAT32,
        "z", 1, sensor_msgs::PointField::FLOAT32,
        "q0", 1, sensor_msgs::PointField::FLOAT32,
        "q1", 1, sensor_msgs::PointField::FLOAT32,
        "q2", 1, sensor_msgs::PointField::FLOAT32,
        "q3", 1, sensor_msgs::PointField::FLOAT32,
        "q4", 1, sensor_msgs::PointField::FLOAT32,
        "q5", 1, sensor_msgs::PointField::FLOAT32,
        "q6", 1, sensor_msgs::PointField::FLOAT32,
        "confidence", 1, sensor_msgs::PointField::FLOAT32,
        "current_visibility", 1, sensor_msgs::PointField::FLOAT32,
        "radius", 1, sensor_msgs::PointField::FLOAT32,
        "eval_timestep", 1, sensor_msgs::PointField::INT32,
        "original_timestep", 1, sensor_msgs::PointField::INT32);
    modifier.resize(selected_count);

    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");
    sensor_msgs::PointCloud2Iterator<float> iter_q0(cloud, "q0");
    sensor_msgs::PointCloud2Iterator<float> iter_q1(cloud, "q1");
    sensor_msgs::PointCloud2Iterator<float> iter_q2(cloud, "q2");
    sensor_msgs::PointCloud2Iterator<float> iter_q3(cloud, "q3");
    sensor_msgs::PointCloud2Iterator<float> iter_q4(cloud, "q4");
    sensor_msgs::PointCloud2Iterator<float> iter_q5(cloud, "q5");
    sensor_msgs::PointCloud2Iterator<float> iter_q6(cloud, "q6");
    sensor_msgs::PointCloud2Iterator<float> iter_conf(cloud, "confidence");
    sensor_msgs::PointCloud2Iterator<float> iter_vis(
        cloud, "current_visibility");
    sensor_msgs::PointCloud2Iterator<float> iter_radius(cloud, "radius");
    sensor_msgs::PointCloud2Iterator<int32_t> iter_eval(
        cloud, "eval_timestep");
    sensor_msgs::PointCloud2Iterator<int32_t> iter_original(
        cloud, "original_timestep");

    flat_index = 0;
    for (const auto& frame : sample_result.frames)
    {
      int original_timestep = -1;
      if (frame.timestep_index >= 0 &&
          frame.timestep_index <
              static_cast<int>(risk_result.eval_to_original_index.size()))
      {
        original_timestep =
            risk_result.eval_to_original_index[
                static_cast<std::size_t>(frame.timestep_index)];
      }

      for (const auto& sample : frame.samples)
      {
        const bool inside = (srv.response.inside_map[flat_index] != 0);
        const float confidence =
            static_cast<float>(srv.response.confidence[flat_index]);
        const float current_visibility =
            static_cast<float>(srv.response.current_visibility[flat_index]);

        if (!isIgnoredRiskLink(sample.link_name) && inside)
        {
          if (frame.q.size() != 7)
          {
            ROS_ERROR_THROTTLE(
                1.0,
                "[trajectory_risk_node] C5.2 body-sweep anchor export "
                "requires 7-DoF q, got %ld",
                static_cast<long>(frame.q.size()));
            return;
          }

          *iter_x = static_cast<float>(sample.center_base.x());
          *iter_y = static_cast<float>(sample.center_base.y());
          *iter_z = static_cast<float>(sample.center_base.z());
          *iter_q0 = static_cast<float>(frame.q(0));
          *iter_q1 = static_cast<float>(frame.q(1));
          *iter_q2 = static_cast<float>(frame.q(2));
          *iter_q3 = static_cast<float>(frame.q(3));
          *iter_q4 = static_cast<float>(frame.q(4));
          *iter_q5 = static_cast<float>(frame.q(5));
          *iter_q6 = static_cast<float>(frame.q(6));
          *iter_conf = confidence;
          *iter_vis = current_visibility;
          *iter_radius = static_cast<float>(
              sample.radius + forbidden_space_body_inflation_m_);
          *iter_eval = static_cast<int32_t>(frame.timestep_index);
          *iter_original = static_cast<int32_t>(original_timestep);

          ++iter_x; ++iter_y; ++iter_z;
          ++iter_q0; ++iter_q1; ++iter_q2; ++iter_q3;
          ++iter_q4; ++iter_q5; ++iter_q6;
          ++iter_conf; ++iter_vis; ++iter_radius;
          ++iter_eval; ++iter_original;
        }

        flat_index += 1;
      }
    }

    forbidden_space_pair_pub_.publish(cloud);
  }

  void publishRiskStats(const RiskResult& result)
  {
    score_pub_.publish(makeFloatMsg(result.score));
    mean_confidence_pub_.publish(makeFloatMsg(result.mean_confidence));
    min_confidence_pub_.publish(makeFloatMsg(result.min_confidence));
    visible_ratio_pub_.publish(makeFloatMsg(result.visible_ratio));
    worst_timestep_pub_.publish(makeIntMsg(result.worst_risk_timestep));
    summary_pub_.publish(makeRiskSummaryMsg(result));
    timestep_summary_pub_.publish(makeTimestepSummaryMsg(result));
    worst_timestep_summary_pub_.publish(makeWorstTimestepSummaryMsg(result));
  }

  void publishTimingStats(
      const RiskResult& result,
      const TimingStats& timing)
  {
    eval_time_pub_.publish(makeFloatMsg(timing.total_time_ms));
    fk_time_pub_.publish(makeFloatMsg(timing.fk_time_ms));
    query_time_pub_.publish(makeFloatMsg(timing.query_time_ms));
    risk_compute_time_pub_.publish(makeFloatMsg(timing.risk_compute_time_ms));
    marker_time_pub_.publish(makeFloatMsg(timing.marker_time_ms));
    input_age_pub_.publish(makeFloatMsg(timing.input_age_ms));
    timing_summary_pub_.publish(makeTimingSummaryMsg(result, timing));
  }

  void publishAttribution(
      const AttributionResult& attr,
      const RiskResult& result)
  {
    attribution_summary_pub_.publish(makeAttributionSummaryMsg(attr));
    timestep_attribution_pub_.publish(makeTimestepAttributionMsg(result));
    link_attribution_pub_.publish(makeLinkAttributionMsg(attr));
    topk_sample_attribution_pub_.publish(makeTopKSampleAttributionMsg(attr));
    risk_frontier_attribution_pub_.publish(
        makeRiskFrontierAttributionMsg(attr));
  }

  void publishActiveSensingTarget(const AttributionResult& attr)
  {
    if (!attr.success || !attr.has_risk_frontier ||
        attr.risk_frontier_samples.empty())
    {
      return;
    }

    const SampleAttributionItem* target = nullptr;
    for (const auto& item : attr.risk_frontier_samples)
    {
      if (item.inside && item.current_visibility <= 0.5)
      {
        target = &item;
        break;
      }
    }

    if (!target)
    {
      return;
    }

    geometry_msgs::PointStamped msg;
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = base_frame_;
    msg.point = toPointMsg(target->center_base);
    active_sensing_target_pub_.publish(msg);

    ROS_INFO_THROTTLE(
        1.0,
        "[trajectory_risk_node] active-sensing target: x=%.3f y=%.3f z=%.3f, confidence=%.3f, visibility=%.3f, link=%s, original_t=%d",
        target->center_base.x(),
        target->center_base.y(),
        target->center_base.z(),
        target->confidence,
        target->current_visibility,
        target->link_name.c_str(),
        target->original_timestep);
  }

  bool shouldPublishMarkers()
  {
    if (marker_publish_rate_ <= 0.0)
    {
      return false;
    }

    const ros::Time now = ros::Time::now();

    if (last_marker_publish_time_.isZero())
    {
      last_marker_publish_time_ = now;
      return true;
    }

    const double dt = (now - last_marker_publish_time_).toSec();
    const double period = 1.0 / marker_publish_rate_;

    if (dt >= period)
    {
      last_marker_publish_time_ = now;
      return true;
    }

    return false;
  }

  void publishFullTrajectoryMarkers(
      const care_confidence_map::TrajectorySampleResult& sample_result,
      const care_confidence_map::QueryConfidence& srv)
  {
    visualization_msgs::MarkerArray array;
    array.markers.push_back(
        makeDeleteAllMarker("trajectory_risk/full_trajectory"));

    if (!show_full_trajectory_markers_)
    {
      full_trajectory_marker_pub_.publish(array);
      return;
    }

    int marker_id = 1;
    std::size_t flat_index = 0;

    for (const auto& frame : sample_result.frames)
    {
      for (const auto& sample : frame.samples)
      {
        const double confidence =
            static_cast<double>(srv.response.confidence[flat_index]);

        const bool inside =
            (srv.response.inside_map[flat_index] != 0);

        array.markers.push_back(
            makeSampleMarker(
                sample,
                "trajectory_risk/full_trajectory",
                marker_id++,
                confidence,
                inside,
                false));

        flat_index += 1;
      }
    }

    full_trajectory_marker_pub_.publish(array);
  }

  void publishWorstTimestepMarkers(
      const care_confidence_map::TrajectorySampleResult& sample_result,
      const care_confidence_map::QueryConfidence& srv,
      const RiskResult& result)
  {
    visualization_msgs::MarkerArray array;
    array.markers.push_back(
        makeDeleteAllMarker("trajectory_risk/worst_timestep"));

    if (!show_worst_timestep_markers_ ||
        result.worst_risk_timestep < 0)
    {
      worst_timestep_marker_pub_.publish(array);
      return;
    }

    int marker_id = 1;
    std::size_t flat_index = 0;

    for (const auto& frame : sample_result.frames)
    {
      for (const auto& sample : frame.samples)
      {
        const double confidence =
            static_cast<double>(srv.response.confidence[flat_index]);

        const bool inside =
            (srv.response.inside_map[flat_index] != 0);

        if (frame.timestep_index == result.worst_risk_timestep)
        {
          array.markers.push_back(
              makeSampleMarker(
                  sample,
                  "trajectory_risk/worst_timestep",
                  marker_id++,
                  confidence,
                  inside,
                  true));
        }

        flat_index += 1;
      }
    }

    array.markers.push_back(
        makeWorstTimestepTextMarker(result, marker_id++));

    worst_timestep_marker_pub_.publish(array);
  }

  visualization_msgs::Marker makeRiskFrontierTargetMarker(
      const AttributionResult& attr,
      int marker_id) const
  {
    visualization_msgs::Marker marker;

    marker.header.frame_id = base_frame_;
    marker.header.stamp = ros::Time::now();
    marker.ns = "trajectory_risk/attribution/risk_frontier_target";
    marker.id = marker_id;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position = toPointMsg(attr.risk_frontier_centroid_base);
    marker.pose.orientation.w = 1.0;

    const double diameter =
        2.0 * std::max(0.02, attr.risk_frontier_radius);
    marker.scale.x = diameter;
    marker.scale.y = diameter;
    marker.scale.z = diameter;

    marker.color.r = 0.0f;
    marker.color.g = 0.85f;
    marker.color.b = 1.0f;
    marker.color.a = 0.35f;
    marker.lifetime = ros::Duration(0.0);
    return marker;
  }

  visualization_msgs::Marker makeRiskFrontierTextMarker(
      const AttributionResult& attr,
      int marker_id) const
  {
    visualization_msgs::Marker marker;

    marker.header.frame_id = base_frame_;
    marker.header.stamp = ros::Time::now();
    marker.ns = "trajectory_risk/attribution/risk_frontier_text";
    marker.id = marker_id;
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position = toPointMsg(attr.risk_frontier_centroid_base);
    marker.pose.position.z +=
        std::max(0.05, attr.risk_frontier_radius + 0.04);
    marker.pose.orientation.w = 1.0;
    marker.scale.z = 0.035;

    marker.color.r = 0.0f;
    marker.color.g = 0.95f;
    marker.color.b = 1.0f;
    marker.color.a = 1.0f;

    std::ostringstream oss;
    oss << "risk frontier"
        << "\nfirst eval t = " << attr.first_risky_eval_timestep
        << " / original t = " << attr.first_risky_original_timestep
        << "\nsafe prefix eval t = "
        << attr.safe_prefix_end_eval_timestep
        << "\nsamples = " << attr.risk_frontier_samples.size()
        << " / radius = " << attr.risk_frontier_radius;
    marker.text = oss.str();
    marker.lifetime = ros::Duration(0.0);
    return marker;
  }

  void publishAttributionMarkers(
      const AttributionResult& attr)
  {
    visualization_msgs::MarkerArray array;
    array.markers.push_back(
        makeDeleteAllMarker("trajectory_risk/attribution"));

    if (!show_attribution_markers_ || !attr.success)
    {
      attribution_marker_pub_.publish(array);
      return;
    }

    int marker_id = 1;

    // Global explanation only: top-K and worst-timestep/worst-link samples.
    // Primary/secondary temporal frontier groups are published on separate topics.
    for (const auto& item : attr.topk_low_confidence_samples)
    {
      array.markers.push_back(
          makeAttributionSampleMarker(
              item,
              "trajectory_risk/attribution/topk",
              marker_id++,
              true));
    }

    for (const auto& item : attr.worst_timestep_worst_link_samples)
    {
      array.markers.push_back(
          makeAttributionSampleMarker(
              item,
              "trajectory_risk/attribution/worst_timestep_worst_link",
              marker_id++,
              false));
    }

    array.markers.push_back(
        makeAttributionTextMarker(attr, marker_id++));

    attribution_marker_pub_.publish(array);
  }

  void publishPrimaryFrontierMarkers(
      const AttributionResult& attr)
  {
    visualization_msgs::MarkerArray array;
    array.markers.push_back(
        makeDeleteAllMarker("trajectory_risk/primary_frontier"));

    if (!show_attribution_markers_ || !attr.success ||
        !attr.has_risk_frontier || attr.risk_frontier_samples.empty())
    {
      primary_frontier_marker_pub_.publish(array);
      return;
    }

    int marker_id = 1;

    for (const auto& item : attr.risk_frontier_samples)
    {
      array.markers.push_back(
          makeColoredFrontierSampleMarker(
              item,
              "trajectory_risk/primary_frontier/samples",
              marker_id++,
              0.0f,
              0.95f,
              1.0f,
              static_cast<float>(attribution_marker_alpha_),
              2.7));
    }

    // Summary sphere only: centroid + bounding radius of primary samples.
    // It is not the actual sensing target used for scoring.
    array.markers.push_back(
        makeFrontierSummarySphereMarker(
            attr.risk_frontier_centroid_base,
            attr.risk_frontier_radius,
            "trajectory_risk/primary_frontier/summary_sphere",
            marker_id++,
            0.0f,
            0.85f,
            1.0f,
            0.22f));

    std::ostringstream oss;
    oss << "primary temporal frontier"
        << "\nfirst eval t = " << attr.first_risky_eval_timestep
        << " / original t = " << attr.first_risky_original_timestep
        << "\nsafe prefix eval t = " << attr.safe_prefix_end_eval_timestep
        << "\nwindow = [" << attr.risk_frontier_window_start_eval
        << "," << attr.risk_frontier_window_end_eval << "]"
        << "\nsamples = " << attr.risk_frontier_samples.size()
        << " / radius = " << attr.risk_frontier_radius;

    array.markers.push_back(
        makeFrontierGroupTextMarker(
            attr.risk_frontier_centroid_base,
            attr.risk_frontier_radius,
            "trajectory_risk/primary_frontier/text",
            marker_id++,
            oss.str(),
            0.0f,
            0.95f,
            1.0f));

    primary_frontier_marker_pub_.publish(array);
  }

  void publishSecondaryFrontierMarkers(
      const AttributionResult& attr)
  {
    visualization_msgs::MarkerArray array;
    array.markers.push_back(
        makeDeleteAllMarker("trajectory_risk/secondary_frontier"));

    if (!show_attribution_markers_ || !attr.success ||
        !attr.has_secondary_frontier || attr.secondary_frontier_samples.empty())
    {
      secondary_frontier_marker_pub_.publish(array);
      return;
    }

    int marker_id = 1;

    for (const auto& item : attr.secondary_frontier_samples)
    {
      array.markers.push_back(
          makeColoredFrontierSampleMarker(
              item,
              "trajectory_risk/secondary_frontier/samples",
              marker_id++,
              0.55f,
              0.35f,
              1.0f,
              static_cast<float>(0.65 * attribution_marker_alpha_),
              2.3));
    }

    array.markers.push_back(
        makeFrontierSummarySphereMarker(
            attr.secondary_frontier_centroid_base,
            attr.secondary_frontier_radius,
            "trajectory_risk/secondary_frontier/summary_sphere",
            marker_id++,
            0.55f,
            0.35f,
            1.0f,
            0.14f));

    std::ostringstream oss;
    oss << "secondary temporal frontier"
        << "\nwindow = [" << attr.secondary_frontier_window_start_eval
        << "," << attr.secondary_frontier_window_end_eval << "]"
        << "\nsamples = " << attr.secondary_frontier_samples.size()
        << " / radius = " << attr.secondary_frontier_radius
        << "\noptional long-horizon bonus";

    array.markers.push_back(
        makeFrontierGroupTextMarker(
            attr.secondary_frontier_centroid_base,
            attr.secondary_frontier_radius,
            "trajectory_risk/secondary_frontier/text",
            marker_id++,
            oss.str(),
            0.55f,
            0.35f,
            1.0f));

    secondary_frontier_marker_pub_.publish(array);
  }

  void publishEmptyOutputsWithError(const std::string& message)
  {
    RiskResult result;
    result.success = false;
    result.message = message;

    publishRiskStats(result);

    AttributionResult attr;
    attr.success = false;
    attr.message = message;
    publishAttribution(attr, result);

    visualization_msgs::MarkerArray full_delete;
    full_delete.markers.push_back(
        makeDeleteAllMarker("trajectory_risk/full_trajectory"));
    full_trajectory_marker_pub_.publish(full_delete);

    visualization_msgs::MarkerArray worst_delete;
    worst_delete.markers.push_back(
        makeDeleteAllMarker("trajectory_risk/worst_timestep"));
    worst_timestep_marker_pub_.publish(worst_delete);

    visualization_msgs::MarkerArray attr_delete;
    attr_delete.markers.push_back(
        makeDeleteAllMarker("trajectory_risk/attribution"));
    attribution_marker_pub_.publish(attr_delete);

    visualization_msgs::MarkerArray primary_delete;
    primary_delete.markers.push_back(
        makeDeleteAllMarker("trajectory_risk/primary_frontier"));
    primary_frontier_marker_pub_.publish(primary_delete);

    visualization_msgs::MarkerArray secondary_delete;
    secondary_delete.markers.push_back(
        makeDeleteAllMarker("trajectory_risk/secondary_frontier"));
    secondary_frontier_marker_pub_.publish(secondary_delete);
  }

  bool getLatestTrajectory(
      trajectory_msgs::JointTrajectory* traj_msg,
      ros::Time* receive_time)
  {
    std::lock_guard<std::mutex> lock(latest_traj_mutex_);

    if (!has_latest_traj_)
    {
      return false;
    }

    const double age =
        (ros::Time::now() - latest_traj_receive_time_).toSec();

    if (!evaluate_stale_trajectory_ &&
        age > stale_trajectory_timeout_)
    {
      return false;
    }

    *traj_msg = latest_traj_;
    *receive_time = latest_traj_receive_time_;
    return true;
  }

  void trajectoryCallback(
      const trajectory_msgs::JointTrajectoryConstPtr& msg)
  {
    if (!msg || msg->points.empty())
    {
      return;
    }

    {
      std::lock_guard<std::mutex> lock(latest_traj_mutex_);
      latest_traj_ = *msg;
      latest_traj_receive_time_ = ros::Time::now();
      has_latest_traj_ = true;
    }

    // C5.31: local/final GCDF geometry exports are latency-sensitive. When
    // enabled, evaluate immediately on a fresh trajectory instead of waiting
    // for the next periodic timer tick. The periodic path remains the default
    // for legacy/main risk diagnostics.
    if (event_driven_eval_)
    {
      evalTimerCallback(ros::TimerEvent());
    }
  }

  void evalTimerCallback(const ros::TimerEvent&)
  {
    const ros::WallTime total_start = ros::WallTime::now();

    trajectory_msgs::JointTrajectory traj_msg;
    ros::Time traj_receive_time;

    if (!getLatestTrajectory(&traj_msg, &traj_receive_time))
    {
      ROS_WARN_THROTTLE(
          2.0,
          "[trajectory_risk_node] Waiting for latest trajectory on %s",
          input_trajectory_topic_.c_str());
      return;
    }

    TimingStats timing;
    timing.input_age_ms =
        (ros::Time::now() - traj_receive_time).toSec() * 1000.0;

    std::vector<Eigen::VectorXd> q_traj;
    std::vector<int> eval_to_original_index;
    std::string error_msg;

    const ros::WallTime convert_start = ros::WallTime::now();

    if (!jointTrajectoryToQTrajectory(
            traj_msg,
            &q_traj,
            &eval_to_original_index,
            &error_msg))
    {
      ROS_WARN_STREAM_THROTTLE(
          2.0,
          "[trajectory_risk_node] Invalid input trajectory: "
              << error_msg);
      publishEmptyOutputsWithError(error_msg);
      return;
    }

    const ros::WallTime convert_end = ros::WallTime::now();
    timing.convert_time_ms = wallMs(convert_start, convert_end);

    const ros::WallTime fk_start = ros::WallTime::now();

    const care_confidence_map::TrajectorySampleResult sample_result =
        evaluator_.computeTrajectorySamples(q_traj);

    const ros::WallTime fk_end = ros::WallTime::now();
    timing.fk_time_ms = wallMs(fk_start, fk_end);

    if (!sample_result.success)
    {
      ROS_WARN_STREAM_THROTTLE(
          2.0,
          "[trajectory_risk_node] FK sample evaluation failed: "
              << sample_result.message);
      publishEmptyOutputsWithError(sample_result.message);
      return;
    }

    if (!refreshBodyPriorBeforeQuery())
    {
      const std::string msg =
          "Failed to refresh current-body confidence prior before trajectory query.";
      ROS_WARN_STREAM_THROTTLE(
          2.0,
          "[trajectory_risk_node] " << msg);
      publishEmptyOutputsWithError(msg);
      return;
    }

    care_confidence_map::QueryConfidence srv;

    const ros::WallTime query_start = ros::WallTime::now();

    if (!queryTrajectoryConfidence(sample_result, &srv))
    {
      const std::string msg =
          "Failed to query confidence map for trajectory samples.";
      ROS_WARN_STREAM_THROTTLE(
          2.0,
          "[trajectory_risk_node] " << msg);
      publishEmptyOutputsWithError(msg);
      return;
    }

    const ros::WallTime query_end = ros::WallTime::now();
    timing.query_time_ms = wallMs(query_start, query_end);

    const ros::WallTime risk_start = ros::WallTime::now();

    const RiskResult result =
        computeRiskResult(
            sample_result,
            srv,
            static_cast<int>(traj_msg.points.size()),
            eval_to_original_index);

    const ros::WallTime risk_end = ros::WallTime::now();
    timing.risk_compute_time_ms = wallMs(risk_start, risk_end);

    const ros::WallTime attribution_start = ros::WallTime::now();

    const AttributionResult attribution =
        computeAttributionResult(
            sample_result,
            srv,
            result);

    const ros::WallTime attribution_end = ros::WallTime::now();
    timing.attribution_time_ms =
        wallMs(attribution_start, attribution_end);

    publishRiskStats(result);
    publishAttribution(attribution, result);
    publishActiveSensingTarget(attribution);
    publishForbiddenSpaceSweepPairs(
        sample_result, srv, result, traj_msg.header.stamp);

    const ros::WallTime marker_start = ros::WallTime::now();

    if (shouldPublishMarkers())
    {
      timing.published_markers = true;
      publishFullTrajectoryMarkers(sample_result, srv);
      publishWorstTimestepMarkers(sample_result, srv, result);
      publishAttributionMarkers(attribution);
      publishPrimaryFrontierMarkers(attribution);
      publishSecondaryFrontierMarkers(attribution);
    }

    const ros::WallTime marker_end = ros::WallTime::now();
    timing.marker_time_ms = wallMs(marker_start, marker_end);

    const ros::WallTime total_end = ros::WallTime::now();
    timing.total_time_ms = wallMs(total_start, total_end);

    publishTimingStats(result, timing);

    ROS_INFO_THROTTLE(
        1.0,
        "[trajectory_risk_node] risk=%.3f, total_ms=%.2f, fk_ms=%.2f, query_ms=%.2f, attr_ms=%.2f, marker_ms=%.2f, input_steps=%d, eval_steps=%d, queried_samples=%d, risk_inside=%d, ignored_links=%s, worst_original_t=%d, worst_link=%s, attr_worst_link=%s, topk=%zu",
        result.score,
        timing.total_time_ms,
        timing.fk_time_ms,
        timing.query_time_ms,
        timing.attribution_time_ms,
        timing.marker_time_ms,
        result.input_num_timesteps,
        result.eval_num_timesteps,
        result.total_samples,
        result.inside_count,
        ignoredRiskLinksToString().c_str(),
        result.worst_risk_original_timestep,
        result.worst_link.c_str(),
        attribution.worst_link_at_worst_timestep.c_str(),
        attribution.topk_low_confidence_samples.size());
  }

  void printSummary() const
  {
    ROS_INFO_STREAM("");
    ROS_INFO_STREAM("========== trajectory_risk_node ==========");
    ROS_INFO_STREAM("robot_urdf_file: " << robot_urdf_file_);
    ROS_INFO_STREAM("body_samples_file: " << body_samples_file_);
    ROS_INFO_STREAM("base_frame: " << base_frame_);
    ROS_INFO_STREAM("input_trajectory_topic: " << input_trajectory_topic_);
    ROS_INFO_STREAM("active_sensing_target_topic: "
                    << active_sensing_target_topic_);
    ROS_INFO_STREAM("confidence_query_service: "
                    << confidence_query_service_);
    ROS_INFO_STREAM("refresh_body_prior_service: "
                    << refresh_body_prior_service_);
    ROS_INFO_STREAM("refresh_body_prior_before_query: "
                    << refresh_body_prior_before_query_);
    ROS_INFO_STREAM("refresh_body_prior_timeout: "
                    << refresh_body_prior_timeout_);
    ROS_INFO_STREAM("eval_rate: " << eval_rate_);
    ROS_INFO_STREAM("event_driven_eval: " << event_driven_eval_);
    ROS_INFO_STREAM("max_eval_timesteps: " << max_eval_timesteps_);
    ROS_INFO_STREAM("query_timeout: " << query_timeout_);
    ROS_INFO_STREAM("marker_publish_rate: " << marker_publish_rate_);
    ROS_INFO_STREAM("marker_alpha: " << marker_alpha_);
    ROS_INFO_STREAM("worst_marker_alpha: " << worst_marker_alpha_);
    ROS_INFO_STREAM("attribution_marker_alpha: " << attribution_marker_alpha_);
    ROS_INFO_STREAM("show_full_trajectory_markers: "
                    << show_full_trajectory_markers_);
    ROS_INFO_STREAM("show_worst_timestep_markers: "
                    << show_worst_timestep_markers_);
    ROS_INFO_STREAM("show_attribution_markers: "
                    << show_attribution_markers_);
    ROS_INFO_STREAM("top_k_samples: " << top_k_samples_);
    ROS_INFO_STREAM("risk_frontier_threshold: "
                    << risk_frontier_threshold_);
    ROS_INFO_STREAM("risk_frontier_window_steps: "
                    << risk_frontier_window_steps_);
    ROS_INFO_STREAM("safe_prefix_margin_steps: "
                    << safe_prefix_margin_steps_);
    ROS_INFO_STREAM("frontier_confidence_threshold: "
                    << frontier_confidence_threshold_);
    ROS_INFO_STREAM("frontier_gap_threshold: "
                    << frontier_gap_threshold_);
    ROS_INFO_STREAM("frontier_radius_margin: "
                    << frontier_radius_margin_);
    ROS_INFO_STREAM("ignored_risk_links: " << ignoredRiskLinksToString());
    ROS_INFO_STREAM("evaluate_stale_trajectory: "
                    << evaluate_stale_trajectory_);
    ROS_INFO_STREAM("stale_trajectory_timeout: "
                    << stale_trajectory_timeout_);
    ROS_INFO_STREAM("forbidden_space_pair_publish_enabled: "
                    << forbidden_space_pair_publish_enabled_);
    ROS_INFO_STREAM("forbidden_space_pair_topic: "
                    << forbidden_space_pair_topic_);
    ROS_INFO_STREAM("forbidden_space_confidence_threshold: "
                    << forbidden_space_confidence_threshold_);
    ROS_INFO_STREAM("forbidden_space_body_inflation_m: "
                    << forbidden_space_body_inflation_m_);
    ROS_INFO_STREAM("Pinocchio nq: " << evaluator_.nq());
    ROS_INFO_STREAM("Pinocchio nv: " << evaluator_.nv());
    ROS_INFO_STREAM("active joints:");
    for (const auto& name : evaluator_.activeJointNames())
    {
      ROS_INFO_STREAM("  " << name);
    }
    ROS_INFO_STREAM("subscribed input:");
    ROS_INFO_STREAM("  " << input_trajectory_topic_);
    ROS_INFO_STREAM("published active-sensing target:");
    ROS_INFO_STREAM("  " << active_sensing_target_topic_);
    ROS_INFO_STREAM("published attribution topics:");
    ROS_INFO_STREAM("  /care_planner/trajectory_risk/attribution_summary");
    ROS_INFO_STREAM("  /care_planner/trajectory_risk/timestep_attribution");
    ROS_INFO_STREAM("  /care_planner/trajectory_risk/link_attribution");
    ROS_INFO_STREAM("  /care_planner/trajectory_risk/topk_sample_attribution");
    ROS_INFO_STREAM("  /care_planner/trajectory_risk/risk_frontier_attribution");
    ROS_INFO_STREAM("  /care_planner/trajectory_risk/attribution_markers");
    ROS_INFO_STREAM("  /care_planner/trajectory_risk/primary_frontier_markers");
    ROS_INFO_STREAM("  /care_planner/trajectory_risk/secondary_frontier_markers");
    ROS_INFO_STREAM("==========================================");
  }

private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  ros::Subscriber trajectory_sub_;
  ros::ServiceClient confidence_query_client_;
  ros::ServiceClient refresh_body_prior_client_;

  ros::Publisher score_pub_;
  ros::Publisher mean_confidence_pub_;
  ros::Publisher min_confidence_pub_;
  ros::Publisher visible_ratio_pub_;
  ros::Publisher worst_timestep_pub_;
  ros::Publisher summary_pub_;
  ros::Publisher timestep_summary_pub_;
  ros::Publisher worst_timestep_summary_pub_;
  ros::Publisher full_trajectory_marker_pub_;
  ros::Publisher worst_timestep_marker_pub_;
  ros::Publisher attribution_marker_pub_;
  ros::Publisher primary_frontier_marker_pub_;
  ros::Publisher secondary_frontier_marker_pub_;
  ros::Publisher active_sensing_target_pub_;

  ros::Publisher eval_time_pub_;
  ros::Publisher fk_time_pub_;
  ros::Publisher query_time_pub_;
  ros::Publisher risk_compute_time_pub_;
  ros::Publisher marker_time_pub_;
  ros::Publisher input_age_pub_;
  ros::Publisher timing_summary_pub_;

  ros::Publisher attribution_summary_pub_;
  ros::Publisher timestep_attribution_pub_;
  ros::Publisher link_attribution_pub_;
  ros::Publisher topk_sample_attribution_pub_;
  ros::Publisher risk_frontier_attribution_pub_;
  ros::Publisher forbidden_space_pair_pub_;

  ros::Timer eval_timer_;

  care_confidence_map::TrajectoryRiskEvaluator evaluator_;

  std::mutex latest_traj_mutex_;
  trajectory_msgs::JointTrajectory latest_traj_;
  ros::Time latest_traj_receive_time_;
  bool has_latest_traj_ = false;

  ros::Time last_marker_publish_time_;

  std::string robot_urdf_file_;
  std::string body_samples_file_;
  std::string base_frame_ = "base_link";
  std::string input_trajectory_topic_ =
      "/care_planner/task_trajectory";
  std::string active_sensing_target_topic_ =
      "/care_planner/active_sensing/target_point";
  std::string output_namespace_ = "/care_planner/trajectory_risk";
  std::string confidence_query_service_ =
      "/care_planner/confidence_map/query";
  std::string refresh_body_prior_service_ =
      "/care_planner/confidence_map/refresh_body_prior";
  std::string forbidden_space_pair_topic_ =
      "/care_planner/trajectory_risk/body_sweep_anchors";

  double eval_rate_ = 20.0;
  int max_eval_timesteps_ = 12;

  double query_timeout_ = 0.10;
  double refresh_body_prior_timeout_ = 0.10;
  double marker_alpha_ = 0.45;
  double worst_marker_alpha_ = 0.85;
  double attribution_marker_alpha_ = 0.90;
  double marker_publish_rate_ = 2.0;
  double stale_trajectory_timeout_ = 1.0;

  int top_k_samples_ = 20;
  int risk_frontier_window_steps_ = 5;
  int safe_prefix_margin_steps_ = 2;
  double risk_frontier_threshold_ = 0.30;
  double frontier_confidence_threshold_ = 0.50;
  double frontier_gap_threshold_ = 0.50;
  double frontier_radius_margin_ = 0.05;
  double forbidden_space_confidence_threshold_ = 0.50;
  double forbidden_space_body_inflation_m_ = 0.0;
  std::vector<std::string> ignored_risk_links_;

  bool refresh_body_prior_before_query_ = false;
  bool event_driven_eval_ = false;
  bool forbidden_space_pair_publish_enabled_ = false;
  bool show_full_trajectory_markers_ = false;
  bool show_worst_timestep_markers_ = true;
  bool show_attribution_markers_ = true;
  bool evaluate_stale_trajectory_ = true;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "trajectory_risk_node");

  TrajectoryRiskNode node;
  if (!node.initialize())
  {
    ROS_ERROR("[trajectory_risk_node] Initialization failed.");
    return 1;
  }

  ros::spin();
  return 0;
}
