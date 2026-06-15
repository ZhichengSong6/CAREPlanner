#include <care_confidence_map/trajectory_risk_evaluator.hpp>
#include <care_confidence_map/QueryConfidence.h>

#include <ros/ros.h>

#include <std_msgs/Float32.h>
#include <std_msgs/Int32.h>
#include <std_msgs/String.h>

#include <geometry_msgs/Point.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
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

    trajectory_sub_ =
        nh_.subscribe(
            input_trajectory_topic_,
            1,
            &TrajectoryRiskNode::trajectoryCallback,
            this);

    score_pub_ =
        nh_.advertise<std_msgs::Float32>(
            "/care_planner/trajectory_risk/score", 1, true);

    mean_confidence_pub_ =
        nh_.advertise<std_msgs::Float32>(
            "/care_planner/trajectory_risk/mean_confidence", 1, true);

    min_confidence_pub_ =
        nh_.advertise<std_msgs::Float32>(
            "/care_planner/trajectory_risk/min_confidence", 1, true);

    visible_ratio_pub_ =
        nh_.advertise<std_msgs::Float32>(
            "/care_planner/trajectory_risk/visible_ratio", 1, true);

    worst_timestep_pub_ =
        nh_.advertise<std_msgs::Int32>(
            "/care_planner/trajectory_risk/worst_timestep", 1, true);

    summary_pub_ =
        nh_.advertise<std_msgs::String>(
            "/care_planner/trajectory_risk/summary", 1, true);

    timestep_summary_pub_ =
        nh_.advertise<std_msgs::String>(
            "/care_planner/trajectory_risk/timestep_query_summary", 1, true);

    worst_timestep_summary_pub_ =
        nh_.advertise<std_msgs::String>(
            "/care_planner/trajectory_risk/worst_timestep_summary", 1, true);

    full_trajectory_marker_pub_ =
        nh_.advertise<visualization_msgs::MarkerArray>(
            "/care_planner/trajectory_risk/full_trajectory_markers", 1, true);

    worst_timestep_marker_pub_ =
        nh_.advertise<visualization_msgs::MarkerArray>(
            "/care_planner/trajectory_risk/worst_timestep_markers", 1, true);

    eval_time_pub_ =
        nh_.advertise<std_msgs::Float32>(
            "/care_planner/trajectory_risk/eval_time_ms", 1, true);

    fk_time_pub_ =
        nh_.advertise<std_msgs::Float32>(
            "/care_planner/trajectory_risk/fk_time_ms", 1, true);

    query_time_pub_ =
        nh_.advertise<std_msgs::Float32>(
            "/care_planner/trajectory_risk/query_time_ms", 1, true);

    risk_compute_time_pub_ =
        nh_.advertise<std_msgs::Float32>(
            "/care_planner/trajectory_risk/risk_compute_time_ms", 1, true);

    marker_time_pub_ =
        nh_.advertise<std_msgs::Float32>(
            "/care_planner/trajectory_risk/marker_time_ms", 1, true);

    input_age_pub_ =
        nh_.advertise<std_msgs::Float32>(
            "/care_planner/trajectory_risk/input_age_ms", 1, true);

    timing_summary_pub_ =
        nh_.advertise<std_msgs::String>(
            "/care_planner/trajectory_risk/timing_summary", 1, true);

    eval_timer_ =
        nh_.createTimer(
            ros::Duration(1.0 / std::max(0.1, eval_rate_)),
            &TrajectoryRiskNode::evalTimerCallback,
            this);

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
    double marker_time_ms = 0.0;
    double total_time_ms = 0.0;
    bool published_markers = false;
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
        "trajectory_risk/confidence_query_service",
        confidence_query_service_,
        "/care_planner/confidence_map/query");

    pnh_.param(
        "trajectory_risk/eval_rate",
        eval_rate_,
        20.0);

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
        "trajectory_risk/evaluate_stale_trajectory",
        evaluate_stale_trajectory_,
        true);

    pnh_.param(
        "trajectory_risk/stale_trajectory_timeout",
        stale_trajectory_timeout_,
        1.0);
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

      ts.total_count = static_cast<int>(frame.samples.size());

      for (const auto& sample : frame.samples)
      {
        const bool inside =
            (srv.response.inside_map[flat_index] != 0);

        const double confidence =
            static_cast<double>(srv.response.confidence[flat_index]);

        const double visible =
            static_cast<double>(
                srv.response.current_visibility[flat_index]);

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
      }

      ts.risk = std::max(0.0, std::min(1.0, 1.0 - ts.mean_confidence));

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
    }

    if (!std::isfinite(result.min_confidence))
    {
      result.min_confidence = 0.0;
    }

    return result;
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
        << ", mean_confidence=" << result.mean_confidence
        << ", min_confidence=" << result.min_confidence
        << ", visible_ratio=" << result.visible_ratio
        << ", input_num_timesteps=" << result.input_num_timesteps
        << ", eval_num_timesteps=" << result.eval_num_timesteps
        << ", max_eval_timesteps=" << max_eval_timesteps_
        << ", samples_per_timestep=" << result.samples_per_timestep
        << ", total_samples=" << result.total_samples
        << ", inside=" << result.inside_count
        << ", outside=" << result.outside_count
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
          << ", inside=" << ts.inside_count
          << ", outside=" << ts.outside_count
          << ", total=" << ts.total_count
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
          << ", timestep_inside=" << ts.inside_count
          << ", timestep_outside=" << ts.outside_count
          << ", timestep_total=" << ts.total_count
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
        << ", marker_ms=" << timing.marker_time_ms
        << ", convert_ms=" << timing.convert_time_ms
        << ", input_age_ms=" << timing.input_age_ms
        << ", eval_rate_target_hz=" << eval_rate_
        << ", target_period_ms=" << (1000.0 / std::max(0.1, eval_rate_))
        << ", input_num_timesteps=" << result.input_num_timesteps
        << ", eval_num_timesteps=" << result.eval_num_timesteps
        << ", total_samples=" << result.total_samples
        << ", markers_published=" << timing.published_markers;

    std_msgs::String msg;
    msg.data = oss.str();
    return msg;
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

  void publishEmptyOutputsWithError(const std::string& message)
  {
    RiskResult result;
    result.success = false;
    result.message = message;

    publishRiskStats(result);

    visualization_msgs::MarkerArray full_delete;
    full_delete.markers.push_back(
        makeDeleteAllMarker("trajectory_risk/full_trajectory"));
    full_trajectory_marker_pub_.publish(full_delete);

    visualization_msgs::MarkerArray worst_delete;
    worst_delete.markers.push_back(
        makeDeleteAllMarker("trajectory_risk/worst_timestep"));
    worst_timestep_marker_pub_.publish(worst_delete);
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
    std::lock_guard<std::mutex> lock(latest_traj_mutex_);

    latest_traj_ = *msg;
    latest_traj_receive_time_ = ros::Time::now();
    has_latest_traj_ = true;
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

    publishRiskStats(result);

    const ros::WallTime marker_start = ros::WallTime::now();

    if (shouldPublishMarkers())
    {
      timing.published_markers = true;
      publishFullTrajectoryMarkers(sample_result, srv);
      publishWorstTimestepMarkers(sample_result, srv, result);
    }

    const ros::WallTime marker_end = ros::WallTime::now();
    timing.marker_time_ms = wallMs(marker_start, marker_end);

    const ros::WallTime total_end = ros::WallTime::now();
    timing.total_time_ms = wallMs(total_start, total_end);

    publishTimingStats(result, timing);

    ROS_INFO_THROTTLE(
        1.0,
        "[trajectory_risk_node] risk=%.3f, total_ms=%.2f, fk_ms=%.2f, query_ms=%.2f, marker_ms=%.2f, input_steps=%d, eval_steps=%d, samples=%d, worst_original_t=%d, worst_link=%s",
        result.score,
        timing.total_time_ms,
        timing.fk_time_ms,
        timing.query_time_ms,
        timing.marker_time_ms,
        result.input_num_timesteps,
        result.eval_num_timesteps,
        result.total_samples,
        result.worst_risk_original_timestep,
        result.worst_link.c_str());
  }

  void printSummary() const
  {
    ROS_INFO_STREAM("");
    ROS_INFO_STREAM("========== trajectory_risk_node ==========");
    ROS_INFO_STREAM("robot_urdf_file: " << robot_urdf_file_);
    ROS_INFO_STREAM("body_samples_file: " << body_samples_file_);
    ROS_INFO_STREAM("base_frame: " << base_frame_);
    ROS_INFO_STREAM("input_trajectory_topic: " << input_trajectory_topic_);
    ROS_INFO_STREAM("confidence_query_service: "
                    << confidence_query_service_);
    ROS_INFO_STREAM("eval_rate: " << eval_rate_);
    ROS_INFO_STREAM("max_eval_timesteps: " << max_eval_timesteps_);
    ROS_INFO_STREAM("query_timeout: " << query_timeout_);
    ROS_INFO_STREAM("marker_publish_rate: " << marker_publish_rate_);
    ROS_INFO_STREAM("marker_alpha: " << marker_alpha_);
    ROS_INFO_STREAM("worst_marker_alpha: " << worst_marker_alpha_);
    ROS_INFO_STREAM("show_full_trajectory_markers: "
                    << show_full_trajectory_markers_);
    ROS_INFO_STREAM("show_worst_timestep_markers: "
                    << show_worst_timestep_markers_);
    ROS_INFO_STREAM("evaluate_stale_trajectory: "
                    << evaluate_stale_trajectory_);
    ROS_INFO_STREAM("stale_trajectory_timeout: "
                    << stale_trajectory_timeout_);
    ROS_INFO_STREAM("Pinocchio nq: " << evaluator_.nq());
    ROS_INFO_STREAM("Pinocchio nv: " << evaluator_.nv());
    ROS_INFO_STREAM("active joints:");
    for (const auto& name : evaluator_.activeJointNames())
    {
      ROS_INFO_STREAM("  " << name);
    }
    ROS_INFO_STREAM("subscribed input:");
    ROS_INFO_STREAM("  " << input_trajectory_topic_);
    ROS_INFO_STREAM("published timing topics:");
    ROS_INFO_STREAM("  /care_planner/trajectory_risk/timing_summary");
    ROS_INFO_STREAM("  /care_planner/trajectory_risk/eval_time_ms");
    ROS_INFO_STREAM("  /care_planner/trajectory_risk/fk_time_ms");
    ROS_INFO_STREAM("  /care_planner/trajectory_risk/query_time_ms");
    ROS_INFO_STREAM("==========================================");
  }

private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  ros::Subscriber trajectory_sub_;
  ros::ServiceClient confidence_query_client_;

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

  ros::Publisher eval_time_pub_;
  ros::Publisher fk_time_pub_;
  ros::Publisher query_time_pub_;
  ros::Publisher risk_compute_time_pub_;
  ros::Publisher marker_time_pub_;
  ros::Publisher input_age_pub_;
  ros::Publisher timing_summary_pub_;

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
  std::string confidence_query_service_ =
      "/care_planner/confidence_map/query";

  double eval_rate_ = 20.0;
  int max_eval_timesteps_ = 12;

  double query_timeout_ = 0.10;
  double marker_alpha_ = 0.45;
  double worst_marker_alpha_ = 0.85;
  double marker_publish_rate_ = 2.0;
  double stale_trajectory_timeout_ = 1.0;

  bool show_full_trajectory_markers_ = false;
  bool show_worst_timestep_markers_ = true;
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