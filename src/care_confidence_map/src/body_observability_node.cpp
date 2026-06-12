#include <care_confidence_map/body_sample_model.hpp>
#include <care_confidence_map/QueryConfidence.h>

#include <ros/ros.h>

#include <geometry_msgs/Point.h>
#include <geometry_msgs/TransformStamped.h>
#include <std_msgs/Float32.h>
#include <std_msgs/String.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <algorithm>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

class BodyObservabilityNode
{
public:
  BodyObservabilityNode()
    : nh_()
    , pnh_("~")
    , tf_buffer_()
    , tf_listener_(tf_buffer_)
  {
  }

  bool initialize()
  {
    loadParams();

    std::string error_msg;
    if (!model_.loadFromYaml(body_samples_file_, &error_msg))
    {
      ROS_ERROR_STREAM("[body_observability_node] Failed to load body samples: "
                       << error_msg);
      return false;
    }

    query_client_ =
        nh_.serviceClient<care_confidence_map::QueryConfidence>(
            confidence_query_service_);

    score_pub_ =
        nh_.advertise<std_msgs::Float32>(
            "/care_planner/body_observability/score", 1, true);

    min_confidence_pub_ =
        nh_.advertise<std_msgs::Float32>(
            "/care_planner/body_observability/min_confidence", 1, true);

    visible_ratio_pub_ =
        nh_.advertise<std_msgs::Float32>(
            "/care_planner/body_observability/visible_ratio", 1, true);

    summary_pub_ =
        nh_.advertise<std_msgs::String>(
            "/care_planner/body_observability/summary", 1, true);

    link_summary_pub_ =
        nh_.advertise<std_msgs::String>(
            "/care_planner/body_observability/link_summary", 1, true);

    marker_pub_ =
        nh_.advertise<visualization_msgs::MarkerArray>(
            "/care_planner/body_observability/markers", 1, true);

    const double period = 1.0 / std::max(0.1, publish_rate_);
    timer_ = nh_.createTimer(
        ros::Duration(period),
        &BodyObservabilityNode::timerCallback,
        this);

    printSummary();
    return true;
  }

private:
  struct TransformedSample
  {
    care_confidence_map::BodySample sample;
    geometry_msgs::Point point_base;
  };

  struct ScoreResult
  {
    int total_count = 0;
    int inside_count = 0;
    int outside_count = 0;
    int visible_count = 0;

    double mean_confidence = 0.0;
    double min_confidence = 0.0;
    double visible_ratio = 0.0;

    std::string worst_link = "none";
    int worst_sample_index_in_link = -1;
    int worst_source_collision_index = -1;
    double worst_confidence = 1.0;
  };

  struct LinkScore
  {
    int total_count = 0;
    int inside_count = 0;
    int outside_count = 0;
    int visible_count = 0;

    double confidence_sum = 0.0;
    double mean_confidence = 0.0;
    double min_confidence = std::numeric_limits<double>::infinity();

    std::string worst_source_type = "none";
    int worst_sample_index_in_link = -1;
    int worst_source_collision_index = -1;
    double worst_confidence = 1.0;
  };

  void loadParams()
  {
    pnh_.param<std::string>(
        "body_observability/body_samples_file",
        body_samples_file_,
        "");

    pnh_.param<std::string>(
        "body_observability/map_frame",
        map_frame_,
        "base_link");

    pnh_.param<std::string>(
        "body_observability/confidence_query_service",
        confidence_query_service_,
        "/care_planner/confidence_map/query");

    pnh_.param(
        "body_observability/publish_rate",
        publish_rate_,
        10.0);

    pnh_.param(
        "body_observability/query_timeout",
        query_timeout_,
        0.05);

    pnh_.param(
        "body_observability/use_risk_samples_only",
        use_risk_samples_only_,
        true);

    pnh_.param(
        "body_observability/publish_markers",
        publish_markers_,
        true);

    pnh_.param(
        "body_observability/marker_alpha",
        marker_alpha_,
        0.75);
  }

  tf2::Transform transformMsgToTf2(
      const geometry_msgs::TransformStamped& msg) const
  {
    tf2::Transform T;
    tf2::fromMsg(msg.transform, T);
    return T;
  }

  geometry_msgs::Point toPointMsg(const tf2::Vector3& p) const
  {
    geometry_msgs::Point out;
    out.x = p.x();
    out.y = p.y();
    out.z = p.z();
    return out;
  }

  bool transformBodySamples(std::vector<TransformedSample>* out)
  {
    out->clear();

    std::map<std::string, tf2::Transform> T_map_frame;

    for (const auto& frame : model_.frames())
    {
      try
      {
        geometry_msgs::TransformStamped tf_msg =
            tf_buffer_.lookupTransform(
                map_frame_,
                frame,
                ros::Time(0),
                ros::Duration(0.005));

        T_map_frame[frame] = transformMsgToTf2(tf_msg);
      }
      catch (const tf2::TransformException& ex)
      {
        ROS_WARN_THROTTLE(
            2.0,
            "[body_observability_node] Missing TF %s -> %s: %s",
            map_frame_.c_str(),
            frame.c_str(),
            ex.what());
      }
    }

    for (const auto& sample : model_.samples())
    {
      if (use_risk_samples_only_ && !sample.include_for_risk)
      {
        continue;
      }

      const auto it = T_map_frame.find(sample.frame_name);
      if (it == T_map_frame.end())
      {
        continue;
      }

      const tf2::Vector3 center_base =
          it->second * sample.center_link;

      TransformedSample ts;
      ts.sample = sample;
      ts.point_base = toPointMsg(center_base);

      out->push_back(ts);
    }

    return !out->empty();
  }

  bool queryConfidence(
      const std::vector<TransformedSample>& transformed_samples,
      care_confidence_map::QueryConfidence* srv)
  {
    srv->request.points.clear();
    srv->request.points.reserve(transformed_samples.size());

    for (const auto& ts : transformed_samples)
    {
      srv->request.points.push_back(ts.point_base);
    }

    if (!query_client_.waitForExistence(ros::Duration(query_timeout_)))
    {
      ROS_WARN_THROTTLE(
          2.0,
          "[body_observability_node] Waiting for service: %s",
          confidence_query_service_.c_str());
      return false;
    }

    if (!query_client_.call(*srv))
    {
      ROS_WARN_THROTTLE(
          2.0,
          "[body_observability_node] Failed to call service: %s",
          confidence_query_service_.c_str());
      return false;
    }

    const std::size_t n = transformed_samples.size();

    if (srv->response.confidence.size() != n ||
        srv->response.current_visibility.size() != n ||
        srv->response.inside_map.size() != n)
    {
      ROS_WARN_THROTTLE(
          2.0,
          "[body_observability_node] Invalid query response size. "
          "request=%zu, confidence=%zu, visibility=%zu, inside_map=%zu",
          n,
          srv->response.confidence.size(),
          srv->response.current_visibility.size(),
          srv->response.inside_map.size());
      return false;
    }

    return true;
  }

  ScoreResult computeScore(
      const std::vector<TransformedSample>& transformed_samples,
      const care_confidence_map::QueryConfidence& srv) const
  {
    ScoreResult result;
    result.total_count = static_cast<int>(transformed_samples.size());

    double confidence_sum = 0.0;
    double min_confidence = std::numeric_limits<double>::infinity();

    for (std::size_t i = 0; i < transformed_samples.size(); ++i)
    {
      const auto& ts = transformed_samples[i];

      const bool inside = (srv.response.inside_map[i] != 0);
      const double confidence =
          static_cast<double>(srv.response.confidence[i]);
      const double visible =
          static_cast<double>(srv.response.current_visibility[i]);

      if (!inside)
      {
        result.outside_count += 1;
        continue;
      }

      result.inside_count += 1;
      confidence_sum += confidence;

      if (visible > 0.5)
      {
        result.visible_count += 1;
      }

      if (confidence < min_confidence)
      {
        min_confidence = confidence;
      }

      if (confidence < result.worst_confidence)
      {
        result.worst_confidence = confidence;
        result.worst_link = ts.sample.link_name;
        result.worst_sample_index_in_link = ts.sample.sample_index_in_link;
        result.worst_source_collision_index =
            ts.sample.source_collision_index;
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
    }

    return result;
  }

  std::map<std::string, LinkScore> computeLinkScores(
      const std::vector<TransformedSample>& transformed_samples,
      const care_confidence_map::QueryConfidence& srv) const
  {
    std::map<std::string, LinkScore> out;

    for (std::size_t i = 0; i < transformed_samples.size(); ++i)
    {
      const auto& ts = transformed_samples[i];
      const std::string& link_name = ts.sample.link_name;

      LinkScore& ls = out[link_name];

      const bool inside = (srv.response.inside_map[i] != 0);
      const double confidence =
          static_cast<double>(srv.response.confidence[i]);
      const double visible =
          static_cast<double>(srv.response.current_visibility[i]);

      ls.total_count += 1;

      if (!inside)
      {
        ls.outside_count += 1;
        continue;
      }

      ls.inside_count += 1;
      ls.confidence_sum += confidence;

      if (visible > 0.5)
      {
        ls.visible_count += 1;
      }

      if (confidence < ls.min_confidence)
      {
        ls.min_confidence = confidence;
      }

      if (confidence < ls.worst_confidence)
      {
        ls.worst_confidence = confidence;
        ls.worst_source_type = ts.sample.source_type;
        ls.worst_sample_index_in_link = ts.sample.sample_index_in_link;
        ls.worst_source_collision_index = ts.sample.source_collision_index;
      }
    }

    for (auto& kv : out)
    {
      LinkScore& ls = kv.second;

      if (ls.inside_count > 0)
      {
        ls.mean_confidence =
            ls.confidence_sum / static_cast<double>(ls.inside_count);
      }
      else
      {
        ls.mean_confidence = 0.0;
        ls.min_confidence = 0.0;
        ls.worst_confidence = 0.0;
      }

      if (!std::isfinite(ls.min_confidence))
      {
        ls.min_confidence = 0.0;
      }
    }

    return out;
  }

  std_msgs::Float32 makeFloatMsg(double value) const
  {
    std_msgs::Float32 msg;
    msg.data = static_cast<float>(value);
    return msg;
  }

  std_msgs::String makeSummaryMsg(const ScoreResult& result) const
  {
    std::ostringstream oss;

    oss << "body_observability: "
        << "mean_confidence=" << result.mean_confidence
        << ", min_confidence=" << result.min_confidence
        << ", visible_ratio=" << result.visible_ratio
        << ", inside=" << result.inside_count
        << ", outside=" << result.outside_count
        << ", total=" << result.total_count
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

  std_msgs::String makeLinkSummaryMsg(
      const std::map<std::string, LinkScore>& link_scores) const
  {
    std::ostringstream oss;

    oss << "body_observability_per_link:";

    for (const auto& kv : link_scores)
    {
      const std::string& link_name = kv.first;
      const LinkScore& ls = kv.second;

      const double visible_ratio =
          (ls.inside_count > 0)
              ? static_cast<double>(ls.visible_count) /
                    static_cast<double>(ls.inside_count)
              : 0.0;

      oss << "\n  " << link_name
          << ": mean=" << ls.mean_confidence
          << ", min=" << ls.min_confidence
          << ", visible_ratio=" << visible_ratio
          << ", inside=" << ls.inside_count
          << ", outside=" << ls.outside_count
          << ", total=" << ls.total_count
          << ", worst_conf=" << ls.worst_confidence
          << ", worst_sample_index_in_link="
          << ls.worst_sample_index_in_link
          << ", worst_collision_index="
          << ls.worst_source_collision_index
          << ", worst_source_type="
          << ls.worst_source_type;
    }

    std_msgs::String msg;
    msg.data = oss.str();
    return msg;
  }

  visualization_msgs::Marker makeDeleteAllMarker() const
  {
    visualization_msgs::Marker marker;
    marker.header.frame_id = map_frame_;
    marker.header.stamp = ros::Time::now();
    marker.ns = "body_observability";
    marker.id = 0;
    marker.action = visualization_msgs::Marker::DELETEALL;
    return marker;
  }

  std_msgs::ColorRGBA confidenceColor(double confidence, bool inside) const
  {
    std_msgs::ColorRGBA c;

    if (!inside)
    {
      c.r = 0.4f;
      c.g = 0.4f;
      c.b = 0.4f;
      c.a = 0.25f;
      return c;
    }

    const double q = std::max(0.0, std::min(1.0, confidence));

    c.r = static_cast<float>(1.0 - q);
    c.g = static_cast<float>(q);
    c.b = 0.05f;
    c.a = static_cast<float>(marker_alpha_);

    return c;
  }

  visualization_msgs::Marker makeSphereMarker(
      const TransformedSample& ts,
      double confidence,
      bool inside,
      int id) const
  {
    visualization_msgs::Marker marker;

    marker.header.frame_id = map_frame_;
    marker.header.stamp = ros::Time::now();

    marker.ns = "body_observability/" + ts.sample.link_name;
    marker.id = id;

    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;

    marker.pose.position = ts.point_base;
    marker.pose.orientation.w = 1.0;

    marker.scale.x = 2.0 * ts.sample.radius;
    marker.scale.y = 2.0 * ts.sample.radius;
    marker.scale.z = 2.0 * ts.sample.radius;

    marker.color = confidenceColor(confidence, inside);

    marker.lifetime =
        ros::Duration(1.0 / std::max(0.1, publish_rate_) * 2.5);

    return marker;
  }

  void publishMarkers(
      const std::vector<TransformedSample>& transformed_samples,
      const care_confidence_map::QueryConfidence& srv)
  {
    if (!publish_markers_)
    {
      return;
    }

    visualization_msgs::MarkerArray array;
    array.markers.push_back(makeDeleteAllMarker());

    int id = 1;

    for (std::size_t i = 0; i < transformed_samples.size(); ++i)
    {
      const bool inside = (srv.response.inside_map[i] != 0);
      const double confidence =
          static_cast<double>(srv.response.confidence[i]);

      array.markers.push_back(
          makeSphereMarker(transformed_samples[i],
                           confidence,
                           inside,
                           id++));
    }

    marker_pub_.publish(array);
  }

  void publishScore(
      const ScoreResult& result,
      const std::map<std::string, LinkScore>& link_scores)
  {
    score_pub_.publish(makeFloatMsg(result.mean_confidence));
    min_confidence_pub_.publish(makeFloatMsg(result.min_confidence));
    visible_ratio_pub_.publish(makeFloatMsg(result.visible_ratio));
    summary_pub_.publish(makeSummaryMsg(result));
    link_summary_pub_.publish(makeLinkSummaryMsg(link_scores));

    ROS_INFO_THROTTLE(
        2.0,
        "[body_observability_node] mean=%.3f, min=%.3f, visible=%.3f, "
        "inside=%d, outside=%d, worst_link=%s, worst_conf=%.3f",
        result.mean_confidence,
        result.min_confidence,
        result.visible_ratio,
        result.inside_count,
        result.outside_count,
        result.worst_link.c_str(),
        result.worst_confidence);

    ROS_INFO_STREAM_THROTTLE(
        5.0,
        "\n" << makeLinkSummaryMsg(link_scores).data);
  }

  void timerCallback(const ros::TimerEvent&)
  {
    std::vector<TransformedSample> transformed_samples;

    if (!transformBodySamples(&transformed_samples))
    {
      ROS_WARN_THROTTLE(
          2.0,
          "[body_observability_node] No transformed body samples available.");
      return;
    }

    care_confidence_map::QueryConfidence srv;
    if (!queryConfidence(transformed_samples, &srv))
    {
      return;
    }

    const ScoreResult result =
        computeScore(transformed_samples, srv);

    const std::map<std::string, LinkScore> link_scores =
        computeLinkScores(transformed_samples, srv);

    publishScore(result, link_scores);
    publishMarkers(transformed_samples, srv);
  }

  void printSummary() const
  {
    ROS_INFO_STREAM("");
    ROS_INFO_STREAM("========== body_observability_node ==========");
    ROS_INFO_STREAM("body_samples_file: " << body_samples_file_);
    ROS_INFO_STREAM("map_frame: " << map_frame_);
    ROS_INFO_STREAM("confidence_query_service: "
                    << confidence_query_service_);
    ROS_INFO_STREAM("publish_rate: " << publish_rate_);
    ROS_INFO_STREAM("use_risk_samples_only: " << use_risk_samples_only_);
    ROS_INFO_STREAM("publish_markers: " << publish_markers_);
    ROS_INFO_STREAM("samples loaded: " << model_.size());
    ROS_INFO_STREAM("risk samples loaded: " << model_.riskSampleCount());
    ROS_INFO_STREAM("frames: " << model_.frames().size());
    for (const auto& frame : model_.frames())
    {
      ROS_INFO_STREAM("  frame: " << frame);
    }
    ROS_INFO_STREAM("=============================================");
  }

private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  ros::ServiceClient query_client_;

  ros::Publisher score_pub_;
  ros::Publisher min_confidence_pub_;
  ros::Publisher visible_ratio_pub_;
  ros::Publisher summary_pub_;
  ros::Publisher link_summary_pub_;
  ros::Publisher marker_pub_;

  ros::Timer timer_;

  care_confidence_map::BodySampleModel model_;

  std::string body_samples_file_;
  std::string map_frame_ = "base_link";
  std::string confidence_query_service_ =
      "/care_planner/confidence_map/query";

  double publish_rate_ = 10.0;
  double query_timeout_ = 0.05;
  double marker_alpha_ = 0.75;

  bool use_risk_samples_only_ = true;
  bool publish_markers_ = true;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "body_observability_node");

  BodyObservabilityNode node;
  if (!node.initialize())
  {
    ROS_ERROR("[body_observability_node] Initialization failed.");
    return 1;
  }

  ros::spin();
  return 0;
}