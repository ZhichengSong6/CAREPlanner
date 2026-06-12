#include <care_confidence_map/body_sample_model.hpp>

#include <ros/ros.h>

#include <geometry_msgs/Point.h>
#include <geometry_msgs/TransformStamped.h>
#include <std_msgs/ColorRGBA.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <map>
#include <string>
#include <vector>

class BodySampleModelNode
{
public:
  BodySampleModelNode()
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
      ROS_ERROR_STREAM("[body_sample_model_node] Failed to load body samples: "
                       << error_msg);
      return false;
    }

    marker_pub_ =
        nh_.advertise<visualization_msgs::MarkerArray>(marker_topic_, 1, true);

    buildLinkColorMap();

    const double period = 1.0 / std::max(0.1, publish_rate_);
    timer_ = nh_.createTimer(
        ros::Duration(period),
        &BodySampleModelNode::timerCallback,
        this);

    printSummary();
    return true;
  }

private:
  void loadParams()
  {
    pnh_.param<std::string>(
        "body_sample_model/body_samples_file",
        body_samples_file_,
        "");

    pnh_.param<std::string>(
        "body_sample_model/map_frame",
        map_frame_,
        "base_link");

    pnh_.param<std::string>(
        "body_sample_model/marker_topic",
        marker_topic_,
        "/care_planner/body_samples/world_markers");

    pnh_.param("body_sample_model/publish_rate", publish_rate_, 10.0);

    pnh_.param("body_sample_model/show_non_risk_samples",
               show_non_risk_samples_,
               true);

    pnh_.param("body_sample_model/show_text_labels",
               show_text_labels_,
               true);

    pnh_.param("body_sample_model/text_scale",
               text_scale_,
               0.035);

    pnh_.param("body_sample_model/marker_alpha",
               marker_alpha_,
               0.45);

    if (body_samples_file_.empty())
    {
      ROS_WARN("[body_sample_model_node] body_samples_file is empty.");
    }
  }

  void buildLinkColorMap()
  {
    link_color_map_.clear();

    const std::vector<std_msgs::ColorRGBA> palette = makePalette();

    int link_idx = 0;
    for (const auto& sample : model_.samples())
    {
      if (link_color_map_.count(sample.link_name) > 0)
      {
        continue;
      }

      std_msgs::ColorRGBA c =
          palette[static_cast<std::size_t>(link_idx) % palette.size()];

      link_color_map_[sample.link_name] = c;
      ++link_idx;
    }
  }

  std::vector<std_msgs::ColorRGBA> makePalette() const
  {
    std::vector<std_msgs::ColorRGBA> out;

    auto color = [](float r, float g, float b, float a)
    {
      std_msgs::ColorRGBA c;
      c.r = r;
      c.g = g;
      c.b = b;
      c.a = a;
      return c;
    };

    out.push_back(color(0.20f, 0.65f, 1.00f, static_cast<float>(marker_alpha_)));
    out.push_back(color(1.00f, 0.55f, 0.15f, static_cast<float>(marker_alpha_)));
    out.push_back(color(0.25f, 0.90f, 0.45f, static_cast<float>(marker_alpha_)));
    out.push_back(color(0.90f, 0.30f, 0.90f, static_cast<float>(marker_alpha_)));
    out.push_back(color(1.00f, 0.85f, 0.20f, static_cast<float>(marker_alpha_)));
    out.push_back(color(0.20f, 0.95f, 0.95f, static_cast<float>(marker_alpha_)));
    out.push_back(color(0.95f, 0.35f, 0.35f, static_cast<float>(marker_alpha_)));
    out.push_back(color(0.65f, 0.65f, 0.65f, static_cast<float>(marker_alpha_)));

    return out;
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

  bool lookupFrameTransforms(
      std::map<std::string, tf2::Transform>& T_map_frame)
  {
    T_map_frame.clear();

    int ready_count = 0;

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
        ++ready_count;
      }
      catch (const tf2::TransformException& ex)
      {
        ROS_WARN_THROTTLE(
            2.0,
            "[body_sample_model_node] Missing TF %s -> %s: %s",
            map_frame_.c_str(),
            frame.c_str(),
            ex.what());
      }
    }

    if (ready_count != last_ready_count_)
    {
      ROS_INFO_STREAM("[body_sample_model_node] Body sample TF ready: "
                      << ready_count << " / " << model_.frames().size());
      last_ready_count_ = ready_count;
    }

    return ready_count > 0;
  }

  visualization_msgs::Marker makeDeleteAllMarker() const
  {
    visualization_msgs::Marker marker;
    marker.header.frame_id = map_frame_;
    marker.header.stamp = ros::Time::now();
    marker.ns = "body_samples_world";
    marker.id = 0;
    marker.action = visualization_msgs::Marker::DELETEALL;
    return marker;
  }

  visualization_msgs::Marker makeSphereMarker(
      const care_confidence_map::BodySample& sample,
      const tf2::Vector3& center_map,
      int marker_id) const
  {
    visualization_msgs::Marker marker;

    marker.header.frame_id = map_frame_;
    marker.header.stamp = ros::Time::now();

    marker.ns = "body_samples_world/" + sample.link_name;
    marker.id = marker_id;

    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;

    marker.pose.position = toPointMsg(center_map);
    marker.pose.orientation.w = 1.0;

    marker.scale.x = 2.0 * sample.radius;
    marker.scale.y = 2.0 * sample.radius;
    marker.scale.z = 2.0 * sample.radius;

    auto it = link_color_map_.find(sample.link_name);
    if (it != link_color_map_.end())
    {
      marker.color = it->second;
    }
    else
    {
      marker.color.r = 0.7f;
      marker.color.g = 0.7f;
      marker.color.b = 0.7f;
      marker.color.a = static_cast<float>(marker_alpha_);
    }

    if (!sample.include_for_risk)
    {
      marker.color.r = 0.45f;
      marker.color.g = 0.45f;
      marker.color.b = 0.45f;
      marker.color.a = 0.25f;
    }

    marker.lifetime = ros::Duration(1.0 / std::max(0.1, publish_rate_) * 2.5);

    return marker;
  }

  visualization_msgs::Marker makeTextMarker(
      const std::string& link_name,
      const tf2::Vector3& center_map,
      int marker_id) const
  {
    visualization_msgs::Marker marker;

    marker.header.frame_id = map_frame_;
    marker.header.stamp = ros::Time::now();

    marker.ns = "body_samples_world_labels";
    marker.id = marker_id;

    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::Marker::ADD;

    marker.pose.position = toPointMsg(center_map + tf2::Vector3(0.0, 0.0, 0.04));
    marker.pose.orientation.w = 1.0;

    marker.scale.z = text_scale_;

    marker.color.r = 1.0f;
    marker.color.g = 1.0f;
    marker.color.b = 1.0f;
    marker.color.a = 0.9f;

    marker.text = link_name;

    marker.lifetime = ros::Duration(1.0 / std::max(0.1, publish_rate_) * 2.5);

    return marker;
  }

  void timerCallback(const ros::TimerEvent&)
  {
    std::map<std::string, tf2::Transform> T_map_frame;
    if (!lookupFrameTransforms(T_map_frame))
    {
      return;
    }

    visualization_msgs::MarkerArray marker_array;
    marker_array.markers.push_back(makeDeleteAllMarker());

    int marker_id = 1;

    std::map<std::string, tf2::Vector3> link_center_sum;
    std::map<std::string, int> link_center_count;

    int published_sample_count = 0;
    int skipped_sample_count = 0;

    for (const auto& sample : model_.samples())
    {
      if (!show_non_risk_samples_ && !sample.include_for_risk)
      {
        ++skipped_sample_count;
        continue;
      }

      const auto tf_it = T_map_frame.find(sample.frame_name);
      if (tf_it == T_map_frame.end())
      {
        ++skipped_sample_count;
        continue;
      }

      const tf2::Vector3 center_map =
          tf_it->second * sample.center_link;

      marker_array.markers.push_back(
          makeSphereMarker(sample, center_map, marker_id++));

      link_center_sum[sample.link_name] += center_map;
      link_center_count[sample.link_name] += 1;

      ++published_sample_count;
    }

    if (show_text_labels_)
    {
      for (const auto& kv : link_center_sum)
      {
        const std::string& link_name = kv.first;
        const int count = link_center_count[link_name];

        if (count <= 0)
        {
          continue;
        }

        const tf2::Vector3 avg_center =
            kv.second / static_cast<double>(count);

        marker_array.markers.push_back(
            makeTextMarker(link_name, avg_center, marker_id++));
      }
    }

    marker_pub_.publish(marker_array);

    ROS_INFO_THROTTLE(
        3.0,
        "[body_sample_model_node] Published %d body samples in %s frame, skipped=%d",
        published_sample_count,
        map_frame_.c_str(),
        skipped_sample_count);
  }

  void printSummary() const
  {
    ROS_INFO_STREAM("");
    ROS_INFO_STREAM("========== body_sample_model_node ==========");
    ROS_INFO_STREAM("body_samples_file: " << body_samples_file_);
    ROS_INFO_STREAM("map_frame: " << map_frame_);
    ROS_INFO_STREAM("marker_topic: " << marker_topic_);
    ROS_INFO_STREAM("publish_rate: " << publish_rate_);
    ROS_INFO_STREAM("show_non_risk_samples: " << show_non_risk_samples_);
    ROS_INFO_STREAM("show_text_labels: " << show_text_labels_);
    ROS_INFO_STREAM("samples: " << model_.size());
    ROS_INFO_STREAM("risk samples: " << model_.riskSampleCount());
    ROS_INFO_STREAM("frames: " << model_.frames().size());
    for (const auto& frame : model_.frames())
    {
      ROS_INFO_STREAM("  frame: " << frame);
    }
    ROS_INFO_STREAM("============================================");
  }

private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  ros::Publisher marker_pub_;
  ros::Timer timer_;

  care_confidence_map::BodySampleModel model_;

  std::string body_samples_file_;
  std::string map_frame_ = "base_link";
  std::string marker_topic_ = "/care_planner/body_samples/world_markers";

  double publish_rate_ = 10.0;
  double text_scale_ = 0.035;
  double marker_alpha_ = 0.45;

  bool show_non_risk_samples_ = true;
  bool show_text_labels_ = true;

  int last_ready_count_ = -1;

  std::map<std::string, std_msgs::ColorRGBA> link_color_map_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "body_sample_model_node");

  BodySampleModelNode node;
  if (!node.initialize())
  {
    ROS_ERROR("[body_sample_model_node] Initialization failed.");
    return 1;
  }

  ros::spin();
  return 0;
}