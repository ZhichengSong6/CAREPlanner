#include <ros/ros.h>

#include <care_confidence_map/body_sample_model.hpp>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/String.h>

#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <boost/bind.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{

struct Sphere
{
  tf2::Vector3 center;
  double radius = 0.0;
};

struct SensorState
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_base{
      new pcl::PointCloud<pcl::PointXYZ>()};
  ros::Time cloud_stamp;
  ros::Time received;
  std::size_t raw_points = 0;
  std::size_t voxel_points = 0;
  std::size_t self_removed = 0;
  std::size_t kept_points = 0;
  std::size_t tf_failures = 0;
  std::size_t latest_tf_fallbacks = 0;
  tf2::Vector3 sensor_origin_base{0.0, 0.0, 0.0};
  bool valid = false;
};

bool finitePoint(const pcl::PointXYZ& p)
{
  return std::isfinite(static_cast<double>(p.x)) &&
         std::isfinite(static_cast<double>(p.y)) &&
         std::isfinite(static_cast<double>(p.z));
}

}  // namespace

class TofFusionSelfFilterNode
{
public:
  TofFusionSelfFilterNode()
      : nh_()
      , pnh_("~")
      , tf_buffer_()
      , tf_listener_(tf_buffer_)
  {
  }

  bool initialize()
  {
    pnh_.param<std::string>("base_frame", base_frame_, "base_link");
    pnh_.param<std::string>(
        "output_topic",
        output_topic_,
        "/care_planner/perception/tof_fused_filtered");
    pnh_.param<std::string>(
        "summary_topic",
        summary_topic_,
        "/care_planner/perception/tof_fusion_summary");
    pnh_.param<std::string>(
        "ray_observation_topic",
        ray_observation_topic_,
        "/care_planner/perception/tof_ray_observations");
    pnh_.param<std::string>("body_samples_file", body_samples_file_, "");

    pnh_.param("input_voxel_leaf_size", input_voxel_leaf_size_, 0.03);
    pnh_.param("output_voxel_leaf_size", output_voxel_leaf_size_, 0.03);
    pnh_.param("self_filter_padding", self_filter_padding_, 0.015);
    pnh_.param(
        "sensor_origin_filter_radius",
        sensor_origin_filter_radius_,
        0.06);
    pnh_.param("sensor_timeout", sensor_timeout_, 0.25);
    pnh_.param("tf_timeout", tf_timeout_, 0.03);
    pnh_.param("allow_latest_tf_fallback", allow_latest_tf_fallback_, true);
    pnh_.param("publish_rate", publish_rate_, 15.0);
    pnh_.param("self_geometry_rate", self_geometry_rate_, 30.0);

    if (!pnh_.getParam("input_topics", input_topics_) ||
        input_topics_.empty())
    {
      ROS_ERROR("[tof_fusion_self_filter] ~input_topics is empty.");
      return false;
    }

    if (!pnh_.getParam("sensor_frames", sensor_frames_) ||
        sensor_frames_.empty())
    {
      ROS_ERROR("[tof_fusion_self_filter] ~sensor_frames is empty.");
      return false;
    }

    if (input_topics_.size() != sensor_frames_.size())
    {
      ROS_ERROR_STREAM(
          "[tof_fusion_self_filter] input_topics size=" << input_topics_.size()
          << " differs from sensor_frames size=" << sensor_frames_.size());
      return false;
    }

    if (body_samples_file_.empty())
    {
      ROS_ERROR("[tof_fusion_self_filter] ~body_samples_file is empty.");
      return false;
    }

    if (input_voxel_leaf_size_ <= 0.0 ||
        output_voxel_leaf_size_ <= 0.0 ||
        self_filter_padding_ < 0.0 ||
        sensor_origin_filter_radius_ < 0.0 ||
        sensor_timeout_ <= 0.0 ||
        tf_timeout_ < 0.0 ||
        publish_rate_ <= 0.0 ||
        self_geometry_rate_ <= 0.0)
    {
      ROS_ERROR("[tof_fusion_self_filter] invalid numeric parameter.");
      return false;
    }

    std::string error_msg;
    if (!body_model_.loadFromYaml(body_samples_file_, &error_msg))
    {
      ROS_ERROR_STREAM(
          "[tof_fusion_self_filter] failed to load body samples: "
          << error_msg);
      return false;
    }

    sensor_states_.resize(input_topics_.size());
    subscribers_.reserve(input_topics_.size());
    for (std::size_t i = 0; i < input_topics_.size(); ++i)
    {
      subscribers_.push_back(
          nh_.subscribe<sensor_msgs::PointCloud2>(
              input_topics_[i],
              1,
              boost::bind(
                  &TofFusionSelfFilterNode::cloudCallback,
                  this,
                  _1,
                  i)));
    }

    output_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
        output_topic_, 1, false);
    ray_observation_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
        ray_observation_topic_, 1, false);
    summary_pub_ = nh_.advertise<std_msgs::String>(
        summary_topic_, 1, true);

    geometry_timer_ = nh_.createTimer(
        ros::Duration(1.0 / self_geometry_rate_),
        &TofFusionSelfFilterNode::geometryTimerCallback,
        this);
    publish_timer_ = nh_.createTimer(
        ros::Duration(1.0 / publish_rate_),
        &TofFusionSelfFilterNode::publishTimerCallback,
        this);

    ROS_INFO_STREAM(
        "[tof_fusion_self_filter] Phase E2 ready: sensors="
        << input_topics_.size()
        << " body_samples=" << body_model_.size()
        << " base_frame=" << base_frame_
        << " input_leaf=" << input_voxel_leaf_size_
        << " output_leaf=" << output_voxel_leaf_size_
        << " padding=" << self_filter_padding_
        << " sensor_origin_radius=" << sensor_origin_filter_radius_);

    for (std::size_t i = 0; i < input_topics_.size(); ++i)
    {
      ROS_INFO_STREAM(
          "  [" << i << "] topic=" << input_topics_[i]
          << " sensor_frame=" << sensor_frames_[i]);
    }

    return true;
  }

private:
  bool lookupTransform(
      const std::string& target,
      const std::string& source,
      const ros::Time& stamp,
      tf2::Transform* out,
      bool* used_latest_fallback = nullptr)
  {
    if (used_latest_fallback)
    {
      *used_latest_fallback = false;
    }

    geometry_msgs::TransformStamped tf_msg;
    try
    {
      tf_msg = tf_buffer_.lookupTransform(
          target,
          source,
          stamp,
          ros::Duration(tf_timeout_));
      tf2::fromMsg(tf_msg.transform, *out);
      return true;
    }
    catch (const tf2::TransformException&)
    {
      if (!allow_latest_tf_fallback_ || stamp.isZero())
      {
        return false;
      }
    }

    try
    {
      tf_msg = tf_buffer_.lookupTransform(
          target,
          source,
          ros::Time(0),
          ros::Duration(tf_timeout_));
      tf2::fromMsg(tf_msg.transform, *out);
      if (used_latest_fallback)
      {
        *used_latest_fallback = true;
      }
      return true;
    }
    catch (const tf2::TransformException&)
    {
      return false;
    }
  }

  void geometryTimerCallback(const ros::TimerEvent&)
  {
    std::unordered_map<std::string, tf2::Transform> frame_tf;
    frame_tf.reserve(
        body_model_.frames().size() + sensor_frames_.size());

    std::size_t missing_frames = 0;
    auto ensure_frame = [&](const std::string& frame) -> bool
    {
      if (frame_tf.find(frame) != frame_tf.end())
      {
        return true;
      }
      tf2::Transform T;
      if (!lookupTransform(base_frame_, frame, ros::Time(0), &T, nullptr))
      {
        ++missing_frames;
        return false;
      }
      frame_tf.emplace(frame, T);
      return true;
    };

    for (const auto& frame : body_model_.frames())
    {
      ensure_frame(frame);
    }
    for (const auto& frame : sensor_frames_)
    {
      ensure_frame(frame);
    }

    std::vector<Sphere> spheres;
    spheres.reserve(body_model_.size() + sensor_frames_.size());

    std::size_t transformed_body_samples = 0;
    for (const auto& sample : body_model_.samples())
    {
      const auto it = frame_tf.find(sample.frame_name);
      if (it == frame_tf.end())
      {
        continue;
      }
      Sphere s;
      s.center = it->second * sample.center_link;
      s.radius = std::max(
          0.0, sample.radius + self_filter_padding_);
      spheres.push_back(s);
      ++transformed_body_samples;
    }

    if (sensor_origin_filter_radius_ > 0.0)
    {
      for (const auto& frame : sensor_frames_)
      {
        const auto it = frame_tf.find(frame);
        if (it == frame_tf.end())
        {
          continue;
        }
        Sphere s;
        s.center = it->second.getOrigin();
        s.radius = sensor_origin_filter_radius_;
        spheres.push_back(s);
      }
    }

    {
      std::lock_guard<std::mutex> lock(geometry_mutex_);
      self_spheres_ = std::move(spheres);
      last_geometry_update_ = ros::Time::now();
      transformed_body_samples_ = transformed_body_samples;
      missing_geometry_frames_ = missing_frames;
    }
  }

  bool isSelfPoint(
      const tf2::Vector3& point,
      const std::vector<Sphere>& spheres) const
  {
    for (const auto& sphere : spheres)
    {
      const tf2::Vector3 delta = point - sphere.center;
      if (delta.length2() <= sphere.radius * sphere.radius)
      {
        return true;
      }
    }
    return false;
  }

  void cloudCallback(
      const sensor_msgs::PointCloud2ConstPtr& msg,
      std::size_t sensor_index)
  {
    if (!msg || sensor_index >= sensor_states_.size())
    {
      return;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr raw(
        new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*msg, *raw);

    pcl::PointCloud<pcl::PointXYZ>::Ptr voxel(
        new pcl::PointCloud<pcl::PointXYZ>());
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(raw);
    const float input_leaf =
        static_cast<float>(input_voxel_leaf_size_);
    voxel_filter.setLeafSize(input_leaf, input_leaf, input_leaf);
    voxel_filter.filter(*voxel);

    tf2::Transform T_base_sensor;
    bool used_latest_fallback = false;
    const std::string source_frame =
        msg->header.frame_id.empty()
            ? sensor_frames_[sensor_index]
            : msg->header.frame_id;
    if (!lookupTransform(
            base_frame_,
            source_frame,
            msg->header.stamp,
            &T_base_sensor,
            &used_latest_fallback))
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      SensorState& state = sensor_states_[sensor_index];
      ++state.tf_failures;
      state.valid = false;
      return;
    }

    std::vector<Sphere> spheres;
    {
      std::lock_guard<std::mutex> lock(geometry_mutex_);
      spheres = self_spheres_;
    }
    if (spheres.empty())
    {
      ROS_WARN_THROTTLE(
          1.0,
          "[tof_fusion_self_filter] waiting for self geometry TFs");
      return;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr kept(
        new pcl::PointCloud<pcl::PointXYZ>());
    kept->points.reserve(voxel->points.size());

    std::size_t self_removed = 0;
    for (const auto& p_sensor : voxel->points)
    {
      if (!finitePoint(p_sensor))
      {
        continue;
      }

      const tf2::Vector3 p_base =
          T_base_sensor * tf2::Vector3(
              static_cast<double>(p_sensor.x),
              static_cast<double>(p_sensor.y),
              static_cast<double>(p_sensor.z));

      if (isSelfPoint(p_base, spheres))
      {
        ++self_removed;
        continue;
      }

      pcl::PointXYZ p;
      p.x = static_cast<float>(p_base.x());
      p.y = static_cast<float>(p_base.y());
      p.z = static_cast<float>(p_base.z());
      kept->points.push_back(p);
    }

    kept->width = static_cast<std::uint32_t>(kept->points.size());
    kept->height = 1;
    kept->is_dense = false;

    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      SensorState& state = sensor_states_[sensor_index];
      state.filtered_base = kept;
      state.cloud_stamp = msg->header.stamp;
      state.received = ros::Time::now();
      state.raw_points = raw->points.size();
      state.voxel_points = voxel->points.size();
      state.self_removed = self_removed;
      state.kept_points = kept->points.size();
      state.sensor_origin_base = T_base_sensor.getOrigin();
      if (used_latest_fallback)
      {
        ++state.latest_tf_fallbacks;
      }
      state.valid = true;
    }
  }

  void publishTimerCallback(const ros::TimerEvent&)
  {
    const ros::Time now = ros::Time::now();

    pcl::PointCloud<pcl::PointXYZ>::Ptr merged(
        new pcl::PointCloud<pcl::PointXYZ>());

    std::vector<SensorState> states;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      states = sensor_states_;
    }

    std::size_t active_sensors = 0;
    std::size_t stale_sensors = 0;
    std::size_t raw_points = 0;
    std::size_t voxel_points = 0;
    std::size_t self_removed = 0;
    std::size_t kept_before_merge = 0;
    std::size_t tf_failures = 0;
    std::size_t latest_tf_fallbacks = 0;

    ros::Time newest_cloud_stamp(0);

    for (const auto& state : states)
    {
      tf_failures += state.tf_failures;
      latest_tf_fallbacks += state.latest_tf_fallbacks;

      if (!state.valid)
      {
        continue;
      }

      const double age = (now - state.received).toSec();
      if (!std::isfinite(age) || age > sensor_timeout_)
      {
        ++stale_sensors;
        continue;
      }

      ++active_sensors;
      raw_points += state.raw_points;
      voxel_points += state.voxel_points;
      self_removed += state.self_removed;
      kept_before_merge += state.kept_points;
      if (state.cloud_stamp > newest_cloud_stamp)
      {
        newest_cloud_stamp = state.cloud_stamp;
      }

      if (state.filtered_base)
      {
        merged->points.insert(
            merged->points.end(),
            state.filtered_base->points.begin(),
            state.filtered_base->points.end());
      }
    }

    merged->width = static_cast<std::uint32_t>(merged->points.size());
    merged->height = 1;
    merged->is_dense = false;

    pcl::PointCloud<pcl::PointXYZ>::Ptr output(
        new pcl::PointCloud<pcl::PointXYZ>());
    if (!merged->points.empty())
    {
      pcl::VoxelGrid<pcl::PointXYZ> output_filter;
      output_filter.setInputCloud(merged);
      const float output_leaf =
          static_cast<float>(output_voxel_leaf_size_);
      output_filter.setLeafSize(
          output_leaf, output_leaf, output_leaf);
      output_filter.filter(*output);
    }

    // Phase E3 transport: preserve ray provenance.  The regular fused cloud
    // remains optimized for visualization/environment geometry, while this
    // observation cloud keeps one sensor origin per retained endpoint.
    std::size_t ray_observation_count = 0;
    for (const auto& state : states)
    {
      if (!state.valid)
      {
        continue;
      }
      const double age = (now - state.received).toSec();
      if (!std::isfinite(age) || age > sensor_timeout_ ||
          !state.filtered_base)
      {
        continue;
      }
      ray_observation_count += state.filtered_base->points.size();
    }

    sensor_msgs::PointCloud2 ray_msg;
    ray_msg.header.frame_id = base_frame_;
    ray_msg.header.stamp =
        newest_cloud_stamp.isZero() ? now : newest_cloud_stamp;
    sensor_msgs::PointCloud2Modifier ray_modifier(ray_msg);
    ray_modifier.setPointCloud2Fields(
        7,
        "x", 1, sensor_msgs::PointField::FLOAT32,
        "y", 1, sensor_msgs::PointField::FLOAT32,
        "z", 1, sensor_msgs::PointField::FLOAT32,
        "sensor_id", 1, sensor_msgs::PointField::INT32,
        "origin_x", 1, sensor_msgs::PointField::FLOAT32,
        "origin_y", 1, sensor_msgs::PointField::FLOAT32,
        "origin_z", 1, sensor_msgs::PointField::FLOAT32);
    ray_modifier.resize(ray_observation_count);

    sensor_msgs::PointCloud2Iterator<float> ray_x(ray_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> ray_y(ray_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> ray_z(ray_msg, "z");
    sensor_msgs::PointCloud2Iterator<int32_t> ray_sensor_id(
        ray_msg, "sensor_id");
    sensor_msgs::PointCloud2Iterator<float> ray_origin_x(
        ray_msg, "origin_x");
    sensor_msgs::PointCloud2Iterator<float> ray_origin_y(
        ray_msg, "origin_y");
    sensor_msgs::PointCloud2Iterator<float> ray_origin_z(
        ray_msg, "origin_z");

    for (std::size_t sensor_id = 0; sensor_id < states.size(); ++sensor_id)
    {
      const auto& state = states[sensor_id];
      if (!state.valid || !state.filtered_base)
      {
        continue;
      }
      const double age = (now - state.received).toSec();
      if (!std::isfinite(age) || age > sensor_timeout_)
      {
        continue;
      }

      for (const auto& p : state.filtered_base->points)
      {
        *ray_x = p.x;
        *ray_y = p.y;
        *ray_z = p.z;
        *ray_sensor_id = static_cast<int32_t>(sensor_id);
        *ray_origin_x =
            static_cast<float>(state.sensor_origin_base.x());
        *ray_origin_y =
            static_cast<float>(state.sensor_origin_base.y());
        *ray_origin_z =
            static_cast<float>(state.sensor_origin_base.z());

        ++ray_x;
        ++ray_y;
        ++ray_z;
        ++ray_sensor_id;
        ++ray_origin_x;
        ++ray_origin_y;
        ++ray_origin_z;
      }
    }
    ray_observation_pub_.publish(ray_msg);

    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(*output, msg);
    msg.header.frame_id = base_frame_;
    msg.header.stamp =
        newest_cloud_stamp.isZero() ? now : newest_cloud_stamp;
    output_pub_.publish(msg);

    std::size_t sphere_count = 0;
    std::size_t transformed_body_samples = 0;
    std::size_t missing_geometry_frames = 0;
    double geometry_age_s = std::numeric_limits<double>::quiet_NaN();
    {
      std::lock_guard<std::mutex> lock(geometry_mutex_);
      sphere_count = self_spheres_.size();
      transformed_body_samples = transformed_body_samples_;
      missing_geometry_frames = missing_geometry_frames_;
      if (!last_geometry_update_.isZero())
      {
        geometry_age_s = (now - last_geometry_update_).toSec();
      }
    }

    std_msgs::String summary;
    std::ostringstream oss;
    oss << "phase=E2"
        << " configured_sensors=" << input_topics_.size()
        << " active_sensors=" << active_sensors
        << " stale_sensors=" << stale_sensors
        << " raw_points=" << raw_points
        << " input_voxel_points=" << voxel_points
        << " self_removed=" << self_removed
        << " kept_before_merge=" << kept_before_merge
        << " fused_points=" << output->points.size()
        << " ray_observations=" << ray_observation_count
        << " self_sphere_count=" << sphere_count
        << " transformed_body_samples=" << transformed_body_samples
        << " missing_geometry_frames=" << missing_geometry_frames
        << " tf_failures=" << tf_failures
        << " latest_tf_fallbacks=" << latest_tf_fallbacks
        << " geometry_age_s=" << geometry_age_s;
    summary.data = oss.str();
    summary_pub_.publish(summary);

    ROS_INFO_STREAM_THROTTLE(
        2.0,
        "[tof_fusion_self_filter] " << summary.data);
  }

private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  care_confidence_map::BodySampleModel body_model_;

  std::string base_frame_;
  std::string output_topic_;
  std::string summary_topic_;
  std::string ray_observation_topic_;
  std::string body_samples_file_;
  std::vector<std::string> input_topics_;
  std::vector<std::string> sensor_frames_;

  double input_voxel_leaf_size_ = 0.03;
  double output_voxel_leaf_size_ = 0.03;
  double self_filter_padding_ = 0.015;
  double sensor_origin_filter_radius_ = 0.06;
  double sensor_timeout_ = 0.25;
  double tf_timeout_ = 0.03;
  bool allow_latest_tf_fallback_ = true;
  double publish_rate_ = 15.0;
  double self_geometry_rate_ = 30.0;

  std::vector<ros::Subscriber> subscribers_;
  ros::Publisher output_pub_;
  ros::Publisher ray_observation_pub_;
  ros::Publisher summary_pub_;
  ros::Timer geometry_timer_;
  ros::Timer publish_timer_;

  std::mutex state_mutex_;
  std::vector<SensorState> sensor_states_;

  std::mutex geometry_mutex_;
  std::vector<Sphere> self_spheres_;
  ros::Time last_geometry_update_;
  std::size_t transformed_body_samples_ = 0;
  std::size_t missing_geometry_frames_ = 0;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "tof_fusion_self_filter");

  TofFusionSelfFilterNode node;
  if (!node.initialize())
  {
    return 1;
  }

  ros::spin();
  return 0;
}
