#!/usr/bin/env cpp
#include <ros/ros.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/String.h>

#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <urdf/model.h>

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
#include <vector>

namespace
{

constexpr double kPi = 3.14159265358979323846;

enum class PrimitiveType
{
  BOX,
  CYLINDER,
  SPHERE,
};

struct Primitive
{
  PrimitiveType type = PrimitiveType::BOX;
  std::string label;
  tf2::Transform T_link_primitive{tf2::Transform::getIdentity()};

  tf2::Vector3 box_size{0.0, 0.0, 0.0};
  double radius = 0.0;
  double length = 0.0;
};

struct LinkGeometry
{
  std::string link_name;
  std::vector<Primitive> primitives;
};

struct TimedLinkGeometry
{
  const LinkGeometry* geometry = nullptr;
  tf2::Transform T_link_base{tf2::Transform::getIdentity()};
};

struct NearestPrimitive
{
  bool valid = false;
  std::string link_name;
  std::string primitive_label;
  double signed_distance = std::numeric_limits<double>::infinity();
};

struct RaySample
{
  pcl::PointXYZ endpoint_base;
  bool hit = false;
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
  std::size_t body_tf_failures = 0;
  std::size_t dropped_for_body_tf = 0;

  tf2::Vector3 sensor_origin_base{0.0, 0.0, 0.0};
  std::vector<RaySample> ray_samples;
  std::size_t ray_self_blocked = 0;

  double callback_ms = std::numeric_limits<double>::quiet_NaN();
  double body_tf_lookup_ms = std::numeric_limits<double>::quiet_NaN();
  double self_filter_ms = std::numeric_limits<double>::quiet_NaN();

  bool valid = false;
};

bool finitePoint(const pcl::PointXYZ& p)
{
  return std::isfinite(static_cast<double>(p.x)) &&
         std::isfinite(static_cast<double>(p.y)) &&
         std::isfinite(static_cast<double>(p.z));
}

double boxSignedDistance(
    const tf2::Vector3& p,
    const tf2::Vector3& size)
{
  const double qx = std::abs(p.x()) - 0.5 * size.x();
  const double qy = std::abs(p.y()) - 0.5 * size.y();
  const double qz = std::abs(p.z()) - 0.5 * size.z();

  const double ox = std::max(qx, 0.0);
  const double oy = std::max(qy, 0.0);
  const double oz = std::max(qz, 0.0);
  const double outside = std::sqrt(ox * ox + oy * oy + oz * oz);
  const double inside = std::min(std::max(qx, std::max(qy, qz)), 0.0);
  return outside + inside;
}

double cylinderSignedDistance(
    const tf2::Vector3& p,
    double radius,
    double length)
{
  const double radial =
      std::sqrt(p.x() * p.x() + p.y() * p.y()) - radius;
  const double axial = std::abs(p.z()) - 0.5 * length;

  const double orad = std::max(radial, 0.0);
  const double oax = std::max(axial, 0.0);
  const double outside = std::sqrt(orad * orad + oax * oax);
  const double inside = std::min(std::max(radial, axial), 0.0);
  return outside + inside;
}

double sphereSignedDistance(
    const tf2::Vector3& p,
    double radius)
{
  return p.length() - radius;
}

tf2::Transform poseToTf(const urdf::Pose& pose)
{
  tf2::Quaternion q(
      pose.rotation.x,
      pose.rotation.y,
      pose.rotation.z,
      pose.rotation.w);
  if (q.length2() <= 1e-18)
  {
    q.setValue(0.0, 0.0, 0.0, 1.0);
  }
  else
  {
    q.normalize();
  }

  return tf2::Transform(
      q,
      tf2::Vector3(
          pose.position.x,
          pose.position.y,
          pose.position.z));
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
    pnh_.param<std::string>(
        "self_filter_urdf_file",
        self_filter_urdf_file_,
        "");

    pnh_.param("input_voxel_leaf_size", input_voxel_leaf_size_, 0.03);
    pnh_.param("output_voxel_leaf_size", output_voxel_leaf_size_, 0.03);
    pnh_.param("sensor_timeout", sensor_timeout_, 0.25);
    pnh_.param("tf_timeout", tf_timeout_, 0.03);
    pnh_.param("publish_rate", publish_rate_, 15.0);
    pnh_.param("ray_pixel_stride", ray_pixel_stride_, 8);
    pnh_.param("ray_horizontal_fov_deg", ray_horizontal_fov_deg_, 55.0);
    pnh_.param("ray_vertical_fov_deg", ray_vertical_fov_deg_, 72.0);
    pnh_.param("ray_max_forward_depth", ray_max_forward_depth_, 0.75);

    pnh_.param(
        "debug_occupancy_watch_enabled",
        debug_occupancy_watch_enabled_,
        false);
    pnh_.param(
        "debug_occupancy_watch_x",
        debug_occupancy_watch_x_,
        0.0);
    pnh_.param(
        "debug_occupancy_watch_y",
        debug_occupancy_watch_y_,
        0.0);
    pnh_.param(
        "debug_occupancy_watch_z",
        debug_occupancy_watch_z_,
        0.0);
    pnh_.param(
        "debug_occupancy_watch_half_extent",
        debug_occupancy_watch_half_extent_,
        0.025);

    pnh_.param(
        "debug_hotspot_watch_enabled",
        debug_hotspot_watch_enabled_,
        false);
    pnh_.param("debug_hotspot_x_min", debug_hotspot_x_min_, -0.025);
    pnh_.param("debug_hotspot_x_max", debug_hotspot_x_max_, 0.075);
    pnh_.param("debug_hotspot_y_min", debug_hotspot_y_min_, 0.025);
    pnh_.param("debug_hotspot_y_max", debug_hotspot_y_max_, 0.125);
    pnh_.param("debug_hotspot_z_min", debug_hotspot_z_min_, 0.225);
    pnh_.param("debug_hotspot_z_max", debug_hotspot_z_max_, 0.425);

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
          "[tof_fusion_self_filter] input_topics size="
          << input_topics_.size()
          << " differs from sensor_frames size="
          << sensor_frames_.size());
      return false;
    }

    if (self_filter_urdf_file_.empty())
    {
      ROS_ERROR(
          "[tof_fusion_self_filter] ~self_filter_urdf_file is empty.");
      return false;
    }

    if (input_voxel_leaf_size_ <= 0.0 ||
        output_voxel_leaf_size_ <= 0.0 ||
        sensor_timeout_ <= 0.0 ||
        tf_timeout_ < 0.0 ||
        publish_rate_ <= 0.0 ||
        ray_pixel_stride_ < 1 ||
        ray_horizontal_fov_deg_ <= 0.0 ||
        ray_vertical_fov_deg_ <= 0.0 ||
        ray_max_forward_depth_ <= 0.0)
    {
      ROS_ERROR("[tof_fusion_self_filter] invalid numeric parameter.");
      return false;
    }

    if (!loadSelfFilterGeometry())
    {
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

    publish_timer_ = nh_.createTimer(
        ros::Duration(1.0 / publish_rate_),
        &TofFusionSelfFilterNode::publishTimerCallback,
        this);

    ROS_WARN_STREAM(
        "[tof_fusion_self_filter] exact dedicated-URDF self filter ready"
        << " sensors=" << input_topics_.size()
        << " urdf=" << self_filter_urdf_file_
        << " links=" << link_geometries_.size()
        << " primitives=" << primitive_count_
        << " runtime_padding_m=0"
        << " timestamp_geometry=cloud_stamp"
        << " target_publish_hz=" << publish_rate_);

    for (std::size_t i = 0; i < input_topics_.size(); ++i)
    {
      ROS_INFO_STREAM(
          "  [" << i << "] topic=" << input_topics_[i]
          << " sensor_frame=" << sensor_frames_[i]);
    }

    return true;
  }

private:
  bool loadSelfFilterGeometry()
  {
    urdf::Model model;
    if (!model.initFile(self_filter_urdf_file_))
    {
      ROS_ERROR_STREAM(
          "[tof_fusion_self_filter] failed to parse dedicated self-filter URDF: "
          << self_filter_urdf_file_);
      return false;
    }

    std::vector<urdf::LinkSharedPtr> links;
    model.getLinks(links);

    link_geometries_.clear();
    primitive_count_ = 0;

    for (const auto& link : links)
    {
      if (!link)
      {
        continue;
      }

      std::vector<urdf::CollisionSharedPtr> collisions =
          link->collision_array;
      if (collisions.empty() && link->collision)
      {
        collisions.push_back(link->collision);
      }
      if (collisions.empty())
      {
        continue;
      }

      LinkGeometry link_geometry;
      link_geometry.link_name = link->name;

      for (std::size_t cidx = 0; cidx < collisions.size(); ++cidx)
      {
        const auto& collision = collisions[cidx];
        if (!collision || !collision->geometry)
        {
          continue;
        }

        Primitive primitive;
        primitive.label =
            link->name + "/collision_" + std::to_string(cidx);
        primitive.T_link_primitive = poseToTf(collision->origin);

        if (collision->geometry->type == urdf::Geometry::BOX)
        {
          const auto* box =
              dynamic_cast<const urdf::Box*>(
                  collision->geometry.get());
          if (!box)
          {
            ROS_ERROR_STREAM(
                "[tof_fusion_self_filter] invalid BOX geometry at "
                << primitive.label);
            return false;
          }
          primitive.type = PrimitiveType::BOX;
          primitive.box_size = tf2::Vector3(
              box->dim.x, box->dim.y, box->dim.z);
        }
        else if (collision->geometry->type ==
                 urdf::Geometry::CYLINDER)
        {
          const auto* cylinder =
              dynamic_cast<const urdf::Cylinder*>(
                  collision->geometry.get());
          if (!cylinder)
          {
            ROS_ERROR_STREAM(
                "[tof_fusion_self_filter] invalid CYLINDER geometry at "
                << primitive.label);
            return false;
          }
          primitive.type = PrimitiveType::CYLINDER;
          primitive.radius = cylinder->radius;
          primitive.length = cylinder->length;
        }
        else if (collision->geometry->type ==
                 urdf::Geometry::SPHERE)
        {
          const auto* sphere =
              dynamic_cast<const urdf::Sphere*>(
                  collision->geometry.get());
          if (!sphere)
          {
            ROS_ERROR_STREAM(
                "[tof_fusion_self_filter] invalid SPHERE geometry at "
                << primitive.label);
            return false;
          }
          primitive.type = PrimitiveType::SPHERE;
          primitive.radius = sphere->radius;
        }
        else
        {
          ROS_ERROR_STREAM(
              "[tof_fusion_self_filter] unsupported collision geometry in "
              << self_filter_urdf_file_
              << " at " << primitive.label
              << ". Dedicated self-filter URDF must use box/cylinder/sphere "
                 "primitives only.");
          return false;
        }

        link_geometry.primitives.push_back(primitive);
        ++primitive_count_;
      }

      if (!link_geometry.primitives.empty())
      {
        link_geometries_.push_back(link_geometry);
      }
    }

    if (link_geometries_.empty() || primitive_count_ == 0)
    {
      ROS_ERROR_STREAM(
          "[tof_fusion_self_filter] dedicated self-filter URDF contains "
          "no supported collision primitives: "
          << self_filter_urdf_file_);
      return false;
    }

    return true;
  }

  bool lookupTransformExact(
      const std::string& target,
      const std::string& source,
      const ros::Time& stamp,
      tf2::Transform* out)
  {
    if (!out)
    {
      return false;
    }

    if (target == source)
    {
      *out = tf2::Transform::getIdentity();
      return true;
    }

    try
    {
      const geometry_msgs::TransformStamped tf_msg =
          tf_buffer_.lookupTransform(
              target,
              source,
              stamp,
              ros::Duration(tf_timeout_));
      tf2::fromMsg(tf_msg.transform, *out);
      return true;
    }
    catch (const tf2::TransformException& ex)
    {
      ROS_WARN_STREAM_THROTTLE(
          1.0,
          "[tof_fusion_self_filter] exact TF unavailable target="
          << target << " source=" << source
          << " stamp=" << stamp.toSec()
          << " error=" << ex.what());
      return false;
    }
  }

  bool buildTimedGeometry(
      const ros::Time& stamp,
      std::vector<TimedLinkGeometry>* timed,
      std::size_t* failure_count)
  {
    if (!timed)
    {
      return false;
    }

    timed->clear();
    timed->reserve(link_geometries_.size());

    std::size_t failures = 0;
    for (const auto& link_geometry : link_geometries_)
    {
      tf2::Transform T_base_link;
      if (!lookupTransformExact(
              base_frame_,
              link_geometry.link_name,
              stamp,
              &T_base_link))
      {
        ++failures;
        continue;
      }

      TimedLinkGeometry item;
      item.geometry = &link_geometry;
      item.T_link_base = T_base_link.inverse();
      timed->push_back(item);
    }

    if (failure_count)
    {
      *failure_count = failures;
    }

    // Fail closed for mapping: if the self model is incomplete at this cloud
    // timestamp, do not publish environmental occupied endpoints from that
    // cloud. A partial self model can turn robot returns into false obstacles.
    return failures == 0 && timed->size() == link_geometries_.size();
  }

  double primitiveSignedDistance(
      const tf2::Vector3& p_link,
      const Primitive& primitive) const
  {
    const tf2::Vector3 p =
        primitive.T_link_primitive.inverse() * p_link;

    switch (primitive.type)
    {
      case PrimitiveType::BOX:
        return boxSignedDistance(p, primitive.box_size);
      case PrimitiveType::CYLINDER:
        return cylinderSignedDistance(
            p, primitive.radius, primitive.length);
      case PrimitiveType::SPHERE:
        return sphereSignedDistance(p, primitive.radius);
    }

    return std::numeric_limits<double>::infinity();
  }

  bool classifySelfPoint(
      const tf2::Vector3& point_base,
      const std::vector<TimedLinkGeometry>& timed,
      NearestPrimitive* nearest = nullptr) const
  {
    bool is_self = false;
    NearestPrimitive best;

    for (const auto& link : timed)
    {
      if (!link.geometry)
      {
        continue;
      }

      const tf2::Vector3 p_link =
          link.T_link_base * point_base;

      for (const auto& primitive : link.geometry->primitives)
      {
        const double d =
            primitiveSignedDistance(p_link, primitive);

        if (d < best.signed_distance)
        {
          best.valid = true;
          best.link_name = link.geometry->link_name;
          best.primitive_label = primitive.label;
          best.signed_distance = d;
        }

        if (d <= 0.0)
        {
          is_self = true;
        }
      }
    }

    if (nearest)
    {
      *nearest = best;
    }
    return is_self;
  }

  bool isHotspotPoint(const tf2::Vector3& point) const
  {
    if (!debug_hotspot_watch_enabled_)
    {
      return false;
    }

    return
        point.x() >= debug_hotspot_x_min_ &&
        point.x() <= debug_hotspot_x_max_ &&
        point.y() >= debug_hotspot_y_min_ &&
        point.y() <= debug_hotspot_y_max_ &&
        point.z() >= debug_hotspot_z_min_ &&
        point.z() <= debug_hotspot_z_max_;
  }

  bool isWatchedOccupancyPoint(const tf2::Vector3& point) const
  {
    if (!debug_occupancy_watch_enabled_)
    {
      return false;
    }

    return
        std::abs(point.x() - debug_occupancy_watch_x_) <=
            debug_occupancy_watch_half_extent_ &&
        std::abs(point.y() - debug_occupancy_watch_y_) <=
            debug_occupancy_watch_half_extent_ &&
        std::abs(point.z() - debug_occupancy_watch_z_) <=
            debug_occupancy_watch_half_extent_;
  }

  void cloudCallback(
      const sensor_msgs::PointCloud2ConstPtr& msg,
      std::size_t sensor_index)
  {
    if (!msg || sensor_index >= sensor_states_.size())
    {
      return;
    }

    const ros::WallTime callback_start = ros::WallTime::now();

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

    const std::string source_frame =
        msg->header.frame_id.empty()
            ? sensor_frames_[sensor_index]
            : msg->header.frame_id;

    tf2::Transform T_base_sensor;
    if (!lookupTransformExact(
            base_frame_,
            source_frame,
            msg->header.stamp,
            &T_base_sensor))
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      SensorState& state = sensor_states_[sensor_index];
      ++state.tf_failures;
      state.valid = false;
      state.callback_ms =
          (ros::WallTime::now() - callback_start).toSec() * 1000.0;
      return;
    }

    const ros::WallTime body_tf_start = ros::WallTime::now();
    std::vector<TimedLinkGeometry> timed_geometry;
    std::size_t body_tf_failures = 0;
    const bool body_geometry_ready =
        buildTimedGeometry(
            msg->header.stamp,
            &timed_geometry,
            &body_tf_failures);
    const double body_tf_lookup_ms =
        (ros::WallTime::now() - body_tf_start).toSec() * 1000.0;

    if (!body_geometry_ready)
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      SensorState& state = sensor_states_[sensor_index];
      state.body_tf_failures += body_tf_failures;
      ++state.dropped_for_body_tf;
      state.valid = false;
      state.body_tf_lookup_ms = body_tf_lookup_ms;
      state.callback_ms =
          (ros::WallTime::now() - callback_start).toSec() * 1000.0;
      return;
    }

    const ros::WallTime self_filter_start = ros::WallTime::now();

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

      if (classifySelfPoint(p_base, timed_geometry, nullptr))
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

    std::vector<RaySample> ray_samples;
    std::size_t ray_self_blocked = 0;

    const std::size_t width = static_cast<std::size_t>(raw->width);
    const std::size_t height = static_cast<std::size_t>(raw->height);

    if (width > 0 &&
        height > 1 &&
        raw->points.size() >= width * height)
    {
      const double h_half =
          0.5 * ray_horizontal_fov_deg_ * kPi / 180.0;
      const double v_half =
          0.5 * ray_vertical_fov_deg_ * kPi / 180.0;
      const double fx =
          (static_cast<double>(width) - 1.0) /
          (2.0 * std::tan(h_half));
      const double fy =
          (static_cast<double>(height) - 1.0) /
          (2.0 * std::tan(v_half));
      const double cx =
          0.5 * (static_cast<double>(width) - 1.0);
      const double cy =
          0.5 * (static_cast<double>(height) - 1.0);

      const std::size_t stride =
          static_cast<std::size_t>(ray_pixel_stride_);

      ray_samples.reserve(
          ((width + stride - 1) / stride) *
          ((height + stride - 1) / stride));

      for (std::size_t v = 0; v < height; v += stride)
      {
        for (std::size_t u = 0; u < width; u += stride)
        {
          const pcl::PointXYZ& p_sensor =
              raw->points[v * width + u];

          RaySample sample;
          if (finitePoint(p_sensor))
          {
            const tf2::Vector3 p_base =
                T_base_sensor * tf2::Vector3(
                    static_cast<double>(p_sensor.x),
                    static_cast<double>(p_sensor.y),
                    static_cast<double>(p_sensor.z));

            NearestPrimitive nearest;
            const bool is_self =
                classifySelfPoint(
                    p_base,
                    timed_geometry,
                    (isHotspotPoint(p_base) ||
                     isWatchedOccupancyPoint(p_base))
                        ? &nearest
                        : nullptr);

            if (isHotspotPoint(p_base))
            {
              ROS_WARN_STREAM(
                  "[TOF_HOTSPOT_HIT]"
                  << " sensor_id=" << sensor_index
                  << " source_frame=" << source_frame
                  << " stamp=" << msg->header.stamp.toSec()
                  << " raw_sensor=[" << p_sensor.x << ","
                  << p_sensor.y << "," << p_sensor.z << "]"
                  << " base=[" << p_base.x() << ","
                  << p_base.y() << "," << p_base.z() << "]"
                  << " is_self=" << static_cast<int>(is_self)
                  << " nearest_self_link="
                  << (nearest.valid
                          ? nearest.link_name
                          : std::string("none"))
                  << " nearest_self_primitive="
                  << (nearest.valid
                          ? nearest.primitive_label
                          : std::string("none"))
                  << " signed_surface_distance="
                  << nearest.signed_distance
                  << " self_filter_model=urdf_primitives"
                  << " runtime_padding_m=0"
                  << " geometry_stamp="
                  << msg->header.stamp.toSec());
            }

            if (isWatchedOccupancyPoint(p_base))
            {
              ROS_WARN_STREAM(
                  "[TOF_OCCUPANCY_WATCH_HIT]"
                  << " sensor_id=" << sensor_index
                  << " source_frame=" << source_frame
                  << " stamp=" << msg->header.stamp.toSec()
                  << " base=[" << p_base.x() << ","
                  << p_base.y() << "," << p_base.z() << "]"
                  << " is_self=" << static_cast<int>(is_self)
                  << " nearest_self_link="
                  << (nearest.valid
                          ? nearest.link_name
                          : std::string("none"))
                  << " nearest_self_primitive="
                  << (nearest.valid
                          ? nearest.primitive_label
                          : std::string("none"))
                  << " signed_surface_distance="
                  << nearest.signed_distance);
            }

            if (is_self)
            {
              ++ray_self_blocked;
              continue;
            }

            sample.endpoint_base.x =
                static_cast<float>(p_base.x());
            sample.endpoint_base.y =
                static_cast<float>(p_base.y());
            sample.endpoint_base.z =
                static_cast<float>(p_base.z());
            sample.hit = true;
          }
          else
          {
            const double x =
                (static_cast<double>(u) - cx) / fx *
                ray_max_forward_depth_;
            const double y =
                (static_cast<double>(v) - cy) / fy *
                ray_max_forward_depth_;

            const tf2::Vector3 p_base =
                T_base_sensor * tf2::Vector3(
                    x,
                    y,
                    ray_max_forward_depth_);

            sample.endpoint_base.x =
                static_cast<float>(p_base.x());
            sample.endpoint_base.y =
                static_cast<float>(p_base.y());
            sample.endpoint_base.z =
                static_cast<float>(p_base.z());
            sample.hit = false;
          }

          ray_samples.push_back(sample);
        }
      }
    }
    else
    {
      // Defensive fallback for an unorganized cloud. The already exact-URDF
      // filtered cloud is safe to expose as finite occupied endpoints.
      ray_samples.reserve(kept->points.size());
      for (const auto& p : kept->points)
      {
        RaySample sample;
        sample.endpoint_base = p;
        sample.hit = true;
        ray_samples.push_back(sample);
      }
    }

    const double self_filter_ms =
        (ros::WallTime::now() - self_filter_start).toSec() * 1000.0;
    const double callback_ms =
        (ros::WallTime::now() - callback_start).toSec() * 1000.0;

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
      state.ray_samples = std::move(ray_samples);
      state.ray_self_blocked = ray_self_blocked;
      state.body_tf_failures += body_tf_failures;
      state.body_tf_lookup_ms = body_tf_lookup_ms;
      state.self_filter_ms = self_filter_ms;
      state.callback_ms = callback_ms;
      state.valid = true;
    }
  }

  void publishTimerCallback(const ros::TimerEvent&)
  {
    const ros::WallTime publish_start = ros::WallTime::now();

    double publish_interval_s =
        std::numeric_limits<double>::quiet_NaN();
    double publish_hz =
        std::numeric_limits<double>::quiet_NaN();

    if (!last_publish_wall_.isZero())
    {
      publish_interval_s =
          (publish_start - last_publish_wall_).toSec();
      if (publish_interval_s > 1e-9)
      {
        publish_hz = 1.0 / publish_interval_s;
      }
    }
    last_publish_wall_ = publish_start;

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
    std::size_t body_tf_failures = 0;
    std::size_t dropped_for_body_tf = 0;
    std::size_t ray_self_blocked = 0;

    double callback_ms_sum = 0.0;
    double callback_ms_max = 0.0;
    std::size_t callback_ms_count = 0;

    double body_tf_ms_sum = 0.0;
    double body_tf_ms_max = 0.0;
    std::size_t body_tf_ms_count = 0;

    double self_filter_ms_sum = 0.0;
    double self_filter_ms_max = 0.0;
    std::size_t self_filter_ms_count = 0;

    ros::Time newest_cloud_stamp(0);

    for (const auto& state : states)
    {
      tf_failures += state.tf_failures;
      body_tf_failures += state.body_tf_failures;
      dropped_for_body_tf += state.dropped_for_body_tf;

      if (std::isfinite(state.callback_ms))
      {
        callback_ms_sum += state.callback_ms;
        callback_ms_max =
            std::max(callback_ms_max, state.callback_ms);
        ++callback_ms_count;
      }

      if (std::isfinite(state.body_tf_lookup_ms))
      {
        body_tf_ms_sum += state.body_tf_lookup_ms;
        body_tf_ms_max =
            std::max(body_tf_ms_max, state.body_tf_lookup_ms);
        ++body_tf_ms_count;
      }

      if (std::isfinite(state.self_filter_ms))
      {
        self_filter_ms_sum += state.self_filter_ms;
        self_filter_ms_max =
            std::max(self_filter_ms_max, state.self_filter_ms);
        ++self_filter_ms_count;
      }

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
      ray_self_blocked += state.ray_self_blocked;

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

    merged->width =
        static_cast<std::uint32_t>(merged->points.size());
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
          output_leaf,
          output_leaf,
          output_leaf);
      output_filter.filter(*output);
    }

    std::size_t ray_observation_count = 0;
    for (const auto& state : states)
    {
      if (!state.valid)
      {
        continue;
      }

      const double age = (now - state.received).toSec();
      if (!std::isfinite(age) || age > sensor_timeout_)
      {
        continue;
      }

      ray_observation_count += state.ray_samples.size();
    }

    sensor_msgs::PointCloud2 ray_msg;
    ray_msg.header.frame_id = base_frame_;
    ray_msg.header.stamp =
        newest_cloud_stamp.isZero()
            ? now
            : newest_cloud_stamp;

    sensor_msgs::PointCloud2Modifier ray_modifier(ray_msg);
    ray_modifier.setPointCloud2Fields(
        8,
        "x", 1, sensor_msgs::PointField::FLOAT32,
        "y", 1, sensor_msgs::PointField::FLOAT32,
        "z", 1, sensor_msgs::PointField::FLOAT32,
        "sensor_id", 1, sensor_msgs::PointField::INT32,
        "origin_x", 1, sensor_msgs::PointField::FLOAT32,
        "origin_y", 1, sensor_msgs::PointField::FLOAT32,
        "origin_z", 1, sensor_msgs::PointField::FLOAT32,
        "hit", 1, sensor_msgs::PointField::UINT8);

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
    sensor_msgs::PointCloud2Iterator<uint8_t> ray_hit(
        ray_msg, "hit");

    for (std::size_t sensor_id = 0;
         sensor_id < states.size();
         ++sensor_id)
    {
      const auto& state = states[sensor_id];
      if (!state.valid)
      {
        continue;
      }

      const double age = (now - state.received).toSec();
      if (!std::isfinite(age) || age > sensor_timeout_)
      {
        continue;
      }

      for (const auto& ray : state.ray_samples)
      {
        *ray_x = ray.endpoint_base.x;
        *ray_y = ray.endpoint_base.y;
        *ray_z = ray.endpoint_base.z;
        *ray_sensor_id = static_cast<int32_t>(sensor_id);
        *ray_origin_x =
            static_cast<float>(state.sensor_origin_base.x());
        *ray_origin_y =
            static_cast<float>(state.sensor_origin_base.y());
        *ray_origin_z =
            static_cast<float>(state.sensor_origin_base.z());
        *ray_hit =
            ray.hit ? static_cast<uint8_t>(1)
                    : static_cast<uint8_t>(0);

        ++ray_x;
        ++ray_y;
        ++ray_z;
        ++ray_sensor_id;
        ++ray_origin_x;
        ++ray_origin_y;
        ++ray_origin_z;
        ++ray_hit;
      }
    }

    ray_observation_pub_.publish(ray_msg);

    sensor_msgs::PointCloud2 output_msg;
    pcl::toROSMsg(*output, output_msg);
    output_msg.header.frame_id = base_frame_;
    output_msg.header.stamp =
        newest_cloud_stamp.isZero()
            ? now
            : newest_cloud_stamp;
    output_pub_.publish(output_msg);

    const double callback_ms_mean =
        callback_ms_count > 0
            ? callback_ms_sum /
                  static_cast<double>(callback_ms_count)
            : std::numeric_limits<double>::quiet_NaN();

    const double body_tf_ms_mean =
        body_tf_ms_count > 0
            ? body_tf_ms_sum /
                  static_cast<double>(body_tf_ms_count)
            : std::numeric_limits<double>::quiet_NaN();

    const double self_filter_ms_mean =
        self_filter_ms_count > 0
            ? self_filter_ms_sum /
                  static_cast<double>(self_filter_ms_count)
            : std::numeric_limits<double>::quiet_NaN();

    const double publish_callback_ms =
        (ros::WallTime::now() - publish_start).toSec() * 1000.0;

    std_msgs::String summary;
    std::ostringstream oss;
    oss
        << "phase=E2"
        << " self_filter_model=urdf_primitives"
        << " runtime_padding_m=0"
        << " timestamp_geometry=cloud_stamp"
        << " urdf_link_count=" << link_geometries_.size()
        << " urdf_primitive_count=" << primitive_count_
        << " configured_sensors=" << input_topics_.size()
        << " active_sensors=" << active_sensors
        << " stale_sensors=" << stale_sensors
        << " raw_points=" << raw_points
        << " input_voxel_points=" << voxel_points
        << " self_removed=" << self_removed
        << " kept_before_merge=" << kept_before_merge
        << " fused_points=" << output->points.size()
        << " ray_observations=" << ray_observation_count
        << " ray_self_blocked=" << ray_self_blocked
        << " tf_failures=" << tf_failures
        << " body_tf_failures=" << body_tf_failures
        << " dropped_for_body_tf=" << dropped_for_body_tf
        << " target_publish_hz=" << publish_rate_
        << " publish_interval_s=" << publish_interval_s
        << " publish_hz=" << publish_hz
        << " publish_callback_ms=" << publish_callback_ms
        << " cloud_callback_ms_mean=" << callback_ms_mean
        << " cloud_callback_ms_max=" << callback_ms_max
        << " body_tf_lookup_ms_mean=" << body_tf_ms_mean
        << " body_tf_lookup_ms_max=" << body_tf_ms_max
        << " self_filter_ms_mean=" << self_filter_ms_mean
        << " self_filter_ms_max=" << self_filter_ms_max;

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

  std::string base_frame_;
  std::string output_topic_;
  std::string summary_topic_;
  std::string ray_observation_topic_;
  std::string self_filter_urdf_file_;

  std::vector<std::string> input_topics_;
  std::vector<std::string> sensor_frames_;

  double input_voxel_leaf_size_ = 0.03;
  double output_voxel_leaf_size_ = 0.03;
  double sensor_timeout_ = 0.25;
  double tf_timeout_ = 0.03;
  double publish_rate_ = 15.0;

  int ray_pixel_stride_ = 8;
  double ray_horizontal_fov_deg_ = 55.0;
  double ray_vertical_fov_deg_ = 72.0;
  double ray_max_forward_depth_ = 0.75;

  bool debug_occupancy_watch_enabled_ = false;
  double debug_occupancy_watch_x_ = 0.0;
  double debug_occupancy_watch_y_ = 0.0;
  double debug_occupancy_watch_z_ = 0.0;
  double debug_occupancy_watch_half_extent_ = 0.025;

  bool debug_hotspot_watch_enabled_ = false;
  double debug_hotspot_x_min_ = -0.025;
  double debug_hotspot_x_max_ = 0.075;
  double debug_hotspot_y_min_ = 0.025;
  double debug_hotspot_y_max_ = 0.125;
  double debug_hotspot_z_min_ = 0.225;
  double debug_hotspot_z_max_ = 0.425;

  std::vector<LinkGeometry> link_geometries_;
  std::size_t primitive_count_ = 0;

  std::vector<ros::Subscriber> subscribers_;
  ros::Publisher output_pub_;
  ros::Publisher ray_observation_pub_;
  ros::Publisher summary_pub_;
  ros::Timer publish_timer_;

  std::mutex state_mutex_;
  std::vector<SensorState> sensor_states_;

  ros::WallTime last_publish_wall_;
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
