#include <ros/ros.h>

#include <XmlRpcValue.h>

#include <care_confidence_map/QueryConfidence.h>

#include <geometry_msgs/Point.h>
#include <geometry_msgs/TransformStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

struct SensorConfig
{
  int id = -1;
  std::string name;
  std::string frame;
};

struct GridPoint
{
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;

  float confidence = 0.0f;
  float current_visibility = 0.0f;
  double last_seen_time = -1.0;
};

struct LocalVisiblePoint
{
  tf2::Vector3 p_sensor;
};

class ConfidenceMapNode
{
public:
  ConfidenceMapNode()
    : nh_()
    , pnh_("~")
    , tf_buffer_()
    , tf_listener_(tf_buffer_)
  {
  }

  bool initialize()
  {
    if (!loadConfidenceMapParams())
    {
      ROS_ERROR("[confidence_map_node] Failed to load confidence_map params.");
      return false;
    }

    if (!loadSensorModelParams())
    {
      ROS_ERROR("[confidence_map_node] Failed to load sensor_model params.");
      return false;
    }

    generateGlobalGrid();
    generateSensorLocalVisiblePoints();

    fov_marker_pub_ =
        nh_.advertise<visualization_msgs::MarkerArray>(fov_marker_topic_, 1, true);

    pointcloud_pub_ =
        nh_.advertise<sensor_msgs::PointCloud2>(pointcloud_topic_, 1, true);

    query_service_ =
        nh_.advertiseService(query_service_name_,
                             &ConfidenceMapNode::handleQueryConfidence,
                             this);

    const double update_period = 1.0 / std::max(0.1, update_rate_);
    const double publish_period = 1.0 / std::max(0.1, publish_rate_);

    update_timer_ =
        nh_.createTimer(ros::Duration(update_period),
                        &ConfidenceMapNode::updateTimerCallback,
                        this);

    publish_timer_ =
        nh_.createTimer(ros::Duration(publish_period),
                        &ConfidenceMapNode::publishTimerCallback,
                        this);

    printSummary();
    return true;
  }

private:
  bool loadConfidenceMapParams()
  {
    ros::NodeHandle& nh = pnh_;

    if (!nh.getParam("confidence_map/frame", map_frame_))
    {
      ROS_ERROR("[confidence_map_node] Missing param: confidence_map/frame");
      return false;
    }

    nh.param("confidence_map/x_min", x_min_, -0.9);
    nh.param("confidence_map/x_max", x_max_,  0.9);
    nh.param("confidence_map/y_min", y_min_, -0.9);
    nh.param("confidence_map/y_max", y_max_,  0.9);
    nh.param("confidence_map/z_min", z_min_,  0.0);
    nh.param("confidence_map/z_max", z_max_,  1.10);

    nh.param("confidence_map/resolution", resolution_, 0.05);
    nh.param("confidence_map/update_rate", update_rate_, 30.0);
    nh.param("confidence_map/publish_rate", publish_rate_, 10.0);
    nh.param("confidence_map/temporal_decay_time", temporal_decay_time_, 2.0);

    nh.param<std::string>(
        "confidence_map/marker_topic",
        marker_topic_,
        "/care_planner/confidence_map/markers");

    nh.param<std::string>(
        "confidence_map/pointcloud_topic",
        pointcloud_topic_,
        "/care_planner/confidence_map/points");

    nh.param<std::string>(
        "confidence_map/fov_marker_topic",
        fov_marker_topic_,
        "/care_planner/confidence_map/fov_markers");

    nh.param<std::string>(
        "confidence_map/query_service",
        query_service_name_,
        "/care_planner/confidence_map/query");

    if (resolution_ <= 0.0)
    {
      ROS_ERROR_STREAM("[confidence_map_node] Invalid resolution: "
                       << resolution_ << ". It must be positive.");
      return false;
    }

    if (update_rate_ <= 0.0 || publish_rate_ <= 0.0)
    {
      ROS_ERROR_STREAM("[confidence_map_node] Invalid rates. update_rate="
                       << update_rate_ << ", publish_rate=" << publish_rate_);
      return false;
    }

    if (temporal_decay_time_ <= 0.0)
    {
      ROS_ERROR_STREAM("[confidence_map_node] Invalid temporal_decay_time: "
                       << temporal_decay_time_);
      return false;
    }

    if (x_max_ <= x_min_ || y_max_ <= y_min_ || z_max_ <= z_min_)
    {
      ROS_ERROR_STREAM("[confidence_map_node] Invalid map bounds.");
      return false;
    }

    nx_ = static_cast<int>(std::floor((x_max_ - x_min_) / resolution_)) + 1;
    ny_ = static_cast<int>(std::floor((y_max_ - y_min_) / resolution_)) + 1;
    nz_ = static_cast<int>(std::floor((z_max_ - z_min_) / resolution_)) + 1;

    if (nx_ <= 0 || ny_ <= 0 || nz_ <= 0)
    {
      ROS_ERROR_STREAM("[confidence_map_node] Invalid grid size: "
                       << nx_ << " x " << ny_ << " x " << nz_);
      return false;
    }

    return true;
  }

  bool loadSensorModelParams()
  {
    ros::NodeHandle& nh = pnh_;

    nh.param<std::string>("sensor_model/forward_axis", forward_axis_, "z");

    nh.param("sensor_model/tof_min_range", tof_min_range_, 0.05);
    nh.param("sensor_model/tof_max_range", tof_max_range_, 0.75);
    nh.param("sensor_model/horizontal_fov_deg", horizontal_fov_deg_, 55.0);
    nh.param("sensor_model/vertical_fov_deg", vertical_fov_deg_, 72.0);
    nh.param("sensor_model/local_resolution", local_resolution_, resolution_);

    if (local_resolution_ <= 0.0)
    {
      ROS_ERROR_STREAM("[confidence_map_node] Invalid sensor_model/local_resolution: "
                       << local_resolution_);
      return false;
    }

    if (tof_max_range_ <= tof_min_range_)
    {
      ROS_ERROR_STREAM("[confidence_map_node] Invalid ToF range: ["
                       << tof_min_range_ << ", " << tof_max_range_ << "]");
      return false;
    }

    if (horizontal_fov_deg_ <= 0.0 || vertical_fov_deg_ <= 0.0)
    {
      ROS_ERROR_STREAM("[confidence_map_node] Invalid FOV: horizontal="
                       << horizontal_fov_deg_
                       << ", vertical=" << vertical_fov_deg_);
      return false;
    }

    if (!parseForwardAxis(forward_axis_, forward_dir_))
    {
      ROS_ERROR_STREAM("[confidence_map_node] Invalid sensor_model/forward_axis: "
                       << forward_axis_
                       << ". Expected one of: x, y, z, -x, -y, -z.");
      return false;
    }

    buildSensorBasis();

    half_h_fov_rad_ = 0.5 * degToRad(horizontal_fov_deg_);
    half_v_fov_rad_ = 0.5 * degToRad(vertical_fov_deg_);
    tan_half_h_fov_ = std::tan(half_h_fov_rad_);
    tan_half_v_fov_ = std::tan(half_v_fov_rad_);

    XmlRpc::XmlRpcValue sensors_xml;
    if (!nh.getParam("sensor_model/sensors", sensors_xml))
    {
      ROS_ERROR("[confidence_map_node] Missing param: sensor_model/sensors");
      return false;
    }

    if (sensors_xml.getType() != XmlRpc::XmlRpcValue::TypeArray)
    {
      ROS_ERROR("[confidence_map_node] sensor_model/sensors must be a list.");
      return false;
    }

    sensors_.clear();

    for (int i = 0; i < sensors_xml.size(); ++i)
    {
      if (sensors_xml[i].getType() != XmlRpc::XmlRpcValue::TypeStruct)
      {
        ROS_WARN("[confidence_map_node] Skip invalid sensor entry %d because it is not a struct.", i);
        continue;
      }

      SensorConfig sensor;

      if (sensors_xml[i].hasMember("id"))
      {
        sensor.id = static_cast<int>(sensors_xml[i]["id"]);
      }

      if (sensors_xml[i].hasMember("name"))
      {
        sensor.name = static_cast<std::string>(sensors_xml[i]["name"]);
      }

      if (sensors_xml[i].hasMember("frame"))
      {
        sensor.frame = static_cast<std::string>(sensors_xml[i]["frame"]);
      }

      if (sensor.frame.empty())
      {
        ROS_WARN("[confidence_map_node] Skip sensor entry %d because frame is empty.", i);
        continue;
      }

      if (sensor.name.empty())
      {
        sensor.name = sensor.frame;
      }

      sensors_.push_back(sensor);
    }

    if (sensors_.empty())
    {
      ROS_ERROR("[confidence_map_node] No valid sensors loaded.");
      return false;
    }

    return true;
  }

  void generateGlobalGrid()
  {
    grid_points_.clear();
    grid_points_.reserve(static_cast<size_t>(nx_) *
                         static_cast<size_t>(ny_) *
                         static_cast<size_t>(nz_));

    for (int ix = 0; ix < nx_; ++ix)
    {
      const double x = x_min_ + static_cast<double>(ix) * resolution_;

      for (int iy = 0; iy < ny_; ++iy)
      {
        const double y = y_min_ + static_cast<double>(iy) * resolution_;

        for (int iz = 0; iz < nz_; ++iz)
        {
          const double z = z_min_ + static_cast<double>(iz) * resolution_;

          GridPoint p;
          p.x = static_cast<float>(x);
          p.y = static_cast<float>(y);
          p.z = static_cast<float>(z);
          p.confidence = 0.0f;
          p.current_visibility = 0.0f;
          p.last_seen_time = -1.0;

          grid_points_.push_back(p);
        }
      }
    }

    ROS_INFO_STREAM("[confidence_map_node] Generated global confidence grid with "
                    << grid_points_.size() << " points.");
  }

  void generateSensorLocalVisiblePoints()
  {
    local_visible_points_.clear();

    const int n_depth =
        static_cast<int>(std::floor((tof_max_range_ - tof_min_range_) / local_resolution_)) + 1;

    for (int id = 0; id < n_depth; ++id)
    {
      const double depth =
          tof_min_range_ + static_cast<double>(id) * local_resolution_;

      if (depth < tof_min_range_ || depth > tof_max_range_)
      {
        continue;
      }

      const double h_max = depth * tan_half_h_fov_;
      const double v_max = depth * tan_half_v_fov_;

      const int n_h =
          static_cast<int>(std::floor((2.0 * h_max) / local_resolution_)) + 1;
      const int n_v =
          static_cast<int>(std::floor((2.0 * v_max) / local_resolution_)) + 1;

      for (int ih = 0; ih < n_h; ++ih)
      {
        const double h =
            -h_max + static_cast<double>(ih) * local_resolution_;

        for (int iv = 0; iv < n_v; ++iv)
        {
          const double v =
              -v_max + static_cast<double>(iv) * local_resolution_;

          LocalVisiblePoint local_point;
          local_point.p_sensor =
              forward_dir_ * depth + right_dir_ * h + up_dir_ * v;

          local_visible_points_.push_back(local_point);
        }
      }

      LocalVisiblePoint center_point;
      center_point.p_sensor = forward_dir_ * depth;
      local_visible_points_.push_back(center_point);
    }

    ROS_INFO_STREAM("[confidence_map_node] Generated sensor-local visible points: "
                    << local_visible_points_.size()
                    << " per sensor at local_resolution="
                    << local_resolution_);
  }

  void resetCurrentVisibility()
  {
    for (auto& p : grid_points_)
    {
      p.current_visibility = 0.0f;
    }
  }

  bool positionToGridIndex(const tf2::Vector3& p_map, int& index) const
  {
    const int ix = static_cast<int>(
        std::llround((p_map.x() - x_min_) / resolution_));
    const int iy = static_cast<int>(
        std::llround((p_map.y() - y_min_) / resolution_));
    const int iz = static_cast<int>(
        std::llround((p_map.z() - z_min_) / resolution_));

    if (ix < 0 || ix >= nx_ ||
        iy < 0 || iy >= ny_ ||
        iz < 0 || iz >= nz_)
    {
      return false;
    }

    index = ix * ny_ * nz_ + iy * nz_ + iz;
    return true;
  }

  void updateConfidenceMapSensorCentric(
      const std::vector<tf2::Transform>& T_map_sensors,
      const ros::Time& now)
  {
    resetCurrentVisibility();

    for (const auto& T_map_sensor : T_map_sensors)
    {
      for (const auto& local_point : local_visible_points_)
      {
        const tf2::Vector3 p_map =
            T_map_sensor * local_point.p_sensor;

        int index = -1;
        if (!positionToGridIndex(p_map, index))
        {
          continue;
        }

        grid_points_[index].current_visibility = 1.0f;
      }
    }

    const double now_sec = now.toSec();

    for (auto& p : grid_points_)
    {
      if (p.current_visibility > 0.5f)
      {
        p.confidence = 1.0f;
        p.last_seen_time = now_sec;
      }
      else
      {
        if (p.last_seen_time >= 0.0)
        {
          const double dt = std::max(0.0, now_sec - p.last_seen_time);
          p.confidence =
              static_cast<float>(std::exp(-dt / temporal_decay_time_));
        }
        else
        {
          p.confidence = 0.0f;
        }
      }
    }

    printConfidenceStatsThrottled();
  }

  bool handleQueryConfidence(
      care_confidence_map::QueryConfidence::Request& req,
      care_confidence_map::QueryConfidence::Response& res)
  {
    res.confidence.clear();
    res.current_visibility.clear();
    res.inside_map.clear();

    res.confidence.reserve(req.points.size());
    res.current_visibility.reserve(req.points.size());
    res.inside_map.reserve(req.points.size());

    for (const auto& point : req.points)
    {
      const tf2::Vector3 p_map(point.x, point.y, point.z);

      int index = -1;
      if (!positionToGridIndex(p_map, index))
      {
        res.confidence.push_back(0.0f);
        res.current_visibility.push_back(0.0f);
        res.inside_map.push_back(0);
        continue;
      }

      const GridPoint& voxel = grid_points_[index];

      res.confidence.push_back(voxel.confidence);
      res.current_visibility.push_back(voxel.current_visibility);
      res.inside_map.push_back(1);
    }

    return true;
  }

  void printConfidenceStatsThrottled() const
  {
    int currently_visible_count = 0;
    int confident_count = 0;
    int ever_seen_count = 0;

    double min_conf = 1.0;
    double max_conf = 0.0;
    double sum_conf = 0.0;

    for (const auto& p : grid_points_)
    {
      const double c = static_cast<double>(p.confidence);
      min_conf = std::min(min_conf, c);
      max_conf = std::max(max_conf, c);
      sum_conf += c;

      if (p.current_visibility > 0.5f)
      {
        ++currently_visible_count;
      }

      if (c > 1e-4)
      {
        ++confident_count;
      }

      if (p.last_seen_time >= 0.0)
      {
        ++ever_seen_count;
      }
    }

    if (grid_points_.empty())
    {
      min_conf = 0.0;
    }

    const double mean_conf =
        grid_points_.empty()
            ? 0.0
            : sum_conf / static_cast<double>(grid_points_.size());

    ROS_INFO_THROTTLE(
        2.0,
        "[confidence_map_node] confidence stats: visible_now=%d / %zu, confident=%d, ever_seen=%d, min=%.3f, max=%.3f, mean=%.3f",
        currently_visible_count,
        grid_points_.size(),
        confident_count,
        ever_seen_count,
        min_conf,
        max_conf,
        mean_conf);
  }

  sensor_msgs::PointCloud2 makeGridPointCloudMsg() const
  {
    sensor_msgs::PointCloud2 cloud;

    cloud.header.frame_id = map_frame_;
    cloud.header.stamp = ros::Time::now();

    cloud.height = 1;
    cloud.width = static_cast<uint32_t>(grid_points_.size());

    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2Fields(
        6,
        "x", 1, sensor_msgs::PointField::FLOAT32,
        "y", 1, sensor_msgs::PointField::FLOAT32,
        "z", 1, sensor_msgs::PointField::FLOAT32,
        "confidence", 1, sensor_msgs::PointField::FLOAT32,
        "current_visibility", 1, sensor_msgs::PointField::FLOAT32,
        "intensity", 1, sensor_msgs::PointField::FLOAT32);

    modifier.resize(grid_points_.size());

    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");
    sensor_msgs::PointCloud2Iterator<float> iter_conf(cloud, "confidence");
    sensor_msgs::PointCloud2Iterator<float> iter_vis(cloud, "current_visibility");
    sensor_msgs::PointCloud2Iterator<float> iter_intensity(cloud, "intensity");

    for (const auto& p : grid_points_)
    {
      *iter_x = p.x;
      *iter_y = p.y;
      *iter_z = p.z;
      *iter_conf = p.confidence;
      *iter_vis = p.current_visibility;
      *iter_intensity = p.confidence;

      ++iter_x;
      ++iter_y;
      ++iter_z;
      ++iter_conf;
      ++iter_vis;
      ++iter_intensity;
    }

    return cloud;
  }

  static bool parseForwardAxis(const std::string& axis_name, tf2::Vector3& axis)
  {
    if (axis_name == "x")
    {
      axis = tf2::Vector3(1.0, 0.0, 0.0);
      return true;
    }
    if (axis_name == "-x")
    {
      axis = tf2::Vector3(-1.0, 0.0, 0.0);
      return true;
    }
    if (axis_name == "y")
    {
      axis = tf2::Vector3(0.0, 1.0, 0.0);
      return true;
    }
    if (axis_name == "-y")
    {
      axis = tf2::Vector3(0.0, -1.0, 0.0);
      return true;
    }
    if (axis_name == "z")
    {
      axis = tf2::Vector3(0.0, 0.0, 1.0);
      return true;
    }
    if (axis_name == "-z")
    {
      axis = tf2::Vector3(0.0, 0.0, -1.0);
      return true;
    }

    return false;
  }

  void buildSensorBasis()
  {
    if (forward_axis_ == "x" || forward_axis_ == "-x")
    {
      right_dir_ = tf2::Vector3(0.0, 1.0, 0.0);
      up_dir_    = tf2::Vector3(0.0, 0.0, 1.0);
    }
    else if (forward_axis_ == "y" || forward_axis_ == "-y")
    {
      right_dir_ = tf2::Vector3(1.0, 0.0, 0.0);
      up_dir_    = tf2::Vector3(0.0, 0.0, 1.0);
    }
    else
    {
      right_dir_ = tf2::Vector3(1.0, 0.0, 0.0);
      up_dir_    = tf2::Vector3(0.0, 1.0, 0.0);
    }
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

  std::vector<tf2::Vector3> computeFrustumCornersInSensorFrame() const
  {
    const double range = tof_max_range_;
    const double half_h = range * tan_half_h_fov_;
    const double half_v = range * tan_half_v_fov_;

    const tf2::Vector3 center = forward_dir_ * range;

    std::vector<tf2::Vector3> corners;
    corners.reserve(4);

    corners.push_back(center + right_dir_ * half_h + up_dir_ * half_v);
    corners.push_back(center + right_dir_ * half_h - up_dir_ * half_v);
    corners.push_back(center - right_dir_ * half_h - up_dir_ * half_v);
    corners.push_back(center - right_dir_ * half_h + up_dir_ * half_v);

    return corners;
  }

  double degToRad(double deg) const
  {
    return deg * M_PI / 180.0;
  }

  visualization_msgs::Marker makeSensorFovMarker(
      const SensorConfig& sensor,
      const tf2::Transform& T_map_sensor,
      int marker_id) const
  {
    visualization_msgs::Marker marker;

    marker.header.frame_id = map_frame_;
    marker.header.stamp = ros::Time::now();

    marker.ns = "sensor_fov";
    marker.id = marker_id;
    marker.type = visualization_msgs::Marker::LINE_LIST;
    marker.action = visualization_msgs::Marker::ADD;

    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.006;

    marker.color.r = 0.0f;
    marker.color.g = 0.8f;
    marker.color.b = 1.0f;
    marker.color.a = 0.9f;

    marker.lifetime = ros::Duration(1.0 / std::max(0.1, publish_rate_) * 2.5);

    const tf2::Vector3 origin_sensor(0.0, 0.0, 0.0);
    const tf2::Vector3 origin_map = T_map_sensor * origin_sensor;

    const auto corners_sensor = computeFrustumCornersInSensorFrame();

    std::vector<tf2::Vector3> corners_map;
    corners_map.reserve(4);
    for (const auto& c_sensor : corners_sensor)
    {
      corners_map.push_back(T_map_sensor * c_sensor);
    }

    auto addLine = [&](const tf2::Vector3& a, const tf2::Vector3& b)
    {
      marker.points.push_back(toPointMsg(a));
      marker.points.push_back(toPointMsg(b));
    };

    for (const auto& c_map : corners_map)
    {
      addLine(origin_map, c_map);
    }

    addLine(corners_map[0], corners_map[1]);
    addLine(corners_map[1], corners_map[2]);
    addLine(corners_map[2], corners_map[3]);
    addLine(corners_map[3], corners_map[0]);

    const tf2::Vector3 center_sensor = forward_dir_ * tof_max_range_;
    const tf2::Vector3 center_map = T_map_sensor * center_sensor;
    addLine(origin_map, center_map);

    return marker;
  }

  visualization_msgs::Marker makeSensorOriginMarker(
      const SensorConfig& sensor,
      const tf2::Transform& T_map_sensor,
      int marker_id) const
  {
    visualization_msgs::Marker marker;

    marker.header.frame_id = map_frame_;
    marker.header.stamp = ros::Time::now();

    marker.ns = "sensor_origin";
    marker.id = marker_id;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;

    const tf2::Vector3 p = T_map_sensor.getOrigin();
    marker.pose.position.x = p.x();
    marker.pose.position.y = p.y();
    marker.pose.position.z = p.z();
    marker.pose.orientation.w = 1.0;

    marker.scale.x = 0.025;
    marker.scale.y = 0.025;
    marker.scale.z = 0.025;

    marker.color.r = 1.0f;
    marker.color.g = 0.4f;
    marker.color.b = 0.0f;
    marker.color.a = 0.9f;

    marker.lifetime = ros::Duration(1.0 / std::max(0.1, publish_rate_) * 2.5);

    return marker;
  }

  visualization_msgs::Marker makeSensorTextMarker(
      const SensorConfig& sensor,
      const tf2::Transform& T_map_sensor,
      int marker_id) const
  {
    visualization_msgs::Marker marker;

    marker.header.frame_id = map_frame_;
    marker.header.stamp = ros::Time::now();

    marker.ns = "sensor_label";
    marker.id = marker_id;
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::Marker::ADD;

    const tf2::Vector3 p = T_map_sensor.getOrigin();
    marker.pose.position.x = p.x();
    marker.pose.position.y = p.y();
    marker.pose.position.z = p.z() + 0.04;
    marker.pose.orientation.w = 1.0;

    marker.scale.z = 0.035;

    marker.color.r = 1.0f;
    marker.color.g = 1.0f;
    marker.color.b = 1.0f;
    marker.color.a = 0.9f;

    marker.text = sensor.name;
    marker.lifetime = ros::Duration(1.0 / std::max(0.1, publish_rate_) * 2.5);

    return marker;
  }

  bool getSensorTransforms(
      std::vector<tf2::Transform>& T_map_sensors,
      visualization_msgs::MarkerArray* marker_array)
  {
    T_map_sensors.clear();

    int ready_count = 0;
    int marker_id = 0;

    for (const auto& sensor : sensors_)
    {
      geometry_msgs::TransformStamped tf_msg;

      try
      {
        tf_msg = tf_buffer_.lookupTransform(
            map_frame_,
            sensor.frame,
            ros::Time(0),
            ros::Duration(0.002));
      }
      catch (const tf2::TransformException& ex)
      {
        ROS_WARN_THROTTLE(
            2.0,
            "[confidence_map_node] Missing TF %s -> %s: %s",
            map_frame_.c_str(),
            sensor.frame.c_str(),
            ex.what());
        continue;
      }

      ++ready_count;

      const tf2::Transform T_map_sensor = transformMsgToTf2(tf_msg);
      T_map_sensors.push_back(T_map_sensor);

      if (marker_array != nullptr)
      {
        marker_array->markers.push_back(
            makeSensorFovMarker(sensor, T_map_sensor, marker_id++));
        marker_array->markers.push_back(
            makeSensorOriginMarker(sensor, T_map_sensor, marker_id++));
        marker_array->markers.push_back(
            makeSensorTextMarker(sensor, T_map_sensor, marker_id++));
      }
    }

    if (ready_count != last_ready_count_)
    {
      ROS_INFO_STREAM("[confidence_map_node] Sensor TF ready: "
                      << ready_count << " / " << sensors_.size());
      last_ready_count_ = ready_count;
    }

    return ready_count > 0;
  }

  void updateTimerCallback(const ros::TimerEvent&)
  {
    std::vector<tf2::Transform> T_map_sensors;
    getSensorTransforms(T_map_sensors, nullptr);

    updateConfidenceMapSensorCentric(T_map_sensors, ros::Time::now());
  }

  void publishTimerCallback(const ros::TimerEvent&)
  {
    visualization_msgs::MarkerArray marker_array;
    std::vector<tf2::Transform> T_map_sensors_unused;
    getSensorTransforms(T_map_sensors_unused, &marker_array);

    fov_marker_pub_.publish(marker_array);

    sensor_msgs::PointCloud2 cloud = makeGridPointCloudMsg();
    pointcloud_pub_.publish(cloud);
  }

  void printSummary() const
  {
    const long long total_points =
        static_cast<long long>(nx_) *
        static_cast<long long>(ny_) *
        static_cast<long long>(nz_);

    ROS_INFO_STREAM("");
    ROS_INFO_STREAM("========== care_confidence_map ==========");
    ROS_INFO_STREAM("map frame: " << map_frame_);
    ROS_INFO_STREAM("bounds:");
    ROS_INFO_STREAM("  x: [" << x_min_ << ", " << x_max_ << "]");
    ROS_INFO_STREAM("  y: [" << y_min_ << ", " << y_max_ << "]");
    ROS_INFO_STREAM("  z: [" << z_min_ << ", " << z_max_ << "]");
    ROS_INFO_STREAM("global resolution: " << resolution_);
    ROS_INFO_STREAM("grid size: "
                    << nx_ << " x "
                    << ny_ << " x "
                    << nz_ << " = "
                    << total_points << " points");
    ROS_INFO_STREAM("update_rate: " << update_rate_);
    ROS_INFO_STREAM("publish_rate: " << publish_rate_);
    ROS_INFO_STREAM("temporal_decay_time: " << temporal_decay_time_);
    ROS_INFO_STREAM("query_service: " << query_service_name_);
    ROS_INFO_STREAM("marker_topic: " << marker_topic_);
    ROS_INFO_STREAM("pointcloud_topic: " << pointcloud_topic_);
    ROS_INFO_STREAM("fov_marker_topic: " << fov_marker_topic_);

    ROS_INFO_STREAM("");
    ROS_INFO_STREAM("sensor model:");
    ROS_INFO_STREAM("  forward_axis: " << forward_axis_);
    ROS_INFO_STREAM("  range: [" << tof_min_range_ << ", " << tof_max_range_ << "]");
    ROS_INFO_STREAM("  horizontal_fov_deg: " << horizontal_fov_deg_);
    ROS_INFO_STREAM("  vertical_fov_deg: " << vertical_fov_deg_);
    ROS_INFO_STREAM("  local_resolution: " << local_resolution_);
    ROS_INFO_STREAM("  local visible points per sensor: "
                    << local_visible_points_.size());

    ROS_INFO_STREAM("");
    ROS_INFO_STREAM("sensors:");
    for (const auto& sensor : sensors_)
    {
      ROS_INFO_STREAM("  id=" << sensor.id
                      << ", name=" << sensor.name
                      << ", frame=" << sensor.frame);
    }

    ROS_INFO_STREAM("=========================================");
  }

private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  ros::Publisher fov_marker_pub_;
  ros::Publisher pointcloud_pub_;
  ros::ServiceServer query_service_;

  ros::Timer update_timer_;
  ros::Timer publish_timer_;

  std::string map_frame_;

  double x_min_ = -0.9;
  double x_max_ =  0.9;
  double y_min_ = -0.9;
  double y_max_ =  0.9;
  double z_min_ =  0.0;
  double z_max_ =  1.10;

  double resolution_ = 0.05;
  double update_rate_ = 30.0;
  double publish_rate_ = 10.0;
  double temporal_decay_time_ = 2.0;

  int nx_ = 0;
  int ny_ = 0;
  int nz_ = 0;

  std::string marker_topic_;
  std::string pointcloud_topic_;
  std::string fov_marker_topic_;
  std::string query_service_name_;

  std::string forward_axis_;
  tf2::Vector3 forward_dir_;
  tf2::Vector3 right_dir_;
  tf2::Vector3 up_dir_;

  double tof_min_range_ = 0.05;
  double tof_max_range_ = 0.75;
  double horizontal_fov_deg_ = 55.0;
  double vertical_fov_deg_ = 72.0;
  double local_resolution_ = 0.05;

  double half_h_fov_rad_ = 0.0;
  double half_v_fov_rad_ = 0.0;
  double tan_half_h_fov_ = 0.0;
  double tan_half_v_fov_ = 0.0;

  std::vector<SensorConfig> sensors_;
  std::vector<GridPoint> grid_points_;
  std::vector<LocalVisiblePoint> local_visible_points_;

  int last_ready_count_ = -1;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "confidence_map_node");

  ConfidenceMapNode node;
  if (!node.initialize())
  {
    ROS_ERROR("[confidence_map_node] Initialization failed.");
    return 1;
  }

  ros::spin();
  return 0;
}