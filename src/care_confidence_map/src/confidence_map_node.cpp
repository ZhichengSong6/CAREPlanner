#include <ros/ros.h>

#include <XmlRpcValue.h>

#include <care_confidence_map/QueryConfidence.h>
#include <care_confidence_map/body_sample_model.hpp>

#include <std_srvs/Trigger.h>
#include <std_msgs/String.h>

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
#include <limits>
#include <sstream>
#include <stdexcept>
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
  // Phase E3 semantic state. 0 = not occupied / unknown-or-free,
  // 1 = most recently observed as occupied. Confidence distinguishes
  // UNKNOWN (low confidence) from observed FREE (high confidence).
  float occupancy = 0.0f;
  double last_seen_time = -1.0;
};

struct LocalVisiblePoint
{
  tf2::Vector3 p_sensor;
};

struct CurrentBodyPriorSphere
{
  std::string link_name;
  tf2::Vector3 center_map;
  double radius = 0.0;
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

    if (!loadCurrentBodyPriorParams())
    {
      ROS_ERROR("[confidence_map_node] Failed to load current_body_prior params.");
      return false;
    }

    generateGlobalGrid();
    if (observation_mode_ == "ideal_fov")
    {
      generateSensorLocalVisiblePoints();
    }
    else
    {
      local_visible_points_.clear();
    }

    fov_marker_pub_ =
        nh_.advertise<visualization_msgs::MarkerArray>(fov_marker_topic_, 1, true);

    current_body_prior_marker_pub_ =
        nh_.advertise<visualization_msgs::MarkerArray>(
            current_body_prior_marker_topic_, 1, true);

    pointcloud_pub_ =
        nh_.advertise<sensor_msgs::PointCloud2>(pointcloud_topic_, 1, true);
    e3_summary_pub_ =
        nh_.advertise<std_msgs::String>(e3_summary_topic_, 1, true);

    if (observation_mode_ == "tof_ray")
    {
      ray_observation_sub_ =
          nh_.subscribe(
              ray_observation_topic_,
              1,
              &ConfidenceMapNode::rayObservationCallback,
              this);
    }

    query_service_ =
        nh_.advertiseService(query_service_name_,
                             &ConfidenceMapNode::handleQueryConfidence,
                             this);

    refresh_body_prior_service_ =
        nh_.advertiseService(refresh_body_prior_service_name_,
                             &ConfidenceMapNode::handleRefreshBodyPrior,
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

    if (current_body_prior_enabled_ && current_body_prior_refresh_on_startup_)
    {
      std::string refresh_msg;
      refreshCurrentBodyPrior(ros::Time::now(), "startup", &refresh_msg);
      ROS_INFO_STREAM("[confidence_map_node] startup current-body prior refresh: "
                      << refresh_msg);
    }

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
        "confidence_map/observation_mode",
        observation_mode_,
        "ideal_fov");
    nh.param<std::string>(
        "confidence_map/ray_observation_topic",
        ray_observation_topic_,
        "/care_planner/perception/tof_ray_observations");
    nh.param<std::string>(
        "confidence_map/e3_summary_topic",
        e3_summary_topic_,
        "/care_planner/confidence_map/e3_summary");
    nh.param(
        "confidence_map/ray_observation_timeout",
        ray_observation_timeout_,
        0.25);
    nh.param(
        "confidence_map/ray_free_start_range",
        ray_free_start_range_,
        0.15);
    nh.param(
        "confidence_map/ray_endpoint_guard_distance",
        ray_endpoint_guard_distance_,
        0.5 * resolution_);
    nh.param(
        "confidence_map/ray_step",
        ray_step_,
        0.5 * resolution_);
    nh.param(
        "confidence_map/ray_min_valid_range",
        ray_min_valid_range_,
        0.10);
    nh.param(
        "confidence_map/ray_max_valid_range",
        ray_max_valid_range_,
        1.50);

    // Phase E5 safety semantics for the current static-obstacle world.
    // Once a voxel is positively observed occupied, a later free traversal
    // must not erase it in one packet. Coarse 5-cm ray rounding and multiple
    // sensors otherwise create false-free holes directly through obstacles.
    nh.param(
        "confidence_map/ray_persistent_occupied",
        ray_persistent_occupied_,
        false);

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

    if (observation_mode_ != "ideal_fov" &&
        observation_mode_ != "tof_ray")
    {
      ROS_ERROR_STREAM(
          "[confidence_map_node] Invalid observation_mode='"
          << observation_mode_ << "'. Expected ideal_fov or tof_ray.");
      return false;
    }

    if (ray_observation_timeout_ <= 0.0 ||
        ray_free_start_range_ < 0.0 ||
        ray_endpoint_guard_distance_ < 0.0 ||
        ray_step_ <= 0.0 ||
        ray_min_valid_range_ < 0.0 ||
        ray_max_valid_range_ <= ray_min_valid_range_)
    {
      ROS_ERROR("[confidence_map_node] Invalid Phase E3 ray parameters.");
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

  bool loadCurrentBodyPriorParams()
  {
    ros::NodeHandle& nh = pnh_;

    nh.param("current_body_prior/enabled",
             current_body_prior_enabled_,
             true);

    nh.param("current_body_prior/refresh_on_startup",
             current_body_prior_refresh_on_startup_,
             true);

    nh.param("current_body_prior/lock_after_complete_refresh",
             current_body_prior_lock_after_complete_refresh_,
             false);

    nh.param("current_body_prior/risk_samples_only",
             current_body_prior_risk_samples_only_,
             false);

    nh.param("current_body_prior/inflation_radius",
             current_body_prior_inflation_radius_,
             0.10);

    nh.param("current_body_prior/tf_timeout",
             current_body_prior_tf_timeout_,
             0.05);

    nh.param<std::string>(
        "current_body_prior/body_samples_file",
        current_body_prior_body_samples_file_,
        "");

    nh.param<std::string>(
        "current_body_prior/refresh_service",
        refresh_body_prior_service_name_,
        "/care_planner/confidence_map/refresh_body_prior");

    nh.param<std::string>(
        "current_body_prior/marker_topic",
        current_body_prior_marker_topic_,
        "/care_planner/confidence_map/current_body_prior_markers");

    if (!current_body_prior_enabled_)
    {
      return true;
    }

    if (current_body_prior_inflation_radius_ < 0.0)
    {
      ROS_ERROR_STREAM("[confidence_map_node] Invalid current_body_prior/inflation_radius: "
                       << current_body_prior_inflation_radius_);
      return false;
    }

    if (current_body_prior_body_samples_file_.empty())
    {
      ROS_ERROR("[confidence_map_node] current_body_prior/body_samples_file is empty.");
      return false;
    }

    std::string error_msg;
    if (!current_body_sample_model_.loadFromYaml(
            current_body_prior_body_samples_file_, &error_msg))
    {
      ROS_ERROR_STREAM("[confidence_map_node] Failed to load body samples for current_body_prior: "
                       << error_msg);
      return false;
    }

    ROS_INFO_STREAM("[confidence_map_node] Loaded "
                    << current_body_sample_model_.size()
                    << " body samples for current_body_prior from "
                    << current_body_prior_body_samples_file_);

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
          p.occupancy = 0.0f;
          p.last_seen_time = -1.0;

          grid_points_.push_back(p);
        }
      }
    }

    ROS_INFO_STREAM("[confidence_map_node] Generated global confidence grid with "
                    << grid_points_.size() << " points.");
  }

  int gridLinearIndex(int ix, int iy, int iz) const
  {
    return ix * ny_ * nz_ + iy * nz_ + iz;
  }

  void markSphereAsKnownClear(
      const tf2::Vector3& center_map,
      double radius,
      const ros::Time& stamp,
      int* updated_cells)
  {
    if (updated_cells == nullptr)
    {
      return;
    }

    const double r = std::max(0.0, radius);
    const double r2 = r * r;

    int ix_min = static_cast<int>(
        std::floor((center_map.x() - r - x_min_) / resolution_));
    int ix_max = static_cast<int>(
        std::ceil((center_map.x() + r - x_min_) / resolution_));
    int iy_min = static_cast<int>(
        std::floor((center_map.y() - r - y_min_) / resolution_));
    int iy_max = static_cast<int>(
        std::ceil((center_map.y() + r - y_min_) / resolution_));
    int iz_min = static_cast<int>(
        std::floor((center_map.z() - r - z_min_) / resolution_));
    int iz_max = static_cast<int>(
        std::ceil((center_map.z() + r - z_min_) / resolution_));

    ix_min = std::max(0, ix_min);
    iy_min = std::max(0, iy_min);
    iz_min = std::max(0, iz_min);
    ix_max = std::min(nx_ - 1, ix_max);
    iy_max = std::min(ny_ - 1, iy_max);
    iz_max = std::min(nz_ - 1, iz_max);

    const double stamp_sec = stamp.toSec();

    for (int ix = ix_min; ix <= ix_max; ++ix)
    {
      const double x = x_min_ + static_cast<double>(ix) * resolution_;
      for (int iy = iy_min; iy <= iy_max; ++iy)
      {
        const double y = y_min_ + static_cast<double>(iy) * resolution_;
        for (int iz = iz_min; iz <= iz_max; ++iz)
        {
          const double z = z_min_ + static_cast<double>(iz) * resolution_;
          const double dx = x - center_map.x();
          const double dy = y - center_map.y();
          const double dz = z - center_map.z();

          if (dx * dx + dy * dy + dz * dz > r2)
          {
            continue;
          }

          GridPoint& gp = grid_points_[gridLinearIndex(ix, iy, iz)];
          gp.last_seen_time = stamp_sec;
          gp.confidence = 1.0f;
          gp.occupancy = 0.0f;

          // This is a robot-body known-clear prior, not a live sensor ray.
          // Do not set current_visibility=1 here.
          gp.current_visibility = 0.0f;

          *updated_cells += 1;
        }
      }
    }
  }

  bool refreshCurrentBodyPrior(
      const ros::Time& now,
      const std::string& reason,
      std::string* message)
  {
    if (current_body_prior_locked_)
    {
      if (message)
      {
        *message = "current_body_prior locked after complete initial refresh";
      }
      ROS_WARN_STREAM_THROTTLE(
          2.0,
          "[confidence_map_node] rejecting body-prior refresh after lock; reason="
              << reason);
      return true;
    }

    last_body_prior_spheres_.clear();
    last_body_prior_refresh_time_ = now;
    last_body_prior_updated_cells_ = 0;
    last_body_prior_transformed_samples_ = 0;
    last_body_prior_skipped_samples_ = 0;

    if (!current_body_prior_enabled_)
    {
      if (message)
      {
        *message = "current_body_prior disabled";
      }
      return true;
    }

    if (current_body_sample_model_.size() == 0)
    {
      if (message)
      {
        *message = "no body samples loaded";
      }
      return false;
    }

    const ros::Duration timeout(
        std::max(0.0, current_body_prior_tf_timeout_));

    for (const auto& sample : current_body_sample_model_.samples())
    {
      if (current_body_prior_risk_samples_only_ && !sample.include_for_risk)
      {
        continue;
      }

      geometry_msgs::TransformStamped tf_msg;
      try
      {
        tf_msg = tf_buffer_.lookupTransform(
            map_frame_,
            sample.frame_name,
            ros::Time(0),
            timeout);
      }
      catch (const tf2::TransformException& ex)
      {
        ++last_body_prior_skipped_samples_;
        ROS_WARN_THROTTLE(
            2.0,
            "[confidence_map_node] current_body_prior missing TF %s -> %s: %s",
            map_frame_.c_str(),
            sample.frame_name.c_str(),
            ex.what());
        continue;
      }

      const tf2::Transform T_map_link = transformMsgToTf2(tf_msg);
      const tf2::Vector3 center_map = T_map_link * sample.center_link;
      const double radius =
          sample.radius + current_body_prior_inflation_radius_;

      markSphereAsKnownClear(
          center_map, radius, now, &last_body_prior_updated_cells_);

      CurrentBodyPriorSphere sphere;
      sphere.link_name = sample.link_name;
      sphere.center_map = center_map;
      sphere.radius = radius;
      last_body_prior_spheres_.push_back(sphere);

      ++last_body_prior_transformed_samples_;
    }

    std::ostringstream oss;
    oss << "reason=" << reason
        << ", transformed_samples=" << last_body_prior_transformed_samples_
        << ", skipped_samples=" << last_body_prior_skipped_samples_
        << ", updated_cells=" << last_body_prior_updated_cells_
        << ", inflation_radius=" << current_body_prior_inflation_radius_;

    if (message)
    {
      *message = oss.str();
    }

    if (last_body_prior_transformed_samples_ <= 0)
    {
      ROS_WARN_STREAM("[confidence_map_node] current_body_prior refresh failed: "
                      << oss.str());
      return false;
    }

    ROS_INFO_STREAM_THROTTLE(
        1.0,
        "[confidence_map_node] current_body_prior refreshed: "
            << oss.str());

    publishCurrentBodyPriorMarkers();

    if (current_body_prior_lock_after_complete_refresh_ &&
        last_body_prior_transformed_samples_ > 0 &&
        last_body_prior_skipped_samples_ == 0)
    {
      current_body_prior_locked_ = true;
      ROS_WARN(
          "[confidence_map_node] initial trusted-free body prior LOCKED; "
          "future refresh requests cannot move/clear the map.");
    }

    return true;
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

  void applyTemporalDecay(const ros::Time& now)
  {
    const double now_sec = now.toSec();
    for (auto& p : grid_points_)
    {
      if (p.current_visibility > 0.5f)
      {
        continue;
      }

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

  void rayObservationCallback(
      const sensor_msgs::PointCloud2ConstPtr& msg)
  {
    if (!msg || observation_mode_ != "tof_ray")
    {
      return;
    }

    if (!msg->header.frame_id.empty() &&
        msg->header.frame_id != map_frame_)
    {
      ++ray_decode_failure_count_;
      ROS_ERROR_STREAM_THROTTLE(
          1.0,
          "[confidence_map_node] Phase E3 ray frame mismatch: msg='"
          << msg->header.frame_id << "' map='" << map_frame_
          << "'. E2 ray observations must be expressed in the map frame.");
      return;
    }

    std::vector<uint8_t> free_mask(grid_points_.size(), 0);
    std::vector<uint8_t> occupied_mask(grid_points_.size(), 0);

    std::size_t valid_rays = 0;
    std::size_t hit_rays = 0;
    std::size_t no_hit_rays = 0;
    std::size_t invalid_range = 0;
    std::size_t out_of_map_endpoints = 0;

    try
    {
      sensor_msgs::PointCloud2ConstIterator<float> x_it(*msg, "x");
      sensor_msgs::PointCloud2ConstIterator<float> y_it(*msg, "y");
      sensor_msgs::PointCloud2ConstIterator<float> z_it(*msg, "z");
      sensor_msgs::PointCloud2ConstIterator<float> ox_it(*msg, "origin_x");
      sensor_msgs::PointCloud2ConstIterator<float> oy_it(*msg, "origin_y");
      sensor_msgs::PointCloud2ConstIterator<float> oz_it(*msg, "origin_z");
      sensor_msgs::PointCloud2ConstIterator<int32_t> sensor_it(
          *msg, "sensor_id");
      sensor_msgs::PointCloud2ConstIterator<uint8_t> hit_it(
          *msg, "hit");

      for (; x_it != x_it.end();
           ++x_it, ++y_it, ++z_it,
           ++ox_it, ++oy_it, ++oz_it, ++sensor_it, ++hit_it)
      {
        (void)(*sensor_it);
        const bool hit = (*hit_it != 0);

        const tf2::Vector3 endpoint(
            static_cast<double>(*x_it),
            static_cast<double>(*y_it),
            static_cast<double>(*z_it));
        const tf2::Vector3 origin(
            static_cast<double>(*ox_it),
            static_cast<double>(*oy_it),
            static_cast<double>(*oz_it));

        if (!std::isfinite(endpoint.x()) ||
            !std::isfinite(endpoint.y()) ||
            !std::isfinite(endpoint.z()) ||
            !std::isfinite(origin.x()) ||
            !std::isfinite(origin.y()) ||
            !std::isfinite(origin.z()))
        {
          ++invalid_range;
          continue;
        }

        const tf2::Vector3 delta = endpoint - origin;
        const double range = delta.length();
        if (!std::isfinite(range) ||
            range < ray_min_valid_range_ ||
            range > ray_max_valid_range_)
        {
          ++invalid_range;
          continue;
        }

        ++valid_rays;
        if (hit)
        {
          ++hit_rays;
        }
        else
        {
          ++no_hit_rays;
        }
        const tf2::Vector3 direction = delta / range;

        const double free_end =
            hit
                ? std::max(
                      ray_free_start_range_,
                      range - ray_endpoint_guard_distance_)
                : range;

        if (free_end >= ray_free_start_range_)
        {
          for (double d = ray_free_start_range_;
               d < free_end;
               d += ray_step_)
          {
            const tf2::Vector3 p = origin + direction * d;
            int index = -1;
            if (positionToGridIndex(p, index))
            {
              free_mask[static_cast<std::size_t>(index)] = 1;
            }
          }
        }

        int endpoint_index = -1;
        if (positionToGridIndex(endpoint, endpoint_index))
        {
          if (hit)
          {
            occupied_mask[static_cast<std::size_t>(endpoint_index)] = 1;
          }
          else
          {
            free_mask[static_cast<std::size_t>(endpoint_index)] = 1;
          }
        }
        else
        {
          ++out_of_map_endpoints;
        }
      }
    }
    catch (const std::runtime_error& ex)
    {
      ++ray_decode_failure_count_;
      ROS_ERROR_STREAM_THROTTLE(
          1.0,
          "[confidence_map_node] Phase E3 ray observation decode failed: "
          << ex.what());
      return;
    }

    resetCurrentVisibility();

    const ros::Time stamp =
        msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp;
    const double stamp_sec = stamp.toSec();

    std::size_t free_cells = 0;
    std::size_t occupied_cells = 0;
    std::size_t occupied_free_clear_suppressed = 0;

    for (std::size_t i = 0; i < grid_points_.size(); ++i)
    {
      GridPoint& gp = grid_points_[i];

      // Occupied endpoint wins over free traversal if multiple sensors touch
      // the same coarse voxel during one fused observation packet.
      if (occupied_mask[i] != 0)
      {
        gp.occupancy = 1.0f;
        gp.current_visibility = 1.0f;
        gp.confidence = 1.0f;
        gp.last_seen_time = stamp_sec;
        ++occupied_cells;
      }
      else if (free_mask[i] != 0)
      {
        // Static-world safety mode: free-space traversal may refresh
        // visibility/confidence, but it cannot erase a previously observed
        // occupied voxel. This prevents last-observation-wins false frees.
        if (ray_persistent_occupied_ && gp.occupancy > 0.5f)
        {
          ++occupied_free_clear_suppressed;
        }
        else
        {
          gp.occupancy = 0.0f;
        }
        gp.current_visibility = 1.0f;
        gp.confidence = 1.0f;
        gp.last_seen_time = stamp_sec;
        ++free_cells;
      }
    }

    applyTemporalDecay(stamp);

    last_ray_observation_received_ = ros::Time::now();
    last_ray_observation_stamp_ = stamp;
    ++ray_packet_count_;
    last_ray_count_ = valid_rays;
    last_hit_ray_count_ = hit_rays;
    last_no_hit_ray_count_ = no_hit_rays;
    last_ray_free_cell_count_ = free_cells;
    last_ray_occupied_cell_count_ = occupied_cells;
    last_ray_occupied_free_clear_suppressed_count_ =
        occupied_free_clear_suppressed;
    total_ray_occupied_free_clear_suppressed_count_ +=
        occupied_free_clear_suppressed;
    last_ray_out_of_map_endpoint_count_ = out_of_map_endpoints;
    last_ray_invalid_range_count_ = invalid_range;

    printConfidenceStatsThrottled();
  }

  std_msgs::String makeE3SummaryMsg(const ros::Time& now) const
  {
    std::size_t visible_now = 0;
    std::size_t known_free = 0;
    std::size_t known_occupied = 0;
    std::size_t unknown = 0;

    for (const auto& gp : grid_points_)
    {
      if (gp.current_visibility > 0.5f)
      {
        ++visible_now;
      }

      if (gp.confidence <= 1e-4f)
      {
        ++unknown;
      }
      else if (gp.occupancy > 0.5f)
      {
        ++known_occupied;
      }
      else
      {
        ++known_free;
      }
    }

    const double observation_age =
        last_ray_observation_received_.isZero()
            ? std::numeric_limits<double>::quiet_NaN()
            : std::max(
                  0.0,
                  (now - last_ray_observation_received_).toSec());

    std_msgs::String msg;
    std::ostringstream oss;
    oss << "phase=E3"
        << " observation_mode=" << observation_mode_
        << " ray_packet_count=" << ray_packet_count_
        << " ray_decode_failure_count=" << ray_decode_failure_count_
        << " last_ray_count=" << last_ray_count_
        << " last_hit_ray_count=" << last_hit_ray_count_
        << " last_no_hit_ray_count=" << last_no_hit_ray_count_
        << " last_free_cell_count=" << last_ray_free_cell_count_
        << " last_occupied_cell_count=" << last_ray_occupied_cell_count_
        << " persistent_occupied=" << static_cast<int>(ray_persistent_occupied_)
        << " last_occupied_free_clear_suppressed_count="
        << last_ray_occupied_free_clear_suppressed_count_
        << " total_occupied_free_clear_suppressed_count="
        << total_ray_occupied_free_clear_suppressed_count_
        << " last_out_of_map_endpoint_count="
        << last_ray_out_of_map_endpoint_count_
        << " last_invalid_range_count=" << last_ray_invalid_range_count_
        << " visible_now=" << visible_now
        << " known_free=" << known_free
        << " known_occupied=" << known_occupied
        << " unknown=" << unknown
        << " observation_age_s=" << observation_age;
    msg.data = oss.str();
    return msg;
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

  bool handleRefreshBodyPrior(
      std_srvs::Trigger::Request&,
      std_srvs::Trigger::Response& res)
  {
    std::string message;
    const bool ok = refreshCurrentBodyPrior(
        ros::Time::now(), "service", &message);

    res.success = ok;
    res.message = message;
    return true;
  }

  void printConfidenceStatsThrottled() const
  {
    int currently_visible_count = 0;
    int confident_count = 0;
    int occupied_count = 0;
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
        if (p.occupancy > 0.5f)
        {
          ++occupied_count;
        }
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
        "[confidence_map_node] confidence stats: visible_now=%d / %zu, confident=%d, occupied=%d, ever_seen=%d, min=%.3f, max=%.3f, mean=%.3f",
        currently_visible_count,
        grid_points_.size(),
        confident_count,
        occupied_count,
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
        7,
        "x", 1, sensor_msgs::PointField::FLOAT32,
        "y", 1, sensor_msgs::PointField::FLOAT32,
        "z", 1, sensor_msgs::PointField::FLOAT32,
        "confidence", 1, sensor_msgs::PointField::FLOAT32,
        "current_visibility", 1, sensor_msgs::PointField::FLOAT32,
        "occupancy", 1, sensor_msgs::PointField::FLOAT32,
        "intensity", 1, sensor_msgs::PointField::FLOAT32);

    modifier.resize(grid_points_.size());

    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");
    sensor_msgs::PointCloud2Iterator<float> iter_conf(cloud, "confidence");
    sensor_msgs::PointCloud2Iterator<float> iter_vis(cloud, "current_visibility");
    sensor_msgs::PointCloud2Iterator<float> iter_occ(cloud, "occupancy");
    sensor_msgs::PointCloud2Iterator<float> iter_intensity(cloud, "intensity");

    for (const auto& p : grid_points_)
    {
      *iter_x = p.x;
      *iter_y = p.y;
      *iter_z = p.z;
      *iter_conf = p.confidence;
      *iter_vis = p.current_visibility;
      *iter_occ = p.occupancy;
      // RViz-compatible intensity: occupied known cells appear stronger than
      // free known cells while unknown remains near zero.
      *iter_intensity =
          p.occupancy > 0.5f ? 2.0f : p.confidence;

      ++iter_x;
      ++iter_y;
      ++iter_z;
      ++iter_conf;
      ++iter_vis;
      ++iter_occ;
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

  visualization_msgs::Marker makeCurrentBodyPriorSphereMarker(
      const CurrentBodyPriorSphere& sphere,
      int marker_id) const
  {
    visualization_msgs::Marker marker;

    marker.header.frame_id = map_frame_;
    marker.header.stamp = ros::Time::now();
    marker.ns = "current_body_prior_spheres";
    marker.id = marker_id;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;

    marker.pose.position.x = sphere.center_map.x();
    marker.pose.position.y = sphere.center_map.y();
    marker.pose.position.z = sphere.center_map.z();
    marker.pose.orientation.w = 1.0;

    marker.scale.x = 2.0 * sphere.radius;
    marker.scale.y = 2.0 * sphere.radius;
    marker.scale.z = 2.0 * sphere.radius;

    marker.color.r = 0.2f;
    marker.color.g = 1.0f;
    marker.color.b = 0.2f;
    marker.color.a = 0.12f;

    marker.lifetime = ros::Duration(0.0);
    return marker;
  }

  visualization_msgs::Marker makeCurrentBodyPriorTextMarker(int marker_id) const
  {
    visualization_msgs::Marker marker;

    marker.header.frame_id = map_frame_;
    marker.header.stamp = ros::Time::now();
    marker.ns = "current_body_prior_label";
    marker.id = marker_id;
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::Marker::ADD;

    marker.pose.position.x = 0.45;
    marker.pose.position.y = 0.45;
    marker.pose.position.z = 1.15;
    marker.pose.orientation.w = 1.0;
    marker.scale.z = 0.04;

    marker.color.r = 0.7f;
    marker.color.g = 1.0f;
    marker.color.b = 0.7f;
    marker.color.a = 1.0f;

    std::ostringstream oss;
    oss << "current body prior"
        << "\nbody samples + " << current_body_prior_inflation_radius_ << " m"
        << "\nconfidence=1, visibility=0"
        << "\nsamples=" << last_body_prior_transformed_samples_
        << ", cells=" << last_body_prior_updated_cells_;

    marker.text = oss.str();
    marker.lifetime = ros::Duration(0.0);
    return marker;
  }

  visualization_msgs::MarkerArray makeCurrentBodyPriorMarkerArray() const
  {
    visualization_msgs::MarkerArray array;

    visualization_msgs::Marker delete_marker;
    delete_marker.header.frame_id = map_frame_;
    delete_marker.header.stamp = ros::Time::now();
    delete_marker.action = visualization_msgs::Marker::DELETEALL;
    array.markers.push_back(delete_marker);

    if (!current_body_prior_enabled_)
    {
      return array;
    }

    int marker_id = 0;
    for (const auto& sphere : last_body_prior_spheres_)
    {
      array.markers.push_back(
          makeCurrentBodyPriorSphereMarker(sphere, marker_id++));
    }

    array.markers.push_back(makeCurrentBodyPriorTextMarker(marker_id++));
    return array;
  }

  void publishCurrentBodyPriorMarkers() const
  {
    if (!current_body_prior_marker_pub_)
    {
      return;
    }
    current_body_prior_marker_pub_.publish(
        makeCurrentBodyPriorMarkerArray());
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
    const ros::Time now = ros::Time::now();

    if (observation_mode_ == "ideal_fov")
    {
      std::vector<tf2::Transform> T_map_sensors;
      getSensorTransforms(T_map_sensors, nullptr);
      updateConfidenceMapSensorCentric(T_map_sensors, now);
      return;
    }

    // In tof_ray mode the observation callback owns the current visible set.
    // Only clear it when the entire E2 observation stream has gone stale;
    // otherwise a 30-Hz timer would flicker visibility between 15-Hz packets.
    if (last_ray_observation_received_.isZero() ||
        (now - last_ray_observation_received_).toSec() >
            ray_observation_timeout_)
    {
      resetCurrentVisibility();
    }

    applyTemporalDecay(now);
    printConfidenceStatsThrottled();
  }

  void publishTimerCallback(const ros::TimerEvent&)
  {
    visualization_msgs::MarkerArray marker_array;
    std::vector<tf2::Transform> T_map_sensors_unused;
    getSensorTransforms(T_map_sensors_unused, &marker_array);

    fov_marker_pub_.publish(marker_array);
    publishCurrentBodyPriorMarkers();

    sensor_msgs::PointCloud2 cloud = makeGridPointCloudMsg();
    pointcloud_pub_.publish(cloud);

    e3_summary_pub_.publish(makeE3SummaryMsg(ros::Time::now()));
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
    ROS_INFO_STREAM("observation_mode: " << observation_mode_);
    ROS_INFO_STREAM("ray_observation_topic: " << ray_observation_topic_);
    ROS_INFO_STREAM("e3_summary_topic: " << e3_summary_topic_);
    ROS_INFO_STREAM("query_service: " << query_service_name_);
    ROS_INFO_STREAM("marker_topic: " << marker_topic_);
    ROS_INFO_STREAM("pointcloud_topic: " << pointcloud_topic_);
    ROS_INFO_STREAM("fov_marker_topic: " << fov_marker_topic_);

    ROS_INFO_STREAM("");
    ROS_INFO_STREAM("current_body_prior:");
    ROS_INFO_STREAM("  enabled: " << current_body_prior_enabled_);
    ROS_INFO_STREAM("  body_samples_file: " << current_body_prior_body_samples_file_);
    ROS_INFO_STREAM("  inflation_radius: " << current_body_prior_inflation_radius_);
    ROS_INFO_STREAM("  refresh_on_startup: " << current_body_prior_refresh_on_startup_);
    ROS_INFO_STREAM("  risk_samples_only: " << current_body_prior_risk_samples_only_);
    ROS_INFO_STREAM("  refresh_service: " << refresh_body_prior_service_name_);
    ROS_INFO_STREAM("  marker_topic: " << current_body_prior_marker_topic_);
    ROS_INFO_STREAM("  last_transformed_samples: " << last_body_prior_transformed_samples_);
    ROS_INFO_STREAM("  last_updated_cells: " << last_body_prior_updated_cells_);

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
  ros::Publisher current_body_prior_marker_pub_;
  ros::Publisher pointcloud_pub_;
  ros::Publisher e3_summary_pub_;
  ros::Subscriber ray_observation_sub_;
  ros::ServiceServer query_service_;
  ros::ServiceServer refresh_body_prior_service_;

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

  std::string observation_mode_ = "ideal_fov";
  std::string ray_observation_topic_ =
      "/care_planner/perception/tof_ray_observations";
  std::string e3_summary_topic_ =
      "/care_planner/confidence_map/e3_summary";
  double ray_observation_timeout_ = 0.25;
  double ray_free_start_range_ = 0.15;
  double ray_endpoint_guard_distance_ = 0.025;
  double ray_step_ = 0.025;
  double ray_min_valid_range_ = 0.10;
  double ray_max_valid_range_ = 1.50;
  bool ray_persistent_occupied_ = false;

  ros::Time last_ray_observation_received_;
  ros::Time last_ray_observation_stamp_;
  std::size_t ray_packet_count_ = 0;
  std::size_t ray_decode_failure_count_ = 0;
  std::size_t last_ray_count_ = 0;
  std::size_t last_hit_ray_count_ = 0;
  std::size_t last_no_hit_ray_count_ = 0;
  std::size_t last_ray_free_cell_count_ = 0;
  std::size_t last_ray_occupied_cell_count_ = 0;
  std::size_t last_ray_occupied_free_clear_suppressed_count_ = 0;
  std::size_t total_ray_occupied_free_clear_suppressed_count_ = 0;
  std::size_t last_ray_out_of_map_endpoint_count_ = 0;
  std::size_t last_ray_invalid_range_count_ = 0;

  int nx_ = 0;
  int ny_ = 0;
  int nz_ = 0;

  std::string marker_topic_;
  std::string pointcloud_topic_;
  std::string fov_marker_topic_;
  std::string query_service_name_;

  bool current_body_prior_enabled_ = true;
  bool current_body_prior_refresh_on_startup_ = true;
  bool current_body_prior_lock_after_complete_refresh_ = false;
  bool current_body_prior_locked_ = false;
  bool current_body_prior_risk_samples_only_ = false;
  double current_body_prior_inflation_radius_ = 0.10;
  double current_body_prior_tf_timeout_ = 0.05;
  std::string current_body_prior_body_samples_file_;
  std::string refresh_body_prior_service_name_ =
      "/care_planner/confidence_map/refresh_body_prior";
  std::string current_body_prior_marker_topic_ =
      "/care_planner/confidence_map/current_body_prior_markers";
  care_confidence_map::BodySampleModel current_body_sample_model_;
  std::vector<CurrentBodyPriorSphere> last_body_prior_spheres_;
  ros::Time last_body_prior_refresh_time_;
  int last_body_prior_updated_cells_ = 0;
  int last_body_prior_transformed_samples_ = 0;
  int last_body_prior_skipped_samples_ = 0;

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