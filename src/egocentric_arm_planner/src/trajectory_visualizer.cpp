#include "egocentric_arm_planner/trajectory_visualizer.hpp"

namespace egocentric_arm_planner {

bool TrajectoryVisualizer::initialize(const ros::NodeHandle& nh,
                                      const ros::NodeHandle& pnh) {
  nh_ = nh;
  pnh_ = pnh;

  pnh_.param<std::string>("topics/target_pose",
                          target_pose_topic_,
                          target_pose_topic_);

  pnh_.param<std::string>("topics/task_trajectory",
                          task_trajectory_topic_,
                          task_trajectory_topic_);

  pnh_.param<std::string>("topics/command_trajectory",
                          command_trajectory_topic_,
                          command_trajectory_topic_);

  pnh_.param<std::string>("visualization/marker_topic",
                          marker_topic_,
                          marker_topic_);

  pnh_.param<double>("visualization/path_line_width",
                     path_line_width_,
                     path_line_width_);

  pnh_.param<double>("visualization/command_line_width",
                     command_line_width_,
                     command_line_width_);

  pnh_.param<double>("visualization/target_axis_length",
                     target_axis_length_,
                     target_axis_length_);

  pnh_.param<double>("visualization/target_axis_width",
                     target_axis_width_,
                     target_axis_width_);

  pnh_.param<double>("visualization/target_sphere_radius",
                     target_sphere_radius_,
                     target_sphere_radius_);

  robot_model_ = std::make_shared<arm_model::RobotModel>();
  if (!robot_model_->initializeFromRosParam(pnh_)) {
    ROS_ERROR("[TrajectoryVisualizer] Failed to initialize RobotModel.");
    return false;
  }

  target_pose_sub_ = nh_.subscribe(
      target_pose_topic_,
      1,
      &TrajectoryVisualizer::targetPoseCallback,
      this);

  task_traj_sub_ = nh_.subscribe(
      task_trajectory_topic_,
      1,
      &TrajectoryVisualizer::taskTrajectoryCallback,
      this);

  command_traj_sub_ = nh_.subscribe(
      command_trajectory_topic_,
      1,
      &TrajectoryVisualizer::commandTrajectoryCallback,
      this);

  marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
      marker_topic_,
      1,
      true);

  ROS_INFO("[TrajectoryVisualizer] Initialized.");
  ROS_INFO_STREAM("[TrajectoryVisualizer] target_pose_topic = " << target_pose_topic_);
  ROS_INFO_STREAM("[TrajectoryVisualizer] task_trajectory_topic = " << task_trajectory_topic_);
  ROS_INFO_STREAM("[TrajectoryVisualizer] command_trajectory_topic = " << command_trajectory_topic_);
  ROS_INFO_STREAM("[TrajectoryVisualizer] marker_topic = " << marker_topic_);

  return true;
}

void TrajectoryVisualizer::targetPoseCallback(
    const geometry_msgs::PoseStampedConstPtr& msg) {
  if (!msg) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_target_pose_ = *msg;
    has_target_pose_ = true;
  }

  publishMarkers();
}

void TrajectoryVisualizer::taskTrajectoryCallback(
    const trajectory_msgs::JointTrajectoryConstPtr& msg) {
  if (!msg) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_task_traj_ = *msg;
    has_task_traj_ = true;
  }

  publishMarkers();
}

void TrajectoryVisualizer::commandTrajectoryCallback(
    const trajectory_msgs::JointTrajectoryConstPtr& msg) {
  if (!msg) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_command_traj_ = *msg;
    has_command_traj_ = true;
  }

  publishMarkers();
}

void TrajectoryVisualizer::publishMarkers() {
  geometry_msgs::PoseStamped target_pose;
  trajectory_msgs::JointTrajectory task_traj;
  trajectory_msgs::JointTrajectory command_traj;

  bool has_target = false;
  bool has_task = false;
  bool has_command = false;

  {
    std::lock_guard<std::mutex> lock(data_mutex_);

    has_target = has_target_pose_;
    has_task = has_task_traj_;
    has_command = has_command_traj_;

    if (has_target) {
      target_pose = latest_target_pose_;
    }

    if (has_task) {
      task_traj = latest_task_traj_;
    }

    if (has_command) {
      command_traj = latest_command_traj_;
    }
  }

  visualization_msgs::MarkerArray marker_array;
  marker_array.markers.push_back(makeDeleteAllMarker());

  if (has_target) {
    appendTargetMarkers(target_pose, marker_array);
  }

  if (has_task) {
    std::vector<geometry_msgs::Point> task_points;
    if (trajectoryToEePath(task_traj, task_points)) {
      marker_array.markers.push_back(
          makePathMarker("task_ee_path",
                         10,
                         robot_model_->baseFrame(),
                         task_points,
                         0.1, 0.6, 1.0, 0.85,
                         path_line_width_));
    }
  }

  if (has_command) {
    std::vector<geometry_msgs::Point> command_points;
    if (trajectoryToEePath(command_traj, command_points)) {
      marker_array.markers.push_back(
          makePathMarker("command_ee_path",
                         20,
                         robot_model_->baseFrame(),
                         command_points,
                         1.0, 0.4, 0.1, 1.0,
                         command_line_width_));
    }
  }

  marker_pub_.publish(marker_array);
}

bool TrajectoryVisualizer::trajectoryToEePath(
    const trajectory_msgs::JointTrajectory& traj,
    std::vector<geometry_msgs::Point>& ee_points) const {
  ee_points.clear();

  if (!robot_model_) {
    ROS_ERROR("[TrajectoryVisualizer] robot_model is null.");
    return false;
  }

  if (traj.points.empty()) {
    return false;
  }

  ee_points.reserve(traj.points.size());

  for (std::size_t i = 0; i < traj.points.size(); ++i) {
    Eigen::VectorXd q;
    if (!buildQFromTrajectoryPoint(traj, i, q)) {
      return false;
    }

    Eigen::Isometry3d T_base_ee;
    if (!robot_model_->getEndEffectorPose(q, T_base_ee)) {
      ROS_WARN_THROTTLE(
          1.0,
          "[TrajectoryVisualizer] Failed to compute EE pose for trajectory point.");
      return false;
    }

    geometry_msgs::Point p;
    p.x = T_base_ee.translation().x();
    p.y = T_base_ee.translation().y();
    p.z = T_base_ee.translation().z();

    ee_points.push_back(p);
  }

  return !ee_points.empty();
}

bool TrajectoryVisualizer::buildQFromTrajectoryPoint(
    const trajectory_msgs::JointTrajectory& traj,
    std::size_t point_index,
    Eigen::VectorXd& q) const {
  if (!robot_model_) {
    return false;
  }

  if (point_index >= traj.points.size()) {
    return false;
  }

  const auto& point = traj.points[point_index];

  if (traj.joint_names.empty()) {
    ROS_ERROR("[TrajectoryVisualizer] trajectory joint_names is empty.");
    return false;
  }

  if (point.positions.size() != traj.joint_names.size()) {
    ROS_ERROR_STREAM(
        "[TrajectoryVisualizer] point.positions size does not match joint_names size. "
            << "positions = "
            << point.positions.size()
            << ", joint_names = "
            << traj.joint_names.size());
    return false;
  }

  q = Eigen::VectorXd::Zero(robot_model_->nq());

  for (std::size_t j = 0; j < traj.joint_names.size(); ++j) {
    int q_idx = -1;
    if (!robot_model_->getJointQIndex(traj.joint_names[j], q_idx)) {
      ROS_ERROR_STREAM("[TrajectoryVisualizer] Failed to get q index for joint: "
                       << traj.joint_names[j]);
      return false;
    }

    if (q_idx < 0 || q_idx >= robot_model_->nq()) {
      ROS_ERROR_STREAM("[TrajectoryVisualizer] Invalid q index for joint: "
                       << traj.joint_names[j]
                       << ", q_idx = "
                       << q_idx);
      return false;
    }

    q[q_idx] = point.positions[j];
  }

  return true;
}

visualization_msgs::Marker TrajectoryVisualizer::makeDeleteAllMarker() const {
  visualization_msgs::Marker marker;
  marker.action = visualization_msgs::Marker::DELETEALL;
  return marker;
}

visualization_msgs::Marker TrajectoryVisualizer::makePathMarker(
    const std::string& ns,
    int id,
    const std::string& frame_id,
    const std::vector<geometry_msgs::Point>& points,
    double r,
    double g,
    double b,
    double a,
    double line_width) const {
  visualization_msgs::Marker marker;

  marker.header.stamp = ros::Time::now();
  marker.header.frame_id = frame_id;

  marker.ns = ns;
  marker.id = id;
  marker.type = visualization_msgs::Marker::LINE_STRIP;
  marker.action = visualization_msgs::Marker::ADD;

  marker.pose.orientation.w = 1.0;

  marker.scale.x = line_width;

  marker.color.r = r;
  marker.color.g = g;
  marker.color.b = b;
  marker.color.a = a;

  marker.points = points;

  return marker;
}

void TrajectoryVisualizer::appendTargetMarkers(
    const geometry_msgs::PoseStamped& target,
    visualization_msgs::MarkerArray& marker_array) const {
  const std::string frame_id =
      target.header.frame_id.empty() ? robot_model_->baseFrame()
                                     : target.header.frame_id;

  visualization_msgs::Marker sphere;
  sphere.header.stamp = ros::Time::now();
  sphere.header.frame_id = frame_id;
  sphere.ns = "target_pose";
  sphere.id = 0;
  sphere.type = visualization_msgs::Marker::SPHERE;
  sphere.action = visualization_msgs::Marker::ADD;
  sphere.pose = target.pose;
  sphere.scale.x = target_sphere_radius_;
  sphere.scale.y = target_sphere_radius_;
  sphere.scale.z = target_sphere_radius_;
  sphere.color.r = 1.0;
  sphere.color.g = 1.0;
  sphere.color.b = 0.1;
  sphere.color.a = 1.0;

  marker_array.markers.push_back(sphere);

  Eigen::Quaterniond q(target.pose.orientation.w,
                       target.pose.orientation.x,
                       target.pose.orientation.y,
                       target.pose.orientation.z);

  if (q.norm() < 1e-9) {
    q = Eigen::Quaterniond::Identity();
  } else {
    q.normalize();
  }

  const Eigen::Matrix3d R = q.toRotationMatrix();

  const Eigen::Vector3d origin(target.pose.position.x,
                               target.pose.position.y,
                               target.pose.position.z);

  const Eigen::Vector3d axes[3] = {
      R.col(0),
      R.col(1),
      R.col(2)
  };

  const double colors[3][3] = {
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
      {0.0, 0.2, 1.0}
  };

  for (int i = 0; i < 3; ++i) {
    visualization_msgs::Marker arrow;
    arrow.header.stamp = ros::Time::now();
    arrow.header.frame_id = frame_id;
    arrow.ns = "target_axes";
    arrow.id = 1 + i;
    arrow.type = visualization_msgs::Marker::ARROW;
    arrow.action = visualization_msgs::Marker::ADD;

    arrow.scale.x = target_axis_width_;
    arrow.scale.y = 2.0 * target_axis_width_;
    arrow.scale.z = 2.0 * target_axis_width_;

    arrow.color.r = colors[i][0];
    arrow.color.g = colors[i][1];
    arrow.color.b = colors[i][2];
    arrow.color.a = 1.0;

    geometry_msgs::Point p0;
    p0.x = origin.x();
    p0.y = origin.y();
    p0.z = origin.z();

    const Eigen::Vector3d end = origin + target_axis_length_ * axes[i];

    geometry_msgs::Point p1;
    p1.x = end.x();
    p1.y = end.y();
    p1.z = end.z();

    arrow.points.push_back(p0);
    arrow.points.push_back(p1);

    marker_array.markers.push_back(arrow);
  }
}

}  // namespace egocentric_arm_planner