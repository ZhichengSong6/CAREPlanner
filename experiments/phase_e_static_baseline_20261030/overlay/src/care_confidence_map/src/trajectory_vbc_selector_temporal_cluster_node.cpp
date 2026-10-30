#include <care_confidence_map/trajectory_risk_evaluator.hpp>
#include <care_confidence_map/QueryConfidence.h>
#include <care_confidence_map/QueryConfidenceSnapshot.h>
#include <care_confidence_map/vbc_evidence.hpp>
#include <care_confidence_map/vbc_primitive_model.hpp>
#include <care_confidence_map/analytic_ground_support.hpp>
#include <care_confidence_map/initial_shell_recovery.hpp>
#include <care_confidence_map/trajectory_motion.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <ros/ros.h>
#include <ros/callback_queue.h>

#include <geometry_msgs/Point.h>
#include <geometry_msgs/PointStamped.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/String.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <sensor_msgs/JointState.h>

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
// VBC candidates are confidence-map voxel centers inside the configured body
// geometry. Primitive sweeps enclose every original piecewise-linear interval,
// including its endpoints; samples retain their legacy discrete semantics.
// VBC margin is independent of GCDF inflation.
// Every low-confidence swept voxel is then evaluated independently with analytic
// VBC. Steering semantics
// are coarser: violating candidates are first grouped into ordered temporal
// layers, each layer spanning at most temporal_layer_mpc_steps original MPC
// intervals.  Every temporal layer is then spatially clustered only for
// diagnostics / physical interpretation.  ALL points from ALL spatial regions
// in the earliest unsafe temporal layer are published as one learned-steering
// active set.
class TrajectoryVbcTemporalClusterNode
{
  friend struct VbcPrimitiveTest;
  friend struct VbcPrimitiveReplay;
public:
  TrajectoryVbcTemporalClusterNode() : nh_(), pnh_("~"), measured_spinner_(1,&measured_queue_) {}
  ~TrajectoryVbcTemporalClusterNode() { measured_spinner_.stop(); }

  bool initialize()
  {
    loadParams();
    // Midpoints without displacement bounds can skip initial shell points.
    // Keep that ablation in the offline geometry API, never a live verifier.
    if (geometry_backend_ == "primitive" && !continuous_motion_bound_enabled_) {
      ROS_ERROR("Primitive VBC requires continuous_motion_bound_enabled=true; sampled-only verification is unsupported");
      return false;
    }
    pnh_.param("trajectory_vbc/include_measured_body", include_measured_body_, false);
    pnh_.param("trajectory_vbc/initial_shell_recovery_enabled", initial_shell_recovery_enabled_, false);
    pnh_.param("trajectory_vbc/initial_shell_execution_margin_m", shell_execution_margin_m_, 0.);
    pnh_.param("trajectory_vbc/fixed_start_rebase_reaudit_enabled", fixed_start_rebase_reaudit_enabled_, false);
    if(fixed_start_rebase_reaudit_enabled_ && !initial_shell_recovery_enabled_)return false;
    pnh_.param("trajectory_vbc/initial_shell_occupied_margin_m", shell_occupied_margin_m_, 0.);
    if (initial_shell_recovery_enabled_ && (geometry_backend_!="primitive" ||
        !std::isfinite(shell_execution_margin_m_) || shell_execution_margin_m_<0. ||
        shell_execution_margin_m_>=swept_volume_margin_m_ || !std::isfinite(shell_occupied_margin_m_))) return false;
    if (include_measured_body_ || initial_shell_recovery_enabled_) {
      // Receive state while the serialized verifier performs FK and map RPCs.
      // Only this immutable pointer is updated here; geometry, evidence and the
      // per-audit FK ledger stay on the original single verification queue.
      auto options=ros::SubscribeOptions::create<sensor_msgs::JointState>(
          "/care_arm/joint_states",1,[this](const sensor_msgs::JointStateConstPtr& msg) {
            std::lock_guard<std::mutex> lock(mutex_);latest_measured_=msg;
          },ros::VoidConstPtr(),&measured_queue_);
      measured_sub_=nh_.subscribe(options);
      measured_spinner_.start();
    }
    if (initial_shell_recovery_enabled_) {
      recovery_context_sub_=nh_.subscribe<std_msgs::String>(
          predicted_trajectory_topic_+"/initial_shell_context",4,
          &TrajectoryVbcTemporalClusterNode::recoveryContextCallback,this);
      recovery_quality_sub_=nh_.subscribe<std_msgs::String>(
          predicted_trajectory_topic_+"/initial_shell_quality",4,
          &TrajectoryVbcTemporalClusterNode::recoveryQualityCallback,this);
      recovery_protocol_pub_=nh_.advertise<std_msgs::String>(
          predicted_trajectory_topic_+"/protocol_status",8,false);
      shell_diagnostic_pub_=nh_.advertise<std_msgs::String>(
          predicted_trajectory_topic_+"/initial_shell_diagnostic",32,false);
    }

    std::string error_msg;
    if (geometry_backend_ != "samples" && geometry_backend_ != "primitive")
    {
      ROS_ERROR_STREAM("Unknown VBC geometry backend: " << geometry_backend_);
      return false;
    }
    const bool geometry_ok = geometry_backend_ == "primitive"
        ? (evaluator_.initializeKinematics(robot_urdf_file_, base_frame_, &error_msg) &&
           primitive_model_.load(primitive_urdf_file_, whole_voxel_support_ ? std::vector<std::string>{} : ignored_risk_links_, &error_msg))
        : evaluator_.initialize(robot_urdf_file_, body_samples_file_, base_frame_, &error_msg);
    if (!geometry_ok)
    {
      ROS_ERROR_STREAM(
          "[trajectory_vbc_temporal] Failed to initialize evaluator: "
          << error_msg);
      return false;
    }
    if(whole_voxel_support_) {
      const char* file=std::getenv("CARE_ANALYTIC_GROUND_FILE");
      const char* apply=std::getenv("CARE_ANALYTIC_GROUND_APPLY");
      if(file && *file && apply && std::string(apply)=="true") {
        try {
          analytic_support_=care_confidence_map::AnalyticGroundSupport::load(file);
          if(!analytic_support_.association_enabled || analytic_support_.frame!=base_frame_)return false;
          clip_analytic_support_=true;
        } catch(const std::exception& e){ROS_ERROR_STREAM(e.what());return false;}
      }
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
        rediscovery_trajectory_timeout_ <= 0.0 ||
        predicted_periodic_refresh_rate_ <= 0.0 ||
        temporal_layer_mpc_steps_ < 0 || region_max_diameter_m_ <= 0.0 ||
        tof_min_range_ < 0.0 || tof_max_range_ <= tof_min_range_ ||
        horizontal_fov_deg_ <= 0.0 || vertical_fov_deg_ <= 0.0 ||
        !std::isfinite(swept_volume_margin_m_) || !std::isfinite(map_resolution_) ||
        !std::isfinite(map_x_min_) || !std::isfinite(map_x_max_) ||
        !std::isfinite(map_y_min_) || !std::isfinite(map_y_max_) ||
        !std::isfinite(map_z_min_) || !std::isfinite(map_z_max_) ||
        swept_volume_margin_m_ < 0.0 || map_resolution_ <= 0.0 ||
        map_x_max_ < map_x_min_ || map_y_max_ < map_y_min_ ||
        map_z_max_ < map_z_min_)
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

    if (geometry_backend_ == "primitive")
    {
      std::vector<care_confidence_map::VbcPrimitiveFrame> initial_geometry;
      if (!primitive_model_.computeTrajectory(evaluator_, {q0}, &initial_geometry, &error_msg))
      {
        ROS_ERROR_STREAM("Invalid primitive frame configuration: " << error_msg);
        return false;
      }
    }

    confidence_query_client_ =
        nh_.serviceClient<care_confidence_map::QueryConfidence>(
            confidence_query_service_, confidence_query_persistent_);
    if(initial_shell_recovery_enabled_) shell_snapshot_client_=nh_.serviceClient<care_confidence_map::QueryConfidenceSnapshot>(
        confidence_query_service_+"_snapshot");

    bool tcp_nodelay = false;
    pnh_.param("trajectory_vbc/tcp_nodelay", tcp_nodelay, false);
    const auto trajectory_transport = ros::TransportHints().tcpNoDelay(tcp_nodelay);
    trajectory_sub_ = nh_.subscribe(
        input_trajectory_topic_, 1,
        &TrajectoryVbcTemporalClusterNode::trajectoryCallback, this, trajectory_transport);
    predicted_trajectory_sub_ = nh_.subscribe(
        predicted_trajectory_topic_, 1,
        &TrajectoryVbcTemporalClusterNode::predictedTrajectoryCallback, this, trajectory_transport);
    if (!rediscovery_trajectory_topic_.empty())
    {
      rediscovery_trajectory_sub_ = nh_.subscribe(
          rediscovery_trajectory_topic_, 1,
          &TrajectoryVbcTemporalClusterNode::rediscoveryTrajectoryCallback, this, trajectory_transport);
    }
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
    active_set_bundle_pub_ = nh_.advertise<std_msgs::Float64MultiArray>(
        active_set_bundle_topic_, 1, true);

    publishCandidateActive(false);
    publishActiveSet({});
    publishActiveSetBundle({}, std::numeric_limits<double>::quiet_NaN());

    timer_ = nh_.createTimer(
        ros::Duration(1.0 / std::max(0.1, eval_rate_)),
        &TrajectoryVbcTemporalClusterNode::timerCallback, this);
    // A unique audit identity may be a few ns ahead of a repeated /clock tick.
    // Retry only that pending identity when time catches up; keep its deadline.
    context_clock_timer_ = nh_.createWallTimer(ros::WallDuration(.001),
        &TrajectoryVbcTemporalClusterNode::contextClockCallback, this);

    printSummary();
    return true;
  }

private:
  double voxelSupportRadius() const {return whole_voxel_support_ ? .5*std::sqrt(3.)*map_resolution_ : 0.;}
  double supportDistance(const care_confidence_map::VbcPrimitive& body,const Eigen::Vector3d& point) const {
    return body.signedDistance(point)-voxelSupportRadius();
  }
  bool proveSupportOutsideContinuousTrajectory(const care_confidence_map::TrajectoryRiskEvaluator& fk,
      const std::vector<Eigen::VectorXd>& q,const Eigen::Vector3d& point,double margin,int* remaining,
      const std::vector<Eigen::VectorXd>& bend={},double* slack=nullptr,std::string* diagnostic=nullptr,
      care_confidence_map::VbcPrimitiveModel::ProofFkCache* cache=nullptr,double desired=0.,
      const std::function<bool()>& may_refine={}) const {
    // Every point of a5cm voxel lies within its half diagonal of the center.
    // 1-Lipschitz point SDF therefore certifies the full voxel at the original
    // physical margin. The returned slack is already relative to margin+r.
    return primitive_model_.provePointOutsideContinuousTrajectory(fk,q,point,margin+voxelSupportRadius(),
        remaining,bend,slack,diagnostic,cache,desired,may_refine);
  }
  struct SweepVoxel
  {
    Eigen::Vector3d point_base = Eigen::Vector3d::Zero();
    std::string link_name = "none";
    int sample_index_in_link = -1;
    std::string source_type = "unknown";
    int source_collision_index = -1;
    std::string source_collision_name;
    double primitive_signed_distance_m = std::numeric_limits<double>::quiet_NaN();
    Eigen::Vector3d sample_center_base = Eigen::Vector3d::Zero();
    double raw_sample_radius_m = 0.0;
    double swept_radius_m = 0.0;
    double point_center_distance_m = 0.0;
    int sweep_eval_timestep = -1;
    int sweep_original_timestep = -1;
    double sweep_time_s = 0.0;
  };

  using GridKey = std::tuple<int, int, int>;

  struct Candidate
  {
    Eigen::Vector3d point_base = Eigen::Vector3d::Zero();
    std::string link_name = "none";
    int sample_index_in_link = -1;
    std::string source_type = "unknown";
    int source_collision_index = -1;
    std::string source_collision_name;
    double primitive_signed_distance_m = std::numeric_limits<double>::quiet_NaN();
    std::string refinement_status = "not_applicable";
    int refinement_fk_before = -1, refinement_fk_after = -1;
    double candidate_margin_m = std::numeric_limits<double>::quiet_NaN();
    double initial_support_distance_m = std::numeric_limits<double>::quiet_NaN();
    bool initial_support_unknown = false;
    bool initial_shell_certified = false;
    bool full_path_excluded = false;
    Eigen::Vector3d sample_center_base = Eigen::Vector3d::Zero();
    double raw_sample_radius_m = 0.0;
    double swept_radius_m = 0.0;
    double point_center_distance_m = 0.0;
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

  struct EvalTiming
  {
    double trajectory_convert_ms = 0.0;
    double body_fk_ms = 0.0;
    double swept_voxel_build_ms = 0.0;
    double confidence_query_ms = 0.0;
    double confidence_ready_ms = 0.0;
    double confidence_rpc_ms = 0.0;
    double candidate_filter_ms = 0.0;
    double sensor_fk_ms = 0.0;
    double visibility_scan_ms = 0.0;
    double cluster_ms = 0.0;
    double total_eval_ms = 0.0;
  };

  static double wallMs(
      const ros::WallTime& begin,
      const ros::WallTime& end = ros::WallTime::now())
  {
    return (end - begin).toSec() * 1000.0;
  }

  std::string timingFields() const
  {
    std::ostringstream oss;
    oss << " trajectory_convert_ms=" << current_eval_timing_.trajectory_convert_ms
        << " input_wait_ms=" << current_input_wait_ms_
        << " geometry_backend=" << geometry_backend_
        << " spatial_semantics=" << (whole_voxel_support_ ? "whole_voxel_sphere_enclosure" : "voxel_centers_discrete_knots")
        << " body_fk_ms=" << current_eval_timing_.body_fk_ms
        << " swept_voxel_build_ms=" << current_eval_timing_.swept_voxel_build_ms
        << " confidence_query_ms=" << current_eval_timing_.confidence_query_ms
        << " confidence_ready_ms=" << current_eval_timing_.confidence_ready_ms
        << " confidence_rpc_ms=" << current_eval_timing_.confidence_rpc_ms
        << " confidence_persistent=" << static_cast<int>(confidence_query_persistent_)
        << " candidate_filter_ms=" << current_eval_timing_.candidate_filter_ms
        << " sensor_fk_ms=" << current_eval_timing_.sensor_fk_ms
        << " visibility_scan_ms=" << current_eval_timing_.visibility_scan_ms
        << " cluster_ms=" << current_eval_timing_.cluster_ms
        << " total_eval_ms=" << current_eval_timing_.total_eval_ms
        << " predicted_periodic_skip_count=" << predicted_periodic_skip_count_
        << " predicted_periodic_refresh_count=" << predicted_periodic_refresh_count_;
    oss << " shell_recovery_count=" << shell_recovery_count_
        << " shell_recovery_evidence=" << shell_recovery_evidence_
        << " context_request_session=" << current_context_session_
        << " context_request_sequence=" << current_context_sequence_
        << " context_prefix_session=" << current_prefix_session_
        << " context_prefix_revision=" << current_prefix_revision_
        << " context_prefix_s=" << current_prefix_s_
        << " wire_header_seq=" << current_eval_wire_seq_;
    return oss.str();
  }

  void recoveryContextCallback(const std_msgs::StringConstPtr& msg) {
    if (!msg || msg->data.size()>256*1024) return;
    try {
      boost::property_tree::ptree t;
      std::istringstream in(msg->data); boost::property_tree::read_json(in,t);
      const auto stamp=t.get<unsigned long long>("trajectory_stamp_ns");
      if (!stamp) return;
      trajectory_msgs::JointTrajectory pending;
      bool recheck=false;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        auto old=recovery_contexts_.find(stamp);
        const bool conflict=old!=recovery_contexts_.end() && old->second!=msg->data;
        const bool first=old==recovery_contexts_.end();
        // Once an identity has conflicting contents it stays invalid, including
        // after a later duplicate of the first message. No certificate ratchet.
        if(conflict) old->second.clear();
        else recovery_contexts_[stamp]=msg->data;
        while (recovery_contexts_.size()>64) recovery_contexts_.erase(recovery_contexts_.begin());
        recheck=(first || conflict) && has_predicted_traj_ && !force_bootstrap_ &&
            latest_predicted_traj_.header.stamp.toNSec()==stamp;
        if(recheck)pending=latest_predicted_traj_;
      }
      if(recheck)evaluate(pending,"predicted","recovery_context_callback");
    } catch (...) {
      ROS_WARN_THROTTLE(1.,"[VBC_PROTOCOL] malformed context without usable audit identity");
    }
  }

  void recoveryQualityCallback(const std_msgs::StringConstPtr& msg) {
    if(!msg || msg->data.size()>2048)return;
    try {
      boost::property_tree::ptree t;std::istringstream in(msg->data);boost::property_tree::read_json(in,t);
      for(const auto* key:{"schema","trajectory_stamp_ns","request_session","request_sequence","required_residual_m"})
        if(t.count(key)!=1)return;
      const auto stamp=t.get<unsigned long long>("trajectory_stamp_ns");
      const double target=t.get<double>("required_residual_m");
      const double reply=t.get<double>("reply_allowance_s",.003);
      if(t.get<int>("schema")!=1 || !stamp || !std::isfinite(target) || target<=0. || target>1.)return;
      if(t.count("reply_allowance_s")>1 || !std::isfinite(reply) || reply<.001 || reply>.01)return;
      std::lock_guard<std::mutex> lock(mutex_);
      const auto context=recovery_contexts_.find(stamp);
      if(context==recovery_contexts_.end() || context->second.empty())return;
      boost::property_tree::ptree c;std::istringstream raw(context->second);boost::property_tree::read_json(raw,c);
      const auto now=ros::Time::now().toNSec();
      if(c.get<std::string>("request_session")!=t.get<std::string>("request_session") ||
          c.get<unsigned long long>("request_sequence")!=t.get<unsigned long long>("request_sequence") ||
          now<stamp || now>c.get<unsigned long long>("deadline_ns"))return;
      recovery_quality_[stamp]=std::max(recovery_quality_[stamp],target);
      recovery_reply_allowance_[stamp]=std::max(recovery_reply_allowance_[stamp],reply);
      while(recovery_quality_.size()>64)recovery_quality_.erase(recovery_quality_.begin());
      while(recovery_reply_allowance_.size()>64)recovery_reply_allowance_.erase(recovery_reply_allowance_.begin());
    } catch(...) {}
  }

  bool recoveryContext(const trajectory_msgs::JointTrajectory& traj,
                       bool* eligible, Eigen::VectorXd* acceleration) {
    *eligible=false;
    current_shared_budget_=false;
    current_geometric_restore_=false; current_margin_recovery_=false; geometric_restore_points_.clear();fixed_start_recovery_points_.clear();
    current_proof_resource_evidence_="none";
    current_restore_mode_epoch_.clear();current_restore_observation_token_.clear();
    current_context_state_="invalid";
    current_context_reason_="malformed_context";
    current_context_session_="none";current_context_sequence_=0;
    current_prefix_session_="none";current_prefix_revision_=0;current_prefix_s_=0.;
    auto reject=[this](const std::string& reason) {
      current_context_reason_=reason;return false;
    };
    std::string raw;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it=recovery_contexts_.find(traj.header.stamp.toNSec());
      if(it==recovery_contexts_.end()) {
        current_context_state_="waiting";return reject("missing_context");
      }
      raw=it->second;
    }
    if(raw.empty())return reject("conflicting_context");
    try {
      boost::property_tree::ptree t;
      std::istringstream in(raw); boost::property_tree::read_json(in,t);
      for (const auto& name:{"schema","trajectory_stamp_ns","request_session","request_sequence","frame","deadline_ns","kind","q","times_s","joint_names"})
        if(t.count(name)!=1)return reject("missing_or_duplicate_field");
      if(t.get<int>("schema")!=2 && t.get<int>("schema")!=3)return reject("unsupported_schema");
      current_context_session_=t.get<std::string>("request_session");
      current_context_sequence_=t.get<unsigned long long>("request_sequence");
      if(current_context_session_.empty() || current_context_session_.size()>80 || !current_context_sequence_ ||
          !std::all_of(current_context_session_.begin(),current_context_session_.end(),[](unsigned char c) {
            return (c>='a' && c<='z') || (c>='A' && c<='Z') || (c>='0' && c<='9') || c=='_' || c=='-';
          })) {current_context_session_="none";return reject("invalid_business_identity");}
      const auto now=ros::Time::now().toNSec();
      const auto deadline=t.get<unsigned long long>("deadline_ns");
      // Header.seq is a ROS topic counter, not the application request identity.
      // Bind the immutable stamp and exact content even when a preceding request
      // was cancelled or rejected before it reached this publisher.
      if(t.get<unsigned long long>("trajectory_stamp_ns")!=traj.header.stamp.toNSec())return reject("audit_stamp_mismatch");
      if(t.get<std::string>("frame")!=traj.header.frame_id)return reject("context_frame_mismatch");
      if(now>deadline || deadline<traj.header.stamp.toNSec() ||
          deadline-traj.header.stamp.toNSec()>500000000ULL)return reject("context_deadline_invalid");
      if(now<traj.header.stamp.toNSec()) {
        current_context_state_="waiting";
        return reject("context_clock_pending");
      }
      if (
          t.get_child("q").size()!=traj.points.size() || t.get_child("times_s").size()!=traj.points.size() ||
          t.get_child("joint_names").size()!=traj.joint_names.size() || traj.joint_names!=evaluator_.activeJointNames())return reject("context_shape_mismatch");
      std::size_t i=0;
      for(const auto& name:t.get_child("joint_names")) {
        if(!name.first.empty() || name.second.get_value<std::string>()!=traj.joint_names.at(i++))return reject("context_joint_mismatch");
      }
      i=0;
      for(const auto& row:t.get_child("q")) {
        if(!row.first.empty() || row.second.size()!=traj.joint_names.size())return reject("context_shape_mismatch");
        std::size_t j=0;
        for(const auto& v:row.second) {
          const double x=v.second.get_value<double>();
          if(!v.first.empty() || !std::isfinite(x) || std::abs(x-traj.points.at(i).positions.at(j++))>1e-12)return reject("context_q_mismatch");
        }
        ++i;
      }
      i=0;
      for(const auto& time:t.get_child("times_s")) {
        const double x=time.second.get_value<double>();
        if(!time.first.empty() || !std::isfinite(x) || std::abs(x-traj.points.at(i++).time_from_start.toSec())>1e-12)return reject("context_time_mismatch");
      }
      if(t.get<int>("schema")==3) {
        current_shared_budget_=true;
        if(t.count("v")!=1)return reject("context_velocity_shape");
        const auto& velocities=t.get_child("v");
        if(velocities.size()!=traj.points.size())return reject("context_velocity_shape");
        std::size_t k=0;
        for(const auto& row:velocities) {
          if(row.second.size()!=traj.points[k].velocities.size())return reject("context_velocity_shape");
          std::size_t j=0;
          for(const auto& item:row.second) {
            const double x=item.second.get_value<double>();
            if(!std::isfinite(x) || x!=traj.points[k].velocities[j++])return reject("context_velocity_mismatch");
          }
          ++k;
        }
      }
      if (t.count("planned_prefix")>1)return reject("duplicate_prefix_contract");
      if (t.count("planned_prefix")) {
        const auto& c=t.get_child("planned_prefix");
        if(c.size()!=7)return reject("invalid_prefix_contract");
        for(const auto& name:{"schema","session","revision","prefix_s","acceleration","brake_dt_s","brake_steps"})
          if(c.count(name)!=1)return reject("invalid_prefix_contract");
        const auto session=c.get<std::string>("session");
        const auto revision=c.get<unsigned long long>("revision");
        const double prefix=c.get<double>("prefix_s");
        if(c.get<int>("schema")!=1 || session.size()!=32 || !revision ||
            !std::all_of(session.begin(),session.end(),[](char x){return (x>='0'&&x<='9')||(x>='a'&&x<='f');}) ||
            !std::isfinite(prefix) || prefix<=0. || traj.points.empty() || prefix>traj.points.back().time_from_start.toSec() ||
            c.get<double>("brake_dt_s")!=t.get<double>("brake_dt_s") ||
            c.get<int>("brake_steps")!=t.get<int>("brake_steps") ||
            c.get_child("acceleration").size()!=traj.joint_names.size() ||
            t.get_child("acceleration").size()!=traj.joint_names.size())return reject("invalid_prefix_contract");
        auto a=t.get_child("acceleration").begin();
        for(const auto& v:c.get_child("acceleration")) {
          const double x=v.second.get_value<double>();
          if(!v.first.empty() || !std::isfinite(x) || x<=0. || x!=(a++)->second.get_value<double>())return reject("prefix_parameters_mismatch");
        }
        current_prefix_session_=session;current_prefix_revision_=revision;current_prefix_s_=prefix;
      }
      const auto kind=t.get<std::string>("kind");
      if(kind=="ordinary") {current_context_state_="validated";current_context_reason_="exact_content_bound";return true;}
      if(kind=="geometric_restore") {
        if(t.count("geometric_restore")!=1 || t.get<int>("schema")!=3 || current_prefix_session_=="none")
          return reject("invalid_geometric_restore_identity");
        const auto& r=t.get_child("geometric_restore");
        for(const auto& name:{"schema","raw_candidate_stamp_ns","query_stamp_ns","mode_epoch","observation_token","points"})
          if(r.count(name)!=1)return reject("invalid_geometric_restore_identity");
        const auto query=r.get<unsigned long long>("query_stamp_ns");
        const int restore_schema=r.get<int>("schema");
        const auto query_time=restore_schema==2 ? r.get<unsigned long long>("query_ros_ns") : query;
        const auto raw_stamp=r.get<unsigned long long>("raw_candidate_stamp_ns");
        const auto token=r.get<std::string>("observation_token");
        const bool margin_recovery=r.count("margin_recovery")==1;
        const bool fixed_start=r.count("fixed_start_recovery_points")==1;
        if(fixed_start && (restore_schema!=2 || margin_recovery))return reject("invalid_fixed_start_recovery");
        if(margin_recovery && (restore_schema!=2 || r.get<std::string>("margin_recovery")!="true"))
          return reject("invalid_margin_recovery_intent");
        if((restore_schema!=1 && restore_schema!=2) || r.size()!=std::size_t((restore_schema==2 ? 7 : 6)+(margin_recovery ? 1 : 0)+(fixed_start ? 1 : 0)) ||
            (restore_schema==2 && r.count("query_ros_ns")!=1) || !r.get<unsigned long long>("mode_epoch") ||
            !query || !query_time || query_time>raw_stamp || raw_stamp>traj.header.stamp.toNSec() || now<query_time || now-query_time>1000000000ULL ||
            token.compare(0,12,"care_obs_v1_")!=0 || token.size()>120 ||
            r.get_child("points").empty() || r.get_child("points").size()>32)
          return reject("invalid_geometric_restore_identity");
        for(const auto& row:r.get_child("points")) {
          if(!row.first.empty() || row.second.size()!=3)return reject("invalid_geometric_restore_points");
          Eigen::Vector3d p;int j=0;
          for(const auto& x:row.second) {if(!x.first.empty())return reject("invalid_geometric_restore_points");p[j++]=x.second.get_value<double>();}
          if(!p.allFinite() || std::any_of(geometric_restore_points_.begin(),geometric_restore_points_.end(),
              [&](const Eigen::Vector3d& old){return (old-p).lpNorm<Eigen::Infinity>()<=1e-5;}))
            return reject("invalid_geometric_restore_points");
          geometric_restore_points_.push_back(p);
        }
        if(fixed_start) {
          const auto& subset=r.get_child("fixed_start_recovery_points");
          if(subset.empty() || subset.size()>32)return reject("invalid_fixed_start_recovery");
          for(const auto& row:subset) {
            if(!row.first.empty() || row.second.size()!=3)return reject("invalid_fixed_start_recovery");
            Eigen::Vector3d p;int j=0;
            for(const auto& x:row.second) {if(!x.first.empty())return reject("invalid_fixed_start_recovery");p[j++]=x.second.get_value<double>();}
            if(!p.allFinite() || std::none_of(geometric_restore_points_.begin(),geometric_restore_points_.end(),
                [&](const Eigen::Vector3d& x){return (p-x).lpNorm<Eigen::Infinity>()<=1e-5;}) ||
                std::any_of(fixed_start_recovery_points_.begin(),fixed_start_recovery_points_.end(),
                [&](const Eigen::Vector3d& x){return (p-x).lpNorm<Eigen::Infinity>()<=1e-5;}))
              return reject("invalid_fixed_start_recovery");
            fixed_start_recovery_points_.push_back(p);
          }
        }
        current_geometric_restore_=true;
        current_margin_recovery_=margin_recovery;
        current_restore_mode_epoch_=r.get<std::string>("mode_epoch");
        current_restore_observation_token_=token;
      }
      if((kind!="repair_prefix_brake_hold" && kind!="geometric_restore") || t.count("acceleration")!=1 ||
          t.get<double>("brake_dt_s")!=.05 || t.get<int>("brake_steps")!=20 ||
          t.get_child("acceleration").size()!=traj.joint_names.size())return reject("invalid_recovery_parameters");
      acceleration->resize(traj.joint_names.size());i=0;
      for(const auto& a:t.get_child("acceleration"))(*acceleration)[i++]=a.second.get_value<double>();
      if(!acceleration->allFinite() || (acceleration->array()<=0.).any())return reject("invalid_recovery_parameters");
      *eligible=true;current_context_state_="validated";current_context_reason_="exact_content_bound";return true;
    } catch (...) { return reject("malformed_context"); }
  }

  void publishRecoveryProtocol(const trajectory_msgs::JointTrajectory& traj) {
    const auto stamp=traj.header.stamp.toNSec();
    const auto value=current_context_state_+":"+current_context_reason_;
    if(recovery_protocol_states_[stamp]==value)return;
    recovery_protocol_states_[stamp]=value;
    while(recovery_protocol_states_.size()>64)recovery_protocol_states_.erase(recovery_protocol_states_.begin());
    std::ostringstream out;
    out<<"trajectory_stamp_ns="<<stamp<<" state="<<current_context_state_
        <<" reason="<<current_context_reason_<<" request_session="<<current_context_session_
        <<" request_sequence="<<current_context_sequence_<<" wire_header_seq="<<traj.header.seq
        <<" path_counterexample_evidence="<<current_shell_path_counterexample_evidence_
        <<" proof_resource_evidence="<<current_proof_resource_evidence_;
    std_msgs::String msg;msg.data=out.str();recovery_protocol_pub_.publish(msg);
    if(current_context_state_!="validated")ROS_WARN_STREAM("[VBC_PROTOCOL] "<<msg.data);
  }

  bool certifyInitialShell(std::vector<Candidate>* candidates,
      const std::vector<Eigen::VectorXd>& q, const Eigen::VectorXd& acceleration,
      int* remaining_fk, const std::vector<Eigen::VectorXd>& curves,
      const trajectory_msgs::JointTrajectory& trajectory,
      care_confidence_map::VbcPrimitiveModel::ProofFkCache& proof_fk_cache) {
    current_shell_path_counterexample_=false;
    current_shell_path_counterexample_evidence_="none";
    current_proof_resource_evidence_="none";
    // Diagnostics preserve the original decisions, identity and FK ledger.
    const auto begin_wall=ros::WallTime::now();
    const auto begin_ros=ros::Time::now();
    const int fk_before=*remaining_fk;
    const int cache_begin_hits=proof_fk_cache.hits,cache_begin_misses=proof_fk_cache.misses,
        cover_begin_hits=proof_fk_cache.cover_hits;
    struct ProvedMotion {
      std::vector<Eigen::VectorXd> q, curves;
      Eigen::Vector3d point;
      double floor, slack;
      std::string stage;
      bool ordinary_unknown;
      int weak_interval;
    };
    std::vector<ProvedMotion> proved_motions;
    double residual_before=0., residual_after=0.;
    int reserve_refinement_fk=0, reserve_refinement_attempts=0;
    std::string residual_stage="none";
    Eigen::Vector3d residual_point=Eigen::Vector3d::Zero();
    double residual_floor=0.;
    bool residual_ordinary_unknown=false;
    int residual_interval=-1;
    ros::Time measured_stamp,snapshot_stamp;
    std::string stage="initial_support",proof_reason="none";
    Eigen::Vector3d proof_point=Eigen::Vector3d::Zero();
    double proof_floor=0., proof_occupancy=-1., proof_confidence=-1.;
    bool proof_shell_point=false;
    bool has_proof_point=false;
    const auto finish=[&](const char* reason) {
      // Only a fresh, complete mandatory work allowance may drive steering.
      // A periodic replay of an already-spent audit cannot mint another event.
      if(std::string(reason)=="continuous_proof_failed" &&
          proof_reason=="fk_budget_exhausted" && fk_before==256 &&
          *remaining_fk==0 && proof_fk_cache.misses==256 && has_proof_point &&
          current_geometric_restore_) {
        std::ostringstream evidence;
        evidence<<std::setprecision(17)<<"{\"schema\":1,\"kind\":\"mandatory_proof_resource_exhausted\","
            <<"\"trajectory_stamp_ns\":\""<<current_eval_stamp_ns_
            <<"\",\"mode_epoch\":\""<<current_restore_mode_epoch_
            <<"\",\"observation_token\":\""<<current_restore_observation_token_
            <<"\",\"stage\":\""<<stage<<"\",\"floor_m\":"<<proof_floor
            <<",\"point\":["<<proof_point.x()<<','<<proof_point.y()<<','<<proof_point.z()
            <<"],\"fk_before\":256,\"fk_spent\":256,\"fk_remaining\":0"
            <<",\"ordinary_unknown\":"<<((stage=="map_path_proof" && !proof_shell_point &&
                proof_occupancy>=0. && proof_occupancy<=.5 && proof_confidence>=0. &&
                proof_confidence<=frontier_confidence_threshold_ && proof_floor==swept_volume_margin_m_)?"true":"false")<<'}';
        current_proof_resource_evidence_=evidence.str();
      }
      current_shell_path_counterexample_=std::string(reason)=="continuous_proof_failed" &&
          proof_reason=="distance_within_margin" &&
          (stage=="map_path_proof" || stage=="support_path_proof");
      if(current_shell_path_counterexample_ && stage=="map_path_proof" &&
          !proof_shell_point && proof_occupancy>=0. && proof_occupancy<=.5 &&
          proof_confidence>=0. && proof_confidence<=frontier_confidence_threshold_ &&
          proof_floor==swept_volume_margin_m_) {
        std::ostringstream evidence;
        evidence<<std::setprecision(17)<<"{\"schema\":1,\"kind\":\"unknown_ordinary_path_floor\","
            <<"\"trajectory_stamp_ns\":\""<<current_eval_stamp_ns_<<"\",\"floor_m\":"<<proof_floor
            <<",\"point\":["<<proof_point.x()<<','<<proof_point.y()<<','<<proof_point.z()<<"]}";
        current_shell_path_counterexample_evidence_=evidence.str();
      }
      const auto end=ros::Time::now();
      std::ostringstream out;
      out<<"trajectory_stamp_ns="<<current_eval_stamp_ns_
          <<" request_session="<<current_context_session_
          <<" request_sequence="<<current_context_sequence_
          <<" result="<<reason<<" stage="<<stage<<" proof_reason="<<proof_reason
          <<" begin_ros_ns="<<begin_ros.toNSec()<<" end_ros_ns="<<end.toNSec()
          <<" measured_stamp_ns="<<measured_stamp.toNSec()
          <<" snapshot_stamp_ns="<<snapshot_stamp.toNSec()
          <<" measured_age_ms="<<(measured_stamp.isZero()?-1.:(end-measured_stamp).toSec()*1000.)
          <<" snapshot_age_ms="<<(snapshot_stamp.isZero()?-1.:(end-snapshot_stamp).toSec()*1000.)
          <<" wall_ms="<<wallMs(begin_wall)
          <<" fk_before="<<fk_before<<" fk_after="<<*remaining_fk
          <<" fk_cache_hits="<<proof_fk_cache.hits-cache_begin_hits<<" fk_cache_misses="<<proof_fk_cache.misses-cache_begin_misses
          <<" fk_cache_total_misses="<<proof_fk_cache.misses<<" cover_hits="<<proof_fk_cache.cover_hits-cover_begin_hits
          <<" residual_before_m="<<residual_before<<" residual_after_m="<<residual_after
          <<" reserve_refinement_fk="<<reserve_refinement_fk
          <<" reserve_refinement_attempts="<<reserve_refinement_attempts
          <<" residual_stage="<<residual_stage<<" residual_floor_m="<<residual_floor
          <<" residual_point="<<residual_point.x()<<','<<residual_point.y()<<','<<residual_point.z();
      if(has_proof_point)out<<" proof_point="<<proof_point.x()<<','<<proof_point.y()<<','<<proof_point.z()
          <<" proof_floor_m="<<proof_floor<<" proof_occupancy="<<proof_occupancy
          <<" proof_confidence="<<proof_confidence<<" proof_shell_point="<<proof_shell_point;
      std_msgs::String msg;msg.data=out.str();shell_diagnostic_pub_.publish(msg);
      ROS_INFO_STREAM("[VBC_SHELL_CERT] "<<msg.data);
      return std::string(reason)=="certified";
    };
    const auto prove=[&](const std::vector<Eigen::VectorXd>& motion,const Eigen::Vector3d& point,
        double floor,const std::vector<Eigen::VectorXd>& bend,double* slack) {
      proof_point=point;has_proof_point=true;proof_floor=floor;
      const bool ok=proveSupportOutsideContinuousTrajectory(evaluator_,motion,point,
          floor,remaining_fk,bend,slack,&proof_reason,&proof_fk_cache);
      if(ok)proved_motions.push_back({motion,bend,point,floor,*slack,stage,
          stage=="map_path_proof" && !proof_shell_point && proof_occupancy>=0. && proof_occupancy<=.5 &&
          proof_confidence>=0. && proof_confidence<=frontier_confidence_threshold_ && floor==swept_volume_margin_m_,
          proof_fk_cache.last_weak_interval});
      return ok;
    };
    std::vector<std::size_t> initial;
    for(std::size_t i=0;i<candidates->size();++i)
      if(candidates->at(i).initial_support_unknown)initial.push_back(i);
    if(initial.empty())return finish("no_initial_support");
    if(initial.size()>32)return finish("initial_support_limit");
    stage="measured_state";
    sensor_msgs::JointStateConstPtr measured;
    {std::lock_guard<std::mutex> lock(mutex_);measured=latest_measured_;}
    if(measured)measured_stamp=measured->header.stamp;
    const auto now=ros::Time::now();
    if(!measured || measured->header.stamp.isZero() ||
        measured->position.size()!=measured->name.size() || measured->velocity.size()!=measured->name.size() ||
        measured->name.size()!=q.front().size() || now<measured->header.stamp ||
        (now-measured->header.stamp).toSec()>.05)return finish("measured_state_invalid_or_stale");
    Eigen::VectorXd actual(q.front().size()), velocity(q.front().size());
    for(int j=0;j<actual.size();++j) {
      const auto& name=evaluator_.activeJointNames()[j];
      if(std::count(measured->name.begin(),measured->name.end(),name)!=1)return finish("measured_joint_identity");
      const auto k=std::distance(measured->name.begin(),std::find(measured->name.begin(),measured->name.end(),name));
      actual[j]=measured->position[k];velocity[j]=measured->velocity[k];
    }
    if(!actual.allFinite() || !velocity.allFinite() || (actual-q.front()).lpNorm<Eigen::Infinity>()>.03)
      return finish("measured_state_invalid_or_changed");
    stage="initial_geometry";
    std::vector<care_confidence_map::VbcPrimitiveFrame> actual_frames,end_frames;
    std::string error;
    if(!primitive_model_.computeTrajectory(evaluator_,{actual},&actual_frames,&error) ||
        !primitive_model_.computeTrajectory(evaluator_,{q.back()},&end_frames,&error))return finish("initial_fk_failed");
    std::map<CandidateKey,double> point_floors;
    std::vector<Eigen::Vector3d> reaudited_points;
    const auto is_fixed_start=[&](const Eigen::Vector3d& p) {
      return std::any_of(fixed_start_recovery_points_.begin(),fixed_start_recovery_points_.end(),
          [&](const Eigen::Vector3d& fixed){return (fixed-p).lpNorm<Eigen::Infinity>()<=1e-5;});
    };
    const auto is_reaudited=[&](const Eigen::Vector3d& p) {
      return std::any_of(reaudited_points.begin(),reaudited_points.end(),
          [&](const Eigen::Vector3d& restored){return (restored-p).lpNorm<Eigen::Infinity>()<=1e-5;});
    };
    for(const auto i:initial) {
      const auto& c=candidates->at(i);double d=1e9;
      for(const auto& b:actual_frames.front().primitives)d=std::min(d,supportDistance(b,c.point_base));
      double floor;bool recovering;
      if(!shell_recovery_.floor(c.point_base,std::min(d,c.initial_support_distance_m),
          swept_volume_margin_m_,shell_execution_margin_m_,&floor,&recovering,
          current_margin_recovery_ && std::any_of(geometric_restore_points_.begin(),geometric_restore_points_.end(),
              [&](const Eigen::Vector3d& p){return (p-c.point_base).lpNorm<Eigen::Infinity>()<=1e-5;})) ||
          (!recovering && !current_geometric_restore_))return finish("initial_floor_rejected");
      if(is_fixed_start(c.point_base) && c.initial_support_distance_m>=swept_volume_margin_m_-1e-12) {
        if(!fixed_start_rebase_reaudit_enabled_)return finish("fixed_start_recovery_already_at_planning_floor");
        if(d<swept_volume_margin_m_)return finish("fixed_start_reaudit_measured_below_planning_floor");
        // This is classification, not permission. All subsequent continuous
        // proofs must establish the planning floor on this exact executable.
        floor=swept_volume_margin_m_;
        reaudited_points.push_back(c.point_base);
      }
      point_floors[care_confidence_map::InitialShellRecovery::key(c.point_base)]=floor;
    }
    const Eigen::VectorXd reference_velocity=velocity;
    std::vector<Eigen::VectorXd> brake{actual};
    std::vector<Eigen::VectorXd> brake_curves;
    for(int step=0;step<20 && velocity.cwiseAbs().maxCoeff()>1e-6;++step) {
      Eigen::VectorXd next=velocity;
      for(int j=0;j<velocity.size();++j)next[j]=std::copysign(std::max(0.,std::abs(velocity[j])-acceleration[j]*.05),velocity[j]);
      brake.push_back(brake.back()+.025*(velocity+next));
      brake_curves.push_back(.05*(velocity-next)/8.);velocity=next;
    }
    if(velocity.cwiseAbs().maxCoeff()>1e-5)return finish("brake_steps_exhausted");
    stage="brake_geometry";
    // The measured stop is a distinct possible motion. Query its complete
    // geometry too, including occupied voxel volumes, instead of checking
    // only the points responsible for the nominal recovery request.
    std::vector<int> indices;std::vector<double> times;
    for(std::size_t k=0;k<brake.size();++k){indices.push_back(k);times.push_back(k*.05);}
    std::vector<care_confidence_map::VbcPrimitiveFrame> brake_frames;
    std::vector<int> brake_indices;std::vector<double> brake_times;
    if(!primitive_model_.computeContinuousTrajectory(evaluator_,brake,indices,times,&brake_frames,&brake_indices,&brake_times,&error,true,brake_curves))return finish("brake_sweep_failed");
    std::vector<Eigen::VectorXd> full_motion{actual};
    full_motion.insert(full_motion.end(),q.begin(),q.end());
    std::vector<Eigen::VectorXd> full_curves{Eigen::VectorXd::Zero(actual.size())};
    full_curves.insert(full_curves.end(),curves.begin(),curves.end());
    std::vector<int> full_indices;std::vector<double> full_times;
    for(std::size_t k=0;k<full_motion.size();++k){full_indices.push_back(k);full_times.push_back(k*.05);}
    std::vector<care_confidence_map::VbcPrimitiveFrame> full_frames;
    std::vector<int> full_out_indices;std::vector<double> full_out_times;
    if(!primitive_model_.computeContinuousTrajectory(evaluator_,full_motion,full_indices,full_times,
        &full_frames,&full_out_indices,&full_out_times,&error,true,full_curves))return finish("full_sweep_failed");
    brake_frames.insert(brake_frames.end(),full_frames.begin(),full_frames.end());
    brake_indices.resize(brake_frames.size());brake_times.resize(brake_frames.size());
    for(std::size_t k=0;k<brake_frames.size();++k){brake_indices[k]=k;brake_times[k]=k*.05;}
    const double occupied_floor=.5*std::sqrt(3.)*map_resolution_+shell_occupied_margin_m_-voxelSupportRadius();
    const double broad_margin=std::max(shell_execution_margin_m_,occupied_floor)+swept_volume_margin_m_-shell_execution_margin_m_;
    std::vector<SweepVoxel> brake_voxels;
    try { brake_voxels=care_confidence_map::buildPrimitiveSweptVoxels<SweepVoxel>(brake_frames,brake_indices,brake_times,
        {{map_x_min_,map_y_min_,map_z_min_},{map_x_max_,map_y_max_,map_z_max_},map_resolution_},broad_margin,voxelSupportRadius()); }
    catch (...) { return finish("brake_voxel_build_failed"); }
    // Declared restoration points need a current map classification even when
    // they lie outside this motion's broad-phase enclosure.
    for(const auto& p:geometric_restore_points_) {
      if(std::none_of(brake_voxels.begin(),brake_voxels.end(),[&](const SweepVoxel& v){
          return (v.point_base-p).lpNorm<Eigen::Infinity>()<=1e-5;})) {
        SweepVoxel v;v.point_base=p;brake_voxels.push_back(v);
      }
    }
    care_confidence_map::QueryConfidenceSnapshot snapshot;
    stage="map_snapshot";
    snapshot.request.horizon_s=0.;snapshot.request.minimum_confidence=frontier_confidence_threshold_;
    for(const auto& v:brake_voxels) {
      geometry_msgs::Point p;p.x=v.point_base.x();p.y=v.point_base.y();p.z=v.point_base.z();snapshot.request.points.push_back(p);
    }
    if(!shell_snapshot_client_.waitForExistence(ros::Duration(query_timeout_)) || !shell_snapshot_client_.call(snapshot))return finish("snapshot_rpc_failed");
    const auto& s=snapshot.response;
    snapshot_stamp=s.snapshot_stamp;
    const double snapshot_age=(ros::Time::now()-s.snapshot_stamp).toSec();
    if(!s.valid || s.clock_fault || s.frame!=base_frame_ || s.resolution!=map_resolution_ || s.instance.empty() ||
        s.snapshot_stamp.isZero() || snapshot_age<0. || snapshot_age>.05 ||
        s.inside_map.size()!=brake_voxels.size() || s.confidence_now.size()!=brake_voxels.size() ||
        s.occupancy.size()!=brake_voxels.size())return finish("snapshot_invalid_or_stale");
    double onset_budget=swept_volume_margin_m_-shell_execution_margin_m_;
    double certified_residual=std::numeric_limits<double>::infinity();
    for(std::size_t i=0;i<brake_voxels.size();++i) {
      if(!s.inside_map[i]) {
        if(std::any_of(geometric_restore_points_.begin(),geometric_restore_points_.end(),[&](const Eigen::Vector3d& p){
            return (p-brake_voxels[i].point_base).lpNorm<Eigen::Infinity>()<=1e-5;}))return finish("restore_point_outside_map");
        continue;
      }
      if(!std::isfinite(s.occupancy[i]) || !std::isfinite(s.confidence_now[i]) ||
          s.occupancy[i]<0. || s.occupancy[i]>1. || s.confidence_now[i]<0. || s.confidence_now[i]>1.)return finish("invalid_map_values");
      const bool occupied=s.occupancy[i]>.5;
      if(!occupied && s.confidence_now[i]>frontier_confidence_threshold_)continue;
      proof_occupancy=s.occupancy[i];proof_confidence=s.confidence_now[i];
      proof_shell_point=point_floors.count(care_confidence_map::InitialShellRecovery::key(brake_voxels[i].point_base));
      const auto point_floor=point_floors.find(care_confidence_map::InitialShellRecovery::key(brake_voxels[i].point_base));
      const bool shell_point=point_floor!=point_floors.end();
      // Re-audit also covers the measured stop at the planning floor for
      // ordinary UNKNOWNs. Only independently owned, still-recovering points
      // may retain their original execution-floor proof in a mixed request.
      const double required=occupied ? occupied_floor : !reaudited_points.empty()
          ? (shell_point ? point_floor->second : swept_volume_margin_m_) : shell_execution_margin_m_;
      double d=1e9;
      for(const auto& b:actual_frames.front().primitives)d=std::min(d,supportDistance(b,brake_voxels[i].point_base));
      onset_budget=std::min(onset_budget,d-required);
      double slack;
      stage="map_brake_proof";
      if(!prove(brake,brake_voxels[i].point_base,required,brake_curves,&slack))return finish("continuous_proof_failed");
      certified_residual=std::min(certified_residual,slack);
      const double path_floor=occupied ? occupied_floor : shell_point ? point_floor->second : swept_volume_margin_m_;
      auto guarded_motion=full_motion;auto guarded_curves=full_curves;
      if(!occupied && !shell_point) {
        const auto observed=std::find_if(candidates->begin(),candidates->end(),[&](const Candidate& c){
          return (c.point_base-brake_voxels[i].point_base).lpNorm<Eigen::Infinity>()<=1e-5 &&
              c.nominally_visible && c.margin_s>=min_margin_s_;
        });
        // Preserve ordinary VBC's observation-before-entry permission. The
        // geometric envelope protects the approach up to that observation;
        // the ordinary deadline and measured execution verifier still apply.
        if(observed!=candidates->end()) {
          const std::size_t count=std::min(guarded_motion.size(),std::size_t(observed->see_eval_timestep+2));
          guarded_motion.resize(count);guarded_curves.resize(count-1);
        }
      }
      stage="map_path_proof";
      if(!prove(guarded_motion,brake_voxels[i].point_base,path_floor,guarded_curves,&slack))return finish("continuous_proof_failed");
      certified_residual=std::min(certified_residual,slack);
      // This full measured-start path proof already establishes the optional
      // ordinary-margin exclusion. A visibility-truncated or owned-floor
      // proof cannot remove an observation candidate.
      if(!occupied && !shell_point && path_floor==swept_volume_margin_m_ &&
          guarded_motion.size()==full_motion.size())
        for(auto& c:*candidates)
          if((c.point_base-brake_voxels[i].point_base).lpNorm<Eigen::Infinity>()<=1e-5)
            c.full_path_excluded=true;
    }
    if(onset_budget<=0. || !std::isfinite(onset_budget))return finish("onset_clearance_exhausted");
    double actual_min=std::numeric_limits<double>::infinity();bool progress=false;
    std::ostringstream points;points<<'[';bool first=true;
    for(const auto i:initial) {
      const auto& c=candidates->at(i);
      double actual_d=1e9,end_d=1e9;
      for(const auto& b:actual_frames.front().primitives)actual_d=std::min(actual_d,supportDistance(b,c.point_base));
      for(const auto& b:end_frames.front().primitives)end_d=std::min(end_d,supportDistance(b,c.point_base));
      double floor;bool recovering;
      if(!shell_recovery_.floor(c.point_base,std::min(actual_d,c.initial_support_distance_m),
          swept_volume_margin_m_,shell_execution_margin_m_,&floor,&recovering,
          current_margin_recovery_ && std::any_of(geometric_restore_points_.begin(),geometric_restore_points_.end(),
              [&](const Eigen::Vector3d& p){return (p-c.point_base).lpNorm<Eigen::Infinity>()<=1e-5;})) ||
          (!recovering && !current_geometric_restore_))return finish("initial_floor_rejected");
      const bool reaudited=is_reaudited(c.point_base);
      floor=point_floors.at(care_confidence_map::InitialShellRecovery::key(c.point_base));
      recovering=floor<swept_volume_margin_m_;
      if(is_fixed_start(c.point_base) && !reaudited &&
          end_d<=std::max(actual_d,c.initial_support_distance_m)+care_confidence_map::InitialShellRecovery::tolerance_m())
        return finish("fixed_start_recovery_no_clearance_progress");
      const double onset_floor=reaudited ? swept_volume_margin_m_ : shell_execution_margin_m_;
      double slack;
      stage="support_path_proof";
      if(!prove(q,c.point_base,floor,curves,&slack))return finish("continuous_proof_failed");
      certified_residual=std::min(certified_residual,slack);
      stage="support_onset_proof";
      if(!prove({actual,q.front()},c.point_base,onset_floor,{},&slack))return finish("continuous_proof_failed");
      certified_residual=std::min(certified_residual,slack);
      stage="support_brake_proof";
      if(!prove(brake,c.point_base,onset_floor,brake_curves,&slack))return finish("continuous_proof_failed");
      certified_residual=std::min(certified_residual,slack);
      actual_min=std::min(actual_min,actual_d);
      progress=progress || end_d>c.initial_support_distance_m+care_confidence_map::InitialShellRecovery::tolerance_m() ||
          (c.nominally_visible && c.see_time_s>0.) ||
          (current_geometric_restore_ && (!recovering || current_margin_recovery_) &&
              (q.back()-q.front()).lpNorm<Eigen::Infinity>()>1e-4);
      if(!first)points<<',';first=false;
      points<<"{\"point\":["<<care_confidence_map::evidenceNumber(c.point_base.x())<<','
          <<care_confidence_map::evidenceNumber(c.point_base.y())<<','<<care_confidence_map::evidenceNumber(c.point_base.z())
          <<"],\"floor_m\":"<<care_confidence_map::evidenceNumber(floor)
          <<",\"actual_m\":"<<care_confidence_map::evidenceNumber(actual_d)
          <<",\"start_m\":"<<care_confidence_map::evidenceNumber(c.initial_support_distance_m)
          <<",\"end_m\":"<<care_confidence_map::evidenceNumber(end_d)<<'}';
    }
    if(!progress)return finish("no_recovery_progress");
    points<<']';
    // A Boolean exclusion proof may stop with an arbitrarily small positive
    // residual. Complete ALL mandatory geometry first, then spend only the
    // same request's remaining FK allowance on stronger proofs. The 2 mm
    // target is optional certificate quality, never a new execution floor or
    // permission to accept an unproved state change. Incremental strengthening
    // retains a complete interval cover even when the target is unattainable.
    residual_before=std::min(certified_residual,onset_budget);
    double requested_residual=0.,reply_allowance=.003;
    {std::lock_guard<std::mutex> lock(mutex_);const auto hint=recovery_quality_.find(current_eval_stamp_ns_);
      if(hint!=recovery_quality_.end())requested_residual=hint->second;
      const auto reply=recovery_reply_allowance_.find(current_eval_stamp_ns_);
      if(reply!=recovery_reply_allowance_.end())reply_allowance=reply->second;}
    // Quality hints schedule work only. The live commit formula remains the gate.
    const double reserve_target=std::min(onset_budget,requested_residual>0. ?
        requested_residual*1.25 : .002);
    std::stable_sort(proved_motions.begin(),proved_motions.end(),
        [](const ProvedMotion& a,const ProvedMotion& b){return a.slack<b.slack;});
    const int refinement_start=*remaining_fk;
    // Match the receiver's existing static-scene certificate-age policy. The
    // measured state and map have different freshness limits: mandatory proof
    // work may already have aged the former past 50 ms while the queried map
    // is still fresh. Do not disable all reserve refinement in that interval.
    // The live state, map limit, FK ledger and final residual check are unchanged.
    const char* continuation=std::getenv("CARE_CONTINUATION_RESEARCH");
    const char* static_repair=std::getenv("CARE_REPAIR_UNKNOWN_SOFT");
    const char* refinement_policy=std::getenv("CARE_CERTIFICATE_REFINEMENT_POLICY");
    const std::string refinement=refinement_policy ? refinement_policy : "inherit";
    const bool aligned=refinement=="aligned" || (refinement=="inherit" &&
        continuation && std::string(continuation)=="true");
    const double measured_age_limit=(aligned && static_repair &&
        std::string(static_repair)=="true") ? .25 : .05;
    const auto may_refine=[&]() {
      const auto now=ros::Time::now();
      return now>=measured_stamp && now>=snapshot_stamp &&
          (now-measured_stamp).toSec()<measured_age_limit-reply_allowance &&
          (now-snapshot_stamp).toSec()<.05-reply_allowance;
    };
    for(auto& proof:proved_motions) {
      if(*remaining_fk<=0 || proof.slack>=reserve_target)break;
      // Do not spend optional work after the evidence's original validity
      // interval. The downstream freshness/identity checks remain authoritative.
      if(!may_refine())break;
      double stronger_slack=0.;std::string refinement_reason;
      ++reserve_refinement_attempts;
      if(proveSupportOutsideContinuousTrajectory(evaluator_,proof.q,
          proof.point,proof.floor,remaining_fk,proof.curves,
          &stronger_slack,&refinement_reason,&proof_fk_cache,reserve_target,may_refine))
        {proof.slack=std::max(proof.slack,stronger_slack);proof.weak_interval=proof_fk_cache.last_weak_interval;}
    }
    reserve_refinement_fk=refinement_start-*remaining_fk;
    certified_residual=std::numeric_limits<double>::infinity();
    for(const auto& proof:proved_motions) {
      if(proof.slack<certified_residual) {
        certified_residual=proof.slack;residual_stage=proof.stage;
        residual_point=proof.point;residual_floor=proof.floor;
        residual_ordinary_unknown=proof.ordinary_unknown;residual_interval=proof.weak_interval;
      }
    }
    residual_after=std::min(certified_residual,onset_budget);
    if(onset_budget<=certified_residual){residual_stage="onset_budget";residual_ordinary_unknown=false;residual_interval=-1;}
    std::ostringstream evidence;
    evidence<<"{\"schema\":2,\"trajectory_stamp_ns\":\""<<current_eval_stamp_ns_
        <<"\",\"measured_stamp_ns\":\""<<measured->header.stamp.toNSec()
        <<"\",\"execution_floor_m\":"<<care_confidence_map::evidenceNumber(shell_execution_margin_m_)
        <<",\"actual_min_distance_m\":"<<care_confidence_map::evidenceNumber(actual_min)
        <<",\"onset_clearance_budget_m\":"<<care_confidence_map::evidenceNumber(onset_budget)
        <<",\"map_snapshot_stamp_ns\":\""<<s.snapshot_stamp.toNSec()<<"\""
        <<",\"certified_residual_m\":"<<care_confidence_map::evidenceNumber(std::min(certified_residual,onset_budget))
        <<",\"residual_stage\":\""<<residual_stage<<"\",\"residual_floor_m\":"<<care_confidence_map::evidenceNumber(residual_floor)
        <<",\"residual_point\":["<<care_confidence_map::evidenceNumber(residual_point.x())<<','
        <<care_confidence_map::evidenceNumber(residual_point.y())<<','<<care_confidence_map::evidenceNumber(residual_point.z())<<']'
        <<",\"residual_ordinary_unknown\":"<<(residual_ordinary_unknown?"true":"false")
        <<",\"residual_interval_index\":"<<residual_interval
        <<",\"requested_residual_m\":"<<care_confidence_map::evidenceNumber(requested_residual)
        <<",\"quality_target_m\":"<<care_confidence_map::evidenceNumber(reserve_target)
        <<",\"reply_allowance_s\":"<<care_confidence_map::evidenceNumber(reply_allowance)
        <<",\"refinement_measured_age_limit_s\":"<<care_confidence_map::evidenceNumber(measured_age_limit)
        <<",\"certificate_end_ros_ns\":\""<<ros::Time::now().toNSec()<<"\""
        <<",\"motion_horizon_s\":"<<care_confidence_map::evidenceNumber(trajectory.points.back().time_from_start.toSec())
        <<",\"measured_stop_contract\":\"ramp_stop_and_linear_start_bridge_v1\""
        <<",\"measured_stop_horizon_s\":"<<care_confidence_map::evidenceNumber(brake_curves.size()*.05)
        <<",\"measured_velocity\":[";
    for(int j=0;j<actual.size();++j){if(j)evidence<<',';evidence<<care_confidence_map::evidenceNumber(reference_velocity[j]);}
    evidence<<"],\"initial_q\":[";
    for(int j=0;j<actual.size();++j){if(j)evidence<<',';evidence<<care_confidence_map::evidenceNumber(q.front()[j]);}
    evidence<<"],\"initial_velocity\":[";
    for(std::size_t j=0;j<trajectory.points.front().velocities.size();++j){if(j)evidence<<',';evidence<<care_confidence_map::evidenceNumber(trajectory.points.front().velocities[j]);}
    evidence<<"],\"remaining_fk\":"<<*remaining_fk<<",\"map_instance\":\""<<s.instance
        <<"\",\"map_epoch\":"<<s.epoch<<",\"map_revision\":"<<s.revision
        <<",\"points\":"<<points.str()<<",\"measured_q\":[";
    for(int j=0;j<actual.size();++j){if(j)evidence<<',';evidence<<care_confidence_map::evidenceNumber(actual[j]);}
    evidence<<"],\"joint_motion_radii\":[";
    for(int j=0;j<actual.size();++j) {
      if(j)evidence<<',';
      const auto bounds=primitive_model_.displacementBounds(evaluator_.activeJointNames(),Eigen::VectorXd::Unit(actual.size(),j));
      evidence<<care_confidence_map::evidenceNumber(*std::max_element(bounds.begin(),bounds.end()));
    }
    evidence<<"]}";
    if(current_geometric_restore_) {
      std::string value=evidence.str();value.pop_back();std::ostringstream extra;
      extra<<value<<",\"contract\":\"geometric_restore\",\"restoration_points\":[";
      for(std::size_t i=0;i<geometric_restore_points_.size();++i) {
        if(i)extra<<',';const auto& p=geometric_restore_points_[i];
        extra<<'['<<care_confidence_map::evidenceNumber(p.x())<<','<<care_confidence_map::evidenceNumber(p.y())<<','<<care_confidence_map::evidenceNumber(p.z())<<']';
      }
      extra<<']';
      if(!fixed_start_recovery_points_.empty()) {
        extra<<",\"fixed_start_recovery_points\":[";
        for(std::size_t i=0;i<fixed_start_recovery_points_.size();++i) {
          if(i)extra<<',';const auto& p=fixed_start_recovery_points_[i];
          extra<<'['<<care_confidence_map::evidenceNumber(p.x())<<','<<care_confidence_map::evidenceNumber(p.y())<<','<<care_confidence_map::evidenceNumber(p.z())<<']';
        }
        extra<<']';
      }
      if(!reaudited_points.empty()) {
        extra<<",\"fixed_start_reaudit_contract\":\"planning_floor_path_onset_stop_v1\""
            <<",\"fixed_start_reaudit_floor_m\":"<<care_confidence_map::evidenceNumber(swept_volume_margin_m_)
            <<",\"fixed_start_reaudit_points\":[";
        for(std::size_t i=0;i<reaudited_points.size();++i) {
          if(i)extra<<',';const auto& p=reaudited_points[i];
          extra<<'['<<care_confidence_map::evidenceNumber(p.x())<<','<<care_confidence_map::evidenceNumber(p.y())<<','<<care_confidence_map::evidenceNumber(p.z())<<']';
        }
        extra<<']';
      }
      extra<<'}';evidence.str("");evidence.clear();evidence<<extra.str();
    }
    shell_recovery_evidence_=evidence.str();shell_recovery_count_=initial.size();
    for(const auto i:initial)candidates->at(i).initial_shell_certified=true;
    stage="complete";
    return finish("certified");
  }

  void loadParams()
  {
    pnh_.param("trajectory_vbc/preserve_periodic_eval", preserve_periodic_eval_, false);
    pnh_.param<std::string>(
        "trajectory_vbc/robot_urdf_file", robot_urdf_file_, "");
    pnh_.param<std::string>(
        "trajectory_vbc/body_samples_file", body_samples_file_, "");
    pnh_.param<std::string>("trajectory_vbc/geometry_backend", geometry_backend_, "samples");
    pnh_.param<std::string>("trajectory_vbc/primitive_urdf_file", primitive_urdf_file_, "");
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
        "trajectory_vbc/active_set_bundle_topic",
        active_set_bundle_topic_,
        "/care_planner/trajectory_risk/vbc_active_set_bundle");
    pnh_.param<std::string>(
        "trajectory_vbc/force_bootstrap_topic",
        force_bootstrap_topic_,
        "/care_planner/trajectory_risk/force_bootstrap");
    pnh_.param<std::string>(
        "trajectory_vbc/rediscovery_trajectory_topic",
        rediscovery_trajectory_topic_, "");
    pnh_.param(
        "trajectory_vbc/rediscovery_trajectory_timeout",
        rediscovery_trajectory_timeout_, 1.0);

    const char* whole=std::getenv("CARE_VBC_WHOLE_VOXEL");
    whole_voxel_support_=whole && std::string(whole)=="true";
    pnh_.param("trajectory_vbc/enabled", enabled_, true);
    pnh_.param("trajectory_vbc/eval_rate", eval_rate_, 20.0);
    pnh_.param(
        "trajectory_vbc/event_driven_eval",
        event_driven_eval_, false);
    pnh_.param(
        "trajectory_vbc/predicted_periodic_refresh_rate",
        predicted_periodic_refresh_rate_, 5.0);
    pnh_.param("trajectory_vbc/max_eval_timesteps", max_eval_timesteps_, 50);
    pnh_.param("trajectory_vbc/query_timeout", query_timeout_, 0.10);
    pnh_.param("trajectory_vbc/confidence_query_persistent", confidence_query_persistent_, false);
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
        "trajectory_vbc/swept_volume_margin_m",
        swept_volume_margin_m_, 0.025);
    pnh_.param(
        "trajectory_vbc/continuous_motion_bound_enabled",
        continuous_motion_bound_enabled_, true);
    pnh_.param(
        "trajectory_vbc/map_resolution",
        map_resolution_, 0.05);
    pnh_.param("trajectory_vbc/map_x_min", map_x_min_, -0.95);
    pnh_.param("trajectory_vbc/map_x_max", map_x_max_, 0.95);
    pnh_.param("trajectory_vbc/map_y_min", map_y_min_, -0.95);
    pnh_.param("trajectory_vbc/map_y_max", map_y_max_, 0.95);
    pnh_.param("trajectory_vbc/map_z_min", map_z_min_, 0.0);
    pnh_.param("trajectory_vbc/map_z_max", map_z_max_, 1.15);
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
    if (geometry_backend_ == "primitive" && input_joint_index.size()!=traj.joint_names.size()) {
      if(error_msg) *error_msg="Duplicate continuous trajectory joint names";
      return false;
    }

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

    std::vector<int> selected;
    if (geometry_backend_ == "primitive") {
      // Dropping a knot changes the piecewise-linear path, including its brake
      // and hold tail. Capacity failures must reject, never simplify the path.
      if (traj.points.size() > 4096) {
        if (error_msg) *error_msg="Continuous trajectory capacity exceeded";
        return false;
      }
      for (std::size_t i=0;i<traj.points.size();++i) selected.push_back(static_cast<int>(i));
    } else selected = makeDownsampleIndices(static_cast<int>(traj.points.size()));
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
      double t = (geometry_backend_ == "primitive" || has_timing)
          ? pt.time_from_start.toSec()
          : static_cast<double>(original_index) * fallback_dt_;
      if (geometry_backend_ == "primitive" && (!q.allFinite() || !std::isfinite(t) || t<0 ||
          (eval_times_s->empty() && t!=0.0) ||
          (!eval_times_s->empty() && t<=eval_times_s->back()))) {
        if(error_msg) *error_msg="Invalid continuous knot positions/timing";
        return false;
      }
      if (!eval_times_s->empty() && t < eval_times_s->back())
        t = eval_times_s->back();
      q_traj->push_back(q);
      original_indices->push_back(original_index);
      eval_times_s->push_back(t);
    }
    return !q_traj->empty();
  }

  bool queryConfidencePoints(
      const std::vector<SweepVoxel>& voxels,
      care_confidence_map::QueryConfidence* srv)
  {
    srv->request.points.clear();
    srv->request.points.reserve(voxels.size());
    for (const auto& voxel : voxels)
    {
      geometry_msgs::Point p;
      p.x = voxel.point_base.x();
      p.y = voxel.point_base.y();
      p.z = voxel.point_base.z();
      srv->request.points.push_back(p);
    }

    const auto ready_begin = ros::WallTime::now();
    bool ready = true;
    if (!confidence_query_persistent_)
      ready = confidence_query_client_.waitForExistence(ros::Duration(query_timeout_));
    else if (!confidence_query_client_.isValid())
    {
      // Reconnect only on a subsequent evaluation; never retry a failed query
      // or reuse its response as a certificate. Keep the existing ready budget.
      confidence_query_client_.shutdown();
      auto probe = nh_.serviceClient<care_confidence_map::QueryConfidence>(confidence_query_service_);
      ready = probe.waitForExistence(ros::Duration(query_timeout_));
      if (ready)
      {
        confidence_query_client_ = nh_.serviceClient<care_confidence_map::QueryConfidence>(
            confidence_query_service_, true);
        ready = confidence_query_client_.isValid();
      }
    }
    current_eval_timing_.confidence_ready_ms = wallMs(ready_begin);
    if (!ready)
      return false;
    const auto rpc_begin = ros::WallTime::now();
    const bool called = confidence_query_client_.call(*srv);
    current_eval_timing_.confidence_rpc_ms = wallMs(rpc_begin);
    if (!called)
    {
      if (confidence_query_persistent_) confidence_query_client_.shutdown();
      return false;
    }
    const std::size_t n = srv->request.points.size();
    return srv->response.confidence.size() == n &&
           srv->response.current_visibility.size() == n &&
           srv->response.inside_map.size() == n;
  }

  int clampGridIndex(int value, int max_index) const
  {
    return std::max(0, std::min(max_index, value));
  }

  std::vector<SweepVoxel> buildSweptVolumeVoxels(
      const care_confidence_map::TrajectorySampleResult& sample_result,
      const std::vector<int>& original_indices,
      const std::vector<double>& eval_times_s) const
  {
    const int nx = static_cast<int>(
        std::floor((map_x_max_ - map_x_min_) / map_resolution_)) + 1;
    const int ny = static_cast<int>(
        std::floor((map_y_max_ - map_y_min_) / map_resolution_)) + 1;
    const int nz = static_cast<int>(
        std::floor((map_z_max_ - map_z_min_) / map_resolution_)) + 1;

    std::map<GridKey, SweepVoxel> earliest;
    for (const auto& frame : sample_result.frames)
    {
      const int k = frame.timestep_index;
      if (k < 0 || k >= static_cast<int>(eval_times_s.size()) ||
          k >= static_cast<int>(original_indices.size()))
        continue;

      for (const auto& sample : frame.samples)
      {
        if (isIgnoredRiskLink(sample.link_name)) continue;

        const double radius =
            std::max(0.0, sample.radius + swept_volume_margin_m_);
        if (radius <= 0.0) continue;

        const int ix0 = clampGridIndex(
            static_cast<int>(std::ceil(
                (sample.center_base.x() - radius - map_x_min_) /
                map_resolution_)),
            nx - 1);
        const int ix1 = clampGridIndex(
            static_cast<int>(std::floor(
                (sample.center_base.x() + radius - map_x_min_) /
                map_resolution_)),
            nx - 1);
        const int iy0 = clampGridIndex(
            static_cast<int>(std::ceil(
                (sample.center_base.y() - radius - map_y_min_) /
                map_resolution_)),
            ny - 1);
        const int iy1 = clampGridIndex(
            static_cast<int>(std::floor(
                (sample.center_base.y() + radius - map_y_min_) /
                map_resolution_)),
            ny - 1);
        const int iz0 = clampGridIndex(
            static_cast<int>(std::ceil(
                (sample.center_base.z() - radius - map_z_min_) /
                map_resolution_)),
            nz - 1);
        const int iz1 = clampGridIndex(
            static_cast<int>(std::floor(
                (sample.center_base.z() + radius - map_z_min_) /
                map_resolution_)),
            nz - 1);

        if (ix0 > ix1 || iy0 > iy1 || iz0 > iz1) continue;

        const double radius_sq = radius * radius + 1e-12;
        for (int ix = ix0; ix <= ix1; ++ix)
        {
          const double x = map_x_min_ + ix * map_resolution_;
          for (int iy = iy0; iy <= iy1; ++iy)
          {
            const double y = map_y_min_ + iy * map_resolution_;
            for (int iz = iz0; iz <= iz1; ++iz)
            {
              const double z = map_z_min_ + iz * map_resolution_;
              const Eigen::Vector3d point(x, y, z);
              if ((point - sample.center_base).squaredNorm() > radius_sq)
                continue;

              const GridKey key(ix, iy, iz);
              const double sweep_time =
                  eval_times_s[static_cast<std::size_t>(k)];
              const auto it = earliest.find(key);
              if (it != earliest.end() &&
                  it->second.sweep_time_s <= sweep_time + 1e-12)
                continue;

              SweepVoxel voxel;
              voxel.point_base = point;
              voxel.link_name = sample.link_name;
              voxel.sample_index_in_link = sample.sample_index_in_link;
              voxel.source_type = sample.source_type;
              voxel.source_collision_index = sample.source_collision_index;
              voxel.sample_center_base = sample.center_base;
              voxel.raw_sample_radius_m = std::max(0.0, sample.radius);
              voxel.swept_radius_m = radius;
              voxel.point_center_distance_m =
                  (point - sample.center_base).norm();
              voxel.sweep_eval_timestep = k;
              voxel.sweep_original_timestep =
                  original_indices[static_cast<std::size_t>(k)];
              voxel.sweep_time_s = sweep_time;
              earliest[key] = voxel;
            }
          }
        }
      }
    }

    std::vector<SweepVoxel> voxels;
    voxels.reserve(earliest.size());
    for (const auto& item : earliest) voxels.push_back(item.second);
    return voxels;
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
    if(!whole_voxel_support_) {
      for(const auto& pose:poses)if(pointVisibleFromSensor(point_base,pose))return true;
      return false;
    }
    Eigen::Vector3d center=point_base,half=Eigen::Vector3d::Constant(map_resolution_/2);
    if(clip_analytic_support_) {
      const auto residual=analytic_support_.residual(point_base,map_resolution_,analytic_support_.supportUpper());
      if(residual.empty)return false; // Covered geometry is not an observation.
      center=(residual.minimum+residual.maximum)/2.;half=(residual.maximum-residual.minimum)/2.;
    }
    for(const auto& pose:poses)if(care_confidence_map::supportAabbInsideFrustum(center,half,pose.rotation_base,pose.translation_base,
        forward_dir_,right_dir_,up_dir_,tof_min_range_,tof_max_range_,tan_half_h_fov_,tan_half_v_fov_))return true;
    return false;
  }

  bool isViolation(const Candidate& c) const
  {
    return !c.initial_shell_certified && (!c.nominally_visible || c.margin_s < min_margin_s_);
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

  // C5.26 coherent blocker handoff.  Geometry and sweep metadata must travel
  // in one ROS message; publish order across two independent topics is not an
  // atomic transaction at the subscriber.
  //
  // data = [generation_seq, sweep_time_s, point_count, x0,y0,z0, ...]
  // An empty/Safe bundle uses point_count=0 and sweep_time_s=NaN.
  void publishActiveSetBundle(
      const std::vector<const Candidate*>& points,
      double sweep_time_s)
  {
    std::vector<const Candidate*> ordered = points;
    std::sort(
        ordered.begin(), ordered.end(),
        [this](const Candidate* a, const Candidate* b) {
          return candidateKey(a->point_base) < candidateKey(b->point_base);
        });

    ++active_set_bundle_seq_;
    std_msgs::Float64MultiArray msg;
    // Optional diagnostic identity, leaving the legacy numeric layout intact.
    msg.layout.dim.resize(1);
    msg.layout.dim[0].label = "care_vbc_v1_" + std::to_string(current_eval_stamp_ns_) +
        "_" + std::to_string(active_set_bundle_seq_);
    msg.data.reserve(3 + ordered.size() * 3);
    msg.data.push_back(static_cast<double>(active_set_bundle_seq_));
    msg.data.push_back(sweep_time_s);
    msg.data.push_back(static_cast<double>(ordered.size()));
    for (const Candidate* c : ordered)
    {
      if (!c) continue;
      msg.data.push_back(c->point_base.x());
      msg.data.push_back(c->point_base.y());
      msg.data.push_back(c->point_base.z());
    }
    active_set_bundle_pub_.publish(msg);
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
    publishActiveSetBundle({}, std::numeric_limits<double>::quiet_NaN());
    has_selected_key_ = false;

    std::ostringstream oss;
    oss << "vbc success=1 has_violation=0"
        << " spatial_support=" << (geometry_backend_ == "primitive" ? "urdf_primitive_volume" : "body_sphere_volume")
        << " continuous_linear_sweep=" << (geometry_backend_ == "primitive")
        << " include_measured_body=" << include_measured_body_
        << " swept_voxel_query_count=" << last_swept_voxel_query_count_
        << " sweep_refinement_checks=" << last_sweep_refinement_checks_
        << " sweep_refinement_excluded=" << last_sweep_refinement_excluded_
        << " sweep_refinement_fk=" << last_sweep_refinement_fk_
        << " voxel_support_radius_m=" << voxelSupportRadius() << " swept_volume_margin_m=" << swept_volume_margin_m_
        << " continuous_motion_bound_enabled=" << continuous_motion_bound_enabled_
        << " reason=" << reason
        << " trajectory_source=" << trajectory_source
        << " trajectory_stamp_ns=" << current_eval_stamp_ns_
        << " evaluation_trigger=" << current_eval_trigger_
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
        << " min_required_margin_s=" << min_margin_s_
        << " vbc_evidence=" << care_confidence_map::vbcEvidence(
             current_eval_stamp_ns_, active_set_bundle_seq_, min_margin_s_,
             map_resolution_, std::vector<Candidate>{}, std::vector<TemporalLayer>{}, geometry_backend_,
             continuous_motion_bound_enabled_,voxelSupportRadius())
        << timingFields();
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

    // Publish the active-set identity before its sweep timestamp.  The online
    // waypoint node intentionally clears its cached sweep whenever the active
    // set changes.  Publishing sweep first therefore creates a transport race:
    // the following active-set callback can erase the fresh sweep and delay or
    // suppress obligation generation.  active_set -> sweep makes one coherent
    // rediscovery transaction even on the first observation of a new blocker.
    std::vector<const Candidate*> active_points;
    active_points.reserve(active_layer.member_indices.size());
    for (const int idx : active_layer.member_indices)
      active_points.push_back(&candidates[static_cast<std::size_t>(idx)]);
    publishActiveSet(active_points);
    publishActiveSetBundle(active_points, active_layer.min_sweep_time_s);

    // Legacy split-topic metadata remains for diagnostics/backward-compatible
    // consumers. C5.26 blocker acquisition consumes the coherent bundle above.
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
        << " spatial_support=" << (geometry_backend_ == "primitive" ? "urdf_primitive_volume" : "body_sphere_volume")
        << " continuous_linear_sweep=" << (geometry_backend_ == "primitive")
        << " include_measured_body=" << include_measured_body_
        << " swept_voxel_query_count=" << last_swept_voxel_query_count_
        << " sweep_refinement_checks=" << last_sweep_refinement_checks_
        << " sweep_refinement_excluded=" << last_sweep_refinement_excluded_
        << " sweep_refinement_fk=" << last_sweep_refinement_fk_
        << " voxel_support_radius_m=" << voxelSupportRadius() << " swept_volume_margin_m=" << swept_volume_margin_m_
        << " continuous_motion_bound_enabled=" << continuous_motion_bound_enabled_
        << " reason=selected_temporal_cluster"
        << " start_support_unknown_count=" << std::count_if(candidates.begin(), candidates.end(),
             [](const Candidate& c) { return c.initial_support_unknown; })
        << " trajectory_source=" << trajectory_source
        << " trajectory_stamp_ns=" << current_eval_stamp_ns_
        << " evaluation_trigger=" << current_eval_trigger_
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
        << " source_type=" << representative.source_type
        << " source_collision_index=" << representative.source_collision_index
        << " primitive_signed_distance_m=" << representative.primitive_signed_distance_m
        << " raw_sample_center_xyz=["
        << representative.sample_center_base.x() << ","
        << representative.sample_center_base.y() << ","
        << representative.sample_center_base.z() << "]"
        << " raw_sample_radius_m=" << representative.raw_sample_radius_m
        << " swept_radius_m=" << representative.swept_radius_m
        << " blocker_center_distance_m="
        << representative.point_center_distance_m
        << " inside_raw_body_sphere="
        << (geometry_backend_ == "primitive" ? -1 : static_cast<int>(
             representative.point_center_distance_m <=
             representative.raw_sample_radius_m + 1e-9))
        << " margin_shell_only="
        << (geometry_backend_ == "primitive" ? static_cast<int>(
             representative.primitive_signed_distance_m > 1e-9) : static_cast<int>(
             representative.point_center_distance_m >
             representative.raw_sample_radius_m + 1e-9 &&
             representative.point_center_distance_m <=
             representative.swept_radius_m + 1e-9))
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
    oss << timingFields()
        << " vbc_evidence=" << care_confidence_map::vbcEvidence(
             current_eval_stamp_ns_, active_set_bundle_seq_, min_margin_s_,
             map_resolution_, candidates, layers, geometry_backend_, continuous_motion_bound_enabled_,voxelSupportRadius());

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

  static unsigned long long stampToNs(const ros::Time& stamp)
  {
    return static_cast<unsigned long long>(stamp.sec) * 1000000000ULL +
           static_cast<unsigned long long>(stamp.nsec);
  }

  void evaluate(
      const trajectory_msgs::JointTrajectory& traj,
      const std::string& trajectory_source,
      const std::string& evaluation_trigger = "timer")
  {
    const ros::WallTime eval_begin = ros::WallTime::now();
    current_eval_timing_ = EvalTiming();
    last_sweep_refinement_checks_ = last_sweep_refinement_excluded_ = 0;
    last_sweep_refinement_fk_ = 0;
    shell_recovery_count_=0;shell_recovery_evidence_="none";current_shared_budget_=false;
    current_geometric_restore_=false;current_margin_recovery_=false;geometric_restore_points_.clear();fixed_start_recovery_points_.clear();
    current_context_session_="none";current_context_sequence_=0;
    current_prefix_session_="none";current_prefix_revision_=0;current_prefix_s_=0.;
    current_eval_wire_seq_=traj.header.seq;
    // Diagnostic only: arrival callback -> evaluation start, independently of
    // trajectory ROS stamp and geometry/service compute time. No scheduling change.
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const ros::WallTime received = trajectory_source == "predicted" ? predicted_received_wall_ :
          trajectory_source == "scp_rediscovery" ? rediscovery_received_wall_ : trajectory_received_wall_;
      current_input_wait_ms_ = received.isZero() ? 0.0 : wallMs(received, eval_begin);
    }

    ros::Time identity_stamp = traj.header.stamp;
    if (identity_stamp.isZero()) identity_stamp = ros::Time::now();
    current_eval_stamp_ns_ = stampToNs(identity_stamp);
    current_eval_trigger_ = evaluation_trigger;

    auto finalize_total = [this, &eval_begin]() {
      current_eval_timing_.total_eval_ms = wallMs(eval_begin);
    };

    if (!enabled_)
    {
      finalize_total();
      publishNoTargetSummary("disabled", trajectory_source, 0);
      return;
    }

    std::vector<Eigen::VectorXd> q_traj;
    std::vector<Eigen::VectorXd> q_curves;
    std::vector<int> original_indices;
    std::vector<double> eval_times_s;
    std::string error_msg;
    const ros::WallTime convert_begin = ros::WallTime::now();
    if (!convertTrajectory(
            traj, &q_traj, &original_indices, &eval_times_s, &error_msg))
    {
      current_eval_timing_.trajectory_convert_ms = wallMs(convert_begin);
      finalize_total();
      ROS_WARN_STREAM_THROTTLE(2.0, "[trajectory_vbc_temporal] " << error_msg);
      return;
    }
    current_eval_timing_.trajectory_convert_ms = wallMs(convert_begin);
    std::vector<int> curve_mapping;
    for(const auto& name:evaluator_.activeJointNames())
      curve_mapping.push_back(std::distance(traj.joint_names.begin(),std::find(traj.joint_names.begin(),traj.joint_names.end(),name)));
    if(geometry_backend_=="primitive")q_curves=care_confidence_map::trajectoryCurvatures(traj,curve_mapping);
    bool shell_eligible=false;Eigen::VectorXd shell_acceleration;
    if(initial_shell_recovery_enabled_ && trajectory_source=="predicted") {
      const bool valid=recoveryContext(traj,&shell_eligible,&shell_acceleration);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        context_clock_wait_stamp_ns_ = current_context_reason_=="context_clock_pending"
            ? traj.header.stamp.toNSec() : 0ULL;
      }
      publishRecoveryProtocol(traj);
      if(!valid)return;
    }

    const ros::WallTime body_fk_begin = ros::WallTime::now();
    care_confidence_map::TrajectorySampleResult sample_result;
    std::vector<care_confidence_map::VbcPrimitiveFrame> primitive_frames;
    std::vector<care_confidence_map::VbcPrimitiveFrame> initial_frames;
    std::vector<int> sweep_indices;
    std::vector<double> sweep_times;
    bool body_ok = false;
    if (geometry_backend_ == "primitive") {
      body_ok = primitive_model_.computeContinuousTrajectory(evaluator_, q_traj,
          original_indices, eval_times_s, &primitive_frames, &sweep_indices,
          &sweep_times, &error_msg, continuous_motion_bound_enabled_,q_curves);
      // Exact initial geometry classifies recovery's starting condition. The
      // interval enclosures above remain the gate; this is not an exemption.
      if (body_ok) body_ok = primitive_model_.computeTrajectory(
          evaluator_, {q_traj.front()}, &initial_frames, &error_msg);
    }
    else
    {
      sample_result = evaluator_.computeTrajectorySamples(q_traj);
      body_ok = sample_result.success;
      error_msg = sample_result.message;
    }
    current_eval_timing_.body_fk_ms = wallMs(body_fk_begin);
    if (!body_ok)
    {
      ROS_WARN_STREAM_THROTTLE(
          2.0, "[trajectory_vbc_temporal] body-sweep FK failed: "
                   << error_msg);
      return;
    }

    if (geometry_backend_ == "primitive" && include_measured_body_) {
      sensor_msgs::JointStateConstPtr measured;
      {std::lock_guard<std::mutex> lock(mutex_); measured=latest_measured_;}
      const double age=measured ? (ros::Time::now()-measured->header.stamp).toSec() : 1.;
      if (!measured || measured->header.stamp.isZero() || age<0 || age>0.15 ||
          measured->name.size()!=measured->position.size()) {
        ROS_WARN_STREAM_THROTTLE(1.0,"VBC actual-body audit unavailable: measured age=" << age);
        return;
      }
      Eigen::VectorXd actual(evaluator_.nq());
      for (int j=0;j<evaluator_.nq();++j) {
        auto it=std::find(measured->name.begin(),measured->name.end(),evaluator_.activeJointNames()[j]);
        if(it==measured->name.end()) {
          ROS_WARN_STREAM_THROTTLE(1.0,"VBC actual-body audit missing joint " << evaluator_.activeJointNames()[j]);
          return;
        }
        actual[j]=measured->position[std::distance(measured->name.begin(),it)];
      }
      std::vector<care_confidence_map::VbcPrimitiveFrame> actual_frames;
      if (!actual.allFinite() || !primitive_model_.computeTrajectory(evaluator_,{actual},&actual_frames,&error_msg)) return;
      actual_frames[0].original_eval_timestep=0;
      primitive_frames.insert(primitive_frames.begin(),std::move(actual_frames[0]));
      sweep_indices.insert(sweep_indices.begin(),original_indices.front());
      sweep_times.insert(sweep_times.begin(),0.0);
    }
    const ros::WallTime swept_begin = ros::WallTime::now();
    std::vector<SweepVoxel> swept_voxels;
    try
    {
      if (geometry_backend_ == "primitive")
        swept_voxels = care_confidence_map::buildPrimitiveSweptVoxels<SweepVoxel>(
            primitive_frames, sweep_indices, sweep_times,
            {{map_x_min_, map_y_min_, map_z_min_}, {map_x_max_, map_y_max_, map_z_max_},
             map_resolution_}, swept_volume_margin_m_,voxelSupportRadius());
      else
        swept_voxels = buildSweptVolumeVoxels(sample_result, original_indices, eval_times_s);
    }
    catch (const std::exception& e)
    {
      // No safe summary or samples fallback: final verifier must time out closed.
      ROS_ERROR_STREAM_THROTTLE(2.0, "VBC geometry evaluation failed: " << e.what());
      return;
    }
    current_eval_timing_.swept_voxel_build_ms = wallMs(swept_begin);
    last_swept_voxel_query_count_ = swept_voxels.size();
    if (swept_voxels.empty())
    {
      finalize_total();
      publishNoTargetSummary("empty_swept_volume", trajectory_source, 0);
      return;
    }

    care_confidence_map::QueryConfidence confidence_srv;
    const ros::WallTime confidence_begin = ros::WallTime::now();
    if (!queryConfidencePoints(swept_voxels, &confidence_srv))
    {
      current_eval_timing_.confidence_query_ms = wallMs(confidence_begin);
      finalize_total();
      ROS_WARN_THROTTLE(
          2.0,
          "[trajectory_vbc_temporal] swept-volume confidence query failed "
          "or returned invalid sizes");
      return;
    }
    current_eval_timing_.confidence_query_ms = wallMs(confidence_begin);

    const ros::WallTime candidate_filter_begin = ros::WallTime::now();
    std::map<CandidateKey, Candidate> candidate_map;
    // Spend a bounded amount of work on coarse-enclosure-only unknown points.
    // A failed proof retains the original voxel AND its original entry time.
    // primitive_frames already includes the same-snapshot measured pose.
    // Its exact margin support must survive any nominal-only exclusion proof.
    if(current_shared_budget_ && !certification_budgets_.count(current_eval_stamp_ns_))
      certification_budgets_[current_eval_stamp_ns_]=256;
    while(certification_budgets_.size()>64)certification_budgets_.erase(certification_budgets_.begin());
    int refinement_fk_remaining = current_shared_budget_ ? certification_budgets_[current_eval_stamp_ns_] : 256;
    // Both candidate exclusion and the mandatory recovery certificate query
    // the same pure FK at many identical configurations. Share those exact
    // frames within this evaluation; otherwise point-by-point exclusion can
    // consume all 256 FK calls before the measured brake is even checked.
    // Point distances, margins, map classes and interval proofs remain fresh.
    care_confidence_map::VbcPrimitiveModel::ProofFkCache local_proof_cache;
    auto& proof_fk_cache=current_shared_budget_ ? proof_audit_caches_[current_eval_stamp_ns_] : local_proof_cache;
    while(proof_audit_caches_.size()>8) {
      auto old=proof_audit_caches_.begin();if(old->first==current_eval_stamp_ns_)++old;
      proof_audit_caches_.erase(old);
    }
    struct BudgetSaver {
      std::map<unsigned long long,int>* budgets;unsigned long long stamp;int* remaining;
      ~BudgetSaver(){if(budgets)(*budgets)[stamp]=*remaining;}
    } save_budget{current_shared_budget_?&certification_budgets_:nullptr,current_eval_stamp_ns_,&refinement_fk_remaining};
    for (std::size_t i = 0; i < swept_voxels.size(); ++i)
    {
      const bool inside = confidence_srv.response.inside_map[i] != 0;
      const double confidence = static_cast<double>(
          confidence_srv.response.confidence[i]);
      const double current_visibility = static_cast<double>(
          confidence_srv.response.current_visibility[i]);
      if (!inside) continue;
      if (confidence > frontier_confidence_threshold_ ||
          current_visibility > 0.5) {
        shell_recovery_.observedFree(swept_voxels[i].point_base); continue;
      }

      const SweepVoxel& voxel = swept_voxels[i];
      const int refinement_before = refinement_fk_remaining;
      std::string refinement_status = geometry_backend_ == "primitive"
          ? (voxel.primitive_signed_distance_m <= swept_volume_margin_m_ + 1e-12
              ? "representative_inside_margin" : "fk_budget_exhausted")
          : "not_applicable";
      if (!(shell_eligible && current_geometric_restore_) && geometry_backend_ == "primitive" &&
          voxel.primitive_signed_distance_m > swept_volume_margin_m_ + 1e-12 &&
          refinement_fk_remaining > 0) {
        // The first enclosing frame can be outside the nominal margin while
        // a later sampled pose actually intersects it. Such a point cannot
        // pass the full-path proof; reuse existing FK to avoid futile splits.
        // For execution this also checks the frozen measured-body frame: a
        // point in its margin cannot be discarded by a nominal-path proof.
        const bool sampled_intersection = std::any_of(
            primitive_frames.begin(), primitive_frames.end(), [&](const auto& frame) {
              return std::any_of(frame.primitives.begin(), frame.primitives.end(),
                  [&](const auto& primitive) {
                    return supportDistance(primitive,voxel.point_base) <=
                        swept_volume_margin_m_ + 1e-12;
                  });
            });
        refinement_status = sampled_intersection ? "sampled_intersection"
            : "continuous_exclusion_not_proven";
        if (!sampled_intersection) ++last_sweep_refinement_checks_;
        if (!sampled_intersection && proveSupportOutsideContinuousTrajectory(
                evaluator_, q_traj, voxel.point_base, swept_volume_margin_m_,
                &refinement_fk_remaining,q_curves,nullptr,nullptr,&proof_fk_cache)) {
          ++last_sweep_refinement_excluded_;
          continue;
        }
      }
      if (refinement_status == "continuous_exclusion_not_proven" && refinement_fk_remaining == 0)
        refinement_status = "continuous_exclusion_budget_exhausted";
      if(shell_eligible && current_geometric_restore_)
        refinement_status="deferred_for_mandatory_certificate";
      const CandidateKey key = candidateKey(voxel.point_base);
      auto it = candidate_map.find(key);
      if (it == candidate_map.end())
      {
        Candidate candidate;
        candidate.point_base = voxel.point_base;
        candidate.link_name = voxel.link_name;
        candidate.sample_index_in_link = voxel.sample_index_in_link;
        candidate.source_type = voxel.source_type;
        candidate.source_collision_index = voxel.source_collision_index;
        candidate.source_collision_name = voxel.source_collision_name;
        candidate.primitive_signed_distance_m = voxel.primitive_signed_distance_m;
        candidate.refinement_status = refinement_status;
        candidate.refinement_fk_before = refinement_before;
        candidate.refinement_fk_after = refinement_fk_remaining;
        candidate.candidate_margin_m = swept_volume_margin_m_;
        if (!initial_frames.empty()) {
          double distance = std::numeric_limits<double>::infinity();
          for (const auto& primitive : initial_frames.front().primitives)
            distance = std::min(distance, supportDistance(primitive,voxel.point_base));
          candidate.initial_support_distance_m = distance;
          candidate.initial_support_unknown = distance <= swept_volume_margin_m_ + 1e-12;
        }
        candidate.sample_center_base = voxel.sample_center_base;
        candidate.raw_sample_radius_m = voxel.raw_sample_radius_m;
        candidate.swept_radius_m = voxel.swept_radius_m;
        candidate.point_center_distance_m = voxel.point_center_distance_m;
        candidate.confidence = confidence;
        candidate.current_visibility = current_visibility;
        candidate.sweep_eval_timestep = voxel.sweep_eval_timestep;
        candidate.sweep_original_timestep = voxel.sweep_original_timestep;
        candidate.sweep_time_s = voxel.sweep_time_s;
        candidate_map.emplace(key, candidate);
      }
      else
      {
        it->second.confidence = std::min(it->second.confidence, confidence);
        if (voxel.sweep_time_s < it->second.sweep_time_s)
        {
          it->second.point_base = voxel.point_base;
          it->second.link_name = voxel.link_name;
          it->second.sample_index_in_link = voxel.sample_index_in_link;
          it->second.source_type = voxel.source_type;
          it->second.source_collision_index = voxel.source_collision_index;
          it->second.source_collision_name = voxel.source_collision_name;
          it->second.primitive_signed_distance_m = voxel.primitive_signed_distance_m;
          it->second.refinement_status = refinement_status;
          it->second.refinement_fk_before = refinement_before;
          it->second.refinement_fk_after = refinement_fk_remaining;
          it->second.candidate_margin_m = swept_volume_margin_m_;
          it->second.sample_center_base = voxel.sample_center_base;
          it->second.raw_sample_radius_m = voxel.raw_sample_radius_m;
          it->second.swept_radius_m = voxel.swept_radius_m;
          it->second.point_center_distance_m = voxel.point_center_distance_m;
          it->second.sweep_eval_timestep = voxel.sweep_eval_timestep;
          it->second.sweep_original_timestep = voxel.sweep_original_timestep;
          it->second.sweep_time_s = voxel.sweep_time_s;
        }
      }
    }

    last_sweep_refinement_fk_ = 256 - refinement_fk_remaining;
    current_eval_timing_.candidate_filter_ms = wallMs(candidate_filter_begin);
    if(current_geometric_restore_) {
      for(const auto& p:geometric_restore_points_) {
        auto& c=candidate_map[candidateKey(p)];
        c.point_base=p;c.initial_support_unknown=true;c.initial_support_distance_m=1e9;
        for(const auto& b:initial_frames.front().primitives)
          c.initial_support_distance_m=std::min(c.initial_support_distance_m,supportDistance(b,p));
      }
    }
    if (candidate_map.empty())
    {
      finalize_total();
      publishNoTargetSummary(
          "no_low_confidence_sweep_candidates", trajectory_source, 0);
      return;
    }

    const ros::WallTime sensor_fk_begin = ros::WallTime::now();
    std::vector<std::vector<care_confidence_map::FramePoseInBase>> sensor_poses(
        q_traj.size());
    for (std::size_t k = 0; k < q_traj.size(); ++k)
    {
      if (!evaluator_.computeFramePosesForConfiguration(
              q_traj[k], sensor_frames_, &sensor_poses[k], &error_msg))
      {
        ROS_WARN_STREAM_THROTTLE(
            2.0, "[trajectory_vbc_temporal] sensor FK failed: " << error_msg);
        current_eval_timing_.sensor_fk_ms = wallMs(sensor_fk_begin);
        finalize_total();
        return;
      }
    }
    current_eval_timing_.sensor_fk_ms = wallMs(sensor_fk_begin);

    const ros::WallTime visibility_begin = ros::WallTime::now();
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
    if(shell_eligible) {
      const bool certified=certifyInitialShell(&candidates,q_traj,shell_acceleration,&refinement_fk_remaining,q_curves,traj,proof_fk_cache);
      if(current_geometric_restore_ && !certified) {
        current_context_state_="invalid";
        current_context_reason_=current_shell_path_counterexample_
            ? "geometric_restore_path_counterexample" : current_proof_resource_evidence_!="none"
            ? "geometric_restore_proof_resource_exhausted" : "geometric_restore_certificate_unavailable";
        publishRecoveryProtocol(traj);finalize_total();return;
      }
      last_sweep_refinement_fk_=256-refinement_fk_remaining;
      if(certified) {
        const auto before=candidates.size();
        candidates.erase(std::remove_if(candidates.begin(),candidates.end(),
            [](const Candidate& c){return c.full_path_excluded;}),candidates.end());
        last_sweep_refinement_excluded_+=before-candidates.size();
      }
    }
    current_eval_timing_.visibility_scan_ms = wallMs(visibility_begin);

    const ros::WallTime cluster_begin = ros::WallTime::now();
    const std::vector<TemporalLayer> layers = buildTemporalLayers(candidates);
    current_eval_timing_.cluster_ms = wallMs(cluster_begin);
    if (layers.empty())
    {
      finalize_total();
      publishNoTargetSummary(
          shell_recovery_count_>0 ? "initial_shell_recovery_certified" :
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
      finalize_total();
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
    finalize_total();
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
    bool evaluate_now = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_traj_ = *msg;
      trajectory_received_wall_ = ros::WallTime::now();
      has_traj_ = true;
      evaluate_now =
          event_driven_eval_ && !force_bootstrap_ &&
          !prefer_predicted_trajectory_;
    }
    if (evaluate_now)
      evaluate(*msg, "bootstrap", "bootstrap_callback");
  }

  void predictedTrajectoryCallback(
      const trajectory_msgs::JointTrajectoryConstPtr& msg)
  {
    if (!msg || msg->points.empty()) return;
    bool evaluate_now = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_predicted_traj_ = *msg;
      predicted_received_wall_ = ros::WallTime::now();
      predicted_traj_received_ = ros::Time::now();
      has_predicted_traj_ = true;
      evaluate_now =
          event_driven_eval_ && !force_bootstrap_ &&
          prefer_predicted_trajectory_;
    }
    if (evaluate_now)
    {
      evaluate(*msg, "predicted", "predicted_callback");
      std::lock_guard<std::mutex> lock(mutex_);
      ros::Time stamp = msg->header.stamp;
      if (stamp.isZero()) stamp = ros::Time::now();
      last_event_evaluated_predicted_stamp_ns_ = stampToNs(stamp);
      last_predicted_full_eval_wall_ = ros::WallTime::now();
    }
  }

  void rediscoveryTrajectoryCallback(
      const trajectory_msgs::JointTrajectoryConstPtr& msg)
  {
    if (!msg || msg->points.empty()) return;
    bool evaluate_now = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_rediscovery_traj_ = *msg;
      rediscovery_received_wall_ = ros::WallTime::now();
      rediscovery_traj_received_ = ros::Time::now();
      has_rediscovery_traj_ = true;
      evaluate_now =
          event_driven_eval_ && force_bootstrap_ &&
          !rediscovery_trajectory_topic_.empty();
    }
    if (evaluate_now)
      evaluate(*msg, "scp_rediscovery", "rediscovery_callback");
  }

  void forceBootstrapCallback(const std_msgs::BoolConstPtr& msg)
  {
    if (!msg) return;

    bool changed = false;
    bool evaluate_rediscovery_now = false;
    trajectory_msgs::JointTrajectory rediscovery_traj;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      changed = force_bootstrap_ != msg->data;
      force_bootstrap_ = msg->data;

      if (event_driven_eval_ && force_bootstrap_ &&
          !rediscovery_trajectory_topic_.empty() &&
          has_rediscovery_traj_)
      {
        const double age_s =
            (ros::Time::now() - rediscovery_traj_received_).toSec();
        if (age_s >= 0.0 &&
            age_s <= rediscovery_trajectory_timeout_)
        {
          rediscovery_traj = latest_rediscovery_traj_;
          evaluate_rediscovery_now = true;
        }
      }
    }

    if (changed)
    {
      ROS_INFO_STREAM(
          "[trajectory_vbc_temporal] force_bootstrap="
          << static_cast<int>(msg->data));
    }

    // C5.34: a solver failure usually toggles force_bootstrap after the SCP
    // query trajectory was already cached. Evaluate that fresh same-SCP
    // trajectory immediately instead of waiting one 20-Hz timer period.
    if (evaluate_rediscovery_now)
      evaluate(
          rediscovery_traj, "scp_rediscovery",
          "force_bootstrap_callback");
  }

  void contextClockCallback(const ros::WallTimerEvent&)
  {
    trajectory_msgs::JointTrajectory pending;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if(!context_clock_wait_stamp_ns_)return;
      if(force_bootstrap_ || !has_predicted_traj_ ||
          latest_predicted_traj_.header.stamp.toNSec()!=context_clock_wait_stamp_ns_) {
        context_clock_wait_stamp_ns_=0;return;
      }
      if(ros::Time::now().toNSec()<context_clock_wait_stamp_ns_)return;
      pending=latest_predicted_traj_;
      context_clock_wait_stamp_ns_=0;
    }
    evaluate(pending,"predicted","context_clock_ready");
  }

  void timerCallback(const ros::TimerEvent&)
  {
    trajectory_msgs::JointTrajectory traj;
    std::string source = "bootstrap";
    bool count_predicted_refresh = false;
    const ros::Time now = ros::Time::now();
    const ros::WallTime wall_now = ros::WallTime::now();
    {
      std::lock_guard<std::mutex> lock(mutex_);

      // C5.30: a PROBE solver failure must be rediscovered on the SAME
      // trajectory that produced the local-GCDF batch. Sparse-SCP publishes
      // that linearization before requesting GCDF. If this dedicated
      // rediscovery trajectory is missing or stale, fail closed and wait;
      // never substitute a task-bootstrap SAFE verdict from another path.
      if (force_bootstrap_ && !rediscovery_trajectory_topic_.empty())
      {
        const double age_s = has_rediscovery_traj_
            ? (now - rediscovery_traj_received_).toSec()
            : std::numeric_limits<double>::infinity();
        const bool rediscovery_fresh =
            has_rediscovery_traj_ &&
            age_s >= 0.0 &&
            age_s <= rediscovery_trajectory_timeout_;
        if (!rediscovery_fresh)
        {
          ROS_WARN_THROTTLE(
              1.0,
              "[trajectory_vbc_temporal] waiting for fresh SCP rediscovery "
              "trajectory on %s (has=%d age=%.3f timeout=%.3f)",
              rediscovery_trajectory_topic_.c_str(),
              static_cast<int>(has_rediscovery_traj_),
              age_s,
              rediscovery_trajectory_timeout_);
          return;
        }
        traj = latest_rediscovery_traj_;
        source = "scp_rediscovery";
      }
      else
      {
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

          if (event_driven_eval_ && !preserve_periodic_eval_)
          {
            ros::Time stamp = traj.header.stamp;
            const unsigned long long stamp_ns =
                stamp.isZero() ? 0ULL : stampToNs(stamp);
            const double refresh_period_s =
                1.0 / std::max(0.1, predicted_periodic_refresh_rate_);
            const double since_full_eval_s =
                last_predicted_full_eval_wall_.toSec() > 0.0
                    ? (wall_now - last_predicted_full_eval_wall_).toSec()
                    : std::numeric_limits<double>::infinity();

            if (stamp_ns != 0ULL &&
                stamp_ns == last_event_evaluated_predicted_stamp_ns_ &&
                since_full_eval_s >= 0.0 &&
                since_full_eval_s < refresh_period_s)
            {
              ++predicted_periodic_skip_count_;
              return;
            }
            count_predicted_refresh = true;
          }
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
    }
    evaluate(traj, source, "timer");
    if (count_predicted_refresh)
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++predicted_periodic_refresh_count_;
      last_predicted_full_eval_wall_ = ros::WallTime::now();
    }
  }

  void printSummary() const
  {
    ROS_INFO_STREAM("");
    ROS_INFO_STREAM(
        "========== trajectory_vbc_selector: bounded temporal clusters ==========");
    ROS_INFO_STREAM("enabled: " << enabled_);
    ROS_INFO_STREAM("event-driven evaluation: "
                    << static_cast<int>(event_driven_eval_)
                    << " (periodic timer retained as fallback)");
    ROS_INFO_STREAM(
        "predicted periodic full-refresh rate: "
        << predicted_periodic_refresh_rate_
        << " Hz when event-driven evaluation is enabled");
    ROS_INFO_STREAM("bootstrap trajectory: " << input_trajectory_topic_);
    ROS_INFO_STREAM("predicted trajectory: " << predicted_trajectory_topic_);
    ROS_INFO_STREAM("rediscovery trajectory: "
                    << (rediscovery_trajectory_topic_.empty()
                            ? std::string("disabled")
                            : rediscovery_trajectory_topic_)
                    << ", timeout=" << rediscovery_trajectory_timeout_ << " s");
    ROS_INFO_STREAM("prefer predicted: " << prefer_predicted_trajectory_
                    << ", timeout=" << predicted_trajectory_timeout_ << " s");
    ROS_INFO_STREAM("candidate resolution: " << candidate_resolution_ << " m");
    ROS_INFO_STREAM("spatial support: " << geometry_backend_ << " volume, margin="
                    << swept_volume_margin_m_ << " m, map_resolution="
                    << map_resolution_ << " m");
    ROS_INFO_STREAM("FK URDF: " << robot_urdf_file_ << "; primitive URDF: "
                    << (geometry_backend_ == "primitive" ? primitive_urdf_file_ : "unused")
                    << "; primitive semantics: continuous linear interval enclosures, 5mm tightness");
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
  ros::Subscriber rediscovery_trajectory_sub_;
  ros::Subscriber force_bootstrap_sub_;
  ros::ServiceClient confidence_query_client_;
  ros::Publisher target_pub_;
  ros::Publisher candidate_active_pub_;
  ros::Publisher summary_pub_;
  ros::Publisher selected_margin_pub_;
  ros::Publisher selected_sweep_time_pub_;
  ros::Publisher selected_see_time_pub_;
  ros::Publisher active_set_points_pub_;
  ros::Publisher active_set_bundle_pub_;
  ros::Timer timer_;
  ros::WallTimer context_clock_timer_;
  unsigned long long context_clock_wait_stamp_ns_=0;

  care_confidence_map::TrajectoryRiskEvaluator evaluator_;
  std::mutex mutex_;
  trajectory_msgs::JointTrajectory latest_traj_;
  trajectory_msgs::JointTrajectory latest_predicted_traj_;
  trajectory_msgs::JointTrajectory latest_rediscovery_traj_;
  ros::Time predicted_traj_received_;
  ros::Time rediscovery_traj_received_;
  bool has_traj_ = false;
  bool has_predicted_traj_ = false;
  bool has_rediscovery_traj_ = false;
  bool force_bootstrap_ = false;
  unsigned long long active_set_bundle_seq_ = 0;

  bool has_selected_key_ = false;
  CandidateKey selected_key_;

  bool enabled_ = true;
  bool prefer_predicted_trajectory_ = false;
  bool event_driven_eval_ = false;
  double eval_rate_ = 20.0;
  double predicted_periodic_refresh_rate_ = 5.0;
  int max_eval_timesteps_ = 50;
  double query_timeout_ = 0.10;
  bool confidence_query_persistent_ = false;
  double fallback_dt_ = 0.05;
  double predicted_trajectory_timeout_ = 0.20;
  double rediscovery_trajectory_timeout_ = 1.0;
  double frontier_confidence_threshold_ = 0.50;
  double candidate_resolution_ = 0.05;
  bool whole_voxel_support_=false;
  care_confidence_map::AnalyticGroundSupport analytic_support_;
  bool clip_analytic_support_=false;
  double swept_volume_margin_m_ = 0.025;
  bool continuous_motion_bound_enabled_ = true;
  double map_resolution_ = 0.05;
  double map_x_min_ = -0.95;
  double map_x_max_ = 0.95;
  double map_y_min_ = -0.95;
  double map_y_max_ = 0.95;
  double map_z_min_ = 0.0;
  double map_z_max_ = 1.15;
  std::size_t last_swept_voxel_query_count_ = 0;
  std::size_t last_sweep_refinement_checks_ = 0;
  std::size_t last_sweep_refinement_excluded_ = 0;
  int last_sweep_refinement_fk_ = 0;
  unsigned long long current_eval_stamp_ns_ = 0;
  std::string current_eval_trigger_ = "startup";
  EvalTiming current_eval_timing_;
  bool preserve_periodic_eval_ = false;
  ros::WallTime trajectory_received_wall_, predicted_received_wall_, rediscovery_received_wall_;
  double current_input_wait_ms_ = 0.0;
  unsigned long long last_event_evaluated_predicted_stamp_ns_ = 0;
  ros::WallTime last_predicted_full_eval_wall_;
  unsigned long long predicted_periodic_skip_count_ = 0;
  unsigned long long predicted_periodic_refresh_count_ = 0;
  double min_margin_s_ = 0.30;
  int temporal_layer_mpc_steps_ = 2;
  double region_max_diameter_m_ = 0.12;

  std::string robot_urdf_file_;
  std::string body_samples_file_;
  std::string geometry_backend_ = "samples";
  std::string primitive_urdf_file_;
  care_confidence_map::VbcPrimitiveModel primitive_model_;
  bool include_measured_body_ = false;
  ros::CallbackQueue measured_queue_;
  ros::AsyncSpinner measured_spinner_;
  ros::Subscriber measured_sub_;
  sensor_msgs::JointStateConstPtr latest_measured_;
  bool initial_shell_recovery_enabled_=false;
  bool fixed_start_rebase_reaudit_enabled_=false;
  double shell_execution_margin_m_=0.;
  double shell_occupied_margin_m_=0.;
  ros::ServiceClient shell_snapshot_client_;
  care_confidence_map::InitialShellRecovery shell_recovery_;
  ros::Subscriber recovery_context_sub_;
  ros::Subscriber recovery_quality_sub_;
  ros::Publisher recovery_protocol_pub_;
  ros::Publisher shell_diagnostic_pub_;
  std::map<unsigned long long,std::string> recovery_contexts_;
  std::map<unsigned long long,std::string> recovery_protocol_states_;
  std::string current_context_state_,current_context_reason_,current_context_session_="none";
  bool current_shell_path_counterexample_=false;
  std::string current_shell_path_counterexample_evidence_="none";
  std::string current_proof_resource_evidence_="none";
  std::string current_restore_mode_epoch_,current_restore_observation_token_;
  unsigned long long current_context_sequence_=0;
  std::string current_prefix_session_="none";
  unsigned long long current_prefix_revision_=0;
  double current_prefix_s_=0.;
  unsigned int current_eval_wire_seq_=0;
  int shell_recovery_count_=0;
  std::string shell_recovery_evidence_="none";
  bool current_shared_budget_=false;
  bool current_geometric_restore_=false;
  bool current_margin_recovery_=false;
  std::vector<Eigen::Vector3d> geometric_restore_points_;
  std::vector<Eigen::Vector3d> fixed_start_recovery_points_;
  std::map<unsigned long long,int> certification_budgets_;
  std::map<unsigned long long,double> recovery_quality_;
  std::map<unsigned long long,double> recovery_reply_allowance_;
  std::map<unsigned long long,care_confidence_map::VbcPrimitiveModel::ProofFkCache> proof_audit_caches_;
  std::string base_frame_ = "base_link";
  std::string input_trajectory_topic_ = "/care_planner/task_trajectory";
  std::string predicted_trajectory_topic_ =
      "/care_planner/mpc/predicted_trajectory";
  std::string rediscovery_trajectory_topic_;
  std::string output_target_topic_ =
      "/care_planner/active_sensing/target_candidate";
  std::string candidate_active_topic_ =
      "/care_planner/active_sensing/target_candidate_active";
  std::string confidence_query_service_ =
      "/care_planner/confidence_map/query";
  std::string active_set_points_topic_ =
      "/care_planner/trajectory_risk/vbc_active_set_points";
  std::string active_set_bundle_topic_ =
      "/care_planner/trajectory_risk/vbc_active_set_bundle";
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
