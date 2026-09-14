#include <care_confidence_map/vbc_primitive_model.hpp>
#include <care_confidence_map/gcdf_primitive_anchors.hpp>
#include <ros/ros.h>
#include <std_msgs/String.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <chrono>
#include <set>
#include <sstream>

namespace care_confidence_map {
class GcdfPrimitiveAnchorNode {
 public:
  GcdfPrimitiveAnchorNode() : pnh_("~") {
    std::string urdf, collision, input, output, error;
    std::vector<std::string> ignored;
    pnh_.param<std::string>("robot_urdf_file",urdf,"");
    pnh_.param<std::string>("primitive_urdf_file",collision,"");
    pnh_.param<std::string>("base_frame",base_,"base_link");
    pnh_.param<std::string>("input_trajectory_topic",input,"");
    pnh_.param<std::string>("output_topic",output,"");
    pnh_.param("body_inflation_m",inflation_,0.);
    pnh_.param("max_eval_timesteps",max_steps_,50);
    if (!pnh_.getParam("ignored_risk_links",ignored) || input.empty() || output.empty() ||
        !std::isfinite(inflation_) || inflation_<0 || max_steps_<1 || max_steps_>50)
      throw std::runtime_error("Explicit ignored links, valid topics/inflation and 1..50 steps required");
    if (!fk_.initializeKinematics(urdf,base_,&error) || fk_.nq()!=7 ||
        !model_.load(collision,ignored,&error)) throw std::runtime_error(error);
    // Validate every active collision frame before accepting any query.
    std::vector<VbcPrimitiveFrame> probe;
    if (!model_.computeTrajectory(fk_,{Eigen::VectorXd::Zero(7)},&probe,&error))
      throw std::runtime_error(error);
    pub_=nh_.advertise<sensor_msgs::PointCloud2>(output,1);
    diagnostic_=nh_.advertise<std_msgs::String>(output+"/geometry_summary",10);
    sub_=nh_.subscribe(input,1,&GcdfPrimitiveAnchorNode::callback,this);
    ROS_INFO_STREAM("GCDF primitive anchors READY shapes="<<model_.primitives().size()
        <<" samples_yaml_loaded=0 inflation="<<inflation_);
  }
 private:
  void callback(const trajectory_msgs::JointTrajectoryConstPtr& msg) {
    const auto started=std::chrono::steady_clock::now();
    try {
      if (!msg || msg->header.stamp.isZero() || msg->points.empty() || msg->points.size()>10000 ||
          (!msg->header.frame_id.empty() && msg->header.frame_id!=base_) || msg->joint_names.size()!=7 ||
          std::set<std::string>(msg->joint_names.begin(),msg->joint_names.end()).size()!=7)
        throw std::invalid_argument("Invalid trajectory identity/frame/joints/dimensions");
      if (msg->header.stamp<=last_stamp_) {
        ROS_WARN_STREAM("GCDF primitive anchor duplicate/stale query dropped: stamp_ns="
            << msg->header.stamp.toNSec() << " last_stamp_ns=" << last_stamp_.toNSec());
        return;
      }
      std::vector<int> order;
      for (const auto& name:fk_.activeJointNames()) {
        auto it=std::find(msg->joint_names.begin(),msg->joint_names.end(),name);
        if (it==msg->joint_names.end()) throw std::invalid_argument("Missing active joint");
        order.push_back(it-msg->joint_names.begin());
      }
      ros::Duration previous(-1.);
      for (const auto& p:msg->points) {
        if (p.positions.size()!=7 || p.time_from_start.toSec()<0 || p.time_from_start<=previous ||
            !std::all_of(p.positions.begin(),p.positions.end(),[](double x){return std::isfinite(x);}))
          throw std::invalid_argument("Nonfinite/dimension/timing error in trajectory");
        previous=p.time_from_start;
      }
      const int n=msg->points.size();
      if (max_steps_==1 && n!=1) throw std::invalid_argument("Single-state audit requires one knot");
      std::vector<int> indices;
      for (int k=0;k<std::min(n,max_steps_);++k)
        indices.push_back(n<=max_steps_?k:static_cast<int>(std::round(double(k)*(n-1)/(max_steps_-1))));
      std::vector<Eigen::VectorXd> q;
      for (int index:indices) {
        Eigen::VectorXd value(7);
        for (int j=0;j<7;++j) value[j]=msg->points[index].positions[order[j]];
        q.push_back(value);
      }
      std::vector<VbcPrimitiveFrame> frames;std::string error;
      if (!model_.computeTrajectory(fk_,q,&frames,&error)) throw std::runtime_error(error);
      const auto fk_done=std::chrono::steady_clock::now();
      std::vector<GcdfPrimitiveAnchor> anchors;
      anchors.reserve(indices.size()*model_.primitives().size());
      for (std::size_t k=0;k<indices.size();++k) for (const auto& shape:frames[k].primitives) {
        GcdfPrimitiveAnchor a;a.shape=shape;a.inflation=inflation_;
        a.eval_timestep=k;a.original_timestep=indices[k];
        for (int j=0;j<7;++j) a.q[j]=static_cast<float>(q[k][j]);
        anchors.push_back(std::move(a));
      }
      auto header=msg->header;header.frame_id=base_;
      const auto cloud=encodePrimitiveAnchors(header,anchors);
      pub_.publish(cloud);last_stamp_=msg->header.stamp;
      std::ostringstream out;
      out<<"geometry_backend=primitive samples_yaml_loaded=0 trajectory_stamp_ns="<<last_stamp_.toNSec()
         <<" knots="<<indices.size()<<" anchors="<<anchors.size()<<" wire_bytes="<<cloud.data.size()
         <<" fk_ms="<<std::chrono::duration<double,std::milli>(fk_done-started).count()
         <<" total_ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
      std_msgs::String diag;diag.data=out.str();diagnostic_.publish(diag);
    } catch (const std::exception& e) {
      // Never emit an empty cloud/batch as a substitute safety certificate.
      ROS_ERROR_STREAM("GCDF primitive anchor rejected: "<<e.what());
    }
  }
  ros::NodeHandle nh_,pnh_;
  ros::Publisher pub_,diagnostic_;ros::Subscriber sub_;
  TrajectoryRiskEvaluator fk_;VbcPrimitiveModel model_;
  std::string base_;double inflation_=0.;int max_steps_=50;ros::Time last_stamp_;
};
}
int main(int argc,char** argv) {
  ros::init(argc,argv,"gcdf_primitive_anchors");
  try {care_confidence_map::GcdfPrimitiveAnchorNode node;ros::spin();}
  catch(const std::exception& e) {ROS_FATAL_STREAM(e.what());return 1;}
  return 0;
}
