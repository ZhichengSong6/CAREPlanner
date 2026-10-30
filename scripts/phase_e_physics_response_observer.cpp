// Passive diagnostic: pair pre/post physics states at every simulation step.
// The ODE motor target is observed after gazebo_ros_control's update callback.
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <array>
#include <fstream>
#include <iomanip>

namespace gazebo {
class CarePhysicsResponseObserver : public ModelPlugin {
 public:
  void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override {
    model_=model;
    const std::array<std::string,7> names{{"joint1","joint2","joint3","joint4","wrist_joint1","wrist_joint2","wrist_joint3"}};
    for(size_t j=0;j<7;++j) joints_[j]=model->GetJoint(names[j]);
    output_.open(sdf->Get<std::string>("output"));
    output_<<std::setprecision(17);
    begin_=event::Events::ConnectWorldUpdateBegin(std::bind(&CarePhysicsResponseObserver::Begin,this,std::placeholders::_1));
    end_=event::Events::ConnectWorldUpdateEnd(std::bind(&CarePhysicsResponseObserver::End,this));
  }
 private:
  physics::ModelPtr model_;
  std::array<physics::JointPtr,7> joints_;
  event::ConnectionPtr begin_,end_;
  std::ofstream output_;
  std::array<double,7> before_{},motor_{};
  uint64_t stamp_=0;
  template<class T> void Array(const T& values) {
    output_<<'[';
    for(size_t j=0;j<7;++j){if(j)output_<<',';output_<<values[j];}
    output_<<']';
  }
  void Begin(const common::UpdateInfo& info) {
    stamp_=uint64_t(info.simTime.sec)*1000000000+info.simTime.nsec;
    for(size_t j=0;j<7;++j){before_[j]=joints_[j]->Position(0);motor_[j]=joints_[j]->GetParam("vel",0);}
  }
  void End() {
    std::array<double,7> after{},velocity{};
    for(size_t j=0;j<7;++j){after[j]=joints_[j]->Position(0);velocity[j]=joints_[j]->GetVelocity(0);}
    output_<<"{\"t_ns\":"<<stamp_<<",\"q_before\":";Array(before_);
    output_<<",\"q_after\":";Array(after);
    output_<<",\"motor\":";Array(motor_);
    output_<<",\"velocity\":";Array(velocity);output_<<"}\n";
    if(stamp_%1000000000==0)output_.flush();
  }
};
GZ_REGISTER_MODEL_PLUGIN(CarePhysicsResponseObserver)
}
