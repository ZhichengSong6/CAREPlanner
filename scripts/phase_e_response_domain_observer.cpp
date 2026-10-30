// Passive, paired physics-step measurements for the finite-force Gazebo study.
// This observer never writes joint commands or changes physics configuration.
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <array>
#include <chrono>
#include <cmath>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace gazebo {
class CareResponseDomainObserver : public ModelPlugin {
 public:
  void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override {
    model_ = model;
    world_ = model->GetWorld();
    physics_ = world_->Physics();
    const std::array<std::string, 7> names{{"joint1", "joint2", "joint3", "joint4",
        "wrist_joint1", "wrist_joint2", "wrist_joint3"}};
    for (size_t j = 0; j < 7; ++j) {
      joints_[j] = model->GetJoint(names[j]);
      if (!joints_[j]) throw std::runtime_error("response observer missing joint");
    }
    const std::string directory = sdf->Get<std::string>("directory");
    output_.open(directory + "/physics_steps.jsonl", std::ios::out);
    if (!output_) throw std::runtime_error("response observer output unavailable");
    output_ << std::setprecision(17);
    std::ofstream(directory + "/expanded_model.sdf") << model->GetSDF()->ToString("");
    std::ofstream(directory + "/expanded_physics.sdf") << physics_->GetSDF()->ToString("");
    std::ofstream domain(directory + "/physics_runtime.json");
    const auto gravity = world_->Gravity();
    domain << std::setprecision(17) << "{\"physics_type\":" << std::quoted(physics_->GetType())
        << ",\"max_step_size\":" << physics_->GetMaxStepSize()
        << ",\"real_time_update_rate\":" << physics_->GetRealTimeUpdateRate()
        << ",\"gravity\":[" << gravity.X() << ',' << gravity.Y() << ',' << gravity.Z()
        << "],\"global_symbol_providers\":{";
    bool first = true;
    for (const char* name : {"dWorldQuickStep", "dWorldStep", "dJointSetHingeParam"}) {
      Dl_info info{};
      void* symbol = dlsym(RTLD_DEFAULT, name);
      if (!symbol || !dladdr(symbol, &info) || !info.dli_fname)
        throw std::runtime_error("response observer unresolved ODE symbol");
      if (!first) domain << ',';
      first = false;
      domain << std::quoted(name) << ':' << std::quoted(info.dli_fname);
    }
    domain << "},\"parameter_types\":{";
    first = true;
    for (const char* name : {"constraint_residual", "rms_error", "num_contacts"}) {
      boost::any parameter;
      physics_->GetParam(name, parameter);
      if (!first) domain << ',';
      first = false;
      domain << std::quoted(name) << ':' << std::quoted(parameter.type().name());
    }
    domain << "},\"scope\":\"passive diagnostic; no response qualification\"}\n";
    domain.close();
    origin_ = std::chrono::steady_clock::now();
    begin_ = event::Events::ConnectWorldUpdateBegin(
        std::bind(&CareResponseDomainObserver::Begin, this, std::placeholders::_1));
    end_ = event::Events::ConnectWorldUpdateEnd(
        std::bind(&CareResponseDomainObserver::End, this));
  }

 private:
  physics::ModelPtr model_;
  physics::WorldPtr world_;
  physics::PhysicsEnginePtr physics_;
  std::array<physics::JointPtr, 7> joints_;
  event::ConnectionPtr begin_, end_;
  std::ofstream output_;
  std::array<double, 7> before_{}, velocity_before_{}, motor_{}, fmax_{};
  std::chrono::steady_clock::time_point origin_;
  uint64_t stamp_ = 0, index_ = 0;
  double step_ = 0., wall_ = 0.;
  bool paired_ = false;

  template <class T> void Array(const T& values) {
    output_ << '[';
    for (size_t j = 0; j < 7; ++j) {
      if (j) output_ << ',';
      if (std::isfinite(values[j])) output_ << values[j];
      else output_ << "null";
    }
    output_ << ']';
  }
  void Parameter(const char* key) {
    boost::any value;
    output_ << ',' << std::quoted(key) << ':';
    if (!physics_->GetParam(key, value)) { output_ << "null"; return; }
    if (value.type() == typeid(double)) {
      const double number = boost::any_cast<double>(value);
      if (std::isfinite(number)) output_ << number;
      else output_ << "null";
    } else if (value.type() == typeid(double*)) {
      // Gazebo's ODE interface returns four RMS categories via an owned array,
      // not a scalar. Read it synchronously in the physics thread.
      const double* numbers = boost::any_cast<double*>(value);
      output_ << '[';
      for (int i = 0; i < 4; ++i) {
        if (i) output_ << ',';
        if (numbers && std::isfinite(numbers[i])) output_ << numbers[i];
        else output_ << "null";
      }
      output_ << ']';
    } else if (value.type() == typeid(int)) output_ << boost::any_cast<int>(value);
    else if (value.type() == typeid(unsigned int)) output_ << boost::any_cast<unsigned int>(value);
    else output_ << "null";
  }
  void Begin(const common::UpdateInfo& info) {
    if (paired_) throw std::runtime_error("response observer unpaired begin");
    paired_ = true;
    stamp_ = uint64_t(info.simTime.sec) * 1000000000 + info.simTime.nsec;
    step_ = physics_->GetMaxStepSize();
    wall_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - origin_).count();
    for (size_t j = 0; j < 7; ++j) {
      before_[j] = joints_[j]->Position(0);
      velocity_before_[j] = joints_[j]->GetVelocity(0);
      motor_[j] = joints_[j]->GetParam("vel", 0);
      fmax_[j] = joints_[j]->GetParam("fmax", 0);
    }
  }
  void End() {
    if (!paired_) return;  // A load may occur after this step's begin callback.
    paired_ = false;
    std::array<double, 7> after{}, velocity{}, motor_after{};
    for (size_t j = 0; j < 7; ++j) {
      after[j] = joints_[j]->Position(0);
      velocity[j] = joints_[j]->GetVelocity(0);
      motor_after[j] = joints_[j]->GetParam("vel", 0);
    }
    output_ << "{\"index\":" << index_++ << ",\"t_ns\":" << stamp_
        << ",\"end_time_ns\":" << world_->SimTime().Double() * 1e9
        << ",\"step_s\":" << step_ << ",\"wall_s\":" << wall_ << ",\"q_before\":";
    Array(before_);
    output_ << ",\"q_after\":"; Array(after);
    output_ << ",\"v_before\":"; Array(velocity_before_);
    output_ << ",\"velocity\":"; Array(velocity);
    output_ << ",\"motor\":"; Array(motor_);
    output_ << ",\"motor_after\":"; Array(motor_after);
    output_ << ",\"fmax\":"; Array(fmax_);
    Parameter("constraint_residual");
    Parameter("rms_error");
    Parameter("num_contacts");
    output_ << "}\n";
    if (index_ % 100 == 0) output_.flush();
  }
};
GZ_REGISTER_MODEL_PLUGIN(CareResponseDomainObserver)
}  // namespace gazebo
