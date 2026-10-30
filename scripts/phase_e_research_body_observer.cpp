// Passive full-link poses for finite-domain research validation.
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <cstdlib>
#include <fstream>
#include <iomanip>

namespace gazebo {
class CareResearchBodyObserver : public ModelPlugin {
 public:
  void Load(physics::ModelPtr model, sdf::ElementPtr) override {
    model_ = model;
    const char* path = std::getenv("CARE_BODY_TRACE_FILE");
    if (!path) return;
    stream_.open(path);
    stream_ << std::setprecision(17);
    update_ = event::Events::ConnectWorldUpdateEnd([this]() {
      const auto stamp = model_->GetWorld()->SimTime();
      const auto ns = uint64_t(stamp.sec) * 1000000000 + stamp.nsec;
      if (last_ && ns >= last_ && ns - last_ < 5000000) return;
      last_ = ns;
      stream_ << "{\"sim_ns\":" << ns << ",\"links\":{";
      bool first = true;
      for (const auto& link : model_->GetLinks()) {
        if (!first) stream_ << ',';
        first = false;
        const auto p = link->WorldPose();
        stream_ << std::quoted(link->GetName()) << ":[" << p.Pos().X() << ','
          << p.Pos().Y() << ',' << p.Pos().Z() << ',' << p.Rot().X() << ','
          << p.Rot().Y() << ',' << p.Rot().Z() << ',' << p.Rot().W() << ']';
      }
      stream_ << "}}\n";
    });
  }
 private:
  physics::ModelPtr model_;
  event::ConnectionPtr update_;
  std::ofstream stream_;
  uint64_t last_ = 0;
};
GZ_REGISTER_MODEL_PLUGIN(CareResearchBodyObserver)
}
