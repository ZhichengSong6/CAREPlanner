// Exercise the actual downstream policy with producer silence after one lease.
// Emits observations only; the independent Python formula is not used here.
#include <care_velocity_controller/expiring_velocity_core.hpp>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>

using Core=care_velocity_controller::ExpiringVelocityCore;
template<class T> void array(const T& a) {
  std::cout << '['; for(size_t j=0;j<a.size();++j) { if(j)std::cout<<','; std::cout<<a[j]; } std::cout<<']';
}
int main() {
  std::cout<<std::setprecision(17);
  std::mt19937 rng(601); std::uniform_real_distribution<double> uniform(-1.,1.);
  for(int test=0;test<300;++test) {
    Core c; c.start(1.); Core::Command cmd; cmd.ingress=1; cmd.valid=true; cmd.stamp=1.001;
    Core::Joints measured{};
    for(size_t j=0;j<7;++j) {
      c.output[j]=uniform(rng)*c.velocity_limits[j]; cmd.velocity[j]=uniform(rng)*c.velocity_limits[j];
      if(test%10==0) c.output[j]=0.; // accelerating: current-speed-only model is wrong
      if(test%10==1) cmd.velocity[j]=0.;
    }
    const double age=std::array<double,3>{{0.,.03,.099}}[test%3];
    double now=cmd.stamp;
    c.update(now,.001,cmd,false,measured);
    if(age>0.) { now+=age; c.update(now,age,cmd,false,measured); }
    if(c.mode!=Core::TRACKING) throw std::runtime_error("invalid test setup");
    const auto initial=c.output;
    const double start=now;
    const double h=std::array<double,5>{{.001,.01,.02,.05,.1}}[(test/3)%5];
    Core::Joints travel{};
    for(int tick=0;tick<20000 && c.mode!=Core::LATCHED;++tick) {
      const double dt=test%2 ? h : h*(.1+.9*std::abs(uniform(rng)));
      for(size_t j=0;j<7;++j) travel[j]+=std::abs(c.output[j])*dt;
      now+=dt; c.update(now,dt,cmd,false,measured);
    }
    if(c.mode!=Core::LATCHED) throw std::runtime_error("test did not stop");
    std::cout<<"{\"age\":"<<age<<",\"gap\":"<<h<<",\"applied\":"; array(initial);
    std::cout<<",\"target\":"; array(cmd.velocity);
    std::cout<<",\"absolute_command_travel\":"; array(travel);
    std::cout<<",\"observed_stop_s\":"<<now-start<<"}\n";
  }
}
