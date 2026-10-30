#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace egocentric_arm_planner {

struct RepairExecutableSettings {
  std::vector<double> prefixes;
  Eigen::VectorXd acceleration;
  double brake_dt = .05;
  int max_brake_steps = 20;
  // When present, u[k] is the endpoint velocity of an acceleration interval.
  // Otherwise preserve the legacy advertised-velocity view for old proposals.
  Eigen::VectorXd initial_velocity;
};

// A position on the exact prefix/brake view, differentiated with respect to
// the original QP's q and u knots. q0 is fixed and is omitted during assembly.
struct RepairExecutableSample {
  Eigen::VectorXd q;
  std::vector<std::pair<int, double>> q_weights;
  std::vector<std::pair<int, Eigen::VectorXd>> u_weights;
  double prefix_s = 0.;
  double time_s = 0.;
  bool brake = false;
};

// Mirrors _repair_prefix_with_braking_tail: interpolate q and advertised u at
// the prefix endpoint, clip each velocity toward zero, integrate trapezoids.
// Midpoints add proposal constraints; they do not replace continuous VBC.
// One latent first brake sample at zero speed allows the QP to create speed;
// nonlinear checking below must rebuild the resulting (possibly longer) tail.
inline bool repairExecutableSamples(const Eigen::MatrixXd& q,
    const Eigen::MatrixXd& u, double dt, const RepairExecutableSettings& settings,
    std::vector<RepairExecutableSample>* samples, std::string* error = nullptr) {
  auto fail = [&](const char* why) { if (error) *error = why; return false; };
  if (!samples) return fail("Missing executable samples output");
  samples->clear();
  const int n = q.rows(), K = u.cols();
  if (n < 1 || K < 1 || q.cols() != K+1 || u.rows() != n ||
      !q.allFinite() || !u.allFinite() || !(dt > 0.) || !std::isfinite(dt) ||
      settings.acceleration.size() != n || !settings.acceleration.allFinite() ||
      (settings.acceleration.array() <= 0.).any() || !(settings.brake_dt > 0.) ||
      !std::isfinite(settings.brake_dt) || settings.max_brake_steps < 1 ||
      settings.max_brake_steps > 20 || settings.prefixes.empty() || settings.prefixes.size() > 4)
    return fail("Invalid executable view contract");
  for (double requested : settings.prefixes) {
    if (!(requested > 0.) || !std::isfinite(requested)) return fail("Invalid executable prefix");
    const double prefix = std::min(requested, K*dt);
    const double knot = prefix/dt;
    const int left = std::min(K-1, static_cast<int>(std::floor(knot)));
    const double alpha = std::min(1., std::max(0., knot-left));
    RepairExecutableSample endpoint;
    endpoint.q = (1.-alpha)*q.col(left)+alpha*q.col(left+1);
    endpoint.q_weights = {{left, 1.-alpha}, {left+1, alpha}};
    endpoint.prefix_s = endpoint.time_s = prefix;
    const bool nodal = settings.initial_velocity.size() == n;
    if (settings.initial_velocity.size() && (!nodal || !settings.initial_velocity.allFinite()))
      return fail("Invalid initial executable velocity");
    auto nodal_sample = [&](int k, double s) {
      RepairExecutableSample sample;
      sample.prefix_s=prefix;sample.time_s=(k+s)*dt;
      const Eigen::VectorXd before=k ? u.col(k-1).eval() : settings.initial_velocity;
      const double bend=.5*dt*s*(1.-s);
      sample.q=(1.-s)*q.col(k)+s*q.col(k+1)+bend*(before-u.col(k));
      sample.q_weights={{k,1.-s},{k+1,s}};
      if(k)sample.u_weights.push_back({k-1,Eigen::VectorXd::Constant(n,bend)});
      sample.u_weights.push_back({k,Eigen::VectorXd::Constant(n,-bend)});
      return sample;
    };
    if(nodal)endpoint=nodal_sample(left,alpha);
    // The actual prefix is piecewise linear. Include its knots and midpoints,
    // including a fractional last interval when the prefix is off the grid.
    for (int k=0;k*dt < prefix-1e-9;++k) {
      RepairExecutableSample s;
      s.q=q.col(k); s.q_weights={{k,1.}}; s.prefix_s=prefix; s.time_s=k*dt;
      samples->push_back(s);
      if(nodal) {
        const double end=std::min(1.,(prefix-k*dt)/dt);
        samples->push_back(nodal_sample(k,.5*end));
        continue;
      }
      if ((k+1)*dt < prefix-1e-9) {
        s.q=.5*(q.col(k)+q.col(k+1)); s.q_weights={{k,.5},{k+1,.5}};
        s.time_s=(k+.5)*dt;
      } else {
        s.q=.5*(q.col(k)+endpoint.q); s.q_weights={{k,.5}};
        for (const auto& term : endpoint.q_weights) s.q_weights.push_back({term.first,.5*term.second});
        s.time_s=.5*(k*dt+prefix);
      }
      samples->push_back(std::move(s));
    }
    samples->push_back(endpoint);
    Eigen::VectorXd v;
    std::vector<std::pair<int,double>> velocity_weights;
    if(nodal) {
      v=(1.-alpha)*(left ? u.col(left-1).eval() : settings.initial_velocity)+alpha*u.col(left);
      if(left)velocity_weights.push_back({left-1,1.-alpha});
      velocity_weights.push_back({left,alpha});
    } else {
      v=(1.-alpha)*u.col(left);
      velocity_weights.push_back({left,1.-alpha});
      if(left+1<K){v+=alpha*u.col(left+1);velocity_weights.push_back({left+1,alpha});}
    }
    Eigen::VectorXd derivative = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd gain = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd position = endpoint.q;
    bool stopped = false;
    for (int step = 0; step < settings.max_brake_steps; ++step) {
      if (step > 0 && v.cwiseAbs().maxCoeff() <= 1e-6) { stopped = true; break; }
      Eigen::VectorXd next(n), next_derivative(n);
      for (int j = 0; j < n; ++j) {
        const double dv = settings.acceleration[j]*settings.brake_dt;
        next[j] = v[j] > dv ? v[j]-dv : v[j] < -dv ? v[j]+dv : 0.;
        next_derivative[j] = std::abs(v[j]) > dv ? derivative[j] : 0.;
      }
      const Eigen::VectorXd next_position = position+.5*settings.brake_dt*(v+next);
      const Eigen::VectorXd next_gain = gain+.5*settings.brake_dt*(derivative+next_derivative);
      for (int midpoint = 1; midpoint >= 0; --midpoint) {
        const double fraction = midpoint ? .5 : 1.;
        RepairExecutableSample s = endpoint;
        s.q = position+fraction*(next_position-position);
        s.time_s = prefix+(step+fraction)*settings.brake_dt;
        s.brake = true;
        Eigen::VectorXd derivative_gain = gain+fraction*(next_gain-gain);
        if(nodal) {
          const double bend=.5*settings.brake_dt*fraction*(1.-fraction);
          s.q+=bend*(v-next);
          derivative_gain+=bend*(derivative-next_derivative);
        }
        for(const auto& term:velocity_weights)
          if(term.second!=0.)s.u_weights.push_back({term.first,term.second*derivative_gain});
        samples->push_back(std::move(s));
      }
      position = next_position; gain = next_gain; v = next; derivative = next_derivative;
    }
    if (!stopped && v.cwiseAbs().maxCoeff() > 1e-5)
      return fail("Executable brake exceeds existing step limit");
  }
  return true;
}

// Additional learned queries use exact nodal prefix midpoints and complete
// brake endpoints. They are appended after the original QP knots, so a raw
// timestep is never mistaken for an executable timestep. Continuous native
// certification still covers every interval. Keep the existing 50-anchor cap.
inline bool repairCdfExecutableSamples(const Eigen::MatrixXd& q,
    const Eigen::MatrixXd& u,double dt,const RepairExecutableSettings& settings,
    std::vector<RepairExecutableSample>* output,std::string* error=nullptr) {
  std::vector<RepairExecutableSample> all;
  if(!output || !repairExecutableSamples(q,u,dt,settings,&all,error))return false;
  output->clear();
  for(const auto& sample:all) {
    if(sample.time_s<=0.)continue;
    const double step=(sample.time_s-sample.prefix_s)/settings.brake_dt;
    if(sample.brake ? std::abs(step-std::round(step))>1e-6 :
        std::abs(sample.time_s/dt-std::round(sample.time_s/dt))<=1e-6)continue;
    output->push_back(sample);
  }
  if(q.cols()+output->size()>50) {
    output->clear();if(error)*error="Executable CDF view exceeds original 50-anchor cap";return false;
  }
  return true;
}
// Read-only preview of the receiver's complete measured-state construction.
// Unlike the differentiated proposal samples above, this re-integrates the
// coordinated velocity ramp from live q/v, then appends the actual brake.
inline bool repairLiveExecutableSamples(const Eigen::MatrixXd& q,
    const Eigen::MatrixXd& u, double dt, const RepairExecutableSettings& settings,
    const Eigen::VectorXd& measured_q, const Eigen::VectorXd& measured_v,
    std::vector<RepairExecutableSample>* samples, std::string* error = nullptr) {
  auto fail=[&](const char* why){if(error)*error=why;return false;};
  std::vector<RepairExecutableSample> validation;
  if (!samples || !repairExecutableSamples(q,u,dt,settings,&validation,error)) return false;
  const int n=q.rows(),K=u.cols();
  if (settings.initial_velocity.size()!=n || measured_q.size()!=n || measured_v.size()!=n ||
      !measured_q.allFinite() || !measured_v.allFinite()) return fail("Invalid live q/v snapshot");
  samples->clear();
  auto velocity=[&](int k)->Eigen::VectorXd {return k ? u.col(k-1).eval() : settings.initial_velocity;};
  bool nodal=true;
  for(int k=0;k<K;++k) {
    const Eigen::VectorXd before=velocity(k);
    if ((q.col(k+1)-q.col(k)-.5*dt*(before+u.col(k))).cwiseAbs().maxCoeff()>1e-6 ||
        ((u.col(k)-before).cwiseAbs()-settings.acceleration*dt).maxCoeff()>1e-6) nodal=false;
  }
  auto at=[&](double t, bool want_velocity)->Eigen::VectorXd {
    const double knot=std::min(K*dt,std::max(0.,t))/dt;
    const int k=std::min(K-1,static_cast<int>(std::floor(knot)));
    const double s=std::min(1.,std::max(0.,knot-k));
    const Eigen::VectorXd before=velocity(k);
    if(want_velocity)return ((1.-s)*before+s*u.col(k)).eval();
    Eigen::VectorXd position=(1.-s)*q.col(k)+s*q.col(k+1);
    if((q.col(k+1)-q.col(k)-.5*dt*(before+u.col(k))).cwiseAbs().maxCoeff()<=1e-9)
      position+=.5*dt*s*(1.-s)*(before-u.col(k));
    return position;
  };
  auto rounded=[](double t){return std::llround(t*1e9)*1e-9;};
  for(double requested:settings.prefixes) {
    const double prefix=std::min(requested,K*dt);
    Eigen::VectorXd position=measured_q,v=measured_v;
    double t=0.;
    auto sample=[&](const Eigen::VectorXd& p,double time,bool brake){
      RepairExecutableSample s;s.q=p;s.time_s=time;s.prefix_s=prefix;s.brake=brake;
      samples->push_back(std::move(s));
    };
    sample(position,0.,false);
    for(int boundary=1;t<prefix-1e-9;++boundary) {
      const double end=std::min(prefix,boundary*dt);
      while(t<end-1e-9) {
        const double next_time=rounded(t+std::min(settings.brake_dt,end-t));
        const double h=next_time-t;
        if(h<=0.)return fail("Invalid live knot time");
        const Eigen::VectorXd desired=nodal ? at(next_time,true) : ((at(next_time,false)-at(t,false))/h).eval();
        const Eigen::VectorXd delta=desired-v;
        double fraction=1.;
        for(int j=0;j<n;++j)if(std::abs(delta[j])>1e-12)
          fraction=std::min(fraction,settings.acceleration[j]*h/std::abs(delta[j]));
        const Eigen::VectorXd next=v+fraction*delta;
        const Eigen::VectorXd next_position=position+.5*h*(v+next);
        sample(.5*(position+next_position)+.125*h*(v-next),t+.5*h,false);
        sample(next_position,next_time,false);
        position=next_position;v=next;t=next_time;
      }
    }
    for(int step=0;step<settings.max_brake_steps && v.cwiseAbs().maxCoeff()>1e-6;++step) {
      const double next_time=rounded(t+settings.brake_dt),h=next_time-t;
      Eigen::VectorXd next(n);
      for(int j=0;j<n;++j)next[j]=std::copysign(std::max(0.,std::abs(v[j])-settings.acceleration[j]*h),v[j]);
      const Eigen::VectorXd next_position=position+.5*h*(v+next);
      sample(.5*(position+next_position)+.125*h*(v-next),t+.5*h,true);
      sample(next_position,next_time,true);
      position=next_position;v=next;t=next_time;
    }
    if(v.cwiseAbs().maxCoeff()>1e-5)return fail("Live brake steps exhausted");
  }
  return true;
}

}  // namespace egocentric_arm_planner
