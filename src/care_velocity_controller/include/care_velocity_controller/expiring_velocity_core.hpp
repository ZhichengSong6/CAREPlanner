#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace care_velocity_controller {
// ROS-independent policy used directly by the controller update and its tests.
// Time is simulation/ROS time. No wall-time or physical deceleration guarantee.
class ExpiringVelocityCore {
 public:
  using Joints = std::array<double, 7>;
  enum Mode : uint8_t { WAITING, TRACKING, BRAKING, LATCHED };
  struct Command {
    Joints velocity{};
    double stamp = 0.;
    uint32_t sequence = 0;
    uint64_t ingress = 0;
    bool valid = false;
  };
  Joints velocity_limits{{2.,2.,2.,2.,2.5,2.5,2.5}};
  Joints acceleration_limits{{3.,3.,4.,4.,6.,6.,6.}};
  double timeout = .1, max_step = .02, stopped_tolerance = .001;
  Mode mode = WAITING;
  bool clock_fault = false;
  Joints output{};
  double command_stamp = 0., fault_stamp = 0., stopped_stamp = 0.;
  uint32_t command_sequence = 0;
  uint64_t rejected = 0, rearm_successes = 0, rearm_rejections = 0;

  void validate() const {
    for (double v : {timeout, max_step, stopped_tolerance})
      if (!std::isfinite(v) || v <= 0.) throw std::invalid_argument("invalid controller timing/tolerance");
    for (size_t j=0; j<7; ++j)
      if (!std::isfinite(velocity_limits[j]) || velocity_limits[j] <= 0. ||
          !std::isfinite(acceleration_limits[j]) || acceleration_limits[j] <= 0.)
        throw std::invalid_argument("invalid joint limits");
  }
  void start(double now) {
    validate();
    mode=WAITING; clock_fault=false; output.fill(0.); latest_=Command(); target_.fill(0.);
    last_time_=floor_=last_stamp_=now;
    command_stamp=fault_stamp=stopped_stamp=0.; command_sequence=0;
    last_ingress_=0; rejected=rearm_successes=rearm_rejections=0;
  }
  void update(double now, double period, const Command& incoming, bool rearm,
              const Joints& measured) {
    const bool bad_clock = !std::isfinite(now) || now < last_time_ ||
                           !std::isfinite(period) || period < 0.;
    const double dt = bad_clock ? 0. : std::max(0., std::min({now-last_time_, period, max_step}));
    if (bad_clock) { clock_fault=true; fault(now); }
    if (std::isfinite(now)) last_time_=now;
    // Expiry is evaluated BEFORE a late replacement is allowed to renew a lease.
    if (mode == TRACKING && now-command_stamp >= timeout) fault(now);
    if (incoming.ingress && incoming.ingress != last_ingress_) {
      last_ingress_=incoming.ingress;
      bool valid = incoming.valid && std::isfinite(incoming.stamp) && incoming.stamp > floor_ &&
                   incoming.stamp > last_stamp_ && incoming.stamp <= now && now-incoming.stamp < timeout;
      for (size_t j=0; j<7; ++j)
        valid = valid && std::isfinite(incoming.velocity[j]) &&
                std::abs(incoming.velocity[j]) <= velocity_limits[j];
      if (valid && !clock_fault) {
        last_stamp_=incoming.stamp; latest_=incoming;
        if (mode == WAITING || mode == TRACKING) accept(incoming);
      } else { ++rejected; }
    }
    if (rearm) {
      const bool fresh_zero = latest_.valid && latest_.stamp > stopped_stamp &&
                              latest_.stamp <= now && now-latest_.stamp < timeout && zero(latest_.velocity, 1e-12);
      if (mode == LATCHED && !clock_fault && fresh_zero && zero(measured, stopped_tolerance)) {
        // Explicit rearm may only apply zero. Subsequent motion needs a newly
        // generated command after this update, never a buffered pre-rearm command.
        accept(latest_); floor_=now; ++rearm_successes;
      } else { ++rearm_rejections; }
    }
    for (size_t j=0; j<7; ++j) {
      const double desired = mode == TRACKING ? target_[j] : 0.;
      const double delta = acceleration_limits[j]*dt;
      output[j] += std::max(-delta, std::min(delta, desired-output[j]));
    }
    if (mode == BRAKING && zero(output, 1e-12)) {
      output.fill(0.); mode=LATCHED; stopped_stamp=now;
    }
  }
 private:
  double last_time_=0., floor_=0., last_stamp_=0.;
  uint64_t last_ingress_=0;
  Joints target_{};
  Command latest_;
  static bool zero(const Joints& values, double tolerance) {
    for (double v : values) if (!std::isfinite(v) || std::abs(v) > tolerance) return false;
    return true;
  }
  void accept(const Command& c) {
    target_=c.velocity; command_stamp=c.stamp; command_sequence=c.sequence; mode=TRACKING;
  }
  void fault(double now) {
    if (mode != BRAKING && mode != LATCHED) { mode=BRAKING; fault_stamp=now; }
  }
};
}  // namespace care_velocity_controller
